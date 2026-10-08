"""Qwen2 decoder surgery for the pinned Transformers implementation."""

import math
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn
from transformers import Qwen2ForCausalLM
from transformers.masking_utils import create_causal_mask

from .outputs import RecurrentOutput, answer_margin
from .completion import CompletionHead
from .memory import PromptMemory
from .prefix import fixed_prefix_layer
from torch.utils.checkpoint import checkpoint
from .interfaces import PromptRouter, ReentryBridge, RecurrentController


def _answer_readout(hidden: Tensor, positions: list[int]) -> Tensor:
    # Integer slices avoid MPS's nondeterministic scatter backward for advanced
    # indexing/gather. Batches are deliberately tiny; positions sync once per forward.
    return torch.stack([hidden[row, position] for row, position in enumerate(positions)])


class DecoderBlock(nn.Module):
    """An ordered group of original decoder objects, with no copies."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.prefix_reuse = False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        memory: PromptMemory | None = None,
    ) -> Tensor:
        for index, layer in enumerate(self.layers):
            if memory is not None:
                hidden = memory.layer_input(index, hidden)
            if self.prefix_reuse and memory is not None:
                hidden = fixed_prefix_layer(layer, hidden, position_embeddings, attention_mask,
                                            memory, index, self.gradient_checkpointing)
                continue
            call = layer
            if self.gradient_checkpointing and torch.is_grad_enabled():
                def call(x, _layer=layer, **kwargs):
                    return checkpoint(_layer, x, use_reentrant=False, **kwargs)
            hidden = call(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )
        return memory.block_output(hidden) if memory is not None else hidden


class RecurrentQwen(nn.Module):
    """P -> R repeated T times -> C, reusing the supplied model's weights.

    Construction freezes and shares the supplied base model in place. Load a
    fresh base for an independent reference after adapters have been attached.
    Only full attention and default RoPE are supported in Stage 0, as used by
    Qwen2.5-0.5B-Instruct. Device/dtype are inherited, never selected here.
    """

    def __init__(
        self, base: Qwen2ForCausalLM, recurrent_start: int = 6, recurrent_end: int = 18,
        *, recurrence_mode: str = "full_sequence",
    ) -> None:
        super().__init__()
        if not isinstance(base, Qwen2ForCausalLM):
            raise TypeError("Stage 0 requires Qwen2ForCausalLM")
        if recurrence_mode not in ("full_sequence", "fixed_prompt"):
            raise ValueError("recurrence_mode must be full_sequence or fixed_prompt")
        self.recurrence_mode = recurrence_mode
        layers = list(base.model.layers)
        if not 0 <= recurrent_start < recurrent_end <= len(layers):
            raise ValueError("Require 0 <= recurrent_start < recurrent_end <= layer count")
        if any(kind != "full_attention" for kind in base.config.layer_types):
            raise ValueError("Stage 0 supports full-attention Qwen layers only")
        if base.config.rope_parameters["rope_type"] != "default":
            raise ValueError("Stage 0 supports default RoPE only")
        if getattr(base, "peft_config", None):
            raise ValueError("Supply an unadapted base; attach LoRA to recurrence afterwards")
        base.requires_grad_(False)
        base.zero_grad(set_to_none=True)
        base.config.use_cache = False
        self.config = base.config
        self.recurrent_start = recurrent_start
        self.recurrent_end = recurrent_end
        self.embed_tokens = base.model.embed_tokens
        self.rotary_emb = base.model.rotary_emb
        self.prelude = DecoderBlock(layers[:recurrent_start])
        self.recurrent = DecoderBlock(layers[recurrent_start:recurrent_end])
        self.coda = DecoderBlock(layers[recurrent_end:])
        self.norm = base.model.norm
        self.lm_head = base.lm_head
        self.completion_head: CompletionHead | RecurrentController | None = None
        self.router = None
        self.bridge = None
        self.state_head = None
        self.train_scope = "lora"
        self.prefix_reuse = False
        self.gradient_checkpointing = False
        self.train(base.training)

    def enable_completion(self, intermediate: int = 128) -> None:
        """Attach a trainable stop readout without changing recurrent inputs."""
        if self.completion_head is not None:
            raise ValueError("Completion head is already attached")
        self.completion_head = CompletionHead(self.config.hidden_size, intermediate).to(
            device=self.embed_tokens.weight.device, dtype=self.embed_tokens.weight.dtype)

    def configure_executor(self, tokenizer: Any, *, train_scope: str = "recurrent",
                           bridge: bool = True, direct_readout: bool = True,
                           controller_size: int = 128, isolated: bool = True,
                           prefix_reuse: bool = False, gradient_checkpointing: bool = False) -> None:
        """Opt-in architecture; legacy forwards/checkpoints keep their semantics."""
        if train_scope not in ("lora", "recurrent", "full"):
            raise ValueError("train_scope must be lora, recurrent, or full")
        if self.recurrence_mode != "fixed_prompt":
            raise ValueError("Executor interfaces require fixed_prompt recurrence")
        if train_scope != "lora" and any(".lora_" in name for name, _ in self.named_parameters()):
            raise ValueError("Full-weight scopes require an unadapted base, not an attached LoRA model")
        if prefix_reuse and self.config.attention_dropout != 0:
            raise ValueError("Prefix reuse requires zero attention dropout for equivalent execution")
        self.train_scope = train_scope
        if train_scope == "recurrent":
            self.recurrent.requires_grad_(True)
        elif train_scope == "full":
            self.requires_grad_(True)
        device, dtype = self.embed_tokens.weight.device, self.embed_tokens.weight.dtype
        if isolated:
            self.router = PromptRouter(tokenizer)
            if controller_size:
                self.completion_head = RecurrentController(self.config.hidden_size, controller_size).to(device=device, dtype=dtype)
        if bridge:
            self.bridge = ReentryBridge(self.config.hidden_size).to(device=device, dtype=dtype)
        if direct_readout:
            self.state_head = nn.Linear(self.config.hidden_size, 26).to(device=device, dtype=dtype)
        self.prefix_reuse = prefix_reuse
        self.gradient_checkpointing = gradient_checkpointing
        for block in (self.prelude, self.recurrent, self.coda):
            block.prefix_reuse = prefix_reuse
            block.gradient_checkpointing = gradient_checkpointing

    def controller_prompt_state(self, input_ids: Tensor, attention_mask: Tensor,
                                answer_positions: Tensor | None = None) -> Tensor:
        """Frozen full-prompt P features [B,H], identical to controller initialization.

        This interface runs neither R nor C and consumes only token IDs/mask.
        It never parses numeric values. Used to cache broader controller prompts.
        """
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or not attention_mask.any(-1).all():
            raise ValueError('Need aligned nonempty [B,S] prompt IDs and mask')
        if answer_positions is None:
            answer_positions = (attention_mask.long() * torch.arange(1, input_ids.shape[1] + 1,
                                device=input_ids.device)).argmax(-1)
        with torch.no_grad():
            context = self.embed_tokens(input_ids)
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            mask = create_causal_mask(config=self.config, inputs_embeds=context,
                attention_mask=attention_mask, past_key_values=None, position_ids=positions)
            context = self.prelude(context, mask, positions, self.rotary_emb(context, positions))
        return _answer_readout(context, answer_positions.tolist())

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        num_loops: int = 1,
        position_ids: Tensor | None = None,
        answer_positions: Tensor | None = None,
        labels: Tensor | None = None,
        allowed_token_ids: Tensor | None = None,
        return_hidden_states: bool = False,
        logits_mode: Literal["answer", "all"] = "answer",
        stop_policy: Callable[[int, Tensor, Tensor], bool] | None = None,
        completion_threshold: float | None = None,
        readout_token_ids: list[int] | None = None,
    ) -> RecurrentOutput:
        """Run complete prompt states through shared recurrence, without KV cache.

        Inputs/mask: [B,S]; positions: [1,S] or [B,S], default arange(S),
        matching ordinary Qwen forward (including padding). Answer positions:
        [B], default last unmasked token. Labels: [B,T] *token IDs*, one
        target per loop, with an explicit allowed answer set [A]. No CE loss
        or causal label shifting is performed. Pass repeated final labels
        explicitly if final-answer margins, rather than step margins, are wanted.
        Optional stop_policy(t, h_t, answer_logits_t) is inference-only, batch 1,
        and may return True to end after loop t. It never sees labels.
        completion_threshold enables the checkpoint's hidden-state-only head,
        independent of task depth, under the num_loops safety cap (batch 1).
        fixed_prompt keeps only the final unmasked position writable; returned
        full-shaped states contain fixed first-pass prefix outputs for diagnostics.
        """
        if type(num_loops) is not int or num_loops < 1:
            raise ValueError("num_loops must be a positive integer")
        if stop_policy is not None and (self.training or torch.is_grad_enabled() or input_ids.shape[0] != 1 or labels is not None or logits_mode != "answer"):
            raise ValueError("Stopping requires eval/no_grad, batch 1, no labels, and answer logits")
        if completion_threshold is not None:
            if (not math.isfinite(completion_threshold) or not 0 < completion_threshold < 1 or
                    self.completion_head is None or stop_policy is not None):
                raise ValueError("Completion stopping requires a head, threshold in (0,1), and no other stop policy")
            if self.training or torch.is_grad_enabled() or input_ids.shape[0] != 1 or labels is not None or logits_mode != "answer":
                raise ValueError("Completion stopping requires eval/no_grad, batch 1, no labels, and answer logits")
        if logits_mode not in ("answer", "all"):
            raise ValueError("logits_mode must be 'answer' or 'all'")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or 0 in input_ids.shape:
            raise ValueError("input_ids must be a nonempty torch.long [batch, sequence] tensor")
        if input_ids.device != self.embed_tokens.weight.device:
            raise ValueError("Input IDs and model must share a device")
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ValueError("Input IDs must be within the model vocabulary")
        batch, sequence = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or attention_mask.device != device:
            raise ValueError("attention_mask must match input_ids shape and device")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must contain only 0 and 1")
        if not attention_mask.bool().any(dim=-1).all():
            raise ValueError("Each example must have at least one unmasked token")
        if position_ids is None:
            position_ids = torch.arange(sequence, device=device).unsqueeze(0)
        if (
            position_ids.shape not in ((1, sequence), (batch, sequence))
            or position_ids.dtype != torch.long
            or position_ids.device != device
            or (position_ids < 0).any()
        ):
            raise ValueError("position_ids must be nonnegative long [1,S] or [B,S] on the input device")
        if answer_positions is None:
            indices = torch.arange(sequence, device=device).expand(batch, -1)
            answer_positions = indices.masked_fill(~attention_mask.bool(), -1).max(-1).values
        if (
            answer_positions.shape != (batch,)
            or answer_positions.dtype != torch.long
            or answer_positions.device != device
            or (answer_positions < 0).any()
            or (answer_positions >= sequence).any()
        ):
            raise ValueError("answer_positions must be long [batch] valid sequence indices on the input device")
        rows = torch.arange(batch, device=device)
        if not attention_mask[rows, answer_positions].bool().all():
            raise ValueError("Answer positions cannot point to padding")
        if self.recurrence_mode == "fixed_prompt":
            last = torch.arange(sequence, device=device).expand(batch, -1).masked_fill(~attention_mask.bool(), -1).max(-1).values
            if not torch.equal(answer_positions, last):
                raise ValueError("fixed_prompt requires the final unmasked position as its working state")
        readout_positions = answer_positions.tolist()
        if (labels is None) != (allowed_token_ids is None):
            raise ValueError("Provide both labels and allowed_token_ids to compute margins")
        if labels is not None and labels.shape != (batch, num_loops):
            raise ValueError("labels must have shape [batch, num_loops]")

        controller_initial = None
        if self.router is not None:
            if logits_mode != "answer":
                raise ValueError("Isolated execution supports answer-position logits only")
            # Controller prompt encoding is isolated even for full-model training.
            controller_initial = self.controller_prompt_state(input_ids, attention_mask, answer_positions)
            input_ids, attention_mask, position_ids = self.router(input_ids, attention_mask)
        hidden = self.embed_tokens(input_ids)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden, position_ids)
        block_args = (causal_mask, position_ids, position_embeddings)
        hidden = self.prelude(hidden, *block_args)
        initial_hidden = hidden if return_hidden_states else None
        bridge_initial = _answer_readout(hidden, readout_positions)
        controller_memory = (self.completion_head.initialize(controller_initial)
                             if isinstance(self.completion_head, RecurrentController) else None)
        recurrent_memory = coda_memory = None
        if self.recurrence_mode == "fixed_prompt":
            write_mask = (torch.arange(sequence, device=device)[None, :] == answer_positions[:, None]).unsqueeze(-1)
            recurrent_memory, coda_memory = PromptMemory(write_mask), PromptMemory(write_mask)
        states, logits, margins, stop_logits, state_logits = [], [], [], [], []
        for loop in range(num_loops):
            if loop and self.bridge is not None:
                working = self.bridge(_answer_readout(hidden, readout_positions), bridge_initial)
                hidden = torch.where(write_mask, working[:, None, :], hidden)
            hidden = (self.recurrent(hidden, *block_args, memory=recurrent_memory)
                      if recurrent_memory is not None else self.recurrent(hidden, *block_args))
            working = _answer_readout(hidden, readout_positions)
            if isinstance(self.completion_head, RecurrentController):
                stop, controller_memory = self.completion_head.advance(working, controller_memory)
                stop_logits.append(stop)
            elif self.completion_head is not None:
                stop_logits.append(self.completion_head(working))
            if self.state_head is not None:
                state_logits.append(self.state_head(working))
            if return_hidden_states:
                states.append(hidden)
            # C stays differentiable even when frozen: readout loss must reach R.
            decoded = self.norm(self.coda(hidden, *block_args, memory=coda_memory)
                                if coda_memory is not None else self.coda(hidden, *block_args))
            readout = decoded if logits_mode == "all" else _answer_readout(decoded, readout_positions)
            if readout_token_ids is None:
                scores = self.lm_head(readout)
            else:
                if labels is not None:
                    raise ValueError("Use full-vocabulary readout when requesting token-ID margins")
                ids = torch.tensor(readout_token_ids, device=device)
                scores = nn.functional.linear(readout, self.lm_head.weight.index_select(0, ids))
            logits.append(scores)
            if labels is not None:
                answer_scores = _answer_readout(scores, readout_positions) if logits_mode == "all" else scores
                margins.append(answer_margin(answer_scores, labels[:, loop], allowed_token_ids))
            if stop_policy is not None and stop_policy(loop + 1, hidden, scores):
                break
            if completion_threshold is not None:
                probability = stop_logits[-1].sigmoid().item()
                if not math.isfinite(probability):
                    raise FloatingPointError("Nonfinite completion probability")
                if probability >= completion_threshold:
                    break
        return RecurrentOutput(
            loop_logits=tuple(logits),
            state_logits=torch.stack(state_logits, dim=1) if state_logits else None,
            hidden_states=tuple(states) if return_hidden_states else None,
            initial_hidden_state=initial_hidden,
            margins=torch.stack(margins, dim=1) if margins else None,
            stop_logits=torch.stack(stop_logits, dim=1) if stop_logits else None,
        )
