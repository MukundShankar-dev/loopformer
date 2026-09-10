"""Qwen2 decoder surgery for the pinned Transformers implementation."""

from typing import Literal

import torch
from torch import Tensor, nn
from transformers import Qwen2ForCausalLM
from transformers.masking_utils import create_causal_mask

from .outputs import RecurrentOutput, answer_margin


def _answer_readout(hidden: Tensor, positions: list[int]) -> Tensor:
    # Integer slices avoid MPS's nondeterministic scatter backward for advanced
    # indexing/gather. Batches are deliberately tiny; positions sync once per forward.
    return torch.stack([hidden[row, position] for row, position in enumerate(positions)])


class DecoderBlock(nn.Module):
    """An ordered group of original decoder objects, with no copies."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )
        return hidden


class RecurrentQwen(nn.Module):
    """P -> R repeated T times -> C, reusing the supplied model's weights.

    Construction freezes and shares the supplied base model in place. Load a
    fresh base for an independent reference after adapters have been attached.
    Only full attention and default RoPE are supported in Stage 0, as used by
    Qwen2.5-0.5B-Instruct. Device/dtype are inherited, never selected here.
    """

    def __init__(
        self, base: Qwen2ForCausalLM, recurrent_start: int = 6, recurrent_end: int = 18
    ) -> None:
        super().__init__()
        if not isinstance(base, Qwen2ForCausalLM):
            raise TypeError("Stage 0 requires Qwen2ForCausalLM")
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
        self.train(base.training)

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
    ) -> RecurrentOutput:
        """Run complete prompt states through shared recurrence, without KV cache.

        Inputs/mask: [B,S]; positions: [1,S] or [B,S], default arange(S),
        matching ordinary Qwen forward (including padding). Answer positions:
        [B], default last unmasked token. Labels: [B,T] *token IDs*, one
        target per loop, with an explicit allowed answer set [A]. No CE loss
        or causal label shifting is performed. Pass repeated final labels
        explicitly if final-answer margins, rather than step margins, are wanted.
        """
        if type(num_loops) is not int or num_loops < 1:
            raise ValueError("num_loops must be a positive integer")
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
        readout_positions = answer_positions.tolist()
        if (labels is None) != (allowed_token_ids is None):
            raise ValueError("Provide both labels and allowed_token_ids to compute margins")
        if labels is not None and labels.shape != (batch, num_loops):
            raise ValueError("labels must have shape [batch, num_loops]")

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
        states, logits, margins = [], [], []
        for loop in range(num_loops):
            hidden = self.recurrent(hidden, *block_args)
            if return_hidden_states:
                states.append(hidden)
            # Frozen C stays differentiable: readout loss must reach R's adapters.
            decoded = self.norm(self.coda(hidden, *block_args))
            readout = decoded if logits_mode == "all" else _answer_readout(decoded, readout_positions)
            scores = self.lm_head(readout)
            logits.append(scores)
            if labels is not None:
                answer_scores = _answer_readout(scores, readout_positions) if logits_mode == "all" else scores
                margins.append(answer_margin(answer_scores, labels[:, loop], allowed_token_ids))
        return RecurrentOutput(
            loop_logits=tuple(logits),
            hidden_states=tuple(states) if return_hidden_states else None,
            initial_hidden_state=initial_hidden,
            margins=torch.stack(margins, dim=1) if margins else None,
        )
