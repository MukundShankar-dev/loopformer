"""Read-only recurrent-state, attention, and adapter-ablation diagnostics."""

from contextlib import contextmanager
from dataclasses import dataclass
import re
from typing import Any, Iterator

import torch
from peft.tuners.lora import LoraLayer
from transformers.masking_utils import create_causal_mask

from scripts.dataset.pointer import PointerExample, SYMBOLS


@contextmanager
def adapters_disabled(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily remove only the trained LoRA contribution; restore on error."""
    layers = [layer for layer in model.recurrent.modules() if isinstance(layer, LoraLayer)]
    if not layers:
        raise ValueError("No recurrent LoRA layers found for adapter ablation")
    previous = [layer.disable_adapters for layer in layers]
    try:
        for layer in layers:
            layer.enable_adapters(False)
        yield
    finally:
        for layer, disabled in zip(layers, previous, strict=True):
            layer.enable_adapters(not disabled)


def rule_symbol_positions(tokenizer: Any, task: PointerExample) -> tuple[list[int], list[int]]:
    """Exact token indices for the source/destination symbols in rule order."""
    offsets = {tuple(span): index for index, span in enumerate(
        tokenizer(task.prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"])}
    rule_line = task.prompt.splitlines()[0]
    pairs = list(re.finditer(r"\( ([A-Z]), ([A-Z])\)", rule_line))
    if [(match.group(1), match.group(2)) for match in pairs] != [tuple(pair) for pair in task.mapping]:
        raise ValueError("Rule text does not match task mapping")
    sources, destinations = [], []
    for match in pairs:
        for group, positions in ((1, sources), (2, destinations)):
            # Tokenizers include the leading space in each symbol token.
            span = (match.start(group) - 1, match.end(group))
            if span not in offsets:
                raise ValueError(f"Rule symbol is not an independent token: {span}")
            positions.append(offsets[span])
    return sources, destinations


class InternalRecorder:
    """Capture answer vectors and optionally R-attention to all 26 rule pairs.

    Hooks are only active around a single batch-one eval/no-grad forward. Each
    extracted attention slice is copied, so a retained slice cannot keep the
    full [sequence, sequence] matrix alive across recurrent passes.
    """

    def __init__(self, model: torch.nn.Module, answer_position: int,
                 source_positions: list[int] | None = None,
                 destination_positions: list[int] | None = None) -> None:
        self.model = model
        self.answer_position = answer_position
        if (source_positions is None) != (destination_positions is None):
            raise ValueError("Provide both rule source and destination positions")
        self.source_positions = source_positions
        self.destination_positions = destination_positions
        self.coda: list[torch.Tensor] = []
        self.sources: list[list[torch.Tensor]] = [[] for _ in model.recurrent.layers]
        self.destinations: list[list[torch.Tensor]] = [[] for _ in model.recurrent.layers]
        self.handles: list[Any] = []

    def __enter__(self) -> "InternalRecorder":
        self.handles.append(self.model.norm.register_forward_hook(self._coda_hook))
        if self.source_positions is not None:
            for index, layer in enumerate(self.model.recurrent.layers):
                self.handles.append(layer.self_attn.register_forward_hook(self._attention_hook(index)))
        return self

    def __exit__(self, *exception: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _coda_hook(self, module: Any, args: Any, output: torch.Tensor) -> None:
        if output.shape[0] != 1:
            raise ValueError("Internal recording requires batch one")
        self.coda.append(output[0, self.answer_position].detach().clone())

    def _attention_hook(self, index: int):
        def hook(module: Any, args: Any, output: tuple[torch.Tensor, torch.Tensor | None]) -> None:
            weights = output[1]
            if weights is None or weights.shape[0] != 1:
                raise ValueError("Eager attention weights unavailable; use the checkpoint's eager loader")
            answer_weights = weights[0, :, self.answer_position, :].detach()
            source_indices = torch.tensor(self.source_positions, device=answer_weights.device)
            destination_indices = torch.tensor(self.destination_positions, device=answer_weights.device)
            self.sources[index].append(answer_weights.index_select(-1, source_indices))
            self.destinations[index].append(answer_weights.index_select(-1, destination_indices))
        return hook

    def finish(self, loops: int) -> dict[str, torch.Tensor]:
        if len(self.coda) != loops:
            raise ValueError(f"Expected {loops} frozen-coda readouts, got {len(self.coda)}")
        result = {"coda": torch.stack(self.coda).float().cpu()}
        if self.source_positions is not None:
            for name, data in (("source_attention", self.sources),
                               ("destination_attention", self.destinations)):
                if any(len(layer) != loops for layer in data):
                    raise ValueError(f"Incomplete {name} capture")
                # [loop, recurrent layer, attention head, rule in prompt order]
                result[name] = torch.stack([torch.stack(layer) for layer in data], dim=1).float().cpu()
        return result


def recurrent_answer_vectors(result: Any, answer_position: int) -> torch.Tensor:
    """Return h0..hT at the answer token, [T+1,H], detached on CPU."""
    if result.initial_hidden_state is None or result.hidden_states is None:
        raise ValueError("Forward did not capture recurrent hidden states")
    states = [result.initial_hidden_state, *result.hidden_states]
    return torch.stack([state[0, answer_position].detach() for state in states]).float().cpu()


def preloop_symbol(model: torch.nn.Module, initial_hidden: torch.Tensor,
                   token_ids: list[int]) -> str:
    """Read frozen C(h0) once, before any recurrent pass, without feedback."""
    if initial_hidden.ndim != 3 or initial_hidden.shape[0] != 1 or model.training or torch.is_grad_enabled():
        raise ValueError("Preloop readout requires eval/no_grad, batch-one full h0")
    sequence = initial_hidden.shape[1]
    attention = torch.ones((1, sequence), dtype=torch.long, device=initial_hidden.device)
    positions = torch.arange(sequence, device=initial_hidden.device).unsqueeze(0)
    mask = create_causal_mask(config=model.config, inputs_embeds=initial_hidden,
                              attention_mask=attention, past_key_values=None, position_ids=positions)
    embeddings = model.rotary_emb(initial_hidden, positions)
    decoded = model.norm(model.coda(initial_hidden, mask, positions, embeddings))
    logits = model.lm_head(decoded[0, -1])
    winner = logits[torch.tensor(token_ids, device=logits.device)].argmax().item()
    return SYMBOLS[winner]


@dataclass
class RidgeReadout:
    mean: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor
    intercept: torch.Tensor
    alpha: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return ((features.float() - self.mean) / self.scale) @ self.weights + self.intercept


def fit_ridge_readout(features: torch.Tensor, targets: torch.Tensor, *, alpha: float = 1.0) -> RidgeReadout:
    """Fixed-alpha 26-class linear decodability probe; mapping-level holdout is external."""
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.numel() or features.shape[0] < 2:
        raise ValueError("Need aligned nonempty [N,H] features and [N] targets")
    if alpha <= 0 or not torch.isfinite(features).all() or not ((targets >= 0) & (targets < len(SYMBOLS))).all():
        raise ValueError("Invalid probe alpha, features, or class targets")
    features = features.float().cpu()
    targets = targets.long().cpu()
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-5)
    centered = (features - mean) / scale
    labels = torch.nn.functional.one_hot(targets, len(SYMBOLS)).float()
    intercept = labels.mean(0)
    gram = centered @ centered.T
    gram.diagonal().add_(alpha)
    dual = torch.linalg.solve(gram, labels - intercept)
    weights = centered.T @ dual
    return RidgeReadout(mean, scale, weights, intercept, alpha)
