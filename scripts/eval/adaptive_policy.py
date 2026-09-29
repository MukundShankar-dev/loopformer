"""Causal stopping rules for one-example recurrent inference."""

import math
from typing import Literal

import torch
from torch import Tensor, nn


class HaltingHead(nn.Module):
    """Five observable scalar features -> one stop logit; no base-model changes."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != 5:
            raise ValueError("Halting features require trailing dimension 5")
        return self.linear(features).squeeze(-1)


def feature_vector(loop: int, depth: int, budget: int, predicted_margin: float,
                   entropy: float, repeated_run: int) -> tuple[float, ...]:
    """Only current/past outputs and the prompt's requested depth are used."""
    if not 1 <= loop <= budget or not 1 <= depth <= budget or repeated_run < 1:
        raise ValueError("Invalid stopping feature indices")
    if not math.isfinite(predicted_margin) or predicted_margin < 0 or not math.isfinite(entropy) or entropy < 0:
        raise ValueError("Invalid confidence features")
    return loop / budget, depth / budget, predicted_margin, entropy, min(repeated_run, 4) / 4


class StoppingRule:
    """Callback accepted by RecurrentQwen.forward(stop_policy=...)."""

    def __init__(self, kind: Literal["requested_depth", "stability", "margin", "entropy", "learned"],
                 *, depth: int, budget: int, token_ids: list[int], k: int = 2,
                 threshold: float = 0.0, head: HaltingHead | None = None) -> None:
        if kind not in ("requested_depth", "stability", "margin", "entropy", "learned"):
            raise ValueError("Unknown stopping rule")
        if not 1 <= depth <= budget or len(token_ids) < 2 or len(set(token_ids)) != len(token_ids) or k < 1:
            raise ValueError("Invalid depth, budget, answer vocabulary, or stability k")
        if not math.isfinite(threshold) or (kind == "learned") != (head is not None):
            raise ValueError("Learned rule requires a head; threshold must be finite")
        self.kind, self.depth, self.budget = kind, depth, budget
        self.token_ids, self.k, self.threshold, self.head = token_ids, k, threshold, head
        self.history: list[int] = []
        self.last_features: tuple[float, ...] | None = None
        self.last_probability: float | None = None

    def __call__(self, loop: int, hidden: Tensor, scores: Tensor) -> bool:
        if hidden.shape[0] != 1 or scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError("Stopping is defined for one example at a time")
        allowed = scores[0, self.token_ids]
        if not torch.isfinite(allowed).all():
            raise ValueError("Nonfinite answer logits")
        probs = allowed.softmax(-1)
        ranked = allowed.topk(2).values
        margin = float((ranked[0] - ranked[1]).item())
        entropy = float((-(probs * probs.clamp_min(1e-30).log()).sum()).item())
        prediction = int(allowed.argmax().item())
        repeated = 1
        for prior in reversed(self.history):
            if prior != prediction:
                break
            repeated += 1
        self.history.append(prediction)
        self.last_features = feature_vector(loop, self.depth, self.budget, margin, entropy, repeated)
        if loop < self.depth:
            return False
        if self.kind == "requested_depth":
            return True
        if self.kind == "stability":
            return repeated >= self.k
        if self.kind == "margin":
            return margin >= self.threshold
        if self.kind == "entropy":
            return entropy <= self.threshold
        assert self.head is not None
        head_device = next(self.head.parameters()).device
        features = torch.tensor(self.last_features, device=head_device, dtype=torch.float32)
        self.last_probability = float(torch.sigmoid(self.head(features)).item())
        return self.last_probability >= self.threshold
