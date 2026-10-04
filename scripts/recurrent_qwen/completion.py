"""Hidden-state-only completion head for explicit-step pointer recurrence."""

import torch
from torch import Tensor, nn


class CompletionHead(nn.Module):
    """One stop logit per answer-position recurrent state [B,H].

    No numeric loop index, parsed requested depth, decoded symbol, or target is
    accepted by this interface. The prompt can encode its requested Steps.
    """

    def __init__(self, width: int, intermediate: int = 128) -> None:
        super().__init__()
        if width < 1 or intermediate < 1:
            raise ValueError("Completion-head widths must be positive")
        self.width = width
        self.intermediate = intermediate
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, intermediate),
                                     nn.GELU(), nn.Linear(intermediate, 1))

    def forward(self, answer_state: Tensor) -> Tensor:
        if answer_state.ndim != 2 or answer_state.shape[-1] != self.width:
            raise ValueError("Completion head requires answer states [batch, hidden]")
        return self.network(answer_state).squeeze(-1)
