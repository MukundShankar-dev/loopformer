"""Per-forward, differentiable prompt memory for a single writable position."""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class PromptMemory:
    """Capture layer inputs on pass one and reuse their non-working positions.

    write_mask is [B,S,1]; captured tensors are [B,S,H]. Never detach: later
    losses must reach adapters that formed the prompt memory on the first pass.
    Objects live for one model forward only, not across examples or updates.
    This is an activation-memory reference path, not an attention KV cache.
    """

    write_mask: Tensor
    layers: list[Tensor] = field(default_factory=list)
    output: Tensor | None = None

    def layer_input(self, index: int, hidden: Tensor) -> Tensor:
        if self.output is None:
            self.layers.append(hidden)
            return hidden
        return torch.where(self.write_mask, hidden, self.layers[index])

    def block_output(self, hidden: Tensor) -> Tensor:
        if self.output is None:
            self.output = hidden
            return hidden
        return torch.where(self.write_mask, hidden, self.output)
