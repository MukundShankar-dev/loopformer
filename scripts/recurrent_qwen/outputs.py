"""Loop outputs and symbolic readout semantics."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RecurrentOutput:
    """Tuple index t-1 represents loop t; h_0 is not a decoded start symbol.

    Logits are [batch, vocab] in answer mode or [batch, sequence, vocab]
    in all-token mode. Optional hidden states are [batch, sequence, hidden].
    Margins are [batch, loops]. Tensors retain autograd unless the caller
    disables it; observations never feed back into the recurrent trajectory.
    """

    loop_logits: tuple[Tensor, ...]
    hidden_states: tuple[Tensor, ...] | None = None
    initial_hidden_state: Tensor | None = None
    margins: Tensor | None = None

    @property
    def logits(self) -> Tensor:
        """The final loop's logits."""
        return self.loop_logits[-1]


def answer_margin(logits: Tensor, labels: Tensor, allowed_token_ids: Tensor) -> Tensor:
    """Correct raw logit minus the best *other allowed* logit, per example.

    Shapes: logits [batch, vocab], labels [batch] (token IDs), allowed IDs
    [answers]. Labels must belong to a unique answer set of at least two IDs.
    A tie has margin zero, so it is not margin-based correctness.
    """
    if logits.ndim != 2 or labels.shape != logits.shape[:1]:
        raise ValueError("Expected logits [batch, vocab] and labels [batch]")
    if labels.dtype != torch.long or allowed_token_ids.dtype != torch.long:
        raise ValueError("Labels and allowed_token_ids must be torch.long token IDs")
    if labels.device != logits.device or allowed_token_ids.device != logits.device:
        raise ValueError("Logits, labels, and allowed_token_ids must share a device")
    if allowed_token_ids.ndim != 1 or allowed_token_ids.numel() < 2:
        raise ValueError("Provide at least two allowed token IDs in a 1D tensor")
    if allowed_token_ids.unique().numel() != allowed_token_ids.numel():
        raise ValueError("Allowed token IDs must be unique")
    if (allowed_token_ids < 0).any() or (allowed_token_ids >= logits.shape[-1]).any():
        raise ValueError("Allowed token IDs must be within the model vocabulary")
    correct = labels[:, None] == allowed_token_ids[None, :]
    if not correct.any(dim=-1).all():
        raise ValueError("Every label must be an allowed token ID")
    scores = logits.index_select(-1, allowed_token_ids)
    wrong = scores.masked_fill(correct, -torch.inf).max(dim=-1).values
    return logits.gather(-1, labels[:, None]).squeeze(-1) - wrong
