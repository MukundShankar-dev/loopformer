"""One loss per valid recurrent transition, with equal example weighting."""

import torch
from torch import Tensor
from torch.nn import functional as F


def symbolic_scores(loop_logits: tuple[Tensor, ...], token_ids: list[int]) -> Tensor:
    """Return [B,T,A]; integer slices avoid scatter-based MPS backward in indexing."""
    if not loop_logits or len(token_ids) < 2 or len(set(token_ids)) != len(token_ids):
        raise ValueError("Need loop logits and at least two unique answer tokens")
    if min(token_ids) < 0 or max(token_ids) >= loop_logits[0].shape[-1]:
        raise ValueError("Answer token outside vocabulary")
    return torch.stack([
        torch.cat([scores[:, token:token + 1] for token in token_ids], dim=-1)
        for scores in loop_logits
    ], dim=1)


def step_loss(scores: Tensor, targets: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """CE over symbols; mean valid loops per example, then mean examples.

    Later losses retain their graph through earlier recurrence. Masked positions
    contribute zero loss/gradient and are not final-answer retention supervision.
    """
    if scores.ndim != 3 or targets.shape != scores.shape[:2] or mask.shape != targets.shape:
        raise ValueError("Expected scores [B,T,A], targets/mask [B,T]")
    if targets.dtype != torch.long or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError("Long targets and boolean mask with a valid target per example required")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite symbolic logits")
    losses = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
    losses = losses * mask
    return (losses.sum(-1) / mask.sum(-1)).mean(), losses


def batch_metrics(scores: Tensor, targets: Tensor, mask: Tensor, losses: Tensor) -> dict:
    """Detached numerator/count statistics, safe to aggregate unequal batches."""
    correct = (scores.detach().argmax(-1) == targets) & mask
    counts = mask.sum(-1)
    return {
        "examples": len(targets),
        "example_loss_sum": (losses.detach().sum(-1) / counts).sum().item(),
        "trajectory_correct": (correct | ~mask).all(-1).sum().item(),
        "loop_counts": mask.sum(0).tolist(),
        "loop_correct": correct.sum(0).tolist(),
        "loop_loss_sums": losses.detach().sum(0).tolist(),
    }


def combine_metrics(parts: list[dict]) -> dict:
    loops = max(len(part["loop_counts"]) for part in parts)
    total = sum(part["examples"] for part in parts)
    per_loop = {}
    for t in range(loops):
        available = [part for part in parts if t < len(part["loop_counts"])]
        count = sum(part["loop_counts"][t] for part in available)
        if count:
            per_loop[str(t + 1)] = {
                "count": count,
                "loss": sum(part["loop_loss_sums"][t] for part in available) / count,
                "accuracy": sum(part["loop_correct"][t] for part in available) / count,
            }
    steps = sum(item["count"] for item in per_loop.values())
    return {
        "examples": total,
        "loss": sum(part["example_loss_sum"] for part in parts) / total,
        "trajectory_accuracy": sum(part["trajectory_correct"] for part in parts) / total,
        "intermediate_accuracy": sum(item["accuracy"] * item["count"] for item in per_loop.values()) / steps,
        "per_loop": per_loop,
    }
