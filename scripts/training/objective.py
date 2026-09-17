"""Nominal transition CE with explicit example or dataset-level loop weighting."""

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


def loop_loss_weights(depths: list[int]) -> list[float]:
    """Weights N/(D*N_t) make the dataset objective mean_t(mean_eligible CE).

    Use counts from the entire selected training set, never a microbatch. This
    gives unbiased minibatch estimates and preserves accumulation equivalence.
    """
    if not depths or any(type(d) is not int or d < 1 for d in depths):
        raise ValueError("Need positive integer training depths")
    maximum = max(depths)
    return [len(depths) / (maximum * sum(d >= t for d in depths))
            for t in range(1, maximum + 1)]


def training_selection_loss(metrics: dict, max_depth: int, reduction: str) -> float:
    """Select on trained depths only; never include OOD tasks' early loops."""
    eligible = [v for d, v in metrics["by_depth"].items() if int(d) <= max_depth]
    if reduction == "example_mean":
        return sum(v["loss"] * v["examples"] for v in eligible) / sum(v["examples"] for v in eligible)
    if reduction != "loop_mean":
        raise ValueError("Unknown loss reduction")
    means = []
    for t in range(1, max_depth + 1):
        values = [v["per_loop"][str(t)] for v in eligible if str(t) in v["per_loop"]]
        count = sum(v["count"] for v in values)
        if not count:
            raise ValueError(f"No trained-range validation targets for loop {t}")
        means.append(sum(v["loss"] * v["count"] for v in values) / count)
    return sum(means) / len(means)


def step_loss(scores: Tensor, targets: Tensor, mask: Tensor, *, loop_weights: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """CE over symbols; default mean valid loops per example, then examples.

    Optional dataset-derived loop_weights replace the within-example mean with
    a weighted sum. Return unweighted masked CE separately for shared metrics.

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
    if loop_weights is not None:
        if (loop_weights.ndim != 1 or len(loop_weights) < scores.shape[1]
                or loop_weights.device != scores.device or not torch.isfinite(loop_weights).all()
                or not (loop_weights > 0).all()):
            raise ValueError("Need positive finite loop weights covering all loops on the scores device")
        return (losses * loop_weights[:scores.shape[1]]).sum(-1).mean(), losses
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
