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


def training_selection_loss(metrics: dict, max_depth: int, reduction: str, depths: list[int] | None = None) -> float:
    """Select on trained depths only; never include OOD tasks' early loops."""
    eligible = [v for d, v in metrics["by_depth"].items() if int(d) in (depths if depths is not None else range(1, max_depth + 1))]
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


def completion_loss(stop_logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Continue at 1..d-1, stop at d; average the two classes per example.

    A padded pass t>d has no loss. The depth is used only to construct labels;
    neither depth nor loop index is passed to the completion head.
    """
    if stop_logits.ndim != 2 or mask.shape != stop_logits.shape or mask.dtype != torch.bool:
        raise ValueError("Expected stop logits and boolean target mask [batch, loops]")
    if not mask.any(-1).all() or not (mask.int().cumprod(-1).bool() == mask).all():
        raise ValueError("Each example needs a nonempty contiguous target prefix")
    if not torch.isfinite(stop_logits).all():
        raise FloatingPointError("Non-finite stop logits")
    depths = mask.sum(-1)
    stop = torch.zeros_like(stop_logits)
    stop.scatter_(1, (depths - 1).unsqueeze(-1), 1)
    losses = F.binary_cross_entropy_with_logits(stop_logits, stop, reduction="none") * mask
    final = losses.gather(1, (depths - 1).unsqueeze(-1)).squeeze(-1)
    prior = (losses.sum(-1) - final) / (depths - 1).clamp_min(1)
    per_example = torch.where(depths > 1, (prior + final) / 2, final)
    return per_example.mean(), losses


def completion_metrics(stop_logits: Tensor, mask: Tensor, losses: Tensor, objective: Tensor) -> dict:
    """Additive counts for microbatch-safe diagnostic aggregation."""
    depths = mask.sum(-1)
    positive = stop_logits.gather(1, (depths - 1).unsqueeze(-1)).squeeze(-1) >= 0
    first = (stop_logits >= 0).int().argmax(-1) + 1
    has_stop = (stop_logits >= 0).any(-1)
    exact = has_stop & (first == depths)
    return {"examples": len(depths), "objective_loss_sum": objective.item() * len(depths),
            "stop_loss_sum": losses.sum().item(),
            "stop_label_count": mask.sum().item(), "stop_positive_correct": positive.sum().item(),
            "continue_correct": ((stop_logits < 0) & mask).sum().item() - (~positive).sum().item(),
            "continue_count": (depths - 1).sum().item(), "first_stop_exact": exact.sum().item(),
            "first_stop_early": (has_stop & (first < depths)).sum().item(),
            "first_stop_late_or_missing": (~has_stop | (first > depths)).sum().item()}


def combine_completion_metrics(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("Need completion metric parts")
    total = sum(part["examples"] for part in parts)
    continues = sum(part["continue_count"] for part in parts)
    return {"loss_per_label": sum(part["stop_loss_sum"] for part in parts) / sum(part["stop_label_count"] for part in parts),
            "objective_loss": sum(part["objective_loss_sum"] for part in parts) / total,
            "stop_at_depth_accuracy": sum(part["stop_positive_correct"] for part in parts) / total,
            "continue_accuracy": sum(part["continue_correct"] for part in parts) / continues if continues else None,
            "first_stop_exact_rate": sum(part["first_stop_exact"] for part in parts) / total,
            "first_stop_early_rate": sum(part["first_stop_early"] for part in parts) / total,
            "first_stop_late_or_missing_rate": sum(part["first_stop_late_or_missing"] for part in parts) / total}


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
