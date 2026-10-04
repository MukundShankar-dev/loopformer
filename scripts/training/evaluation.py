"""Per-loop evaluation with nominal targets and an explicitly separate final readout."""

from contextlib import nullcontext
import csv
import math
from pathlib import Path
from typing import Callable

import torch

from scripts.dataset.pointer import SYMBOLS
from scripts.recurrent_qwen.model import RecurrentQwen
from .data import EncodedExample, collate
from .objective import (batch_metrics, combine_metrics, completion_loss, completion_metrics,
                        combine_completion_metrics, step_loss, symbolic_scores)


def evaluate(model: RecurrentQwen, examples: list[EncodedExample], token_ids: list[int], pad_id: int,
             *, batch_size: int = 1, loops: int | None = None,
             output: Path | None = None, profile_ranges: bool = False, progress: Callable[[int, int], None] | None = None,
             completion_loss_weight: float = 0.0) -> dict:
    """Nominal loss/accuracy only at t<=d; final-target sweep is observational.

    Extra loops do not receive targets or retention loss. Their final correctness
    must not be interpreted as damage from a valid continuing pointer transition.
    """
    if not examples or batch_size < 1:
        raise ValueError("Evaluation requires examples and a positive batch size")
    if completion_loss_weight < 0 or (completion_loss_weight > 0 and getattr(model, "completion_head", None) is None):
        raise ValueError("Positive completion evaluation weight requires a model head")
    completion_enabled = getattr(model, "completion_head", None) is not None
    def scope(name: str):
        return torch.profiler.record_function("pointer/" + name) if profile_ranges else nullcontext()

    device = str(next(model.parameters()).device)
    was_training = model.training
    model.eval()
    parts, rows, depth_parts, stop_parts, stop_depth_parts = [], [], {}, [], {}
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                items = examples[start:start + batch_size]
                with scope("input_prepare_transfer"):
                    batch = collate(items, pad_id, device, loops)
                with scope("forward"):
                    result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], num_loops=batch["targets"].shape[1])
                with scope("loss"):
                    scores = symbolic_scores(result.loop_logits, token_ids)
                    _, losses = step_loss(scores, batch["targets"], batch["target_mask"])
                    if completion_enabled:
                        stop_objective, stop_losses = completion_loss(result.stop_logits, batch["target_mask"])
                with scope("metrics_and_rows"):
                    parts.append(batch_metrics(scores, batch["targets"], batch["target_mask"], losses))
                    if completion_enabled:
                        stop_parts.append(completion_metrics(result.stop_logits, batch["target_mask"],
                                                             stop_losses, stop_objective))
                    for row, item in enumerate(items):
                        depth_parts.setdefault(len(item.targets), []).append(batch_metrics(
                            scores[row:row + 1], batch["targets"][row:row + 1], batch["target_mask"][row:row + 1], losses[row:row + 1],
                        ))
                        if completion_enabled:
                            one_mask = batch["target_mask"][row:row + 1]
                            one_logits = result.stop_logits[row:row + 1]
                            one_objective, one_losses = completion_loss(one_logits, one_mask)
                            stop_depth_parts.setdefault(len(item.targets), []).append(completion_metrics(
                                one_logits, one_mask, one_losses, one_objective))
                    predictions = scores.argmax(-1).cpu().tolist()
                    values = scores.cpu().tolist()
                    for row_index, (item, predicted, logits) in enumerate(zip(items, predictions, values)):
                        final = item.targets[-1]
                        for t, (prediction, scores_t) in enumerate(zip(predicted, logits), 1):
                            ranked = sorted(scores_t, reverse=True)
                            predicted_margin = ranked[0] - ranked[1]
                            normalized = [math.exp(value - ranked[0]) for value in scores_t]
                            total = sum(normalized)
                            probabilities = [value / total for value in normalized]
                            answer_entropy = -sum(p * math.log(p) for p in probabilities if p)
                            target = item.targets[t - 1] if t <= len(item.targets) else None
                            rows.append({
                                "example_id": item.task.example_id, "task_depth": len(item.targets), "loop": t,
                                "split": item.task.split, "seed": item.task.seed, "family": item.task.family,
                                "initial_state": item.task.initial_state,
                                "prediction": SYMBOLS[prediction],
                                "intermediate_target": SYMBOLS[target] if target is not None else "",
                                "intermediate_correct": prediction == target if target is not None else "",
                                "final_target": SYMBOLS[final], "final_correct": prediction == final,
                                "post_completion": t > len(item.targets),
                                "intermediate_loss": losses[row_index, t - 1].item() if target is not None else "",
                                "intermediate_margin": (scores_t[target] - max(value for j, value in enumerate(scores_t) if j != target)) if target is not None else "",
                                "final_margin": scores_t[final] - max(value for j, value in enumerate(scores_t) if j != final),
                                "predicted_margin": predicted_margin, "answer_entropy": answer_entropy,
                                "target_rank": (1 + sum(value > scores_t[target] or (value == scores_t[target] and j < target)
                                                        for j, value in enumerate(scores_t))) if target is not None else "",
                                "top_symbols": " ".join(SYMBOLS[j] for j in sorted(range(len(scores_t)), key=lambda j: (-scores_t[j], j))[:3]),
                            })
                            if completion_enabled:
                                rows[-1]["stop_logit"] = result.stop_logits[row_index, t - 1].item()
                                rows[-1]["stop_probability"] = result.stop_logits[row_index, t - 1].sigmoid().item()
                                rows[-1]["stop_target"] = (t == len(item.targets)) if t <= len(item.targets) else ""
                                rows[-1]["stop_loss"] = stop_losses[row_index, t - 1].item() if t <= len(item.targets) else ""
                if progress:
                    progress(min(start + batch_size, len(examples)), len(examples))
    finally:
        model.train(was_training)
    with scope("aggregate_and_export"):
        summary = combine_metrics(parts)
        summary["by_depth"] = {str(depth): combine_metrics(values) for depth, values in sorted(depth_parts.items())}
        if completion_enabled:
            summary["completion"] = combine_completion_metrics(stop_parts)
            for depth, values in stop_depth_parts.items():
                summary["by_depth"][str(depth)]["completion"] = combine_completion_metrics(values)
        nominal_final = [row for row in rows if row["loop"] == row["task_depth"]]
        summary["final_accuracy"] = sum(row["final_correct"] for row in nominal_final) / len(nominal_final)
        summary["depth_by_loop"] = {}
        for depth in sorted({row["task_depth"] for row in rows}):
            summary["depth_by_loop"][str(depth)] = {}
            for t in sorted({row["loop"] for row in rows if row["task_depth"] == depth}):
                selected = [row for row in rows if row["task_depth"] == depth and row["loop"] == t]
                summary["depth_by_loop"][str(depth)][str(t)] = {
                    "count": len(selected), "final_accuracy": sum(row["final_correct"] for row in selected) / len(selected),
                }
        if output:
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return summary
