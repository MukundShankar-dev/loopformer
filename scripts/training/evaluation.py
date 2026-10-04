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
from .precision import autocast_context
from .objective import forward_symbols
from .objective import (batch_metrics, combine_metrics, completion_loss, completion_metrics,
                        combine_completion_metrics, step_loss, symbolic_scores)


def evaluate(model: RecurrentQwen, examples: list[EncodedExample], token_ids: list[int], pad_id: int,
             *, batch_size: int = 1, loops: int | None = None,
             output: Path | None = None, profile_ranges: bool = False, progress: Callable[[int, int], None] | None = None,
             completion_loss_weight: float = 0.0, precision: str = "float32", record_states: bool = False) -> dict:
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
                with scope("forward"), autocast_context(device, precision):
                    result, scores = forward_symbols(model, batch["input_ids"], batch["attention_mask"], token_ids, num_loops=batch["targets"].shape[1], **({"return_hidden_states": True} if record_states else {}))
                state_values = None
                direct_scores = getattr(result, "state_logits", None)
                direct_predictions = direct_scores.argmax(-1).cpu().tolist() if direct_scores is not None else None
                direct_losses = (step_loss(direct_scores.float(), batch["targets"], batch["target_mask"])[1].cpu().tolist()
                                 if direct_scores is not None else None)
                if record_states:
                    positions = batch["attention_mask"].sum(-1) - 1
                    indices = torch.arange(len(items), device=batch["input_ids"].device)
                    working = torch.stack([h[indices, positions].float() for h in result.hidden_states], dim=1)
                    initial = result.initial_hidden_state[indices, positions].float()
                    previous = torch.cat([initial[:, None], working[:, :-1]], dim=1)
                    state_values = torch.stack([working.norm(dim=-1), (working - previous).norm(dim=-1),
                                                torch.nn.functional.cosine_similarity(working, previous, dim=-1)], dim=-1).cpu().tolist()
                if completion_enabled:
                    result_stop_logits = result.stop_logits.float()
                with scope("loss"):
                    _, losses = step_loss(scores, batch["targets"], batch["target_mask"])
                    if completion_enabled:
                        stop_objective, stop_losses = completion_loss(result_stop_logits, batch["target_mask"])
                with scope("metrics_and_rows"):
                    # Transfer once per batch; row-level metrics must not force a
                    # GPU synchronization for every scalar/loop.
                    scores, losses = scores.cpu(), losses.cpu()
                    batch["targets"] = batch["targets"].cpu()
                    batch["target_mask"] = batch["target_mask"].cpu()
                    if completion_enabled:
                        result_stop_logits = result_stop_logits.cpu()
                        stop_losses, stop_objective = stop_losses.cpu(), stop_objective.cpu()
                    parts.append(batch_metrics(scores, batch["targets"], batch["target_mask"], losses))
                    if completion_enabled:
                        stop_parts.append(completion_metrics(result_stop_logits, batch["target_mask"],
                                                             stop_losses, stop_objective))
                    for row, item in enumerate(items):
                        depth_parts.setdefault(len(item.targets), []).append(batch_metrics(
                            scores[row:row + 1], batch["targets"][row:row + 1], batch["target_mask"][row:row + 1], losses[row:row + 1],
                        ))
                        if completion_enabled:
                            one_mask = batch["target_mask"][row:row + 1]
                            one_logits = result_stop_logits[row:row + 1]
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
                            if direct_predictions is not None:
                                direct = direct_predictions[row_index][t - 1]
                                rows[-1].update(r_prediction=SYMBOLS[direct],
                                    r_intermediate_correct=direct == target if target is not None else "",
                                    r_c_agree=direct == prediction,
                                    r_intermediate_loss=direct_losses[row_index][t - 1] if target is not None else "")
                            if record_states:
                                rows[-1].update(dict(zip(("working_norm", "working_delta_norm", "working_cosine_previous"), state_values[row_index][t-1])))
                            if completion_enabled:
                                rows[-1]["stop_logit"] = result_stop_logits[row_index, t - 1].item()
                                rows[-1]["stop_probability"] = result_stop_logits[row_index, t - 1].sigmoid().item()
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
        if rows and "r_prediction" in rows[0]:
            nominal = [r for r in rows if not r["post_completion"]]
            summary["direct_state"] = {
                "intermediate_accuracy": sum(r["r_intermediate_correct"] for r in nominal) / len(nominal),
                "r_c_agreement": sum(r["r_c_agree"] for r in nominal) / len(nominal),
                "transition_mean_loss": sum(r["r_intermediate_loss"] for r in nominal) / len(nominal),
                "per_loop": {str(t): {
                    "accuracy": sum(r["r_intermediate_correct"] for r in nominal if r["loop"] == t) / sum(r["loop"] == t for r in nominal),
                    "count": sum(r["loop"] == t for r in nominal)} for t in sorted({r["loop"] for r in nominal})}}
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
