"""Per-loop evaluation with nominal targets and an explicitly separate final readout."""

import csv
from pathlib import Path
from typing import Callable

import torch

from scripts.dataset.pointer import SYMBOLS
from scripts.recurrent_qwen.model import RecurrentQwen
from .data import EncodedExample, collate
from .objective import batch_metrics, combine_metrics, step_loss, symbolic_scores


def evaluate(model: RecurrentQwen, examples: list[EncodedExample], token_ids: list[int], pad_id: int,
             *, batch_size: int = 1, loops: int | None = None,
             output: Path | None = None, progress: Callable[[int, int], None] | None = None) -> dict:
    """Nominal loss/accuracy only at t<=d; final-target sweep is observational.

    Extra loops do not receive targets or retention loss. Their final correctness
    must not be interpreted as damage from a valid continuing pointer transition.
    """
    if not examples or batch_size < 1:
        raise ValueError("Evaluation requires examples and a positive batch size")
    device = str(next(model.parameters()).device)
    was_training = model.training
    model.eval()
    parts, rows, depth_parts = [], [], {}
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                items = examples[start:start + batch_size]
                batch = collate(items, pad_id, device, loops)
                result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], num_loops=batch["targets"].shape[1])
                scores = symbolic_scores(result.loop_logits, token_ids)
                _, losses = step_loss(scores, batch["targets"], batch["target_mask"])
                parts.append(batch_metrics(scores, batch["targets"], batch["target_mask"], losses))
                for row, item in enumerate(items):
                    depth_parts.setdefault(len(item.targets), []).append(batch_metrics(
                        scores[row:row + 1], batch["targets"][row:row + 1], batch["target_mask"][row:row + 1], losses[row:row + 1],
                    ))
                predictions = scores.argmax(-1).cpu().tolist()
                values = scores.cpu().tolist()
                for item, predicted, logits in zip(items, predictions, values):
                    final = item.targets[-1]
                    for t, (prediction, scores_t) in enumerate(zip(predicted, logits), 1):
                        target = item.targets[t - 1] if t <= len(item.targets) else None
                        rows.append({
                            "example_id": item.task.example_id, "task_depth": len(item.targets), "loop": t,
                            "prediction": SYMBOLS[prediction],
                            "intermediate_target": SYMBOLS[target] if target is not None else "",
                            "intermediate_correct": prediction == target if target is not None else "",
                            "final_target": SYMBOLS[final], "final_correct": prediction == final,
                            "post_completion": t > len(item.targets),
                            "final_margin": scores_t[final] - max(value for j, value in enumerate(scores_t) if j != final),
                        })
                if progress:
                    progress(min(start + batch_size, len(examples)), len(examples))
    finally:
        model.train(was_training)
    summary = combine_metrics(parts)
    summary["by_depth"] = {str(depth): combine_metrics(values) for depth, values in sorted(depth_parts.items())}
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
