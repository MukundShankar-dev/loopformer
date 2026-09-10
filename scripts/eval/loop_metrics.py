"""Trajectory diagnostics derived from the shared evaluator's CSV records."""

from collections import Counter, defaultdict
import csv
from pathlib import Path


def summarize_trajectories(path: Path) -> tuple[list[dict], dict]:
    """Summarize nominal execution separately from the full final-target sweep.

    First-error indices are 1-based and blank for perfect nominal trajectories.
    Repeated decoded symbols are observations, not evidence of inert hidden states.
    """
    groups = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["example_id"]].append(row)
    if not groups:
        raise ValueError("No trajectories to summarize")
    examples = []
    for example_id, rows in groups.items():
        rows.sort(key=lambda row: int(row["loop"]))
        depth = int(rows[0]["task_depth"])
        if [int(row["loop"]) for row in rows] != list(range(1, len(rows) + 1)) or len(rows) < depth:
            raise ValueError(f"Incomplete or duplicate loop trajectory: {example_id}")
        nominal = rows[:depth]
        first_error = next((int(row["loop"]) for row in nominal if row["intermediate_correct"] != "True"), None)
        first_final = next((int(row["loop"]) for row in rows if row["final_correct"] == "True"), None)
        examples.append({
            "example_id": example_id, "split": rows[0]["split"], "seed": rows[0]["seed"],
            "task_depth": depth, "initial_state": rows[0]["initial_state"],
            "targets": " ".join(row["intermediate_target"] for row in nominal),
            "predictions": " ".join(row["prediction"] for row in rows), "executed_loops": len(rows),
            "trajectory_correct": first_error is None,
            "nominal_final_correct": nominal[-1]["final_correct"] == "True",
            "first_error_loop": first_error if first_error is not None else "",
            "correct_prefix_length": depth if first_error is None else first_error - 1,
            "first_final_correct_loop": first_final if first_final is not None else "",
            "nominal_repeated_predictions": sum(a["prediction"] == b["prediction"] for a, b in zip(nominal, nominal[1:])),
        })

    def aggregate(items: list[dict]) -> dict:
        return {
            "examples": len(items),
            "trajectory_correct": sum(item["trajectory_correct"] for item in items),
            "first_error_counts": dict(Counter(str(item["first_error_loop"] or "none") for item in items)),
            "first_final_correct_counts": dict(Counter(str(item["first_final_correct_loop"] or "never") for item in items)),
            "mean_correct_prefix_length": sum(item["correct_prefix_length"] for item in items) / len(items),
            "nominal_repeated_predictions": sum(item["nominal_repeated_predictions"] for item in items),
            "nominal_adjacent_pairs": sum(item["task_depth"] - 1 for item in items),
        }

    return examples, {
        "overall": aggregate(examples),
        "by_depth": {str(d): aggregate([item for item in examples if item["task_depth"] == d])
                     for d in sorted({item["task_depth"] for item in examples})},
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write nonempty flat records with a header."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def depth_loop_rows(metrics: dict) -> list[dict]:
    """Flatten the full sweep; nominal loss/accuracy are blank after task depth."""
    rows = []
    for depth, loops in metrics["depth_by_loop"].items():
        for loop, values in loops.items():
            nominal = metrics["by_depth"][depth]["per_loop"].get(loop)
            rows.append({
                "task_depth": int(depth), "loop": int(loop), "examples": values["count"],
                "post_completion": int(loop) > int(depth),
                "final_accuracy": values["final_accuracy"],
                "intermediate_accuracy": nominal["accuracy"] if nominal else "",
                "intermediate_loss": nominal["loss"] if nominal else "",
            })
    return rows
