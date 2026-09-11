"""Fixed-final-target dynamics on explicitly validated absorbing-terminal tasks."""

from collections import defaultdict
import csv
import math
from pathlib import Path

from scripts.dataset.pointer import PointerExample, SYMBOLS, validate_example


def overscaling_metrics(path: Path, tasks: list[PointerExample]) -> tuple[dict[str, list[dict]], dict]:
    """Return adjacent transitions, rates, and continuously correct survival.

    Report all-loop observations separately from t>=d transitions. Only the latter
    measure post-nominal repair/damage. Each rate uses the same cohort at t and
    t+1. Empty conditional denominators yield None (blank CSV / null JSON).
    Survival begins at the first final-correct readout at or after d; unobserved
    offsets are censored, and later recovery never restores continuous survival.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["example_id"]].append(row)
    if not tasks or len({t.example_id for t in tasks}) != len(tasks) or set(groups) != {t.example_id for t in tasks}:
        raise ValueError("Trajectories must cover exactly the unique selected tasks")
    transitions, solutions = [], []
    budgets = set()
    for task in tasks:
        validate_example(task)
        if dict(task.mapping)[task.final_state] != task.final_state:
            raise ValueError("Dynamics require an absorbing-terminal mapping")
        rows = sorted(groups[task.example_id], key=lambda r: int(r["loop"]))
        depth = task.task_depth
        if [int(r["loop"]) for r in rows] != list(range(1, len(rows) + 1)) or len(rows) <= depth:
            raise ValueError("Require complete, unique loops and at least one loop past every task depth")
        budgets.add(len(rows))
        correct = []
        for t, row in enumerate(rows, 1):
            value = row["prediction"] == task.final_state
            if (int(row["task_depth"]) != depth or row["final_target"] != task.final_state
                    or row["prediction"] not in SYMBOLS or row["final_correct"] != str(value)
                    or not math.isfinite(float(row["final_margin"]))):
                raise ValueError(f"Inconsistent final-target trajectory: {task.example_id}")
            correct.append(value)
        first = next((t for t in range(depth, len(rows) + 1) if correct[t - 1]), None)
        solutions.append({
            "example_id": task.example_id, "task_depth": depth,
            "first_correct_at_or_after_depth": first,
            "early_final_match": any(correct[:depth - 1]),
            "nominal_correct": correct[depth - 1], "last_correct": correct[-1],
            "correct": correct,
        })
        for t in range(1, len(rows)):
            a, b = correct[t - 1:t + 1]
            transitions.append({
                "example_id": task.example_id, "task_depth": depth, "loop": t, "next_loop": t + 1,
                "post_nominal": t >= depth, "final_target": task.final_state,
                "prediction": rows[t - 1]["prediction"], "next_prediction": rows[t]["prediction"],
                "correct": a, "next_correct": b,
                "transition": f"{'right' if a else 'wrong'}_to_{'right' if b else 'wrong'}",
                "final_margin": float(rows[t - 1]["final_margin"]),
                "next_final_margin": float(rows[t]["final_margin"]),
            })
    if len(budgets) != 1:
        raise ValueError("Every example must have the same loop budget")
    loops = budgets.pop()

    def fraction(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    rates, survival, summaries = [], [], {}
    for depth in [None, *sorted({t.task_depth for t in tasks})]:
        label = "all" if depth is None else str(depth)
        cohort = [s for s in solutions if depth is None or s["task_depth"] == depth]
        selected = [r for r in transitions if depth is None or r["task_depth"] == depth]
        for scope in ("all_loops_observational", "post_nominal"):
            for t in range(1, loops):
                pairs = [r for r in selected if r["loop"] == t and (scope != "post_nominal" or r["post_nominal"])]
                counts = {kind: sum(r["transition"] == kind for r in pairs) for kind in
                          ("wrong_to_wrong", "wrong_to_right", "right_to_right", "right_to_wrong")}
                wrong = counts["wrong_to_wrong"] + counts["wrong_to_right"]
                right = counts["right_to_right"] + counts["right_to_wrong"]
                rates.append({
                    "task_depth": label, "scope": scope, "loop": t, "next_loop": t + 1,
                    "examples": len(pairs), **counts, "wrong_denominator": wrong, "right_denominator": right,
                    "repair_rate": fraction(counts["wrong_to_right"], wrong),
                    "damage_rate": fraction(counts["right_to_wrong"], right),
                    "accuracy_at_t": fraction(right, len(pairs)),
                    "accuracy_at_next": fraction(counts["wrong_to_right"] + counts["right_to_right"], len(pairs)),
                    "net_gain": fraction(counts["wrong_to_right"] - counts["right_to_wrong"], len(pairs)),
                })
        solved = [s for s in cohort if s["first_correct_at_or_after_depth"] is not None]
        for offset in range(loops):
            observed = [s for s in solved if s["first_correct_at_or_after_depth"] + offset <= loops]
            survived = sum(all(s["correct"][s["first_correct_at_or_after_depth"] - 1:
                                           s["first_correct_at_or_after_depth"] + offset]) for s in observed)
            survival.append({
                "task_depth": label, "extra_loops": offset, "solved": len(solved),
                "never_solved": len(cohort) - len(solved), "observed": len(observed),
                "censored": len(solved) - len(observed), "continuously_correct": survived,
                "survival": fraction(survived, len(observed)),
            })
        summaries[label] = {
            "examples": len(cohort), "solved_at_or_after_depth": len(solved),
            "never_solved_at_or_after_depth": len(cohort) - len(solved),
            "early_final_matches": sum(s["early_final_match"] for s in cohort),
            "nominal_final_accuracy": sum(s["nominal_correct"] for s in cohort) / len(cohort),
            "last_loop_accuracy": sum(s["last_correct"] for s in cohort) / len(cohort),
            "hold_nominal_prediction_baseline_accuracy": sum(s["nominal_correct"] for s in cohort) / len(cohort),
        }
    solution_rows = [{k: v for k, v in s.items() if k != "correct"} for s in solutions]
    return {"transitions.csv": transitions, "transition_rates.csv": rates,
            "survival.csv": survival, "solutions.csv": solution_rows}, summaries
