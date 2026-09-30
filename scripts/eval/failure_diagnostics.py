"""Reference-checked nominal failure diagnostics from complete fixed-depth traces."""

from collections import Counter, defaultdict
import csv
from pathlib import Path

from scripts.dataset.pointer import PointerExample, SYMBOLS, validate_example


def analyze_failures(path: Path, tasks: list[PointerExample], loops: int) -> tuple[list[dict], list[dict], dict]:
    """Return first failures, depth/loop risk sets, and summary; no post-depth claims.

    Risk sets contain only examples with every preceding nominal step correct.
    Categories use the non-repeating reference path, including the initial state.
    """
    references = {t.example_id: t for t in tasks}
    if not references or len(references) != len(tasks):
        raise ValueError("Need nonempty, unique reference tasks")
    for task in tasks:
        validate_example(task)
    groups = defaultdict(dict)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key, loop = row["example_id"], int(row["loop"])
            if key not in references or loop in groups[key]:
                raise ValueError("Unknown example or duplicate loop in trace")
            groups[key][loop] = row
    if groups.keys() != references.keys():
        raise ValueError("Trace/reference example coverage differs")
    failures, counts = [], defaultdict(lambda: [0, 0])
    for key, task in references.items():
        rows = groups[key]
        if sorted(rows) != list(range(1, loops + 1)) or loops < task.task_depth:
            raise ValueError(f"Incomplete fixed-depth trace: {key}")
        path_states = [task.initial_state, *task.intermediate_states]
        first_error = None
        for t, row in sorted(rows.items()):
            prediction = row["prediction"]
            if (int(row["task_depth"]) != task.task_depth or prediction not in SYMBOLS
                    or row["initial_state"] != task.initial_state
                    or row["final_target"] != task.final_state
                    or row["final_correct"] != str(prediction == task.final_state)):
                raise ValueError(f"Invalid trace metadata/scoring: {key}")
            if t > task.task_depth:
                if row["intermediate_target"] or row["intermediate_correct"]:
                    raise ValueError("Post-depth intermediate labels must be empty")
                continue
            correct = prediction == task.intermediate_states[t - 1]
            if row["intermediate_target"] != task.intermediate_states[t - 1] or row["intermediate_correct"] != str(correct):
                raise ValueError(f"Reference target/scoring mismatch: {key}")
            if first_error is not None:
                continue
            counts[task.task_depth, t][0] += 1
            if correct:
                continue
            first_error = t
            counts[task.task_depth, t][1] += 1
            category = ("repeat_previous" if prediction == path_states[t - 1] else
                        "earlier_path" if prediction in path_states[:t - 1] else
                        "future_path" if prediction in path_states[t + 1:] else "off_path")
            failure = {"example_id": key, "task_depth": task.task_depth, "first_error_loop": t,
                       "correct_prefix_length": t - 1, "prediction": prediction,
                       "target": path_states[t], "previous_state": path_states[t - 1], "category": category}
            for field in ("intermediate_margin", "predicted_margin", "answer_entropy", "target_rank", "top_symbols"):
                failure[field] = row.get(field, "")
                failure["previous_" + field] = rows[t - 1].get(field, "") if t > 1 else ""
            failures.append(failure)
    risk = []
    for depth in sorted({t.task_depth for t in tasks}):
        for t in range(1, depth + 1):
            n, f = counts[depth, t]
            risk.append({"task_depth": depth, "loop": t, "correct_prefix_examples": n,
                         "first_failures": f, "conditional_failure_rate": f / n if n else ""})
    summary = {"examples": len(tasks), "complete_trajectories": len(tasks) - len(failures),
               "first_error_categories": dict(Counter(r["category"] for r in failures)),
               "semantics": "Nominal first errors only; empty risk-set rates are undefined; no hidden-state or post-completion damage claim"}
    return failures, risk, summary
