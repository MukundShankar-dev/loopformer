"""Denominator-explicit summaries for the frozen pointer mechanism suite."""

from collections import Counter, defaultdict
from statistics import mean

import torch

from scripts.dataset.pointer import PointerExample


def first_error(rows: list[dict]) -> int | None:
    return next((row["loop"] for row in rows if not row["correct"]), None)


def baseline_case(task: PointerExample, group: str, rows: list[dict]) -> dict:
    error = first_error(rows)
    path = [task.initial_state, *task.intermediate_states]
    category = ""
    if error is not None:
        prediction = rows[error - 1]["prediction"]
        category = ("repeat_previous" if prediction == path[error - 1] else
                    "earlier_path" if prediction in path[:error - 1] else
                    "future_path" if prediction in path[error + 1:] else "off_path")
    return {"example_id": task.example_id, "group": group, "task_depth": task.task_depth,
            "first_error_loop": error or "", "correct_prefix_length": (error - 1) if error else task.task_depth,
            "complete": error is None, "first_error_category": category,
            "first_error_prediction": rows[error - 1]["prediction"] if error else "",
            "first_error_target": rows[error - 1]["target"] if error else ""}


def baseline_summary(cases: list[dict]) -> tuple[dict, list[dict]]:
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["group"], case["task_depth"])].append(case)
    by_depth = {}
    for (group, depth), members in sorted(grouped.items()):
        by_depth.setdefault(group, {})[str(depth)] = {
            "examples": len(members), "complete": sum(case["complete"] for case in members),
            "preloop_matches_start": sum(case["preloop_matches_start"] for case in members),
            "preloop_matches_first_target": sum(case["preloop_matches_first_target"] for case in members),
            "mean_correct_prefix": mean(case["correct_prefix_length"] for case in members),
            "first_errors": dict(Counter(str(case["first_error_loop"]) for case in members if case["first_error_loop"])),
        }
    risk = []
    for (group, depth), members in sorted(grouped.items()):
        for loop in range(1, depth + 1):
            at_risk = sum(case["correct_prefix_length"] >= loop - 1 for case in members)
            failures = sum(case["first_error_loop"] == loop for case in members)
            risk.append({"group": group, "task_depth": depth, "loop": loop,
                         "correct_prefix_examples": at_risk, "first_failures": failures,
                         "conditional_failure_rate": failures / at_risk if at_risk else ""})
    return {"by_depth": by_depth,
            "first_error_categories": dict(Counter(case["first_error_category"] for case in cases if case["first_error_category"]))}, risk


def paired_summary(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["variant"])].append(row)
    result = {}
    for (group, variant), items in sorted(grouped.items()):
        examples = defaultdict(list)
        for item in items:
            examples[item["example_id"]].append(item)
        result.setdefault(group, {})[variant] = {
            "examples": len(examples), "paired_steps": len(items),
            "examples_with_token_length_change": len({item["example_id"] for item in items
                                                       if item["token_length_changed"]}),
            "equivariant_predictions": sum(item["equivariant_prediction"] for item in items),
            "variant_correct_steps": sum(item["variant_correct"] for item in items),
            "baseline_correct_steps": sum(item["baseline_correct"] for item in items),
            "variant_complete_trajectories": sum(all(row["variant_correct"] for row in values)
                                                 for values in examples.values()),
            "baseline_complete_trajectories": sum(all(row["baseline_correct"] for row in values)
                                                  for values in examples.values()),
        }
    return result


def directed_summary(rows: list[dict]) -> dict:
    probes = [row for row in rows if row["loop"] == row["probe_loop"]]
    pairs = defaultdict(dict)
    for row in probes:
        key = (row["phase"], row["example_id"], row["replacement_index"])
        pairs[key][row["condition"]] = row
    result = {}
    for phase in ("early", "critical"):
        matched = [(key, pair["relevant"], pair["irrelevant"]) for key, pair in pairs.items()
                   if key[0] == phase and set(pair) == {"relevant", "irrelevant"}
                   and pair["relevant"]["edited_prefix_correct"] and pair["irrelevant"]["edited_prefix_correct"]]
        follows = [key for key, relevant, _ in matched if relevant["edited_correct"]]
        per_case = defaultdict(list)
        for key, relevant, irrelevant in matched:
            per_case[key[1]].append(relevant["edited_correct"])
        result[phase] = {
            "attempted_pairs": sum(key[0] == phase for key in pairs),
            "matched_correct_prefix_pairs": len(matched),
            "replacement_targets_attempted": dict(Counter(row["new_destination"] for row in probes
                                                        if row["phase"] == phase and row["condition"] == "relevant")),
            "relevant_follows_new_target": len(follows),
            "irrelevant_keeps_original_target": sum(irrelevant["edited_correct"] for _, _, irrelevant in matched),
            "relevant_prediction_changed": sum(relevant["prediction_changed"] for _, relevant, _ in matched),
            "irrelevant_prediction_changed": sum(irrelevant["prediction_changed"] for _, _, irrelevant in matched),
            "cases_with_both_replacements_retaining_prefix": sum(len(values) == 2 for values in per_case.values()),
            "cases_following_both_distinct_replacements": sum(len(values) == 2 and all(values)
                                                              for values in per_case.values()),
        }
        if phase == "early":
            by_variant = defaultdict(dict)
            for row in rows:
                if row["phase"] == "early":
                    by_variant[(row["example_id"], row["replacement_index"], row["condition"])][row["loop"]] = row
            continuation = {}
            for offset in (1, 2, 3):
                eligible = []
                for key, relevant, irrelevant in matched:
                    loops = range(relevant["probe_loop"] + 1, relevant["probe_loop"] + offset + 1)
                    relevant_rows = by_variant[(key[1], key[2], "relevant")]
                    irrelevant_rows = by_variant[(key[1], key[2], "irrelevant")]
                    if relevant["edited_correct"] and irrelevant["edited_correct"] and all(
                            loop in relevant_rows and loop in irrelevant_rows for loop in loops):
                        eligible.append((relevant_rows, irrelevant_rows, loops))
                continuation[str(offset)] = {
                    "eligible_after_both_correct_edit_steps": len(eligible),
                    "relevant_contiguous_correct": sum(all(relevant_rows[loop]["edited_correct"] for loop in loops)
                                                       for relevant_rows, _, loops in eligible),
                    "irrelevant_contiguous_correct": sum(all(irrelevant_rows[loop]["edited_correct"] for loop in loops)
                                                         for _, irrelevant_rows, loops in eligible),
                }
            result[phase]["branch_continuation"] = continuation
    return result


def attention_summary(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        bucket = "early_1_to_6" if row["loop"] <= 6 else "near_7_to_8" if row["loop"] <= 8 else "late_9_plus"
        grouped[(row["group"], bucket, bool(row["correct"]))].append(row)
    return {f"{group}/{bucket}/{'correct' if correct else 'wrong'}": {
                "layer_loop_rows": len(items),
                "mean_destination_top1_heads": mean(item["destination_top1_heads"] for item in items),
                "mean_source_top1_heads": mean(item["source_top1_heads"] for item in items),
            } for (group, bucket, correct), items in sorted(grouped.items())}


def attention_rows(task: PointerExample, group: str, original: list[dict],
                   source: torch.Tensor, destination: torch.Tensor) -> list[dict]:
    """Summarize attention from Answer to the reference active rule, per R layer."""
    if source.shape != destination.shape or source.ndim != 4 or source.shape[0] != task.task_depth or source.shape[-1] != len(task.mapping):
        raise ValueError("Attention capture shape does not match task")
    pair_index = {key: index for index, (key, _) in enumerate(task.mapping)}
    path = [task.initial_state, *task.intermediate_states]
    rows = []
    for t in range(task.task_depth):
        active = pair_index[path[t]]
        for layer in range(source.shape[1]):
            row = {"example_id": task.example_id, "group": group, "task_depth": task.task_depth,
                   "loop": t + 1, "recurrent_layer": layer, "correct": original[t]["correct"],
                   "active_source": path[t], "active_destination": path[t + 1]}
            for name, tensor in (("source", source), ("destination", destination)):
                values = tensor[t, layer]
                active_values = values[:, active]
                ranks = (values > active_values[:, None]).sum(-1) + 1
                row[f"{name}_mean_active_rank"] = ranks.float().mean().item()
                row[f"{name}_top1_heads"] = int((ranks == 1).sum().item())
                row[f"{name}_mean_active_attention"] = active_values.mean().item()
                row[f"{name}_max_active_attention"] = active_values.max().item()
            rows.append(row)
    return rows
