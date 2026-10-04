"""Matched-instance checkpoint and displayed-horizon comparisons."""

from collections import defaultdict
import random
from statistics import mean
from typing import Any

from rich.progress import track

from scripts.dataset.pointer import PointerExample, SYMBOLS, render_prompt
from scripts.dataset.symbols import validate_prompt_tokens
from scripts.eval.pointer_probes import inspect_example
from scripts.training.data import EncodedExample


def _encoded(task: PointerExample, prompt: str, tokenizer: Any,
             token_map: dict[str, int], max_tokens: int) -> EncodedExample:
    validate_prompt_tokens(tokenizer, prompt, token_map)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_tokens:
        raise ValueError(f"Comparison prompt exceeds context: {task.example_id}")
    return EncodedExample(task, ids, [SYMBOLS.index(symbol) for symbol in task.intermediate_states])


def comparison_prompts(task: PointerExample, train_max_depth: int,
                       fixed_horizon: int) -> dict[str, tuple[int, str]]:
    """Keep Rules/Start fixed; only the displayed Steps text can differ."""
    if task.task_depth <= train_max_depth or fixed_horizon <= task.task_depth:
        raise ValueError("Comparison requires a deep task and a longer fixed horizon")
    return {
        "native": (task.task_depth, task.prompt),
        "trained_cue": (train_max_depth, render_prompt(task.mapping, task.initial_state, train_max_depth)),
        "fixed_long_cue": (fixed_horizon, render_prompt(task.mapping, task.initial_state, fixed_horizon)),
    }


def comparison_checkpoint(model: Any, tokenizer: Any, spec: dict,
                          selected: list[tuple[str, PointerExample]], checkpoint: str,
                          *, original_rows: dict[str, list[dict]] | None,
                          console: Any) -> tuple[list[dict], list[dict], list[dict]]:
    """Score identical mappings at three cues and both six/full actual unrolls.

    Short-unroll outputs are a prefix-invariance contract, not an independent
    generalization score. A trained-depth cue is valid only through that depth.
    """
    train_max = int(spec["train_max_depth"])
    fixed_horizon = max(task.task_depth for group, task in selected if group == "deep") + 1
    token_map = dict(zip(spec["symbols"], spec["token_ids"], strict=True))
    rows, prefix_rows, inputs = [], [], []
    for group, task in track(selected, description=f"Matched cues · {checkpoint}", console=console):
        prompts = (comparison_prompts(task, train_max, fixed_horizon) if group == "deep"
                   else {"native": (task.task_depth, task.prompt)})
        for cue, (displayed, prompt) in prompts.items():
            item = _encoded(task, prompt, tokenizer, token_map, model.config.max_position_embeddings)
            if cue == "native" and original_rows is not None:
                full = original_rows[task.example_id]
            else:
                full = inspect_example(model, item, spec["token_ids"], tokenizer.pad_token_id)
            if len(full) != task.task_depth:
                raise ValueError("Full comparison trajectory length mismatch")
            inputs.append({"example_id": task.example_id, "group": group, "cue": cue,
                           "displayed_steps": displayed, "executed_loops": task.task_depth,
                           "short_executed_loops": train_max if group == "deep" else None,
                           "checkpoint": checkpoint, "prompt": prompt,
                           "input_ids": item.input_ids, "targets": task.intermediate_states})
            if group == "deep":
                prefix_item = EncodedExample(task, item.input_ids, item.targets[:train_max])
                short = inspect_example(model, prefix_item, spec["token_ids"], tokenizer.pad_token_id)
                for full_row, short_row in zip(full[:train_max], short, strict=True):
                    unchanged = full_row["prediction"] == short_row["prediction"]
                    prefix_rows.append({"checkpoint": checkpoint, "example_id": task.example_id,
                                        "task_depth": task.task_depth, "cue": cue,
                                        "displayed_steps": displayed, "loop": full_row["loop"],
                                        "full_prediction": full_row["prediction"],
                                        "short_prediction": short_row["prediction"],
                                        "prefix_invariant": unchanged})
                    if not unchanged:
                        raise ValueError("Changing only requested actual unroll changed an earlier prediction")
            for step in full:
                rows.append({"checkpoint": checkpoint, "example_id": task.example_id,
                             "group": group, "task_depth": task.task_depth, "cue": cue,
                             "initial_state": task.initial_state,
                             "displayed_steps": displayed, "executed_loops": task.task_depth,
                             "input_tokens": len(item.input_ids), "loop": step["loop"],
                             "target": step["target"], "prediction": step["prediction"],
                             "correct": step["correct"],
                             "within_displayed_horizon": step["loop"] <= displayed,
                             "target_margin": step["target_margin"]})
    return rows, prefix_rows, inputs


def comparison_cases(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["checkpoint"], row["example_id"], row["cue"])].append(row)
    result = []
    for (checkpoint, example_id, cue), items in sorted(grouped.items()):
        items.sort(key=lambda row: row["loop"])
        first_error = next((row["loop"] for row in items if not row["correct"]), None)
        path = [items[0]["initial_state"], *(row["target"] for row in items)]
        category = ""
        if first_error is not None:
            prediction = items[first_error - 1]["prediction"]
            category = ("repeat_previous" if prediction == path[first_error - 1] else
                        "earlier_path" if prediction in path[:first_error - 1] else
                        "future_path" if prediction in path[first_error + 1:] else "off_path")
        result.append({"checkpoint": checkpoint, "example_id": example_id,
                       "group": items[0]["group"], "task_depth": items[0]["task_depth"],
                       "cue": cue, "displayed_steps": items[0]["displayed_steps"],
                       "first_error_loop": first_error or "",
                       "first_error_category": category,
                       "first_error_within_displayed_horizon":
                       first_error <= items[0]["displayed_steps"] if first_error else "",
                       "correct_prefix_length": first_error - 1 if first_error else len(items),
                       "complete_scored_trajectory": first_error is None,
                       "valid_prefix_correct": all(row["correct"] for row in items
                                                   if row["within_displayed_horizon"])})
    return result


def _bootstrap_paired(values: list[tuple[int, float]], seed: int,
                      draws: int = 2000) -> list[float]:
    """Stratified case-level percentile interval for the paired mean change."""
    if not values:
        return []
    groups = defaultdict(list)
    for depth, difference in values:
        groups[depth].append(difference)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        resampled = [rng.choice(group) for _, group in sorted(groups.items())
                     for _ in range(len(group))]
        samples.append(mean(resampled))
    samples.sort()
    return [samples[int(.025 * draws)], samples[int(.975 * draws)]]


def summarize_comparison(rows: list[dict], checkpoint_order: list[str],
                         train_max_depth: int, seed: int) -> dict:
    """Use mapping-level pairs; never count correlated loop rows as cases."""
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["checkpoint"], row["example_id"], row["cue"])].append(row)
    for group in by_key.values():
        group.sort(key=lambda row: row["loop"])
    reference = checkpoint_order[0]
    by_checkpoint = {}
    for checkpoint in checkpoint_order:
        group_summary = {}
        for group in ("validation", "deep"):
            cases = [values for (name, _, cue), values in by_key.items()
                     if name == checkpoint and cue == "native" and values[0]["group"] == group]
            loop_groups = defaultdict(list)
            for case in cases:
                for row in case:
                    loop_groups[row["loop"]].append(row)
            group_summary[group] = {
                "examples": len(cases),
                "complete": sum(all(row["correct"] for row in case) for case in cases),
                "by_depth": {str(depth): {
                    "examples": sum(case[0]["task_depth"] == depth for case in cases),
                    "complete": sum(case[0]["task_depth"] == depth and all(row["correct"] for row in case)
                                    for case in cases)}
                    for depth in sorted({case[0]["task_depth"] for case in cases})},
                "by_loop": {str(loop): {"correct": sum(row["correct"] for row in items),
                                        "examples": len(items)}
                            for loop, items in sorted(loop_groups.items())},
            }
        by_checkpoint[checkpoint] = group_summary

    cue_pairs = {}
    for checkpoint in checkpoint_order:
        cue_pairs[checkpoint] = {}
        deep_ids = sorted({key[1] for key in by_key if key[0] == checkpoint
                           and key[2] == "native" and by_key[key][0]["group"] == "deep"})
        for cue in ("trained_cue", "fixed_long_cue"):
            valid_differences, stress_differences = [], []
            same_length_differences = []
            segment_differences = defaultdict(list)
            same_length_segment_differences = defaultdict(list)
            segment_steps = defaultdict(list)
            valid_same, valid_total, stress_same, stress_total = 0, 0, 0, 0
            changed_cases, total_cases = 0, 0
            token_changes = 0
            for example_id in deep_ids:
                native, changed = by_key[(checkpoint, example_id, "native")], by_key[(checkpoint, example_id, cue)]
                if len(native) != len(changed):
                    raise ValueError("Matched cue trajectories have different lengths")
                total_cases += 1
                token_changes += native[0]["input_tokens"] != changed[0]["input_tokens"]
                changed_cases += any(a["prediction"] != b["prediction"]
                                     for a, b in zip(native, changed, strict=True))
                valid = [(a, b) for a, b in zip(native, changed, strict=True)
                         if b["within_displayed_horizon"]]
                stress = [(a, b) for a, b in zip(native, changed, strict=True)
                          if not b["within_displayed_horizon"]]
                valid_total += len(valid)
                valid_same += sum(a["prediction"] == b["prediction"] for a, b in valid)
                stress_total += len(stress)
                stress_same += sum(a["prediction"] == b["prediction"] for a, b in stress)
                valid_differences.append((native[0]["task_depth"],
                                          mean(int(b["correct"]) - int(a["correct"]) for a, b in valid)))
                if native[0]["input_tokens"] == changed[0]["input_tokens"]:
                    same_length_differences.append(valid_differences[-1])
                for segment, subset in (("early_1_to_train_max", [(a, b) for a, b in valid
                                                                  if b["loop"] <= train_max_depth]),
                                        ("late_after_train_max", [(a, b) for a, b in valid
                                                                  if b["loop"] > train_max_depth])):
                    if subset:
                        segment_differences[segment].append((native[0]["task_depth"],
                            mean(int(b["correct"]) - int(a["correct"]) for a, b in subset)))
                        if native[0]["input_tokens"] == changed[0]["input_tokens"]:
                            same_length_segment_differences[segment].append(segment_differences[segment][-1])
                        segment_steps[segment].extend(subset)
                if stress:
                    stress_differences.append((native[0]["task_depth"],
                                               mean(int(b["correct"]) - int(a["correct"]) for a, b in stress)))
            cue_pairs[checkpoint][cue] = {
                "paired_examples": total_cases,
                "examples_with_token_length_change": token_changes,
                "examples_with_any_prediction_change": changed_cases,
                "valid_steps": valid_total, "valid_prediction_matches_native": valid_same,
                "valid_mean_case_accuracy_difference": mean(value for _, value in valid_differences),
                "valid_case_bootstrap_95": _bootstrap_paired(valid_differences, seed),
                "same_token_length_cases": len(same_length_differences),
                "same_token_length_valid_mean_case_accuracy_difference": (
                    mean(value for _, value in same_length_differences)
                    if same_length_differences else None),
                "same_token_length_case_bootstrap_95": _bootstrap_paired(same_length_differences, seed + 5),
                "beyond_displayed_horizon_stress_steps": stress_total,
                "stress_prediction_matches_native": stress_same,
                "stress_mean_case_accuracy_difference": (mean(value for _, value in stress_differences)
                                                        if stress_differences else None),
                "stress_case_bootstrap_95": _bootstrap_paired(stress_differences, seed + 1),
                "valid_segments": {segment: {
                    "paired_examples": len(segment_differences[segment]),
                    "paired_steps": len(items),
                    "prediction_matches_native": sum(a["prediction"] == b["prediction"] for a, b in items),
                    "mean_case_accuracy_difference": mean(value for _, value in segment_differences[segment]),
                    "case_bootstrap_95": _bootstrap_paired(segment_differences[segment], seed + 4),
                    "same_token_length_cases": len(same_length_segment_differences[segment]),
                    "same_token_length_mean_case_accuracy_difference": (
                        mean(value for _, value in same_length_segment_differences[segment])
                        if same_length_segment_differences[segment] else None),
                    "same_token_length_case_bootstrap_95": _bootstrap_paired(
                        same_length_segment_differences[segment], seed + 6),
                } for segment, items in segment_steps.items()},
            }

    checkpoint_pairs = {}
    reference_ids = {key[1] for key in by_key if key[0] == reference and key[2] == "native"}
    for checkpoint in checkpoint_order[1:]:
        pairs = {}
        current_ids = {key[1] for key in by_key if key[0] == checkpoint and key[2] == "native"}
        if reference_ids != current_ids:
            raise ValueError("Checkpoint cohorts do not match")
        for group in ("validation", "deep"):
            differences, prefix_differences = [], []
            better, worse, tied = 0, 0, 0
            cases = 0
            for example_id in sorted(reference_ids):
                original = by_key[(reference, example_id, "native")]
                if original[0]["group"] != group:
                    continue
                changed = by_key[(checkpoint, example_id, "native")]
                if len(original) != len(changed):
                    raise ValueError("Checkpoint trajectories have different lengths")
                cases += 1
                depth = original[0]["task_depth"]
                difference = mean(int(b["correct"]) - int(a["correct"])
                                  for a, b in zip(original, changed, strict=True))
                differences.append((depth, difference))
                first_a = next((row["loop"] for row in original if not row["correct"]), depth + 1)
                first_b = next((row["loop"] for row in changed if not row["correct"]), depth + 1)
                prefix_delta = first_b - first_a
                prefix_differences.append((depth, float(prefix_delta)))
                better += prefix_delta > 0
                worse += prefix_delta < 0
                tied += prefix_delta == 0
            pairs[group] = {
                "paired_examples": cases, "longer_correct_prefix": better,
                "shorter_correct_prefix": worse, "same_correct_prefix": tied,
                "mean_case_step_accuracy_difference": mean(value for _, value in differences),
                "case_bootstrap_95": _bootstrap_paired(differences, seed + 2),
                "mean_correct_prefix_change": mean(value for _, value in prefix_differences),
                "prefix_case_bootstrap_95": _bootstrap_paired(prefix_differences, seed + 3),
            }
        checkpoint_pairs[checkpoint] = pairs
    return {"semantics": "Paired mapping-level summaries; trained-cue loops beyond the displayed Steps are stress tests, not normal task accuracy. Bootstrap resamples examples within depth, not loop rows.",
            "checkpoints": by_checkpoint, "cue_pairs": cue_pairs,
            "checkpoint_pairs_vs_first": checkpoint_pairs}
