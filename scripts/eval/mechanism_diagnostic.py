"""One frozen-checkpoint suite for pointer execution, shortcuts, and readout limits."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table

from scripts.dataset.pointer import PointerExample, SYMBOLS
from scripts.dataset.symbols import validate_prompt_tokens
from scripts.eval.loop_metrics import write_csv
from scripts.eval.mechanism_comparison import comparison_cases, comparison_checkpoint, summarize_comparison
from scripts.eval.mechanism_metrics import (attention_rows, attention_summary, baseline_case,
                                             baseline_summary, directed_summary, first_error,
                                             paired_summary)
from scripts.eval.mechanism_recording import (InternalRecorder, adapters_disabled,
                                               fit_ridge_readout, preloop_symbol, recurrent_answer_vectors,
                                               rule_symbol_positions)
from scripts.eval.mechanism_variants import Variant, directed_edge_variants, structural_variants
from scripts.eval.pointer_probes import inspect_example
from scripts.eval.pointer_task import sha256_file
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.data import EncodedExample, encode_tasks, read_tasks


def _fixed_cohorts(validation: list[PointerExample], deep: list[PointerExample],
                   train_max_depth: int, fit_per_depth: int, eval_per_depth: int,
                   deep_per_depth: int) -> list[tuple[str, PointerExample]]:
    """Select disjoint mapping-level cohorts before seeing any checkpoint output."""
    cohorts = []
    for depth in sorted({task.task_depth for task in validation}):
        group = [task for task in validation if task.task_depth == depth]
        fit_count = fit_per_depth if depth <= train_max_depth else 0
        if len(group) < fit_count + eval_per_depth:
            raise ValueError(f"Validation depth {depth} lacks fit plus held-out examples")
        cohorts.extend(("fit", task) for task in group[:fit_count])
        cohorts.extend(("validation", task) for task in group[fit_count:fit_count + eval_per_depth])
    for depth in sorted({task.task_depth for task in deep}):
        if depth <= train_max_depth:
            raise ValueError("Deep diagnostic split must start beyond trained depth")
        group = [task for task in deep if task.task_depth == depth]
        if len(group) < deep_per_depth:
            raise ValueError(f"Deep depth {depth} lacks diagnostic examples")
        cohorts.extend(("deep", task) for task in group[:deep_per_depth])
    fingerprints = [task.mapping_sha256 for _, task in cohorts]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Selected cohorts share a rule table")
    return cohorts


def _encode_variant(task: PointerExample, variant: Variant, tokenizer, token_map: dict[str, int],
                    max_tokens: int) -> EncodedExample:
    validate_prompt_tokens(tokenizer, variant.prompt, token_map)
    ids = tokenizer.encode(variant.prompt, add_special_tokens=False)
    if len(ids) > max_tokens:
        raise ValueError(f"{task.example_id}/{variant.name}: prompt exceeds model context")
    return EncodedExample(task, ids, [SYMBOLS.index(symbol) for symbol in variant.targets])


def _write_rows(path: Path, rows: list[dict]) -> None:
    if rows:
        write_csv(path, rows)
    else:
        path.write_text("example_id\n")


def _save_state_vectors(path: Path, records: list[dict], max_depth: int) -> None:
    hidden = records[0]["recurrent"].shape[-1]
    count = len(records)
    r = np.full((count, max_depth + 1, hidden), np.nan, dtype=np.float32)
    c = np.full((count, max_depth, hidden), np.nan, dtype=np.float32)
    targets = np.full((count, max_depth), 255, dtype=np.uint8)
    lengths = np.empty(count, dtype=np.int16)
    for index, record in enumerate(records):
        depth = record["task"].task_depth
        lengths[index] = depth
        r[index, :depth + 1] = record["recurrent"].numpy()
        c[index, :depth] = record["coda"].numpy()
        targets[index, :depth] = [SYMBOLS.index(symbol) for symbol in record["task"].intermediate_states]
    np.savez_compressed(path, example_ids=np.asarray([record["task"].example_id for record in records]),
                        groups=np.asarray([record["group"] for record in records]), lengths=lengths,
                        recurrent_answer=r, coda_answer=c, reference_targets=targets)


def _probe_rows(records: list[dict], train_max_depth: int) -> tuple[list[dict], dict]:
    fit = [record for record in records if record["group"] == "fit"]
    if not fit:
        raise ValueError("No representation-probe fitting examples")
    x_r, x_c, y = [], [], []
    for record in fit:
        depth = record["task"].task_depth
        count = min(depth, train_max_depth)
        x_r.append(record["recurrent"][1:count + 1])
        x_c.append(record["coda"][:count])
        y.extend(SYMBOLS.index(symbol) for symbol in record["task"].intermediate_states[:count])
    targets = torch.tensor(y, dtype=torch.long)
    r_probe = fit_ridge_readout(torch.cat(x_r), targets, alpha=1.0)
    c_probe = fit_ridge_readout(torch.cat(x_c), targets, alpha=1.0)
    rows = []
    for record in records:
        task = record["task"]
        r_predictions = r_probe.predict(record["recurrent"][1:]).argmax(-1).tolist()
        c_predictions = c_probe.predict(record["coda"]).argmax(-1).tolist()
        for loop, (r_index, c_index, observed) in enumerate(
                zip(r_predictions, c_predictions, record["rows"], strict=True), 1):
            target = task.intermediate_states[loop - 1]
            rows.append({"example_id": task.example_id, "group": record["group"],
                         "task_depth": task.task_depth, "loop": loop, "target": target,
                         "model_prediction": observed["prediction"], "model_correct": observed["correct"],
                         "r_probe_prediction": SYMBOLS[r_index], "r_probe_correct": SYMBOLS[r_index] == target,
                         "c_probe_prediction": SYMBOLS[c_index], "c_probe_correct": SYMBOLS[c_index] == target})
    buckets = defaultdict(list)
    for row in rows:
        interval = ("fit_1_to_6" if row["group"] == "fit" else
                    "early_1_to_6" if row["loop"] <= train_max_depth else
                    "near_7_to_8" if row["loop"] <= train_max_depth + 2 else "farther")
        buckets[(row["group"], interval)].append(row)
    summary = {f"{group}/{bucket}": {"steps": len(items),
               "model_correct": sum(item["model_correct"] for item in items),
               "r_probe_correct": sum(item["r_probe_correct"] for item in items),
               "c_probe_correct": sum(item["c_probe_correct"] for item in items)}
               for (group, bucket), items in sorted(buckets.items())}
    heldout = [row for row in rows if row["group"] == "validation" and row["loop"] <= train_max_depth]
    model_accuracy = sum(row["model_correct"] for row in heldout) / len(heldout) if heldout else None
    threshold = max(0.8, model_accuracy - 0.05) if model_accuracy is not None else None
    r_accuracy = sum(row["r_probe_correct"] for row in heldout) / len(heldout) if heldout else None
    c_accuracy = sum(row["c_probe_correct"] for row in heldout) / len(heldout) if heldout else None
    summary["interpretation_gate"] = {
        "heldout_early_steps": len(heldout),
        "model_early_accuracy": model_accuracy,
        "r_probe_early_accuracy": r_accuracy,
        "c_probe_early_accuracy": c_accuracy,
        "minimum_for_late_readout_interpretation": threshold,
        "r_probe_adequate": r_accuracy >= threshold if threshold is not None else False,
        "c_probe_adequate": c_accuracy >= threshold if threshold is not None else False,
        "semantics": "Linear decodability only; below-threshold early probes cannot locate late representation loss.",
    }
    return rows, summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--comparison-model", type=Path, action="append", default=[],
                        help="Additional compatible checkpoints on the identical selected examples")
    parser.add_argument("--validation", type=Path, default=root / "data/pointer/seed-17/validation.jsonl")
    parser.add_argument("--deep", type=Path, default=root / "data/pointer/seed-17/depth_test.jsonl")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--fit-per-depth", type=int, default=24)
    parser.add_argument("--eval-per-depth", type=int, default=8)
    parser.add_argument("--deep-per-depth", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.fit_per_depth, args.eval_per_depth, args.deep_per_depth, args.threads) < 1 or args.output.exists():
        parser.error("Positive cohort sizes/threads and a new output directory required")
    checkpoint_paths = [args.model, *args.comparison_model]
    if len({str(path.resolve()) for path in checkpoint_paths}) != len(checkpoint_paths):
        parser.error("Comparison checkpoints must be distinct")
    if len({path.name for path in checkpoint_paths}) != len(checkpoint_paths):
        parser.error("Comparison checkpoint directory names must be distinct")
    for checkpoint in checkpoint_paths:
        for name in ("recurrent_config.json", "adapter_model.pt", "tokenizer.json"):
            if not (checkpoint / name).is_file():
                parser.error(f"Missing checkpoint file: {checkpoint / name}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    console = Console()
    with console.status("Loading frozen checkpoint and validated datasets…"):
        model, tokenizer, spec = load_recurrent_checkpoint(args.model, device=args.device, download=args.download)
        validation = read_tasks(args.validation, "validation")
        deep = read_tasks(args.deep, "depth_test")
    train_max_depth = int(spec["train_max_depth"])
    cohorts = _fixed_cohorts(validation, deep, train_max_depth, args.fit_per_depth,
                             args.eval_per_depth, args.deep_per_depth)
    trainable_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    if not any(trainable_before.values()) or any(
            enabled and (not name.startswith("recurrent.") or ".lora_" not in name)
            for name, enabled in trainable_before.items()):
        raise ValueError("Checkpoint exposes trainable parameters outside recurrent LoRA")
    token_map = dict(zip(spec["symbols"], spec["token_ids"], strict=True))
    counts = {group: sum(selected == group for selected, _ in cohorts)
              for group in ("fit", "validation", "deep")}
    console.print("[bold cyan]Pointer mechanism diagnostic · frozen checkpoint[/bold cyan]")
    console.print(f"{args.model} · {args.device} · float32 · batch 1 · "
                  f"{counts['fit']} probe-fit / {counts['validation']} validation / {counts['deep']} deep examples")
    console.print("Full original trajectories, paired controls, directed branches, and compact internal readouts")
    args.output.mkdir(parents=True, exist_ok=False)
    summary_path = args.output / "summary.json"
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "scripts.eval.mechanism_diagnostic", *sys.argv[1:]],
        "checkpoint": str(args.model.resolve()), "device": args.device, "dtype": "float32",
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else args.device,
        "platform": platform.platform(), "python": sys.version,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "numpy")},
        "seed": args.seed, "model_spec": spec, "train_max_depth": train_max_depth,
        "trainable_adapter_parameters": sum(parameter.numel() for parameter in model.parameters()
                                            if parameter.requires_grad),
        "fit_per_depth": args.fit_per_depth, "eval_per_depth": args.eval_per_depth,
        "deep_per_depth": args.deep_per_depth,
        "cohort_ids": {group: [task.example_id for selected, task in cohorts if selected == group]
                       for group in ("fit", "validation", "deep")},
        "data_sha256": {"validation": sha256_file(args.validation), "deep": sha256_file(args.deep)},
        "adapter_sha256": sha256_file(args.model / "adapter_model.pt"),
        "checkpoint_metadata_sha256": sha256_file(args.model / "recurrent_config.json"),
        "tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
        "comparison_checkpoints": {path.name: {
            "path": str(path.resolve()),
            "adapter_sha256": sha256_file(path / "adapter_model.pt"),
            "metadata_sha256": sha256_file(path / "recurrent_config.json"),
            "tokenizer_sha256": sha256_file(path / "tokenizer.json")}
            for path in checkpoint_paths},
        "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in
                          [Path(__file__), Path(__file__).with_name("mechanism_variants.py"),
                           Path(__file__).with_name("mechanism_recording.py"),
                           Path(__file__).with_name("mechanism_metrics.py"),
                           Path(__file__).with_name("mechanism_comparison.py"),
                           Path(__file__).with_name("pointer_probes.py"),
                           root / "scripts/recurrent_qwen/model.py",
                           root / "scripts/recurrent_qwen/checkpoint.py",
                           root / "scripts/recurrent_qwen/lora_utils.py",
                           root / "scripts/training/data.py",
                           root / "scripts/dataset/pointer.py",
                           root / "scripts/dataset/symbols.py"]},
        "semantics": "One fixed development cohort and predeclared controls; no checkpoint update, "
                     "no external state transfer during any inference. Representation probes are offline "
                     "linear decodability tests, not evidence of a causal model algorithm.",
    }
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    records, original_rows, cases, attention, inputs = [], [], [], [], []
    for group, task in track(cohorts, description="Original trajectories and hidden states", console=console):
        item = encode_tasks([task], tokenizer, token_map, model.config.max_position_embeddings)[0]
        position = len(item.input_ids) - 1
        capture_attention = group != "fit"
        source_positions, destination_positions = (rule_symbol_positions(tokenizer, task)
                                                    if capture_attention else (None, None))
        vectors, initial_states = [], []
        with InternalRecorder(model, position, source_positions, destination_positions) as recorder:
            rows = inspect_example(model, item, spec["token_ids"], tokenizer.pad_token_id,
                                   state_sink=lambda output: (vectors.append(recurrent_answer_vectors(output, position)),
                                                              initial_states.append(output.initial_hidden_state.detach().clone())))
        if len(vectors) != 1 or len(initial_states) != 1:
            raise ValueError("Missing recurrent-state capture")
        with torch.no_grad():
            before_recurrence = preloop_symbol(model, initial_states[0], spec["token_ids"])
        internal = recorder.finish(task.task_depth)
        record = {"group": group, "task": task, "item": item, "rows": rows,
                  "recurrent": vectors[0], "coda": internal["coda"]}
        records.append(record)
        original_rows.extend({"example_id": task.example_id, "group": group, "task_depth": task.task_depth, **row}
                             for row in rows)
        case = baseline_case(task, group, rows)
        case.update(preloop_prediction=before_recurrence,
                    preloop_matches_start=before_recurrence == task.initial_state,
                    preloop_matches_first_target=before_recurrence == task.intermediate_states[0])
        cases.append(case)
        inputs.append({"example_id": task.example_id, "group": group, "variant": "original",
                       "prompt": task.prompt, "input_ids": item.input_ids,
                       "targets": task.intermediate_states})
        if capture_attention:
            attention.extend(attention_rows(task, group, rows, internal["source_attention"],
                                            internal["destination_attention"]))
    _save_state_vectors(args.output / "answer_states.npz", records,
                        max(task.task_depth for _, task in cohorts))
    probe_rows, probe_summary = _probe_rows(records, train_max_depth)
    paired, directed = [], []
    evaluated = [record for record in records if record["group"] != "fit"]
    for record in track(evaluated, description="Paired controls and directed edits", console=console):
        task, group, original = record["task"], record["group"], record["rows"]
        with adapters_disabled(model):
            off = inspect_example(model, record["item"], spec["token_ids"], tokenizer.pad_token_id)
        for t, (base, changed) in enumerate(zip(original, off, strict=True), 1):
            paired.append({"example_id": task.example_id, "group": group, "variant": "adapter_off",
                           "task_depth": task.task_depth, "loop": t,
                           "baseline_input_tokens": len(record["item"].input_ids),
                           "variant_input_tokens": len(record["item"].input_ids),
                           "token_length_changed": False,
                           "baseline_prediction": base["prediction"], "variant_prediction": changed["prediction"],
                           "baseline_correct": base["correct"], "variant_correct": changed["correct"],
                           "equivariant_prediction": base["prediction"] == changed["prediction"],
                           "variant_target_margin": changed["target_margin"]})
        for variant in structural_variants(task):
            item = _encode_variant(task, variant, tokenizer, token_map, model.config.max_position_embeddings)
            changed = inspect_example(model, item, spec["token_ids"], tokenizer.pad_token_id)
            rename = dict(zip(SYMBOLS, variant.symbol_map, strict=True)) if variant.symbol_map else None
            for t, (base, observed) in enumerate(zip(original, changed, strict=True), 1):
                expected_prediction = rename[base["prediction"]] if rename else base["prediction"]
                paired.append({"example_id": task.example_id, "group": group, "variant": variant.name,
                               "task_depth": task.task_depth, "loop": t,
                               "baseline_input_tokens": len(record["item"].input_ids),
                               "variant_input_tokens": len(item.input_ids),
                               "token_length_changed": len(item.input_ids) != len(record["item"].input_ids),
                               "baseline_prediction": base["prediction"], "variant_prediction": observed["prediction"],
                               "baseline_correct": base["correct"], "variant_correct": observed["correct"],
                               "equivariant_prediction": expected_prediction == observed["prediction"],
                               "variant_target_margin": observed["target_margin"]})
            inputs.append({"example_id": task.example_id, "group": group, "variant": variant.name,
                           "prompt": variant.prompt, "input_ids": item.input_ids,
                           "targets": variant.targets, "changed_source": variant.changed_source,
                           "old_destination": variant.old_destination,
                           "new_destination": variant.new_destination,
                           "symbol_map": variant.symbol_map})
        if group != "deep":
            continue
        error = first_error(original)
        phases = []
        if error is None or error > train_max_depth:
            phases.append(("early", train_max_depth))
        if error is not None and error > train_max_depth:
            phases.append(("critical", error))
        for phase, probe_loop in phases:
            variants = directed_edge_variants(task, probe_loop,
                                              exclude_prediction=original[probe_loop - 1]["prediction"],
                                              branch=phase, replacements=2)
            for variant in variants:
                item = _encode_variant(task, variant, tokenizer, token_map, model.config.max_position_embeddings)
                changed = inspect_example(model, item, spec["token_ids"], tokenizer.pad_token_id)
                index = int(variant.name.rsplit("_", 1)[-1])
                edited_prefix_correct = all(row["correct"] for row in changed[:probe_loop - 1])
                for t, observed in enumerate(changed, 1):
                    base = original[t - 1]
                    directed.append({"example_id": task.example_id, "task_depth": task.task_depth,
                                     "phase": phase, "condition": "relevant" if "_relevant_" in variant.name else "irrelevant",
                                     "replacement_index": index, "loop": t, "probe_loop": probe_loop,
                                     "baseline_input_tokens": len(record["item"].input_ids),
                                     "variant_input_tokens": len(item.input_ids),
                                     "source": variant.changed_source, "old_destination": variant.old_destination,
                                     "new_destination": variant.new_destination,
                                     "baseline_prediction": base["prediction"], "baseline_target": base["target"],
                                     "edited_prediction": observed["prediction"], "edited_target": observed["target"],
                                     "edited_correct": observed["correct"],
                                     "prediction_changed": base["prediction"] != observed["prediction"],
                                     "edited_prefix_correct": edited_prefix_correct,
                                     "earlier_prediction_changes": sum(a["prediction"] != b["prediction"] for a, b in
                                                                       zip(original[:probe_loop - 1], changed[:probe_loop - 1], strict=True)),
                                     "edited_target_rank": observed["target_rank"],
                                     "edited_target_margin": observed["target_margin"]})
                inputs.append({"example_id": task.example_id, "group": group, "variant": variant.name,
                               "prompt": variant.prompt, "input_ids": item.input_ids,
                               "targets": variant.targets, "probe_loop": probe_loop,
                               "changed_source": variant.changed_source,
                               "old_destination": variant.old_destination,
                               "new_destination": variant.new_destination})
    _write_rows(args.output / "original.csv", original_rows)
    _write_rows(args.output / "cases.csv", cases)
    _write_rows(args.output / "attention.csv", attention)
    _write_rows(args.output / "probe_predictions.csv", probe_rows)
    _write_rows(args.output / "paired.csv", paired)
    _write_rows(args.output / "directed.csv", directed)
    baseline, risk = baseline_summary(cases)
    _write_rows(args.output / "first_error_risk.csv", risk)
    input_path = args.output / "inputs.jsonl"
    input_path.write_text("".join(json.dumps(item) + "\n" for item in inputs))
    selected = [(record["group"], record["task"]) for record in evaluated]
    checkpoint_order = [path.name for path in checkpoint_paths]
    original_by_id = {record["task"].example_id: record["rows"] for record in evaluated}
    comparison_rows, prefix_rows, comparison_inputs = comparison_checkpoint(
        model, tokenizer, spec, selected, args.model.name,
        original_rows=original_by_id, console=console)
    if trainable_before != {name: parameter.requires_grad for name, parameter in model.named_parameters()}:
        raise ValueError("Adapter ablation did not restore the original gradient scope")
    del model, tokenizer, recorder, initial_states
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    for checkpoint in args.comparison_model:
        with console.status(f"Loading comparison checkpoint {checkpoint.name}…"):
            compared, compared_tokenizer, compared_spec = load_recurrent_checkpoint(
                checkpoint, device=args.device, download=args.download)
        if compared_spec != spec or sha256_file(checkpoint / "tokenizer.json") != metadata["tokenizer_sha256"]:
            raise ValueError(f"Comparison checkpoint has different model/tokenizer specification: {checkpoint}")
        extra_rows, extra_prefix, extra_inputs = comparison_checkpoint(
            compared, compared_tokenizer, compared_spec, selected, checkpoint.name,
            original_rows=None, console=console)
        comparison_rows.extend(extra_rows)
        prefix_rows.extend(extra_prefix)
        comparison_inputs.extend(extra_inputs)
        del compared, compared_tokenizer
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    _write_rows(args.output / "comparison.csv", comparison_rows)
    _write_rows(args.output / "comparison_cases.csv", comparison_cases(comparison_rows))
    _write_rows(args.output / "comparison_prefix.csv", prefix_rows)
    (args.output / "comparison_inputs.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in comparison_inputs))
    for checkpoint in checkpoint_paths:
        expected = metadata["comparison_checkpoints"][checkpoint.name]
        for filename, key in (("adapter_model.pt", "adapter_sha256"),
                              ("recurrent_config.json", "metadata_sha256"),
                              ("tokenizer.json", "tokenizer_sha256")):
            if sha256_file(checkpoint / filename) != expected[key]:
                raise ValueError(f"Checkpoint file changed during read-only diagnostic: {checkpoint / filename}")
    metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    baseline=baseline, paired_controls=paired_summary(paired),
                    directed_edges=directed_summary(directed),
                    representation_probe=probe_summary, attention=attention_summary(attention),
                    matched_comparison=summarize_comparison(comparison_rows, checkpoint_order,
                                                            train_max_depth, args.seed),
                    artifacts_sha256={path.name: sha256_file(path) for path in args.output.iterdir()
                                      if path.is_file() and path.name != "summary.json"})
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    table = Table(title="Frozen mechanism diagnostic · selected development cohorts")
    table.add_column("Cohort / variant")
    table.add_column("Correct / steps", justify="right")
    for group in ("validation", "deep"):
        by_depth = metadata["baseline"]["by_depth"].get(group, {})
        complete = sum(value["complete"] for value in by_depth.values())
        total = sum(value["examples"] for value in by_depth.values())
        console.print(f"{group} original complete trajectories: {complete}/{total}")
    for group in ("validation", "deep"):
        for variant in ("adapter_off", "rule_order", "symbol_rename", "steps_plus_one", "irrelevant_edge"):
            value = metadata["paired_controls"].get(group, {}).get(variant)
            if value:
                table.add_row(f"{group} / {variant}", f"{value['variant_correct_steps']}/{value['paired_steps']}")
    console.print(table)
    for phase, value in metadata["directed_edges"].items():
        console.print(f"{phase}: relevant follows {value['relevant_follows_new_target']}/"
                      f"{value['matched_correct_prefix_pairs']} matched edits; "
                      f"two-target cases {value['cases_following_both_distinct_replacements']}/"
                      f"{value['cases_with_both_replacements_retaining_prefix']}")
    gate = metadata["representation_probe"]["interpretation_gate"]
    console.print(f"Held-out early readout: R probe {gate['r_probe_early_accuracy']:.1%} "
                  f"({'adequate' if gate['r_probe_adequate'] else 'inconclusive'}), "
                  f"C probe {gate['c_probe_early_accuracy']:.1%} "
                  f"({'adequate' if gate['c_probe_adequate'] else 'inconclusive'}); "
                  f"interpretation threshold {gate['minimum_for_late_readout_interpretation']:.1%}")
    for name in checkpoint_order:
        value = metadata["matched_comparison"]["checkpoints"][name]["deep"]
        console.print(f"{name} native deep complete trajectories: {value['complete']}/{value['examples']}")
        late = metadata["matched_comparison"]["cue_pairs"][name]["fixed_long_cue"]["valid_segments"]["late_after_train_max"]
        same_length_change = late["same_token_length_mean_case_accuracy_difference"]
        rendered_change = f"{same_length_change:+.1%}" if same_length_change is not None else "n/a"
        console.print(f"{name} Steps-long late paired change, same token count: "
                      f"{rendered_change} across {late['same_token_length_cases']} mappings")
    for name, values in metadata["matched_comparison"]["checkpoint_pairs_vs_first"].items():
        deep = values["deep"]
        console.print(f"{name} versus {checkpoint_order[0]}: deep prefix longer {deep['longer_correct_prefix']}, "
                      f"shorter {deep['shorter_correct_prefix']}, same {deep['same_correct_prefix']}")
    console.print(f"[green]Saved[/green] {args.output}")


if __name__ == "__main__":
    main()
