"""Probe a frozen recurrent checkpoint with relevant and irrelevant rule edits."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table

from scripts.dataset.pointer import SYMBOLS
from scripts.dataset.symbols import validate_prompt_tokens
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_probes import inspect_example
from scripts.eval.pointer_task import load_examples, sha256_file
from scripts.eval.rule_edits import paired_rule_edits
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.training.data import EncodedExample, encode_tasks


def _summarize(pairs: list[dict]) -> dict:
    result = {}
    for phase in ("early", "late"):
        result[phase] = {}
        for condition in ("relevant", "irrelevant"):
            rows = [row for row in pairs if row["phase"] == phase and row["condition"] == condition]
            retained = [row for row in rows if row["edited_prefix_correct"]]
            result[phase][condition] = {
                "attempted": len(rows),
                "prefix_retained": len(retained),
                "follows_edited_target": sum(row["edited_correct"] for row in retained),
                "prediction_changed": sum(row["prediction_changed"] for row in retained),
                "baseline_correct": sum(row["baseline_correct"] for row in retained),
            }
        by_id = {row["example_id"]: row for row in pairs
                 if row["phase"] == phase and row["condition"] == "irrelevant"}
        matched = [(row, by_id[row["example_id"]]) for row in pairs
                   if row["phase"] == phase and row["condition"] == "relevant"
                   and row["example_id"] in by_id and row["edited_prefix_correct"]
                   and by_id[row["example_id"]]["edited_prefix_correct"]]
        result[phase]["matched"] = {
            "both_prefixes_retained": len(matched),
            "relevant_follows_new_edge": sum(relevant["edited_correct"] for relevant, _ in matched),
            "irrelevant_keeps_original_target": sum(irrelevant["edited_correct"] for _, irrelevant in matched),
            "relevant_prediction_changed": sum(relevant["prediction_changed"] for relevant, _ in matched),
            "irrelevant_prediction_changed": sum(irrelevant["prediction_changed"] for _, irrelevant in matched),
        }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=root / "data/pointer/seed-17/depth_test.jsonl")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--limit", type=int, default=32, help="Fixed dataset prefix, selected before inference")
    parser.add_argument("--early-loop", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.limit, args.early_loop, args.threads) < 1 or args.output.exists():
        parser.error("Positive counts and a new output directory required")
    for name in ("recurrent_config.json", "adapter_model.pt", "tokenizer.json"):
        if not (args.model / name).is_file():
            parser.error(f"Missing checkpoint file: {args.model / name}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    tasks = load_examples(args.data, args.limit)
    if any(task.task_depth <= args.early_loop for task in tasks):
        parser.error("Every selected task must be deeper than --early-loop")
    console = Console()
    with console.status("Loading frozen checkpoint…"):
        model, tokenizer, spec = load_recurrent_checkpoint(args.model, device=args.device, download=args.download)
    token_map = dict(zip(spec["symbols"], spec["token_ids"], strict=True))
    args.output.mkdir(parents=True, exist_ok=False)
    summary_path = args.output / "summary.json"
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "scripts.eval.rule_edit_probe", *sys.argv[1:]],
        "model": str(args.model.resolve()), "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "adapter_sha256": sha256_file(args.model / "adapter_model.pt"),
        "checkpoint_metadata_sha256": sha256_file(args.model / "recurrent_config.json"),
        "tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
        "source_sha256": {str(path.relative_to(root)): sha256_file(path)
                          for path in (Path(__file__), Path(__file__).with_name("rule_edits.py"),
                                       Path(__file__).with_name("pointer_probes.py"),
                                       root / "scripts/recurrent_qwen/model.py")},
        "device": args.device, "dtype": "float32", "seed": args.seed,
        "fixed_prefix_examples": len(tasks), "early_loop": args.early_loop,
        "semantics": "Complete prompt is encoded once per variant and recurrent state is never externally reset. "
                     "Late cases have an observed correct original prefix and first error after early_loop; "
                     "only edited-prefix-retained cases support transition comparison. "
                     "Edits perturb the input from loop one, so outcomes are diagnostic rather than causal isolation.",
    }
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    states, pairs, inputs = [], [], []
    eligible = 0
    for task in track(tasks, description="Rule-edit diagnostic", console=console):
        original_item = encode_tasks([task], tokenizer, token_map, model.config.max_position_embeddings)[0]
        original = inspect_example(model, original_item, spec["token_ids"], tokenizer.pad_token_id)
        for row in original:
            states.append({"example_id": task.example_id, "phase": "original", "condition": "original", **row})
        first_error = next((row["loop"] for row in original if not row["correct"]), None)
        if first_error is None or first_error <= args.early_loop:
            continue
        eligible += 1
        for phase, loop in (("early", args.early_loop), ("late", first_error)):
            baseline = original[loop - 1]
            if phase == "early" and not baseline["correct"]:
                raise AssertionError("Early control is not a correct original transition")
            for edit in paired_rule_edits(task, loop):
                validate_prompt_tokens(tokenizer, edit.prompt, token_map)
                ids = tokenizer.encode(edit.prompt, add_special_tokens=False)
                if len(ids) != len(original_item.input_ids):
                    raise ValueError(f"Edited prompt changed token count: {task.example_id}")
                if len(ids) > model.config.max_position_embeddings:
                    raise ValueError(f"Edited prompt exceeds context: {task.example_id}")
                item = EncodedExample(task, ids, [SYMBOLS.index(symbol) for symbol in edit.targets])
                edited = inspect_example(model, item, spec["token_ids"], tokenizer.pad_token_id)
                prefix_correct = all(row["correct"] for row in edited[:-1])
                for row in edited:
                    states.append({"example_id": task.example_id, "phase": phase,
                                   "condition": edit.condition, **row})
                current = edited[-1]
                pairs.append({
                    "example_id": task.example_id, "task_depth": task.task_depth,
                    "phase": phase, "loop": loop, "first_error_loop": first_error,
                    "condition": edit.condition, "source": edit.source,
                    "old_destination": edit.old_destination, "new_destination": edit.new_destination,
                    "baseline_prediction": baseline["prediction"], "baseline_target": baseline["target"],
                    "baseline_correct": baseline["correct"],
                    "edited_prediction": current["prediction"], "edited_target": current["target"],
                    "edited_correct": current["correct"], "prediction_changed": current["prediction"] != baseline["prediction"],
                    "edited_prefix_correct": prefix_correct,
                    "earlier_prediction_changes": sum(a["prediction"] != b["prediction"]
                                                      for a, b in zip(original[:loop - 1], edited[:-1], strict=True)),
                    "edited_target_rank": current["target_rank"],
                    "edited_target_margin": current["target_margin"],
                    "edited_predicted_margin": current["predicted_margin"],
                })
                inputs.append({"example_id": task.example_id, "phase": phase, "loop": loop,
                               "condition": edit.condition, "source": edit.source,
                               "old_destination": edit.old_destination, "new_destination": edit.new_destination,
                               "prompt": edit.prompt, "input_ids": ids, "reference_targets_through_loop": edit.targets})
    write_csv(args.output / "states.csv", states)
    if pairs:
        write_csv(args.output / "pairs.csv", pairs)
    else:
        (args.output / "pairs.csv").write_text("example_id,phase,condition,loop\n")
    inputs_path = args.output / "inputs.jsonl"
    inputs_path.write_text("".join(json.dumps(row) + "\n" for row in inputs))
    metadata.update(status="complete", eligible_late_first_errors=eligible,
                    counts=_summarize(pairs), inputs_sha256=sha256_file(inputs_path))
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    table = Table(title="Edited-target following after a correct edited prefix")
    table.add_column("Phase / edit")
    table.add_column("Retained / attempted", justify="right")
    table.add_column("Followed / retained", justify="right")
    for phase, conditions in metadata["counts"].items():
        for condition in ("relevant", "irrelevant"):
            count = conditions[condition]
            table.add_row(f"{phase} / {condition}", f"{count['prefix_retained']}/{count['attempted']}",
                          f"{count['follows_edited_target']}/{count['prefix_retained']}")
    console.print(table)
    for phase, conditions in metadata["counts"].items():
        matched = conditions["matched"]
        console.print(f"{phase} matched prefixes: {matched['both_prefixes_retained']} · "
                      f"relevant follows new edge {matched['relevant_follows_new_edge']} · "
                      f"irrelevant keeps original target {matched['irrelevant_keeps_original_target']}")
    console.print(f"[green]Saved[/green] {args.output} · {eligible}/{len(tasks)} eligible late first errors")


if __name__ == "__main__":
    main()
