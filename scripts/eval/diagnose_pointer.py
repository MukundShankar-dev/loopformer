"""Analyze an existing full-loop run against its reference data; no model loading."""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scripts.eval.failure_diagnostics import analyze_failures
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import load_examples, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", type=Path, required=True, help="Local source JSONL; must match saved dataset hash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    saved = json.loads((args.run / "summary.json").read_text())
    if saved.get("status") != "complete" or saved.get("stop_policy") or saved.get("task_variant"):
        parser.error("Requires a complete, fixed-depth original-pointer run")
    if sha256_file(args.data) != saved["data_sha256"]:
        parser.error("Dataset hash differs from source run")
    tasks = load_examples(args.data, saved["selected_examples"])
    failures, risk, summary = analyze_failures(args.run / "trajectories.csv", tasks, saved["loops"])
    args.output.mkdir(parents=True, exist_ok=False)
    # Empty failure cohorts remain valid and have an explicit zero in summary.
    if failures:
        write_csv(args.output / "first_failures.csv", failures)
    write_csv(args.output / "conditional_failures.csv", risk)
    summary.update(status="complete", source_run=str(args.run.resolve()), data_sha256=saved["data_sha256"],
                   source_summary_sha256=sha256_file(args.run / "summary.json"),
                   trajectories_sha256=sha256_file(args.run / "trajectories.csv"),
                   analyzer_sha256=sha256_file(Path(__file__).with_name("failure_diagnostics.py")))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = Table(title="First nominal failures")
    table.add_column("Category"); table.add_column("Count", justify="right")
    for category, count in summary["first_error_categories"].items():
        table.add_row(category, str(count))
    console = Console(); console.print(table)
    console.print(f"{summary['complete_trajectories']}/{len(tasks)} complete trajectories · saved {args.output}")


if __name__ == "__main__":
    main()
