"""Offline adaptive-compute analysis of a complete loop_test run."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from rich.console import Console
from rich.table import Table

from .adaptive_metrics import analyze, load_traces
from .loop_metrics import write_csv
from .pointer_task import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Complete loop_test or overscaling_test output directory")
    parser.add_argument("--output", type=Path, required=True, help="New derived-analysis directory")
    parser.add_argument("--stability-k", type=int, default=2)
    parser.add_argument("--margin-threshold", type=float, help="Top-two predicted-answer margin threshold")
    parser.add_argument("--entropy-threshold", type=float, help="Allowed-answer entropy threshold")
    parser.add_argument("--oracle-window", type=int, default=1, help="Future-correct window for separate optimistic oracle")
    parser.add_argument("--allow-observational", action="store_true",
                        help="Analyze continuing-pointer traces without interpreting post-depth changes as damage")
    args = parser.parse_args()
    if args.stability_k < 1 or args.oracle_window < 1:
        parser.error("--stability-k must be positive")
    source = args.run / "summary.json"
    trajectories = args.run / "trajectories.csv"
    if not source.is_file() or not trajectories.is_file():
        parser.error("Run needs summary.json and trajectories.csv")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    metadata = json.loads(source.read_text())
    if metadata.get("status") != "complete":
        parser.error("Only complete runs can be analyzed")
    variant = metadata.get("task_variant", "continuing-pointer")
    if variant != "absorbing-terminal-v1" and not args.allow_observational:
        parser.error("Nonterminal traces require --allow-observational; damage labels are not valid")
    traces = load_traces(trajectories)
    if len(traces) != metadata.get("selected_examples") or traces[0].budget != metadata.get("loops"):
        parser.error("Trace coverage differs from source summary")
    rows, summary = analyze(traces, stability_k=args.stability_k,
                            margin_threshold=args.margin_threshold,
                            entropy_threshold=args.entropy_threshold,
                            oracle_window=args.oracle_window)
    if variant != "absorbing-terminal-v1":
        for row in rows:
            row["category"] = "observational_" + row["category"]
            row["first_final_target_departure_loop"] = row.pop("first_damage_loop")
        summary["categories"] = {"observational_" + key: count
                                  for key, count in summary["categories"].items()}
        for depth in summary["by_depth"].values():
            depth["categories"] = {"observational_" + key: count
                                   for key, count in depth["categories"].items()}
    args.output.mkdir(parents=True)
    write_csv(args.output / "examples.csv", rows)
    root = Path(__file__).resolve().parents[2]
    analysis_sources = ("scripts/eval/adaptive_analyze.py", "scripts/eval/adaptive_metrics.py", "scripts/eval/loop_metrics.py")
    report = {"analysis_source_sha256": {name: sha256_file(root / name) for name in analysis_sources}, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "command": [sys.executable, "-m", "scripts.eval.adaptive_analyze", *sys.argv[1:]],
              "source_run": str(args.run.resolve()), "source_summary_sha256": sha256_file(source),
              "source_trajectories_sha256": sha256_file(trajectories),
              "checkpoint": metadata.get("checkpoint"),
              "local_checkpoint_sha256": metadata.get("local_checkpoint_sha256"),
              "source_git_head": metadata.get("git_head"),
              "data_sha256": metadata.get("data_sha256"),
              "source_sha256": metadata.get("source_sha256"), "task_variant": variant,
              "interpretation": "terminal" if variant == "absorbing-terminal-v1" else "observational_only",
              "oracle_rule": "earliest t>=depth correct through observed horizon; fallback to horizon",
              "first_correct_rule": "earliest t>=depth correct; optimistic labeled reference",
              "stability_k": args.stability_k, "margin_threshold": args.margin_threshold,
              "entropy_threshold": args.entropy_threshold, "oracle_window": args.oracle_window,
              "compute_unit": "recurrent loops, not latency",
              **summary}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    console = Console()
    table = Table(title="Offline stopping analysis")
    for name in ("Policy", "Accuracy", "Mean loops", "Loop savings"):
        table.add_column(name)
    for name, metrics in summary["policies"].items():
        table.add_row(name, f"{metrics['accuracy']:.2%}", f"{metrics['mean_loops']:.2f}",
                      f"{metrics['loop_savings_fraction']:.2%}")
    console.print(table)
    console.print(f"[bold]Interpretation: {report['interpretation']}[/bold] · {args.output.resolve()}")


if __name__ == "__main__":
    main()
