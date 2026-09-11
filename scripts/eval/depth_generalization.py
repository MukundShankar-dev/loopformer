"""Paired full-loop sweeps on depths 1–8 and 9–16, with relative-depth reporting."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from rich.console import Console
from rich.table import Table

from scripts.eval.depth_comparison import compare_depth_runs
from scripts.eval.loop_metrics import write_csv


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=Path, required=True, help="Complete depth-4 step directory")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=Path, help="Complete depth-6 step directory")
    model.add_argument("--model-run", type=Path, help="Resolve best_checkpoint.json inside this new training run")
    parser.add_argument("--data", type=Path, action="append", help="Repeat for disjoint depth ranges; defaults to seed-17 validation and depth_test")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="New parent directory for all four runs and comparison")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; no model loading, subprocess inference, or writes")
    args = parser.parse_args()
    if args.model_run:
        best = json.loads((args.model_run / "best_checkpoint.json").read_text())
        args.model = (args.model_run / best["path"]).resolve()
        if args.model.parent != args.model_run.resolve():
            parser.error("Best checkpoint must be a step directory inside --model-run")
    data = args.data or [root / "data/pointer/seed-17/validation.jsonl", root / "data/pointer/seed-17/depth_test.jsonl"]
    output = args.output or root / "eval/pointer_depth_comparison" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    for path in data:
        if not path.is_file():
            parser.error(f"Missing dataset: {path}")
    models = {"reference": args.reference_model, "candidate": args.model}
    specs = {}
    for role, path in models.items():
        specs[role] = json.loads((path / "recurrent_config.json").read_text())
        if not args.dry_run and not (path / "adapter_model.pt").is_file():
            parser.error(f"Missing adapter weights: {path}")
    if specs["reference"]["train_max_depth"] >= specs["candidate"]["train_max_depth"]:
        parser.error("Candidate training maximum must exceed the reference maximum")
    console = Console()
    console.print("[bold cyan]Depth generalization · same datasets for both checkpoints[/bold cyan]")
    commands, runs = [], {role: [] for role in models}
    for role, path in models.items():
        console.print(f"{role}: {path} · trained through {specs[role]['train_max_depth']}")
        for index, dataset in enumerate(data):
            destination = output / f"{role}-{index}-{dataset.stem}"
            runs[role].append(destination)
            commands.append([sys.executable, "-m", "scripts.eval.loop_test", "--model", str(path),
                             "--data", str(dataset), "--device", args.device, "--output", str(destination)])
            console.print(f"  {dataset} → {destination}")
    if args.dry_run:
        console.print("[green]Dry run: plan only; no inference or writes.[/green]")
        return
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "commands": commands,
                "interpretation": "Matched absolute-depth data; equal offsets compare different depths, not paired instances or equal training compute"}
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    rows = compare_depth_runs(runs["reference"], runs["candidate"])
    write_csv(output / "comparison.csv", rows)
    table = Table(title="Complete trajectories · absolute and relative depth")
    for label in ("Model", "Depth", "Beyond training", "Examples", "All steps correct"):
        table.add_column(label)
    for row in rows:
        table.add_row(row["role"], str(row["task_depth"]), f"{row['steps_beyond_training']:+d}",
                      str(row["examples"]), f"{row['trajectory_accuracy']:.1%}")
    console.print(table)
    summary_path.write_text(json.dumps({**manifest, "status": "complete", "comparison": rows}, indent=2) + "\n")
    console.print(f"Saved comparison and individual full-loop artifacts to {output.resolve()}")


if __name__ == "__main__":
    main()
