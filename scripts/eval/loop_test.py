"""Evaluate every recurrent loop of a saved pointer checkpoint; no training."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Saved recurrent step directory (with adapter weights and tokenizer)")
    parser.add_argument("--data", type=Path, default=root / "data/pointer/seed-17/test.jsonl")
    parser.add_argument("--loops", type=int, help="Loops for every example; default maximum selected task depth; must cover all targets")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--download", action="store_true", help="Allow downloading the recorded frozen base; tokenizer and adapters must be local")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--limit", type=int, help="Evaluate the first N examples")
    subset.add_argument("--test", nargs="?", type=int, const=3, choices=(2, 3), help="Show 2 or 3 inputs and their per-loop scoring")
    parser.add_argument("--output", type=Path, help="New output directory; default eval/pointer_loops/<timestamp>-<run>-<step>")
    args = parser.parse_args()
    for name in ("batch_size", "threads", "loops", "limit"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    for path in (args.data, args.model / "recurrent_config.json", args.model / "adapter_model.pt"):
        if not path.is_file():
            parser.error(f"Required file missing: {path}. Use the complete checkpoint on the training device.")
    started = datetime.now(timezone.utc)
    output = args.output or root / "eval/pointer_loops" / f"{started:%Y%m%dT%H%M%S.%fZ}-{args.model.parent.name}-{args.model.name}"
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output")

    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text

    from scripts.eval.loop_metrics import depth_loop_rows, summarize_trajectories, write_csv
    from scripts.eval.pointer_task import load_examples, sha256_file, synchronize
    from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
    from scripts.training.data import encode_tasks
    from scripts.training.evaluation import evaluate

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select an available device explicitly")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; select an available device explicitly")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    examples = load_examples(args.data, args.test if args.test is not None else args.limit)
    loops = args.loops or max(item.task_depth for item in examples)
    if loops < max(item.task_depth for item in examples):
        parser.error("--loops must cover the deepest selected task; no targets are truncated")
    console = Console()
    console.print("[bold cyan]Pointer task · full recurrent loop evaluation[/bold cyan]")
    console.print(f"{len(examples):,} examples · {loops} loops each · {args.device} · float32 · batch {args.batch_size}")
    console.print("Raw dataset prompts · A–Z readout · nominal targets only · cache disabled")
    console.print(f"Results: {output.resolve()}")
    if args.test:
        console.print("[yellow]TEST MODE: small subset with inputs and per-loop scoring.[/yellow]")
    loading_started = perf_counter()
    with console.status("Loading saved adapters, tokenizer, and frozen base…"):
        model, tokenizer, spec = load_recurrent_checkpoint(args.model, device=args.device, download=args.download)
        if tokenizer.pad_token_id is None:
            raise ValueError("Checkpoint tokenizer requires a pad token")
        items = encode_tasks(examples, tokenizer, dict(zip(spec["symbols"], spec["token_ids"], strict=True)),
                             model.config.max_position_embeddings)
    loading_seconds = perf_counter() - loading_started
    metadata = {
        "status": "running", "schema_version": 1, "started_utc": started.isoformat(),
        "command": [sys.executable, "-m", "scripts.eval.loop_test", *sys.argv[1:]],
        "checkpoint": str(args.model.resolve()), "model": spec,
        "data": str(args.data.resolve()), "data_sha256": sha256_file(args.data),
        "selected_examples": len(examples), "limit": args.limit, "test_mode": args.test is not None,
        "loops": loops, "device": args.device, "dtype": "float32", "batch_size": args.batch_size,
        "seed": args.seed, "threads": args.threads, "use_cache": False, "deterministic_algorithms": True,
        "scoring": "allowed_symbol_argmax_at_each_loop", "prompt_format": "dataset_raw",
        "target_semantics": "Intermediate targets only for 1 <= loop <= task_depth; final-target sweep is observational",
        "loss_reduction": "Mean nominal loops per example, then mean examples; per-loop means use eligible examples",
        "tie_breaking": "First symbol in checkpoint order (A–Z); positive margin is stricter than argmax correctness",
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "peft", "rich")},
        "local_checkpoint_sha256": {path.name: sha256_file(path) for path in sorted(args.model.iterdir())
                                    if path.is_file() and path.name != "training_state.pt"},
        "source_sha256": {str(path.relative_to(root)): sha256_file(path)
                          for pattern in ("scripts/eval/*.py", "scripts/recurrent_qwen/*.py", "scripts/training/*.py", "scripts/dataset/*.py")
                          for path in sorted(root.glob(pattern))},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "loading_and_encoding_seconds": loading_seconds,
        "throughput_definition": "Questions and executed example-loops per evaluation second, including scoring/trajectory CSV writing; excludes loading and final diagnostic exports",
    }
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    device = torch.device(args.device)
    synchronize(device)
    began = perf_counter()
    with Progress(TextColumn("Evaluating"), BarColumn(), TextColumn("{task.completed:.0f}/{task.total:.0f}"),
                  TimeElapsedColumn(), TimeRemainingColumn(), TextColumn("{task.fields[rates]}"), console=console) as progress:
        task = progress.add_task("Evaluating", total=len(items), rates="")

        def update(done: int, total: int) -> None:
            rate = done / max(perf_counter() - began, 1e-9)
            progress.update(task, completed=done, rates=f"{rate:.2f} q/s · {rate * loops:.2f} example-loops/s")

        metrics = evaluate(model, items, spec["token_ids"], tokenizer.pad_token_id, batch_size=args.batch_size,
                           loops=loops, output=output / "trajectories.csv", progress=update)
    synchronize(device)
    elapsed = perf_counter() - began
    example_rows, diagnostics = summarize_trajectories(output / "trajectories.csv")
    write_csv(output / "examples.csv", example_rows)
    write_csv(output / "depth_by_loop.csv", depth_loop_rows(metrics))
    summary_path.write_text(json.dumps({
        **metadata, **metrics, "diagnostics": diagnostics, "status": "complete",
        "evaluation_seconds": elapsed, "executed_example_loops": len(items) * loops,
        "questions_per_second": len(items) / elapsed, "example_loops_per_second": len(items) * loops / elapsed,
    }, indent=2) + "\n")
    if args.test:
        for item, row in zip(items, example_rows, strict=True):
            console.print(Panel(Text(item.task.prompt), title=item.task.example_id))
            table = Table(title="Per-loop readout")
            for column in ("Loop", "Prediction", "Target", "Verdict"):
                table.add_column(column)
            for t, prediction in enumerate(row["predictions"].split(), 1):
                target = item.task.intermediate_states[t - 1] if t <= item.task.task_depth else ""
                verdict = ("[green]RIGHT[/green]" if prediction == target else "[red]WRONG[/red]") if target else "post-completion"
                table.add_row(str(t), prediction, target or "—", verdict)
            console.print(table)
    table = Table(title="Nominal execution by task depth")
    for column in ("Depth", "Examples", "Step accuracy", "All steps correct", "Final accuracy"):
        table.add_column(column)
    for depth, values in metrics["by_depth"].items():
        table.add_row(depth, str(values["examples"]), f"{values['intermediate_accuracy']:.2%}",
                      f"{values['trajectory_accuracy']:.2%}", f"{metrics['depth_by_loop'][depth][depth]['final_accuracy']:.2%}")
    console.print(table)
    console.print(f"[bold]All steps correct: {metrics['trajectory_accuracy']:.2%} · final accuracy: {metrics['final_accuracy']:.2%}[/bold]")
    console.print(f"Saved trajectories.csv, examples.csv, depth_by_loop.csv, and summary.json to {output.resolve()}")


if __name__ == "__main__":
    main()
