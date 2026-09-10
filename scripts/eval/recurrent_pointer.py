"""Final-answer readout for Stage 1 checkpoints, composed by naive_test."""

import csv
from argparse import Namespace
from datetime import datetime
import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Iterator

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from scripts.dataset.pointer import PointerExample
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.model import RecurrentQwen
from .pointer_task import load_examples, sha256_file, summarize_results, synchronize


def evaluate_recurrent_batches(model: RecurrentQwen, tokenizer: Any, examples: list[PointerExample], spec: dict,
                               *, batch_size: int) -> Iterator[tuple[list[dict], float]]:
    """Read A–Z logits at loop d, using raw prompts and no generated text.

    A mixed-depth batch executes max(d) shared loops; each row is scored at its
    own requested depth. Report the executed example-loops, including padding.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    tokenizer.padding_side = "right"
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompts = [example.prompt for example in batch]
        inputs = tokenizer(prompts, padding=True, add_special_tokens=False,
                           return_token_type_ids=False, return_tensors="pt").to(device)
        if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
            raise ValueError("Prompt exceeds model context; no truncation performed")
        loops = max(example.task_depth for example in batch)
        synchronize(device)
        began = perf_counter()
        with torch.inference_mode():
            result = model(**inputs, num_loops=loops)
            predictions = [
                int(result.loop_logits[example.task_depth - 1][row, spec["token_ids"]].argmax())
                for row, example in enumerate(batch)
            ]
        synchronize(device)
        elapsed = perf_counter() - began
        rows = []
        for example, prediction, length in zip(batch, predictions, inputs["attention_mask"].sum(-1).tolist()):
            symbol = spec["symbols"][prediction]
            rows.append({
                "example_id": example.example_id, "split": example.split, "seed": example.seed,
                "task_depth": example.task_depth, "target": example.final_state,
                "response": symbol, "prediction": symbol, "valid_answer": True,
                "correct": symbol == example.final_state, "prompt_tokens": length,
                "generated_tokens": 0, "generated_token_ids": "[]", "stop_reason": "recurrent_depth",
                "readout_loop": example.task_depth, "executed_loops": loops,
                "prediction_token_id": spec["token_ids"][prediction],
                "prompt": example.prompt, "model_input": example.prompt,
            })
        yield rows, elapsed


def run_checkpoint_eval(args: Namespace, root: Path, output: Path, started_utc: datetime) -> None:
    console = Console()
    console.print("[bold cyan]Pointer task · recurrent checkpoint final-answer readout[/bold cyan]")
    console.print("Dataset raw prompt · A–Z argmax at loop = requested steps · cache disabled")
    examples = load_examples(args.data, args.test if args.test is not None else args.limit)
    with console.status("Loading recurrent checkpoint and its frozen base…"):
        model, tokenizer, spec = load_recurrent_checkpoint(Path(args.model), device=args.device,
                                                          dtype=args.dtype, download=args.download)
    console.print(f"{len(examples):,} questions · {args.device} · {args.dtype} · batch {args.batch_size}")
    if args.test:
        console.print("[yellow]TEST MODE: inspecting a small subset, not the full test set.[/yellow]")
    checkpoint = Path(args.model)
    metadata = {
        "status": "running", "started_utc": started_utc.isoformat(), "model": spec,
        "command": [sys.executable, "-m", "scripts.eval.naive_test", *sys.argv[1:]],
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "peft", "rich")},
        "checkpoint": str(checkpoint.resolve()), "device": args.device, "dtype": args.dtype,
        "batch_size": args.batch_size, "seed": args.seed, "threads": args.threads,
        "test_mode": args.test is not None, "selected_examples": len(examples),
        "data_sha256": sha256_file(args.data), "data": str(args.data.resolve()),
        "local_checkpoint_sha256": {path.name: sha256_file(path) for path in checkpoint.iterdir()
                                    if path.is_file() and path.name != "training_state.pt"},
        "source_sha256": {str(path.relative_to(root)): sha256_file(path)
                          for pattern in ("scripts/eval/*.py", "scripts/recurrent_qwen/*.py") for path in root.glob(pattern)},
        "scoring": "allowed_symbol_argmax_at_requested_depth", "prompt_format": "dataset_raw",
        "use_cache": False, "deterministic_algorithms": True,
        "throughput_definition": "example-loops/s counts all executed recurrent loops per example, including mixed-depth batch padding; questions/s excludes loading",
    }
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    all_rows, forward_seconds = [], 0.0
    began = perf_counter()
    with (output / "predictions.csv").open("w", newline="") as handle, Progress(
        TextColumn("Evaluating"), BarColumn(), TextColumn("{task.completed:.0f}/{task.total:.0f}"),
        TimeElapsedColumn(), TimeRemainingColumn(), TextColumn("{task.fields[rates]}"), console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(examples), rates="")
        writer = None
        for rows, seconds in evaluate_recurrent_batches(model, tokenizer, examples, spec, batch_size=args.batch_size):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            all_rows.extend(rows)
            forward_seconds += seconds
            if args.test:
                for row in rows:
                    console.print(Panel(Text(row["model_input"]), title=f"Model input · {row['example_id']} · {row['readout_loop']} loops"))
                    verdict = "[green]RIGHT[/green]" if row["correct"] else "[red]WRONG[/red]"
                    console.print(f"{verdict} · expected: {row['target']} · readout: {row['prediction']} · token ID: {row['prediction_token_id']}")
            metrics = summarize_results(all_rows, forward_seconds, perf_counter() - began)
            loop_rate = sum(row["executed_loops"] for row in all_rows) / forward_seconds
            progress.update(task, advance=len(rows), rates=f"{metrics['accuracy']:.1%} acc · {loop_rate:.2f} example-loops/s · {metrics['questions_per_second']:.2f} q/s")
    metrics.pop("generation_seconds")
    metrics.pop("generated_tokens_per_second")
    metrics.update(forward_seconds=forward_seconds, example_loops_per_second=loop_rate)
    summary_path.write_text(json.dumps({**metadata, **metrics, "status": "complete"}, indent=2) + "\n")
    table = Table(title="Final-answer accuracy at requested depth")
    for label in ("Steps", "Correct", "Total", "Accuracy"):
        table.add_column(label)
    for depth, values in metrics["by_depth"].items():
        table.add_row(depth, str(values["correct"]), str(values["total"]), f"{values['accuracy']:.2%}")
    console.print(table)
    console.print(f"[bold]Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})[/bold]")
    console.print(f"Saved predictions.csv and summary.json to {output.resolve()}")
