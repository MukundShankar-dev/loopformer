"""Evaluate ordinary or saved recurrent final answers on the pointer test set.

Run from the repository root: python -m scripts.eval.naive_test --help
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face ID, saved model directory, or full state-dict file")
    parser.add_argument("--base-model", help="Architecture/config and default tokenizer source for a standalone state dict")
    parser.add_argument("--tokenizer", help="Override the tokenizer source")
    parser.add_argument("--revision", help="Model/base revision; defaults to pinned Qwen revision for the project model, main otherwise")
    parser.add_argument("--download", action="store_true", help="Explicitly allow missing Hub files to download (default: cached/local only)")
    parser.add_argument("--data", type=Path, default=root / "data/pointer/seed-17/test.jsonl")
    parser.add_argument("--prompt", type=Path, default=root / "prompts/pointer_task.txt")
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--limit", type=int, help="Evaluate only the first N rows (default: entire test file)")
    subset.add_argument("--test", nargs="?", type=int, const=3, choices=(2, 3), help="Run 3 (or 2) examples and show exact prompts, responses, and scoring")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="New results directory; default eval/pointer_task/<timestamp>-<model>")
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.threads) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size, token budget, thread count, and limit must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    recurrent = (Path(args.model) / "recurrent_config.json").is_file()
    for path in ((args.data,) if recurrent else (args.data, args.prompt)):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    started_utc = datetime.now(timezone.utc)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model).strip("-.")[-80:] or "model"
    mode_label = "test-" if args.test is not None else ""
    output = args.output or root / "eval/pointer_task" / f"{started_utc:%Y%m%dT%H%M%S.%fZ}-{mode_label}{name}"
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output")

    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text

    from scripts.eval.loading import load_model
    from scripts.eval.pointer_task import (
        evaluate_batches, load_examples, sha256_file, summarize_results, validate_template,
    )

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; choose an available --device explicitly")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose an available --device explicitly")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if recurrent:
        from scripts.eval.recurrent_pointer import run_checkpoint_eval
        incompatible = {"--base-model", "--tokenizer", "--revision", "--prompt", "--prompt-format", "--max-new-tokens"}
        supplied = {argument.split("=", 1)[0] for argument in sys.argv[1:]}
        if supplied & incompatible:
            parser.error("Recurrent checkpoints fix their base, tokenizer, raw prompt, and loop readout; omit " + ", ".join(sorted(supplied & incompatible)))
        run_checkpoint_eval(args, root, output, started_utc)
        return
    console = Console()
    console.print("[bold cyan]Pointer task · ordinary-model final-answer baseline[/bold cyan]")
    examples = load_examples(args.data, args.test if args.test is not None else args.limit)
    template = args.prompt.read_text()
    validate_template(template)
    console.print(f"{args.model} · {args.device} · {args.dtype} · batch {args.batch_size} · {len(examples):,} questions")
    console.print(f"{args.prompt_format} prompt · greedy · max {args.max_new_tokens} new tokens · cache disabled")
    console.print(f"Results: {output.resolve()}")
    if args.test is not None:
        console.print(f"[bold yellow]TEST MODE: showing {len(examples)} examples and their scoring; this is not the full test set.[/bold yellow]")
    loading_started = perf_counter()
    with console.status("Loading model and tokenizer…"):
        model, tokenizer, model_info = load_model(
            args.model, base_model=args.base_model, tokenizer_source=args.tokenizer,
            revision=args.revision, device=args.device, dtype=args.dtype, download=args.download,
        )
    loading_seconds = perf_counter() - loading_started
    if args.prompt_format == "chat" and not tokenizer.chat_template:
        parser.error("Tokenizer has no chat template; choose --prompt-format raw")

    # Hash local checkpoint artifacts; Hub models are identified by resolved commit.
    checkpoint = Path(model_info["source"])
    weight_files = [checkpoint] if checkpoint.is_file() else sorted(
        [*checkpoint.glob("*.safetensors"), *checkpoint.glob("*.bin"), *checkpoint.glob("*.json")]
    ) if checkpoint.is_dir() else []
    sources = [*sorted((root / "scripts/eval").glob("*.py")), root / "scripts/dataset/pointer.py"]
    metadata = {
        "status": "running", "started_utc": started_utc.isoformat(),
        "command": [sys.executable, "-m", "scripts.eval.naive_test", *sys.argv[1:]],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": model_info, "model_config": model.config.to_dict(),
        "local_checkpoint_sha256": {str(path.resolve()): sha256_file(path) for path in weight_files},
        "tokenizer_backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() if tokenizer.is_fast else None,
        "chat_template": tokenizer.chat_template if args.prompt_format == "chat" else None,
        "prompt_template": template, "prompt_sha256": sha256_file(args.prompt),
        "data_sha256": sha256_file(args.data), "selected_examples": len(examples),
        "test_mode": args.test is not None,
        "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in sources},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "rich")},
        "scoring": "unconstrained_generation_exact_uppercase_symbol_after_strip",
        "use_cache": False, "do_sample": False, "deterministic_algorithms": True,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": model.generation_config.eos_token_id if model.generation_config.eos_token_id is not None else tokenizer.eos_token_id,
        "loading_seconds": loading_seconds,
        "throughput_definition": "generated_tokens/s uses synchronized model.generate time, including prefill and EOS but excluding batch padding; questions/s uses evaluation-loop wall time, excluding loading",
    }
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    all_rows = []
    generation_seconds = 0.0
    evaluation_started = perf_counter()
    with (output / "predictions.csv").open("w", newline="") as handle, Progress(
        TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed:.0f}/{task.total:.0f}"),
        TimeElapsedColumn(), TimeRemainingColumn(),
        TextColumn("{task.fields[rates]}"), console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(examples), rates="")
        writer = None
        for rows, seconds in evaluate_batches(
            model, tokenizer, examples, template, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, prompt_format=args.prompt_format,
        ):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            if args.test is not None:
                for row in rows:
                    console.print(Panel(Text(row["model_input"]), title=f"Model input · {row['example_id']} · {row['task_depth']} steps"))
                    console.print(Panel(Text(repr(row["response"])), title="Decoded response (repr preserves whitespace)"))
                    verdict = "[bold green]RIGHT[/bold green]" if row["correct"] else "[bold red]WRONG[/bold red]"
                    prediction = row["prediction"] or "<invalid answer format>"
                    console.print(f"{verdict} · expected: {row['target']} · parsed: {prediction} · stop: {row['stop_reason']}")
                    console.print(Text(f"Generated token IDs: {row['generated_token_ids']}"))
            all_rows.extend(rows)
            generation_seconds += seconds
            metrics = summarize_results(all_rows, generation_seconds, perf_counter() - evaluation_started)
            progress.update(task, advance=len(rows), rates=(
                f"{metrics['accuracy']:.1%} acc · {metrics['generated_tokens_per_second']:.1f} tok/s · "
                f"{metrics['questions_per_second']:.2f} q/s"
            ))
    metrics = summarize_results(all_rows, generation_seconds, perf_counter() - evaluation_started)
    summary_path.write_text(json.dumps({**metadata, "status": "complete", **metrics}, indent=2) + "\n")
    table = Table(title="Final-answer accuracy by depth")
    for column in ("Steps", "Correct", "Total", "Accuracy"):
        table.add_column(column)
    for depth, values in metrics["by_depth"].items():
        table.add_row(depth, str(values["correct"]), str(values["total"]), f"{values['accuracy']:.2%}")
    console.print(table)
    console.print(f"[bold]Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})[/bold]")
    console.print(f"Invalid answers: {metrics['invalid_answers']} · token-budget stops: {metrics['token_budget_stops']}")
    console.print(f"{metrics['generated_tokens_per_second']:.2f} generated tok/s · {metrics['questions_per_second']:.2f} questions/s")
    console.print(f"Saved predictions.csv and summary.json to {output.resolve()}")


if __name__ == "__main__":
    main()
