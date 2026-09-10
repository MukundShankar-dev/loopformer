"""Generate or verify reproducible Stage 1 pointer data (no model training).

Run from the repository root: python -m scripts.dataset --help
"""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.dataset.dataset import (
    DatasetConfig, SPLITS, generate_dataset, make_record, preview_indices,
    verify_dataset, write_dataset,
)
from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION, validate_symbols


def show_summary(console: Console, manifest: dict) -> None:
    table = Table(title="Validated pointer dataset")
    for column in ("Split", "Examples", "Depths (count)", "Prompt tokens"):
        table.add_column(column)
    for split in SPLITS:
        summary = manifest["splits"][split]
        depths = ", ".join(f"{depth}: {count}" for depth, count in sorted(summary["depth_counts"].items(), key=lambda item: int(item[0])))
        table.add_row(split, f"{summary['count']:,}", depths, f"{summary['prompt_tokens_min']}–{summary['prompt_tokens_max']}")
    console.print(table)
    console.print("[green]PASS[/green] Exact targets, tokenizer contexts, unique rule tables, split boundaries, and seed replay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-count", type=int, default=10_000)
    parser.add_argument("--validation-count", type=int, default=1_000)
    parser.add_argument("--test-count", type=int, default=1_000)
    parser.add_argument("--depth-test-count", type=int, default=1_000)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-train-depth", type=int, default=8)
    parser.add_argument("--max-eval-depth", type=int, default=16)
    parser.add_argument("--output", type=Path, help="New directory; default data/pointer/seed-<seed>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", nargs="?", const=5, type=int, choices=(5, 10), help="Preview up to 5 (default) or 10 examples spanning depths; write no dataset files")
    mode.add_argument("--verify", type=Path, metavar="DIRECTORY", help="Read and independently verify an existing dataset")
    args = parser.parse_args()
    config = DatasetConfig(**{key: getattr(args, key) for key in asdict(DatasetConfig())})
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    output = args.output or Path("data/pointer") / f"seed-{args.seed}"
    if args.verify is None and args.dry_run is None and output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output (never overwritten)")

    from transformers import AutoTokenizer

    console = Console()
    console.print("[bold cyan]Stage 1 · pointer data[/bold cyan]")
    console.print(f"Cached tokenizer only: {MODEL_ID} @ {MODEL_REVISION}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    if args.verify is not None:
        with console.status("Verifying saved files and replaying seeds…"):
            manifest = verify_dataset(args.verify, tokenizer)
        show_summary(console, manifest)
        console.print(f"Verified: {args.verify.resolve()}")
        return

    token_ids = validate_symbols(tokenizer)
    console.print(f"[green]PASS[/green] 26 unique single-token symbols in rule, start, and answer contexts; seed {config.seed}")
    console.print(f"Training/instance depths: {config.min_depth}–{config.max_train_depth}; held-out depths: {config.max_train_depth + 1}–{config.max_eval_depth}")
    if args.dry_run is not None:
        distribution = Table(title="Configured dataset distribution (not preview frequencies)")
        for column in ("Split", "Total", "Steps: example count"):
            distribution.add_column(column)
        for split in SPLITS:
            depths = config.depths(split)
            quotient, remainder = divmod(config.count(split), len(depths))
            counts = ", ".join(f"{depth}: {quotient + (index < remainder)}" for index, depth in enumerate(depths))
            distribution.add_row(split, f"{config.count(split):,}", counts)
        console.print(distribution)
        selected = preview_indices(config, args.dry_run)
        console.print("[cyan]Preview spans shallow through deep tasks; it is not a random frequency sample.[/cyan]")
        for split, index in selected:
            record = make_record(config, split, index, tokenizer, token_ids)
            console.print(Panel(Text(record["prompt"]), title=f"{record['example_id']} · depth {record['task_depth']}"))
            table = Table(title=f"Start {record['initial_state']} · seed {record['seed']}")
            for column in ("Step / loop", "Expected symbol", "Answer token ID"):
                table.add_column(column)
            for step, (symbol, token_id) in enumerate(zip(record["intermediate_states"], record["intermediate_token_ids"]), 1):
                table.add_row(str(step), symbol, str(token_id))
            console.print(table)
            console.print(f"[bold green]Final: {record['final_state']}[/bold green] · answer position {record['answer_position']}")
        console.print(f"[yellow]Dry run complete: {len(selected)} examples; no dataset files written.[/yellow]")
        return

    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "scripts/dataset").glob("*.py"))
    provenance = {
        "command": [sys.executable, "-m", "scripts.dataset", *sys.argv[1:]],
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("transformers", "tokenizers", "huggingface-hub", "rich")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    }
    with console.status("Generating and checking every example…"):
        dataset = generate_dataset(config, tokenizer, token_ids)
        write_dataset(output, dataset, config, token_ids, provenance)
    with console.status("Reading files back and independently replaying seeds…"):
        manifest = verify_dataset(output, tokenizer)
    show_summary(console, manifest)
    console.print(f"Dataset and manifest: [bold]{output.resolve()}[/bold]")
    console.print("Data milestone complete. No training performed.")


if __name__ == "__main__":
    main()
