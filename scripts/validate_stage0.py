"""Load Qwen, create P/R/C, and verify Stage 0 with Rich output and JSON evidence.

Run from the repository root: python -m scripts.validate_stage0 --help
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--download", action="store_true", help="Explicitly allow missing Hub files to download.")
    parser.add_argument("--recurrent-start", type=int, default=6)
    parser.add_argument("--recurrent-end", type=int, default=18)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="JSON destination; default artifacts/stage0/<timestamp>-<device>.json")
    args = parser.parse_args()
    if min(args.loops) < 1 or max(args.loops) < 2:
        parser.error("--loops must be positive and include a depth > 1")
    if min(args.rank, args.alpha, args.threads) < 1:
        parser.error("--rank, --alpha, and --threads must be positive")
    if not 0 <= args.recurrent_start < args.recurrent_end <= 24:
        parser.error("Require 0 <= recurrent-start < recurrent-end <= 24")

    import torch
    from rich.console import Console
    from rich.table import Table
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    from scripts.recurrent_qwen.validation import validate_stage0

    console = Console()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; no automatic device fallback is performed")
    if not __debug__:
        parser.error("Validation uses assertions; run Python without -O")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    started = datetime.now(timezone.utc)
    start = perf_counter()
    console.print(f"[bold]Stage 0[/bold] · {model_id}")
    console.print(f"{args.device} · float32 · eager attention · no KV cache · seed {args.seed}")
    console.print(f"Revision: {args.revision}; downloads allowed: {args.download}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision, local_files_only=not args.download)
    base = Qwen2ForCausalLM.from_pretrained(
        model_id, revision=args.revision, local_files_only=not args.download,
        dtype=torch.float32, attn_implementation="eager", use_safetensors=True,
    ).to(args.device).eval()
    prompts = ["The sky is blue.", "What is two plus two? Answer briefly."]
    cases = {}
    for padding_side in ("left", "right"):
        tokenizer.padding_side = padding_side
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)
        cases[f"{padding_side}_padded"] = dict(encoded)
    cases["unpadded"] = dict(tokenizer(prompts[0], return_tensors="pt").to(args.device))
    console.print(f"Loaded {len(base.model.layers)} layers; hidden size {base.config.hidden_size}")
    partition = Table(title="Model surgery (zero-based layer indices)")
    partition.add_column("Part")
    partition.add_column("Layers")
    partition.add_column("Trainability")
    for name, first, end, trainability in (
        ("P: prelude", 0, args.recurrent_start, "Frozen"),
        ("R: shared recurrence", args.recurrent_start, args.recurrent_end, "q/v LoRA only"),
        ("C: coda", args.recurrent_end, len(base.model.layers), "Frozen"),
    ):
        partition.add_row(name, f"{first}–{end - 1}" if first < end else "Empty", trainability)
    console.print(partition)
    console.print(f"Embeddings, norm, and LM head frozen; LoRA rank {args.rank}, alpha {args.alpha}; no bridge")
    measured = validate_stage0(
        base, cases, recurrent_start=args.recurrent_start, recurrent_end=args.recurrent_end,
        rank=args.rank, alpha=args.alpha, loop_counts=tuple(args.loops), progress=console.print,
    )
    root = Path(__file__).resolve().parents[1]
    source_paths = sorted([
        *root.glob("scripts/**/*.py"), *root.glob("tests/*.py"),
        root / "requirements.txt",
    ])
    report = {
        "started_utc": started.isoformat(),
        "command": [sys.executable, "-m", "scripts.validate_stage0", *sys.argv[1:]],
        "model": model_id,
        "resolved_model_revision": base.config._commit_hash,
        "model_config": base.config.to_dict(),
        "device": args.device, "dtype": "float32", "attention_implementation": "eager",
        "use_cache": False, "seed": args.seed, "threads": args.threads,
        "deterministic_algorithms": True,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "prompts": prompts,
        "inputs": {name: {key: value.tolist() for key, value in values.items()} for name, values in cases.items()},
        **measured,
        "total_seconds": perf_counter() - start,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
    }
    output = args.output or Path("artifacts/stage0") / f"{started:%Y%m%dT%H%M%SZ}-{args.device}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    table = Table(title="Stage 0 validation")
    table.add_column("Check")
    table.add_column("Result")
    for name, status in measured["checks"].items():
        table.add_row(name.replace("_", " "), f"[green]{status}[/green]")
    console.print(table)
    maximum = max(item["max_absolute_error"] for item in measured["equivalence"])
    console.print(f"Max T=1 logit error: {maximum:.3g}; trainable parameters: {measured['parameters']['trainable']:,}")
    console.print(f"Elapsed: {report['total_seconds']:.2f}s; peak process RSS: {report['peak_process_rss_bytes'] / 2**30:.2f} GiB")
    console.print(f"Evidence: {output.resolve()}")
    console.print("[green]Gate 0 passed.[/green] No research training performed; stop before pointer data.")


if __name__ == "__main__":
    main()
