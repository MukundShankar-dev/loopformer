"""Configure and launch Stage 1 training; python -m scripts.training.train_pointer."""

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/stage1_pointer.json")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), help="Override the config device explicitly")
    parser.add_argument("--output", type=Path, help="New run directory; default models/stage1_pointer/<timestamp>")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="Resume optimizer/RNG/cursor with matching config")
    initialization.add_argument("--init-from", type=Path, help="Initialize adapters only; new optimizer/RNG/counter, compatible architecture required")
    parser.add_argument("--allow-batch-change", action="store_true", help="With --resume, allow batch/accumulation changes only when examples per optimizer update stay equal")
    parser.add_argument("--batch-size", type=int, help="Explicit microbatch override")
    parser.add_argument("--smoke-test", action="store_true", help="Ten disposable updates and small validation; writes a separate run")
    parser.add_argument("--download", action="store_true", help="Allow missing base model/tokenizer files to download")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data/tokenizer and print the budget; do not load model weights or write output")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default=os.environ.get("WANDB_MODE", "disabled"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "loopformer"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    args = parser.parse_args()
    if args.allow_batch_change and not args.resume:
        parser.error("--allow-batch-change requires --resume")

    import torch
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from scripts.dataset.pointer import SYMBOLS
    from scripts.dataset.symbols import validate_symbols
    from scripts.eval.pointer_task import sha256_file
    from scripts.recurrent_qwen.checkpoint import restore_adapters
    from scripts.training.config import read_config
    from scripts.training.data import collate, encode_tasks, read_tasks, select_tasks
    from scripts.training.gates import initialize, validate_completion_gradients
    from scripts.training.initialization import validate_initialization
    from scripts.training.runner import resume_identity, train, validate_resume_identity

    config = read_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.smoke_test:
        if args.resume or args.init_from:
            parser.error("Smoke test requires fresh initialization")
        config = replace(config, max_steps=10, validation_per_depth=2, train_probe_per_depth=1)
    config.validate()
    if config.completion_loss_weight and args.init_from:
        parser.error("Completion training requires fresh adapters or --resume; --init-from is not supported")
    if config.recurrence_mode != "full_sequence" and args.init_from:
        parser.error("fixed_prompt training requires fresh adapters or matching --resume")
    output = args.output or root / "models/stage1_pointer" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if output.exists() and not args.dry_run:
        parser.error("Output directory already exists; choose a new --output")
    if not args.dry_run:
        if config.device == "mps" and not torch.backends.mps.is_available():
            parser.error("MPS unavailable; choose --device cpu explicitly")
        if config.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA unavailable; choose --device cpu explicitly")
    if not args.dry_run and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA device does not support BF16")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    console = Console()
    console.print("[bold cyan]Stage 1 · shared recurrent pointer training[/bold cyan]")
    train_path, validation_path = root / config.train_data, root / config.validation_data
    with console.status("Validating data, symbol contexts, and selected examples…"):
        train_tasks, validation_tasks = read_tasks(train_path, "train"), read_tasks(validation_path, "validation")
        if {task.mapping_sha256 for task in train_tasks} & {task.mapping_sha256 for task in validation_tasks}:
            raise ValueError("Training and validation contain overlapping rule tables")
        selected = select_tasks(train_tasks, config.train_max_depth, config.seed, limit=config.train_limit, depths=config.train_depths)
        validation_selected = select_tasks(validation_tasks, config.validation_max_depth, config.seed + 1, per_depth=config.validation_per_depth)
        probe_selected = select_tasks(selected, config.train_max_depth, config.seed + 2, per_depth=config.train_probe_per_depth, depths=config.train_depths)
        tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision, local_files_only=args.dry_run or not args.download)
        token_map = validate_symbols(tokenizer)
        if tokenizer.pad_token_id is None:
            raise ValueError("Training tokenizer needs a pad token")
        train_items = encode_tasks(selected, tokenizer, token_map, config.max_prompt_tokens)
        validation_items = encode_tasks(validation_selected, tokenizer, token_map, config.max_prompt_tokens)
        probe_items = encode_tasks(probe_selected, tokenizer, token_map, config.max_prompt_tokens)
    token_ids = [token_map[symbol] for symbol in SYMBOLS]
    spec = {"base_model": config.model, "revision": config.revision,
            "recurrent_start": config.recurrent_start, "recurrent_end": config.recurrent_end,
            "lora_rank": config.lora_rank, "lora_alpha": config.lora_alpha,
            "symbols": list(SYMBOLS), "token_ids": token_ids,
            "prompt_format": "dataset_raw", "loss_vocabulary": "symbols", "train_max_depth": config.train_max_depth}
    if config.train_depths is not None:
        spec["train_depths"] = config.train_depths
    if config.loss_reduction != "example_mean":
        spec["loss_reduction"] = config.loss_reduction
    if config.recurrence_mode != "full_sequence":
        spec["recurrence_mode"] = config.recurrence_mode
    if config.completion_loss_weight:
        spec["completion_head"] = {"intermediate": config.completion_head_hidden_size}
        spec["completion_loss_weight"] = config.completion_loss_weight
    if config.precision != "float32" or config.attention != "eager":
        spec["training_compute"] = {"precision": config.precision, "attention": config.attention}
    if (config.train_scope != "lora" or config.isolated_controller or config.reentry_bridge
            or config.state_loss_weight or config.prefix_reuse or config.gradient_checkpointing):
        spec["executor"] = {
            "train_scope": config.train_scope, "isolated": config.isolated_controller,
            "bridge": config.reentry_bridge, "direct_readout": config.state_loss_weight > 0,
            "controller_size": config.completion_head_hidden_size if config.completion_loss_weight else 0,
            "prefix_reuse": config.prefix_reuse, "gradient_checkpointing": config.gradient_checkpointing,
        }
    initialization_metadata = None
    if args.init_from:
        source_spec = validate_initialization(args.init_from, spec)
        initialization_metadata = {"checkpoint": str(args.init_from.resolve()), "source_train_max_depth": source_spec["train_max_depth"],
                                   "mode": "adapters_only", "optimizer_rng_cursor": "fresh"}
        if not args.dry_run:
            saved_tokenizer = AutoTokenizer.from_pretrained(args.init_from, local_files_only=True, trust_remote_code=False)
            if saved_tokenizer.backend_tokenizer.to_str() != tokenizer.backend_tokenizer.to_str():
                raise ValueError("Initialization tokenizer differs from the configured tokenizer")
            initialization_metadata["sha256"] = {name: sha256_file(args.init_from / name)
                                                  for name in ("adapter_model.pt", "recurrent_config.json", "tokenizer.json")}
    effective_batch = config.batch_size * config.gradient_accumulation
    planned = ((len(train_items) + effective_batch - 1) // effective_batch) * config.epochs
    planned = min(planned, config.max_steps) if config.max_steps else planned
    table = Table(title="Run configuration")
    table.add_column("Setting")
    table.add_column("Value")
    for label, value in (
        ("Model", f"{config.model} @ {config.revision}"),
        ("Device / dtype", f"{config.device} / {config.precision} compute, float32 parameters · {config.attention} · deterministic={config.deterministic}"),
        ("Recurrence", f"{config.recurrence_mode} · scope={config.train_scope} · isolated={config.isolated_controller} · bridge={config.reentry_bridge}"),
        ("Training", f"{len(train_items):,} examples · depths {config.train_depths or list(range(1, config.train_max_depth + 1))} · {config.epochs} epochs"),
        ("Validation / train probe", f"{len(validation_items)} / {len(probe_items)} examples · validation depths 1–{config.validation_max_depth}"),
        ("Depth bucketing / validation batch", f"{config.bucket_by_depth} / {config.validation_batch_size or config.batch_size}"),
        ("Batch / accumulation", f"{config.batch_size} × {config.gradient_accumulation} = {effective_batch} examples/update"),
        ("Updates / learning rate", f"{planned} / {config.learning_rate:g}"),
        ("Loss", f"26-symbol CE · {config.loss_reduction} · nominal targets only" +
         (f" + {config.completion_loss_weight:g} × hidden-state completion BCE" if config.completion_loss_weight else "")),
        ("Prompt", "Dataset raw rules/start/steps · no few-shot examples"),
        ("Output", str(output.resolve())),
        ("Initialization", str(args.init_from) + " · adapters only, new optimizer" if args.init_from else
         (str(args.resume) + " · resume" if args.resume else f"Fresh {config.train_scope} adaptation")),
    ):
        table.add_row(label, value)
    console.print(table)
    if args.dry_run:
        console.print("[green]Dry run passed.[/green] No model weights loaded, training performed, or output files written.")
        return
    with console.status("Loading base and checking T=1 equivalence, weight sharing, and two-loop gradients…"):
        base = Qwen2ForCausalLM.from_pretrained(
            config.model, revision=config.revision, local_files_only=not args.download,
            dtype=torch.float32, attn_implementation=config.attention, trust_remote_code=False,
        ).to(config.device).eval()
        if max(len(item.input_ids) for item in [*train_items, *validation_items]) > base.config.max_position_embeddings:
            raise ValueError("Selected prompts exceed model context size")
        probe = collate(train_items[:1], tokenizer.pad_token_id, config.device)
        model, gate = initialize(base, config, probe, token_ids)
        if "executor" in spec:
            model.configure_executor(tokenizer, **spec["executor"])
            from scripts.training.executor_gates import validate_executor
            gate["executor"] = validate_executor(model, probe, token_ids)
        if config.completion_loss_weight:
            if model.completion_head is None:
                model.enable_completion(config.completion_head_hidden_size)
            if not config.isolated_controller:
                gate["completion"] = validate_completion_gradients(model, probe)
            gate["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gate["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if config.precision == "bf16":
        from scripts.training.precision import validate_compute_batch
        console.print("Checking BF16 loss/gradients at the deepest training microbatch…")
        gate["compute_probe"] = validate_compute_batch(model, train_items, tokenizer.pad_token_id, token_ids, config)
        console.print(f"[green]Compute probe passed[/green]: {gate['compute_probe']}")
    console.print(f"[green]Architecture/loss-path checks passed[/green] · {gate['trainable_parameters']:,} trainable parameters")
    spec["revision"] = getattr(base.config, "_commit_hash", None) or config.revision
    if args.init_from:
        validate_initialization(args.init_from, spec)
        restore_adapters(model, args.init_from)
        spec["initialization"] = initialization_metadata
    if args.resume:
        saved_spec = json.loads((args.resume / "recurrent_config.json").read_text())
        if "initialization" in saved_spec:
            initialization_metadata = saved_spec["initialization"]
            spec["initialization"] = initialization_metadata
    data_identity = {"train_sha256": sha256_file(train_path), "validation_sha256": sha256_file(validation_path),
                     "selected_ids_sha256": hashlib.sha256(json.dumps([[item.task.example_id for item in items] for items in (train_items, validation_items, probe_items)]).encode()).hexdigest(),
                     "spec": spec, "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    identity = resume_identity(config, data_identity)
    resume = None
    batch_change = None
    if args.resume:
        if not (args.resume / "recurrent_config.json").is_file():
            raise ValueError("Resume directory is not a completed recurrent checkpoint")
        resume = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        batch_change = validate_resume_identity(resume["identity"], identity, allow_batch_change=args.allow_batch_change)
        restore_adapters(model, args.resume)
        if batch_change:
            console.print("[yellow]Resuming with changed microbatches; optimizer and data cursor retained. "
                          "Floating-point results may differ.[/yellow]")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    sources = [*root.glob("scripts/training/*.py"), *root.glob("scripts/recurrent_qwen/*.py"), *root.glob("scripts/dataset/*.py"), root / "scripts/eval/pointer_task.py"]
    metadata = {"status": "running", "command": [sys.executable, "-m", "scripts.training.train_pointer", *sys.argv[1:]],
                "gate": gate, "identity": identity, "resume": str(args.resume) if args.resume else None,
                "initialization": initialization_metadata, "batch_change": batch_change,
                "python": platform.python_version(), "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "peft", "rich")},
                "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in sources},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    state = {"phase": "Starting", "step": 0, "total_steps": planned}
    bar = Progress(TextColumn("Updates"), BarColumn(), TextColumn("{task.completed:.0f}/{task.total:.0f}"), TimeElapsedColumn(), auto_refresh=False)
    task = bar.add_task("Training", total=planned)

    def render():
        dashboard = Table.grid(padding=(0, 2))
        dashboard.add_column()
        dashboard.add_column()
        dashboard.add_row("Phase", state["phase"])
        if "train" in state:
            values = state["train"]
            dashboard.add_row("Training", f"objective {values['objective_loss']:.4f} · example CE {values['loss']:.4f} · step accuracy {values['intermediate_accuracy']:.1%} · trajectory {values['trajectory_accuracy']:.1%}")
            dashboard.add_row("Loop losses", "  ".join(f"L{t}: {value['loss']:.3f}" for t, value in values["per_loop"].items()))
            if "completion" in values:
                stop = values["completion"]
                dashboard.add_row("Completion", f"loss {stop['objective_loss']:.4f} · exact stop {stop['first_stop_exact_rate']:.1%} · early {stop['first_stop_early_rate']:.1%}")
        if "validation" in state:
            values = state["validation"]
            dashboard.add_row("Validation", f"loss {values['loss']:.4f} · step accuracy {values['intermediate_accuracy']:.1%} · trajectory {values['trajectory_accuracy']:.1%}")
            dashboard.add_row("Checkpoint selection", f"{config.loss_reduction} · trained-depth loss {values['selection_loss']:.4f}")
            if "completion" in values:
                stop = values["completion"]
                dashboard.add_row("Validation stop", f"loss {stop['objective_loss']:.4f} · exact {stop['first_stop_exact_rate']:.1%} · early {stop['first_stop_early_rate']:.1%}")
        if "train_probe" in state:
            dashboard.add_row("Fixed train probe", f"trajectory {state['train_probe']['trajectory_accuracy']:.1%}")
        if "gradient_norm_before_clip" in state:
            dashboard.add_row("Optimization", f"grad norm {state['gradient_norm_before_clip']:.3g} · lr {state['learning_rate']:.3g}")
            dashboard.add_row("Resources", f"{state['examples_per_second']:.2f} examples/s · host RAM peak {state['peak_process_rss_bytes'] / 2**30:.2f} GiB")
            if "cuda_peak_bytes" in state:
                dashboard.add_row("CUDA tensors", f"between updates {state['cuda_allocated_bytes'] / 2**30:.2f} GiB · update peak {state['update_peak_allocated_bytes'] / 2**30:.2f} GiB · lifetime peak {state['cuda_peak_bytes'] / 2**30:.2f} GiB")
            dashboard.add_row("Estimated remaining", str(timedelta(seconds=round(state["eta_seconds"]))))
        return Group(Panel(dashboard, title="Stage 1 training"), bar)

    from scripts.training.tracking import tracking_run
    with tracking_run(output, metadata, config.to_dict(), mode=args.wandb_mode,
                      project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name) as tracker, \
            Live(render(), console=console, refresh_per_second=4) as live:
        def progress(event: dict) -> None:
            state.update(event)
            bar.update(task, completed=state["step"], total=state["total_steps"])
            live.update(render())
        result = train(model, tokenizer, train_items, validation_items, probe_items, config, output, spec, identity,
                       resume=resume, allow_batch_change=args.allow_batch_change, progress=progress, event_sink=tracker.log)
        tracker.summary(result)
        progress({"phase": "Complete", "step": result["step"]})
    (output / "run.json").write_text(json.dumps({**metadata, "status": "complete"}, indent=2) + "\n")
    console.print(f"[green]Run complete.[/green] Last checkpoint: {result['last_checkpoint']}")
    console.print(f"Metrics and validation trajectories: {output.resolve()}")


if __name__ == "__main__":
    main()
