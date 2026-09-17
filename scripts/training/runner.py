"""Optimizer updates, validation, progress events, and resumable checkpoints."""

import json
import math
from pathlib import Path
import random
import resource
import sys
from time import perf_counter
from typing import Any, Callable

import torch

from scripts.recurrent_qwen.checkpoint import save_checkpoint
from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.eval.pointer_task import synchronize
from .config import TrainingConfig
from .data import EncodedExample, collate
from .evaluation import evaluate
from .objective import batch_metrics, combine_metrics, step_loss, symbolic_scores, loop_loss_weights, training_selection_loss


def memory_usage(device: str) -> dict:
    values = {"peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    if device == "mps":
        values.update(mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
    elif device == "cuda":
        values.update(cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_peak_bytes=torch.cuda.max_memory_allocated())
    return values


def rng_state(device: str) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if device == "mps":
        state["device"] = torch.mps.get_rng_state()
    elif device == "cuda":
        state["device"] = torch.cuda.get_rng_state()
    return state


def restore_rng(state: dict, device: str) -> None:
    torch.set_rng_state(state["cpu"])
    if "device" in state:
        if device == "mps":
            torch.mps.set_rng_state(state["device"])
        elif device == "cuda":
            torch.cuda.set_rng_state(state["device"])


def resume_identity(config: TrainingConfig, data_identity: dict) -> dict:
    # A continuation can extend its budget or change reporting cadence. All
    # sampling, optimization, model, validation, and device settings must match.
    settings = config.to_dict()
    # Historical checkpoints predate this field and used equal-example loss.
    if settings["loss_reduction"] == "example_mean":
        settings.pop("loss_reduction")
    for name in ("epochs", "max_steps", "eval_every", "save_every"):
        settings.pop(name)
    return {"config": settings, "data": data_identity}


def validate_resume_identity(saved: dict, current: dict, *, allow_batch_change: bool = False) -> dict | None:
    """Allow an explicit microbatch repartition, preserving examples per update.

    All other identity fields stay strict. Returns the change for provenance;
    matching effective batches preserve update groups, not bitwise numerics.
    """
    if saved == current:
        return None
    if allow_batch_change:
        previous, requested = dict(saved["config"]), dict(current["config"])
        old_batch = {name: previous.pop(name) for name in ("batch_size", "gradient_accumulation")}
        new_batch = {name: requested.pop(name) for name in ("batch_size", "gradient_accumulation")}
        valid = all(type(value) is int and value > 0 for value in (*old_batch.values(), *new_batch.values()))
        if (valid and {**saved, "config": previous} == {**current, "config": requested}
                and math.prod(old_batch.values()) == math.prod(new_batch.values())):
            return {"previous": old_batch, "current": new_batch,
                    "effective_batch": math.prod(new_batch.values()), "bitwise_equivalent": False}
    raise ValueError("Resume identity differs. --allow-batch-change permits only batch_size/"
                     "gradient_accumulation changes with the same product; all other identity fields must match.")


def train(
    model: RecurrentQwen, tokenizer: Any, train_items: list[EncodedExample], validation: list[EncodedExample],
    train_probe: list[EncodedExample], config: TrainingConfig, output: Path, spec: dict,
    identity: dict, *, resume: dict | None = None, allow_batch_change: bool = False,
    progress: Callable[[dict], None] = lambda event: None,
) -> dict:
    """Accumulate the configured objective; checkpoint only completed updates."""
    config.validate()
    token_ids, pad_id = spec["token_ids"], tokenizer.pad_token_id
    weights = (torch.tensor(loop_loss_weights([len(item.targets) for item in train_items]),
                            device=config.device, dtype=next(model.parameters()).dtype)
               if config.loss_reduction == "loop_mean" else None)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, foreach=False)
    step, epoch, offset, best_loss = 0, 0, 0, float("inf")
    if resume:
        validate_resume_identity(resume["identity"], identity, allow_batch_change=allow_batch_change)
        optimizer.load_state_dict(resume["optimizer"])
        step, epoch, offset, best_loss = (resume[key] for key in ("step", "next_epoch", "next_offset", "best_loss"))
        restore_rng(resume["rng"], config.device)
    effective_batch = config.batch_size * config.gradient_accumulation
    total_steps = math.ceil(len(train_items) / effective_batch) * config.epochs
    total_steps = min(total_steps, config.max_steps) if config.max_steps else total_steps
    if step >= total_steps:
        raise ValueError("Checkpoint already reached this budget; increase epochs/max_steps to continue")
    started, train_seconds, validation_seconds, completed_here, evaluations = perf_counter(), 0.0, 0.0, 0, 0
    last_saved = None
    latest_validation = None
    current_best_path = None
    history = (output / "metrics.jsonl").open("w")

    def log(event: dict) -> None:
        history.write(json.dumps({"elapsed_seconds": perf_counter() - started, **event}, allow_nan=False) + "\n")
        history.flush()

    def validate() -> bool:
        nonlocal best_loss, validation_seconds, evaluations, latest_validation
        begin = perf_counter()
        latest_validation = evaluate(
            model, validation, token_ids, pad_id, batch_size=config.batch_size,
            loops=config.validation_max_depth, output=output / f"validation-step-{step:06d}.csv",
            progress=lambda done, total: progress({"phase": f"Validation {done}/{total}"}),
        )
        probe = evaluate(
            model, train_probe, token_ids, pad_id, batch_size=config.batch_size,
            output=output / f"train-probe-step-{step:06d}.csv",
            progress=lambda done, total: progress({"phase": f"Train probe {done}/{total}"}),
        )
        selection_loss = training_selection_loss(latest_validation, config.train_max_depth, config.loss_reduction)
        latest_validation["selection_loss"] = selection_loss
        latest_validation["selection_reduction"] = config.loss_reduction
        improved = selection_loss < best_loss
        best_loss = min(best_loss, selection_loss)
        seconds = perf_counter() - begin
        validation_seconds += seconds
        evaluations += 1
        log({"event": "validation", "step": step, "validation": latest_validation, "train_probe": probe,
             "best_selection_loss": best_loss, "seconds": seconds, **memory_usage(config.device)})
        progress({"validation": latest_validation, "train_probe": probe})
        return improved

    def checkpoint(improved: bool) -> None:
        nonlocal last_saved, current_best_path
        path = output / f"step-{step:06d}"
        if last_saved != step:
            progress({"phase": f"Saving step {step}"})
            save_checkpoint(path, model, tokenizer, spec, {
                "step": step, "next_epoch": epoch, "next_offset": offset,
                "best_loss": best_loss, "identity": identity, "config": config.to_dict(),
                "optimizer": optimizer.state_dict(), "rng": rng_state(config.device),
            })
            last_saved = step
            (output / "last_checkpoint.json").write_text(json.dumps({"path": path.name, "step": step}) + "\n")
            log({"event": "checkpoint", "step": step, "path": str(path), "best": improved})
        if improved:
            current_best_path = str(path)
            (output / "best_checkpoint.json").write_text(json.dumps({"path": path.name, "step": step, "selection_loss": best_loss}) + "\n")

    try:
        log({"event": "objective", "loss_reduction": config.loss_reduction,
             "loop_weights": weights.tolist() if weights is not None else None,
             "training_examples": len(train_items)})
        progress({"step": step, "total_steps": total_steps, "phase": "Initial validation"})
        improved = validate()
        checkpoint(improved)
        while epoch < config.epochs and step < total_steps:
            order = list(range(len(train_items)))
            random.Random(config.seed + epoch).shuffle(order)
            while offset < len(order) and step < total_steps:
                group = [train_items[index] for index in order[offset:offset + effective_batch]]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                parts = []
                objective_sum = 0.0
                rate = config.learning_rate * min(1.0, (step + 1) / max(1, config.warmup_steps))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = rate
                synchronize(torch.device(config.device))
                begin = perf_counter()
                for start in range(0, len(group), config.batch_size):
                    items = group[start:start + config.batch_size]
                    batch = collate(items, pad_id, config.device)
                    result = model(batch["input_ids"], batch["attention_mask"], num_loops=batch["targets"].shape[1])
                    scores = symbolic_scores(result.loop_logits, token_ids)
                    loss, losses = step_loss(scores, batch["targets"], batch["target_mask"], loop_weights=weights)
                    (loss * len(items) / len(group)).backward()
                    objective_sum += loss.detach().item() * len(items)
                    parts.append(batch_metrics(scores, batch["targets"], batch["target_mask"], losses))
                    progress({"phase": f"Epoch {epoch + 1}/{config.epochs} · accumulation {min(start + len(items), len(group))}/{len(group)}",
                              "train": {**combine_metrics(parts), "objective_loss": objective_sum / (start + len(items))}, "learning_rate": rate})
                    del result, scores, loss, losses
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad and parameter.grad is not None:
                        raise RuntimeError(f"Frozen parameter received a gradient: {name}")
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True, foreach=False)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                synchronize(torch.device(config.device))
                seconds = perf_counter() - begin
                train_seconds += seconds
                completed_here += 1
                step += 1
                offset += len(group)
                epoch_finished = offset == len(order)
                if epoch_finished:
                    epoch += 1
                    offset = 0
                metrics = combine_metrics(parts)
                metrics["objective_loss"] = objective_sum / len(group)
                event = {"event": "train", "step": step, "next_epoch": epoch, "next_offset": offset,
                         "train": metrics, "learning_rate": rate, "gradient_norm_before_clip": norm.item(),
                         "update_seconds": seconds, "examples_per_second": len(group) / seconds,
                         "valid_transitions_per_second": sum(len(item.targets) for item in group) / seconds,
                         "eta_seconds": (total_steps - step) * (train_seconds / completed_here + validation_seconds / evaluations / config.eval_every),
                         **memory_usage(config.device)}
                log(event)
                progress(event)
                improved = False
                if step % config.eval_every == 0 or epoch_finished or step == total_steps:
                    improved = validate()
                if step % config.save_every == 0 or epoch_finished or step == total_steps or improved:
                    checkpoint(improved)
                if epoch_finished:
                    break
        result = {"status": "complete", "step": step, "next_epoch": epoch, "next_offset": offset,
                  "best_selection_loss": best_loss, "last_checkpoint": str(output / f"step-{last_saved:06d}"),
                  "best_checkpoint_in_this_run": current_best_path,
                  "validation": latest_validation, "training_seconds": train_seconds,
                  "validation_seconds": validation_seconds, "wall_seconds": perf_counter() - started,
                  **memory_usage(config.device)}
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        history.close()
