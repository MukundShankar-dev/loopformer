"""Opt-in profiling scopes and disposable optimizer-update measurements."""

from contextlib import contextmanager
from math import ceil
from time import perf_counter
from typing import Callable, Iterator

import torch

from scripts.eval.pointer_task import synchronize
from scripts.training.data import EncodedExample, collate
from scripts.training.objective import batch_metrics, step_loss, symbolic_scores


@contextmanager
def module_ranges(model: torch.nn.Module) -> Iterator[None]:
    """Annotate the existing forward without changing tensors or retaining states."""
    handles, active = [], []

    def before(name):
        def hook(module, args):
            scope = torch.profiler.record_function(name)
            scope.__enter__()
            active.append(scope)
        return hook

    def after(module, args, output):
        active.pop().__exit__(None, None, None)

    try:
        for name in ("prelude", "recurrent", "coda", "lm_head"):
            module = getattr(model, name)
            handles.append(module.register_forward_pre_hook(before("pointer/" + name)))
            handles.append(module.register_forward_hook(after, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        while active:
            active.pop().__exit__(None, None, None)


def measure(operation: Callable[[], object], device: torch.device) -> tuple[float, object]:
    """Synchronized wall time, including CPU work inside operation."""
    synchronize(device)
    began = perf_counter()
    result = operation()
    synchronize(device)
    return perf_counter() - began, result


def distribution(values: list[float]) -> dict:
    if not values:
        raise ValueError("Need timing samples")
    ordered = sorted(values)
    n = len(ordered)
    return {"samples_seconds": values, "median_seconds": (ordered[(n - 1) // 2] + ordered[n // 2]) / 2,
            "p95_seconds": ordered[ceil(.95 * n) - 1], "p95_method": "nearest rank"}


def memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"cuda_memory": None}
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_free_bytes": free, "device_total_bytes": total}


def training_update(model: torch.nn.Module, items: list[EncodedExample], token_ids: list[int],
                    pad_id: int, optimizer: torch.optim.Optimizer, batch_size: int,
                    *, diagnostics: bool = False) -> dict:
    """One disposable equal-example update; same sample group for every partition.

    Uses the production collation/objective and backward through all nominal loops.
    Never loads optimizer state or writes a checkpoint. Caller restores adapters
    and constructs a fresh optimizer before each measurement.
    """
    if not items or batch_size < 1 or len(items) % batch_size:
        raise ValueError("Microbatch must divide the effective update group")
    device = str(next(model.parameters()).device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    executed, supervised = 0, sum(len(x.targets) for x in items)
    parts = []
    for start in range(0, len(items), batch_size):
        group = items[start:start + batch_size]
        with torch.profiler.record_function("pointer/input_prepare_transfer"):
            batch = collate(group, pad_id, device)
        loops = batch["targets"].shape[1]
        executed += len(group) * loops
        with torch.profiler.record_function("pointer/forward"):
            output = model(batch["input_ids"], batch["attention_mask"], num_loops=loops)
        with torch.profiler.record_function("pointer/loss"):
            scores = symbolic_scores(output.loop_logits, token_ids)
            loss, losses = step_loss(scores, batch["targets"], batch["target_mask"])
        with torch.profiler.record_function("pointer/backward"):
            (loss * len(group) / len(items)).backward()
        with torch.profiler.record_function("pointer/metrics"):
            parts.append(batch_metrics(scores, batch["targets"], batch["target_mask"], losses))
        del output, scores, loss, losses
    named = {n: p for n, p in model.named_parameters() if p.requires_grad} if diagnostics else {}
    before = {n: p.detach().clone() for n, p in named.items()}
    gradient_norms = {n: p.grad.detach().norm().item() if p.grad is not None else None for n, p in named.items()}
    with torch.profiler.record_function("pointer/optimizer"):
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"executed_transitions": executed, "supervised_transitions": supervised,
            "gradient_norm_before_clip": norm.item(), "clipped": norm.item() > 1,
            "loss": sum(p["example_loss_sum"] for p in parts) / len(items),
            "adapter_diagnostics": {n: {"gradient_norm_before_clip": gradient_norms[n],
                                         "update_norm": (p.detach() - before[n]).norm().item()}
                                    for n, p in named.items()}}
