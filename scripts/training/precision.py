"""Explicit compute precision; parameters and optimizer state stay float32."""
from contextlib import nullcontext
from typing import TYPE_CHECKING, ContextManager

if TYPE_CHECKING:
    from scripts.recurrent_qwen.model import RecurrentQwen
    from .config import TrainingConfig
    from .data import EncodedExample
import torch


def autocast_context(device: str, precision: str) -> ContextManager:
    """BF16 autocast is CUDA-only; never silently choose a different dtype."""
    if precision == "float32":
        return nullcontext()
    if precision != "bf16" or torch.device(device).type != "cuda":
        raise ValueError("BF16 autocast requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def validate_compute_batch(model: "RecurrentQwen", items: list["EncodedExample"], pad_id: int,
                           token_ids: list[int], config: "TrainingConfig") -> dict:
    """Backward at maximum training depth without updating any parameter."""
    from time import perf_counter
    from .data import collate
    from .objective import symbolic_scores, step_loss, completion_loss
    deepest = max(len(x.targets) for x in items)
    selected = [x for x in items if len(x.targets) == deepest][:config.batch_size]
    if len(selected) != config.batch_size:
        raise ValueError("Need one full deepest-depth batch for the compute probe")
    batch = collate(selected, pad_id, config.device)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = perf_counter()
    with autocast_context(config.device, config.precision):
        result = model(batch['input_ids'], batch['attention_mask'], num_loops=deepest)
    scores = symbolic_scores(result.loop_logits, token_ids).float()
    loss, _ = step_loss(scores, batch['targets'], batch['target_mask'])
    if config.completion_loss_weight:
        halt, _ = completion_loss(result.stop_logits.float(), batch['target_mask'])
        loss = loss + config.completion_loss_weight * halt
    loss.backward()
    gradients = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received gradient: {name}")
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            gradients.append(parameter.grad.float().norm())
    norm = torch.stack(gradients).norm().item()
    if norm <= 0:
        raise RuntimeError("Compute probe produced no gradient")
    torch.cuda.synchronize()
    output = {'depth':deepest, 'batch_size':len(selected), 'loss':loss.item(), 'gradient_norm':norm,
              'forward_backward_seconds':perf_counter()-start, 'cuda_peak_gib':torch.cuda.max_memory_allocated()/2**30}
    model.zero_grad(set_to_none=True)
    return output
