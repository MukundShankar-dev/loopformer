"""Startup architecture checks on the actual model and selected training input."""

import torch
from transformers import Qwen2ForCausalLM

from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
from .config import TrainingConfig
from .objective import symbolic_scores
from .objective import completion_loss


def initialize(base: Qwen2ForCausalLM, config: TrainingConfig, inputs: dict, token_ids: list[int]) -> tuple[RecurrentQwen, dict]:
    """Gate fresh zero-B adapters before training, including the actual loss path."""
    base.eval()
    base.config.use_cache = False
    x = inputs["input_ids"][:1]
    mask = inputs["attention_mask"][:1]
    position = int(torch.arange(mask.shape[1], device=mask.device).masked_fill(~mask[0].bool(), -1).max().item())
    with torch.no_grad():
        # Match the wrapper's one-position LM-head projection. Projecting the
        # entire prompt first can introduce shape-dependent rounding on CUDA.
        readout = torch.tensor([position], device=x.device)
        reference = base(input_ids=x, attention_mask=mask, use_cache=False,
                         logits_to_keep=readout).logits[:, 0].clone()
    model = RecurrentQwen(base, config.recurrent_start, config.recurrent_end,
                          recurrence_mode=config.recurrence_mode).eval()
    attach_recurrent_lora(model, rank=config.lora_rank, alpha=config.lora_alpha)
    with torch.no_grad():
        actual = model(x, mask, num_loops=1).logits
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
        error = (actual - reference).abs().max().item()
    del actual, reference
    calls = []
    hook = model.recurrent.register_forward_hook(lambda module, args, output: calls.append((id(module), tuple(id(p) for p in module.parameters()))))
    try:
        result = model(x, mask, num_loops=2, return_hidden_states=True)
    finally:
        hook.remove()
    if len(calls) != 2 or calls[0] != calls[1]:
        raise RuntimeError("Recurrent weights are not shared")
    if config.recurrence_mode == "fixed_prompt":
        prefix = torch.arange(mask.shape[1], device=mask.device)[None, :] < position
        torch.testing.assert_close(result.hidden_states[0][prefix], result.hidden_states[1][prefix], atol=0, rtol=0)
    for hidden in result.hidden_states:
        hidden.retain_grad()
    scores = symbolic_scores(result.loop_logits, token_ids)
    # The last-loop loss alone must propagate through both recurrent states.
    torch.nn.functional.cross_entropy(scores[:, -1], inputs["targets"][:1, 0]).backward()
    hidden_norms = [(hidden.grad[:, position].norm() if config.recurrence_mode == "fixed_prompt" else hidden.grad.norm()).item()
                    if hidden.grad is not None else 0 for hidden in result.hidden_states]
    if not all(value > 0 for value in hidden_norms):
        raise RuntimeError("Later-loop gradient does not reach earlier states")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if ".lora_" not in name or not name.startswith("recurrent.") or parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Invalid trainable parameter/gradient: {name}")
        elif parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received a gradient: {name}")
    model.zero_grad(set_to_none=True)
    return model, {"passed": True, "t1_max_logit_error": error, "shared_weights": True,
                   "recurrence_mode": config.recurrence_mode,
                   "frozen_gradients": 0, "hidden_gradient_norms": hidden_norms,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def validate_completion_gradients(model: RecurrentQwen, inputs: dict) -> dict:
    """Check the stop-only path reaches the shared recurrence and head."""
    if model.completion_head is None:
        raise ValueError("Attach the completion head before checking its gradients")
    model.zero_grad(set_to_none=True)
    result = model(inputs["input_ids"][:1], inputs["attention_mask"][:1],
                   num_loops=2, return_hidden_states=True)
    for hidden in result.hidden_states:
        hidden.retain_grad()
    mask = torch.tensor([[True, True]], device=result.stop_logits.device)
    loss, _ = completion_loss(result.stop_logits, mask)
    loss.backward()
    mask_input = inputs["attention_mask"][0]
    position = int(torch.arange(len(mask_input), device=mask_input.device).masked_fill(~mask_input.bool(), -1).max().item())
    hidden_norms = [(hidden.grad[:, position].norm() if model.recurrence_mode == "fixed_prompt" else hidden.grad.norm()).item()
                    if hidden.grad is not None else 0.0
                    for hidden in result.hidden_states]
    if not all(value > 0 for value in hidden_norms):
        raise RuntimeError("Completion loss does not reach every recurrent state")
    head_grads = 0
    adapter_grads = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Invalid completion-path gradient: {name}")
            if name.startswith("completion_head."):
                head_grads += int(parameter.grad.abs().sum() > 0)
            elif name.startswith("recurrent.") and ".lora_" in name:
                adapter_grads += int(parameter.grad.abs().sum() > 0)
            else:
                raise RuntimeError(f"Unexpected trainable parameter: {name}")
        elif parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received a completion gradient: {name}")
    if not head_grads or not adapter_grads:
        raise RuntimeError("Completion gradient must update the head and recurrent adapters")
    model.zero_grad(set_to_none=True)
    return {"passed": True, "head_tensors_with_gradient": head_grads,
            "adapter_tensors_with_gradient": adapter_grads,
            "hidden_gradient_norms": hidden_norms}
