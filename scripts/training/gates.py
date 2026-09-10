"""Startup architecture checks on the actual model and selected training input."""

import torch
from transformers import Qwen2ForCausalLM

from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
from .config import TrainingConfig
from .objective import symbolic_scores


def initialize(base: Qwen2ForCausalLM, config: TrainingConfig, inputs: dict, token_ids: list[int]) -> tuple[RecurrentQwen, dict]:
    """Gate fresh zero-B adapters before training, including the actual loss path."""
    base.eval()
    base.config.use_cache = False
    x = inputs["input_ids"][:1]
    mask = inputs["attention_mask"][:1]
    position = int(mask.sum().item()) - 1
    with torch.no_grad():
        reference = base(input_ids=x, attention_mask=mask, use_cache=False).logits[:, position].clone()
    model = RecurrentQwen(base, config.recurrent_start, config.recurrent_end).eval()
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
    for hidden in result.hidden_states:
        hidden.retain_grad()
    scores = symbolic_scores(result.loop_logits, token_ids)
    # The last-loop loss alone must propagate through both recurrent states.
    torch.nn.functional.cross_entropy(scores[:, -1], inputs["targets"][:1, 0]).backward()
    hidden_norms = [hidden.grad.norm().item() if hidden.grad is not None else 0 for hidden in result.hidden_states]
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
                   "frozen_gradients": 0, "hidden_gradient_norms": hidden_norms,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
