"""Stage 0 checks shared by the pretrained CLI and a small integration test."""

from collections.abc import Callable
from time import perf_counter
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import Qwen2ForCausalLM

from .lora_utils import attach_recurrent_lora
from .model import RecurrentQwen


def validate_stage0(
    base: Qwen2ForCausalLM,
    cases: dict[str, dict[str, Tensor]],
    *,
    recurrent_start: int = 6,
    recurrent_end: int = 18,
    rank: int = 8,
    alpha: int = 16,
    loop_counts: tuple[int, ...] = (1, 2, 4),
    atol: float = 1e-5,
    rtol: float = 1e-5,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Assert architecture contracts; return serializable measured evidence.

    Mutates/freezes base and attaches fresh LoRA. No optimizer step is taken.
    The supplied cases must be short, nonempty input_ids/attention_mask pairs;
    equivalence compares every vocabulary logit at every sequence position.
    """
    if not cases or not loop_counts or min(loop_counts) < 1 or max(loop_counts) < 2:
        raise ValueError("Provide input cases and positive loop counts including a depth > 1")
    base.eval()
    base.config.use_cache = False
    originals = tuple(base.parameters())
    original_ids = {id(p) for p in originals}
    original_layers = tuple(base.model.layers)
    progress("Computing ordinary-Qwen reference logits")
    with torch.no_grad():
        references = {name: base(**inputs, use_cache=False).logits for name, inputs in cases.items()}
    model = RecurrentQwen(base, recurrent_start, recurrent_end).eval()
    assert tuple(model.recurrent.layers) == original_layers[recurrent_start:recurrent_end]
    errors = []
    for adapted in (False, True):
        if adapted:
            attach_recurrent_lora(model, rank=rank, alpha=alpha)
        progress(f"Checking T=1 equivalence ({'with' if adapted else 'without'} LoRA)")
        with torch.no_grad():
            for name, inputs in cases.items():
                actual = model(**inputs, logits_mode="all").logits
                expected = references[name]
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                difference = (actual - expected).abs()
                errors.append({
                    "case": name,
                    "lora": adapted,
                    "max_absolute_error": difference.max().item(),
                    "mean_absolute_error": difference.mean().item(),
                    "logits_shape": list(actual.shape),
                })
    del references

    progress("Checking shared weights, loop observability, and configurable depth")
    parameters_before = tuple(id(p) for p in model.parameters())
    calls: list[tuple[int, tuple[int, ...], Tensor, Tensor]] = []
    block_calls: list[str] = []
    layer_calls: list[int] = []
    handles = [model.recurrent.register_forward_hook(
        lambda module, args, output: calls.append(
            (id(module), tuple(id(p) for p in module.parameters()), args[0], output)
        )
    )]
    for name in ("prelude", "coda"):
        handles.append(getattr(model, name).register_forward_hook(
            lambda module, args, output, name=name: block_calls.append(name)
        ))
    for layer in model.recurrent.layers:
        handles.append(layer.register_forward_hook(
            lambda module, args, output: layer_calls.append(id(module))
        ))
    inputs = next(iter(cases.values()))
    batch, sequence = inputs["input_ids"].shape
    device = inputs["input_ids"].device
    allowed = inputs["input_ids"].unique()
    if allowed.numel() < 2:
        raise ValueError("Validation inputs must contain at least two distinct token IDs")
    recurrent_ids = tuple(id(p) for p in model.recurrent.parameters())
    trajectories = []
    try:
        with torch.no_grad():
            for depth in loop_counts:
                calls.clear()
                block_calls.clear()
                layer_calls.clear()
                labels = allowed[torch.arange(depth, device=device) % allowed.numel()].expand(batch, -1)
                start = perf_counter()
                output = model(**inputs, num_loops=depth, return_hidden_states=True,
                               labels=labels, allowed_token_ids=allowed)
                if device.type == "mps":
                    torch.mps.synchronize()
                seconds = perf_counter() - start
                assert len(calls) == depth
                assert block_calls.count("prelude") == 1 and block_calls.count("coda") == depth
                assert layer_calls == [id(layer) for layer in model.recurrent.layers] * depth
                assert all(call[0] == id(model.recurrent) and call[1] == recurrent_ids for call in calls)
                assert tuple(id(p) for p in model.parameters()) == parameters_before
                assert calls[0][2] is output.initial_hidden_state
                assert all(calls[t][2] is calls[t - 1][3] for t in range(1, depth))
                assert len(output.hidden_states) == len(output.loop_logits) == depth
                assert output.margins.shape == (batch, depth)
                for hidden, logits in zip(output.hidden_states, output.loop_logits):
                    assert hidden.shape == (batch, sequence, model.config.hidden_size)
                    assert logits.shape == (batch, model.config.vocab_size)
                    assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
                assert torch.isfinite(output.margins).all()
                trajectories.append({
                    "num_loops": depth,
                    "seconds": seconds,
                    "recurrent_calls": len(calls),
                    "recurrent_layer_calls": len(layer_calls),
                    "prelude_calls": block_calls.count("prelude"),
                    "coda_calls": block_calls.count("coda"),
                    "predicted_token_ids": [scores.argmax(-1).tolist() for scores in output.loop_logits],
                    "diagnostic_labels": labels.tolist(),
                    "allowed_token_ids": allowed.tolist(),
                    "margins": output.margins.tolist(),
                })
    finally:
        for handle in handles:
            handle.remove()
    calls.clear()
    del output

    progress("Checking two-loop backward through the frozen coda (no optimizer step)")
    model.train()
    model.zero_grad(set_to_none=True)
    start = perf_counter()
    output = model(**inputs, num_loops=2, return_hidden_states=True)
    for hidden in output.hidden_states:
        hidden.retain_grad()
    # Arbitrary token labels test graph connectivity, not task competence.
    loss = F.cross_entropy(output.logits.float(), inputs["input_ids"][:, 0])
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    backward_seconds = perf_counter() - start
    assert all(not p.requires_grad and p.grad is None for p in originals)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == (recurrent_end - recurrent_start) * 2 * 2
    assert {id(p) for p in model.parameters()} - original_ids == {id(p) for _, p in trainable}
    gradients = {}
    for name, parameter in trainable:
        assert name.startswith("recurrent.") and (".lora_A." in name or ".lora_B." in name)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        norm = parameter.grad.float().norm().item()
        if ".lora_B." in name:
            assert norm > 0, name
        gradients[name] = {"elements": parameter.numel(), "gradient_l2": norm}
    hidden_gradients = [hidden.grad.float().norm().item() for hidden in output.hidden_states]
    assert all(value > 0 and torch.isfinite(torch.tensor(value)) for value in hidden_gradients)
    result = {
        "gate_passed": True,
        "checks": {name: "PASS" for name in (
            "t1_equivalence", "shared_weights", "gradient_scope", "observability", "configurable_depth"
        )},
        "split": {"prelude": [0, recurrent_start], "recurrent": [recurrent_start, recurrent_end],
                  "coda": [recurrent_end, len(original_layers)], "end_indices_exclusive": True},
        "lora": {"rank": rank, "alpha": alpha, "targets": ["q_proj", "v_proj"], "dropout": 0.0},
        "bridge": None,
        "tolerance": {"atol": atol, "rtol": rtol},
        "equivalence": errors,
        "trajectories": trajectories,
        "parameters": {"original": sum(p.numel() for p in originals),
                       "trainable": sum(p.numel() for _, p in trainable),
                       "trainable_tensors": len(trainable)},
        "gradient_check": {"loss": loss.item(), "forward_backward_seconds": backward_seconds,
                           "hidden_gradient_l2": hidden_gradients, "parameters": gradients,
                           "frozen_parameters_with_grad": 0, "optimizer_steps": 0},
    }
    model.zero_grad(set_to_none=True)
    return result
