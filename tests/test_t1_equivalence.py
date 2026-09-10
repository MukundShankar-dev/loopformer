import pytest
import torch

from scripts.recurrent_qwen import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora


@pytest.mark.parametrize("split", [(1, 3), (0, 4), (0, 2), (2, 4)])
@pytest.mark.parametrize("padding", ["none", "left", "right"])
def test_full_logits_match_before_and_after_lora(base, inputs, split, padding):
    mask = torch.ones_like(inputs)
    if padding == "left":
        mask[0, :2] = 0
    elif padding == "right":
        mask[0, -2:] = 0
    # Explicit positions exercise the same interface as callers that offset
    # or reset positions for padded sequences.
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        expected = base(inputs, attention_mask=mask, position_ids=positions, use_cache=False).logits
        model = RecurrentQwen(base, *split)
        for adapted in (False, True):
            if adapted:
                attach_recurrent_lora(model, rank=2, alpha=4)
            actual = model(inputs, mask, position_ids=positions, logits_mode="all").logits
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert model.embed_tokens.weight is model.lm_head.weight


def test_default_positions_and_sdpa(base, inputs):
    base.set_attn_implementation("sdpa")
    with torch.no_grad():
        expected = base(inputs, use_cache=False).logits
        actual = RecurrentQwen(base, 1, 3)(inputs, logits_mode="all").logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_causality(base, inputs):
    model = RecurrentQwen(base, 1, 3)
    changed = inputs.clone()
    changed[:, -1] = 23
    with torch.no_grad():
        a = model(inputs, num_loops=3, logits_mode="all")
        b = model(changed, num_loops=3, logits_mode="all")
    for first, second in zip(a.loop_logits, b.loop_logits):
        torch.testing.assert_close(first[:, :-1], second[:, :-1])
