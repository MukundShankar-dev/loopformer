"""Startup equivalence must compare matching readouts without hiding errors."""

from dataclasses import replace

import pytest
import torch

from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.training.config import TrainingConfig
from scripts.training.gates import initialize


def gate_inputs():
    # Right padding makes the answer position differ from the last column.
    return {
        "input_ids": torch.tensor([[3, 5, 7, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "targets": torch.tensor([[0]]),
    }


def gate_config():
    return replace(TrainingConfig(), recurrent_start=1, recurrent_end=3,
                   lora_rank=2, lora_alpha=4)


def test_gate_matches_reference_projection_at_unpadded_answer(base):
    projections = []
    hook = base.lm_head.register_forward_pre_hook(
        lambda module, args: projections.append(args[0].detach().clone())
    )
    try:
        _, gate = initialize(base, gate_config(), gate_inputs(), [3, 5, 7])
    finally:
        hook.remove()
    reference, actual = projections[:2]
    assert reference.shape == (1, 1, base.config.hidden_size)
    assert actual.shape == (1, base.config.hidden_size)
    torch.testing.assert_close(reference[:, 0], actual, atol=1e-5, rtol=1e-5)
    assert gate["passed"]


def test_gate_still_rejects_incorrect_recurrent_logits(base, monkeypatch):
    original = RecurrentQwen.forward

    def corrupted(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        output.logits[0, 0] += 0.01
        return output

    monkeypatch.setattr(RecurrentQwen, "forward", corrupted)
    with pytest.raises(AssertionError, match="not close"):
        initialize(base, gate_config(), gate_inputs(), [3, 5, 7])
