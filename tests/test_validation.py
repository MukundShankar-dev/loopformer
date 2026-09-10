import torch

from scripts.recurrent_qwen.validation import validate_stage0


def test_gate_runner_on_tiny_qwen(base, inputs):
    report = validate_stage0(
        base, {"tiny": {"input_ids": inputs, "attention_mask": torch.ones_like(inputs)}},
        recurrent_start=1, recurrent_end=3, rank=2, alpha=4,
        loop_counts=(1, 3), progress=lambda message: None,
    )
    assert report["gate_passed"]
    assert report["parameters"]["trainable_tensors"] == 8
    assert report["gradient_check"]["optimizer_steps"] == 0
