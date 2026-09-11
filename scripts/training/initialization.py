"""Validate adapter-only initialization separately from exact training resume."""

import json
from pathlib import Path

from scripts.recurrent_qwen.checkpoint import FORMAT


def validate_initialization(path: Path, expected: dict) -> dict:
    """Allow a new training depth/budget, never a different adapter interpretation.

    This metadata check loads no tensors. Actual adapter shape/finite-value checks
    remain in restore_adapters. A lower recorded training maximum would conceal
    the source adapter's exposure and is rejected.
    """
    source = json.loads((path / "recurrent_config.json").read_text())
    if source.get("format") != FORMAT:
        raise ValueError("Initialization requires a completed recurrent checkpoint")
    for key in ("base_model", "revision", "recurrent_start", "recurrent_end", "lora_rank", "lora_alpha",
                "symbols", "token_ids", "prompt_format", "loss_vocabulary"):
        if source.get(key) != expected[key]:
            raise ValueError(f"Initialization checkpoint differs in {key}")
    if not isinstance(source.get("train_max_depth"), int) or source["train_max_depth"] > expected["train_max_depth"]:
        raise ValueError("Cannot reduce the recorded training depth below the source checkpoint's exposure")
    return source
