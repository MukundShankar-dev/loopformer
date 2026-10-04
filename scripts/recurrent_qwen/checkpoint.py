"""Portable recurrent adapters plus explicit base-model and tokenizer identity."""

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from .lora_utils import attach_recurrent_lora
from .model import RecurrentQwen
from scripts.dataset.pointer import SYMBOLS


FORMAT = "loopformer-stage1-v1"
COMPLETION_FORMAT = "loopformer-stage1-completion-v1"
FIXED_PROMPT_FORMAT = "loopformer-stage1-fixed-prompt-v1"


def checkpoint_mode(spec: dict) -> str:
    """Legacy checkpoints mean full-sequence recurrence; never infer new modes."""
    mode = spec.get("recurrence_mode", "full_sequence")
    if spec.get("format") not in (FORMAT, COMPLETION_FORMAT, FIXED_PROMPT_FORMAT):
        raise ValueError("Unsupported recurrent checkpoint format")
    expected = "fixed_prompt" if spec["format"] == FIXED_PROMPT_FORMAT else "full_sequence"
    if mode != expected:
        raise ValueError("Checkpoint format and recurrence_mode disagree")
    return mode


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path: Path, model: RecurrentQwen, tokenizer: Any, spec: dict, training_state: dict) -> None:
    """Write a new checkpoint directory; metadata is the final completion marker."""
    if spec.get("recurrence_mode", "full_sequence") != model.recurrence_mode:
        raise ValueError("Checkpoint spec and recurrence_mode disagree")
    weights = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    completion = model.completion_head is not None
    expected_completion = spec.get("completion_head")
    if completion != (expected_completion is not None):
        raise ValueError("Checkpoint spec and completion head disagree")
    if completion and expected_completion != {"intermediate": model.completion_head.intermediate}:
        raise ValueError("Checkpoint completion-head architecture differs from spec")
    if not weights or any(not ((name.startswith("recurrent.") and ".lora_" in name) or
                               (completion and name.startswith("completion_head."))) for name in weights):
        raise ValueError("Checkpoint expects recurrent LoRA and optional completion head only")
    path.mkdir(parents=True, exist_ok=False)
    torch.save(cpu_tree(weights), path / "adapter_model.pt")
    torch.save(cpu_tree(training_state), path / "training_state.pt")
    tokenizer.save_pretrained(path)
    version = FIXED_PROMPT_FORMAT if model.recurrence_mode == "fixed_prompt" else (COMPLETION_FORMAT if completion else FORMAT)
    (path / "recurrent_config.json").write_text(json.dumps({**spec, "format": version}, indent=2) + "\n")


def restore_adapters(model: RecurrentQwen, path: Path) -> None:
    if checkpoint_mode(json.loads((path / "recurrent_config.json").read_text())) != model.recurrence_mode:
        raise ValueError("Adapter checkpoint recurrence_mode differs from model")
    state = torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True)
    expected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(state) != set(expected):
        raise ValueError("Adapter checkpoint names do not match the recurrent model")
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = state[name]
            if tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid adapter tensor: {name}")
            parameter.copy_(tensor)


def load_recurrent_checkpoint(path: Path, *, device: str = "cpu", dtype: str = "float32",
                              download: bool = False) -> tuple[RecurrentQwen, Any, dict]:
    spec = json.loads((path / "recurrent_config.json").read_text())
    mode = checkpoint_mode(spec)
    if spec.get("prompt_format") != "dataset_raw" or spec.get("loss_vocabulary") != "symbols":
        raise ValueError("Unsupported recurrent checkpoint format/prompt/loss")
    completion = spec["format"] == COMPLETION_FORMAT or (mode == "fixed_prompt" and "completion_head" in spec)
    if completion and (not isinstance(spec.get("completion_head"), dict) or
                       type(spec["completion_head"].get("intermediate")) is not int or
                       spec["completion_head"]["intermediate"] < 1):
        raise ValueError("Invalid saved completion-head architecture")
    if not completion and "completion_head" in spec:
        raise ValueError("Legacy checkpoint cannot contain a completion head")
    if spec.get("symbols") != list(SYMBOLS) or len(set(spec.get("token_ids", []))) != len(SYMBOLS):
        raise ValueError("Checkpoint requires the complete A–Z symbol vocabulary")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    for symbol, token in zip(spec["symbols"], spec["token_ids"], strict=True):
        if tokenizer.encode(" " + symbol, add_special_tokens=False) != [token]:
            raise ValueError("Checkpoint tokenizer and symbol IDs disagree")
    base = Qwen2ForCausalLM.from_pretrained(
        spec["base_model"], revision=spec["revision"], local_files_only=not download,
        dtype=getattr(torch, dtype), attn_implementation="eager", trust_remote_code=False,
    ).to(device).eval()
    model = RecurrentQwen(base, spec["recurrent_start"], spec["recurrent_end"], recurrence_mode=mode)
    attach_recurrent_lora(model, rank=spec["lora_rank"], alpha=spec["lora_alpha"])
    if completion:
        model.enable_completion(spec["completion_head"]["intermediate"])
    restore_adapters(model, path)
    model.eval()
    return model, tokenizer, spec
