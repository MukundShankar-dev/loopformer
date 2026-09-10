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
    path.mkdir(parents=True, exist_ok=False)
    weights = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not weights or any(not name.startswith("recurrent.") or ".lora_" not in name for name in weights):
        raise ValueError("Checkpoint expects only recurrent LoRA to be trainable")
    torch.save(cpu_tree(weights), path / "adapter_model.pt")
    torch.save(cpu_tree(training_state), path / "training_state.pt")
    tokenizer.save_pretrained(path)
    (path / "recurrent_config.json").write_text(json.dumps({**spec, "format": FORMAT}, indent=2) + "\n")


def restore_adapters(model: RecurrentQwen, path: Path) -> None:
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
    if spec.get("format") != FORMAT or spec.get("prompt_format") != "dataset_raw" or spec.get("loss_vocabulary") != "symbols":
        raise ValueError("Unsupported recurrent checkpoint format/prompt/loss")
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
    model = RecurrentQwen(base, spec["recurrent_start"], spec["recurrent_end"])
    attach_recurrent_lora(model, rank=spec["lora_rank"], alpha=spec["lora_alpha"])
    restore_adapters(model, path)
    model.eval()
    return model, tokenizer, spec
