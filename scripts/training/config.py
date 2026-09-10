"""Explicit, JSON-backed configuration for the first pointer experiment."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION


@dataclass(frozen=True)
class TrainingConfig:
    model: str = MODEL_ID
    revision: str = MODEL_REVISION
    train_data: str = "data/pointer/seed-17/train.jsonl"
    validation_data: str = "data/pointer/seed-17/validation.jsonl"
    device: str = "cpu"
    seed: int = 17
    threads: int = 4
    recurrent_start: int = 6
    recurrent_end: int = 18
    lora_rank: int = 8
    lora_alpha: int = 16
    train_max_depth: int = 4
    validation_max_depth: int = 8
    train_limit: int | None = None
    validation_per_depth: int = 8
    train_probe_per_depth: int = 4
    max_prompt_tokens: int = 256
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 0.0002
    warmup_steps: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    eval_every: int = 100
    save_every: int = 100
    loss_vocabulary: str = "symbols"

    def validate(self) -> None:
        positive = ("threads", "lora_rank", "lora_alpha", "train_max_depth", "validation_max_depth",
                    "validation_per_depth", "train_probe_per_depth", "max_prompt_tokens", "epochs",
                    "batch_size", "gradient_accumulation", "eval_every", "save_every")
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("train_limit", "max_steps"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")
        if not 1 <= self.train_max_depth <= self.validation_max_depth <= 25:
            raise ValueError("Require 1 <= train_max_depth <= validation_max_depth <= 25")
        if not 0 <= self.recurrent_start < self.recurrent_end:
            raise ValueError("Invalid recurrent layer range")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 or type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("Invalid seed or warmup_steps")
        if self.device not in ("cpu", "mps", "cuda") or self.loss_vocabulary != "symbols":
            raise ValueError("Use cpu/mps/cuda and loss_vocabulary='symbols'")
        for name in ("learning_rate", "weight_decay", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {name}")

    def to_dict(self) -> dict:
        return asdict(self)


def read_config(path: Path) -> TrainingConfig:
    config = TrainingConfig(**json.loads(path.read_text()))
    config.validate()
    return config
