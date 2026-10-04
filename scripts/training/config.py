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
    recurrence_mode: str = "full_sequence"
    train_scope: str = "lora"
    isolated_controller: bool = False
    reentry_bridge: bool = False
    state_loss_weight: float = 0.0
    prefix_reuse: bool = False
    gradient_checkpointing: bool = False
    lr_schedule: str = "constant"
    schedule_steps: int | None = None
    min_lr_ratio: float = 0.1
    lora_rank: int = 8
    lora_alpha: int = 16
    train_depths: list[int] | None = None
    precision: str = "float32"
    attention: str = "eager"
    validation_batch_size: int | None = None
    deterministic: bool = True
    bucket_by_depth: bool = False
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
    loss_reduction: str = "example_mean"
    completion_loss_weight: float = 0.0
    completion_head_hidden_size: int = 128

    def validate(self) -> None:
        if self.train_scope not in ("lora", "recurrent", "full"):
            raise ValueError("train_scope must be lora/recurrent/full")
        for name in ("isolated_controller", "reentry_bridge", "prefix_reuse", "gradient_checkpointing"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (self.train_scope != "lora" or self.isolated_controller or self.reentry_bridge or self.state_loss_weight or self.prefix_reuse or self.gradient_checkpointing) and self.recurrence_mode != "fixed_prompt":
            raise ValueError("Executor interfaces require fixed_prompt")
        if not math.isfinite(self.state_loss_weight) or self.state_loss_weight < 0:
            raise ValueError("state_loss_weight must be finite and nonnegative")
        if self.lr_schedule not in ("constant", "cosine") or not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid learning-rate schedule")
        if self.lr_schedule == "cosine" and (type(self.schedule_steps) is not int or self.schedule_steps <= self.warmup_steps):
            raise ValueError("Cosine requires explicit schedule_steps > warmup_steps (preserved on resume)")
        if self.precision not in ("float32", "bf16") or self.attention not in ("eager", "sdpa"):
            raise ValueError("precision must be float32/bf16 and attention eager/sdpa")
        if type(self.deterministic) is not bool or type(self.bucket_by_depth) is not bool:
            raise ValueError("deterministic must be boolean")
        if self.precision == "bf16" and self.device != "cuda":
            raise ValueError("BF16 training requires CUDA; no silent precision fallback")
        if self.validation_batch_size is not None and (type(self.validation_batch_size) is not int or self.validation_batch_size < 1):
            raise ValueError("validation_batch_size must be a positive integer")
        if self.train_depths is not None:
            if (not self.train_depths or any(type(d) is not int for d in self.train_depths)
                    or sorted(set(self.train_depths)) != self.train_depths
                    or min(self.train_depths) < 1 or max(self.train_depths) != self.train_max_depth):
                raise ValueError("train_depths must be sorted unique positive integers ending at train_max_depth")
        if self.recurrence_mode not in ("full_sequence", "fixed_prompt"):
            raise ValueError("recurrence_mode must be full_sequence or fixed_prompt")
        if self.loss_reduction not in ("example_mean", "loop_mean"):
            raise ValueError("loss_reduction must be example_mean or loop_mean")
        positive = ("threads", "lora_rank", "lora_alpha", "train_max_depth", "validation_max_depth",
                    "validation_per_depth", "train_probe_per_depth", "max_prompt_tokens", "epochs",
                    "batch_size", "gradient_accumulation", "eval_every", "save_every")
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.completion_head_hidden_size) is not int or self.completion_head_hidden_size < 1:
            raise ValueError("completion_head_hidden_size must be a positive integer")
        for name in ("train_limit", "max_steps"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")
        if not 1 <= self.train_max_depth <= self.validation_max_depth <= 256:
            raise ValueError("Require 1 <= train_max_depth <= validation_max_depth <= 2566")
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
        if not math.isfinite(self.completion_loss_weight) or self.completion_loss_weight < 0:
            raise ValueError("completion_loss_weight must be finite and nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)


def read_config(path: Path) -> TrainingConfig:
    config = TrainingConfig(**json.loads(path.read_text()))
    config.validate()
    return config
