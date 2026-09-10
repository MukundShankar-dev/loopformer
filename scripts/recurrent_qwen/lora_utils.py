"""Attach PEFT LoRA only to the shared recurrent decoder group."""

from peft import LoraConfig, inject_adapter_in_model

from .model import RecurrentQwen


def attach_recurrent_lora(
    model: RecurrentQwen,
    *,
    rank: int = 8,
    alpha: int = 16,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
) -> None:
    """Mutate R in place; zero-initialized B preserves the original forward.

    No bias training, dropout, bridge, or adapters outside R. Original weights
    remain frozen. An initial zero gradient for A is expected when B is zero.
    """
    if rank < 1 or alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    valid = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= valid:
        raise ValueError(f"Choose unique LoRA targets from {sorted(valid)}")
    if getattr(model.recurrent, "peft_config", None):
        raise ValueError("Recurrent LoRA is already attached")
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(targets),
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )
    inject_adapter_in_model(config, model.recurrent)
