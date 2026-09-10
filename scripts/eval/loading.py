"""Explicit ordinary causal-LM loading; no recurrent wrapper or implicit downloads."""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION


def load_state_file(model: Any, path: Path) -> None:
    """Load tensor weights only, allowing omitted aliases of tied parameters.

    Full pickled model objects and partial adapter checkpoints are not accepted.
    A tied parameter may be stored under one alias, but conflicting aliases fail.
    """
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    elif path.suffix in (".pt", ".pth", ".bin"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        raise ValueError("Weight files must be .safetensors, .pt, .pth, or .bin")
    if not isinstance(state, dict) or not state or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("Expected a tensor state dict or {'state_dict': tensor_dict}")
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    covered = set()
    for names in aliases.values():
        supplied = [name for name in names if name in state]
        if supplied:
            covered.update(names)
            if any(not torch.equal(state[supplied[0]], state[name]) for name in supplied[1:]):
                raise ValueError(f"Conflicting weights for tied parameters: {supplied}")
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys) - covered
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint does not match the architecture: missing={sorted(missing)}, unexpected={incompatible.unexpected_keys}")


def load_model(
    source: str, *, base_model: str | None, tokenizer_source: str | None,
    revision: str | None, device: str, dtype: str, download: bool,
) -> tuple[Any, Any, dict]:
    """Load a Hub ID, save_pretrained directory, or standalone full state dict."""
    path = Path(source).expanduser()
    if (source.startswith(("/", ".", "~")) or path.suffix in (".pt", ".pth", ".bin", ".safetensors")) and not path.exists():
        raise ValueError(f"Local checkpoint does not exist: {path}")
    if path.is_dir() and (path / "adapter_config.json").exists():
        raise ValueError("An adapter-only directory is not a full ordinary model; merge/export it with its base model first")
    source = str(path.resolve()) if path.exists() else source
    if base_model and not path.is_file():
        raise ValueError("--base-model is only used with a standalone weight file")
    architecture = base_model or (str(path.parent.resolve()) if path.is_file() else source)
    if path.is_file() and not base_model and not (path.parent / "config.json").exists():
        raise ValueError("A standalone weight file needs --base-model MODEL_OR_CONFIG_DIR, or config.json beside it")
    resolved_revision = revision or (MODEL_REVISION if architecture == MODEL_ID else "main")
    options = {"revision": resolved_revision, "local_files_only": not download, "trust_remote_code": False}
    token_source = tokenizer_source or architecture
    token_options = options if token_source == architecture else {"local_files_only": not download, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(token_source, **token_options)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define a pad or EOS token for batching")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    torch_dtype = getattr(torch, dtype)
    if path.is_file():
        config = AutoConfig.from_pretrained(architecture, **options)
        model = AutoModelForCausalLM.from_config(config, dtype=torch_dtype, attn_implementation="eager")
        load_state_file(model, path)
    else:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            source, **options, dtype=torch_dtype, attn_implementation="eager", weights_only=True,
            output_loading_info=True,
        )
        problems = {key: loading_info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading_info.get(key)}
        if problems:
            raise ValueError(f"Checkpoint did not load completely: {problems}")
    model.to(device).eval()
    model.config.use_cache = False
    return model, tokenizer, {
        "source": source, "base_model": architecture,
        "requested_revision": resolved_revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_source": token_source,
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "pad_token_id": tokenizer.pad_token_id,
    }
