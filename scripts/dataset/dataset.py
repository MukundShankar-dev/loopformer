"""Seeded splits, JSONL persistence, and independent pointer-data validation."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.dataset.pointer import PointerExample, example_seed, generate_example, validate_example
from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION, validate_prompt_tokens, validate_symbols


SPLITS = ("train", "validation", "test", "depth_test")


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 17
    train_count: int = 10_000
    validation_count: int = 1_000
    test_count: int = 1_000
    depth_test_count: int = 1_000
    min_depth: int = 1
    max_train_depth: int = 8
    max_eval_depth: int = 16

    def validate(self) -> None:
        if any(type(value) is not int for value in asdict(self).values()):
            raise ValueError("Dataset configuration values must be integers")
        if self.seed < 0 or min(self.count(split) for split in SPLITS) < 1:
            raise ValueError("seed must be nonnegative and all split counts must be positive")
        if not 1 <= self.min_depth <= self.max_train_depth < self.max_eval_depth <= 25:
            raise ValueError("Require 1 <= min_depth <= max_train_depth < max_eval_depth <= 25")

    def count(self, split: str) -> int:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return getattr(self, f"{split}_count")

    def depths(self, split: str) -> range:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if split == "depth_test":
            return range(self.max_train_depth + 1, self.max_eval_depth + 1)
        return range(self.min_depth, self.max_train_depth + 1)


def preview_indices(config: DatasetConfig, count: int) -> list[tuple[str, int]]:
    """Select actual records spanning available depths, without sampling programs.

    This is a depth showcase, not a random sample of split frequencies. Keep only
    enough candidate indices to fill a preview, even for very large split counts.
    """
    config.validate()
    if count not in (5, 10):
        raise ValueError("Preview count must be 5 or 10")
    candidates = [
        (split, index, config.depths(split)[index % len(config.depths(split))])
        for split in SPLITS
        for index in range(min(config.count(split), count * len(config.depths(split))))
    ]
    depths = sorted({depth for _, _, depth in candidates})
    count = min(count, len(candidates))
    selected = []
    for position in range(count):
        target = depths[position * (len(depths) - 1) // (count - 1)]
        candidate = min(candidates, key=lambda item: (
            abs(item[2] - target), (SPLITS.index(item[0]) - position) % len(SPLITS), item[1],
        ))
        candidates.remove(candidate)
        selected.append(candidate[:2])
    return selected


def make_record(config: DatasetConfig, split: str, index: int, tokenizer: Any, token_ids: dict[str, int]) -> dict:
    """Index determines balanced depth and an independently derived random seed."""
    depths = config.depths(split)
    depth = depths[index % len(depths)]
    example = generate_example(example_seed(config.seed, split, index), depth, split, index)
    validate_example(example)
    position = validate_prompt_tokens(tokenizer, example.prompt, token_ids)
    return {
        **example.to_dict(),
        "answer_position": position,
        "prompt_token_count": position + 1,
        "intermediate_token_ids": [token_ids[symbol] for symbol in example.intermediate_states],
        "final_token_id": token_ids[example.final_state],
    }


def generate_dataset(config: DatasetConfig, tokenizer: Any, token_ids: dict[str, int]) -> dict[str, list[dict]]:
    config.validate()
    dataset = {}
    seen = set()
    for split in SPLITS:
        records = []
        for index in range(config.count(split)):
            record = make_record(config, split, index, tokenizer, token_ids)
            fingerprint = record["mapping_sha256"]
            if fingerprint in seen:
                raise ValueError("Repeated rule table within/across splits; select a different master seed")
            seen.add(fingerprint)
            records.append(record)
        dataset[split] = records
    return dataset


def encode_records(records: list[dict]) -> bytes:
    return "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records).encode()


def summarize(records: list[dict]) -> dict:
    return {
        "count": len(records),
        "depth_counts": dict(sorted(Counter(str(record["task_depth"]) for record in records).items())),
        "prompt_tokens_min": min(record["prompt_token_count"] for record in records),
        "prompt_tokens_max": max(record["prompt_token_count"] for record in records),
        "sha256": hashlib.sha256(encode_records(records)).hexdigest(),
    }


def write_dataset(
    output: Path, dataset: dict[str, list[dict]], config: DatasetConfig,
    token_ids: dict[str, int], provenance: dict,
) -> dict:
    """Write a new directory only; the manifest is the final completion marker."""
    manifest = {
        "schema_version": 1,
        "generator": "pointer-v1",
        "config": asdict(config),
        "tokenizer": {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "add_special_tokens": False, "chat_template": False,
            "answer_text_prefix": " ", "symbol_token_ids": token_ids,
        },
        "semantics": "explicit_steps_unique_nominal_path_no_post_completion_targets",
        "splits": {split: summarize(dataset[split]) for split in SPLITS},
        "validation": {"unique_mapping_tables": True, "exact_targets": True, "prompt_symbol_tokens": True},
        "provenance": provenance,
    }
    output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        (output / f"{split}.jsonl").write_bytes(encode_records(dataset[split]))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_dataset(output: Path, tokenizer: Any) -> dict:
    """Check bytes, labels, split boundaries, tokenizer contexts, and seed replay.

    Regeneration alone is insufficient: validate_example separately reparses the
    rendered rules and executes them to check the actual stored trajectory.
    """
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["generator"] != "pointer-v1":
        raise ValueError("Unsupported dataset version")
    config = DatasetConfig(**manifest["config"])
    config.validate()
    token_ids = validate_symbols(tokenizer)
    if manifest["tokenizer"] != {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "add_special_tokens": False, "chat_template": False,
        "answer_text_prefix": " ", "symbol_token_ids": token_ids,
    }:
        raise ValueError("Tokenizer metadata mismatch")
    seen = set()
    for split in SPLITS:
        payload = (output / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split}: file checksum mismatch")
        records = [json.loads(line) for line in payload.splitlines()]
        if len(records) != config.count(split) or summarize(records) != manifest["splits"][split]:
            raise ValueError(f"{split}: split count/summary mismatch")
        for index, record in enumerate(records):
            example = PointerExample(**{key: record[key] for key in PointerExample.__dataclass_fields__})
            validate_example(example)
            if record != make_record(config, split, index, tokenizer, token_ids):
                raise ValueError(f"{split} record {index}: seed replay/token metadata mismatch")
            if example.mapping_sha256 in seen:
                raise ValueError("Repeated rule table within/across splits")
            seen.add(example.mapping_sha256)
    return manifest
