"""Validated raw task inputs and balanced, deterministic training subsets."""

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import torch

from scripts.dataset.pointer import PointerExample, SYMBOLS, validate_example
from scripts.dataset.symbols import validate_prompt_tokens
from scripts.eval.pointer_task import sha256_file


@dataclass
class EncodedExample:
    task: PointerExample
    input_ids: list[int]
    targets: list[int]  # Class indices in SYMBOLS, not tokenizer IDs; loop 1 is index 0.


def read_tasks(path: Path, expected_split: str) -> list[PointerExample]:
    manifest = json.loads((path.parent / "manifest.json").read_text())
    info = manifest["splits"][expected_split]
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"{path}: dataset checksum mismatch")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != info["count"]:
        raise ValueError(f"{path}: dataset count mismatch")
    tasks = []
    for record in records:
        task = PointerExample(**{key: record[key] for key in PointerExample.__dataclass_fields__})
        validate_example(task)
        if task.split != expected_split:
            raise ValueError(f"Expected {expected_split} records in {path}")
        tasks.append(task)
    if len({task.mapping_sha256 for task in tasks}) != len(tasks):
        raise ValueError("Repeated rule tables in dataset")
    return tasks


def select_tasks(tasks: list[PointerExample], max_depth: int, seed: int,
                 *, limit: int | None = None, per_depth: int | None = None) -> list[PointerExample]:
    """Shuffle within depth, then interleave depths so small prefixes are balanced."""
    groups = []
    rng = random.Random(seed)
    for depth in range(1, max_depth + 1):
        group = [task for task in tasks if task.task_depth == depth]
        if not group:
            raise ValueError(f"No examples at required depth {depth}")
        rng.shuffle(group)
        groups.append(group[:per_depth] if per_depth else group)
    selected = [group[index] for index in range(max(map(len, groups))) for group in groups if index < len(group)]
    return selected[:limit] if limit else selected


def encode_tasks(tasks: list[PointerExample], tokenizer: Any, token_ids: dict[str, int], max_tokens: int) -> list[EncodedExample]:
    result = []
    for task in tasks:
        validate_prompt_tokens(tokenizer, task.prompt, token_ids)
        ids = tokenizer.encode(task.prompt, add_special_tokens=False)
        if len(ids) > max_tokens:
            raise ValueError(f"{task.example_id}: {len(ids)} prompt tokens exceeds {max_tokens}; no truncation")
        result.append(EncodedExample(task, ids, [SYMBOLS.index(symbol) for symbol in task.intermediate_states]))
    return result


def collate(examples: list[EncodedExample], pad_id: int, device: str, loops: int | None = None) -> dict:
    """Right-pad inputs; labels/mask are [B,T], with no post-completion targets."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(item.input_ids) for item in examples)
    depth = max(len(item.targets) for item in examples)
    loops = depth if loops is None else loops
    if type(loops) is not int or loops < depth:
        raise ValueError("Loop budget must cover every nominal target")
    inputs = torch.full((len(examples), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(inputs)
    labels = torch.zeros((len(examples), loops), dtype=torch.long)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for row, item in enumerate(examples):
        inputs[row, :len(item.input_ids)] = torch.tensor(item.input_ids)
        attention[row, :len(item.input_ids)] = 1
        labels[row, :len(item.targets)] = torch.tensor(item.targets)
        mask[row, :len(item.targets)] = True
    return {key: tensor.to(device) for key, tensor in {
        "input_ids": inputs, "attention_mask": attention, "targets": labels, "target_mask": mask,
    }.items()}
