"""Pointer baseline data, prompting, generation, and exact final-answer scoring."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from string import Formatter
from time import perf_counter
from typing import Any, Iterator

import torch
from transformers import GenerationConfig

from scripts.dataset.pointer import PointerExample, SYMBOLS, validate_example


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(path: Path, limit: int | None = None) -> list[PointerExample]:
    """Validate every row in the selected file before evaluating its prefix."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    examples = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        record = json.loads(line)
        example = PointerExample(**{key: record[key] for key in PointerExample.__dataclass_fields__})
        validate_example(example)
        if example.example_id in seen:
            raise ValueError(f"Duplicate example ID at line {number}: {example.example_id}")
        seen.add(example.example_id)
        examples.append(example)
    if not examples:
        raise ValueError("Dataset is empty")
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        split = manifest["splits"].get(path.stem)
        if split is None or sha256_file(path) != split["sha256"] or len(examples) != split["count"]:
            raise ValueError("Dataset file does not match its manifest")
    return examples[:limit] if limit else examples


def validate_template(template: str) -> None:
    fields = set()
    for _, field, spec, conversion in Formatter().parse(template):
        if field is not None:
            if field not in {"rules", "start", "steps"} or spec or conversion:
                raise ValueError("Prompt placeholders must be plain {rules}, {start}, and {steps}")
            fields.add(field)
    if fields != {"rules", "start", "steps"}:
        raise ValueError("Prompt must include {rules}, {start}, and {steps}")


def render_task(template: str, example: PointerExample) -> str:
    """Interpolate only task inputs; never supply target fields to the template."""
    rules = " ".join(f"( {source}, {target})" for source, target in example.mapping)
    return template.format(rules=rules, start=example.initial_state, steps=example.task_depth).strip()


def score_response(response: str, target: str) -> dict:
    prediction = response.strip()
    valid = prediction in SYMBOLS
    return {"prediction": prediction if valid else "", "valid_answer": valid, "correct": valid and prediction == target}


def trim_generated(ids: list[int], eos_ids: set[int]) -> tuple[list[int], str]:
    """Count through the first EOS, excluding batch padding after it."""
    for index, token in enumerate(ids):
        if token in eos_ids:
            return ids[:index + 1], "eos"
    return ids, "max_new_tokens"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluate_batches(
    model: Any, tokenizer: Any, examples: list[PointerExample], template: str,
    *, batch_size: int, max_new_tokens: int, prompt_format: str,
) -> Iterator[tuple[list[dict], float]]:
    """Yield scored rows and synchronized generation seconds for each batch.

    Autoregressive tokens are ordinary model generation, not recurrent-depth
    transitions. Scoring is unconstrained exact symbol matching, not an A–Z logit
    restriction. The tokenizer belongs to the evaluated model, not the dataset.
    """
    validate_template(template)
    if batch_size < 1 or max_new_tokens < 1 or prompt_format not in ("chat", "raw"):
        raise ValueError("Positive batch/token budgets and chat/raw prompt format required")
    if prompt_format == "chat" and not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template; choose --prompt-format raw explicitly")
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    generation = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=False,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        prompts = [render_task(template, example) for example in batch]
        model_inputs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            ) if prompt_format == "chat" else prompt for prompt in prompts
        ]
        inputs = tokenizer(
            model_inputs, padding=True, add_special_tokens=False,
            return_token_type_ids=False, return_tensors="pt",
        )
        width = inputs["input_ids"].shape[1]
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit and width + max_new_tokens > context_limit:
            raise ValueError(f"Prompt plus output budget exceeds model context: {width}+{max_new_tokens}>{context_limit}; no truncation performed")
        inputs = inputs.to(model.device)
        synchronize(model.device)
        started = perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=generation)
        synchronize(model.device)
        elapsed = perf_counter() - started
        continuations = generated[:, width:].tolist()
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        rows = []
        for example, prompt, model_input, ids, prompt_length in zip(batch, prompts, model_inputs, continuations, prompt_lengths):
            ids, reason = trim_generated(ids, eos_ids)
            response = tokenizer.decode(ids, skip_special_tokens=True)
            rows.append({
                "example_id": example.example_id, "split": example.split, "seed": example.seed,
                "task_depth": example.task_depth, "target": example.final_state,
                "response": response, **score_response(response, example.final_state),
                "prompt_tokens": prompt_length, "generated_tokens": len(ids),
                "stop_reason": reason, "generated_token_ids": json.dumps(ids),
                "prompt": prompt, "model_input": model_input,
            })
        yield rows, elapsed


def summarize_results(rows: list[dict], generation_seconds: float, evaluation_seconds: float) -> dict:
    if not rows or min(generation_seconds, evaluation_seconds) <= 0:
        raise ValueError("Summary requires results and positive timing measurements")
    by_depth = defaultdict(list)
    for row in rows:
        by_depth[row["task_depth"]].append(row)
    correct = sum(row["correct"] for row in rows)
    tokens = sum(row["generated_tokens"] for row in rows)
    return {
        "total": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "invalid_answers": sum(not row["valid_answer"] for row in rows),
        "token_budget_stops": sum(row["stop_reason"] == "max_new_tokens" for row in rows),
        "generated_tokens": tokens, "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "generation_seconds": generation_seconds, "evaluation_seconds": evaluation_seconds,
        "generated_tokens_per_second": tokens / generation_seconds,
        "questions_per_second": len(rows) / evaluation_seconds,
        "by_depth": {
            str(depth): {"total": len(items), "correct": sum(row["correct"] for row in items),
                         "accuracy": sum(row["correct"] for row in items) / len(items)}
            for depth, items in sorted(by_depth.items())
        },
    }
