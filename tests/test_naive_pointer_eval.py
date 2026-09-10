import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from scripts.dataset.pointer import generate_example
from scripts.eval.loading import load_model, load_state_file
from scripts.eval.pointer_task import (
    evaluate_batches, load_examples, render_task, score_response, summarize_results,
    trim_generated, validate_template,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "prompts/pointer_task.txt").read_text()


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """Local random model/tokenizer; never downloads or loads pretrained weights."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocabulary = {word: index for index, word in enumerate(["[PAD]", "[EOS]", "[UNK]", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "Rules", "Start", "Steps", "Answer", ":", "(", ",", ")", "1", "2"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}\nAnswer:",
    )
    config = Qwen2Config(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=1024, tie_word_embeddings=True,
        pad_token_id=0, eos_token_id=1,
    )
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(config).eval()
    directory = tmp_path / "model"
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory, model, tokenizer


@pytest.mark.parametrize("response,valid,correct", [
    (" C\n", True, True), ("D", True, False), ("c", False, False),
    ("Answer: C", False, False), ("C.", False, False), ("", False, False),
    ("A C", False, False),
])
def test_strict_final_answer_scoring(response, valid, correct):
    assert score_response(response, "C")["valid_answer"] is valid
    assert score_response(response, "C")["correct"] is correct


def test_prompt_inputs_only_and_corrupt_data_rejected(tmp_path):
    validate_template(TEMPLATE)
    example = generate_example(17, 2, "test", 0)
    prompt = render_task(TEMPLATE, example)
    assert f"Start: {example.initial_state}" in prompt
    assert "Steps: 2" in prompt
    for invalid in ("{rules} {start}", "{rules} {start} {steps} {final_state}"):
        with pytest.raises(ValueError, match="Prompt"):
            validate_template(invalid)
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(example.to_dict()) + "\n")
    assert load_examples(path) == [example]
    record = example.to_dict()
    record["final_state"] = example.initial_state
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="labels"):
        load_examples(path)


def test_generated_token_count_and_accuracy_denominators():
    assert trim_generated([4, 1, 0, 0], {1}) == ([4, 1], "eos")
    assert trim_generated([4, 5], {1}) == ([4, 5], "max_new_tokens")
    rows = [
        {"task_depth": depth, "correct": correct, "valid_answer": valid,
         "generated_tokens": 2, "prompt_tokens": 10, "stop_reason": "eos"}
        for depth, correct, valid in [(1, True, True), (1, False, False), (2, False, True)]
    ]
    summary = summarize_results(rows, generation_seconds=2, evaluation_seconds=3)
    assert summary["accuracy"] == 1 / 3
    assert summary["invalid_answers"] == 1
    assert summary["by_depth"]["1"]["accuracy"] == .5
    assert summary["generated_tokens_per_second"] == 3
    assert summary["questions_per_second"] == 1


def test_batch_scoring_excludes_prompt_and_padding(tiny_checkpoint, monkeypatch):
    _, model, tokenizer = tiny_checkpoint
    tokenizer.padding_side = "left"
    examples = [generate_example(17, 1, "test", 0), generate_example(18, 2, "test", 1)]

    def generate(**kwargs):
        assert kwargs["generation_config"].do_sample is False
        assert kwargs["generation_config"].use_cache is False
        ids = [[tokenizer.convert_tokens_to_ids(example.final_state), 1, 0] for example in examples]
        return torch.cat([kwargs["input_ids"], torch.tensor(ids)], dim=1)

    monkeypatch.setattr(model, "generate", generate)
    batches = list(evaluate_batches(model, tokenizer, examples, TEMPLATE, batch_size=2, max_new_tokens=3, prompt_format="chat"))
    rows, seconds = batches[0]
    assert seconds > 0
    assert all(row["correct"] and row["generated_tokens"] == 2 for row in rows)
    assert all(row["response"] == example.final_state for row, example in zip(rows, examples))


@pytest.mark.parametrize("kind", ["directory", "pt", "safetensors"])
def test_model_sources_preserve_weights(tiny_checkpoint, tmp_path, kind):
    directory, original, _ = tiny_checkpoint
    source = directory
    base = None
    if kind == "pt":
        source = tmp_path / "weights.pt"
        torch.save({"state_dict": original.state_dict()}, source)
        base = str(directory)
    elif kind == "safetensors":
        source = directory / "model.safetensors"
    loaded, tokenizer, _ = load_model(
        str(source), base_model=base, tokenizer_source=None, revision=None,
        device="cpu", dtype="float32", download=False,
    )
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[key]), key
    assert tokenizer.padding_side == "left"
    assert loaded.training is False


def test_partial_state_dict_rejected(tiny_checkpoint, tmp_path):
    _, model, _ = tiny_checkpoint
    path = tmp_path / "partial.pt"
    torch.save({"lm_head.weight": model.lm_head.weight.detach()}, path)
    with pytest.raises(ValueError, match="missing"):
        load_state_file(model, path)


@pytest.mark.parametrize("test_count", [None, 2, 3])
def test_cli_outputs_and_optional_prompt_inspection(tiny_checkpoint, tmp_path, test_count):
    directory, _, _ = tiny_checkpoint
    data = tmp_path / "test.jsonl"
    data.write_text("".join(json.dumps(generate_example(17 + index, index + 1, "test", index).to_dict()) + "\n" for index in range(3)))
    output = tmp_path / "results"
    command = [
        sys.executable, "-m", "scripts.eval.naive_test", "--model", str(directory),
        "--data", str(data), "--output", str(output), "--batch-size", "2", "--max-new-tokens", "2",
    ]
    if test_count is not None:
        command.append("--test")
        if test_count == 2:
            command.append("2")
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    assert "Accuracy:" in result.stdout
    if test_count is None:
        assert "Rules:" not in result.stdout and "Decoded response" not in result.stdout
    else:
        assert "TEST MODE" in result.stdout
        assert result.stdout.count("Model input ·") == test_count
        assert result.stdout.count("Decoded response") == test_count
        assert "expected:" in result.stdout and "parsed:" in result.stdout
    summary = json.loads((output / "summary.json").read_text())
    with (output / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["status"] == "complete" and summary["total"] == len(rows) == (test_count or 3)
    assert summary["test_mode"] is (test_count is not None)
    assert summary["correct"] == sum(row["correct"] == "True" for row in rows)
    assert summary["generated_tokens"] == sum(int(row["generated_tokens"]) for row in rows)
    assert summary["local_checkpoint_sha256"] and summary["data_sha256"]
    if test_count is not None:
        assert result.stdout.count("RIGHT ·") == summary["correct"]
        assert result.stdout.count("WRONG ·") == summary["total"] - summary["correct"]
    repeated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert repeated.returncode != 0 and "already exists" in repeated.stderr
