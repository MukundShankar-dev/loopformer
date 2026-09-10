from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.dataset.dataset import (
    DatasetConfig, SPLITS, encode_records, generate_dataset, make_record, preview_indices,
    summarize, verify_dataset, write_dataset,
)
from scripts.dataset.pointer import (
    SYMBOLS, check_predictions, execute, generate_example, mapping_fingerprint,
    parse_mapping, validate_example,
)
from scripts.dataset.symbols import MODEL_ID, MODEL_REVISION, validate_symbols


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("Pinned Qwen tokenizer is not cached; integration checks require local files")


@pytest.fixture(scope="module")
def token_ids(tokenizer):
    return validate_symbols(tokenizer)


@pytest.fixture
def small_config():
    return DatasetConfig(train_count=16, validation_count=8, test_count=8, depth_test_count=8)


def test_user_example_and_step_scoring():
    mapping = parse_mapping("(A,C) (C,D) (D, E)")
    assert execute(mapping, "A", 2) == ["C", "D"]
    assert execute(mapping, "A", 0) == []
    assert check_predictions(mapping, "A", [" C", "D"], steps=2) == [True, True]
    assert check_predictions(mapping, "A", ["D", "E"], steps=2) == [False, False]
    assert check_predictions(mapping, "A", ["The answer is C"], steps=2) == [False]
    with pytest.raises(ValueError, match="at most"):
        check_predictions(mapping, "A", ["C", "D", "E"], steps=2)
    with pytest.raises(ValueError, match="at most"):
        check_predictions(mapping, "A", "CD", steps=2)


@pytest.mark.parametrize("text", ["", "(A,C) garbage", "(A,C) (A,D)", "(AA,C)", "(A,C),(C,D)"])
def test_malformed_mapping_rejected(text):
    with pytest.raises(ValueError):
        parse_mapping(text)


def test_missing_rules_and_cycles_have_explicit_semantics():
    with pytest.raises(ValueError, match="step 2"):
        execute({"A": "B"}, "A", 2)
    assert execute({"A": "B", "B": "A"}, "A", 4) == ["B", "A", "B", "A"]
    assert execute({"A": "A"}, "A", 3) == ["A", "A", "A"]
    with pytest.raises(ValueError, match="nonnegative"):
        execute({"A": "A"}, "A", -1)


@pytest.mark.parametrize("depth", [1, 2, 8, 16, 25])
def test_generated_tables_and_exact_paths(depth):
    for seed in range(20):
        example = generate_example(seed, depth, "train", 0)
        validate_example(example)
        table = dict(example.mapping)
        assert set(table) == set(SYMBOLS)
        assert len(example.mapping) == 26
        current = example.initial_state
        # Step through the saved pairs directly, separately from execute().
        expected = []
        for _ in range(depth):
            current = next(target for source, target in example.mapping if source == current)
            expected.append(current)
        assert expected == example.intermediate_states
        assert expected[-1] == example.final_state
        assert len(set([example.initial_state, *expected])) == depth + 1
        assert generate_example(seed, depth, "train", 0) == example
    assert generate_example(1, depth, "train", 0).mapping != generate_example(2, depth, "train", 0).mapping


def test_invalid_depth_and_corrupt_targets_rejected():
    for depth in (0, 26, 1.5, True):
        with pytest.raises(ValueError, match="depth"):
            generate_example(17, depth, "train", 0)
    example = generate_example(17, 3, "train", 0)
    with pytest.raises(ValueError, match="labels"):
        validate_example(replace(example, intermediate_states=[example.initial_state] * 3))
    with pytest.raises(ValueError, match="Prompt"):
        validate_example(replace(example, prompt=example.prompt.replace("Steps: 3", "Steps: 2")))
    assert mapping_fingerprint(dict(example.mapping)) == mapping_fingerprint(dict(reversed(example.mapping)))


def test_real_tokenizer_prompt_and_answer_contract(tokenizer, token_ids, small_config):
    assert len(set(token_ids.values())) == 26
    for split in ("train", "depth_test"):
        record = make_record(small_config, split, 7, tokenizer, token_ids)
        prefix = tokenizer.encode(record["prompt"], add_special_tokens=False)
        assert record["answer_position"] == len(prefix) - 1
        assert record["prompt_token_count"] == len(prefix)
        for symbol in SYMBOLS:
            assert tokenizer.encode(record["prompt"] + " " + symbol, add_special_tokens=False) == [*prefix, token_ids[symbol]]
        assert [tokenizer.decode([value]).strip() for value in record["intermediate_token_ids"]] == record["intermediate_states"]


def test_balanced_disjoint_splits_and_prefix_reproducibility(tokenizer, token_ids, small_config):
    dataset = generate_dataset(small_config, tokenizer, token_ids)
    repeated = generate_dataset(small_config, tokenizer, token_ids)
    fingerprints = []
    for split in SPLITS:
        assert encode_records(dataset[split]) == encode_records(repeated[split])
        counts = summarize(dataset[split])["depth_counts"]
        assert max(counts.values()) - min(counts.values()) <= 1
        assert set(map(int, counts)) == set(small_config.depths(split))
        fingerprints.extend(record["mapping_sha256"] for record in dataset[split])
    assert len(fingerprints) == len(set(fingerprints))
    larger = generate_dataset(replace(small_config, train_count=24), tokenizer, token_ids)
    assert larger["train"][:16] == dataset["train"]
    assert larger["depth_test"] == dataset["depth_test"]
    different = generate_dataset(replace(small_config, seed=18), tokenizer, token_ids)
    assert different["train"][0]["mapping"] != dataset["train"][0]["mapping"]


@pytest.mark.parametrize("change", [{"seed": -1}, {"train_count": 0}, {"max_eval_depth": 26}, {"max_train_depth": 16}, {"min_depth": 0}])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        replace(DatasetConfig(), **change).validate()


@pytest.mark.parametrize("count", [5, 10])
def test_preview_spans_depths_and_uses_real_distinct_records(small_config, count):
    selected = preview_indices(small_config, count)
    assert len(selected) == len(set(selected)) == count
    depths = []
    for split, index in selected:
        assert 0 <= index < small_config.count(split)
        depths.append(small_config.depths(split)[index % len(small_config.depths(split))])
    assert depths == sorted(depths)
    assert min(depths) == 1 and max(depths) == 16
    assert len(set(depths)) == count


def test_preview_respects_tiny_split_counts():
    config = DatasetConfig(train_count=1, validation_count=1, test_count=1, depth_test_count=1)
    selected = preview_indices(config, 10)
    assert len(selected) == len(set(selected)) == 4
    assert all(index == 0 for _, index in selected)


def test_saved_dataset_verification_detects_corruption(tmp_path, tokenizer, token_ids, small_config):
    dataset = generate_dataset(small_config, tokenizer, token_ids)
    output = tmp_path / "dataset"
    manifest = write_dataset(output, dataset, small_config, token_ids, {})
    assert verify_dataset(output, tokenizer) == manifest
    with pytest.raises(FileExistsError):
        write_dataset(output, dataset, small_config, token_ids, {})
    path = output / "train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_dataset(output, tokenizer)
    # Even updating the manifest checksum must not hide a wrong target.
    dataset["train"][0]["final_state"] = dataset["train"][0]["initial_state"]
    path.write_bytes(encode_records(dataset["train"]))
    manifest["splits"]["train"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="labels"):
        verify_dataset(output, tokenizer)


@pytest.mark.parametrize("count", [5, 10])
def test_cli_dry_run_prints_without_creating_output(tmp_path, tokenizer, count):
    output = tmp_path / "must-not-exist" / "dataset"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "scripts.dataset", "--dry-run", str(count), "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True,
    )
    assert f"Dry run complete: {count} examples" in result.stdout
    assert result.stdout.count("Final:") == count
    assert "Configured dataset distribution" in result.stdout
    assert "depth 16" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_packaged_cli_writes_and_verifies_with_source_provenance(tmp_path, tokenizer):
    output = tmp_path / "dataset"
    command = [sys.executable, "-B", "-m", "scripts.dataset", "--output", str(output)]
    for split in SPLITS:
        command.extend([f"--{split.replace('_', '-')}-count", "8"])
    result = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True,
    )
    assert "Data milestone complete" in result.stdout
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["provenance"]["source_sha256"]
    assert all(name.startswith("scripts/dataset/") for name in manifest["provenance"]["source_sha256"])
    assert all(manifest["splits"][split]["count"] == 8 for split in SPLITS)
