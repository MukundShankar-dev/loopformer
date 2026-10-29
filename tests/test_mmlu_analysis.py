"""Independent MMLU scoring rejects changed prompts and catches paired losses."""

import hashlib
import json
from copy import deepcopy

import pytest

from scripts.eval.mmlu_analysis import audit_row, paired_summary, read_samples, render_context


def test_raw_likelihood_audit_and_prompt_integrity():
    doc = {"question": "Which?", "choices": ["one", "two", "three", "four"], "answer": 0}
    context = render_context("Subject\n\n", [doc] * 5, doc)
    responses = [[-1.0, False], [-1.0, False], [-2.0, False], [-3.0, False]]
    row = {"doc": doc, "target": 0, "arguments": [[context, " " + letter] for letter in "ABCD"],
           "prompt_hash": hashlib.sha256(context.encode()).hexdigest(), "filtered_resps": responses,
           "resps": [[value] for value in responses], "acc": 1}
    assert audit_row(row, doc, context) == (0, True)  # First maximum on a tie.
    damaged = deepcopy(row)
    damaged["arguments"][0][0] += " A"
    with pytest.raises(ValueError, match="Prompt"):
        audit_row(damaged, doc, context)
    damaged = deepcopy(row)
    damaged["acc"] = 0
    with pytest.raises(ValueError, match="accuracy"):
        audit_row(damaged, doc, context)


def test_paired_effect_preserves_damage_and_repair():
    rows = [
        {"subject": "one", "base_correct": True, "recurrent_correct": False, "base_prediction": 0, "recurrent_prediction": 1},
        {"subject": "one", "base_correct": False, "recurrent_correct": True, "base_prediction": 1, "recurrent_prediction": 0},
        {"subject": "two", "base_correct": True, "recurrent_correct": False, "base_prediction": 0, "recurrent_prediction": 1},
        {"subject": "two", "base_correct": True, "recurrent_correct": True, "base_prediction": 0, "recurrent_prediction": 0},
    ]
    summary = paired_summary(rows, repetitions=1000)
    assert summary["base_accuracy"] == 0.75
    assert summary["recurrent_accuracy"] == 0.5
    assert summary["difference"] == -0.25
    assert summary["base_correct_recurrent_wrong"] == 2
    assert summary["base_wrong_recurrent_correct"] == 1
    assert summary["prediction_agreement"] == 0.25
    assert summary["difference_ci95"][0] <= -0.25 <= summary["difference_ci95"][1]


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    records = [{"question": "first\u2028second\u0085third"}, {"question": "another"}]
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    assert read_samples(path) == records
