"""Headroom metrics and result readers for the code-benchmark probe."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.codeprobe.headroom import pass_at_k, read_evalplus, read_livecodebench, summarize


def test_pass_at_k_matches_exact_combinatorics():
    for n, c, k in [(16, 0, 1), (16, 3, 1), (16, 3, 4), (16, 15, 2), (10, 10, 5)]:
        exact = 1 - math.comb(n - c, k) / math.comb(n, k)
        assert pass_at_k(n, c, k) == pytest.approx(exact, abs=1e-12)
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)


def test_summary_splits_learnable_problems():
    results = {"a": [True] * 4, "b": [True, False, False, False], "c": [False] * 4, "d": [False] * 4}
    row = summarize(results, ks=(1, 4))
    assert (row["always_solved"], row["sometimes_solved"], row["never_solved"]) == (0.25, 0.25, 0.5)
    assert row["pass@1"] == pytest.approx((1 + 0.25) / 4)
    assert row["pass@4"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unequal sample counts"):
        summarize({"a": [True], "b": [True, False]})


def test_evalplus_plus_requires_both_test_sets(tmp_path):
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps({"eval": {"HumanEval/0": [
        {"base_status": "pass", "plus_status": "pass"},
        {"base_status": "pass", "plus_status": "fail"},
        {"base_status": "fail", "plus_status": "pass"}]}}))
    assert read_evalplus(path) == {"HumanEval/0": [True, False, False]}
    assert read_evalplus(path, "base") == {"HumanEval/0": [True, True, False]}
    path.write_text(json.dumps({"HumanEval/0": []}))
    with pytest.raises(ValueError, match="'eval' key"):
        read_evalplus(path)


def test_livecodebench_groups_and_date_cutoff(tmp_path):
    path = tmp_path / "eval_all.json"
    path.write_text(json.dumps([
        {"question_id": "q1", "difficulty": "easy", "contest_date": "2024-10-01T00:00:00", "graded_list": [True, True]},
        {"question_id": "q2", "difficulty": "easy", "contest_date": "2025-02-01T00:00:00", "graded_list": [True, False]},
        {"question_id": "q3", "difficulty": "hard", "contest_date": "2025-03-01T00:00:00", "graded_list": [False, False]}]))
    groups = read_livecodebench(path, since="2025-01-01")
    assert groups["all"] == {"q2": [True, False], "q3": [False, False]}
    assert set(groups["easy"]) == {"q2"} and set(groups["hard"]) == {"q3"}
    path.write_text(json.dumps([{"question_id": "q1"}]))
    with pytest.raises(ValueError, match="lacks"):
        read_livecodebench(path)


def test_cli_writes_summary_and_refuses_overwrite(tmp_path):
    results = tmp_path / "eval_results.json"
    results.write_text(json.dumps({"eval": {"t": [{"base_status": "pass", "plus_status": "pass"}] * 16}}))
    output = tmp_path / "summary.json"
    command = [sys.executable, "-m", "scripts.codeprobe.headroom", "--evalplus", f"humaneval={results}", "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert json.loads(output.read_text())["benchmarks"]["humaneval"]["always_solved"] == 1.0
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "Output exists" in again.stderr


def test_probe_commands_execute_samples_only_in_the_sandbox(tmp_path):
    from scripts.codeprobe.commands import build_commands, load_config
    config = load_config(Path('configs/probe_headroom.json'))
    steps = build_commands(config)
    assert {s['kind'] for s in steps if 'evaluate' in ' '.join(s['argv'])} == {'sandbox'}
    assert all(s['argv'][2] == 'scripts.codeprobe.evaluate_sandboxed' for s in steps if s['kind'] == 'sandbox')
    assert not any('--evaluate' in s['argv'] for s in steps)  # LiveCodeBench's in-process evaluation
    assert all(m['revision'] in ' '.join(s['argv']) for m in config['models'] for s in steps
               if s['kind'] == 'gpu' and m['name'] in s['name'])
    assert sum('--greedy' in s['argv'] for s in steps) == len(config['models']) * len(config['evalplus_datasets'])
    path = tmp_path / 'probe.json'
    for change, message in (({'models': [{'name': 'm', 'id': 'org/m', 'revision': None}]}, 'Pin model revisions'),
                            ({'output': str(tmp_path)}, 'exists')):
        path.write_text(json.dumps({**config, **change}))
        result = subprocess.run([sys.executable, '-m', 'scripts.codeprobe.commands', '--config', str(path), '--run'],
                                capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr
    dry = subprocess.run([sys.executable, '-m', 'scripts.codeprobe.commands'], capture_output=True, text=True, check=True)
    assert 'lcb_runner.runner.main' in dry.stdout and 'evaluate_sandboxed' in dry.stdout
