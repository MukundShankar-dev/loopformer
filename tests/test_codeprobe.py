"""Headroom metrics and result readers for the code-benchmark probe."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from sandbox_marks import needs_sandbox
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


def test_livecodebench_groups_and_date_window(tmp_path):
    path = tmp_path / "eval_all.json"
    path.write_text(json.dumps([
        {"question_id": "q1", "difficulty": "easy", "contest_date": "2024-10-01T00:00:00", "graded_list": [True, True]},
        {"question_id": "q2", "difficulty": "easy", "contest_date": "2025-02-01T00:00:00", "graded_list": [True, False]},
        {"question_id": "q3", "difficulty": "hard", "contest_date": "2025-03-01T00:00:00", "graded_list": [False, False]}]))
    groups = read_livecodebench(path, dates=("2024-10-01", "2025-02-01"))
    assert groups["all"] == {"q1": [True, True], "q2": [True, False]}
    assert set(groups["easy"]) == {"q1", "q2"} and "hard" not in groups
    assert set(read_livecodebench(path)["all"]) == {"q1", "q2", "q3"}
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
    sandboxed = {'scripts.codeprobe.evaluate_sandboxed', 'scripts.codeprobe.evaluate_livecodebench'}
    assert {s['argv'][2] for s in steps if s['kind'] == 'sandbox'} == sandboxed
    assert not any('--evaluate' in s['argv'] for s in steps)  # LiveCodeBench's in-process evaluation
    lcb = config['livecodebench']
    assert all(lcb['problems'] in ' '.join(s['argv']) for s in steps if 'livecodebench_codegen' in ' '.join(s['argv']))
    assert all(m['revision'] in ' '.join(s['argv']) for m in config['models'] for s in steps
               if s['kind'] == 'gpu' and m['name'] in s['name'])
    assert sum('--greedy' in s['argv'] for s in steps) == len(config['models']) * len(config['evalplus_datasets'])
    path = tmp_path / 'probe.json'
    for change, flags, message in (({'models': [{'name': 'm', 'id': 'org/m', 'revision': None}]}, [], 'Pin model revisions'),
                                   ({'output': str(tmp_path)}, [], 'pass --resume'),
                                   ({'output': str(tmp_path / 'none')}, ['--resume'], 'does not exist')):
        path.write_text(json.dumps({**config, **change}))
        result = subprocess.run([sys.executable, '-m', 'scripts.codeprobe.commands', '--config', str(path), '--run',
                                 *flags], capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr
    assert all('creates' in s for s in steps if s['kind'] != 'network')
    dry = subprocess.run([sys.executable, '-m', 'scripts.codeprobe.commands'], capture_output=True, text=True, check=True)
    assert 'livecodebench_codegen' in dry.stdout and 'evaluate_livecodebench' in dry.stdout


def test_livecodebench_tests_decode_without_loading_classes(tmp_path):
    import base64
    import os
    import pickle
    import zlib

    from scripts.codeprobe.livecodebench_data import decode_private_tests

    def stored(value) -> str:
        return base64.b64encode(zlib.compress(pickle.dumps(value))).decode()

    tests = json.dumps([{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}])
    assert decode_private_tests(tests) == tests
    assert decode_private_tests(stored(tests)) == tests
    marker = tmp_path / "ran"

    class Payload:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        decode_private_tests(stored(Payload()))
    assert not marker.exists()
    with pytest.raises(ValueError, match="pickled JSON text"):
        decode_private_tests(stored([1, 2]))


def test_livecodebench_codegen_refuses_host_evaluation(tmp_path):
    command = [sys.executable, "-m", "scripts.codeprobe.livecodebench_codegen", "--lcb", str(tmp_path),
               "--problems", str(tmp_path), "--output", str(tmp_path / "out.json"), "--", "--model", "org/m"]
    for flag in ("--evaluate", "--continue_existing_with_eval", "--not_fast"):
        result = subprocess.run([*command, flag], capture_output=True, text=True)
        assert result.returncode != 0 and f"refused LiveCodeBench flags: ['{flag}']" in result.stderr
    assert not (tmp_path / "out.json").exists()


LIVECODEBENCH = Path.home() / "src" / "LiveCodeBench"
PROBE_VENV = Path(".venv-probe").absolute()


@needs_sandbox
@pytest.mark.skipif(not (LIVECODEBENCH / "lcb_runner").is_dir() or not PROBE_VENV.is_dir(),
                    reason="needs the LiveCodeBench checkout and .venv-probe (docs/setup.md)")
def test_livecodebench_evaluator_grades_in_the_sandbox(tmp_path):
    from scripts.codeprobe.evaluate_livecodebench import evaluate_problem

    def problem(question_id, platform, tests, metadata):
        public, hidden = tests[:1], tests[1:]
        (tmp_path / f"{question_id}.json").write_text(json.dumps(hidden))
        return {"question_title": question_id, "question_content": "", "platform": platform,
                "question_id": question_id, "contest_id": "c", "contest_date": "2024-01-01T00:00:00",
                "starter_code": "", "difficulty": "easy", "public_test_cases": json.dumps(public),
                "metadata": json.dumps(metadata)}

    add = problem("add", "atcoder", [{"input": f"{a} {b}\n", "output": f"{a + b}\n", "testtype": "stdin"}
                                     for a, b in [(1, 2), (5, 7), (-3, 3)]], {})
    codes = ["a, b = map(int, input().split())\nprint(a + b)", "print(3)", "import os\nos.system('true')\nprint(3)"]
    verdict = evaluate_problem(add, codes, tmp_path / "add.json", LIVECODEBENCH, PROBE_VENV, timeout=6, workers=2,
                               memory_gb=4)
    assert verdict["graded_list"] == [True, False, False]
    double = problem("double", "leetcode", [{"input": f"{json.dumps(xs)}", "output": json.dumps([2 * x for x in xs]),
                                             "testtype": "functional"} for xs in ([1], [2, 3], [])],
                     {"func_name": "double"})
    codes = ["class Solution:\n    def double(self, xs):\n        return [2 * x for x in xs]",
             "class Solution:\n    def double(self, xs):\n        return xs"]
    verdict = evaluate_problem(double, codes, tmp_path / "double.json", LIVECODEBENCH, PROBE_VENV, timeout=6,
                               workers=2, memory_gb=4)
    assert verdict["graded_list"] == [True, False]
    hog = "class Solution:\n    def double(self, xs):\n        block = bytearray(8 << 30)\n        return [2 * x for x in xs]"
    verdict = evaluate_problem(double, [codes[0], hog, codes[0]], tmp_path / "double.json", LIVECODEBENCH, PROBE_VENV,
                               timeout=6, workers=2, memory_gb=2)
    assert verdict["graded_list"] == [True, False, True]  # the hog fails alone; the others keep their verdicts
    assert json.loads(verdict["metadata"][1]) == {"error_message": "sandbox: out of memory"}
