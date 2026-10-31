"""Problem-set storage: hidden tests round-trip exactly and load only on demand."""
import json

import pytest

from sandbox_marks import needs_sandbox
from scripts.codetasks.problems import Problem, in_memory, read_problems, write_problems
from scripts.sandbox.grader import TestCase, Tests


def problem(problem_id: str, hidden: Tests) -> Problem:
    public = Tests("stdio", (TestCase("1\n", "2\n"),))
    return Problem(problem_id, "test", "Add one.", public, in_memory(hidden), {"difficulty": "easy"})


def test_problem_set_round_trip_and_lazy_hidden_tests(tmp_path):
    stdio = Tests("stdio", (TestCase("1\n", "2\n"), TestCase("x\ny\n", "line\n\n")))
    call = Tests("call", (TestCase('[1, {"__tuple__": [2]}]', "3.5"),), function="f", atol=1e-6,
                 imports="from typing import *\n", seconds=6)
    directory = tmp_path / "set"
    assert write_problems(directory, iter([problem("Mbpp/2", stdio), problem("abc_1", call)])) == 2
    loaded = read_problems(directory)
    assert [p.id for p in loaded] == ["Mbpp/2", "abc_1"] and loaded[0].metadata == {"difficulty": "easy"}
    assert loaded[0].public == Tests("stdio", (TestCase("1\n", "2\n"),))
    assert "x\\ny" not in (directory / "problems.jsonl").read_text()  # hidden cases live only in hidden/
    assert loaded[0].hidden() == stdio and loaded[1].hidden() == call
    assert json.loads((directory / "problems.jsonl").read_text().splitlines()[0])["hidden_cases"] == 2
    (directory / "hidden" / "abc_1.jsonl").unlink()
    loaded = read_problems(directory)  # reading does not touch hidden files
    with pytest.raises(FileNotFoundError):
        loaded[1].hidden()


def test_problem_set_refuses_overwrite_and_id_collisions(tmp_path):
    tests = Tests("stdio", (TestCase("", ""),))
    with pytest.raises(FileExistsError):
        write_problems(tmp_path, [])
    with pytest.raises(ValueError, match="collides"):
        write_problems(tmp_path / "set", [problem("a/b", tests), problem("a_b", tests)])


def test_livecodebench_conversion_follows_its_test_semantics(tmp_path):
    from scripts.codetasks.livecodebench import IMPORTS, SECONDS, problems
    from scripts.sandbox.values import decode

    prepared = tmp_path / "prepared"
    (prepared / "tests").mkdir(parents=True)
    base = {"question_title": "t", "question_content": "Q ", "contest_id": "c", "contest_date": "2024-01-01T00:00:00",
            "difficulty": "easy"}
    leetcode = {**base, "question_id": "3000", "platform": "leetcode",
                "starter_code": "class Solution:\n    def f(self, xs: List[int], k: int) -> List[int]:\n        ",
                "metadata": json.dumps({"func_name": "f"}),
                "public_test_cases": json.dumps([{"input": "[1, 2]\n3", "output": "[4, 5]", "testtype": "functional"}])}
    atcoder = {**base, "question_id": "abc1_a", "platform": "atcoder", "starter_code": "", "metadata": "{}",
               "public_test_cases": json.dumps([{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}])}
    (prepared / "tests" / "3000.json").write_text(json.dumps([{"input": "[]\n0", "output": "[]", "testtype": "functional"}]))
    (prepared / "tests" / "abc1_a.json").write_text(json.dumps([{"input": "2 2\n", "output": "4\n", "testtype": "stdin"}]))
    empty = {**atcoder, "question_id": "abc2_c", "public_test_cases": "[]"}
    (prepared / "problems.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (leetcode, atcoder, empty)))
    excluded = {}
    call, stdio = problems(prepared, "lcb-test", excluded)
    assert excluded == {"abc2_c": "no public examples"}
    assert call.public.kind == "call" and call.public.function == "f"
    assert [decode(json.loads(case.input)) for case in call.hidden().cases] == [[[1, 2], 3], [[], 0]]
    assert [json.loads(case.expected) for case in call.hidden().cases] == [[4, 5], []]
    assert call.statement.startswith("Q\n\nStarter code:\n```python\nclass Solution:") and call.metadata["platform"] == "leetcode"
    assert stdio.statement == "Q" and stdio.public.cases == (TestCase("1 2\n", "3\n"),)
    assert stdio.hidden() == Tests("stdio", (TestCase("1 2\n", "3\n"), TestCase("2 2\n", "4\n")),
                                   imports=IMPORTS, seconds=SECONDS)
    assert (call.public.imports, call.public.seconds) == (call.hidden().imports, call.hidden().seconds) == (IMPORTS, SECONDS)
    (prepared / "tests" / "abc1_a.json").write_text(json.dumps([{"input": "", "output": "", "testtype": "functional"}]))
    with pytest.raises(ValueError, match="unexpected test types"):
        stdio.hidden()


def test_kodcode_inputs_are_literal_calls_inside_asserts():
    from scripts.codetasks.kodcode import Excluded, call_inputs

    source = """
from solution import f
import pytest

from sandbox_marks import needs_sandbox

def test_values():
    assert f([1, 2], 3) == 4
    assert f([1, 2], 3) == 4  # duplicate
    assert f(x=1) == 2        # keyword: skipped
    assert abs(f((1.5,), 0) - 1) < 1e-9
    value = f([9], 9)          # not in an assert
    assert f(*args) == 0       # starred: skipped
    assert f(make(), 1) == 0   # not literal: skipped

def test_raises():
    with pytest.raises(ValueError):
        f([], -1)
"""
    assert call_inputs(source, "f") == [[[1, 2], 3], [(1.5,), 0]]
    from scripts.codetasks.kodcode import convert
    row = {"question_id": "q", "question": "Q", "solution": "def f(x):\n    return x", "style": "instruct",
           "subset": "Algorithm", "gpt_difficulty": "easy", "benchmark_similarity": 0.1,
           "test_info": [{"function_name": "f", "function_declaration": "def f(x):"}],
           "test": "def test():\n    assert f(b'a') == b'a'\n    assert f(b'b')\n    assert f(b'c')"}
    with pytest.raises(Excluded, match="cannot encode a value of type bytes"):
        convert(row, 0.0, "kodcode-test")
    with pytest.raises(Excluded, match="does not parse"):
        call_inputs("def broken(:", "f")


def test_deepcoder_tests_statements_and_solutions():
    from scripts.codetasks.deepcoder import (PRIMEINTELLECT_PREFIX, PRIMEINTELLECT_SUFFIX, Excluded, clean_solution,
                                            clean_statement, stdio_cases)

    taco = json.dumps({"inputs": ["1\n", "2\n", "3\n"], "outputs": ["1\n", "4\n", "9\n"]})
    assert stdio_cases(taco)[1] == TestCase("2\n", "4\n")
    prime = json.dumps([{"type": "stdin_stdout", "input": str(i), "output": str(i)} for i in range(3)])
    assert len(stdio_cases(prime)) == 3
    for bad, message in [(json.dumps({"inputs": [[1]] * 3, "outputs": [[1]] * 3, "fn_name": "f"}), "function-call"),
                         (json.dumps({"inputs": [[1]] * 3, "outputs": ["1"] * 3}), "non-string"),
                         (json.dumps({"inputs": ["1"], "outputs": ["1"]}), "1 tests")]:
        with pytest.raises(Excluded, match=message):
            stdio_cases(bad)
    assert clean_statement("primeintellect", PRIMEINTELLECT_PREFIX + "Body." + PRIMEINTELLECT_SUFFIX) == "Body."
    assert clean_solution("```python\nprint(1)\n```") == "print(1)\n" and clean_solution("print(2)") == "print(2)"


def test_dedup_containment_finds_rewordings_not_shared_phrasing():
    from scripts.codetasks.dedup import DEDUP_N, DEDUP_THRESHOLD, EvalIndex

    index = EvalIndex({"mbpp:119": "Write a python function to find the element that appears only once in a sorted "
                                   "array. assert search([1,1,2,2,3]) == 3"}, DEDUP_N)
    rewording = ("Write a python function to find the element that appears only once in a sorted array. All other "
                 "elements appear exactly twice.")
    generic = "Write a python function to find the length of the longest valid parentheses substring."
    assert index.overlap(rewording)[0] >= DEDUP_THRESHOLD and index.overlap(rewording)[1] == "mbpp:119"
    assert index.overlap(generic)[0] < DEDUP_THRESHOLD


@needs_sandbox
def test_reference_outputs_ignore_prints_and_catch_nondeterminism():
    from scripts.codetasks.reference import ReferenceFailed, reference_outputs

    chatty = "print('loading')\ndef f(x):\n    print('called', x)\n    return x * 2"
    assert reference_outputs(chatty, "f", [[1], [(2,)]], repeats=2) == [2, {"__tuple__": [2, 2]}]
    with pytest.raises(ReferenceFailed, match="nondeterministic"):
        reference_outputs("import random\ndef f(x):\n    return random.random()", "f", [[1]], repeats=2)
    assert reference_outputs("def f(x):\n    return 1 // x", "f", [[0]]) == [{"__error__": "ZeroDivisionError"}]
