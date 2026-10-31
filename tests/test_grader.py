"""Grader contracts: comparison and classification on the host, and cheats that
would beat a naive grader but must fail here."""
import dataclasses
import hashlib
import json
import textwrap
import uuid

import pytest

from sandbox_marks import needs_sandbox
from scripts.sandbox.grader import (
    TestCase, Tests, exception_type, failure, grade, parse_value, same_stdout, same_value)
from scripts.sandbox.runner import Limits, RunResult
from scripts.sandbox.values import BIG_INT, decode, encode


def stdio(*pairs: tuple[str, str]) -> Tests:
    return Tests("stdio", tuple(TestCase(i, e) for i, e in pairs))


def call(function: str, *pairs: tuple[list, object], atol: float = 0.0) -> Tests:
    cases = tuple(TestCase(json.dumps(encode(a)), json.dumps(encode(e))) for a, e in pairs)
    return Tests("call", cases, function=function, atol=atol)


def observed(exit_code: int = 0, stderr: str = "", timed_out: bool = False, oom_killed: bool = False) -> RunResult:
    return RunResult(exit_code, timed_out, oom_killed, 0.1, 0.1, "", stderr, False, False, unit="x")


SUM = call("add", ([1, 2], 3), ([5, -5], 0), ([10, 20], 30))


# Host-side comparison and classification: run anywhere.

def test_stdout_comparison_ignores_spacing_and_numeric_formatting_only():
    assert same_stdout("1 2\n3\n", "1   2\n3")
    assert same_stdout("0.5\n", "0.50\n") and same_stdout("10\n", "1e1\n")
    assert not same_stdout("1 2\n", "1\n2\n")
    assert not same_stdout("1 2\n", "1 2 3\n")
    assert not same_stdout("abc\n", "ABC\n")
    assert not same_stdout("NaN\n", "nan\n") and not same_stdout("1\n", "")


def test_tagged_values_round_trip_and_keep_types_distinct():
    for value in [(1, [2, (3,)]), {3, 1, 2}, {"a": (1,), (1, 2): {4}}, {"__tuple__": 1}, 1 + 2j, [], (), 1.5, None, "x"]:
        assert decode(encode(value)) == value and type(decode(encode(value))) is type(value)
    assert encode((1, 2)) != encode([1, 2]) and encode({1, "a"}) == encode({"a", 1})
    nested = {frozenset({1, 2}), frozenset()}
    assert decode(encode(nested)) == nested and decode(encode({(frozenset({3}),): 1})) == {(frozenset({3}),): 1}
    with pytest.raises(TypeError, match="object"):
        encode(object())


def test_huge_integers_cross_as_decimal_strings():
    import sys
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        for value in [10 ** 5000, -(10 ** 5000) - 7, [BIG_INT, BIG_INT - 1]]:
            assert decode(encode(value)) == value
        assert encode(BIG_INT - 1) == BIG_INT - 1 and encode(True) is True
        assert same_value(encode(10 ** 5000), encode(10 ** 5000), 0.0)
        assert not same_value(encode(10 ** 5000), encode(10 ** 5000 + 1), 0.0)
        assert not same_value(encode(10 ** 5000), 1e308, 0.0)
        text = json.dumps(encode([10 ** 5000]))
    finally:
        sys.set_int_max_str_digits(limit)
    assert parse_value(text)[0]  # the host parses the encoded form under the default limit


@needs_sandbox
def test_huge_integer_results_grade_by_value():
    tests = Tests("call", (TestCase("[5000]", json.dumps({"__int__": "1" + "0" * 5000})),), function="power")
    assert grade("def power(n):\n    return 10 ** n", tests).outcome == "pass"
    assert grade("def power(n):\n    return 10 ** n + 1", tests).outcome == "wrong_answer"


def test_value_comparison_is_strict_about_types_and_tolerant_only_for_floats():
    assert same_value([1, [2, {"a": None}]], [1, [2, {"a": None}]], 0.0)
    assert same_value(1, True, 0.0) and same_value([True], [1], 0.0)  # Python ==, which EvalPlus uses
    assert not same_value(2, True, 0.0)
    assert not same_value("1", 1, 0.0) and not same_value([1], [1, 1], 0.0)
    assert not same_value({"a": 1}, {"a": 1, "b": 2}, 0.0)
    assert same_value(0.3, 0.1 + 0.2, 0.0)  # within EvalPlus's relative 1e-7
    assert same_value(1.0, 1.0 + 1e-3, 1e-2) and not same_value(1.0, 1.0 + 1e-3, 0.0)
    assert same_value(2, 2.0, 0.0)


def test_adversarial_program_output_cannot_crash_the_host():
    deep = "[" * 200_000 + "]" * 200_000
    assert parse_value(deep) == (False, None)
    assert parse_value("9" * 10_000) == (False, None)  # over Python's int-conversion limit
    assert parse_value('"ok" "again"') == (False, None)
    parsed, value = parse_value("[" * 500 + "]" * 500)
    assert parsed and not same_value(1, value, 0.0)  # recursion follows the expected value, not the program's


def test_failure_classes():
    traceback = "Traceback (most recent call last):\n  File \"x\", line 1\n{}\n"
    assert failure(observed(timed_out=True, exit_code=-9)) == "timeout"
    assert failure(observed(exit_code=128 + 24)) == "timeout"
    assert failure(observed(exit_code=137, oom_killed=True)) == "memory"
    assert failure(observed(1, traceback.format("MemoryError"))) == "memory"
    assert failure(observed(1, traceback.format("ModuleNotFoundError: No module named 'numpy'"))) == "import_error"
    assert failure(observed(1, traceback.format("ZeroDivisionError: division by zero"))) == "runtime_error:ZeroDivisionError"
    assert failure(observed(1, traceback.format("json.decoder.JSONDecodeError: bad"))) == "runtime_error:JSONDecodeError"
    assert failure(observed(1, "custom message only")) == "runtime_error:exit"
    assert failure(observed(128 + 11)) == "runtime_error:signal11"
    assert failure(observed(0)) is None
    assert exception_type(traceback.format("\x1b]52;c;evil\x07: x")) is None  # names are sanitised


def test_tests_are_validated():
    with pytest.raises(ValueError, match="function name"):
        Tests("call", (TestCase("[]", "1"),))
    with pytest.raises(ValueError, match="function name"):
        Tests("stdio", (TestCase("", ""),), function="f")
    with pytest.raises(ValueError, match="JSON array"):
        Tests("call", (TestCase('{"a": 1}', "1"),), function="f")
    with pytest.raises(ValueError, match="no test cases"):
        Tests("stdio", ())


# Grading real programs.

@needs_sandbox
def test_stdio_grading_and_pass_fraction():
    tests = stdio(("1 2\n", "3\n"), ("4 4\n", "8\n"), ("5 1\n", "6\n"), ("2 2\n", "4\n"))
    assert grade("a, b = map(int, input().split())\nprint(a + b)", tests).outcome == "pass"
    partial = grade("a, b = map(int, input().split())\nprint(a + b if a % 2 == 0 else 0)", tests)
    assert partial.pass_fraction == 0.5 and partial.outcome == "wrong_answer"
    assert [r.outcome for r in partial.results] == ["wrong_answer", "pass", "wrong_answer", "pass"]


@needs_sandbox
def test_call_grading_function_and_solution_class_styles():
    assert grade("def add(a, b):\n    print('debug output')\n    return a + b", SUM).outcome == "pass"
    leetcode = "class Solution:\n    def add(self, a, b):\n        return a + b"
    assert grade(leetcode, SUM).outcome == "pass"
    shapes = call("f", (["tuple"], (1, 2)), (["set"], {1, 2, 3}), (["dict"], {(1, 2): "a"}), ([(5, 6)], [5, 6]))
    solution = "def f(kind):\n    return {'tuple': (1, 2), 'set': {3, 1, 2}, 'dict': {(1, 2): 'a'}}.get(kind) or list(kind)"
    assert grade(solution, shapes).outcome == "pass"  # tuple arguments and results survive the boundary
    assert grade("def f(kind): return [1, 2]", call("f", (["tuple"], (1, 2)))).outcome == "wrong_answer"
    # Where Python's == would give a different verdict: floats within tolerance.
    assert grade("def f(): return 1.001", call("f", ([], 1.0), atol=1e-2)).outcome == "pass"
    assert grade("def f(): return 1.001", call("f", ([], 1.0))).outcome == "wrong_answer"


@needs_sandbox
def test_failure_outcomes_of_real_programs():
    cases = {
        "def add(a, b) return a": "syntax_error",
        "import numpy_that_does_not_exist\ndef add(a, b): return a + b": "import_error",
        "def add(a, b): return a / 0": "runtime_error:ZeroDivisionError",
        "def add(a, b):\n    while True: pass": "timeout",
        "def add(a, b):\n    x = bytearray(1 << 32)": "memory",
        "def subtract(a, b): return a - b": "runtime_error:AttributeError",
        "def add(a, b): return a - b": "wrong_answer",
    }
    for solution, outcome in cases.items():
        result = grade(solution, SUM, workers=3)
        assert result.outcome == outcome and result.pass_fraction == 0.0, (solution, result.outcome)
    assert len(grade("def add(a, b) return a", SUM).results) == len(SUM.cases)


@needs_sandbox
def test_imports_run_in_the_solution_namespace():
    leetcode = "class Solution:\n    def total(self, xs: List[int]) -> int:\n        return sum(xs)\n"
    tests = call("total", ([[1, 2]], 3))
    assert grade(leetcode, tests).outcome == "runtime_error:NameError"
    assert grade(leetcode, dataclasses.replace(tests, imports="from typing import *\n")).outcome == "pass"
    script = "if __name__ == '__main__':\n    print(sqrt(int(input())))\n"
    tests = stdio(("4\n", "2.0\n"))
    assert grade(script, tests).outcome == "runtime_error:NameError"
    assert grade(script, dataclasses.replace(tests, imports="from math import *\n")).outcome == "pass"


@needs_sandbox
def test_imports_keep_the_solution_line_numbers():
    tests = Tests("stdio", (TestCase("", ""),), imports="import math\n" * 5)
    result = grade("x = 1\nraise ValueError('here')\n", tests)
    assert result.outcome == "runtime_error:ValueError" and 'solution.py", line 2' in result.results[0].stderr


@needs_sandbox
def test_tests_time_limit_replaces_the_callers():
    tests = stdio(("", "done\n"))
    slow = "import time\ntime.sleep(2)\nprint('done')\n"
    assert grade(slow, tests, limits=Limits(wall_seconds=1, cpu_seconds=1)).outcome == "timeout"
    assert grade(slow, dataclasses.replace(tests, seconds=4), limits=Limits(wall_seconds=1, cpu_seconds=1)).outcome == "pass"


@needs_sandbox
def test_long_correct_output_is_not_truncated():
    expected = "\n".join(str(i) for i in range(50_000)) + "\n"  # about 290 kB, over the default output cap
    assert grade("for i in range(50_000): print(i)", stdio(("", expected))).outcome == "pass"


# Cheats that beat naive graders.

@needs_sandbox
def test_claiming_success_does_not_pass():
    stdio_tests = stdio(("1 2\n", "3\n"))
    for cheat in ["print('pass')", "import sys; sys.exit(0)", "print('All tests passed'); raise SystemExit(0)", ""]:
        assert grade(cheat, stdio_tests).outcome == "wrong_answer", cheat
    for cheat in [
        "import os\ndef add(a, b): os._exit(0)",  # exit before the harness writes a result
        "def add(a, b): raise SystemExit(0)",
        "import json\njson.dumps = lambda *a, **k: '3'\ndef add(a, b): return None",  # patch the harness's encoder
    ]:
        assert grade(cheat, SUM).outcome != "pass", cheat


@needs_sandbox
def test_always_equal_objects_do_not_pass():
    """Defeats graders that compare in the program's process with `==`."""
    cheat = textwrap.dedent("""
        class AlwaysEqual(int):
            def __eq__(self, other): return True
            __hash__ = int.__hash__
        def add(a, b): return AlwaysEqual(0)
    """)
    assert grade(cheat, SUM).outcome == "wrong_answer"  # int subclass encodes as 0
    assert grade("class X:\n    __eq__ = lambda s, o: True\ndef add(a, b): return X()", SUM).outcome == "runtime_error:TypeError"


@needs_sandbox
def test_tampering_with_the_result_channel_only_hurts():
    for fd in (3, 4):  # 3 is the harness's result channel; 4 is not open
        cheat = f"import os\ndef add(a, b):\n    os.write({fd}, b'3')\n    return a + b"
        assert grade(cheat, SUM).outcome != "pass", fd
    for size in (60_000, 300_000):  # parsed by the host / truncated before parsing
        nesting = f"import os\ndef add(a, b):\n    os.write(3, b'[' * {size})\n    os._exit(0)"
        assert grade(nesting, SUM).outcome == "wrong_answer"  # and the host survived


@needs_sandbox
def test_expected_values_are_unreachable_from_the_program():
    secret = f"SECRET-{uuid.uuid4().hex}"
    tests = call("f", ([1], secret), ([2], secret))
    # The program knows only the secret's hash, so finding it means it leaked in.
    search = textwrap.dedent(f"""
        import gc, hashlib, os, re, sys
        DIGEST = {hashlib.sha256(secret.encode()).hexdigest()!r}
        PATTERN = re.compile(rb'SECRET-[0-9a-f]{{32}}')
        def leaked(data):
            return any(hashlib.sha256(m).hexdigest() == DIGEST for m in PATTERN.findall(data))
        def f(x):
            places = []
            for obj in gc.get_objects():
                items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else ()
                if any(isinstance(item, str) and leaked(item.encode(errors='ignore')) for item in items):
                    places.append('memory')
            for root, _, files in os.walk('/'):
                if root.startswith(('/proc', '/usr', '/dev')) and not root.startswith('/dev/shm'):  # ttys block
                    continue
                for name in files:
                    try:
                        if leaked(open(os.path.join(root, name), 'rb').read(1 << 20)):
                            places.append(os.path.join(root, name))
                    except OSError:
                        pass
            for pid in [p for p in os.listdir('/proc') if p.isdigit()]:
                for part in ('cmdline', 'environ'):
                    try:
                        if leaked(open(f'/proc/{{pid}}/{{part}}', 'rb').read()):
                            places.append(f'/proc/{{pid}}/{{part}}')
                    except OSError:
                        pass
            print('FOUND' if places else 'NOT FOUND', places, file=sys.stderr)
            return 'guess'
    """)
    result = grade(search, tests, limits=Limits(wall_seconds=30, cpu_seconds=30))
    assert result.outcome == "wrong_answer"
    assert all("NOT FOUND []" in r.stderr for r in result.results), [r.stderr[-300:] for r in result.results]
    control = grade(search, call("f", ([secret], "x")), limits=Limits(wall_seconds=30, cpu_seconds=30))
    assert "FOUND ['memory'" in control.results[0].stderr  # the search finds the secret when it is given as input


@needs_sandbox
def test_cases_cannot_see_each_other():
    tests = stdio(("first\n", "1\n"), ("second\n", "1\n"), ("third\n", "1\n"))
    solution = textwrap.dedent("""
        import os
        seen = os.listdir('/work') + os.listdir('/tmp')
        open('/work/' + input().strip(), 'w').write('x')
        print(len(seen) + 1)
    """)
    assert grade(solution, tests, workers=1).outcome == "pass"
