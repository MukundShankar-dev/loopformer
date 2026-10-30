"""Grade a submitted program against tests, deciding pass or fail on the host.

Each test case is a separate sandbox run on a fresh copy of the submission,
given only that case's input. Expected outputs stay on the host, so a
submission cannot pass by exiting early, tampering with the harness, printing
"pass" or reading the answers: whatever it emits is compared here against a
value it never saw.

Failure classes are parsed from the program's stderr, which the program
controls. They are diagnostics, never evidence of a pass.

Two kinds of tests:
- `stdio`: input is stdin; expected is stdout, compared line by line on
  whitespace-separated tokens, with numeric tokens compared as decimals
  (modelled on LiveCodeBench's checker).
- `call`: input is a JSON array of arguments to `function`; expected is the
  JSON return value, compared structurally, with floats within `atol`.
"""
import dataclasses
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from scripts.sandbox.runner import Limits, RunResult, run

HARNESS = (Path(__file__).parent / "call_harness.py").read_text()
SYNTAX_CHECK = """\
import sys
try:
    compile(open('/code/solution.py', encoding='utf-8').read(), 'solution.py', 'exec')
except (SyntaxError, ValueError) as error:
    sys.exit(f'{type(error).__name__}: {error}')
"""
TEST_LIMITS = Limits(wall_seconds=4, cpu_seconds=4)
EXCEPTION_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]{0,63})(?::|$)")
DETAIL_CHARACTERS = 2000
SIGNAL_CPU = 24


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # not a pytest class
    input: str  # stdio: stdin text; call: JSON array of arguments
    expected: str  # stdio: expected stdout; call: JSON of the expected return value


@dataclass(frozen=True)
class Tests:
    __test__ = False  # not a pytest class
    kind: Literal["stdio", "call"]
    cases: tuple[TestCase, ...]
    function: str | None = None  # call only
    atol: float = 0.0  # call only: absolute tolerance for floats

    def __post_init__(self) -> None:
        if self.kind not in ("stdio", "call"):
            raise ValueError(f"unknown test kind {self.kind!r}")
        if (self.kind == "call") != bool(self.function):
            raise ValueError("call tests need a function name; stdio tests take none")
        if not self.cases:
            raise ValueError("no test cases")
        if self.kind == "call":
            for case in self.cases:
                if not isinstance(json.loads(case.input), list):
                    raise ValueError(f"call input must be a JSON array: {case.input[:80]!r}")
                json.loads(case.expected)


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest class
    """`stdout` and `stderr` are untrusted and truncated. Results of hidden
    tests must not be shown to the agent."""

    outcome: str  # pass / wrong_answer / timeout / memory / import_error / runtime_error:<type> / syntax_error
    wall_seconds: float
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Grade:
    outcome: str  # pass if every test passed, else the first failing test's outcome
    pass_fraction: float
    results: tuple[TestResult, ...]


def exception_type(stderr: str) -> str | None:
    """The exception named on the last line of a traceback, if any."""
    lines = [line for line in stderr.splitlines() if line.strip()]
    if not lines or "Traceback (most recent call last):" not in stderr:
        return None
    match = EXCEPTION_LINE.match(lines[-1])
    return match.group(1).rsplit(".", 1)[-1] if match else None


def failure(result: RunResult) -> str | None:
    """Outcome class of a run that did not exit cleanly; None if it did."""
    if result.timed_out or result.exit_code == 128 + SIGNAL_CPU:
        return "timeout"
    name = exception_type(result.stderr)
    if result.oom_killed or name == "MemoryError":
        return "memory"
    if result.exit_code == 0:
        return None
    if name in ("ImportError", "ModuleNotFoundError"):
        return "import_error"
    if name:
        return f"runtime_error:{name}"
    if result.exit_code > 128:
        return f"runtime_error:signal{result.exit_code - 128}"
    return "runtime_error:exit"


def decimal(token: str) -> Decimal | None:
    try:
        value = Decimal(token)
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def same_stdout(expected: str, actual: str) -> bool:
    expected_lines = [line.split() for line in expected.strip().splitlines()]
    actual_lines = [line.split() for line in actual.strip().splitlines()]
    if len(expected_lines) != len(actual_lines):
        return False
    for want_line, got_line in zip(expected_lines, actual_lines):
        if len(want_line) != len(got_line):
            return False
        for want, got in zip(want_line, got_line):
            if want != got and (decimal(want) is None or decimal(want) != decimal(got)):
                return False
    return True


def same_value(expected, actual, atol: float) -> bool:
    """Structural equality that recurses along `expected`, so the depth of an
    adversarial `actual` cannot exhaust the host's stack."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, float) or isinstance(actual, float):
            return math.isclose(expected, actual, rel_tol=0.0, abs_tol=atol) or expected == actual
        return expected == actual
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(same_value(e, a, atol) for e, a in zip(expected, actual)))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(same_value(expected[key], actual[key], atol) for key in expected))
    return type(expected) is type(actual) and expected == actual


def parse_value(text: str) -> tuple[bool, object]:
    """(parsed, value) for a program's JSON output; (False, None) for anything malformed."""
    try:
        return True, json.loads(text)
    except (ValueError, RecursionError):  # bad JSON, oversized integers, adversarial nesting
        return False, None


def case_limits(limits: Limits, case: TestCase) -> Limits:
    """Room for an output twice the expected size, so long correct outputs are never truncated."""
    output = max(limits.output_bytes, 2 * len(case.expected.encode()) + 1024)
    return dataclasses.replace(limits, output_bytes=output, file_bytes=max(limits.file_bytes, 2 * output))


def run_case(solution: str, tests: Tests, case: TestCase, limits: Limits) -> TestResult:
    limits = case_limits(limits, case)
    if tests.kind == "stdio":
        result = run({"solution.py": solution}, entry="solution.py", stdin=case.input, limits=limits)
    else:
        request = json.dumps({"function": tests.function, "arguments": json.loads(case.input)})
        result = run({"solution.py": solution, "harness.py": HARNESS}, entry="harness.py", stdin=request, limits=limits)
    outcome = failure(result)
    if outcome is None:
        if result.stdout_truncated:
            outcome = "wrong_answer"
        elif tests.kind == "stdio":
            outcome = "pass" if same_stdout(case.expected, result.stdout) else "wrong_answer"
        else:
            parsed, value = parse_value(result.stdout)
            matched = parsed and same_value(json.loads(case.expected), value, tests.atol)
            outcome = "pass" if matched else "wrong_answer"
    return TestResult(outcome, result.wall_seconds, result.stdout[:DETAIL_CHARACTERS], result.stderr[-DETAIL_CHARACTERS:])


def grade(solution: str, tests: Tests, limits: Limits = TEST_LIMITS, workers: int = 1) -> Grade:
    """Compile-check once, then run every case; `workers` sandbox runs at a time."""
    check = run({"solution.py": solution, "check.py": SYNTAX_CHECK}, entry="check.py", limits=limits)
    if check.exit_code != 0:
        outcome = failure(check) if check.timed_out or check.oom_killed else "syntax_error"
        detail = TestResult(outcome, check.wall_seconds, "", check.stderr[-DETAIL_CHARACTERS:])
        return Grade(outcome, 0.0, (detail,) * len(tests.cases))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = tuple(pool.map(lambda case: run_case(solution, tests, case, limits), tests.cases))
    failed = [result.outcome for result in results if result.outcome != "pass"]
    return Grade(failed[0] if failed else "pass", 1 - len(failed) / len(results), results)
