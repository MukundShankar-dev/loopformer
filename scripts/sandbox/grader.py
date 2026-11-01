"""Grade a submitted program against tests, deciding pass or fail on the host.

Each stdio test case is a separate sandbox run on a fresh copy of the
submission, given only that case's input. Call tests run in batches: one
sandbox run loads the submission once and calls it on up to BATCH_CASES
inputs in turn, as EvalPlus and LiveCodeBench run all of a solution's tests in
one process. Only a value a call returned is decided from a batch; every other
case (an exception, a crash, output that is malformed or out of order, or
cases left when the batch hit its limits) is re-run alone, and so is every
passing case of a batch that did not exit cleanly, since alone such a case
fails after returning (a thread that never ends, a crashing exit handler). A
whole batch gets one case's time limit; as in those harnesses, state a case
warms (a cache) carries to later cases in its batch.
Expected outputs stay on the host, so a submission cannot pass by exiting
early, tampering with the harness, printing "pass" or reading the answers:
whatever it emits is compared here against a value it never saw.

Failure classes are parsed from the program's stderr, which the program
controls. They are diagnostics, never evidence of a pass.

Two kinds of tests:
- `stdio`: input is stdin; expected is stdout, compared line by line on
  whitespace-separated tokens, with numeric tokens compared as decimals
  (modelled on LiveCodeBench's checker).
- `call`: input is a JSON array of arguments to `function`; expected is the
  return value. Both use the tagged JSON of scripts/sandbox/values.py, so
  tuples and sets survive. Values are compared structurally, with two floats
  equal within `atol` or a relative 1e-7 (as EvalPlus does).

`imports` is code run in the solution's namespace before it, as
LiveCodeBench does with its fixed import block; the solution file itself is
unchanged, so tracebacks keep its line numbers. `seconds`, when set, is the
per-test time limit (wall and CPU) in place of the caller's.
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
VALUES = (Path(__file__).parent / "values.py").read_text()
FLOAT_RELATIVE_TOLERANCE = 1e-7
SYNTAX_CHECK = """\
import sys
try:
    compile(open('/code/solution.py', encoding='utf-8').read(), 'solution.py', 'exec')
except (SyntaxError, ValueError) as error:
    sys.exit(f'{type(error).__name__}: {error}')
"""
IMPORTS_LAUNCHER = """\
import runpy
namespace = {}
exec(compile(open('/code/imports.py', encoding='utf-8').read(), 'imports.py', 'exec'), namespace)
namespace.pop('__builtins__', None)
runpy.run_path('/code/ENTRY', init_globals=namespace, run_name='__main__')
"""
TEST_LIMITS = Limits(wall_seconds=4, cpu_seconds=4)
BATCH_CASES = 64
BATCH_INPUT_BYTES = 1 << 20  # of JSON arguments per batch
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
    imports: str = ""  # run in the solution's namespace first
    seconds: float | None = None  # per-test time limit; None keeps the caller's limits

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
    pass_fraction: float  # of all cases; cases not run (after `first_failure` stopped) count as failed
    results: tuple[TestResult, ...]  # of the cases run, in order


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
    """Python `==` on encoded values (as EvalPlus checks), with float tolerance.
    Numbers compare by value, so `True == 1` as in Python; tagged tuples and
    lists differ, as they do under `==`. Tolerance applies only between two
    floats: EvalPlus requires matching types when values are not `==`, so an
    int is not close to a float (an exact int against a reference's rounded
    float fails there too). Recurses along `expected`, so the depth of an
    adversarial `actual` cannot exhaust the host's stack."""
    if type(expected) is float and type(actual) is float:
        return math.isclose(expected, actual, rel_tol=FLOAT_RELATIVE_TOLERANCE, abs_tol=atol) or expected == actual
    if isinstance(expected, (bool, int, float)) and isinstance(actual, (bool, int, float)):
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


def case_limits(limits: Limits, tests: Tests, case: TestCase) -> Limits:
    """The tests' own time limit, if any, and room for an output twice the expected size, so long correct outputs
    are never truncated."""
    if tests.seconds is not None:
        limits = dataclasses.replace(limits, wall_seconds=tests.seconds, cpu_seconds=tests.seconds)
    output = max(limits.output_bytes, 2 * len(case.expected.encode()) + 1024)
    return dataclasses.replace(limits, output_bytes=output, file_bytes=max(limits.file_bytes, 2 * output))


def with_imports(files: dict[str, str], entry: str, imports: str) -> tuple[dict[str, str], str]:
    """Files and entry point that run `imports` in `entry`'s namespace before `entry`."""
    if not imports:
        return files, entry
    launcher = IMPORTS_LAUNCHER.replace("ENTRY", entry)
    return {**files, "imports.py": imports, "launch_with_imports.py": launcher}, "launch_with_imports.py"


def call_outcome(tests: Tests, case: TestCase, value) -> str:
    return "pass" if same_value(json.loads(case.expected), value, tests.atol) else "wrong_answer"


def batches(cases: tuple[TestCase, ...]) -> list[tuple[TestCase, ...]]:
    """Consecutive runs of at most BATCH_CASES cases and BATCH_INPUT_BYTES of input (a larger case goes alone)."""
    groups, current, size = [], [], 0
    for case in cases:
        if current and (len(current) == BATCH_CASES or size + len(case.input) > BATCH_INPUT_BYTES):
            groups.append(tuple(current))
            current, size = [], 0
        current.append(case)
        size += len(case.input)
    return groups + [tuple(current)]


def run_batch(solution: str, tests: Tests, cases: tuple[TestCase, ...], limits: Limits) -> list[TestResult | None]:
    """One sandbox run calling the solution on each case; a result for each case that returned a value in order,
    None for the rest. Output past a case's own output limit fails it, as it would alone. Each decided case carries
    the batch's stderr, which no single case owns."""
    alone = [case_limits(limits, tests, case) for case in cases]
    output = sum(case.output_bytes + 64 for case in alone)
    limits = dataclasses.replace(alone[0], output_bytes=output, file_bytes=max(limits.file_bytes, 2 * output))
    request = json.dumps({"function": tests.function, "batch": [json.loads(case.input) for case in cases],
                          "imports": tests.imports})
    result = run({"solution.py": solution, "harness.py": HARNESS, "values.py": VALUES}, entry="harness.py",
                 stdin=request, limits=limits)
    decided: list[TestResult | None] = [None] * len(cases)
    for number, line in enumerate(result.stdout.split("\n")[:-1][:len(cases)]):  # complete lines only
        parsed, frame = parse_value(line)
        if not (parsed and isinstance(frame, dict) and type(frame.get("case")) is int and frame["case"] == number):
            break  # malformed or forged: this and later cases are re-run alone
        if frame.keys() != {"case", "value"}:
            continue  # the call raised
        text = json.dumps(frame["value"])
        too_long = len(text.encode()) > alone[number].output_bytes
        outcome = "wrong_answer" if too_long else call_outcome(tests, cases[number], frame["value"])
        decided[number] = TestResult(outcome, result.wall_seconds / len(cases), text[:DETAIL_CHARACTERS],
                                     result.stderr[-DETAIL_CHARACTERS:])
    if failure(result) is not None or result.stdout_truncated:
        decided = [r if r is not None and r.outcome != "pass" else None for r in decided]
    return decided


def run_case(solution: str, tests: Tests, case: TestCase, limits: Limits) -> TestResult:
    limits = case_limits(limits, tests, case)
    if tests.kind == "stdio":
        files, entry = with_imports({"solution.py": solution}, "solution.py", tests.imports)
        result = run(files, entry=entry, stdin=case.input, limits=limits)
    else:
        request = json.dumps({"function": tests.function, "arguments": json.loads(case.input),
                              "imports": tests.imports})
        files = {"solution.py": solution, "harness.py": HARNESS, "values.py": VALUES}
        result = run(files, entry="harness.py", stdin=request, limits=limits)
    outcome = failure(result)
    if outcome is None:
        if result.stdout_truncated:
            outcome = "wrong_answer"
        elif tests.kind == "stdio":
            outcome = "pass" if same_stdout(case.expected, result.stdout) else "wrong_answer"
        else:
            parsed, value = parse_value(result.stdout)
            outcome = call_outcome(tests, case, value) if parsed else "wrong_answer"
    return TestResult(outcome, result.wall_seconds, result.stdout[:DETAIL_CHARACTERS], result.stderr[-DETAIL_CHARACTERS:])


def run_group(solution: str, tests: Tests, cases: tuple[TestCase, ...], limits: Limits,
              first_failure: bool) -> list[TestResult]:
    """A batch of call cases, then each undecided case alone; with `first_failure`, stops at the first failure."""
    decided = run_batch(solution, tests, cases, limits) if len(cases) > 1 else [None]
    results = []
    for case, result in zip(cases, decided):
        results.append(result or run_case(solution, tests, case, limits))
        if first_failure and results[-1].outcome != "pass":
            break
    return results


def grade(solution: str, tests: Tests, limits: Limits = TEST_LIMITS, workers: int = 1,
          first_failure: bool = False) -> Grade:
    """Compile-check once, then run the cases (call tests in batches), `workers` sandbox runs at a time. With
    `first_failure`, grading stops after a round of `workers` runs with a failure: enough for a pass/fail reward,
    much cheaper on the many submissions that fail early."""
    check = run({"solution.py": solution, "check.py": SYNTAX_CHECK}, entry="check.py", limits=limits)
    if check.exit_code != 0:
        outcome = failure(check) if check.timed_out or check.oom_killed else "syntax_error"
        detail = TestResult(outcome, check.wall_seconds, "", check.stderr[-DETAIL_CHARACTERS:])
        return Grade(outcome, 0.0, (detail,) * len(tests.cases))
    groups = batches(tests.cases) if tests.kind == "call" else [(case,) for case in tests.cases]
    chunk = workers if first_failure else len(groups)
    results: list[TestResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(groups), chunk):
            round_groups = groups[start:start + chunk]
            for cases, group in zip(round_groups, pool.map(
                    lambda cases: run_group(solution, tests, cases, limits, first_failure), round_groups)):
                results += group
                if len(group) < len(cases):
                    break  # stopped at a failure: later groups' results would leave a gap in `results`
            if first_failure and any(result.outcome != "pass" for result in results):
                break
    failed = [result.outcome for result in results if result.outcome != "pass"]
    passed = len(results) - len(failed)
    return Grade(failed[0] if failed else "pass", passed / len(tests.cases), tuple(results))
