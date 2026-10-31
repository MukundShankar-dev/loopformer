"""Coding problems for the v2 agent: a statement, public tests and hidden tests.

The agent sees `statement` and `public`; `hidden` decides success and must
never reach it. A problem set is a directory: `problems.jsonl` (one problem
per line, without hidden tests) and `hidden/<file>.jsonl` per problem (a
header line with the test kind, then one `[input, expected]` case per line).
Hidden tests load only when `Problem.hidden()` is called, because some
problems' tests reach hundreds of megabytes.
"""
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from scripts.sandbox.grader import TestCase, Tests


@dataclass(frozen=True)
class Problem:
    id: str
    source: str  # dataset and version, e.g. "mbppplus-v0.2.0"
    statement: str
    public: Tests
    hidden: Callable[[], Tests]  # loads the hidden tests
    metadata: dict = field(default_factory=dict)  # e.g. contest date, difficulty, platform


def _same(tests: Tests) -> Tests:
    return tests


def in_memory(tests: Tests) -> Callable[[], Tests]:
    return partial(_same, tests)  # picklable, so problems can go to grading processes


def _header(tests: Tests) -> dict:
    return {"kind": tests.kind, "function": tests.function, "atol": tests.atol, "imports": tests.imports,
            "seconds": tests.seconds}


def _tests_to_json(tests: Tests) -> dict:
    return {**_header(tests), "cases": [[case.input, case.expected] for case in tests.cases]}


def _tests_from_json(data: dict) -> Tests:
    cases = tuple(TestCase(input, expected) for input, expected in data["cases"])
    return Tests(data["kind"], cases, function=data["function"], atol=data["atol"], imports=data.get("imports", ""),
                 seconds=data.get("seconds"))


def _read_hidden(path: Path) -> Tests:
    with path.open() as handle:
        header = json.loads(next(handle))
        cases = tuple(TestCase(*json.loads(line)) for line in handle)
    return Tests(header["kind"], cases, function=header["function"], atol=header["atol"],
                 imports=header.get("imports", ""), seconds=header.get("seconds"))  # older sets lack both


def hidden_file(problem_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", problem_id) + ".jsonl"


def write_problems(directory: Path, problems: Iterable[Problem]) -> int:
    """Write a problem set, loading one problem's hidden tests at a time; returns the count."""
    if directory.exists():
        raise FileExistsError(f"{directory} exists; choose a new path")
    (directory / "hidden").mkdir(parents=True)
    count = 0
    with (directory / "problems.jsonl").open("w") as handle:
        for problem in problems:
            path = directory / "hidden" / hidden_file(problem.id)
            if path.exists():
                raise ValueError(f"problem ID {problem.id!r} collides with another problem's hidden file")
            hidden = problem.hidden()
            with path.open("w") as tests:
                tests.write(json.dumps(_header(hidden)) + "\n")
                for case in hidden.cases:
                    tests.write(json.dumps([case.input, case.expected]) + "\n")
            handle.write(json.dumps({"id": problem.id, "source": problem.source, "statement": problem.statement,
                                     "public": _tests_to_json(problem.public), "hidden_cases": len(hidden.cases),
                                     "metadata": problem.metadata}) + "\n")
            count += 1
    return count


def read_problems(directory: Path) -> list[Problem]:
    problems = []
    for line in (directory / "problems.jsonl").read_text().splitlines():
        data = json.loads(line)
        problems.append(Problem(data["id"], data["source"], data["statement"], _tests_from_json(data["public"]),
                                partial(_read_hidden, directory / "hidden" / hidden_file(data["id"])),
                                data["metadata"]))
    return problems
