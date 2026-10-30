"""Coding problems for the v2 agent: a statement, public tests and hidden tests.

The agent sees `statement` and `public`; `hidden` decides success and must
never reach it. Stored one JSON object per line.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.sandbox.grader import TestCase, Tests


@dataclass(frozen=True)
class Problem:
    id: str
    source: str  # dataset and version, e.g. "mbppplus-v0.2.0"
    statement: str
    public: Tests
    hidden: Tests


def _tests_to_json(tests: Tests) -> dict:
    return {"kind": tests.kind, "function": tests.function, "atol": tests.atol,
            "cases": [[case.input, case.expected] for case in tests.cases]}


def _tests_from_json(data: dict) -> Tests:
    cases = tuple(TestCase(input, expected) for input, expected in data["cases"])
    return Tests(data["kind"], cases, function=data["function"], atol=data["atol"])


def write_problems(path: Path, problems: list[Problem]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for problem in problems:
            handle.write(json.dumps({"id": problem.id, "source": problem.source, "statement": problem.statement,
                                     "public": _tests_to_json(problem.public),
                                     "hidden": _tests_to_json(problem.hidden)}) + "\n")


def read_problems(path: Path) -> list[Problem]:
    problems = []
    for line in path.read_text().splitlines():
        data = json.loads(line)
        problems.append(Problem(data["id"], data["source"], data["statement"],
                                _tests_from_json(data["public"]), _tests_from_json(data["hidden"])))
    return problems
