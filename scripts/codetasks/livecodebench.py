"""Convert prepared LiveCodeBench problems into a problem set.

    python -m scripts.codetasks.livecodebench --problems data/livecodebench/release_v6-0fe84c3-before-2024-11-12 \\
        --output data/agent/livecodebench-release_v6-0fe84c3-before-2024-11-12

Reads the output of scripts/codeprobe/livecodebench_data.py, whose hidden
tests are already decoded without unpickling classes. Test semantics follow
LiveCodeBench's evaluator: LeetCode problems (with `func_name`) are call tests
whose input is one JSON value per line and whose expected value is JSON;
AtCoder and Codeforces problems are stdin/stdout tests. Public tests are
LiveCodeBench's public examples; hidden tests are public plus private, the set
LiveCodeBench grades on. Every problem's tests carry LiveCodeBench's import
block (IMPORTS), which its evaluator runs before each solution: LeetCode
starter code uses `List` without importing it. Each test's time limit is
LiveCodeBench's default, SECONDS of wall time. The statement includes
LeetCode's starter code, which names the method to write. Problems without public examples are
excluded and listed in `excluded.json`, because the agent needs one.
"""
import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console

from scripts.codetasks.problems import Problem, write_problems
from scripts.sandbox.grader import TestCase, Tests
from scripts.sandbox.values import encode

# lcb_runner/evaluation/testing_util.py `import_string` at the pinned commit (configs/probe_headroom.json).
IMPORTS = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
           "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\n"
           "from statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\n"
           "from io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\n"
           "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\n"
           "import copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\n"
           "import operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n")
SECONDS = 6  # lcb_runner/runner/parser.py --timeout default: a per-test wall-clock alarm
METADATA = ("question_title", "platform", "contest_id", "contest_date", "difficulty", "starter_code")


def to_cases(tests: list[dict], call: bool) -> tuple[TestCase, ...]:
    kinds = {test["testtype"] for test in tests}
    if kinds - {"functional" if call else "stdin"}:
        raise ValueError(f"unexpected test types {sorted(kinds)}")
    if not call:
        return tuple(TestCase(test["input"], test["output"]) for test in tests)
    return tuple(TestCase(json.dumps(encode([json.loads(line) for line in test["input"].split("\n")])),
                          json.dumps(encode(json.loads(test["output"])))) for test in tests)


def statement(record: dict) -> str:
    if not record["starter_code"]:
        return record["question_content"].strip()
    return f"{record['question_content'].strip()}\n\nStarter code:\n```python\n{record['starter_code'].strip()}\n```"


def convert(record: dict, tests_path: Path, source: str) -> Problem:
    function = json.loads(record["metadata"]).get("func_name")
    kind = "call" if function else "stdio"
    public = to_cases(json.loads(record["public_test_cases"]), bool(function))
    load = lambda: Tests(kind, public + to_cases(json.loads(tests_path.read_text()), bool(function)),
                         function=function, imports=IMPORTS, seconds=SECONDS)
    return Problem(id=record["question_id"], source=source, statement=statement(record),
                   public=Tests(kind, public, function=function, imports=IMPORTS, seconds=SECONDS), hidden=load,
                   metadata={key: record[key] for key in METADATA})


def problems(directory: Path, source: str, excluded: dict[str, str]) -> Iterator[Problem]:
    """Converted problems; records each excluded problem's reason in `excluded`."""
    for line in (directory / "problems.jsonl").open():
        record = json.loads(line)
        if not json.loads(record["public_test_cases"]):
            excluded[record["question_id"]] = "no public examples"
            continue
        yield convert(record, directory / "tests" / f"{record['question_id']}.json", source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", type=Path, required=True, help="output of scripts.codeprobe.livecodebench_data")
    parser.add_argument("--output", type=Path, required=True, help="problem-set directory to create")
    args = parser.parse_args()
    source = f"livecodebench-{args.problems.name}"
    excluded = {}
    count = write_problems(args.output, problems(args.problems, source, excluded))
    (args.output / "excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    Console().print(f"{count} LiveCodeBench problems written to {args.output}; excluded {excluded}")


if __name__ == "__main__":
    main()
