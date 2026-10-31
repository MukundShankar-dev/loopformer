"""Convert a sample of KodCode-V1 into a training problem set.

    .venv-probe/bin/python -m scripts.codetasks.kodcode --data ~/src/training-data/kodcode-v1 --count 15000 \\
        --output data/agent/train-kodcode-v1-15k

KodCode's tests are pytest functions whose asserts embed expected values. This
keeps only the call inputs: every call to the problem's function with literal
positional arguments inside an `assert`. Expected values come from running
KodCode's reference solution on those inputs in the sandbox, twice, so
nondeterministic references are excluded. Kept: GPT-4o difficulty easy or
medium, one function, function style (not `online_judge`), not the `Package`
or `Docs` subsets (third-party libraries, documentation lookup), at least
MIN_CASES distinct inputs, KodCode benchmark similarity below
MAX_BENCHMARK_SIMILARITY, and no statement overlap with an evaluation set
(scripts/codetasks/dedup.py). Candidates are shuffled with --seed and
converted in that order until --count problems are kept; the public test is
the first input.
"""
import argparse
import ast
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console

from scripts.codetasks.dedup import DEDUP_N, DEDUP_THRESHOLD, EvalIndex, eval_statements
from scripts.codetasks.problems import Problem, in_memory, write_problems
from scripts.codetasks.reference import FLOAT_ATOL, ReferenceFailed, is_floats, reference_outputs
from scripts.sandbox.grader import TestCase, Tests
from scripts.sandbox.runner import Limits
from scripts.sandbox.values import decode, encode

REVISION = "97d81f0"
COLUMNS = ("question_id", "question", "solution", "test", "test_info", "style", "subset", "gpt_difficulty",
           "benchmark_similarity")
MIN_CASES = 3
MAX_BENCHMARK_SIMILARITY = 0.9
LIMITS = Limits(wall_seconds=60, cpu_seconds=60, memory_bytes=1 << 30, output_bytes=64 << 20, file_bytes=64 << 20)


class Excluded(Exception):
    pass


def call_inputs(test_source: str, function: str) -> list[list]:
    """Distinct literal positional-argument lists of calls to `function` inside asserts, in order of appearance."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError as error:
        raise Excluded(f"test does not parse: {error}") from None
    seen, inputs = set(), []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for call in ast.walk(node.test):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == function
                    and not call.keywords and not any(isinstance(argument, ast.Starred) for argument in call.args)):
                try:
                    arguments = [ast.literal_eval(argument) for argument in call.args]
                except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                    continue
                key = repr(arguments)
                if key not in seen:
                    seen.add(key)
                    inputs.append(arguments)
    return inputs


def statement(row: dict, declaration: str) -> str:
    if row["style"] == "complete":  # the question is already a signature and docstring
        return row["question"].strip()
    return f"{row['question'].strip()}\n\nFunction signature:\n```python\n{declaration.strip()}\n```"


def candidate(row: dict) -> str | None:
    """Why a row is not a candidate, or None."""
    if row["gpt_difficulty"] not in ("easy", "medium"):
        return "difficulty"
    if row["style"] == "online_judge" or row["subset"] in ("Package", "Docs"):
        return "style or subset"
    if len(row["test_info"] or []) != 1:
        return "not one function"
    if row["benchmark_similarity"] is None or row["benchmark_similarity"] >= MAX_BENCHMARK_SIMILARITY:
        return "KodCode benchmark similarity"
    return None


def convert(row: dict, overlap: float, source: str) -> Problem:
    info = row["test_info"][0]
    function = info["function_name"]
    inputs = call_inputs(row["test"], function)
    if len(inputs) < MIN_CASES:
        raise Excluded(f"{len(inputs)} literal inputs")
    try:
        encode(inputs)
    except TypeError as error:  # e.g. bytes literals, which tagged JSON does not carry
        raise Excluded(f"input: {error}") from None
    try:
        outputs = reference_outputs(row["solution"], function, inputs, limits=LIMITS, repeats=2)
    except ReferenceFailed as error:
        raise Excluded(f"reference: {str(error)[:200]}") from None
    if any(isinstance(output, dict) and "__error__" in output for output in outputs):
        raise Excluded("reference raised on an assert input")
    atol = FLOAT_ATOL if any(is_floats(decode(output)) for output in outputs) else 0.0
    cases = tuple(TestCase(json.dumps(encode(arguments)), json.dumps(output)) for arguments, output in zip(inputs, outputs))
    return Problem(id=row["question_id"], source=source, statement=statement(row, info["function_declaration"]),
                   public=Tests("call", cases[:1], function=function, atol=atol),
                   hidden=in_memory(Tests("call", cases, function=function, atol=atol)),
                   metadata={key: row[key] for key in ("subset", "style", "gpt_difficulty", "benchmark_similarity")}
                   | {"eval_overlap": overlap})


def main() -> None:
    import pyarrow.parquet as pq  # only in the probe environment

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="KodCode-V1 snapshot with data/train-*.parquet")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-sets", type=Path, nargs="+", required=True, help="problem sets to deduplicate against")
    parser.add_argument("--confirmation", type=Path, default=Path("configs/confirmation_livecodebench.json"))
    parser.add_argument("--livecodebench-raw", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new output")
    sys.set_int_max_str_digits(0)  # decoding trusted reference outputs, which may be huge integers
    console = Console()
    index = EvalIndex(eval_statements(args.eval_sets, args.confirmation, args.livecodebench_raw), DEDUP_N)
    rows, excluded = [], {}
    for path in sorted((args.data / "data").glob("train-*.parquet")):
        for row in pq.read_table(path, columns=list(COLUMNS)).to_pylist():
            reason = candidate(row)
            if reason is None:
                overlap, match = index.overlap(row["question"])
                if overlap >= DEDUP_THRESHOLD:
                    excluded[row["question_id"]] = f"overlaps {match} ({overlap:.2f})"
                else:
                    rows.append((row, overlap))
    console.print(f"{len(rows)} candidates; {len(excluded)} removed as overlapping an evaluation statement")
    random.Random(args.seed).shuffle(rows)
    source = f"kodcode-v1-{REVISION}"

    def attempt(item: tuple[dict, float]) -> Problem | str:
        try:
            return convert(item[0], item[1], source)
        except Excluded as reason:
            return str(reason)

    def kept():
        count, position = 0, 0
        with ThreadPoolExecutor(args.workers) as pool:
            while count < args.count and position < len(rows):
                batch = rows[position:position + 4 * args.workers]
                for (row, _), result in zip(batch, pool.map(attempt, batch)):
                    if isinstance(result, str):
                        excluded[row["question_id"]] = result
                    elif count < args.count:
                        count += 1
                        yield result
                position += len(batch)
                if position % (64 * args.workers) == 0:
                    console.print(f"{count} kept of {position} tried", markup=False)

    written = write_problems(args.output, kept())
    (args.output / "excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    if written < args.count:
        raise SystemExit(f"only {written} of {args.count} problems could be converted")
    console.print(f"{written} KodCode problems written to {args.output}")


if __name__ == "__main__":
    main()
