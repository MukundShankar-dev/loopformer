"""Convert a sample of DeepCoder's TACO and PrimeIntellect subsets into a training problem set.

    .venv-probe/bin/python -m scripts.codetasks.deepcoder --data ~/src/training-data/deepcoder --count 5000 \\
        --output data/agent/train-deepcoder-5k

Only stdin/stdout problems with string tests and at least MIN_CASES tests
are kept (DeepCoder's function-call problems are left out; KodCode covers
that style). A problem is kept only if one of its first SOLUTIONS_TRIED
reference solutions passes every test with our grader in the sandbox, and
its statement does not overlap an evaluation set
(scripts/codetasks/dedup.py). PrimeIntellect's fixed instruction prefix and
suffix are removed from statements. IDs are "<subset>-<row>" in the pinned
files' order. Candidates are shuffled with --seed and verified in that order
until --count problems are kept; the public test is the first test.
"""
import argparse
import json
import random
from pathlib import Path

from rich.console import Console

from scripts.codetasks.dedup import DEDUP_N, DEDUP_THRESHOLD, EvalIndex, eval_statements
from scripts.codetasks.problems import Problem, in_memory, write_problems
from scripts.sandbox.grader import TestCase, Tests, grade

REVISION = "177913a"
SUBSETS = ("taco", "primeintellect")
MIN_CASES = 3
SOLUTIONS_TRIED = 3
PRIMEINTELLECT_PREFIX = "Solve the following coding problem using the programming language python:\n\n"
PRIMEINTELLECT_SUFFIX = ("\n\nThe input will be stdin and you should print your solution to stdout\n\n\n"
                         "Now solve the problem and return the code.")


class Excluded(Exception):
    pass


def stdio_cases(tests_json: str) -> tuple[TestCase, ...]:
    tests = json.loads(tests_json)
    if isinstance(tests, dict):  # TACO: {"inputs": [...], "outputs": [...], optional "fn_name"}
        if tests.get("fn_name"):
            raise Excluded("function-call tests")
        pairs = list(zip(tests["inputs"], tests["outputs"], strict=True))
    else:  # PrimeIntellect: [{"type", "input", "output"}, ...]
        if any(test.get("type") != "stdin_stdout" for test in tests):
            raise Excluded("function-call tests")
        pairs = [(test["input"], test["output"]) for test in tests]
    if not all(isinstance(value, str) for pair in pairs for value in pair):
        raise Excluded("non-string tests")
    if len(pairs) < MIN_CASES:
        raise Excluded(f"{len(pairs)} tests")
    return tuple(TestCase(input, output) for input, output in pairs)


def clean_statement(subset: str, text: str) -> str:
    if subset == "primeintellect":
        text = text.removeprefix(PRIMEINTELLECT_PREFIX).removesuffix(PRIMEINTELLECT_SUFFIX)
    return text.strip()


def clean_solution(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text


def convert(subset: str, row_number: int, row: dict, overlap: float, source: str, workers: int) -> Problem:
    cases = stdio_cases(row["tests"])
    hidden = Tests("stdio", cases)
    for solution in list(row["solutions"])[:SOLUTIONS_TRIED]:
        if grade(clean_solution(solution), hidden, workers=workers).outcome == "pass":
            break
    else:
        raise Excluded(f"none of the first {SOLUTIONS_TRIED} solutions passes in our sandbox")
    return Problem(id=f"{subset}-{row_number}", source=source, statement=clean_statement(subset, row["problem"]),
                   public=Tests("stdio", cases[:1]), hidden=in_memory(hidden),
                   metadata={"subset": subset, "eval_overlap": overlap})


def read_rows(data: Path, wanted: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """The rows at these (subset, row number) positions, each file read once, keeping the first solutions only."""
    import pyarrow.parquet as pq  # only in the probe environment

    rows, want = {}, set(wanted)
    for subset in {subset for subset, _ in wanted}:
        row_number = 0
        for path in sorted((data / subset).glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):  # bounded memory: tests reach 69 MB
                for row in batch.to_pylist():
                    if (subset, row_number) in want:
                        rows[(subset, row_number)] = {**row, "solutions": list(row["solutions"])[:SOLUTIONS_TRIED]}
                    row_number += 1
    return rows


def main() -> None:
    import pyarrow.parquet as pq  # only in the probe environment

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="DeepCoder snapshot with taco/ and primeintellect/")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-sets", type=Path, nargs="+", required=True, help="problem sets to deduplicate against")
    parser.add_argument("--confirmation", type=Path, default=Path("configs/confirmation_livecodebench.json"))
    parser.add_argument("--livecodebench-raw", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, help="parallel test cases per problem")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new output")
    console = Console()
    index = EvalIndex(eval_statements(args.eval_sets, args.confirmation, args.livecodebench_raw), DEDUP_N)
    candidates, excluded = [], {}
    for subset in SUBSETS:
        row_number = 0
        for path in sorted((args.data / subset).glob("*.parquet")):
            for row in pq.read_table(path, columns=["problem"]).to_pylist():
                overlap, match = index.overlap(clean_statement(subset, row["problem"]))
                if overlap >= DEDUP_THRESHOLD:
                    excluded[f"{subset}-{row_number}"] = f"overlaps {match} ({overlap:.2f})"
                else:
                    candidates.append((subset, row_number, overlap))
                row_number += 1
    console.print(f"{len(candidates)} candidates; {len(excluded)} removed as overlapping an evaluation statement")
    random.Random(args.seed).shuffle(candidates)
    source = f"deepcoder-{REVISION}"

    def kept():
        count, position = 0, 0
        while count < args.count and position < len(candidates):
            chunk = candidates[position:position + 500]
            rows = read_rows(args.data, [(subset, row_number) for subset, row_number, _ in chunk])
            for subset, row_number, overlap in chunk:
                if count == args.count:
                    return
                try:
                    problem = convert(subset, row_number, rows[(subset, row_number)], overlap, source, args.workers)
                except Excluded as reason:
                    excluded[f"{subset}-{row_number}"] = str(reason)
                    continue
                count += 1
                yield problem
            position += len(chunk)
            console.print(f"{count} kept of {position} tried", markup=False)

    written = write_problems(args.output, kept())
    (args.output / "excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    if written < args.count:
        raise SystemExit(f"only {written} of {args.count} problems could be converted")
    console.print(f"{written} DeepCoder problems written to {args.output}")


if __name__ == "__main__":
    main()
