"""Grade EvalPlus samples with our sandbox grader and compare with EvalPlus's verdicts.

M0b's harness gate needs our grader to agree with the official tool. This
grades each sample's `solution` against the problem's hidden tests and, given
an EvalPlus eval_results.json for the same samples, compares per sample with
EvalPlus's plus verdict (original and extra tests both pass). Samples of tasks
excluded from our problem set are skipped and counted.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console

from scripts.codetasks.problems import read_problems
from scripts.sandbox.grader import grade


def read_samples(path: Path) -> dict[str, list[str]]:
    samples = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        samples[row["task_id"]].append(row["solution"])
    return samples


def evalplus_verdicts(path: Path) -> dict[str, list[bool]]:
    data = json.loads(path.read_text())["eval"]
    return {task_id: [row["base_status"] == "pass" and row["plus_status"] == "pass" for row in rows]
            for task_id, rows in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True, help="jsonl of {task_id, solution}")
    parser.add_argument("--evalplus", type=Path, help="EvalPlus eval_results.json for the same samples")
    parser.add_argument("--output", type=Path, required=True, help="directory for per-sample rows and summary")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new directory")
    problems = {problem.id: problem for problem in read_problems(args.problems)}
    samples = read_samples(args.samples)
    official = evalplus_verdicts(args.evalplus) if args.evalplus else {}
    if official and set(official) != set(samples):
        parser.error("EvalPlus results and samples cover different tasks")
    args.output.mkdir(parents=True)
    rows, skipped = [], 0
    console = Console()
    with (args.output / "samples.jsonl").open("w") as handle:
        for index, (task_id, solutions) in enumerate(sorted(samples.items())):
            if task_id not in problems:
                skipped += len(solutions)
                continue
            for number, solution in enumerate(solutions):
                result = grade(solution, problems[task_id].hidden, workers=args.workers)
                row = {"task_id": task_id, "sample": number, "outcome": result.outcome,
                       "pass_fraction": result.pass_fraction, "ours": result.outcome == "pass",
                       "evalplus": official[task_id][number] if official else None}
                handle.write(json.dumps(row) + "\n")
                rows.append(row)
            if index % 25 == 0:
                console.print(f"{index + 1}/{len(samples)} tasks graded")
    summary = {"samples": len(rows), "skipped_excluded_tasks": skipped,
               "ours_pass_rate": sum(row["ours"] for row in rows) / len(rows),
               "outcomes": Counter(row["outcome"] for row in rows)}
    if official:
        pairs = Counter((row["ours"], row["evalplus"]) for row in rows)
        summary.update({
            "evalplus_pass_rate": sum(row["evalplus"] for row in rows) / len(rows),
            "agreement": (pairs[(True, True)] + pairs[(False, False)]) / len(rows),
            "both_pass": pairs[(True, True)], "both_fail": pairs[(False, False)],
            "ours_only": pairs[(True, False)], "evalplus_only": pairs[(False, True)],
            "disagreements": [f"{row['task_id']}#{row['sample']}" for row in rows if row["ours"] != row["evalplus"]],
        })
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.print({key: value for key, value in summary.items() if key != "disagreements"})


if __name__ == "__main__":
    main()
