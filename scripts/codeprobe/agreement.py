"""Grade benchmark samples with our sandbox grader and compare with the official tool's verdicts.

M0b's harness gate needs our grader to agree with the official tools. This
grades each sample against the problem's hidden tests and compares per sample
with the official verdict:
- EvalPlus: `--samples` (codegen .jsonl) and optionally `--evalplus`
  (eval_results.json for the same samples); the verdict is "plus" (original
  and extra tests both pass).
- LiveCodeBench: `--livecodebench`, an `eval_all.json` from
  scripts/codeprobe/evaluate_livecodebench.py, which holds both the code and
  the verdicts.
Samples of problems excluded from our problem set are skipped and counted.
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


def livecodebench_samples(path: Path) -> tuple[dict[str, list[str]], dict[str, list[bool]]]:
    rows = json.loads(path.read_text())
    return ({row["question_id"]: row["code_list"] for row in rows},
            {row["question_id"]: [bool(verdict) for verdict in row["graded_list"]] for row in rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", type=Path, required=True, help="problem-set directory")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--samples", type=Path, help="EvalPlus codegen jsonl of {task_id, solution}")
    sources.add_argument("--livecodebench", type=Path, help="eval_all.json with code_list and graded_list")
    parser.add_argument("--evalplus", type=Path, help="with --samples: EvalPlus eval_results.json for them")
    parser.add_argument("--output", type=Path, required=True, help="directory for per-sample rows and summary")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new directory")
    if args.evalplus and not args.samples:
        parser.error("--evalplus goes with --samples")
    if args.livecodebench:
        samples, official = livecodebench_samples(args.livecodebench)
    else:
        samples = read_samples(args.samples)
        official = evalplus_verdicts(args.evalplus) if args.evalplus else {}
    if official and set(official) != set(samples):
        parser.error("official results and samples cover different problems")
    problems = {problem.id: problem for problem in read_problems(args.problems)}
    args.output.mkdir(parents=True)
    rows, skipped = [], 0
    console = Console()
    with (args.output / "samples.jsonl").open("w") as handle:
        for index, (task_id, solutions) in enumerate(sorted(samples.items())):
            if task_id not in problems:
                skipped += len(solutions)
                continue
            hidden = problems[task_id].hidden()
            for number, solution in enumerate(solutions):
                result = grade(solution, hidden, workers=args.workers)
                row = {"task_id": task_id, "sample": number, "outcome": result.outcome,
                       "pass_fraction": result.pass_fraction, "ours": result.outcome == "pass",
                       "official": official[task_id][number] if official else None}
                handle.write(json.dumps(row) + "\n")
                rows.append(row)
            if index % 25 == 0:
                console.print(f"{index + 1}/{len(samples)} problems graded")
    summary = {"samples": len(rows), "skipped_excluded_problems": skipped,
               "ours_pass_rate": sum(row["ours"] for row in rows) / len(rows),
               "outcomes": Counter(row["outcome"] for row in rows)}
    if official:
        pairs = Counter((row["ours"], row["official"]) for row in rows)
        summary.update({
            "official_pass_rate": sum(row["official"] for row in rows) / len(rows),
            "agreement": (pairs[(True, True)] + pairs[(False, False)]) / len(rows),
            "both_pass": pairs[(True, True)], "both_fail": pairs[(False, False)],
            "ours_only": pairs[(True, False)], "official_only": pairs[(False, True)],
            "disagreements": [f"{row['task_id']}#{row['sample']}" for row in rows if row["ours"] != row["official"]],
        })
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.print({key: value for key, value in summary.items() if key != "disagreements"})


if __name__ == "__main__":
    main()
