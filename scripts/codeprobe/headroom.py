"""Headroom summary for code benchmarks from official per-sample grading output.

RL with execution rewards mostly turns sometimes-solved problems into reliably
solved ones; problems never solved in n samples give no learning signal. This
module reads per-sample pass/fail from EvalPlus and LiveCodeBench result files
and reports, per benchmark, the always/sometimes/never-solved fractions and
unbiased pass@k (Chen et al., 2021).

The readers fail loudly if a result file does not have the expected layout.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"Need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    # Product form avoids huge binomials.
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def summarize(results: dict[str, list[bool]], ks: tuple[int, ...] = (1, 4, 16)) -> dict:
    """Headroom for one benchmark; results map problem ID -> per-sample passes.

    Every problem must have the same sample count n, so pass@k is comparable.
    """
    if not results:
        raise ValueError("No problems to summarize")
    counts = {len(samples) for samples in results.values()}
    if len(counts) != 1:
        raise ValueError(f"Problems have unequal sample counts: {sorted(counts)}")
    n = counts.pop()
    solved = np.array([sum(samples) for samples in results.values()])
    return {
        "problems": len(results),
        "samples_per_problem": n,
        "always_solved": float(np.mean(solved == n)),
        "sometimes_solved": float(np.mean((solved > 0) & (solved < n))),
        "never_solved": float(np.mean(solved == 0)),
        **{f"pass@{k}": float(np.mean([pass_at_k(n, int(c), k) for c in solved])) for k in ks if k <= n},
    }


def read_evalplus(path: Path, tests: str = "plus") -> dict[str, list[bool]]:
    """EvalPlus `*_eval_results.json`: {"eval": {task_id: [{"base_status", "plus_status"}, ...]}}.

    "plus" requires both the original and the extra tests to pass, matching
    EvalPlus's HumanEval+/MBPP+ score; "base" uses the original tests only.
    """
    if tests not in ("base", "plus"):
        raise ValueError("tests must be 'base' or 'plus'")
    data = json.loads(path.read_text())
    if "eval" not in data:
        raise ValueError(f"{path}: expected an EvalPlus eval_results file with an 'eval' key")
    results = {}
    for task_id, samples in data["eval"].items():
        passes = []
        for sample in samples:
            if "base_status" not in sample:
                raise ValueError(f"{path}: sample for {task_id} lacks base_status")
            ok = sample["base_status"] == "pass"
            if tests == "plus":
                ok = ok and sample.get("plus_status") == "pass"
            passes.append(ok)
        results[task_id] = passes
    return results


def read_livecodebench(path: Path, dates: tuple[str, str] | None = None) -> dict[str, dict[str, list[bool]]]:
    """LiveCodeBench `eval_all` list: problems with `graded_list`.

    Returns problems grouped by difficulty plus "all". `dates` (two
    YYYY-MM-DD, inclusive) keeps problems whose contest_date is in that window,
    for comparison with numbers published on a window.
    """
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a LiveCodeBench eval_all list")
    groups: dict[str, dict[str, list[bool]]] = defaultdict(dict)
    for problem in data:
        missing = {"question_id", "graded_list", "difficulty", "contest_date"} - set(problem)
        if missing:
            raise ValueError(f"{path}: problem lacks {sorted(missing)}")
        if dates is not None and not dates[0] <= str(problem["contest_date"])[:10] <= dates[1]:
            continue
        passes = [bool(x) for x in problem["graded_list"]]
        groups["all"][problem["question_id"]] = passes
        groups[problem["difficulty"]][problem["question_id"]] = passes
    if not groups:
        raise ValueError(f"{path}: no problems in {dates}")
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evalplus", type=Path, action="append", default=[],
                        help="NAME=PATH to an EvalPlus eval_results.json; repeatable")
    parser.add_argument("--livecodebench", type=Path, action="append", default=[],
                        help="NAME=PATH to a LiveCodeBench eval_all.json; repeatable")
    parser.add_argument("--dates", help="LiveCodeBench contest-date window START:END (YYYY-MM-DD, inclusive)")
    parser.add_argument("--output", type=Path, required=True, help="summary JSON to write")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    dates = tuple(args.dates.split(":")) if args.dates else None
    if dates is not None and len(dates) != 2:
        parser.error("--dates takes START:END")
    summary = {}
    for spec in args.evalplus:
        name, path = str(spec).split("=", 1)
        summary[name] = summarize(read_evalplus(Path(path)))
    for spec in args.livecodebench:
        name, path = str(spec).split("=", 1)
        for group, results in read_livecodebench(Path(path), dates).items():
            summary[f"{name}/{group}"] = summarize(results)
    if not summary:
        parser.error("Give at least one --evalplus or --livecodebench result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"livecodebench_dates": dates, "benchmarks": summary}, indent=2) + "\n")
    table = Table(title="Headroom (fractions of problems)")
    for column in ("benchmark", "n", "problems", "always", "sometimes", "never", "pass@1", "pass@16"):
        table.add_column(column)
    for name, row in summary.items():
        table.add_row(name, str(row["samples_per_problem"]), str(row["problems"]),
                      *(f"{row[key]:.3f}" for key in ("always_solved", "sometimes_solved", "never_solved")),
                      f"{row['pass@1']:.3f}", f"{row['pass@16']:.3f}" if "pass@16" in row else "—")
    Console().print(table)


if __name__ == "__main__":
    main()
