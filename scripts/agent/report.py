"""Summarize agent episodes, optionally against a baseline run on the same problems.

    python -m scripts.agent.report --episodes eval/agent/<run>/episodes.jsonl --problems <problem set> \\
        [--by difficulty platform] [--baseline eval/agent/<other>/episodes.jsonl]

Writes report.json and report.md next to the episodes. Each group (all
problems, then each value of each `--by` metadata key) reports:
- solved: the fraction of episodes whose submission passes the hidden tests,
  with a 95% bootstrap interval over problems (episodes of one problem are
  correlated, so problems are the resampling unit), and pass@k;
- selected@k: the breadth baseline, which submits the first of k attempts
  that passes the public tests (the first attempt if none does); the exact
  expectation over k-subsets of each problem's attempts in seed order;
- public_pass_hidden_fail: of submissions that pass the public tests, the
  fraction failing hidden tests (overfitting to the examples);
- stops, hidden outcome classes, tool calls per episode, and repairs
  (solved episodes with an earlier turn failing the public tests: a
  `run_public_tests` call or a repair-mode submission);
- compute per episode: FLOPs, tokens processed and generated, model and
  sandbox seconds.
With `--baseline`, the paired difference in per-problem solve rate (this run
minus baseline) with its bootstrap interval, and problem wins and losses.
"""
import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
from rich.console import Console

from scripts.codeprobe.headroom import summarize
from scripts.codetasks.problems import read_problems

DRAWS = 10_000
PASS_KS = (1, 2, 4, 8, 16)  # those up to the episodes per problem are reported


def read_episodes(path: Path) -> dict[str, list[dict]]:
    """Episodes per problem ID, each problem with the same number of episodes."""
    episodes = defaultdict(list)
    for line in path.read_text().splitlines():
        episode = json.loads(line)
        episodes[episode["problem_id"]].append(episode)
    counts = {len(group) for group in episodes.values()}
    if len(counts) > 1:
        raise ValueError(f"{path}: problems have unequal episode counts {sorted(counts)}")
    return dict(episodes)


def bootstrap_mean(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean of `values` (one per problem)."""
    draws = np.random.default_rng(seed).choice(values, size=(DRAWS, len(values))).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def solved(episode: dict) -> bool:
    return episode["hidden_outcome"] == "pass"


def repaired(episode: dict) -> bool:
    """Solved after some turn failed the public tests (run_public_tests, or a repair submission)."""
    return solved(episode) and any(turn["public_pass_fraction"] is not None and turn["outcome"] != "pass"
                                   for turn in episode["turns"])


def selected_at_k(group: list[dict], k: int) -> float:
    """Expected hidden-test success of the breadth baseline on a random k-subset of `group`, attempts in seed order.
    A public-passing attempt is chosen when no earlier public-passing attempt is in the subset; a failing one when
    no public-passing attempt is in the subset and it is the subset's first."""
    attempts = sorted(group, key=lambda episode: episode["seed"])
    public = [episode["public_outcome"] == "pass" for episode in attempts]
    n, total = len(attempts), 0
    for index, episode in enumerate(attempts):
        if public[index]:
            subsets = comb(n - 1 - sum(public[:index]), k - 1)
        else:  # the other k - 1 are all later failing attempts
            subsets = comb(sum(not passed for passed in public[index + 1:]), k - 1)
        total += subsets * solved(episode)
    return total / comb(n, k)


def group_summary(episodes: dict[str, list[dict]]) -> dict:
    flat = [episode for group in episodes.values() for episode in group]
    per_problem = np.array([np.mean([solved(e) for e in group]) for group in episodes.values()])
    public_pass = [e for e in flat if e["public_outcome"] == "pass"]
    turn = lambda e, key: sum(t[key] for t in e["turns"])
    mean = lambda values: float(np.mean(values))
    return {
        "solved": float(per_problem.mean()),
        "solved_interval": bootstrap_mean(per_problem),
        **{key: value for key, value in summarize({p: [solved(e) for e in g] for p, g in episodes.items()},
                                                  ks=PASS_KS).items() if key != "pass@1"},
        **{f"selected@{k}": float(np.mean([selected_at_k(group, k) for group in episodes.values()]))
           for k in PASS_KS if k <= min(len(group) for group in episodes.values())},
        "public_pass": len(public_pass) / len(flat),
        "public_pass_hidden_fail": mean([not solved(e) for e in public_pass]) if public_pass else None,
        "stops": {key: count / len(flat) for key, count in Counter(e["stop"] for e in flat).most_common()},
        "hidden_outcomes": {key: count / len(flat)
                            for key, count in Counter(str(e["hidden_outcome"]) for e in flat).most_common()},
        "turns": mean([len(e["turns"]) for e in flat]),
        "tool_calls": {tool: mean([sum(t["tool"] == tool for t in e["turns"]) for e in flat])
                       for tool in ("run_python", "run_public_tests", "submit")},
        "repaired": mean([repaired(e) for e in flat]),
        "tflops": mean([e["flops"] / 1e12 for e in flat]),
        "tokens_processed": mean([e["tokens_processed"] for e in flat]),
        "tokens_generated": mean([turn(e, "tokens") for e in flat]),
        "model_seconds": mean([turn(e, "model_seconds") for e in flat]),
        "sandbox_seconds": mean([turn(e, "sandbox_seconds") for e in flat]),
    }


def comparison(episodes: dict[str, list[dict]], baseline: dict[str, list[dict]]) -> dict:
    if episodes.keys() != baseline.keys():
        raise ValueError("the run and the baseline cover different problems")
    ids = sorted(episodes)
    rate = lambda run: np.array([np.mean([solved(e) for e in run[p]]) for p in ids])
    difference = rate(episodes) - rate(baseline)
    tflops = lambda run: np.mean([e["flops"] for p in ids for e in run[p]]) / 1e12
    return {"problems": len(ids), "solved_difference": float(difference.mean()),
            "solved_difference_interval": bootstrap_mean(difference),
            "wins": int((difference > 0).sum()), "losses": int((difference < 0).sum()),
            "tflops": float(tflops(episodes)), "baseline_tflops": float(tflops(baseline))}


def groups(episodes: dict[str, list[dict]], metadata: dict[str, dict], keys: list[str]) -> dict[str, dict]:
    result = {"all": episodes}
    for key in keys:
        values = defaultdict(dict)
        for problem_id, group in episodes.items():
            values[str(metadata[problem_id].get(key))][problem_id] = group
        result |= {f"{key}={value}": group for value, group in sorted(values.items())}
    return result


def markdown(report: dict) -> str:
    percent = lambda value: "–" if value is None else f"{100 * value:.1f}"
    lines = [f"# Agent report: {report['episodes']}", "",
             f"{report['problems']} problems, {report['episodes_per_problem']} episodes each.", "",
             "| Group | Problems | Solved % (95% CI) | Public pass % | Public pass, hidden fail % | Repaired % "
             "| Turns | TFLOPs | Model s | Sandbox s |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, group in report["groups"].items():
        low, high = group["solved_interval"]
        lines.append(f"| {name} | {group['problems']} | {percent(group['solved'])} ({percent(low)}–{percent(high)}) "
                     f"| {percent(group['public_pass'])} | {percent(group['public_pass_hidden_fail'])} "
                     f"| {percent(group['repaired'])} | {group['turns']:.2f} | {group['tflops']:.2f} "
                     f"| {group['model_seconds']:.1f} | {group['sandbox_seconds']:.1f} |")
    every = report["groups"]["all"]
    for prefix, name in (("pass@", "pass@k, oracle"), ("selected@", "selected@k, by public tests")):
        values = {key: value for key, value in every.items() if key.startswith(prefix)}
        if values:
            lines += ["", f"{name} (all): " + ", ".join(f"{key} {percent(value)}%" for key, value in values.items())]
    for title, key in (("Stops", "stops"), ("Hidden outcomes", "hidden_outcomes"), ("Tool calls per episode", "tool_calls")):
        values = every[key]
        shown = (f"{name} {value:.2f}" if key == "tool_calls" else f"{name} {percent(value)}%"
                 for name, value in values.items())
        lines += ["", f"{title} (all): " + ", ".join(shown)]
    if "comparison" in report:
        c = report["comparison"]
        low, high = c["solved_difference_interval"]
        lines += ["", f"## Against {report['baseline']}", "",
                  f"Solved difference {100 * c['solved_difference']:+.1f} points (95% CI {100 * low:+.1f} to "
                  f"{100 * high:+.1f}) over {c['problems']} problems; wins {c['wins']}, losses {c['losses']}. "
                  f"TFLOPs per episode {c['tflops']:.2f} vs {c['baseline_tflops']:.2f}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True, help="problem-set directory the episodes ran on")
    parser.add_argument("--by", nargs="*", default=[], help="problem metadata keys to group by")
    parser.add_argument("--baseline", type=Path, help="episodes of another run on the same problems")
    args = parser.parse_args()
    episodes = read_episodes(args.episodes)
    metadata = {problem.id: problem.metadata for problem in read_problems(args.problems)}
    missing = episodes.keys() - metadata.keys()
    if missing:
        raise ValueError(f"episodes for problems not in {args.problems}: {sorted(missing)[:5]}")
    report = {"episodes": str(args.episodes), "problems": len(episodes),
              "episodes_per_problem": len(next(iter(episodes.values()))),
              "groups": {name: group_summary(group) for name, group in groups(episodes, metadata, args.by).items()}}
    if args.baseline:
        report |= {"baseline": str(args.baseline), "comparison": comparison(episodes, read_episodes(args.baseline))}
    (args.episodes.parent / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    text = markdown(report)
    (args.episodes.parent / "report.md").write_text(text)
    Console(soft_wrap=True).print(text, markup=False)


if __name__ == "__main__":
    main()
