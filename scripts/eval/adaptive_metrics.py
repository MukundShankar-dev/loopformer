"""Offline stopping analysis of existing complete recurrent trajectories.

The true target is used only for scoring and explicitly labeled oracles. Causal
policies use predictions and optional target-free confidence features.
"""

from collections import Counter, defaultdict
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable


@dataclass(frozen=True)
class Trace:
    example_id: str
    depth: int
    predictions: tuple[str, ...]
    correct: tuple[bool, ...]
    complete_trajectory: bool
    predicted_margins: tuple[float | None, ...]
    entropies: tuple[float | None, ...]

    @property
    def budget(self) -> int:
        return len(self.predictions)


def load_traces(path: Path) -> list[Trace]:
    """Require complete 1..T trajectories and one common budget."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["example_id"]].append(row)
    if not groups:
        raise ValueError("No trajectory rows")
    traces = []
    for example_id, rows in groups.items():
        rows.sort(key=lambda r: int(r["loop"]))
        depth = int(rows[0]["task_depth"])
        if depth < 1 or [int(r["loop"]) for r in rows] != list(range(1, len(rows) + 1)) or len(rows) < depth:
            raise ValueError(f"Incomplete, duplicate, or short trajectory: {example_id}")
        target = rows[0]["final_target"]
        if any(int(r["task_depth"]) != depth or r["final_target"] != target or
               r.get("split") != rows[0].get("split") or r.get("seed") != rows[0].get("seed") or
               r["final_correct"] != str(r["prediction"] == target) for r in rows):
            raise ValueError(f"Inconsistent final target/correctness: {example_id}")
        nominal = rows[:depth]
        if any(r["intermediate_correct"] != str(r["prediction"] == r["intermediate_target"])
               for r in nominal):
            raise ValueError(f"Inconsistent intermediate correctness: {example_id}")
        def optional(name: str) -> tuple[float | None, ...]:
            values = tuple(float(r[name]) if r.get(name) not in (None, "") else None for r in rows)
            if any(v is not None and (v < 0 or not math.isfinite(v)) for v in values):
                raise ValueError(f"Invalid {name}: {example_id}")
            return values
        traces.append(Trace(example_id, depth, tuple(r["prediction"] for r in rows),
                            tuple(r["final_correct"] == "True" for r in rows),
                            all(r["intermediate_correct"] == "True" for r in nominal),
                            optional("predicted_margin"), optional("answer_entropy")))
    if len({trace.budget for trace in traces}) != 1:
        raise ValueError("Every example must have the same loop budget")
    return traces


def oracle_stop(trace: Trace, *, rule: str = "suffix", window: int = 1) -> int | None:
    """Earliest t>=d that is correct through T or for a declared window."""
    if rule not in ("suffix", "window") or window < 1:
        raise ValueError("Require suffix/window rule and positive window")
    for t in range(trace.depth, trace.budget + 1):
        end = trace.budget if rule == "suffix" else t + window - 1
        if end <= trace.budget and all(trace.correct[t - 1:end]):
            return t
    return None


def describe_trace(trace: Trace) -> dict:
    """Mutually exclusive terminal-style categories; caller labels task variant."""
    eligible = trace.correct[trace.depth - 1:]
    first = next((trace.depth + i for i, right in enumerate(eligible) if right), None)
    first_all = next((i for i, right in enumerate(trace.correct, 1) if right), None)
    damage = next((trace.depth + i for i, (a, b) in enumerate(zip(eligible, eligible[1:])) if a and not b), None)
    if first is None:
        category = "unsolved"
    elif any(eligible[i] and not eligible[i + 1] and any(eligible[i + 2:])
             for i in range(len(eligible) - 1)):
        category = "unstable"
    elif damage is not None:
        category = "overthinking"
    elif first == trace.depth:
        category = "early_stable"
    else:
        category = "needs_compute"
    last_run = next((i for i in range(trace.budget, 0, -1) if not trace.correct[i - 1]), 0) + 1 if trace.correct[-1] else None
    return {"example_id": trace.example_id, "task_depth": trace.depth, "category": category,
            "first_correct_loop": first_all, "first_eligible_correct_loop": first,
            "stable_from_first_eligible": first is not None and all(trace.correct[first - 1:]),
            "first_damage_loop": damage, "last_correct_region_start": last_run,
            "oracle_suffix_stop": oracle_stop(trace), "fixed_loops": trace.budget,
            "complete_trajectory": trace.complete_trajectory}


def stable_prediction_stop(trace: Trace, k: int) -> int:
    """Causal stability policy with minimum depth d and T fallback."""
    if k < 1:
        raise ValueError("k must be positive")
    for t in range(trace.depth, trace.budget + 1):
        if t >= k and len(set(trace.predictions[t - k:t])) == 1:
            return t
    return trace.budget


def margin_stop(trace: Trace, threshold: float) -> int:
    """Causal top-two predicted-answer margin policy; never use label margin."""
    if not math.isfinite(threshold):
        raise ValueError("Margin threshold must be finite")
    if any(value is None for value in trace.predicted_margins):
        raise ValueError("Predicted margins absent; rerun full-loop evaluation with feature export")
    for t in range(trace.depth, trace.budget + 1):
        if trace.predicted_margins[t - 1] >= threshold:
            return t
    return trace.budget


def entropy_stop(trace: Trace, threshold: float) -> int:
    """Causal allowed-answer entropy policy; requires target-free export."""
    if not math.isfinite(threshold):
        raise ValueError("Entropy threshold must be finite")
    if any(value is None for value in trace.entropies):
        raise ValueError("Answer entropies absent; rerun full-loop evaluation with feature export")
    for t in range(trace.depth, trace.budget + 1):
        if trace.entropies[t - 1] <= threshold:
            return t
    return trace.budget


def summarize_stops(traces: list[Trace], stops: list[int], *, rule: str = "suffix") -> dict:
    """Score stopped final answers and recurrent-loop accounting on one cohort."""
    if not traces or len(stops) != len(traces) or len({t.budget for t in traces}) != 1:
        raise ValueError("Require one matched cohort and stop for every trace")
    if any(type(stop) is not int or not trace.depth <= stop <= trace.budget
           for trace, stop in zip(traces, stops, strict=True)):
        raise ValueError("Stops must be integers between task depth and budget")
    ordered = sorted(stops)
    n, budget = len(traces), traces[0].budget
    safe = [oracle_stop(t, rule=rule) for t in traces]
    eligible = [(stop, point) for stop, point in zip(stops, safe) if point is not None]
    return {"examples": n, "accuracy": sum(t.correct[s - 1] for t, s in zip(traces, stops)) / n,
            "complete_trajectory_accuracy": mean(t.complete_trajectory for t in traces),
            "mean_loops": mean(stops), "p50_loops": ordered[(n * 50 + 99) // 100 - 1],
            "p95_loops": ordered[(n * 95 + 99) // 100 - 1],
            "total_loops": sum(stops), "fixed_total_loops": n * budget,
            "loop_savings_fraction": 1 - sum(stops) / (n * budget),
            "no_safe_stop": n - len(eligible),
            "premature_stop": sum(not t.correct[s - 1] and any(t.correct[s:])
                                  for t, s in zip(traces, stops)),
            "unnecessary_compute": sum(stop > point for stop, point in eligible),
            "unnecessary_compute_denominator": len(eligible)}


def analyze(traces: list[Trace], *, stability_k: int = 2,
            margin_threshold: float | None = None, entropy_threshold: float | None = None,
            oracle_window: int = 1) -> tuple[list[dict], dict]:
    """Full cohort, requested-depth, oracle, and prediction-stability baselines."""
    if not traces or oracle_window < 1:
        raise ValueError("Require traces and a positive oracle window")
    descriptions = [describe_trace(t) for t in traces]
    policies: dict[str, Callable[[Trace], int]] = {
        "fixed_budget": lambda t: t.budget,
        "requested_depth": lambda t: t.depth,
        "oracle_suffix": lambda t: oracle_stop(t) or t.budget,
        "oracle_first_correct": lambda t: oracle_stop(t, rule="window") or t.budget,
        f"prediction_stability_k{stability_k}": lambda t: stable_prediction_stop(t, stability_k),
        f"oracle_window_{oracle_window}": lambda t: oracle_stop(t, rule="window", window=oracle_window) or t.budget,
    }
    if margin_threshold is not None:
        policies[f"margin_threshold_{margin_threshold}"] = lambda t: margin_stop(t, margin_threshold)
    if entropy_threshold is not None:
        policies[f"entropy_threshold_{entropy_threshold}"] = lambda t: entropy_stop(t, entropy_threshold)
    summaries = {name: summarize_stops(traces, [policy(t) for t in traces]) for name, policy in policies.items()}
    by_depth = {}
    for depth in sorted({t.depth for t in traces}):
        subset = [t for t in traces if t.depth == depth]
        by_depth[str(depth)] = {"categories": dict(Counter(describe_trace(t)["category"] for t in subset)),
                                "policies": {name: summarize_stops(subset, [policy(t) for t in subset])
                                             for name, policy in policies.items()}}
    frontier = []
    for budget in range(max(t.depth for t in traces), traces[0].budget + 1):
        prefix = [Trace(t.example_id, t.depth, t.predictions[:budget], t.correct[:budget],
                        t.complete_trajectory, t.predicted_margins[:budget], t.entropies[:budget]) for t in traces]
        frontier.append({"budget": budget, "fixed": summarize_stops(prefix, [budget] * len(prefix)),
                         "oracle_suffix": summarize_stops(prefix, [oracle_stop(t) or budget for t in prefix])})
    return descriptions, {"categories": dict(Counter(row["category"] for row in descriptions)),
                          "policies": summaries, "by_depth": by_depth, "frontier": frontier}
