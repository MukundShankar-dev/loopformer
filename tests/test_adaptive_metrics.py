"""Offline stopping contracts: target semantics, causality, fallback and aggregation."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.eval.adaptive_metrics import (Trace, analyze, describe_trace, load_traces,
                                           margin_stop, oracle_stop, stable_prediction_stop,
                                           summarize_stops)


def trace(name, depth, right, predictions=None, margins=None):
    n = len(right)
    return Trace(name, depth, tuple(predictions or ("A" if c else "B" for c in right)), tuple(right),
                 all(right[:depth]), tuple(margins or [None] * n), tuple([None] * n))


def test_categories_and_oracles_are_finite_horizon():
    early = trace("early", 2, [False, True, True, True])
    need = trace("need", 2, [False, False, True, True])
    damage = trace("damage", 2, [False, True, False, False])
    recover = trace("recover", 2, [False, True, False, True])
    never = trace("never", 2, [False] * 4)
    assert [describe_trace(t)["category"] for t in (early, need, damage, recover, never)] == [
        "early_stable", "needs_compute", "overthinking", "unstable", "unsolved"]
    assert [oracle_stop(t) for t in (early, need, damage, recover, never)] == [2, 3, None, 4, None]
    assert oracle_stop(damage, rule="window") == 2
    assert oracle_stop(recover, rule="window", window=2) is None
    assert describe_trace(recover)["first_damage_loop"] == 2
    assert describe_trace(recover)["last_correct_region_start"] == 4
    assert describe_trace(early)["first_correct_loop"] == 2


def test_causal_stability_and_matched_compute():
    stable_wrong = trace("w", 2, [False, False, True, True], ["X", "X", "Y", "Y"])
    stable_right = trace("r", 2, [False, True, True, True], ["X", "Y", "Y", "Y"])
    assert stable_prediction_stop(stable_wrong, 2) == 2  # Can confidently stop wrong.
    assert stable_prediction_stop(stable_right, 2) == 3
    with pytest.raises(ValueError, match="margins absent"):
        margin_stop(stable_wrong, 1)
    margin = trace("m", 2, [False, False, True, True], margins=[0, 3, 0, 4])
    assert margin_stop(margin, 2) == 2  # Label correctness is never inspected.
    metrics = summarize_stops([stable_wrong, stable_right], [2, 3])
    assert metrics["accuracy"] == .5 and metrics["mean_loops"] == 2.5
    assert metrics["loop_savings_fraction"] == .375
    assert metrics["premature_stop"] == 1
    assert metrics["no_safe_stop"] == 0
    assert metrics["unnecessary_compute"] == 1
    with pytest.raises(ValueError, match="between task depth"):
        summarize_stops([stable_wrong], [1])
    rows, summary = analyze([stable_wrong, stable_right])
    assert len(rows) == 2 and summary["policies"]["fixed_budget"]["accuracy"] == 1


def test_loader_and_cli_reject_nonterminal_without_explicit_observational(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    path = run / "trajectories.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "task_depth", "loop", "prediction", "final_target",
                                                    "final_correct", "intermediate_target", "intermediate_correct"])
        writer.writeheader()
        for t, prediction, target in ((1, "A", "A"), (2, "B", "B"), (3, "B", "")):
            writer.writerow({"example_id": "one", "task_depth": 2, "loop": t, "prediction": prediction,
                             "final_target": "B", "final_correct": prediction == "B", "intermediate_target": target,
                             "intermediate_correct": prediction == target if target else ""})
    (run / "summary.json").write_text(json.dumps({"status": "complete", "selected_examples": 1, "loops": 3}))
    traces = load_traces(path)
    assert traces[0].complete_trajectory and oracle_stop(traces[0]) == 2
    cmd = [sys.executable, "-m", "scripts.eval.adaptive_analyze", str(run), "--output", str(tmp_path / "out")]
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    subprocess.run([*cmd, "--allow-observational"], capture_output=True, check=True)
    result = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert result["interpretation"] == "observational_only"
    assert result["policies"]["oracle_suffix"]["mean_loops"] == 2
    with path.open("a") as handle:
        handle.write("one,2,3,B,B,True,,\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_traces(path)
