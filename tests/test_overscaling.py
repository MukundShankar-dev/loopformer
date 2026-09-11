"""Terminal semantics and conditional dynamics, independently prescribed."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.dataset.pointer import SYMBOLS, execute, generate_example
from scripts.dataset.terminal import absorbing_terminal
from scripts.eval.loop_metrics import write_csv
from scripts.eval.overscaling_metrics import overscaling_metrics


def test_terminal_transform_preserves_nominal_path_and_source():
    for depth in (1, 4, 8, 25):
        source = generate_example(17 + depth, depth, "test", depth)
        original = source.to_dict()
        terminal = absorbing_terminal(source)
        assert source.to_dict() == original
        assert terminal.example_id == source.example_id and terminal.seed == source.seed
        assert terminal.intermediate_states == source.intermediate_states
        assert execute(dict(terminal.mapping), terminal.initial_state, 64) == [
            *source.intermediate_states, *([source.final_state] * (64 - depth))]
        assert all(a == b for a, b in zip(source.mapping, terminal.mapping) if a[0] != source.final_state)
        assert absorbing_terminal(terminal) == terminal


def test_repair_damage_cohorts_censoring_and_recovery(tmp_path):
    tasks = [absorbing_terminal(generate_example(17 + i, d, "test", i)) for i, d in enumerate((1, 2))]
    sequences = [[True, False, True, True, False], [False, False, True, False, True]]
    rows = []
    for task, sequence in zip(tasks, sequences):
        wrong = next(s for s in SYMBOLS if s != task.final_state)
        for t, right in enumerate(sequence, 1):
            rows.append({"example_id": task.example_id, "task_depth": task.task_depth, "loop": t,
                         "prediction": task.final_state if right else wrong, "final_target": task.final_state,
                         "final_correct": right, "final_margin": 1 if right else -1})
    path = tmp_path / "trajectories.csv"
    write_csv(path, rows)
    artifacts, summary = overscaling_metrics(path, tasks)
    rates = {r["loop"]: r for r in artifacts["transition_rates.csv"]
             if r["task_depth"] == "all" and r["scope"] == "post_nominal"}
    assert rates[1]["examples"] == 1 and rates[1]["damage_rate"] == 1 and rates[1]["repair_rate"] is None
    assert rates[2]["repair_rate"] == 1 and rates[2]["damage_rate"] is None
    assert rates[3]["damage_rate"] == .5 and rates[3]["net_gain"] == -.5
    assert rates[4]["repair_rate"] == rates[4]["damage_rate"] == 1 and rates[4]["net_gain"] == 0
    for r in artifacts["transition_rates.csv"]:
        if r["examples"]:
            assert r["accuracy_at_next"] - r["accuracy_at_t"] == pytest.approx(r["net_gain"])
        else:
            assert r["repair_rate"] is r["damage_rate"] is r["net_gain"] is None
    survival = [r for r in artifacts["survival.csv"] if r["task_depth"] == "all"]
    assert survival[0]["survival"] == 1
    assert survival[2]["observed"] == 2 and survival[2]["survival"] == 0  # Recovery doesn't restore survival.
    assert survival[3]["observed"] == 1 and survival[3]["censored"] == 1
    assert summary["all"]["hold_nominal_prediction_baseline_accuracy"] == .5
    assert len(artifacts["transitions.csv"]) == 8
    write_csv(path, rows[:-1])
    with pytest.raises(ValueError, match="same loop budget"):
        overscaling_metrics(path, tasks)
    write_csv(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="complete, unique"):
        overscaling_metrics(path, tasks)
    # An early final match on a depth-2 task is not a completed solution; an
    # entirely unsolved cohort has undefined survival, not perfect survival.
    for row in rows:
        early = row["task_depth"] == 2 and row["loop"] == 1
        row["prediction"] = row["final_target"] if early else next(s for s in SYMBOLS if s != row["final_target"])
        row["final_correct"] = early
        row["final_margin"] = 1 if early else -1
    write_csv(path, rows)
    artifacts, summary = overscaling_metrics(path, tasks)
    assert summary["all"]["early_final_matches"] == 1
    assert summary["all"]["never_solved_at_or_after_depth"] == 2
    assert all(r["observed"] == 0 and r["survival"] is None for r in artifacts["survival.csv"])


def test_dry_run_needs_no_checkpoint_and_writes_nothing(tmp_path):
    data = tmp_path / "test.jsonl"
    data.write_text("".join(json.dumps(generate_example(17 + i, i + 1, "test", i).to_dict()) + "\n" for i in range(5)))
    before = data.read_bytes()
    output = tmp_path / "unused"
    result = subprocess.run([sys.executable, "-m", "scripts.eval.overscaling_test", "--data", str(data),
                             "--dry-run", "--output", str(output)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert "DRY RUN" in result.stdout and "Reference:" in result.stdout
    assert not output.exists() and data.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["test.jsonl"]
