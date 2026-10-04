"""Summarize requested-count interpolation separately from depth extrapolation."""
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table


def summarize_cohorts(results: Path) -> dict:
    """Aggregate existing per-depth metrics without fitting thresholds."""
    cohorts = {"trained_counts": ("validation", [1,2,3,4,5,6,8,10,12]),
               "held_out_counts": ("validation", [7,9,11]),
               "unseen_depths": ("depth_test", list(range(13,21)))}
    output = {}
    for name, (split, depths) in cohorts.items():
        full = json.loads((results / split / "summary.json").read_text())
        stopped = json.loads((results / (split + "-stopped") / "summary.json").read_text())
        a, b = full["by_depth"], stopped["by_depth"]
        count = sum(a[str(d)]["examples"] for d in depths)
        if any(a[str(d)]["examples"] != b[str(d)]["examples"] for d in depths):
            raise ValueError("Full and stopped cohorts differ")
        output[name] = {"depths": depths, "examples": count,
                       "trajectory_accuracy": sum(a[str(d)]["trajectory_accuracy"] * a[str(d)]["examples"] for d in depths) / count,
                       **{k: sum(b[str(d)][k] * b[str(d)]["examples"] for d in depths) / count
                          for k in ("accuracy", "exact_stop_rate", "joint_success_rate", "early_stop_rate", "cap_fallback_rate")}}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_cohorts(args.results)
    (args.results / "cohorts.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = Table("Cohort", "Full trajectory", "Stopped answer", "Exact stop + answer")
    for name, v in summary.items():
        table.add_row(name, f"{v['trajectory_accuracy']:.1%}", f"{v['accuracy']:.1%}", f"{v['joint_success_rate']:.1%}")
    Console().print(table)


if __name__ == "__main__":
    main()
