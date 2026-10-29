"""Export compact, independently rescored native benchmark runs for playback."""

import csv
import hashlib
import json
from pathlib import Path

from scripts.demo.inference import DemoRequest, score_trace
from scripts.dataset.pointer import execute


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "eval/pointer_benchmark/final-full-20261008/independent"
    tasks = {t["example_id"]: t for t in map(json.loads, (source / "native_tasks.jsonl").read_text().splitlines())}
    decisions = {r["example_id"]: r for r in csv.DictReader((source / "native/decisions.csv").open())}
    selections = [
        ("benchmark-307-full_cycle-000000-cce88da52549973d-steps-12", "12 steps · correct cycle"),
        ("benchmark-307-random_function-000000-991ed24ca01e4bf0-steps-12", "12 steps · first error at 4"),
        ("benchmark-307-full_cycle-000000-cce88da52549973d-steps-256", "256 steps · a long cycle"),
    ]
    selected_ids = {item[0] for item in selections}
    grouped = {key: [] for key in selected_ids}
    for row in csv.DictReader((source / "native/trajectories.csv").open()):
        if row["example_id"] in selected_ids:
            grouped[row["example_id"]].append(row)
    examples = []
    for key, label in selections:
        task = tasks[key]
        request = DemoRequest.parse({"mapping": task["mapping"], "start": task["initial_state"], "steps": task["task_depth"]})
        mapping = dict(request.pairs)
        targets = execute(mapping, request.start, len(grouped[key]))
        previous = request.start
        rows = []
        for index, native in enumerate(grouped[key]):
            assert int(native["loop"]) == index + 1
            assert native["intermediate_target"] == targets[index]
            correct = native["prediction"] == targets[index]
            assert correct == (native["intermediate_correct"] == "True")
            rows.append({"loop": index + 1, "prediction": native["prediction"],
                         "expected": targets[index], "correct": correct,
                         "edge_valid": mapping[previous] == native["prediction"],
                         "stop_probability": float(native["stop_probability"]),
                         "direct_prediction": None})
            previous = native["prediction"]
        summary = score_trace(request, rows, decisions[key]["stop_reason"])
        for own, native in [("answer_correct", "correct"), ("complete_trajectory", "complete_trajectory"),
                            ("exact_stop", "exact_stop"), ("joint_success", "joint_success")]:
            assert summary[own] == (decisions[key][native] == "True")
        examples.append({"schema_version": 1, "source": "recorded", "label": label,
                         "status": "complete", "checkpoint": "controller-shared-number-precision-seed83",
                         "request": request.payload(), "threshold": .5, "safety_cap": 272,
                         "trace": rows, "summary": summary,
                         "provenance": {"example_id": key,
                                        "report": "docs/history.md",
                                        "trajectory_sha256": hashlib.sha256((source / "native/trajectories.csv").read_bytes()).hexdigest()}})
    target = ROOT / "demo/web/examples.json"
    target.write_text(json.dumps(examples, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported {len(examples)} independently rescored recorded runs to {target}")


if __name__ == "__main__":
    main()
