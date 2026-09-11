"""Compare absolute depth and extrapolation distance on matching evaluations."""

import json
from pathlib import Path


def compare_depth_runs(reference: list[Path], candidate: list[Path]) -> list[dict]:
    """Return per-depth rows; equal offsets are not paired examples or equal compute.

    Each corresponding pair must use the same full dataset and inference settings.
    Different files must cover disjoint depths. No test score selects a checkpoint.
    """
    if not reference or len(reference) != len(candidate):
        raise ValueError("Supply matching nonempty reference and candidate run lists")
    rows, seen = [], set()
    identities = {}
    for left, right in zip(reference, candidate, strict=True):
        pair = [json.loads((path / "summary.json").read_text()) for path in (left, right)]
        for s in pair:
            if s["status"] != "complete" or s.get("test_mode") or s.get("limit") is not None or s.get("task_variant"):
                raise ValueError("Comparison requires complete, full, original-task evaluations")
        for key in ("data_sha256", "selected_examples", "loops", "dtype", "device", "batch_size", "seed", "scoring", "prompt_format", "source_sha256"):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"Paired evaluations differ in {key}")
        for key in ("base_model", "revision", "symbols", "token_ids"):
            if pair[0]["model"][key] != pair[1]["model"][key]:
                raise ValueError(f"Checkpoint comparison differs in {key}")
        if pair[0]["model"]["train_max_depth"] >= pair[1]["model"]["train_max_depth"]:
            raise ValueError("Candidate must have a larger recorded training depth")
        if pair[0]["by_depth"].keys() != pair[1]["by_depth"].keys():
            raise ValueError("Paired evaluations cover different depths")
        for role, path, s in zip(("reference", "candidate"), (left, right), pair, strict=True):
            identity = (s["checkpoint"], s["local_checkpoint_sha256"])
            if role in identities and identities[role] != identity:
                raise ValueError("Each role must use the same checkpoint across all files")
            identities[role] = identity
            maximum = s["model"]["train_max_depth"]
            for d, metrics in s["by_depth"].items():
                depth = int(d)
                if (role, depth) in seen:
                    raise ValueError("Input files overlap in task depth")
                seen.add((role, depth))
                offset = depth - maximum
                rows.append({
                    "role": role, "checkpoint": s["checkpoint"], "evaluation": str(path.resolve()),
                    "data_sha256": s["data_sha256"], "train_max_depth": maximum, "task_depth": depth,
                    "steps_beyond_training": offset,
                    "region": "trained" if offset <= 0 else ("near_ood" if offset <= 2 else "far_ood"),
                    "examples": metrics["examples"], "trajectory_accuracy": metrics["trajectory_accuracy"],
                    "final_accuracy": s["depth_by_loop"][d][d]["final_accuracy"], "loss": metrics["loss"],
                })
    return sorted(rows, key=lambda r: (r["role"], r["task_depth"]))
