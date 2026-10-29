"""Independently audit paired MMLU records and summarize retained accuracy."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from rich.console import Console
import yaml

from scripts.eval.paper_baseline import atomic_json
from scripts.eval.pointer_task import sha256_file


def render_question(doc: dict) -> str:
    """Reconstruct the standard four-choice MMLU text from its dataset row."""
    return doc["question"].strip() + "\n" + "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", doc["choices"], strict=True)
    ) + "\nAnswer:"


def render_context(description: str, dev: list[dict], doc: dict) -> str:
    """Five first-in-order dev answers followed by the unanswered test item."""
    examples = [render_question(row) + " " + "ABCD"[row["answer"]] for row in dev]
    return description + "\n\n".join([*examples, render_question(doc)])


def audit_row(row: dict, doc: dict, context: str) -> tuple[int, bool]:
    """Check raw inputs/targets and rescore A–D without the inference scorer."""
    if row["doc"] != doc or row["target"] != doc["answer"]:
        raise ValueError("Logged test document or target differs from the pinned dataset")
    if row["arguments"] != [[context, " " + letter] for letter in "ABCD"]:
        raise ValueError("Prompt or continuation differs from the reconstructed five-shot protocol")
    expected_hash = hashlib.sha256(context.encode()).hexdigest()
    if row["prompt_hash"] != expected_hash:
        raise ValueError("Logged prompt hash differs")
    responses = row["filtered_resps"]
    if len(responses) != 4 or row["resps"] != [[response] for response in responses]:
        raise ValueError("Expected exactly four unmodified single-repeat likelihood responses")
    scores = [float(response[0]) for response in responses]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Nonfinite answer likelihood")
    prediction = max(range(4), key=lambda i: scores[i])
    correct = prediction == doc["answer"]
    if int(row["acc"]) != int(correct):
        raise ValueError("Saved accuracy disagrees with independent likelihood argmax")
    return prediction, correct


def paired_summary(rows: list[dict], *, seed: int = 0, repetitions: int = 10000) -> dict:
    """Question-weighted paired effects with resampling inside each subject."""
    if not rows:
        raise ValueError("No paired questions")
    count = len(rows)
    base = sum(row["base_correct"] for row in rows)
    recurrent = sum(row["recurrent_correct"] for row in rows)
    damage = sum(row["base_correct"] and not row["recurrent_correct"] for row in rows)
    repair = sum(row["recurrent_correct"] and not row["base_correct"] for row in rows)
    rng = np.random.default_rng(seed)
    draws = np.zeros(repetitions)
    for subject in sorted({row["subject"] for row in rows}):
        group = [row for row in rows if row["subject"] == subject]
        decreases = sum(row["base_correct"] and not row["recurrent_correct"] for row in group)
        increases = sum(row["recurrent_correct"] and not row["base_correct"] for row in group)
        sampled = rng.multinomial(len(group), np.array([decreases, len(group) - decreases - increases, increases]) / len(group), size=repetitions)
        draws += sampled[:, 2] - sampled[:, 0]
    interval = np.quantile(draws / count, [0.025, 0.975]).tolist()
    return {"questions": count, "base_correct": base, "recurrent_correct": recurrent,
            "base_accuracy": base / count, "recurrent_accuracy": recurrent / count,
            "difference": (recurrent - base) / count, "difference_ci95": interval,
            "base_correct_recurrent_wrong": damage, "base_wrong_recurrent_correct": repair,
            "prediction_agreement": sum(row["base_prediction"] == row["recurrent_prediction"] for row in rows) / count,
            "bootstrap": {"repetitions": repetitions, "seed": seed, "unit": "paired question, stratified within subject"}}


def read_samples(path: Path) -> list[dict]:
    """Read physical JSONL lines, preserving Unicode separators inside strings."""
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mmlu_retention.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["output"])
    if json.loads((root / "protocol.json").read_text()) != config:
        raise ValueError("Frozen protocol differs")
    freezes = {arm: json.loads((root / arm / "freeze.json").read_text()) for arm in ("base", "recurrent")}
    for field in ("protocol", "checkpoint_sha256", "source_sha256", "evaluation_sha256", "packages", "git_commit", "device"):
        if freezes["base"][field] != freezes["recurrent"][field]:
            raise ValueError(f"Arm identities differ at {field}")
    preflight = json.loads((root / "preflight/preflight.json").read_text())
    if not preflight["passed"]:
        raise ValueError("Reference-scoring preflight did not pass")
    if not json.loads((root / "architecture_control.json").read_text())["passed"]:
        raise ValueError("Exact checkpoint-architecture control did not pass")
    subject_counts = preflight["subject_counts"]
    harness_results = {arm: json.loads((root / arm / "results.json").read_text()) for arm in freezes}
    paired, subjects, sources = [], [], {}
    for name, expected in sorted(subject_counts.items()):
        subject = name.removeprefix("mmlu_")
        definition = yaml.safe_load((root / "task_configs" / (name + ".yaml")).read_text())
        dataset = load_dataset(config["dataset"], subject, revision=config["dataset_revision"])
        if len(dataset["test"]) != expected or len(dataset["dev"]) != config["num_fewshot"]:
            raise ValueError("Pinned dataset coverage differs")
        if dataset["test"]._fingerprint != preflight["dataset_fingerprints"][name]:
            raise ValueError("Dataset fingerprint differs from preflight")
        dev = list(dataset["dev"])
        records = {}
        for arm in freezes:
            path = root / arm / (name + ".jsonl")
            sources[str(path.relative_to(root))] = sha256_file(path)
            values = read_samples(path)
            records[arm] = {row["doc_id"]: row for row in values}
            if len(records[arm]) != len(values) or set(records[arm]) != set(range(expected)):
                raise ValueError("Missing or duplicate MMLU test IDs")
        local = []
        category = definition["tag"].removeprefix("mmlu_").removesuffix("_tasks")
        for index, doc in enumerate(dataset["test"]):
            context = render_context(definition["description"], dev, doc)
            base_prediction, base_correct = audit_row(records["base"][index], doc, context)
            recurrent_prediction, recurrent_correct = audit_row(records["recurrent"][index], doc, context)
            local.append({"subject": subject, "category": category, "doc_id": index, "target": doc["answer"],
                          "base_prediction": base_prediction, "recurrent_prediction": recurrent_prediction,
                          "base_correct": base_correct, "recurrent_correct": recurrent_correct,
                          "prompt_sha256": hashlib.sha256(context.encode()).hexdigest()})
        statistics = {"subject": subject, "category": category, "questions": expected,
                      "base_accuracy": sum(row["base_correct"] for row in local) / expected,
                      "recurrent_accuracy": sum(row["recurrent_correct"] for row in local) / expected}
        statistics["difference"] = statistics["recurrent_accuracy"] - statistics["base_accuracy"]
        for arm in freezes:
            if abs(statistics[arm + "_accuracy"] - harness_results[arm]["results"][name]["acc,none"]) > 1e-12:
                raise ValueError("Subject accuracy disagrees with harness aggregate")
        paired.extend(local)
        subjects.append(statistics)
    if len(subjects) != config["expected_subjects"] or len(paired) != config["expected_questions"]:
        raise ValueError("Incomplete paired benchmark")
    summary = paired_summary(paired, seed=config["seed"])
    for arm in freezes:
        if abs(summary[arm + "_accuracy"] - harness_results[arm]["groups"]["mmlu"]["acc,none"]) > 1e-12:
            raise ValueError("Overall accuracy disagrees with harness aggregate")
    summary["categories"] = {}
    for category in sorted({row["category"] for row in paired}):
        subset = [row for row in paired if row["category"] == category]
        statistics = {"questions": len(subset), "base_accuracy": sum(row["base_correct"] for row in subset) / len(subset),
                      "recurrent_accuracy": sum(row["recurrent_correct"] for row in subset) / len(subset)}
        for arm in freezes:
            reported = harness_results[arm]["groups"]["mmlu_" + category]["acc,none"]
            if abs(reported - statistics[arm + "_accuracy"]) > 1e-12:
                raise ValueError("Category accuracy disagrees with harness aggregate")
        summary["categories"][category] = statistics
    summary["subjects"] = len(subjects)
    summary["macro_subject_base_accuracy"] = float(np.mean([row["base_accuracy"] for row in subjects]))
    summary["macro_subject_recurrent_accuracy"] = float(np.mean([row["recurrent_accuracy"] for row in subjects]))
    for filename, rows in (("paired.csv", paired), ("subjects.csv", subjects)):
        with (root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(root / "comparison.json", summary)
    atomic_json(root / "independent_audit.json", {"passed": True, "questions_per_arm": len(paired),
                "subjects": len(subjects), "protocol_sha256": sha256_file(root / "protocol.json"),
                "architecture_control_sha256": sha256_file(root / "architecture_control.json"),
                "sample_sha256": sources, "audit_source_sha256": sha256_file(Path(__file__)),
                "comparison_sha256": sha256_file(root / "comparison.json"),
                "checks": ["pinned dataset documents and targets", "first-five-dev prompt reconstruction",
                           "paired byte-identical prompts", "all test IDs exactly once", "four raw likelihoods",
                           "independent argmax and accuracy", "subject/category/overall harness agreement", "matched arm provenance"]})
    Console().print(summary)


if __name__ == "__main__":
    main()
