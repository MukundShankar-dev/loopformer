"""Gated supervised training of a small stopping head from terminal traces."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from rich.console import Console

from .adaptive_metrics import Trace, analyze, load_traces, oracle_stop, summarize_stops
from .adaptive_policy import HaltingHead, feature_vector
from .pointer_task import sha256_file


def _features(trace: Trace, t: int) -> tuple[float, ...]:
    if trace.predicted_margins[t - 1] is None or trace.entropies[t - 1] is None:
        raise ValueError("Halting training needs predicted_margin and answer_entropy; regenerate terminal traces")
    repeated = 1
    for prior in range(t - 2, -1, -1):
        if trace.predictions[prior] != trace.predictions[t - 1]:
            break
        repeated += 1
    return feature_vector(t, trace.depth, trace.budget, trace.predicted_margins[t - 1],
                          trace.entropies[t - 1], repeated)


def training_examples(traces: list[Trace], *, label_rule: str = "first_correct") -> tuple[torch.Tensor, torch.Tensor]:
    """Supervised terminal stop labels; T is always a forced fallback."""
    if label_rule not in ("first_correct", "suffix"):
        raise ValueError("Unknown oracle label rule")
    vectors, labels = [], []
    for trace in traces:
        safe = oracle_stop(trace)
        for t in range(trace.depth, trace.budget):
            vectors.append(_features(trace, t))
            labels.append(int(trace.correct[t - 1]) if label_rule == "first_correct"
                          else int(safe is not None and t >= safe))
    if not vectors or len(set(labels)) < 2:
        raise ValueError("Need both safe-stop and continue labels before fitting a head")
    return torch.tensor(vectors, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def headroom_gate(traces: list[Trace], *, label_rule: str = "first_correct",
                  min_extra_savings: float = .05, quality_tolerance: float = .02) -> dict:
    """Require terminal oracle to beat simple policies at comparable quality."""
    if label_rule not in ("first_correct", "suffix"):
        raise ValueError("Unknown oracle label rule")
    if not 0 <= min_extra_savings < 1 or not 0 <= quality_tolerance < 1:
        raise ValueError("Invalid gate thresholds")
    _, analysis = analyze(traces)
    policies = analysis["policies"]
    oracle = policies["oracle_first_correct" if label_rule == "first_correct" else "oracle_suffix"]
    simple = [policies[name] for name in ("fixed_budget", "requested_depth", "prediction_stability_k2")]
    comparable = [row for row in simple if row["accuracy"] >= oracle["accuracy"] - quality_tolerance]
    best = max((row["loop_savings_fraction"] for row in comparable), default=0)
    best_simple_accuracy = max(row["accuracy"] for row in simple)
    passed = (oracle["accuracy"] >= best_simple_accuracy - quality_tolerance
              and oracle["loop_savings_fraction"] >= best + min_extra_savings)
    return {"passed": passed, "label_rule": label_rule, "oracle_accuracy": oracle["accuracy"],
            "oracle_loop_savings_fraction": oracle["loop_savings_fraction"],
            "best_comparable_simple_savings_fraction": best,
            "best_simple_accuracy": best_simple_accuracy,
            "min_extra_savings": min_extra_savings, "quality_tolerance": quality_tolerance,
            "comparison_policies": ("fixed_budget", "requested_depth", "prediction_stability_k2")}


def fit_head(traces: list[Trace], *, label_rule: str = "first_correct", epochs: int = 100, learning_rate: float = .01,
             seed: int = 17) -> tuple[HaltingHead, dict]:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("Positive epochs and learning rate required")
    inputs, labels = training_examples(traces, label_rule=label_rule)
    torch.manual_seed(seed)
    head = HaltingHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    positive = labels.sum().item()
    weight = torch.tensor((len(labels) - positive) / positive)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(inputs), labels, pos_weight=weight)
        loss.backward()
        optimizer.step()
    head.eval()
    return head, {"training_rows": len(labels), "stop_labels": int(positive),
                  "continue_labels": int(len(labels) - positive), "epochs": epochs,
                  "learning_rate": learning_rate, "seed": seed, "final_train_bce": float(loss.item())}



def development_curve(head: HaltingHead, traces: list[Trace], *, label_rule: str) -> dict:
    """Frozen head on development traces; report a curve and stop-label Brier."""
    inputs, labels = training_examples(traces, label_rule=label_rule)
    with torch.inference_mode():
        probabilities = torch.sigmoid(head(inputs))
    brier = float(((probabilities - labels) ** 2).mean().item())
    curve = {}
    with torch.inference_mode():
        for threshold in (.1, .25, .5, .75, .9):
            stops = []
            for trace in traces:
                stop = trace.budget
                for t in range(trace.depth, trace.budget):
                    features = torch.tensor(_features(trace, t), dtype=torch.float32)
                    if float(torch.sigmoid(head(features)).item()) >= threshold:
                        stop = t
                        break
                stops.append(stop)
            curve[str(threshold)] = summarize_stops(traces, stops)
    return {"stop_label_brier": brier, "threshold_curve": curve,
            "evaluation_split": "development_only"}

def _run(path: Path) -> tuple[dict, list[Trace]]:
    metadata = json.loads((path / "summary.json").read_text())
    if metadata.get("status") != "complete" or metadata.get("task_variant") != "absorbing-terminal-v1":
        raise ValueError("Head training requires a complete absorbing-terminal full-loop run")
    traces = load_traces(path / "trajectories.csv")
    if len(traces) != metadata.get("selected_examples") or traces[0].budget != metadata.get("loops"):
        raise ValueError("Trace coverage differs from run metadata")
    return metadata, traces


def load_head(path: Path, *, checkpoint: Path) -> HaltingHead:
    """Refuse use with a different recurrent checkpoint."""
    manifest = json.loads((path / "manifest.json").read_text())
    adapter = checkpoint / "adapter_model.pt"
    if (manifest.get("checkpoint_adapter_sha256") != sha256_file(adapter)
            or manifest.get("feature_version") != 1):
        raise ValueError("Halting head checkpoint/features do not match the recurrent checkpoint")
    head = HaltingHead()
    head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
    return head.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--dev-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-rule", choices=("first_correct", "suffix"), default="first_correct")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a fresh directory")
    train_meta, train = _run(args.train_run)
    dev_meta, dev = _run(args.dev_run)
    if train_meta.get("data_sha256") == dev_meta.get("data_sha256") or {t.example_id for t in train} & {t.example_id for t in dev}:
        parser.error("Train and development runs must have distinct files and example IDs")
    adapter_hash = train_meta.get("local_checkpoint_sha256", {}).get("adapter_model.pt")
    if not adapter_hash or adapter_hash != dev_meta.get("local_checkpoint_sha256", {}).get("adapter_model.pt"):
        parser.error("Train and development runs must use the same recorded adapter")
    gate = headroom_gate(dev, label_rule=args.label_rule)
    if not gate["passed"]:
        parser.error(f"No declared oracle headroom beyond comparable heuristics: {gate}")
    head, fit = fit_head(train, label_rule=args.label_rule, epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed)
    dev_evaluation = development_curve(head, dev, label_rule=args.label_rule)
    args.output.mkdir(parents=True)
    torch.save(head.state_dict(), args.output / "head.pt")
    manifest = {"status": "fitted_unconfirmed", "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": [sys.executable, "-m", "scripts.eval.adaptive_head", *sys.argv[1:]],
                "feature_version": 1, "label_rule": args.label_rule,
                "label_semantics": "first_correct trains current correctness" if args.label_rule == "first_correct" else "suffix trains finite-horizon stability",
                "checkpoint_adapter_sha256": adapter_hash,
                "train_trajectories_sha256": sha256_file(args.train_run / "trajectories.csv"),
                "dev_trajectories_sha256": sha256_file(args.dev_run / "trajectories.csv"),
                "headroom_gate": gate, "fit": fit, "development_evaluation": dev_evaluation}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Console().print(f"[green]Fitted gated head[/green] · {args.output.resolve()}")


if __name__ == "__main__":
    main()
