"""Stopped pointer inference, composed by the existing full-loop checkpoint CLI."""

from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from scripts.dataset.pointer import SYMBOLS
from scripts.training.data import EncodedExample, collate
from .adaptive_policy import HaltingHead, StoppingRule
from .pointer_task import synchronize


def evaluate_stopping(model: torch.nn.Module, items: list[EncodedExample], token_ids: list[int], pad_id: int,
                      *, loops: int, policy: str, k: int = 2, threshold: float = .5,
                      head: HaltingHead | None = None) -> tuple[list[dict], list[dict], dict]:
    """Execute batch-1 recurrence until stop; measure synchronized per-example time.

    Timing includes prelude, recurrent steps, coda readouts and policy; excludes
    loading, tokenization, CPU export and CSV writes. No hidden state is reused
    across examples. Complete trajectory accuracy only scores observed nominal
    steps; every policy waits at least until the example's task depth.
    """
    if not items or loops < max(len(item.targets) for item in items):
        raise ValueError("Loop budget must cover every selected task")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    trajectories, decisions, latencies = [], [], []
    try:
        with torch.inference_mode():
            for item in items:
                batch = collate([item], pad_id, str(device), loops)
                rule = StoppingRule(policy, depth=len(item.targets), budget=loops,
                                    token_ids=token_ids, k=k, threshold=threshold, head=head)
                synchronize(device)
                start = perf_counter()
                result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               num_loops=loops, stop_policy=rule)
                synchronize(device)
                latency = perf_counter() - start
                latencies.append(latency)
                predictions, features = [], []
                for t, logits in enumerate(result.loop_logits, 1):
                    allowed = logits[0, token_ids]
                    prediction = int(allowed.argmax())
                    predicted = SYMBOLS[prediction]
                    predictions.append(predicted)
                    probs = allowed.softmax(-1)
                    margin = float((allowed.topk(2).values[0] - allowed.topk(2).values[1]).item())
                    entropy = float((-(probs * probs.clamp_min(1e-30).log()).sum()).item())
                    features.append((margin, entropy))
                    target = item.task.intermediate_states[t - 1] if t <= len(item.targets) else ""
                    trajectories.append({"example_id": item.task.example_id, "task_depth": len(item.targets),
                                         "loop": t, "prediction": predicted, "intermediate_target": target,
                                         "intermediate_correct": predicted == target if target else "",
                                         "final_target": item.task.final_state,
                                         "final_correct": predicted == item.task.final_state,
                                         "predicted_margin": margin, "answer_entropy": entropy,
                                         "stopped": t == len(result.loop_logits)})
                stopped = len(predictions)
                if stopped < len(item.targets):
                    raise RuntimeError("Stopping before nominal task depth")
                decisions.append({"example_id": item.task.example_id, "task_depth": len(item.targets),
                                  "budget": loops, "executed_loops": stopped, "policy": policy,
                                  "prediction": predictions[-1], "target": item.task.final_state,
                                  "correct": predictions[-1] == item.task.final_state,
                                  "complete_trajectory": predictions[:len(item.targets)] == list(item.task.intermediate_states),
                                  "predicted_margin": features[-1][0], "answer_entropy": features[-1][1],
                                  "stop_probability": rule.last_probability if rule.last_probability is not None else "",
                                  "latency_seconds": latency})
    finally:
        model.train(was_training)
    ordered_latency = sorted(latencies)
    ordered_loops = sorted(row["executed_loops"] for row in decisions)
    n = len(items)
    def percentile(values: list[float | int], level: int) -> float | int:
        return values[(n * level + 99) // 100 - 1]
    summary = {"examples": n, "accuracy": mean(row["correct"] for row in decisions),
               "complete_trajectory_accuracy": mean(row["complete_trajectory"] for row in decisions),
               "executed_example_loops": sum(row["executed_loops"] for row in decisions),
               "fixed_budget_example_loops": n * loops,
               "loop_savings_fraction": 1 - sum(row["executed_loops"] for row in decisions) / (n * loops),
               "mean_loops": mean(row["executed_loops"] for row in decisions),
               "p50_loops": percentile(ordered_loops, 50), "p95_loops": percentile(ordered_loops, 95),
               "total_inference_seconds": sum(latencies), "mean_latency_seconds": mean(latencies),
               "p50_latency_seconds": percentile(ordered_latency, 50),
               "p95_latency_seconds": percentile(ordered_latency, 95),
               "timing_scope": "Synchronized per-example model forward and stopping callback; batch 1; excludes tokenization, loading, and CSV export"}
    return trajectories, decisions, summary
