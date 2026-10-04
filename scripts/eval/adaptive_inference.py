"""Stopped pointer inference, composed by the existing full-loop checkpoint CLI."""

import math
from statistics import mean
from time import perf_counter
from typing import Callable

import torch

from scripts.dataset.pointer import SYMBOLS
from scripts.training.data import EncodedExample, collate
from scripts.training.objective import forward_symbols
from .adaptive_policy import HaltingHead, StoppingRule
from .pointer_task import synchronize


def evaluate_stopping(model: torch.nn.Module, items: list[EncodedExample], token_ids: list[int], pad_id: int,
                      *, loops: int, policy: str, k: int = 2, threshold: float = .5,
                      head: HaltingHead | None = None,
                      progress: Callable[[int, int], None] | None = None) -> tuple[list[dict], list[dict], dict]:
    """Execute batch-1 recurrence until stop; measure synchronized per-example time.

    Timing includes prelude, recurrent steps, coda readouts and policy; excludes
    loading, tokenization, CPU export and CSV writes. No hidden state is reused
    across examples. Complete trajectory accuracy requires every nominal step
    to have been executed correctly. Historical policies wait until task depth; completion uses only the
    checkpoint's hidden-state head and can stop before depth or hit a smaller cap.
    """
    completion = policy == "completion"
    if not items or loops < 1 or (not completion and loops < max(len(item.targets) for item in items)):
        raise ValueError("Loop budget must cover every selected task")
    if completion and (getattr(model, "completion_head", None) is None or head is not None or
                       not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("Completion stopping requires the checkpoint head and threshold in (0,1)")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    trajectories, decisions, latencies = [], [], []
    try:
        with torch.inference_mode():
            for item in items:
                batch = collate([item], pad_id, str(device))
                rule = None if completion else StoppingRule(policy, depth=len(item.targets), budget=loops,
                                                            token_ids=token_ids, k=k, threshold=threshold, head=head)
                synchronize(device)
                start = perf_counter()
                stopping = {"completion_threshold": threshold} if completion else {"stop_policy": rule}
                if completion:
                    result, selected_scores = forward_symbols(model, batch["input_ids"], batch["attention_mask"],
                                                              token_ids, num_loops=loops, **stopping)
                else:
                    result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   num_loops=loops, **stopping)
                synchronize(device)
                latency = perf_counter() - start
                latencies.append(latency)
                predictions, features = [], []
                for t, logits in enumerate(result.loop_logits, 1):
                    allowed = selected_scores[0, t - 1] if completion else logits[0, token_ids]
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
                    if completion:
                        trajectories[-1].update(stop_logit=result.stop_logits[0, t - 1].item(),
                                                stop_probability=result.stop_logits[0, t - 1].sigmoid().item())
                stopped = len(predictions)
                if not completion and stopped < len(item.targets):
                    raise RuntimeError("Stopping before nominal task depth")
                probability = result.stop_logits[0, -1].sigmoid().item() if completion else rule.last_probability
                decisions.append({"example_id": item.task.example_id, "task_depth": len(item.targets),
                                  "budget": loops, "executed_loops": stopped, "policy": policy,
                                  "prediction": predictions[-1], "target": item.task.final_state,
                                  "correct": predictions[-1] == item.task.final_state,
                                  "complete_trajectory": predictions[:len(item.targets)] == list(item.task.intermediate_states),
                                  "predicted_margin": features[-1][0], "answer_entropy": features[-1][1],
                                  "stop_probability": probability if probability is not None else "",
                                  "latency_seconds": latency})
                if completion:
                    signaled = probability >= threshold
                    depth = len(item.targets)
                    decisions[-1].update(stop_reason="head" if signaled else "cap",
                        exact_stop=signaled and stopped == depth, early_stop=signaled and stopped < depth,
                        late_stop=signaled and stopped > depth, cap_fallback=not signaled,
                        joint_success=signaled and stopped == depth and predictions[-1] == item.task.final_state)
                if progress:
                    progress(len(decisions), len(items))
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
    if completion:
        def rates(rows: list[dict]) -> dict:
            return {"examples": len(rows), "accuracy": mean(row["correct"] for row in rows),
                    **{name + "_rate": mean(row[name] for row in rows)
                       for name in ("exact_stop", "early_stop", "late_stop", "cap_fallback", "joint_success")}}
        summary["completion"] = rates(decisions)
        summary["by_depth"] = {str(d): rates([row for row in decisions if row["task_depth"] == d])
                               for d in sorted({row["task_depth"] for row in decisions})}
        summary["timing_scope"] = "Synchronized batch-1 forward including memory preparation, every coda readout, completion head and threshold check; excludes tokenization, loading and export"
    return trajectories, decisions, summary
