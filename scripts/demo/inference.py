"""Stream native loop observations without reference-state feedback."""

from dataclasses import dataclass
import math
from queue import Queue
from threading import Event, Thread
from time import perf_counter
from typing import Any, Iterator

import torch

from scripts.dataset.pointer import SYMBOLS, execute, render_prompt


SAFETY_CAP = 272
STOP_THRESHOLD = 0.5


@dataclass(frozen=True)
class DemoRequest:
    """Ordered full A–Z table; order is part of the model's actual prompt."""

    pairs: tuple[tuple[str, str], ...]
    start: str
    steps: int

    @classmethod
    def parse(cls, value: Any, *, max_steps: int = 256) -> "DemoRequest":
        if not isinstance(value, dict) or set(value) != {"mapping", "start", "steps"}:
            raise ValueError("Supply exactly mapping, start, and steps")
        pairs = value["mapping"]
        if not isinstance(pairs, list) or len(pairs) != 26:
            raise ValueError("Supply all 26 A–Z rules, with one destination per node")
        if any(not isinstance(p, list) or len(p) != 2 or
               any(not isinstance(s, str) or s not in SYMBOLS for s in p) for p in pairs):
            raise ValueError("Each rule must be a pair of uppercase A–Z letters")
        if {p[0] for p in pairs} != set(SYMBOLS):
            raise ValueError("Each A–Z source must occur exactly once")
        if not isinstance(value["start"], str) or value["start"] not in SYMBOLS:
            raise ValueError("Start must be an uppercase A–Z letter")
        if type(value["steps"]) is not int or not 1 <= value["steps"] <= max_steps:
            raise ValueError(f"Steps must be an integer from 1 to {max_steps}")
        return cls(tuple(tuple(p) for p in pairs), value["start"], value["steps"])

    def payload(self) -> dict:
        return {"mapping": [list(p) for p in self.pairs], "start": self.start, "steps": self.steps}

    def prompt(self) -> str:
        return render_prompt([list(p) for p in self.pairs], self.start, self.steps)


class InferenceCancelled(Exception):
    """The client closed its generator; stop at the next completed readout."""


def score_trace(request: DemoRequest, rows: list[dict], stop_reason: str) -> dict:
    """Final letter, strict requested prefix and timing are separate metrics."""
    expected = execute(dict(request.pairs), request.start, request.steps)
    predictions = [row["prediction"] for row in rows]
    exact_stop = stop_reason == "head" and len(rows) == request.steps
    correct = bool(predictions) and predictions[-1] == expected[-1]
    complete = len(predictions) >= request.steps and predictions[:request.steps] == expected
    return {"answer_correct": correct, "complete_trajectory": complete,
            "exact_stop": exact_stop, "joint_success": correct and exact_stop,
            "stop_reason": stop_reason, "executed_loops": len(rows),
            "first_error": next((r["loop"] for r in rows[:request.steps] if not r["correct"]), None)}


def stream_inference(model: Any, tokenizer: Any, spec: dict, request: DemoRequest,
                     *, checkpoint_id: str, cancel: Event | None = None,
                     loop_cap: int | None = None) -> Iterator[dict]:
    """Yield cumulative JSON snapshots as loops arrive, then a final summary.

    One worker owns one native forward. The bounded 272-loop cap also bounds
    the queue. A consumer must serialize calls to its model (Gradio concurrency
    one). Closing the iterator cancels and joins the worker before releasing GPU
    ownership. Targets are computed only in the consumer, never passed to R.
    """
    if model.training or model.completion_head is None:
        raise ValueError("Demo requires an eval-mode model with learned completion")
    loop_cap = SAFETY_CAP if loop_cap is None else loop_cap
    if type(loop_cap) is not int or loop_cap < 1:
        raise ValueError("Loop cap must be a positive integer")
    cancel = cancel if cancel is not None else Event()
    queue: Queue = Queue()
    rows: list[dict] = []
    mapping = dict(request.pairs)
    reference = request.start
    previous_prediction = request.start
    started = perf_counter()
    metadata = {"schema_version": 1, "source": "live", "checkpoint": checkpoint_id,
                "request": request.payload(), "threshold": STOP_THRESHOLD, "safety_cap": loop_cap}

    def observe(loop, scores, stop_logits, state_logits):
        if cancel.is_set():
            raise InferenceCancelled()
        scores = scores[0].float().cpu()
        probability = float(stop_logits.sigmoid().item())
        if not torch.isfinite(scores).all() or not math.isfinite(probability):
            raise FloatingPointError("Nonfinite model output")
        top = scores.topk(3)
        probabilities = scores.softmax(-1)
        queue.put(("loop", {"loop": loop, "prediction": SYMBOLS[int(scores.argmax())],
                   "direct_prediction": SYMBOLS[int(state_logits[0].argmax())] if state_logits is not None else None,
                   "stop_probability": probability,
                   "top3": [{"symbol": SYMBOLS[i], "probability": float(probabilities[i])}
                            for i in top.indices.tolist()],
                   "arrival_seconds": perf_counter() - started}))

    def run():
        try:
            if cancel.is_set():
                raise InferenceCancelled()
            device = next(model.parameters()).device
            encoded = tokenizer(request.prompt(), add_special_tokens=False, return_tensors="pt")
            with torch.inference_mode():
                result = model(input_ids=encoded["input_ids"].to(device),
                               attention_mask=encoded["attention_mask"].to(device),
                               num_loops=loop_cap, completion_threshold=STOP_THRESHOLD,
                               readout_token_ids=spec["token_ids"], loop_callback=observe)
            signaled = result.stop_logits[0, -1].sigmoid().item() >= STOP_THRESHOLD
            queue.put(("done", "head" if signaled else "cap"))
        except InferenceCancelled:
            queue.put(("cancelled", None))
        except Exception as error:
            # Transport the original failure to the owning generator, never hide it.
            queue.put(("error", error))

    worker = Thread(target=run, name="loopformer-inference", daemon=True)
    worker.start()
    try:
        while True:
            kind, value = queue.get()
            if kind == "error":
                raise value
            if kind == "cancelled":
                return
            if kind == "done":
                yield {**metadata, "status": "complete", "trace": list(rows),
                       "summary": score_trace(request, rows, value),
                       "inference_seconds": perf_counter() - started}
                return
            reference = mapping[reference]
            value.update(expected=reference, correct=value["prediction"] == reference,
                         edge_valid=mapping[previous_prediction] == value["prediction"])
            previous_prediction = value["prediction"]
            rows.append(value)
            yield {**metadata, "status": "streaming", "trace": list(rows)}
    finally:
        cancel.set()
        worker.join()
