"""Native streaming fidelity, boundary validation and explicit stopping results."""

from copy import deepcopy
from threading import Event

import pytest
import torch

from scripts.dataset.pointer import SYMBOLS
from scripts.demo.inference import DemoRequest, score_trace, stream_inference
from scripts.recurrent_qwen.model import RecurrentQwen
from test_executor_upgrade import setup, configured


def payload(steps=9):
    return {"mapping": [[s, SYMBOLS[(i + 1) % 26]] for i, s in enumerate(SYMBOLS)],
            "start": "A", "steps": steps}


@pytest.mark.parametrize("change", [
    {"steps": True}, {"steps": 0}, {"steps": 257}, {"steps": 1.2},
    {"start": "AA"}, {"mapping": [["A", "B"]] * 26},
    {"mapping": [[s, "<script>"] for s in SYMBOLS]}, {"extra": 1},
])
def test_request_rejects_invalid_boundaries(change):
    with pytest.raises(ValueError):
        DemoRequest.parse({**payload(), **change})


def test_request_preserves_order_and_explicit_limits():
    data = payload(256)
    data["mapping"].reverse()
    request = DemoRequest.parse(data)
    assert request.payload() == data
    assert request.prompt().startswith("Rules: ( Z, A)")
    with pytest.raises(ValueError):
        DemoRequest.parse(data, max_steps=26)


def test_observer_cannot_change_logits_or_stopping(base, inputs):
    model = RecurrentQwen(base, 1, 3).eval()
    model.enable_completion(8)
    model.eval()
    seen = []
    def observe(t, scores, stop, state):
        seen.append((t, scores.clone(), stop.clone()))
        scores.fill_(123)
        stop.fill_(123)
    with torch.inference_mode():
        expected = model(inputs[:1], num_loops=5, completion_threshold=.5)
        actual = model(inputs[:1], num_loops=5, completion_threshold=.5, loop_callback=observe)
    assert len(seen) == len(expected.loop_logits)
    torch.testing.assert_close(actual.stop_logits, expected.stop_logits, rtol=0, atol=0)
    for t, scores, _ in seen:
        torch.testing.assert_close(scores, expected.loop_logits[t - 1], rtol=0, atol=0)
        torch.testing.assert_close(actual.loop_logits[t - 1], scores, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Loop observation"):
        model(inputs[:1], loop_callback=observe)


@pytest.mark.parametrize("stop_bias,reason", [(10., "head"), (-10., "cap")])
def test_stream_matches_native_and_keeps_final_snapshot(setup, monkeypatch, stop_bias, reason):
    base, tokenizer, tokens, _, _ = setup
    model = configured(base, tokenizer, prefix_reuse=True)
    with torch.no_grad():
        model.completion_head.readout.weight.zero_()
        model.completion_head.readout.bias.fill_(stop_bias)
    monkeypatch.setattr("scripts.demo.inference.SAFETY_CAP", 4)
    request = DemoRequest.parse(payload(3))
    encoded = tokenizer(request.prompt(), add_special_tokens=False, return_tensors="pt")
    with torch.inference_mode():
        native = model(**encoded, num_loops=4, completion_threshold=.5,
                       readout_token_ids=list(tokens.values()))
    frames = list(stream_inference(model, tokenizer, {"token_ids": list(tokens.values())},
                                  request, checkpoint_id="tiny-test"))
    assert frames[-1]["status"] == "complete"
    assert frames[-1]["summary"]["stop_reason"] == reason
    assert frames[-1]["summary"]["exact_stop"] is False
    assert len(frames) == len(native.loop_logits) + 1
    assert len(frames[0]["trace"]) == 1
    for row, scores in zip(frames[-1]["trace"], native.loop_logits, strict=True):
        assert row["prediction"] == SYMBOLS[int(scores[0].argmax())]
    event = Event()
    event.set()
    assert list(stream_inference(model, tokenizer, {"token_ids": list(tokens.values())},
                                request, checkpoint_id="tiny-test", cancel=event)) == []


def test_wrong_length_correct_letter_is_not_joint_success():
    request = DemoRequest.parse(payload(27))
    rows = [{"loop": 1, "prediction": "B", "correct": True}]
    result = score_trace(request, rows, "head")
    assert result["answer_correct"] is True
    assert result["joint_success"] is False
    assert result["complete_trajectory"] is False


def test_stream_propagates_inference_failure(setup):
    base, tokenizer, tokens, _, _ = setup
    model = configured(base, tokenizer)
    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")
    model.forward = fail
    with pytest.raises(RuntimeError, match="inference failed"):
        list(stream_inference(model, tokenizer, {"token_ids": list(tokens.values())},
                              DemoRequest.parse(payload()), checkpoint_id="tiny-test"))
