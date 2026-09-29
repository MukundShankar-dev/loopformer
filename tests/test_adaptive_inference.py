"""Actual stopped execution and gated head contracts on tiny models/traces."""

import torch
import pytest
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.dataset.pointer import SYMBOLS, generate_example
from scripts.eval.adaptive_head import development_curve, fit_head, headroom_gate, training_examples
from scripts.eval.adaptive_inference import evaluate_stopping
from scripts.eval.adaptive_metrics import Trace
from scripts.eval.adaptive_policy import HaltingHead, StoppingRule
from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.training.data import EncodedExample


def tiny_model():
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        tie_word_embeddings=True, pad_token_id=0, eos_token_id=1,
        attn_implementation="eager",
    )).eval()
    return RecurrentQwen(base, 1, 2).eval()


def test_stopping_reuses_state_and_preserves_fixed_forward():
    torch.manual_seed(17)
    model = tiny_model()
    inputs = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        fixed = model(inputs, num_loops=4)
        never = model(inputs, num_loops=4, stop_policy=lambda t, h, s: False)
        rule = StoppingRule("requested_depth", depth=2, budget=4, token_ids=list(range(3, 29)))
        stopped = model(inputs, num_loops=4, stop_policy=rule)
    assert len(fixed.loop_logits) == len(never.loop_logits) == 4
    assert len(stopped.loop_logits) == 2
    for index in range(4):
        torch.testing.assert_close(fixed.loop_logits[index], never.loop_logits[index])
    torch.testing.assert_close(fixed.loop_logits[1], stopped.loop_logits[1])
    assert len(rule.history) == 2
    with pytest.raises(ValueError, match="eval/no_grad"):
        model(inputs, num_loops=4, stop_policy=rule)


def test_stopped_evaluator_measures_actual_loops_and_latency():
    torch.manual_seed(17)
    model = tiny_model()
    tasks = [generate_example(20 + depth, depth, "test", depth) for depth in (1, 3)]
    items = [EncodedExample(task, [3, 4, 5], [SYMBOLS.index(s) for s in task.intermediate_states]) for task in tasks]
    rows, decisions, summary = evaluate_stopping(model, items, list(range(3, 29)), 0,
                                                   loops=4, policy="requested_depth")
    assert [r["executed_loops"] for r in decisions] == [1, 3]
    assert summary["executed_example_loops"] == 4 and summary["fixed_budget_example_loops"] == 8
    assert summary["loop_savings_fraction"] == .5
    assert summary["p95_latency_seconds"] > 0
    assert len(rows) == 4 and all(r["stopped"] == (r["loop"] == r["task_depth"]) for r in rows)
    assert model.training is False


def test_head_gate_and_supervised_labels():
    traces = [Trace(str(i), 2, ("B", "B", "A", "A", "A"),
                    (False, False, True, True, True), False,
                    (1., 1., 2., 3., 3.), (2., 2., 1., .5, .5)) for i in range(4)]
    gate = headroom_gate(traces)
    assert gate["passed"] and gate["oracle_loop_savings_fraction"] == .4
    features, labels = training_examples(traces)
    assert features.shape == (12, 5) and labels.sum() == 8
    head, fit = fit_head(traces, epochs=3)
    assert isinstance(head, HaltingHead) and fit["training_rows"] == 12
    assert head(features).shape == (12,)
    curve = development_curve(head, traces, label_rule="first_correct")
    assert set(curve["threshold_curve"]) == {"0.1", "0.25", "0.5", "0.75", "0.9"}
    assert 0 <= curve["stop_label_brier"] <= 1
    with pytest.raises(ValueError, match="predicted_margin"):
        training_examples([Trace("x", 2, traces[0].predictions, traces[0].correct, False,
                                 (None,) * 5, (None,) * 5)])


def test_causal_rule_thresholds_and_head_identity(tmp_path):
    from scripts.eval.adaptive_head import load_head
    from scripts.eval.pointer_task import sha256_file
    logits = torch.zeros(1, 32)
    logits[0, 3] = 4
    hidden = torch.zeros(1, 2, 16)
    margin = StoppingRule("margin", depth=2, budget=4, token_ids=[3, 4], threshold=2)
    assert not margin(1, hidden, logits) and margin(2, hidden, logits)
    entropy = StoppingRule("entropy", depth=2, budget=4, token_ids=[3, 4], threshold=.2)
    assert not entropy(1, hidden, logits) and entropy(2, hidden, logits)
    stability = StoppingRule("stability", depth=2, budget=4, token_ids=[3, 4], k=2)
    assert not stability(1, hidden, logits) and stability(2, hidden, logits)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.pt").write_bytes(b"toy-adapter")
    directory = tmp_path / "head"
    directory.mkdir()
    head = HaltingHead()
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.fill_(10)
    torch.save(head.state_dict(), directory / "head.pt")
    import json
    (directory / "manifest.json").write_text(json.dumps({
        "feature_version": 1, "checkpoint_adapter_sha256": sha256_file(checkpoint / "adapter_model.pt")}))
    loaded = load_head(directory, checkpoint=checkpoint)
    rule = StoppingRule("learned", depth=2, budget=4, token_ids=[3, 4],
                        threshold=.5, head=loaded)
    assert not rule(1, hidden, logits) and rule(2, hidden, logits)
    assert rule.last_probability > .5
    (checkpoint / "adapter_model.pt").write_bytes(b"different")
    with pytest.raises(ValueError, match="do not match"):
        load_head(directory, checkpoint=checkpoint)
