"""Fixed prompt memory, learned stopping and portable architecture contracts."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
from scripts.recurrent_qwen.checkpoint import checkpoint_mode, restore_adapters
from scripts.training.config import TrainingConfig
from scripts.training.gates import initialize, validate_completion_gradients
from scripts.training.runner import resume_identity, validate_resume_identity
from scripts.eval.adaptive_inference import evaluate_stopping
from scripts.dataset.pointer import SYMBOLS, generate_example
from scripts.training.data import EncodedExample


@pytest.mark.parametrize('padding', ['none', 'left', 'right'])
@pytest.mark.parametrize('split', [(0, 4), (1, 3), (0, 1), (3, 4)])
def test_one_pass_matches_base_and_layer_prefixes_stay_fixed(base, padding, split):
    x = torch.tensor([[3, 5, 7, 9], [11, 13, 17, 19]])
    mask = torch.ones_like(x)
    if padding != 'none':
        mask[0, 0 if padding == 'left' else -1] = 0
    with torch.no_grad():
        expected = base(x, attention_mask=mask, use_cache=False).logits
    model = RecurrentQwen(base, *split, recurrence_mode='fixed_prompt')
    captured = [[] for _ in [*model.recurrent.layers, *model.coda.layers]]
    hooks = [layer.register_forward_pre_hook(lambda m, args, target=target: target.append(args[0].detach().clone()))
             for layer, target in zip([*model.recurrent.layers, *model.coda.layers], captured)]
    try:
        with torch.no_grad():
            result = model(x, mask, num_loops=4, logits_mode='all', return_hidden_states=True)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(result.loop_logits[0], expected, atol=1e-5, rtol=1e-5)
    positions = torch.arange(4).expand(2, -1).masked_fill(~mask.bool(), -1).max(-1).values
    prefix = (torch.arange(4)[None, :] < positions[:, None]) & mask.bool()
    for calls in captured:
        assert len(calls) == 4
        for later in calls[1:]:
            torch.testing.assert_close(later[prefix], calls[0][prefix], rtol=0, atol=0)
    for later in result.hidden_states[1:]:
        torch.testing.assert_close(later[prefix], result.hidden_states[0][prefix], rtol=0, atol=0)
    assert not torch.equal(result.hidden_states[0][1, -1], result.hidden_states[-1][1, -1])


def test_reused_memory_matches_independent_recomputation_and_gradients(base, inputs):
    """Independent reference resets the P prefix and recomputes it every pass.

    In a causal decoder its prefix cannot depend on the final working position,
    so the gradient must equal reuse of the differentiable first-pass memory.
    """
    model = RecurrentQwen(base, 1, 3, recurrence_mode='fixed_prompt')
    attach_recurrent_lora(model, rank=2, alpha=4)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.03)
    reference = deepcopy(model)
    result = model(inputs, num_loops=3, return_hidden_states=True)
    for state in result.hidden_states:
        state.retain_grad()
    # Run the independent reference through its existing full-sequence block.
    from transformers.masking_utils import create_causal_mask
    positions = torch.arange(inputs.shape[1])[None]
    args = (create_causal_mask(config=reference.config, inputs_embeds=reference.embed_tokens(inputs),
            attention_mask=torch.ones_like(inputs), past_key_values=None, position_ids=positions),
            positions, reference.rotary_emb(reference.embed_tokens(inputs), positions))
    initial = reference.prelude(reference.embed_tokens(inputs), *args)
    state = initial[:, -1:]
    expected = []
    for _ in range(3):
        hidden = reference.recurrent(torch.cat([initial[:, :-1], state], dim=1), *args)
        state = hidden[:, -1:]
        expected.append(reference.lm_head(reference.norm(reference.coda(hidden, *args))[:, -1]))
    for actual, wanted in zip(result.loop_logits, expected):
        torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=1e-5)
    result.logits.square().sum().backward()
    expected[-1].square().sum().backward()
    assert all(s.grad is not None and s.grad[:, -1].abs().sum() > 0 for s in result.hidden_states)
    for (name, actual), (_, wanted) in zip(model.named_parameters(), reference.named_parameters()):
        if actual.requires_grad:
            assert actual.grad is not None and wanted.grad is not None, name
            torch.testing.assert_close(actual.grad, wanted.grad, atol=1e-6, rtol=1e-4)
        else:
            assert actual.grad is None and wanted.grad is None


def test_fixed_memory_gate_head_scope_and_call_isolation(base, inputs):
    config = replace(TrainingConfig(), recurrence_mode='fixed_prompt', recurrent_start=1,
                     recurrent_end=3, lora_rank=2, lora_alpha=4)
    batch = dict(input_ids=inputs, attention_mask=torch.ones_like(inputs), targets=torch.tensor([[0], [1]]))
    model, gate = initialize(base, config, batch, [3, 5, 7])
    assert gate['passed']
    model.enable_completion(8)
    assert validate_completion_gradients(model, batch)['passed']
    with torch.no_grad():
        first = model(inputs, num_loops=3)
        model(inputs.flip(1), num_loops=4)
        again = model(inputs, num_loops=3)
        short = model(inputs, num_loops=1)
    torch.testing.assert_close(first.logits, again.logits, rtol=0, atol=0)
    torch.testing.assert_close(first.loop_logits[0], short.logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match='final unmasked'):
        model(inputs, answer_positions=torch.tensor([1, 2]))
    with pytest.raises(ValueError, match='recurrence_mode'):
        RecurrentQwen(base, recurrence_mode='unknown')
    with pytest.raises(ValueError, match='differs'):
        validate_resume_identity(resume_identity(replace(config, recurrence_mode='full_sequence'), {}),
                                 resume_identity(config, {}))


def test_completion_stops_without_depth_input_and_reports_cap(base):
    model = RecurrentQwen(base, 1, 3, recurrence_mode='fixed_prompt').eval()
    model.enable_completion(8)
    tasks = [generate_example(32 + d, d, 'test', d) for d in (1, 3)]
    items = [EncodedExample(t, [3, 4, 5], [SYMBOLS.index(s) for s in t.intermediate_states]) for t in tasks]
    for p in model.completion_head.parameters():
        p.data.zero_()
    model.completion_head.network[-1].bias.data.fill_(10)
    with torch.no_grad():
        full = model(torch.tensor([[3, 4, 5]]), num_loops=4)
        stopped = model(torch.tensor([[3, 4, 5]]), num_loops=4, completion_threshold=.5)
    assert len(stopped.loop_logits) == 1
    torch.testing.assert_close(stopped.logits, full.loop_logits[0], rtol=0, atol=0)
    seen = []
    hook = model.register_forward_pre_hook(lambda m, args, kwargs: seen.append(kwargs.copy()), with_kwargs=True)
    try:
        rows, decisions, summary = evaluate_stopping(model, items, list(range(3, 29)), 0, loops=4, policy='completion')
    finally:
        hook.remove()
    assert all(set(k) == {'input_ids', 'attention_mask', 'num_loops', 'completion_threshold', 'readout_token_ids'} for k in seen)
    assert all(k['readout_token_ids'] == list(range(3, 29)) for k in seen)
    assert all(k['num_loops'] == 4 for k in seen)
    assert [r['executed_loops'] for r in decisions] == [1, 1]
    assert summary['completion']['exact_stop_rate'] == .5
    assert summary['completion']['early_stop_rate'] == .5
    assert summary['completion']['cap_fallback_rate'] == 0
    assert len(rows) == 2 and 'stop_probability' in rows[0]
    model.completion_head.network[-1].bias.data.fill_(-10)
    _, decisions, summary = evaluate_stopping(model, items, list(range(3, 29)), 0, loops=1, policy='completion')
    assert all(r['stop_reason'] == 'cap' for r in decisions)
    assert summary['completion']['cap_fallback_rate'] == 1
    assert summary['completion']['exact_stop_rate'] == 0  # cap at d is not learned completion
    assert summary['completion']['joint_success_rate'] == 0
    with pytest.raises(ValueError, match='eval/no_grad'):
        model(torch.tensor([[3, 4, 5]]), completion_threshold=.5)


def test_checkpoint_modes_fail_closed(base, tmp_path):
    assert checkpoint_mode({'format': 'loopformer-stage1-v1'}) == 'full_sequence'
    with pytest.raises(ValueError, match='disagree'):
        checkpoint_mode({'format': 'loopformer-stage1-fixed-prompt-v1'})
    model = RecurrentQwen(base, 1, 3, recurrence_mode='fixed_prompt')
    (tmp_path / 'recurrent_config.json').write_text(json.dumps({'format': 'loopformer-stage1-completion-v1'}))
    with pytest.raises(ValueError, match='recurrence_mode'):
        restore_adapters(model, tmp_path)


@pytest.mark.parametrize('crossing', [1, 2, 3, None])
def test_completion_timing_categories_and_stopped_prefix(base, crossing):
    # A prescribed test head isolates timing and scoring from learned quality.
    class PrescribedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            self.calls += 1
            return hidden.new_full((len(hidden),), 10. if self.calls == crossing else -10.)

    model = RecurrentQwen(base, 1, 3, recurrence_mode='fixed_prompt').eval()
    model.completion_head = PrescribedHead()
    task = generate_example(19, 2, 'test', 0)
    item = EncodedExample(task, [3, 4, 5], [SYMBOLS.index(s) for s in task.intermediate_states])
    rows, decisions, summary = evaluate_stopping(model, [item], list(range(3, 29)), 0, loops=4, policy='completion')
    assert len(rows) == (crossing or 4) == decisions[0]['executed_loops']
    assert model.completion_head.calls == len(rows)
    for name, expected in [('exact_stop', crossing == 2), ('early_stop', crossing == 1),
                           ('late_stop', crossing == 3), ('cap_fallback', crossing is None)]:
        assert summary['completion'][name + '_rate'] == float(expected)
    with torch.no_grad():
        full = model(torch.tensor([[3, 4, 5]]), num_loops=4)
    for t, row in enumerate(rows):
        assert row['prediction'] == SYMBOLS[int(full.loop_logits[t][0, list(range(3, 29))].argmax())]
