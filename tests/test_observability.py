import pytest
import torch

from scripts.recurrent_qwen import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
from scripts.recurrent_qwen.outputs import answer_margin


def test_outputs_and_readout_positions(base, inputs):
    model = RecurrentQwen(base, 1, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    labels = torch.tensor([[5, 7, 9], [9, 5, 7]])
    allowed = torch.tensor([5, 7, 9])
    with torch.no_grad():
        full = model(inputs, mask, num_loops=3, logits_mode="all", return_hidden_states=True,
                     labels=labels, allowed_token_ids=allowed)
        compact = model(inputs, mask, num_loops=3, labels=labels, allowed_token_ids=allowed)
        explicit = model(inputs, mask, num_loops=3, answer_positions=torch.tensor([0, 1]))
    assert full.initial_hidden_state.shape == (2, 4, 32)
    assert len(full.hidden_states) == len(full.loop_logits) == 3
    assert full.margins.shape == (2, 3)
    assert compact.hidden_states is compact.initial_hidden_state is None
    for loop, (hidden, logits) in enumerate(zip(full.hidden_states, full.loop_logits)):
        assert hidden.shape == (2, 4, 32)
        assert logits.shape == (2, 4, 47)
        torch.testing.assert_close(compact.loop_logits[loop], logits[torch.arange(2), [1, 3]])
        torch.testing.assert_close(explicit.loop_logits[loop], logits[torch.arange(2), [0, 1]])
        expected = []
        for row, pos in enumerate([1, 3]):
            target = labels[row, loop].item()
            others = [token for token in allowed.tolist() if token != target]
            expected.append(logits[row, pos, target] - logits[row, pos, others].max())
        torch.testing.assert_close(full.margins[:, loop], torch.stack(expected))
    torch.testing.assert_close(compact.margins, full.margins)


def test_margin_excludes_disallowed_tokens_and_handles_ties():
    logits = torch.tensor([[100., 3., 1., 3.], [100., 2., 5., 1.]], requires_grad=True)
    margin = answer_margin(logits, torch.tensor([1, 2]), torch.tensor([1, 2, 3]))
    torch.testing.assert_close(margin, torch.tensor([0., 3.]))
    margin.sum().backward()
    assert (logits.grad[:, 0] == 0).all()


@pytest.mark.parametrize("allowed,labels", [([1], [1]), ([1, 1], [1]), ([1, 4], [1]), ([1, 2], [3])])
def test_invalid_margin_inputs(allowed, labels):
    with pytest.raises(ValueError):
        answer_margin(torch.zeros(1, 4), torch.tensor(labels), torch.tensor(allowed))


@pytest.mark.parametrize("kwargs", [
    {"num_loops": 0}, {"num_loops": True}, {"logits_mode": "invalid"},
    {"attention_mask": torch.zeros(2, 4)},
    {"attention_mask": torch.full((2, 4), 2)},
    {"answer_positions": torch.tensor([4, 0])},
    {"position_ids": torch.zeros(3, 4, dtype=torch.long)},
    {"labels": torch.tensor([[1], [2]])},
    {"labels": torch.tensor([[1, 2], [1, 2]]), "allowed_token_ids": torch.tensor([1, 2])},
])
def test_invalid_forward_inputs(base, inputs, kwargs):
    with pytest.raises(ValueError):
        RecurrentQwen(base, 1, 3)(inputs, **kwargs)


@pytest.mark.parametrize("split", [(-1, 3), (2, 2), (1, 5)])
def test_invalid_split(base, split):
    with pytest.raises(ValueError):
        RecurrentQwen(base, *split)


def test_rejects_repeated_adapter_attachment(base):
    model = RecurrentQwen(base, 1, 3)
    attach_recurrent_lora(model)
    with pytest.raises(ValueError, match="already attached"):
        attach_recurrent_lora(model)
