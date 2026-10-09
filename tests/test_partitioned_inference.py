"""Compare exact prefix factorization with original dense model execution."""
from copy import deepcopy

import pytest
import torch

from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
from scripts.recurrent_qwen.partitioned import partitioned_forward


@pytest.mark.parametrize('mode', ['full_sequence', 'fixed_prompt'])
@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
def test_partitioned_preserves_count_specific_states_and_stop_logits(base, mode, attention):
    model = RecurrentQwen(deepcopy(base), 1, 3, recurrence_mode=mode).eval()
    model.config._attn_implementation = attention
    attach_recurrent_lora(model, rank=2, alpha=4)
    model.enable_completion(8)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.02)
    prefixes = torch.tensor([[3, 5, 7], [11, 13, 17]])
    suffixes = torch.tensor([[9, 19, 0], [23, 29, 31], [9, 19, 0], [23, 29, 31]])
    mask = (suffixes != 0).long()
    indices = torch.tensor([0, 0, 1, 1])
    ids = torch.cat((prefixes[indices], suffixes), 1)
    fullmask = torch.cat((torch.ones(4, 3, dtype=torch.long), mask), 1)
    tokens = [3, 5, 7, 9]
    with torch.inference_mode():
        dense = model(ids, fullmask, num_loops=9, readout_token_ids=tokens, return_hidden_states=True)
        factored = partitioned_forward(model, prefixes, suffixes, mask, indices, tokens, 9)
    torch.testing.assert_close(factored.scores, torch.stack(dense.loop_logits, 1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(factored.stops, dense.stop_logits, atol=1e-5, rtol=1e-5)
    ends = fullmask.sum(-1) - 1
    torch.testing.assert_close(factored.working, dense.hidden_states[-1][torch.arange(4), ends], atol=1e-5, rtol=1e-5)
    assert torch.equal(factored.scores.argmax(-1), torch.stack(dense.loop_logits, 1).argmax(-1))
    assert not torch.equal(factored.scores[0], factored.scores[1])


@pytest.mark.parametrize('mode', ['full_sequence', 'fixed_prompt'])
def test_readout_horizon_pruning_preserves_all_observed_predictions(base, mode):
    model=RecurrentQwen(deepcopy(base),1,3,recurrence_mode=mode).eval()
    prefix=torch.tensor([[3,5]])
    suffix=torch.tensor([[7,9],[11,13],[17,19]])
    mask=torch.ones_like(suffix);indices=torch.zeros(3,dtype=torch.long)
    tokens=[3,5,7,9];horizons=torch.tensor([1,3,8])
    full=partitioned_forward(model,prefix,suffix,mask,indices,tokens,8)
    short=partitioned_forward(model,prefix,suffix,mask,indices,tokens,8,horizons)
    for i,n in enumerate(horizons):
        torch.testing.assert_close(short.scores[i,:n],full.scores[i,:n],atol=1e-5,rtol=1e-5)
    model.enable_completion(8)
    for parameter in model.completion_head.parameters(): parameter.data.zero_()
    model.completion_head.network[-1].bias.data.fill_(10)
    full=partitioned_forward(model,prefix,suffix,mask,indices,tokens,8)
    short=partitioned_forward(model,prefix,suffix,mask,indices,tokens,8,horizons)
    assert torch.equal(short.stops.argmax(-1),full.stops.argmax(-1))
    for i,n in enumerate(horizons):
        torch.testing.assert_close(short.scores[i,:n],full.scores[i,:n],atol=1e-5,rtol=1e-5)
    # A head that never crosses must retain every row to the safety cap, even
    # after its requested horizon; otherwise missing-stop answers are invented.
    model.completion_head.network[-1].bias.data.fill_(-10)
    full=partitioned_forward(model,prefix,suffix,mask,indices,tokens,8)
    short=partitioned_forward(model,prefix,suffix,mask,indices,tokens,8,horizons)
    assert torch.equal(short.scores,full.scores)
    assert torch.equal(short.stops,full.stops)
