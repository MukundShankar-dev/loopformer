from copy import deepcopy
import pytest
import torch
from scripts.recurrent_qwen.model import RecurrentQwen
from scripts.recurrent_qwen.lora_utils import attach_recurrent_lora
from scripts.training.config import TrainingConfig
from scripts.training.precision import autocast_context


def test_sdpa_recurrence_matches_eager_outputs_and_gradients(base, inputs):
    eager=RecurrentQwen(base,1,3,recurrence_mode='fixed_prompt')
    attach_recurrent_lora(eager,rank=2,alpha=4)
    for name,p in eager.named_parameters():
        if 'lora_B' in name:torch.nn.init.normal_(p,std=.01)
    sdpa=deepcopy(eager)
    sdpa.config._attn_implementation='sdpa'
    mask=torch.ones_like(inputs);mask[0,-1]=0
    a=eager(inputs,mask,num_loops=4).loop_logits
    b=sdpa(inputs,mask,num_loops=4).loop_logits
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-5)
    sum(x.square().mean() for x in a).backward()
    sum(x.square().mean() for x in b).backward()
    for (n,p),(_,q) in zip(eager.named_parameters(),sdpa.named_parameters()):
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4)
        else:assert p.grad is None and q.grad is None


def test_precision_fails_explicitly_on_unsupported_device():
    with pytest.raises(ValueError):TrainingConfig(precision='bf16',device='cpu').validate()
    with pytest.raises(ValueError):autocast_context('cpu','bf16')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_bf16_shared_recurrence_backward_cuda(base, inputs):
    if not torch.cuda.is_bf16_supported():pytest.skip('BF16 unavailable')
    base.config._attn_implementation='sdpa'
    model=RecurrentQwen(base.cuda(),1,3,recurrence_mode='fixed_prompt')
    attach_recurrent_lora(model,rank=2,alpha=4)
    model.enable_completion(8)
    with autocast_context('cuda','bf16'):
        out=model(inputs.cuda(),num_loops=12)
    loss=sum(x.float().square().mean() for x in out.loop_logits)+out.stop_logits.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_cohort_aggregation_separates_held_out_counts(tmp_path):
    import json
    from scripts.eval.depth_cohorts import summarize_cohorts
    for split,depths in [('validation',range(1,13)),('depth_test',range(13,21))]:
        full={};stopped={}
        for d in depths:
            value=float(d in [7,9,11])
            full[str(d)]={'examples':2,'trajectory_accuracy':value}
            stopped[str(d)]={'examples':2,**{k:value for k in ['accuracy','exact_stop_rate','joint_success_rate','early_stop_rate','cap_fallback_rate']}}
        for suffix,values in [('',full),('-stopped',stopped)]:
            p=tmp_path/(split+suffix);p.mkdir();(p/'summary.json').write_text(json.dumps({'by_depth':values}))
    result=summarize_cohorts(tmp_path)
    assert result['held_out_counts']['joint_success_rate']==1
    assert result['trained_counts']['joint_success_rate']==0
    assert result['unseen_depths']['examples']==16
