"""Trace targets, scientific scoring, gradients and ordinary checkpoint contract."""
from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from test_naive_pointer_eval import tiny_checkpoint
from scripts.dataset.pointer import generate_example, generate_unconditioned_example
from scripts.eval.pointer_task import evaluate_batches
from scripts.eval.trace_baselines import read_chunk, save_chunk
from scripts.trace_task import encode_trace, score_trace, set_trainable, trace_collate, trace_loss, trace_states
from scripts.training.trace_runner import checkpoint, rate_at_step, validate
from scripts.eval.trace_analysis import audit_row
from scripts.eval.paper_baseline import text_hash
from scripts.eval.cached_generation import cached_batches

TEMPLATE=Path('prompts/pointer_trace.txt').read_text()


def test_trace_metrics_do_not_accept_cycle_alias_or_repaired_prefix():
    task=generate_unconditioned_example(17,3,'test',0,'random_function')
    gold=task.intermediate_states
    exact=score_trace(' '.join(gold),task,'eos')
    assert exact['strict_success'] and exact['exact_stop'] and exact['first_failure'] is None
    # A correct final letter from a short/long cyclic trace does not fix termination.
    short=score_trace(gold[-1],task,'eos')
    assert short['correct'] and not short['joint_success'] and short['early_stop']
    long=score_trace(' '.join(gold+[gold[-1]]),task,'eos')
    assert long['correct'] and long['complete_trajectory'] and long['late_stop'] and not long['strict_success']
    wrong='Z' if gold[0]!='Z' else 'Y'
    repaired=score_trace(' '.join([wrong,*gold[1:]]),task,'eos')
    assert repaired['joint_success'] and not repaired['strict_success'] and repaired['first_symbol_error']==1
    no_eos=score_trace(' '.join(gold),task,'max_new_tokens')
    assert no_eos['complete_trajectory'] and no_eos['missing_stop'] and not no_eos['exact_stop']
    assert score_trace('Answer: '+' '.join(gold),task,'eos')['malformed']
    assert trace_states(' A\nB ') == ['A','B']
    assert trace_states('AB') is None and trace_states('') is None


def test_response_ce_matches_full_causal_loss_and_gradients(tiny_checkpoint):
    _,model,tokenizer=tiny_checkpoint
    items=[encode_trace(generate_example(17+i,1+i,'train',i),tokenizer,TEMPLATE,1024) for i in range(2)]
    batch=trace_collate(items,tokenizer.pad_token_id,'cpu')
    model.eval();loss,_=trace_loss(model,batch);loss.backward()
    gradients={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    full=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
    selected=full.gather(1,batch['positions'][:,:,None].expand(-1,-1,full.shape[-1]))
    ce=torch.nn.functional.cross_entropy(selected.flatten(0,1),batch['targets'].flatten(),reduction='none').reshape_as(batch['targets'])
    reference=((ce*batch['target_mask']).sum(1)/batch['target_mask'].sum(1)).mean()
    torch.testing.assert_close(loss,reference)
    reference.backward()
    for k,p in model.named_parameters():
        if k in gradients:torch.testing.assert_close(p.grad,gradients[k],atol=1e-7,rtol=1e-5)
    assert all(x.response_ids[-1]==tokenizer.eos_token_id for x in items)
    with pytest.raises(ValueError,match='no truncation'):encode_trace(items[0].task,tokenizer,TEMPLATE,2)


def test_only_matching_middle_layers_receive_gradient(tiny_checkpoint):
    _,_,tokenizer=tiny_checkpoint
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,tie_word_embeddings=True)).eval()
    names=set_trainable(model,'middle',1,2)
    assert names and all(k.startswith('model.layers.1.') for k in names)
    item=encode_trace(generate_example(17,2,'train',0),tokenizer,TEMPLATE,1024)
    loss,_=trace_loss(model,trace_collate([item],0,'cpu'));loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.requires_grad)>0


def test_full_sft_trains_embeddings_head_and_every_layer(tiny_checkpoint):
    _,model,_=tiny_checkpoint
    names=set_trainable(model,'full',0,1)
    assert 'model.embed_tokens.weight' in names and 'model.norm.weight' in names
    assert all(p.requires_grad for p in model.parameters())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert json.loads(Path('configs/trace_sft.json').read_text())['train_scope']=='full'


def test_independent_trace_audit_rejects_forged_success(tiny_checkpoint):
    _,model,tokenizer=tiny_checkpoint
    task=generate_example(17,2,'test',0)
    from scripts.eval.pointer_task import render_task
    response=' '.join(task.intermediate_states)
    ids=tokenizer.encode(response,add_special_tokens=False)+[tokenizer.eos_token_id]
    prompt=tokenizer.apply_chat_template([dict(role='user',content=render_task(TEMPLATE,task))],tokenize=False,add_generation_prompt=True)
    row=dict(response=response,generated_token_ids=json.dumps(ids),generated_tokens=len(ids),
        target=task.final_state,task_depth=2,stop_reason='eos',input_sha256=text_hash(prompt),
        **score_trace(response,task,'eos'))
    assert audit_row(row,task,2,tokenizer,TEMPLATE,dict(max_new_tokens=272),{tokenizer.eos_token_id})['strict_success']
    row['exact_stop']=False
    with pytest.raises(ValueError,match='exact_stop'):audit_row(row,task,2,tokenizer,TEMPLATE,dict(max_new_tokens=272),{1})


def test_ordinary_training_resume_preserves_updates(tiny_checkpoint,tmp_path,monkeypatch):
    from scripts.training import trace_runner
    from scripts.eval.loading import load_model
    directory,_,tokenizer=tiny_checkpoint
    c=json.loads(Path('configs/trace_sft.json').read_text())
    c.update(device='cpu',precision='float32',train_scope='full',train_layer_start=0,train_layer_end=1,
        gradient_checkpointing=False,batch_size=1,gradient_accumulation=2,eval_every=1,save_every=1,train_depths=[1,2])
    items=[encode_trace(generate_example(17+i,1+i%2,'train',i),tokenizer,TEMPLATE,1024) for i in range(6)]
    def fresh(source=directory):
        return load_model(str(source),base_model=None,tokenizer_source=None,revision=None,device='cpu',dtype='float32',download=False)[0]
    full=fresh();trace_runner.run(full,tokenizer,items,items[:2],c,{'contract':1},tmp_path/'full')
    real_save=trace_runner.checkpoint
    def interrupted(model,tok,path,state):
        real_save(model,tok,path,state)
        if path.name=='last' and state['step']==1:raise RuntimeError('Simulated interruption')
    monkeypatch.setattr(trace_runner,'checkpoint',interrupted)
    with pytest.raises(RuntimeError,match='Simulated'):
        trace_runner.run(fresh(),tokenizer,items,items[:2],c,{'contract':1},tmp_path/'resumed')
    monkeypatch.setattr(trace_runner,'checkpoint',real_save)
    continued=fresh(tmp_path/'resumed'/'last')
    result=trace_runner.run(continued,tokenizer,items,items[:2],c,{'contract':1},tmp_path/'resumed',resume=tmp_path/'resumed'/'last')
    assert result['steps']==3
    for key,value in full.state_dict().items():torch.testing.assert_close(value,continued.state_dict()[key],atol=0,rtol=0)


def test_cached_generation_matches_ordinary_uncached(tiny_checkpoint):
    _,model,tokenizer=tiny_checkpoint;tokenizer.padding_side='left';model.set_attn_implementation('sdpa')
    examples=[generate_example(17,1,'test',0),generate_example(18,2,'test',1)]
    options=dict(batch_size=2,max_new_tokens=4)
    original=list(evaluate_batches(model,tokenizer,examples,TEMPLATE,prompt_format='chat',**options))[0][0]
    cached=list(cached_batches(model,tokenizer,examples,TEMPLATE,**options))[0][0]
    assert [r['generated_token_ids'] for r in original]==[r['generated_token_ids'] for r in cached]


def test_cached_trace_scoring_keeps_task_identity_across_batches(tiny_checkpoint,monkeypatch):
    _,model,tokenizer=tiny_checkpoint;tokenizer.padding_side='left'
    tasks=[generate_example(17+i,1+i%3,'test',i) for i in range(5)]
    offset=0
    def generated(**kwargs):
        nonlocal offset
        assert kwargs['generation_config'].use_cache is True
        n=len(kwargs['input_ids']);part=tasks[offset:offset+n];offset+=n
        sequences=[tokenizer.encode(' '.join(t.intermediate_states),add_special_tokens=False)+[1] for t in part]
        width=max(map(len,sequences));ids=torch.zeros((n,width),dtype=torch.long)
        for i,s in enumerate(sequences):ids[i,:len(s)]=torch.tensor(s)
        return torch.cat((kwargs['input_ids'],ids),1)
    monkeypatch.setattr(model,'generate',generated)
    rows=[r for batch,_ in cached_batches(model,tokenizer,tasks,TEMPLATE,batch_size=2,max_new_tokens=10) for r in batch]
    assert len(rows)==5 and all(r['strict_success'] for r in rows)
    assert [r['example_id'] for r in rows]==[t.example_id for t in tasks]


def test_saved_ordinary_checkpoint_and_atomic_chunks(tiny_checkpoint,tmp_path):
    _,model,tokenizer=tiny_checkpoint
    state=dict(step=1,best_loss=.5,identity={'test':True},optimizer={},rng={})
    checkpoint(model,tokenizer,tmp_path/'best',state)
    assert (tmp_path/'best'/'config.json').exists() and not (tmp_path/'best'/'recurrent_config.json').exists()
    row=dict(prompt='p',model_input='input',example_id='x',task_depth=1)
    path=tmp_path/'chunk.jsonl.gz';save_chunk(path,[row],1.2,[0],1,'hash')
    header,rows=read_chunk(path)
    assert header['indices']==[0] and rows[0]['graph_index']==0 and 'prompt' not in rows[0]
    assert rows[0]['input_sha256'] and header['freeze_sha256']=='hash'


def test_schedule_matches_executor_and_validation_is_labelled(tiny_checkpoint):
    _,model,tokenizer=tiny_checkpoint
    c=json.loads(Path('configs/trace_sft.json').read_text())
    assert rate_at_step(c,100)==c['learning_rate']
    assert rate_at_step(c,2250)==pytest.approx(c['learning_rate']*.1)
    c.update(device='cpu',precision='float32',batch_size=2,train_depths=[1])
    items=[encode_trace(generate_example(17+i,1+i%2,'validation',i),tokenizer,TEMPLATE,1024) for i in range(4)]
    val=validate(model,items,c,0)
    assert val['selection_loss']==val['by_depth']['1']['loss']
    assert 'teacher_forced_response_accuracy' in val['by_depth']['2']
