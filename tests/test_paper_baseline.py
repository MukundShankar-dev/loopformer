import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.dataset.benchmark import count_variant
from scripts.eval import paper_baseline, paper_baseline_analysis
from scripts.eval.paper_baseline import read_chunk, text_hash, write_chunk
from scripts.eval.paper_baseline_analysis import audit_row
from scripts.eval.paper_suite import load_graphs
from scripts.eval.pointer_task import render_task, score_response, sha256_file


class Tokenizer:
    chat_template = 'test chat template'
    backend_tokenizer = SimpleNamespace(to_str=lambda: 'stable tokenizer')

    def apply_chat_template(self, messages, **kwargs):
        return f"user: {messages[0]['content']}\nassistant:"

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i != 0)

    def save_pretrained(self, path):
        path.mkdir(exist_ok=True); (path/'tokenizer.json').write_text('{}')


def example():
    tasks, _ = load_graphs(Path('data/pointer/benchmark-seeds307-311-313/graphs.jsonl'))
    return tasks[0]


def row_for(task, n, tokenizer, template, response=None, budget=False):
    variant = count_variant(task, n)
    prompt = render_task(template, variant)
    model_input = tokenizer.apply_chat_template([dict(content=prompt)])
    response = variant.final_state if response is None else response
    ids = [ord(c) for c in response] + ([] if budget else [0])
    return dict(example_id=variant.example_id, task_depth=n, target=variant.final_state,
        graph_index=0, response=response, **score_response(response, variant.final_state),
        generated_token_ids=json.dumps(ids), generated_tokens=len(ids), prompt_tokens=100,
        stop_reason='max_new_tokens' if budget else 'eos', prompt=prompt, model_input=model_input,
        input_sha256=text_hash(model_input))


def test_raw_audit_rejects_wrong_targets_inputs_and_scoring_and_counts_invalid_answers():
    task=example(); tokenizer=Tokenizer(); template='Rules: {rules}\nStart: {start}\nSteps: {steps}\nAnswer:'
    row=row_for(task,1,tokenizer,template); target=count_variant(task,1).final_state
    config=dict(max_new_tokens=8)
    assert audit_row(row,task,1,target,tokenizer,template,config,{0})[:3] == (True,True,False)
    for key,value,match in [('target','?', 'target'), ('input_sha256','changed','input'), ('correct',False,'score')]:
        with pytest.raises(ValueError,match=match):
            audit_row({**row,key:value},task,1,target,tokenizer,template,config,{0})
    invalid=row_for(task,1,tokenizer,template,response='Answer Y')
    # The eight-token continuation has no EOS and is invalid even though it contains a letter.
    invalid['generated_token_ids']=json.dumps([ord(c) for c in 'Answer Y']); invalid['generated_tokens']=8
    invalid['stop_reason']='max_new_tokens'
    assert audit_row(invalid,task,1,target,tokenizer,template,config,{0})[:3] == (False,False,True)
    with pytest.raises(ValueError,match='budget'):
        audit_row({**row,'stop_reason':'max_new_tokens'},task,1,target,tokenizer,template,config,{0})


def test_chunk_requires_exact_graph_count_coverage_and_preserves_raw_response(tmp_path):
    task=example(); tokenizer=Tokenizer(); template='{rules}\n{start}\n{steps}'
    rows=[row_for(task,n,tokenizer,template) for n in [1,2]]
    path=tmp_path/'graph-0000.jsonl.gz'; write_chunk(path,0,rows,2.5,3.)
    header,saved=read_chunk(path,0,[1,2])
    assert header['generation_seconds']==2.5 and header['evaluation_seconds']==3.
    assert saved[0]['response']==rows[0]['response'] and saved[0]['input_sha256']==text_hash(rows[0]['model_input'])
    assert 'model_input' not in saved[0] and 'prompt' not in saved[0]
    for index,counts in [(1,[1,2]),(0,[1]),(0,[2,1])]:
        with pytest.raises(ValueError,match='coverage'): read_chunk(path,index,counts)


def test_complete_resume_never_loads_model_and_audit_checks_every_count(tmp_path,monkeypatch):
    root=tmp_path/'suite';root.mkdir();task=example();graph_path=Path('data/pointer/benchmark-seeds307-311-313/graphs.jsonl')
    protocol=dict(graphs=str(graph_path),graph_count=1,requests=list(range(1,257)),
                  models=[dict(name='ce_full')],uncertainty=dict(bootstrap_seed=239,bootstrap_repeats=20))
    (root/'protocol.json').write_text(json.dumps(protocol))
    prompt_path=tmp_path/'prompt.txt';prompt_path.write_text('Rules: {rules}\nStart: {start}\nSteps: {steps}\nAnswer:')
    cfg=json.loads(Path('configs/pointer_analysis_baseline.json').read_text())
    cfg.update(prompt=str(prompt_path),prompt_sha256=sha256_file(prompt_path),graphs_sha256=sha256_file(graph_path))
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(cfg))
    cached=tmp_path/'cache';cached.mkdir();(cached/'config.json').write_text('{}')
    tokenizer=Tokenizer();calls=[]
    model=SimpleNamespace(set_attn_implementation=lambda value:None, requires_grad_=lambda value:None,
                          generation_config=SimpleNamespace(eos_token_id=0))
    monkeypatch.setattr(paper_baseline,'snapshot_download',lambda *a,**k:str(cached))
    monkeypatch.setattr(paper_baseline,'load_graphs',lambda path:([task],[dict(dataset_seed=307,graph_mode='random_function')]))
    def load(*args,**kwargs):
        calls.append('load');return model,tokenizer,dict(resolved_model_revision=cfg['revision'])
    monkeypatch.setattr(paper_baseline,'load_model',load)
    monkeypatch.setattr(paper_baseline,'native_gate',lambda *a:dict(passed=True,questions=54))
    def batches(model,tokenizer,variants,template,**kwargs):
        rows=[row_for(task,v.task_depth,tokenizer,template) for v in variants]
        yield rows,1.
    monkeypatch.setattr(paper_baseline,'evaluate_batches',batches)
    paper_baseline.extract(root,config_path);paper_baseline.extract(root,config_path)
    assert calls==['load']
    directory=root/'ordinary_qwen';summary=json.loads((directory/'summary.json').read_text())
    assert summary['queries']==256 and summary['accuracy']==1.
    np.savez(root/'outcomes.npz',ce_full__nominal_final_correct=np.zeros((1,256),dtype=bool))
    meta=dict(dataset_seed=307,graph_mode='random_function',cycle_period=2,transient_length=11)
    monkeypatch.setattr(paper_baseline_analysis,'load_graphs',lambda path:([task],[meta]))
    monkeypatch.setattr(paper_baseline_analysis.AutoTokenizer,'from_pretrained',lambda *a,**k:tokenizer)
    paper_baseline_analysis.analyze(root)
    audit=json.loads((directory/'independent_audit.json').read_text())
    assert audit['passed'] and audit['queries']==256 and audit['accuracy']==1.
    # A cached summary cannot mask changed raw data or a changed inference recipe.
    with (directory/'graph-0000.jsonl.gz').open('ab') as handle:handle.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):
        paper_baseline_analysis.analyze(root)
    cfg['batch_size']=8;config_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='identity'):
        paper_baseline.extract(root,config_path)
