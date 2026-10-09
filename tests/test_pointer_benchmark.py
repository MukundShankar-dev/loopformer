"""Frozen confirmation contracts: disjoint sampling, exact timing and native fidelity."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_executor_upgrade import setup
from test_controller_training import source_checkpoint
from scripts.dataset.benchmark import count_variant, independent_graphs
from scripts.dataset.dataset import DatasetConfig, generate_dataset, write_dataset
from scripts.dataset.pointer import execute, generate_unconditioned_example
from scripts.eval.benchmark_metrics import clustered_summary, criteria_result, quality_arrays, wilson_interval
from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes
from scripts.recurrent_qwen.interfaces import AffineSuffixController, SharedNumberController
from scripts.training.controller import export_checkpoint


def criteria():
    return {'overall_joint_success': .95, 'overall_complete_trajectory': .95, 'overall_exact_stop': .99,
            'minimum_depth_joint_success': .90, 'minimum_depth_complete_trajectory': .90, 'minimum_depth_exact_stop': .95}


def test_sampling_replays_and_rejects_overlap_without_depth_conditioning():
    config = {'seeds': [211,223], 'graph_modes': ['random_function','permutation','full_cycle'],
              'graphs_per_seed_mode': 2, 'max_depth': 256}
    first, meta = independent_graphs(config, set())
    second, _ = independent_graphs(config, set())
    assert first == second and len({t.mapping_sha256 for t in first}) == 12
    shorter, _ = independent_graphs({**config, 'max_depth': 12}, set())
    assert [(t.mapping,t.initial_state) for t in first] == [(t.mapping,t.initial_state) for t in shorter]
    excluded = {first[0].mapping_sha256}
    fresh, _ = independent_graphs(config, excluded)
    assert not excluded & {t.mapping_sha256 for t in fresh}
    assert count_variant(first[0], 99).mapping == first[0].mapping
    assert all(meta[i]['cycle_period']==26 for i,t in enumerate(first) if meta[i]['graph_mode']=='full_cycle')


def test_exact_stop_rejects_cyclic_coincidence_and_missing_cap():
    task = generate_unconditioned_example(30, 3, 'benchmark', 0)
    # Make a fixed-point reference; every letter is right, but timing still matters.
    task = replace(task, initial_state='A', mapping=[[chr(65+i),'A'] for i in range(26)])
    predicted = torch.zeros(1,4,dtype=torch.long)
    arrays, rows = quality_arrays([task],predicted,[1,1,None],3)
    assert arrays['nominal_final_correct'].all() and arrays['complete_trajectory'].all()
    assert arrays['exact_stop'].tolist()==[[True,False,False]]
    assert arrays['joint_success'].tolist()==[[True,False,False]]
    assert arrays['correct_letter_wrong_time'].tolist()==[[False,True,True]]
    assert arrays['missing_stop'][0,2] and not criteria_result(arrays,criteria())['passed']


def test_prefix_error_is_not_hidden_by_recovery_and_conditional_denominators():
    task = generate_unconditioned_example(31, 3, 'benchmark', 0)
    targets = torch.tensor([[ord(s)-65 for s in execute(dict(task.mapping),task.initial_state,4)]])
    targets[0,1]=(targets[0,1]+1)%26
    arrays, rows = quality_arrays([task],targets,[1,2,3],3)
    assert arrays['complete_trajectory'].tolist()==[[True,False,False]]
    assert arrays['nominal_final_correct'][0,2] and not arrays['strict_success'][0,2]
    assert rows[2]['correct_prefix_graphs']==0 and rows[2]['conditional_transition_accuracy'] is None


def test_cluster_bootstrap_keeps_paired_horizons_together():
    arrays={k:np.array([[True,True],[False,False]]) for k in ('nominal_final_correct','complete_trajectory',
        'exact_stop','early_stop','late_stop','missing_stop','stopped_answer_correct','joint_success',
        'strict_success','correct_letter_wrong_time')}
    meta=[{'dataset_seed':211,'graph_mode':'random_function'}]*2
    result=clustered_summary(arrays,meta,np.array([True,True]),repeats=100,seed=239)
    assert result['graphs']==2 and result['queries']==4
    assert result['metrics']['joint_success']['rate']==.5
    # Resampling two graphs yields 0, .5, 1 rather than falsely four independent observations.
    assert result['metrics']['joint_success']['graph_bootstrap_95']==[0.,1.]


@pytest.mark.parametrize('kind',['affine_suffix','shared_number'])
def test_frozen_benchmark_cli_native_agrees_on_failures_and_files_unchanged(setup,kind):
    model, tokenizer, tokens, path = source_checkpoint(setup)
    head=(AffineSuffixController if kind=='affine_suffix' else SharedNumberController)(model.config.hidden_size)
    with torch.no_grad():
        for parameter in head.context.parameters():
            parameter.zero_()
        head.readout.weight.zero_();head.readout.bias.fill_(1)  # Always stops at one; deliberate failures.
    export_checkpoint(path/'source',path/'affine',head,{'step':0})
    dataset_config=DatasetConfig(seed=61,train_count=2,validation_count=2,test_count=2,
        depth_test_count=2,max_train_depth=2,max_eval_depth=3)
    old=generate_dataset(dataset_config,tokenizer,tokens)
    write_dataset(path/'old',old,dataset_config,tokens,{})
    config={'model':str(path/'affine'),'existing_data':str(path/'old'),'seeds':[211],
        'graphs_per_seed_mode':1,'graph_modes':['random_function','full_cycle'],'max_depth':4,
        'safety_cap':6,'threshold':.5,'batch_size':2,'native_counts':[1,4],
        'native_graphs_per_seed_mode':1,'bootstrap_repeats':10,'bootstrap_seed':239,'criteria':criteria()}
    config_path=path/'config.json';config_path.write_text(json.dumps(config))
    before=checkpoint_hashes(path/'affine')
    freeze=path/'freeze.json';freeze.write_text(json.dumps({'checkpoint':str(path/'affine'),
        'inference_sha256':before,'threshold':.5,'safety_cap':6}))
    args=[sys.executable,'-m','scripts.eval.frozen_pointer_benchmark','--config',str(config_path),
        '--freeze',str(freeze),'--output',str(path/'result'),'--dataset-output',str(path/'benchmark'),
        '--device','cpu','--wandb-mode','disabled']
    result=subprocess.run(args,capture_output=True,text=True,env={**os.environ,'WANDB_MODE':'disabled'})
    assert result.returncode==0,result.stdout+result.stderr
    summary=json.loads((path/'result/summary.json').read_text())
    assert summary['native_fidelity']=={'passed':True,'questions':4,'mismatches':[]}
    assert not summary['acceptance']['passed'] and summary['inference_files_unchanged']
    assert checkpoint_hashes(path/'affine')==before
    from scripts.eval.audit_pointer_benchmark import audit_benchmark
    assert audit_benchmark(path/'result',path/'benchmark/graphs.jsonl')['decisions_checked']==8
    import csv
    import gzip
    decisions=path/'result/decisions.csv.gz'
    original_bytes=decisions.read_bytes()
    with gzip.open(decisions,'rt') as handle:
        rows=list(csv.DictReader(handle))
    rows[1]['exact_stop']='True'  # Correct letters cannot turn a wrong loop into exact timing.
    with gzip.open(decisions,'wt',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match='Decision flag exact_stop differs'):
        audit_benchmark(path/'result',path/'benchmark/graphs.jsonl')
    decisions.write_bytes(original_bytes)
    if kind=='shared_number':
        # Component reuse must preserve negative outcomes and reject a changed R.
        with torch.no_grad():head.cell.bias.add_(-.25)
        export_checkpoint(path/'affine',path/'refined',head,{'step':0})
        settings={'model':str(path/'refined'),'source_results':str(path/'result'),
            'graphs':str(path/'benchmark/graphs.jsonl'),'native_counts':[1,4],
            'numeric_rollout_max':4,'numeric_reader_max':4,'output':str(path/'composition'),'device':'cpu'}
        component_config=path/'component.json';component_config.write_text(json.dumps(settings))
        component_args=[sys.executable,'-m','scripts.eval.countdown_precision_test','--config',str(component_config)]
        failed=subprocess.run(component_args,capture_output=True,text=True)
        assert failed.returncode!=0,failed.stdout+failed.stderr
        composition=json.loads((path/'composition/summary.json').read_text())
        assert composition['status']=='complete' and not composition['passed']
        assert composition['matching_count_timings']==4 and composition['native_fidelity']['passed']
        payload=torch.load(path/'refined/adapter_model.pt',weights_only=True)
        first=next(k for k in payload if k.startswith('recurrent.'))
        payload[first]=payload[first]+.001
        torch.save(payload,path/'refined/adapter_model.pt')
        settings['output']=str(path/'invalid-composition');component_config.write_text(json.dumps(settings))
        assert subprocess.run(component_args,capture_output=True,text=True).returncode!=0
        assert not (path/'invalid-composition').exists()
    repeat=subprocess.run(args,capture_output=True,text=True)
    assert repeat.returncode!=0 and 'refuse overwrite' in repeat.stderr


def test_per_count_boundary_interval_does_not_claim_certainty():
    low, high = wilson_interval(128,128)
    assert .97 < low < 1 and high == 1
    low, high = wilson_interval(0,128)
    assert low == 0 and 0 < high < .03


def test_reserved_reuse_requires_full_frozen_model_and_policy():
    from scripts.eval.freeze_pointer_benchmark import verify_reserved_result
    frozen={'inference_sha256':{'adapter_model.pt':'abc'},'test_data_sha256':'def','threshold':.5,'safety_cap':272}
    summary={'status':'complete','local_checkpoint_sha256':frozen['inference_sha256'],'data_sha256':'def',
        'stop_policy':'completion','stop_threshold':.5,'loops':272,'selected_examples':1536,'limit':None,'test_mode':False}
    verify_reserved_result(summary,frozen,1536)
    for bad in ({'selected_examples':3},{'stop_threshold':.6},{'data_sha256':'other'},
                {'local_checkpoint_sha256':{'adapter_model.pt':'other'}},{'status':'running'}):
        with pytest.raises(ValueError):
            verify_reserved_result({**summary,**bad},frozen,1536)
