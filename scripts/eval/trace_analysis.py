"""Independently audit raw trace generations and render a separate comparison bundle."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from scripts.eval.benchmark_metrics import METRICS, clustered_summary, grouped_rows
from scripts.eval.loop_metrics import write_csv
from scripts.eval.paper_baseline import atomic_json, text_hash
from scripts.eval.paper_suite import load_graphs
from scripts.eval.pointer_task import load_examples, sha256_file
from scripts.eval.trace_baselines import read_chunk


def audit_row(row: dict,task,count: int,tokenizer,template: str,config: dict,eos_ids: set[int]) -> dict:
    """Scalar dictionary execution and direct decoding; no generation scorer reused."""
    state=task.initial_state;mapping=dict(task.mapping);gold=[]
    for _ in range(count):state=mapping[state];gold.append(state)
    prompt=template.format(rules=' '.join(f'( {a}, {b})' for a,b in task.mapping),start=task.initial_state,steps=count).strip()
    model_input=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
    if text_hash(model_input)!=row['input_sha256'] or row['target']!=gold[-1] or row['task_depth']!=count:
        raise ValueError('Trace query input/label identity differs')
    ids=json.loads(row['generated_token_ids'])
    if not ids or any(type(i) is not int or i<0 for i in ids):raise ValueError('Invalid raw token IDs')
    eos=ids[-1] in eos_ids
    if (any(i in eos_ids for i in ids[:-1]) or len(ids)!=row['generated_tokens'] or len(ids)>config['max_new_tokens']
        or row['stop_reason']!=('eos' if eos else 'max_new_tokens') or (not eos and len(ids)!=config['max_new_tokens'])):
        raise ValueError('Trace budget/EOS semantics differ')
    response=tokenizer.decode(ids[:-1] if eos else ids,skip_special_tokens=False)
    valid=re.fullmatch(r'[A-Z](?:\s+[A-Z])*',response.strip(),flags=re.ASCII) is not None
    states=response.split() if valid else [];length=len(states)
    same=[a==b for a,b in zip(states,gold)]
    final=bool(states) and states[-1]==gold[-1]
    prefix=valid and length>=count and all(same[:count])
    exact=valid and eos and length==count
    mismatch=next((i+1 for i,hit in enumerate(same) if not hit),None)
    failure=1 if not valid else mismatch if mismatch else length+1 if length<count else None
    metrics=dict(nominal_final_correct=valid and length>=count and states[count-1]==gold[-1],
        complete_trajectory=prefix,exact_stop=exact,early_stop=valid and eos and length<count,
        late_stop=valid and eos and length>count,missing_stop=not eos,
        stopped_answer_correct=final,joint_success=final and exact,strict_success=prefix and exact,
        correct_letter_wrong_time=final and not exact,first_symbol_error=mismatch,first_failure=failure,
        emitted_states=length,malformed=not valid,prefix_correct=failure-1 if failure else count)
    if response!=row['response'] or valid!=row['valid_answer'] or states!=row['states'] or final!=row['correct']:
        raise ValueError('Trace raw response score differs')
    for key,value in metrics.items():
        if row[key]!=value:raise ValueError(f'Independent trace audit differs at {key}')
    metrics['termination_category']=('malformed' if not valid else 'budget' if not eos else
        'early' if length<count else 'late' if length>count else 'exact')
    return metrics


def audit_arm(root: Path,arm: str,tasks: list,metadata: list,config: dict) -> dict:
    directory=root/arm;summary=json.loads((directory/'summary.json').read_text())
    frozen=json.loads((directory/'freeze.json').read_text());identity=frozen['identity']
    if summary['status']!='complete' or summary['smoke'] or not summary['weights_unchanged']:
        raise ValueError('Only complete full runs enter the comparison')
    if identity['config']!=config or summary['freeze_sha256']!=sha256_file(directory/'freeze.json'):
        raise ValueError('Frozen trace identity differs')
    template=(directory/'prompt.txt').read_text();tokenizer=AutoTokenizer.from_pretrained(directory/'tokenizer',local_files_only=True)
    loaded=json.loads((directory/'loaded.json').read_text())
    if text_hash(template)!=identity['prompt_sha256'] or text_hash(tokenizer.backend_tokenizer.to_str())!=loaded['tokenizer_sha256']:
        raise ValueError('Trace saved prompt/tokenizer differs')
    eos=loaded['eos_token_id'];eos_ids=set(eos if isinstance(eos,list) else [eos])
    shape=len(tasks),len(config['requests']);arrays={k:np.zeros(shape,dtype=bool) for k in (*METRICS,'malformed')}
    arrays.update(emitted_states=np.zeros(shape,dtype=np.int32),first_failure=np.zeros(shape,dtype=np.int32),
        first_symbol_error=np.zeros(shape,dtype=np.int32))
    seen=set();hashes={};categories=Counter();tokens=0
    for path in sorted((directory/'chunks').glob('*.jsonl.gz')):
        hashes[path.name]=sha256_file(path)
        if hashes[path.name]!=summary['chunks_sha256'].get(path.name):raise ValueError('Raw trace chunk hash differs')
        header,rows=read_chunk(path)
        if header['freeze_sha256']!=summary['freeze_sha256'] or header['indices']!=[r['graph_index'] for r in rows]:
            raise ValueError('Trace chunk header identity/coverage differs')
        for row in rows:
            i=row['graph_index'];n=row['task_depth'];key=(i,n)
            if key in seen or not 0<=i<len(tasks) or n not in config['requests']:raise ValueError('Repeated/out-of-panel query')
            seen.add(key);j=config['requests'].index(n)
            if row['example_id']!=f'{tasks[i].example_id}-steps-{n}':raise ValueError('Trace graph ID differs')
            metrics=audit_row(row,tasks[i],n,tokenizer,template,config,eos_ids)
            for k in arrays:arrays[k][i,j]=metrics[k] or 0
            categories[metrics['termination_category']]+=1;tokens+=row['generated_tokens']
    if len(seen)!=shape[0]*shape[1] or set(hashes)!=set(summary['chunks_sha256']):raise ValueError('Trace query/chunk coverage incomplete')
    if tokens!=summary['generated_tokens'] or abs(float(arrays['strict_success'].mean())-summary['strict_success'])>1e-12:
        raise ValueError('Trace aggregate disagrees with independently audited rows')
    testpath=directory/'reserved_test.jsonl.gz'
    if sha256_file(testpath)!=summary['reserved_test_sha256'] or sha256_file(Path(config['test']))!=identity['test_sha256']:
        raise ValueError('Reserved test identity differs')
    header,rows=read_chunk(testpath);test=load_examples(Path(config['test']));testmetrics=[]
    if len(test)!=len(rows) or header['freeze_sha256']!=summary['freeze_sha256']:raise ValueError('Test query coverage differs')
    for row,task in zip(rows,test,strict=True):
        if row['example_id']!=task.example_id:raise ValueError('Test query ID differs')
        testmetrics.append(audit_row(row,task,task.task_depth,tokenizer,template,config,eos_ids))
    testbydepth=[]
    for n in sorted({t.task_depth for t in test}):
        cohort=[m for t,m in zip(test,testmetrics,strict=True) if t.task_depth==n]
        testbydepth.append(dict(depth=n,questions=len(cohort),**{k:float(np.mean([m[k] for m in cohort])) for k in METRICS}))
    write_csv(directory/'reserved_test_by_depth.csv',testbydepth)
    np.savez_compressed(directory/'outcomes.npz',**arrays,counts=np.asarray(config['requests']))
    write_csv(directory/'strata.csv',grouped_rows({k:arrays[k] for k in METRICS},metadata))
    counts=np.asarray(config['requests']);intervals={}
    for name,mask in dict(all=counts>0,trained=np.isin(counts,[1,2,3,4,5,6,8,10,12]),
        heldout=np.isin(counts,[7,9,11]),extrapolation=counts>12).items():
        intervals[name]=clustered_summary(arrays,metadata,mask,repeats=config['bootstrap_repeats'],seed=config['bootstrap_seed'])
    atomic_json(directory/'cluster_intervals.json',intervals)
    report=dict(passed=True,queries=len(seen),reserved_test_queries=len(rows),chunks_sha256=hashes,
        termination_categories=dict(categories),metrics={k:float(arrays[k].mean()) for k in METRICS},
        outcomes_sha256=sha256_file(directory/'outcomes.npz'),freeze_sha256=sha256_file(directory/'freeze.json'),
        native_sensitivity=json.loads((directory/'native_fidelity.json').read_text()),
        analysis_code_sha256=sha256_file(Path(__file__)))
    atomic_json(directory/'independent_audit.json',report)
    return arrays


def render(root: Path,config: dict,arms: dict,metadata: list) -> None:
    """Separate trace figures; no overwrite of the completed 25-family bundle."""
    plots=root/'plots';plots.mkdir(exist_ok=True);depth=np.asarray(config['requests'])
    reference=Path(config['reference_results']);sources={'trace_protocol':sha256_file(root/'protocol.json')}
    previous=json.loads((reference/'protocol.json').read_text())
    if previous['graphs']!=config['graphs'] or previous['requests']!=config['requests']:
        raise ValueError('Recurrent comparison uses different graphs/counts')
    with np.load(reference/'outcomes.npz',allow_pickle=False) as data:
        if not np.array_equal(data['counts'],depth):raise ValueError('Recurrent outcome count order differs')
        final={k:data[f'final__{k}'] for k in METRICS}
    with np.load(reference/'ordinary_qwen'/'outcomes.npz',allow_pickle=False) as data:
        original_final=data['correct']
    series={'Original Qwen · trace':arms['original'],'Ordinary SFT · trace':arms['sft'],'Final recurrent + controller':final}
    styles=[dict(color='#8856a7',ls=':',marker='o'),dict(color='#d95f02',ls='--',marker='s'),dict(color='#1b9e77',ls='-',marker='^')]
    def save(fig,name):
        fig.savefig(plots/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(plots/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,key,title in zip(axes,['stopped_answer_correct','joint_success','strict_success'],
        ['Returned final letter','Final letter + exact termination','All transitions + exact termination'],strict=True):
        for (label,values),style in zip(series.items(),styles,strict=True):
            ax.plot(depth,values[key].mean(0)*100,label=label,markevery=(styles.index(style)*5,24),ms=4,**style)
        if key=='stopped_answer_correct':ax.plot(depth,original_final.mean(0)*100,label='Original Qwen · final-only',color='#555555',ls='-.')
        ax.set(xlabel='Requested transitions',ylabel='Success (%)',title=title,ylim=(-2,102));ax.grid(alpha=.2)
        ax.legend(fontsize=7,loc='lower left')
    save(fig,'01_quality_by_depth')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,arm in enumerate(('original','sft')):
        values=arms[arm];categories=[]
        for n in range(len(depth)):
            malformed=values['malformed'][:,n];budget=values['missing_stop'][:,n]&~malformed
            categories.append([values['exact_stop'][:,n].mean(),values['early_stop'][:,n].mean(),
                values['late_stop'][:,n].mean(),budget.mean(),malformed.mean()])
        # Mutually exclusive partition; cyclic final-letter aliases never alter it.
        rates=np.asarray(categories).T
        if not np.allclose(rates.sum(0),1):raise ValueError('Termination categories do not partition queries')
        axes[0,col].stackplot(depth,rates*100,labels=['Exact EOS length','Early EOS','Late EOS','Budget without EOS','Malformed'],
            colors=['#1b9e77','#d95f02','#7570b3','#666666','#e7298a'])
        axes[0,col].set(title=f'{arm}: termination',xlabel='Requested transitions',ylabel='Queries (%)',ylim=(0,100))
        axes[0,col].legend(fontsize=7,loc='upper right')
        # Condition on fixed N=256. First error depends on the requested prompt.
        failure=values['first_failure'][:,-1];symbol=values['first_symbol_error'][:,-1]
        for data,label,color in ((failure,'First prefix failure (incl. missing/format)', '#d95f02'),(symbol,'First wrong emitted symbol','#7570b3')):
            histogram=np.bincount(data,minlength=257)[1:257]
            axes[1,col].step(depth,np.cumsum(histogram)/len(metadata)*100,where='post',label=label,color=color,
                ls='--' if data is symbol else '-',lw=1.7 if data is symbol else 2.6)
        axes[1,col].set(title=f'{arm}: failure onset at request N=256',xlabel='Transition position',ylabel='Graphs failed by position (%)',ylim=(0,100))
        axes[1,col].legend(fontsize=7);axes[1,col].grid(alpha=.2)
    save(fig,'02_trace_failures')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    modes=['random_function','permutation','full_cycle']
    for ax,arm in zip(axes,('original','sft'),strict=True):
        matrix=np.array([arms[arm]['strict_success'][[m['graph_mode']==mode for m in metadata]].mean(0)*100 for mode in modes])
        view=ax.imshow(matrix,aspect='auto',interpolation='nearest',vmin=0,vmax=100,cmap='viridis',extent=(.5,256.5,2.5,-.5))
        ax.set(yticks=[0,1,2],yticklabels=modes,xlabel='Requested transitions',title=f'{arm}: strict success by graph type')
        fig.colorbar(view,ax=ax,label='Success (%)')
    save(fig,'03_graph_strata')
    metrics=Path(config['training'])/'metrics.jsonl';rows=[json.loads(l) for l in metrics.read_text().splitlines()]
    train=[r for r in rows if r['event']=='train'];val=[r for r in rows if r['event']=='validation']
    fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    axes[0].plot([r['step'] for r in train],[r['train']['loss'] for r in train],alpha=.45,label='Training response CE')
    axes[0].plot([r['step'] for r in val],[r['validation']['selection_loss'] for r in val],marker='o',label='Trained-count validation CE')
    axes[0].set(xlabel='Optimizer updates',ylabel='Mean response-token CE per example',title='Ordinary SFT loss');axes[0].legend()
    marker_shapes=['o','s','^','D','x'];line_styles=['-','--',':','-.','--']
    for j,d in enumerate(('6','7','9','11','12')):
        axes[1].plot([r['step'] for r in val],[r['validation']['by_depth'][d]['teacher_forced_response_accuracy']*100 for r in val],
            marker=marker_shapes[j],markevery=(j,5),ls=line_styles[j],label=f'N={d}'+(' (held out)' if d in ('7','9','11') else ''))
    axes[1].set(xlabel='Optimizer updates',ylabel='Teacher-forced whole response (%)',ylim=(-2,102),title='Teacher-forced validation, not free generation');axes[1].legend()
    for j,d in enumerate(('6','7','9','11','12')):
        axes[2].plot([r['step'] for r in val],[r['validation']['free_generation'][d]['strict_success']*100 for r in val],
            marker=marker_shapes[j],markevery=(j,5),ls=line_styles[j],label=f'N={d}'+(' (held out)' if d in ('7','9','11') else ''))
    axes[2].set(xlabel='Optimizer updates',ylabel='Strict trace + EOS success (%)',ylim=(-2,102),title='Small free-generation validation panel');axes[2].legend()
    save(fig,'04_sft_training')
    # Paired graph-cluster differences against the frozen final recurrent system.
    rng=np.random.default_rng(config['bootstrap_seed']);groups={}
    for i,m in enumerate(metadata):groups.setdefault((m['dataset_seed'],m['graph_mode']),[]).append(i)
    draws=np.concatenate([rng.choice(ids,(config['bootstrap_repeats'],len(ids)),replace=True) for ids in groups.values()],1)
    deltas=[]
    for arm in ('original','sft'):
        for metric in ('stopped_answer_correct','joint_success','strict_success'):
            for cohort,mask in [('all',depth>0),('extrapolation',depth>12)]:
                differences=final[metric][:,mask].mean(1)-arms[arm][metric][:,mask].mean(1)
                lo,hi=np.quantile(differences[draws].mean(1),[.025,.975])
                deltas.append(dict(arm=arm,metric=metric,cohort=cohort,recurrent_minus_trace=float(differences.mean()),low=float(lo),high=float(hi)))
    write_csv(root/'paired_deltas.csv',deltas)
    sources.update(recurrent_outcomes=sha256_file(reference/'outcomes.npz'),final_only_outcomes=sha256_file(reference/'ordinary_qwen'/'outcomes.npz'),training_metrics=sha256_file(metrics))
    for arm in ('original','sft'):sources[arm]=sha256_file(root/arm/'outcomes.npz')
    atomic_json(plots/'provenance.json',dict(sources_sha256=sources,plot_source_sha256=sha256_file(Path(__file__))))
    (plots/'index.md').write_text('# Ordinary trace baseline comparisons\n\n'
        'Separate addition to the completed population figures. Trace termination is emitted-state length plus EOS, not latent loop count.\n\n'
        +''.join(f'- [{name}](./{name}.png) · [PDF](./{name}.pdf)\n' for name in ('01_quality_by_depth','02_trace_failures','03_graph_strata','04_sft_training'))
        +'\nOrdinary SFT adapts all Qwen weights on the same examples as R, using full-vocabulary autoregressive CE. It has more trainable capacity than R. The final controller additionally saw counts 1–63; this is not a fully exposure-matched stopping comparison.\n')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('eval/pointer_traces/seed61-20261010'))
    args=parser.parse_args();root=args.input;config=json.loads((root/'protocol.json').read_text())
    if sha256_file(Path(config['graphs']))!=config['graphs_sha256']:raise ValueError('Panel checksum differs')
    tasks,metadata=load_graphs(Path(config['graphs']))
    arrays={arm:audit_arm(root,arm,tasks,metadata,config) for arm in ('original','sft')}
    render(root,config,arrays,metadata)
    atomic_json(root/'completion.json',dict(status='complete',independent_audits_passed=True,
        queries_per_arm=len(tasks)*len(config['requests']),plots='plots/index.md'))
    (root/'COMPLETE').write_text('Independent trace audits and comparison plots complete.\n')


if __name__=='__main__':main()
