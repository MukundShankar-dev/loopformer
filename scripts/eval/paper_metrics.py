"""Independent raw-table scoring for the full paper panel and failure analyses."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.dataset.pointer import SYMBOLS
from scripts.eval.benchmark_metrics import wilson_interval
from scripts.eval.loop_metrics import write_csv
from scripts.eval.paper_suite import load_graphs
from scripts.eval.pointer_failure_metrics import first_error_category, reference_trace
from scripts.eval.pointer_task import sha256_file


METRICS = ('nominal_final_correct','complete_trajectory','exact_stop','joint_success','strict_success',
           'early_stop','late_stop','missing_stop','stopped_answer_correct','correct_letter_wrong_time')


def score_arrays(predictions: np.ndarray, stops: np.ndarray, targets: np.ndarray,
                 counts: np.ndarray) -> dict[str,np.ndarray]:
    """Inputs uint symbol IDs; -2=not applicable, -1=no crossing. No cap success."""
    g,k = stops.shape
    if targets.shape[0] != g or len(counts)!=k or predictions.shape[0]!=g:
        raise ValueError('Misaligned graph/count tensors')
    isolated = predictions.ndim==2
    p = np.broadcast_to(predictions[:,None,:],(g,k,predictions.shape[-1])) if isolated else predictions
    if p.shape[:2]!=(g,k) or p.shape[-1]<max(counts) or targets.shape[1]<p.shape[-1]:
        raise ValueError('Incomplete trajectory coverage')
    hits = p == targets[:,None,:p.shape[-1]]
    first = np.where((~hits).any(-1),(~hits).argmax(-1)+1,0).astype('int16')
    final = np.take_along_axis(hits, np.broadcast_to(counts[None,:,None]-1,(g,k,1)),-1)[...,0]
    complete = (first==0)|(first>counts[None,:])
    result = dict(nominal_final_correct=final,complete_trajectory=complete,first_error_loop=np.where(first<=counts[None,:],first,0).astype('int16'))
    if (stops==-2).all():
        return {**result,**{m:None for m in METRICS[2:]}}
    if (stops==-2).any() or (stops>p.shape[-1]).any() or (stops==0).any():
        raise ValueError('Invalid stopping semantics')
    exact = stops==counts[None,:]
    actual = np.where(stops>0,stops,p.shape[-1])
    returned = np.take_along_axis(p,actual[...,None]-1,-1)[...,0]
    wanted = targets[:,counts-1]
    correct = returned==wanted
    result.update(exact_stop=exact,joint_success=exact&correct,strict_success=exact&complete,
        early_stop=(stops>0)&(stops<counts[None,:]),late_stop=stops>counts[None,:],missing_stop=stops==-1,
        stopped_answer_correct=correct,correct_letter_wrong_time=correct&~exact)
    return result


def diagnostic_rows(tasks: list, metadata: list[dict], predictions: np.ndarray,
                    direct: np.ndarray, counts: np.ndarray, stops: np.ndarray, model: str,
                    requests: list[int]) -> tuple[list[dict],list[dict],list[dict]]:
    """Fixed-prompt survival/risk sets, mutually exclusive first errors, episodes."""
    first_rows, loop_rows, episodes = [],[],[]
    for n in requests:
        j = counts.tolist().index(n)
        traces = predictions[:,j,:n] if predictions.ndim==3 else predictions[:,:n]
        targets = np.array([[ord(c)-65 for c in reference_trace(dict(t.mapping),t.initial_state,n)] for t in tasks])
        hit = traces==targets
        errors = np.where((~hit).any(-1),(~hit).argmax(-1)+1,n+1)
        for loop in range(1,n+1):
            at_risk = int((errors>=loop).sum())
            failed = int((errors==loop).sum())
            loop_rows.append(dict(model=model,requested_depth=n,loop=loop,graphs=len(tasks),
                survivors=int((errors>loop).sum()),at_risk=at_risk,first_failures=failed,
                survival=float((errors>loop).mean()),hazard=failed/at_risk if at_risk else None,
                step_correct=int(hit[:,loop-1].sum()),step_accuracy=float(hit[:,loop-1].mean())))
        for i,(task,meta) in enumerate(zip(tasks,metadata,strict=True)):
            error=int(errors[i]); wrong=~hit[i]
            prior_wrong=int(wrong[:-1].sum())
            recovered=int((wrong[:-1]&hit[i,1:]).sum())
            consistent=sum(int(traces[i,l]==ord(dict(task.mapping)[SYMBOLS[traces[i,l-1]]])-65) for l in range(1,n))
            base=dict(model=model,requested_depth=n,**meta)
            if error<=n:
                t=error-1; target=''.join(SYMBOLS[c] for c in targets[i]); predicted=SYMBOLS[traces[i,t]]
                r=SYMBOLS[direct[i,t]] if direct.size else None
                category=first_error_category(dict(task.mapping),task.initial_state,target,predicted,t)
                first_rows.append(dict(**base,loop=error,target=target[t],prediction=predicted,
                    category=category,r_prediction=r,readout_category=(None if r is None else
                       'R correct / C wrong' if r==target[t] else 'R and C wrong, same letter' if r==predicted else 'R and C wrong, different letters'),
                    wrong_loops=int(wrong.sum()),correct_prefix=error-1,correct_final=bool(hit[i,-1]),
                    prior_wrong_transitions=prior_wrong,recoveries=recovered,
                    decoded_edge_consistent=consistent,edge_exposure=n-1))
            cursor=0
            while cursor<n:
                if not wrong[cursor]: cursor+=1; continue
                end=cursor+1
                while end<n and wrong[end]: end+=1
                episodes.append(dict(**base,start_loop=cursor+1,duration=end-cursor,right_censored=end==n))
                cursor=end
    return first_rows,loop_rows,episodes


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('eval/pointer_analysis/paper-20261009'))
    args=parser.parse_args(); root=args.input
    config=json.loads((root/'protocol.json').read_text()); tasks,metadata=load_graphs(Path(config['graphs']))
    counts=np.array(config['requests']); g=len(tasks); k=len(counts); cap=config['safety_cap']
    targets=np.array([[ord(c)-65 for c in reference_trace(dict(t.mapping),t.initial_state,cap)] for t in tasks],dtype='uint8')
    table=[]; first_all=[]; loop_all=[]; episode_all=[]; execution_stats=[]; timing_stats=[]; arm_arrays={}; hashes={}
    for arm in config['models']:
        name=arm['name']; directory=root/name
        summary=json.loads((directory/'summary.json').read_text())
        if summary['status']!='complete' or not summary['weights_unchanged'] or not summary['native_fidelity']:
            raise ValueError(f'{name}: incomplete or unverified extraction')
        chunks=sorted(directory.glob('graphs-*.npz')); coverage=[]
        parts=[];stops=[];direct=[]
        for path in chunks:
            with np.load(path,allow_pickle=False) as z:
                coverage.extend(z['graph_indices'].tolist())
                if not np.array_equal(z['counts'],counts): raise ValueError('Chunk count mismatch')
                parts.append(z['predictions']);stops.append(z['stops']);
                if z['direct'].size: direct.append(z['direct'])
            hashes[str(path.relative_to(root))]=sha256_file(path)
        if coverage!=list(range(g)): raise ValueError('Missing/repeated/reordered graph chunk')
        p=np.concatenate(parts);stop=np.concatenate(stops);r=np.concatenate(direct) if direct else np.empty(0,dtype='uint8')
        values=score_arrays(p,stop,targets,counts)
        arm_arrays[name]={m:v for m,v in values.items() if v is not None}
        arm_arrays[name]['stops']=stop
        for j,n in enumerate(counts):
            for metric in METRICS:
                array=values[metric]
                success=int(array[:,j].sum()) if array is not None else None
                low,high=wilson_interval(success,g) if success is not None else (None,None)
                table.append(dict(model=name,requested_depth=int(n),metric=metric,successes=success,
                                  graphs=g,rate=success/g if success is not None else None,low=low,high=high))
        f,l,e=diagnostic_rows(tasks,metadata,p,r,counts,stop,name,config['fixed_prompt_failure_requests'])
        first_all.extend(f);loop_all.extend(l);episode_all.extend(e)
        for n in config['fixed_prompt_failure_requests']:
            selected=[row for row in f if row['requested_depth']==n]
            wrong=sum(row['wrong_loops'] for row in selected)
            prior=sum(row['prior_wrong_transitions'] for row in selected)
            recovery=sum(row['recoveries'] for row in selected)
            prefix=sum(row['correct_prefix'] for row in selected)+(g-len(selected))*n
            execution_stats.append(dict(model=name,requested_depth=n,graphs=g,failed_graphs=len(selected),
                restricted_mean_correct_prefix=prefix/g,wrong_loop_fraction=wrong/(g*n),
                prior_wrong_transitions=prior,reference_recoveries=recovery,
                conditional_reference_recovery=recovery/prior if prior else None,
                wrong_episodes=sum(row['requested_depth']==n for row in e),
                right_censored_episodes=sum(row['requested_depth']==n and row['right_censored'] for row in e)))
        if values['exact_stop'] is not None:
            for j,n in enumerate(counts):
                residual=stop[:,j][stop[:,j]>0]-n
                timing_stats.append(dict(model=name,requested_depth=int(n),graphs=g,observed_crossings=len(residual),
                    signed_mean=float(residual.mean()) if len(residual) else None,
                    absolute_mean=float(abs(residual).mean()) if len(residual) else None,
                    median=float(np.median(residual)) if len(residual) else None,
                    p10=float(np.quantile(residual,.1)) if len(residual) else None,
                    p90=float(np.quantile(residual,.9)) if len(residual) else None))
        diagnostic=np.stack([p[:,counts.tolist().index(n)] if p.ndim==3 else p for n in config['fixed_prompt_failure_requests']],1)
        np.savez_compressed(directory/'diagnostic_traces.npz',predictions=diagnostic,direct=r,counts=np.array(config['fixed_prompt_failure_requests']))
        # Preserve compact per-query outcomes and all requested traces in source chunks.
        with gzip.open(directory/'decisions.csv.gz','wt',newline='') as handle:
            fields=['model','graph_index','requested_depth','first_stop','first_error_loop',*METRICS]
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for i in range(g):
                for j,n in enumerate(counts):
                    row=dict(model=name,graph_index=i,requested_depth=int(n),first_stop=int(stop[i,j]) if stop[i,j]>0 else None,
                             first_error_loop=int(values['first_error_loop'][i,j]) or None)
                    row.update({m:bool(values[m][i,j]) if values[m] is not None else None for m in METRICS})
                    writer.writerow(row)
    write_csv(root/'quality_by_depth.csv',table);write_csv(root/'first_errors.csv',first_all)
    write_csv(root/'fixed_request_loops.csv',loop_all);write_csv(root/'wrong_episodes.csv',episode_all)
    write_csv(root/'execution_statistics.csv',execution_stats);write_csv(root/'timing_residuals.csv',timing_stats)
    # Same graph bootstrap draws across all arms, stratified within nine seed/mode strata.
    rng=np.random.default_rng(config['uncertainty']['bootstrap_seed'])
    groups={}
    for i,m in enumerate(metadata): groups.setdefault((m['dataset_seed'],m['graph_mode']),[]).append(i)
    draws=np.concatenate([rng.choice(ids,size=(config['uncertainty']['bootstrap_repeats'],len(ids)),replace=True) for ids in groups.values()],1)
    intervals=[]
    ranges=[('all',1,256),('1-6',1,6),('7-12',7,12),('13-32',13,32),('33-64',33,64),('65-128',65,128),('129-256',129,256)]
    for name,arrays in arm_arrays.items():
        for metric in METRICS:
            if metric not in arrays: continue
            for label,a,b in ranges:
                pergraph=arrays[metric][:,a-1:b].mean(-1)
                boot=pergraph[draws].mean(-1);lo,hi=np.quantile(boot,[.025,.975])
                intervals.append(dict(model=name,metric=metric,depth_range=label,rate=float(pergraph.mean()),low=float(lo),high=float(hi),graphs=g))
    write_csv(root/'cluster_intervals.csv',intervals)
    deltas=[]
    for name,arrays in arm_arrays.items():
        if name=='final': continue
        for metric in ('nominal_final_correct','complete_trajectory'):
            for label,a,b in ranges:
                d=(arm_arrays['final'][metric][:,a-1:b].mean(-1)-arrays[metric][:,a-1:b].mean(-1))
                lo,hi=np.quantile(d[draws].mean(-1),[.025,.975])
                deltas.append(dict(baseline=name,comparison='final',metric=metric,depth_range=label,
                                   delta=float(d.mean()),low=float(lo),high=float(hi)))
    write_csv(root/'paired_deltas.csv',deltas)
    flat={f'{name}__{key}':value for name,arrays in arm_arrays.items() for key,value in arrays.items()}
    np.savez_compressed(root/'outcomes.npz',**flat,counts=counts,targets=targets)
    audit=dict(passed=True,graphs=g,queries_per_model=g*k,models=list(arm_arrays),
               exact_reference='Dictionary traversal from raw rules, never persisted intermediate labels',
               source_chunk_sha256=hashes,protocol_sha256=sha256_file(root/'protocol.json'),
               graph_sha256=sha256_file(Path(config['graphs'])),rows=len(table),
               missing_stopping='Undefined for headless CE; cap fallback never counts as learned stopping')
    (root/'reference_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__': main()
