"""Graph-cluster structure strata and first-failure risk exposures."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.eval.benchmark_metrics import wilson_interval
from scripts.eval.loop_metrics import write_csv
from scripts.eval.paper_suite import load_graphs
from scripts.eval.pointer_failure_metrics import reference_trace


def bin_name(value:int,bins:list)->str:
    for low,high in bins:
        if low<=value<=high:return str(low) if low==high else f'{low}–{high}'
    raise ValueError('Feature outside declared bins')


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=Path('eval/pointer_analysis/paper-20261009'))
    args=parser.parse_args();root=args.input;config=json.loads((root/'protocol.json').read_text())
    tasks,metadata=load_graphs(Path(config['graphs']));outcomes=np.load(root/'outcomes.npz',allow_pickle=False)
    features=[]
    for task,meta in zip(tasks,metadata,strict=True):
        degrees=Counter(dict(task.mapping).values())
        features.append(dict(**meta,cycle_bin=bin_name(meta['cycle_period'],config['cycle_bins']),
            transient_bin=bin_name(meta['transient_length'],config['transient_bins']),
            orbit_size=meta['cycle_period']+meta['transient_length'],max_indegree=max(degrees.values()),
            first_revisit_loop=meta['cycle_period']+meta['transient_length']))
    write_csv(root/'graph_features.csv',features)
    rows=[];exposures=[]
    requests=config['fixed_prompt_failure_requests']
    for arm in config['models']:
        name=arm['name'];prefix=outcomes[f'{name}__complete_trajectory']
        for n in requests:
            for field in ('graph_mode','dataset_seed','cycle_bin','transient_bin','max_indegree','orbit_size'):
                groups=defaultdict(list)
                for i,f in enumerate(features):
                    # Within-mode period/transient/indegree strata avoid pooled mixture claims.
                    groups[(f['graph_mode'],f[field]) if field not in ('graph_mode','dataset_seed') else ('all',f[field])].append(i)
                for (mode,value),ids in groups.items():
                    failed=int((~prefix[ids,n-1]).sum());lo,hi=wilson_interval(failed,len(ids))
                    rows.append(dict(model=name,requested_depth=n,feature=field,graph_mode=mode,value=value,
                        graphs=len(ids),failures=failed,failure_rate=failed/len(ids),low=lo,high=hi))
        with np.load(root/name/'diagnostic_traces.npz',allow_pickle=False) as z:
            traces=z['predictions']
        for j,n in enumerate(requests):
            numerators=Counter();denominators=Counter()
            # Every exposure is a transition still at risk of its FIRST error.
            for i,(task,f) in enumerate(zip(tasks,features,strict=True)):
                mapping=dict(task.mapping);targets=reference_trace(mapping,task.initial_state,n)
                positions={source:index+1 for index,(source,_) in enumerate(task.mapping)}
                visited={task.initial_state}
                for loop in range(1,n+1):
                    entering=task.initial_state if loop==1 else targets[loop-2]
                    slot=positions[entering];slotbin='1–7' if slot<=7 else '8–13' if slot<=13 else '14–20' if slot<=20 else '21–26'
                    keys=[('active_rule_position',slotbin),('target_letter',targets[loop-1]),
                          ('first_revisit_proximity','before' if loop<f['first_revisit_loop'] else 'at' if loop==f['first_revisit_loop'] else 'after')]
                    wrong=chr(int(traces[i,j,loop-1])+65)!=targets[loop-1]
                    for field,value in keys:
                        key=f['graph_mode'],field,value
                        denominators[key]+=1;numerators[key]+=wrong
                    if wrong:break
                    visited.add(targets[loop-1])
            for (mode,field,value),exposed in denominators.items():
                failures=numerators[mode,field,value]
                exposures.append(dict(model=name,requested_depth=n,graph_mode=mode,feature=field,value=value,
                    at_risk_transitions=exposed,first_failures=failures,conditional_failure_rate=failures/exposed,
                    interval_scope='Descriptive correlated transition exposures; no binomial interval'))
    write_csv(root/'structure_failure_rates.csv',rows);write_csv(root/'first_error_exposures.csv',exposures)


if __name__=='__main__':main()
