"""Second, scalar reference audit of every saved paper decision and aggregate.

Does not call the vectorized scorer. Raw rule dictionaries produce expected
bytes; prefix equality, first error, returned symbol and stop categories are
reconstructed independently for every graph/count pair.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from rich.progress import track

from scripts.eval.paper_suite import load_graphs
from scripts.eval.paper_metrics import METRICS
from scripts.eval.pointer_task import sha256_file


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=Path('eval/pointer_analysis/paper-20261009'))
    args=parser.parse_args();root=args.input;config=json.loads((root/'protocol.json').read_text());tasks,metadata=load_graphs(Path(config['graphs']))
    cap=config['safety_cap'];targets=[]
    for task in tasks:
        mapping=dict(task.mapping);state=task.initial_state;trace=[]
        for _ in range(cap):state=mapping[state];trace.append(ord(state)-65)
        targets.append(bytes(trace))
    quality=list(csv.DictReader((root/'quality_by_depth.csv').open()));expected_aggregate={};audited=0
    for arm in track(config['models'],description='Independent scalar decision audit'):
        name=arm['name'];directory=root/name
        chunks=sorted(directory.glob('graphs-*.npz'));predictions=[];stops=[];indices=[]
        for path in chunks:
            with np.load(path,allow_pickle=False) as z:
                predictions.append(z['predictions']);stops.append(z['stops']);indices.extend(z['graph_indices'].tolist())
        if indices!=list(range(len(tasks))):raise ValueError('Independent chunk coverage failed')
        p=np.concatenate(predictions);stop=np.concatenate(stops);totals={n:{m:0 for m in METRICS} for n in config['requests']}
        with gzip.open(directory/'decisions.csv.gz','rt',newline='') as handle:
            reader=csv.DictReader(handle)
            for i in range(len(tasks)):
                target=targets[i]
                for j,n in enumerate(config['requests']):
                    row=next(reader)
                    if row['model']!=name or int(row['graph_index'])!=i or int(row['requested_depth'])!=n:raise ValueError('Decision identity/order mismatch')
                    pred=(p[i,j] if p.ndim==3 else p[i]).tobytes()
                    complete=pred[:n]==target[:n]
                    first=next((t+1 for t in range(n) if pred[t]!=target[t]),None) if not complete else None
                    expected=dict(nominal_final_correct=pred[n-1]==target[n-1],complete_trajectory=complete)
                    t=int(stop[i,j]);crossing=t if t>0 else None
                    if t==-2:
                        expected.update(dict.fromkeys(METRICS[2:]))
                    else:
                        actual=t if t>0 else cap;returned=pred[actual-1]==target[n-1];exact=t==n
                        expected.update(exact_stop=exact,joint_success=returned and exact,strict_success=complete and exact,
                            early_stop=0<t<n,late_stop=t>n,missing_stop=t==-1,stopped_answer_correct=returned,
                            correct_letter_wrong_time=returned and not exact)
                    if row['first_stop']!=(str(crossing) if crossing is not None else '') or row['first_error_loop']!=(str(first) if first is not None else ''):
                        raise ValueError('Saved first error/crossing differs from independent scalar execution')
                    for metric,value in expected.items():
                        if row[metric] != (str(value) if value is not None else ''):raise ValueError(f'Independent audit failed {name},g{i},N{n},{metric}')
                        if value is None:totals[n][metric]=None
                        else:totals[n][metric]+=value
                    audited+=1
            if next(reader,None) is not None:raise ValueError('Unexpected trailing decisions')
        for n in config['requests']:
            for metric,total in totals[n].items():expected_aggregate[name,n,metric]=total
    for row in quality:
        key=row['model'],int(row['requested_depth']),row['metric'];total=expected_aggregate[key]
        if row['successes']!=(str(total) if total is not None else '') or int(row['graphs'])!=len(tasks):raise ValueError('Aggregate differs from independently audited decisions')
        if total is not None and abs(float(row['rate'])-total/len(tasks))>1e-12:raise ValueError('Wrong rate denominator')
    # Verify final reuse against the original full benchmark, including cyclic timing flags.
    final_rows=gzip.open(root/'final/decisions.csv.gz','rt',newline='')
    measured={(int(r['graph_index']),int(r['requested_depth'])):r for r in csv.DictReader(final_rows)}
    source=Path(config['source_results'])/'decisions.csv.gz';source_count=0
    with gzip.open(source,'rt',newline='') as handle:
        for row in csv.DictReader(handle):
            other=measured[int(row['graph_index']),int(row['depth'])]
            for metric in METRICS:
                if row[metric]!=other[metric]:raise ValueError(f'Final reuse differs from independently rescored source: {metric}')
            source_count+=1
    if source_count!=len(tasks)*len(config['requests']):raise ValueError('Original final reuse coverage differs')
    result=dict(passed=True,scalar_decisions_checked=audited,graphs=len(tasks),models=len(config['models']),
                aggregate_cells_checked=len(quality),original_final_decisions_crosschecked=source_count,
                source_sha256=sha256_file(source),audit_code_sha256=sha256_file(Path(__file__)))
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
