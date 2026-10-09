"""Read-only progress/ETA from atomic full-suite chunks, without model loading."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
from rich.console import Console
from rich.table import Table


def suite_status(root:Path)->dict:
    protocol=root/'protocol.json'
    if not protocol.exists():return {'status':'not started','models':[]}
    config=json.loads(protocol.read_text());rows=[]
    for arm in config['models']:
        directory=root/arm['name'];graphs=0;seconds=0.;chunks=0
        for path in directory.glob('graphs-*.npz'):
            if '.tmp.' in path.name:continue
            with ZipFile(path) as archive:
                indices=np.load(BytesIO(archive.read('graph_indices.npy')),allow_pickle=False)
                duration=float(np.load(BytesIO(archive.read('seconds.npy')),allow_pickle=False))
            graphs+=len(indices);seconds+=duration;chunks+=1
        summary=json.loads((directory/'summary.json').read_text()) if (directory/'summary.json').exists() else {}
        average=seconds/graphs if graphs else None
        rows.append(dict(model=arm['name'],graphs=graphs,total=config['graph_count'],
            native_checked=(directory/'native_fidelity.json').exists(),state=summary.get('status','extracting' if graphs else 'pending'),
            inference_seconds=seconds,remaining_inference_seconds=(config['graph_count']-graphs)*average if average is not None else None))
    snapshots=sum((p/'summary.json').exists() for p in (root/'executor_snapshots').glob('step-*'))
    return dict(status='complete' if (root/'COMPLETE').exists() else 'failed' if (root/'FAILED').exists() else 'running' if (root/'RUNNING').exists() else 'incomplete',
                models=rows,snapshots_complete=snapshots,snapshots_total=len(config['final_executor_history']['retained_weight_steps']),
                scored=(root/'reference_audit.json').exists(),independently_audited=(root/'independent_audit.json').exists(),
                plots_ready=(root/'plots/manifest.json').exists(),eta_scope='Observed extraction only; native gates, later models, scoring and plots excluded')


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=Path('eval/pointer_analysis/paper-20261009'));parser.add_argument('--json',action='store_true');args=parser.parse_args()
    result=suite_status(args.input)
    if args.json:print(json.dumps(result));return
    table=Table(title=f"Pointer paper suite · {result['status']}")
    for column in ('Model','Graphs','Native gate','State','Extraction ETA'):table.add_column(column)
    for row in result['models']:
        eta=row['remaining_inference_seconds'];label=f'{eta/3600:.1f} h' if eta is not None else 'unmeasured'
        table.add_row(row['model'],f"{row['graphs']}/{row['total']}",'passed' if row['native_checked'] else 'pending',row['state'],label)
    console=Console();console.print(table)
    if result['models']:
        console.print(f"Executor snapshots: {result['snapshots_complete']}/{result['snapshots_total']} · independent audit: {result['independently_audited']} · figures: {result['plots_ready']}")
        console.print(result['eta_scope'])


if __name__=='__main__':main()
