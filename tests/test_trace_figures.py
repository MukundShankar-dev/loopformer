"""Exercise the whole plotting bundle with explicit synthetic, non-result arrays."""
import json
from pathlib import Path

import numpy as np

from scripts.eval.benchmark_metrics import METRICS
from scripts.eval.trace_analysis import render


def test_trace_figures_and_termination_partition(tmp_path):
    root=tmp_path/'trace';root.mkdir();reference=tmp_path/'reference';reference.mkdir()
    (reference/'ordinary_qwen').mkdir();training=tmp_path/'training';training.mkdir()
    shape=(9,256);counts=np.arange(1,257)
    def values(cap):
        exact=np.broadcast_to(counts<=cap,shape).copy();zeros=np.zeros(shape,dtype=bool)
        result={k:zeros.copy() for k in METRICS}
        for k in ('nominal_final_correct','complete_trajectory','stopped_answer_correct','joint_success','strict_success','exact_stop'):result[k]=exact.copy()
        result['early_stop']=~exact
        result.update(malformed=zeros,first_failure=np.where(exact,0,cap+1),first_symbol_error=np.zeros(shape,dtype=int))
        return result
    arms=dict(original=values(3),sft=values(12));final=values(256)
    np.savez_compressed(reference/'outcomes.npz',counts=counts,**{f'final__{k}':final[k] for k in METRICS})
    np.savez_compressed(reference/'ordinary_qwen'/'outcomes.npz',correct=values(1)['stopped_answer_correct'])
    for arm in arms:
        (root/arm).mkdir();np.savez_compressed(root/arm/'outcomes.npz',**arms[arm])
    validation=dict(selection_loss=.5,by_depth={str(d):dict(teacher_forced_response_accuracy=.75) for d in (6,7,9,11,12)},
        free_generation={str(d):dict(strict_success=.5) for d in (6,7,9,11,12)})
    rows=[dict(event='train',step=1,train=dict(loss=1.)),dict(event='validation',step=0,validation=validation),dict(event='validation',step=1,validation=validation)]
    (training/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config=dict(graphs='synthetic',requests=counts.tolist(),reference_results=str(reference),training=str(training),bootstrap_seed=239,bootstrap_repeats=5)
    (reference/'protocol.json').write_text(json.dumps(config))
    (root/'protocol.json').write_text(json.dumps(config))
    metadata=[dict(dataset_seed=seed,graph_mode=mode) for seed in (307,311,313) for mode in ('random_function','permutation','full_cycle')]
    render(root,config,arms,metadata)
    assert len(list((root/'plots').glob('*.png')))==4
    assert len(list((root/'plots').glob('*.pdf')))==4
    assert (root/'paired_deltas.csv').exists() and (root/'plots'/'provenance.json').exists()
