"""Reproduce the saved-artifact review; run from repo root with .venv/bin/python.

No inference. Uses the existing reference/first-failure validator. Outputs are
exploratory descriptive statistics, never thresholds fitted for confirmation.
"""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.training.data import read_tasks
from scripts.eval.failure_diagnostics import analyze_failures

OUT = Path(__file__).resolve().parent
PROBE = ROOT / 'eval/pointer_probes/baseline2500-20260930T005754Z-2318'
inputs = {}


def digest(path):
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    digest(path)
    return list(csv.DictReader(path.open(newline='')))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def med(rows, field):
    return median(float(r[field]) for r in rows) if rows else None


runs = {}
checkpoint_results = []
for summary_path in sorted((ROOT / 'eval/pointer_loops').glob('*fresh30k*/summary.json')):
    digest(summary_path)
    summary = json.loads(summary_path.read_text())
    step = int(summary_path.parent.name.rsplit('-', 1)[1])
    split = Path(summary['data']).stem
    data = ROOT / 'data/pointer/seed-17' / f'{split}.jsonl'
    digest(data)
    assert inputs[str(data.relative_to(ROOT))] == summary['data_sha256']
    tasks = read_tasks(data, split)
    failures, risk, aggregate = analyze_failures(summary_path.parent / 'trajectories.csv', tasks, summary['loops'])
    groups = defaultdict(list)
    for r in read_csv(summary_path.parent / 'trajectories.csv'):
        if r['intermediate_target']:
            groups[r['example_id']].append(r)
    for rows in groups.values():
        rows.sort(key=lambda r: int(r['loop']))
    first = {r['example_id']: r['first_error_loop'] for r in failures}
    runs[split, step] = groups, first
    checkpoint_results.append(dict(step=step, split=split, **aggregate))
    write_csv(f'{split}-{step}-risk.csv', risk)
    write_csv(f'{split}-{step}-first-errors.csv', failures)

pairs = []
for split in ['validation', 'depth_test']:
    base, bf = runs[split, 2500]
    for step in [3250, 3750]:
        later, lf = runs[split, step]
        assert base.keys() == later.keys()
        for key, rows in base.items():
            assert [r['intermediate_target'] for r in rows] == [r['intermediate_target'] for r in later[key]]
            d = len(rows)
            a, b = bf.get(key, d + 1), lf.get(key, d + 1)
            pairs.append(dict(split=split, later_checkpoint=step, example_id=key, depth=d,
                              first_error_2500=a if a<=d else '', first_error_later=b if b<=d else '',
                              prefix_2500=a-1, prefix_later=b-1,
                              complete_2500=a>d, complete_later=b>d))
write_csv('checkpoint-pairs.csv', pairs)
comparisons=[]
for split in ['validation','depth_test']:
    for step in [3250,3750]:
        for depths in [None,*range(1,17)]:
            rows=[r for r in pairs if r['split']==split and r['later_checkpoint']==step and (depths is None or r['depth']==depths)]
            if not rows:continue
            comparisons.append(dict(split=split, later_checkpoint=step, depth=depths, n=len(rows),
                lost_complete=sum(r['complete_2500'] and not r['complete_later'] for r in rows),
                gained_complete=sum(not r['complete_2500'] and r['complete_later'] for r in rows),
                earlier_failure=sum(r['prefix_later']<r['prefix_2500'] for r in rows),
                later_failure=sum(r['prefix_later']>r['prefix_2500'] for r in rows)))

# Original first-failure windows; historical traces only contain target margins.
windows=[]
for (split,step),(groups,first) in runs.items():
    for key,t in first.items():
        for offset in [-3,-2,-1,0]:
            if t+offset>=1:
                r=groups[key][t+offset-1]
                windows.append(dict(split=split,checkpoint=step,example_id=key,first_error=t,offset=offset,
                                    target_margin=float(r['intermediate_margin'])))
write_csv('failure-windows.csv',windows)

states=read_csv(PROBE/'states.csv')
groups=defaultdict(list)
for r in states:groups[r['example_id'],r['variant']].append(r)
for rs in groups.values():rs.sort(key=lambda r:int(r['loop']))
probe_windows=[]
aligned=[]
for (key,variant),rs in groups.items():
    if variant!='original':continue
    old=runs['depth_test',2500][0][key]
    assert [r['prediction'] for r in rs]==[r['prediction'] for r in old]
    assert [r['target'] for r in rs]==[r['intermediate_target'] for r in old]
    failures=[int(r['loop']) for r in rs if r['correct']=='False']
    if not failures:continue
    t=failures[0]
    for offset in [-3,-2,-1,0]:
        if t+offset>=1:
            probe_windows.append(dict(first_error=t,offset=offset,**rs[t+offset-1]))
    if t>6:
        suffix=groups[key,'suffix'][t-7]
        assert suffix['target']==rs[t-1]['target']
        row=dict(example_id=key,original_loop=t,suffix_loop=t-6)
        for label,r in [('original',rs[t-1]),('suffix',suffix)]:
            for field in ['prediction','target','correct','target_rank','top_symbols','answer_entropy','predicted_margin','target_margin','answer_relative_update_norm','sequence_relative_update_norm']:
                row[label+'_'+field]=r[field]
        aligned.append(row)
write_csv('probe-failure-windows.csv',probe_windows)
write_csv('probe-first-error-pairs.csv',aligned)
fields=['target_rank','answer_entropy','predicted_margin','target_margin','answer_relative_update_norm','sequence_relative_update_norm']
probe_summary=[]
# Same late-failure cohort for every offset: no varying denominator.
late_ids={r['example_id'] for r in probe_windows if r['first_error']>6}
for offset in [-3,-2,-1,0]:
    rows=[r for r in probe_windows if r['offset']==offset and r['example_id'] in late_ids]
    probe_summary.append(dict(offset=offset,n=len(rows),**{k:med(rows,k) for k in fields}))
paired_summary={label:{k:med(aligned,label+'_'+k) for k in fields} for label in ['original','suffix']}
whole_failure_windows=[]
for step in [2500,3250,3750]:
    for offset in [-3,-2,-1,0]:
        rows=[r for r in windows if r['split']=='depth_test' and r['checkpoint']==step
              and r['first_error']>6 and r['offset']==offset]
        whole_failure_windows.append(dict(checkpoint=step,offset=offset,n=len(rows),target_margin=med(rows,'target_margin')))
trained_range=[]
for step in [2500,3250,3750]:
    gs,_=runs['validation',step]
    eligible=[rs for rs in gs.values() if len(rs)<=6]
    trained_range.append(dict(checkpoint=step,n=len(eligible),complete=sum(all(r['intermediate_correct']=='True' for r in rs) for rs in eligible)))
result=dict(trained_range=trained_range,whole_failure_windows=whole_failure_windows,checkpoints=checkpoint_results,comparisons=comparisons,late_probe_failure_windows=probe_summary,
            aligned_first_error_medians=paired_summary,
            first_error_target_rank_counts=dict(Counter(r['original_target_rank'] for r in aligned)),
            source_files=inputs)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_files','comparisons']},indent=2))
print('Overall checkpoint comparisons:',json.dumps([r for r in comparisons if r['depth'] is None],indent=2))
