import csv,json,statistics,hashlib
from pathlib import Path
from collections import defaultdict,Counter
root=next(Path('eval/pointer_loops').glob('depth6-fixed*'))
out={}
# Offline artifact audit; run from the repository root. No model loading.
for split in ['validation','depth_test']:
 summary=json.loads((root/split/'summary.json').read_text())
 assert hashlib.sha256((Path('data/pointer/seed-17')/(split+'.jsonl')).read_bytes()).hexdigest()==summary['data_sha256']
 data={r['example_id']:r for r in map(json.loads,(Path('data/pointer/seed-17')/(split+'.jsonl')).read_text().splitlines())}
 traces=defaultdict(list)
 for r in csv.DictReader((root/split/'trajectories.csv').open()):traces[r['example_id']].append(r)
 risk=defaultdict(lambda:Counter());cats=Counter();separable=Counter();stop_correct=Counter();perdepth={}
 for eid,rs in traces.items():
  ex=data[eid];d=ex['task_depth'];path=[ex['initial_state']]+ex['intermediate_states'];mapping=dict(ex['mapping']);alive=True
  for t,r in enumerate(rs[:d],1):
   assert r['intermediate_target']==path[t]
   ok=r['prediction']==path[t]
   if alive:
    risk[t]['eligible']+=1;risk[t]['correct']+=ok
    if not ok:
     p=r['prediction'];cats['repeat_previous' if p==path[t-1] else 'earlier_path' if p in path[:t-1] else 'future_path' if p in path[t+1:] else 'off_path']+=1
   alive=alive and ok
  z=[float(r['stop_logit']) for r in rs]
  separable[str(d)]+=int(d==1 or z[d-1]>max(z[:d-1]))
  first=next((i+1 for i,x in enumerate(z) if x>=0),20)
  stop_correct['correct_pointer_when_stopped']+=rs[first-1]['intermediate_correct']=='True'
  stop_correct['total']+=1
  if str(d) not in perdepth:perdepth[str(d)]={'pre':[],'at':[]}
  if d>1:perdepth[str(d)]['pre'].append(max(z[:d-1]))
  perdepth[str(d)]['at'].append(z[d-1])
 out[split]={'risk':dict(risk),'first_error_types':cats,'threshold_separable_counts':separable,'pointer_at_stop':stop_correct,'logits':{d:{k:statistics.median(v) if v else None for k,v in vals.items()} for d,vals in perdepth.items()}}
 print(split,json.dumps(out[split],indent=2))
print('CHECKPOINT VALIDATION depth7,8 fulltrajectory/exactstop out of32')
for p in sorted(Path('models/stage1_pointer/depth6-fixed-prompt-seed37').glob('validation-step-*.csv')):
 g=defaultdict(list)
 for r in csv.DictReader(p.open()):g[r['example_id']].append(r)
 vals={}
 for dep in [7,8]:
  groups=[rs[:dep] for rs in g.values() if int(rs[0]['task_depth'])==dep]
  vals[dep]=[sum(all(r['intermediate_correct']=='True' for r in rs) for rs in groups),sum(next((i+1 for i,r in enumerate(rs) if float(r['stop_logit'])>=0),0)==dep for rs in groups)]
 print(p.stem,vals)
Path(__file__).with_name('analysis.json').write_text(json.dumps(out,indent=2)+'\n')

# Exact sweep of shared logit thresholds. This fits development labels and is
# an optimistic diagnostic bound, not a selected or confirmed stopping policy.
all_traces=defaultdict(list)
for split in ['validation','depth_test']:
 for r in csv.DictReader((root/split/'trajectories.csv').open()):all_traces[r['example_id']].append(r)
threshold_results={}
for lo,hi in [(1,6),(7,8),(9,10),(9,16),(1,16)]:
 rows=[rs for rs in all_traces.values() if lo<=int(rs[0]['task_depth'])<=hi]
 events=defaultdict(int)
 for rs in rows:
  d=int(rs[0]['task_depth']);a=max([float(r['stop_logit']) for r in rs[:d-1]],default=-1e9);b=float(rs[d-1]['stop_logit'])
  if b>a:events[a]+=1;events[b]-=1
 best=cur=0;threshold=None;keys=sorted(events)
 for i,x in enumerate(keys[:-1]):
  cur+=events[x]
  if cur>best:best=cur;threshold=(x+keys[i+1])/2
 threshold_results[f'{lo}-{hi}']={'best_exact':best,'examples':len(rows),'logit_threshold':threshold}
print('Optimistic shared-threshold bounds:',threshold_results)
out['threshold_bounds']=threshold_results
out['artifact_root']=str(root)
out['inputs_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*/trajectories.csv'))}
Path(__file__).with_name('analysis.json').write_text(json.dumps(out,indent=2)+'\n')
