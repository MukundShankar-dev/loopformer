import gc,inspect,json,time,torch,numpy as np
from pathlib import Path
import scripts.recurrent_qwen.partitioned as original
from scripts.eval.paper_suite import load_graphs,partition_inputs
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
source=inspect.getsource(original.partitioned_forward)
source=source.replace('            if not need.any():', '            kept = need.nonzero().flatten()\n            if not kept.numel():')
source=source.replace('[need]', '.index_select(0, kept)')
namespace=dict(vars(original));exec(source,namespace);faster=namespace['partitioned_forward']
p=json.loads(Path('configs/pointer_analysis_protocol.json').read_text());tasks,_=load_graphs(Path(p['graphs']))
torch.set_num_threads(4);torch.manual_seed(239);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
rows=[]
for name in ['ce_full','fixed6']:
 arm=next(x for x in p['models'] if x['name']==name)
 model,tok,spec=load_recurrent_checkpoint(Path(arm['path']),device='cuda');model.eval().requires_grad_(False);model.config._attn_implementation='sdpa'
 inputs=partition_inputs(tasks[:4],tok,p['requests'],'cuda');h=torch.tensor(p['requests'],device='cuda').repeat(4)
 before=None;seconds=[]
 for mode,fn in [('original',original.partitioned_forward),('single_index',faster)]:
  torch.cuda.synchronize();t=time.perf_counter();r=fn(model,*inputs,spec['token_ids'],272,horizons=h);torch.cuda.synchronize();seconds.append(time.perf_counter()-t)
  if before is None: before=r.scores.cpu().clone();before_stops=r.stops.cpu().clone() if r.stops is not None else None
  else:
   equal=torch.equal(before,r.scores.cpu());stop_equal=before_stops is None if r.stops is None else torch.equal(before_stops,r.stops.cpu())
   print(json.dumps({'model':name,'original_seconds':seconds[0],'single_index_seconds':seconds[1],'speed_ratio':seconds[0]/seconds[1],'bitwise_scores_equal':equal,'bitwise_stops_equal':stop_equal}),flush=True)
   rows.append({'model':name,'original_seconds':seconds[0],'single_index_seconds':seconds[1],'speed_ratio':seconds[0]/seconds[1],'bitwise_scores_equal':equal,'bitwise_stops_equal':stop_equal})
   Path('/tmp/paper_index_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
  del r;gc.collect()
 del model,inputs,before,before_stops;gc.collect();torch.cuda.empty_cache()
