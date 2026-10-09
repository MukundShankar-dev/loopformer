import json,time,gc,torch
from pathlib import Path
from scripts.eval.paper_suite import load_graphs,partition_inputs
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.partitioned import partitioned_forward
p=json.loads(Path('configs/pointer_analysis_protocol.json').read_text());tasks,_=load_graphs(Path(p['graphs']))
torch.set_num_threads(4);torch.manual_seed(239);torch.use_deterministic_algorithms(True)
model,tok,spec=load_recurrent_checkpoint(Path(p['models'][0]['path']),device='cuda');model.eval().requires_grad_(False);model.config._attn_implementation='sdpa'
rows=[];reference=None
for g in (4,8,16):
 inputs=partition_inputs(tasks[:g],tok,p['requests'],'cuda');h=torch.tensor(p['requests'],device='cuda').repeat(g).clamp(max=32)
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 torch.cuda.synchronize();tick=time.perf_counter();r=partitioned_forward(model,*inputs,spec['token_ids'],32,horizons=h);torch.cuda.synchronize();sec=time.perf_counter()-tick
 pred=r.scores[:4*256].argmax(-1).cpu();same=True if reference is None else bool(torch.equal(reference,pred));reference=pred if reference is None else reference
 row=dict(graph_batch=g,loops=32,seconds=sec,graphs_per_second=g/sec,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,first_four_graphs_decoded_identical=same,scope='Bounded 32-loop deterministic FP32 profile, not full-horizon batch-equivalence gate');rows.append(row);print(json.dumps(row),flush=True);del r,inputs;gc.collect()
Path('/tmp/paper_batch_profile.json').write_text(json.dumps(rows,indent=2)+'\n')
