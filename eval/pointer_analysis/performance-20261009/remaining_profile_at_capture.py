import gc,json,time,torch
from pathlib import Path
from scripts.eval.paper_suite import load_graphs,partition_inputs
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.partitioned import partitioned_forward
p=json.loads(Path('configs/pointer_analysis_protocol.json').read_text());tasks,_=load_graphs(Path(p['graphs']))
torch.set_num_threads(4);torch.manual_seed(239);torch.use_deterministic_algorithms(True)
measurements=[]
for arm in p['models'][1:4]:
 model,tok,spec=load_recurrent_checkpoint(Path(arm['path']),device='cuda');model.eval().requires_grad_(False);model.config._attn_implementation='sdpa'
 inputs=partition_inputs(tasks[:4],tok,p['requests'],'cuda');h=torch.tensor(p['requests'],device='cuda').repeat(4)
 torch.cuda.synchronize();tick=time.perf_counter()
 result=partitioned_forward(model,*inputs,spec['token_ids'],p['safety_cap'],horizons=h)
 torch.cuda.synchronize();seconds=time.perf_counter()-tick
 row={'model':arm['name'],'graphs':4,'requests_per_graph':256,'cap':272,'seconds':seconds,'estimated_population_hours':seconds/4*1350/3600,'scope':'One full-horizon first-graph chunk. Timing estimate only; excludes native fidelity, tokenization, loading, export and other graph strata.'}
 measurements.append(row);print(json.dumps(row),flush=True)
 Path('/tmp/paper_remaining_profile.json').write_text(json.dumps(measurements,indent=2)+'\n')
 del result,inputs,model;gc.collect();torch.cuda.empty_cache()
