import gc,json,time,torch,numpy as np
from pathlib import Path
from scripts.eval.paper_suite import load_graphs,partition_inputs
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
from scripts.recurrent_qwen.partitioned import partitioned_forward
from peft.tuners.lora.layer import LoraLayer
p=json.loads(Path('configs/pointer_analysis_protocol.json').read_text());tasks,_=load_graphs(Path(p['graphs']))
torch.set_num_threads(4);torch.manual_seed(239);torch.use_deterministic_algorithms(True)
reference=np.load('eval/pointer_analysis/paper-20261009/ce_full/graphs-0000-0004.npz')['predictions']
rows=[]
for mode in ['tf32','merged_lora']:
 torch.set_float32_matmul_precision('high' if mode=='tf32' else 'highest')
 model,tok,spec=load_recurrent_checkpoint(Path(p['models'][0]['path']),device='cuda');model.eval().requires_grad_(False);model.config._attn_implementation='sdpa'
 if mode=='merged_lora':
  for module in model.modules():
   if isinstance(module,LoraLayer):module.merge(safe_merge=True)
 inputs=partition_inputs(tasks[:4],tok,p['requests'],'cuda');h=torch.tensor(p['requests'],device='cuda').repeat(4)
 torch.cuda.synchronize();tick=time.perf_counter()
 result=partitioned_forward(model,*inputs,spec['token_ids'],272,horizons=h)
 torch.cuda.synchronize();seconds=time.perf_counter()-tick
 prediction=result.scores.argmax(-1).cpu().numpy().astype('uint8').reshape(4,256,272)
 valid=np.arange(272)[None,None,:]<np.array(p['requests'])[None,:,None]
 bad=(prediction!=reference)&valid
 final_bad=np.array([prediction[:,j,n-1]!=reference[:,j,n-1] for j,n in enumerate(p['requests'])]).sum()
 row={'candidate':mode,'seconds':seconds,'reference_chunk_seconds':float(np.load('eval/pointer_analysis/paper-20261009/ce_full/graphs-0000-0004.npz')['seconds']),'changed_observed_readouts':int(bad.sum()),'changed_queries':int(bad.any(-1).sum()),'changed_final_readouts':int(final_bad),'exact_decoded_agreement':not bool(bad.any()),'fp32_matmul_precision':torch.get_float32_matmul_precision(),'weights_on_disk_modified':False,'scope':'First four full-horizon CE graphs, every requested N; bounded candidate rejection test, not full population approval'}
 rows.append(row);print(json.dumps(row),flush=True)
 Path('/tmp/paper_acceleration_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
 del result,inputs,model;gc.collect();torch.cuda.empty_cache()
