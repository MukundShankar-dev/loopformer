"""Verify final composition using the already-confirmed, unchanged executor."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import torch
from rich.console import Console

from scripts.dataset.pointer import PointerExample, SYMBOLS
from scripts.eval.frozen_pointer_benchmark import checkpoint_hashes, controller_panel, native_fidelity
from scripts.eval.audit_pointer_benchmark import audit_benchmark
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/controller_number_precision_evaluation.json'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();settings=json.loads(args.config.read_text())
    if args.dry_run:
        Console().print('Preview: reuse unchanged confirmed R, verify all 256 count stops, 36 native calls, numeric rollouts through 8192; no loading/writes.');return
    output=Path(settings['output'])
    if output.exists():raise ValueError('Precision evaluation output exists; refuse overwrite')
    source_results=Path(settings['source_results'])
    source_summary=json.loads((source_results/'summary.json').read_text())
    if source_summary['status']!='complete' or not source_summary['native_fidelity']['passed']:
        raise ValueError('Need completed source graph/native confirmation')
    maximum,cap=source_summary['config']['max_depth'],source_summary['config']['safety_cap']
    source_model=Path(source_summary['checkpoint'])
    if checkpoint_hashes(source_model)!=source_summary['checkpoint_sha256']:
        raise ValueError('Confirmed source model files changed')
    audit_benchmark(source_results,Path(settings['graphs']))
    before=checkpoint_hashes(Path(settings['model']))
    old=torch.load(source_model/'adapter_model.pt',weights_only=True,map_location='cpu')
    new=torch.load(Path(settings['model'])/'adapter_model.pt',weights_only=True,map_location='cpu')
    if set(old)!=set(new):raise ValueError('Model tensor names differ')
    preserved=[k for k in old if not k.startswith('completion_head.cell.')]
    for k in preserved:torch.testing.assert_close(old[k],new[k],atol=0,rtol=0)
    del old,new
    manifest=json.loads((source_results/'dataset_manifest.json').read_text())
    if sha256_file(Path(settings['graphs']))!=manifest['graphs_sha256']:
        raise ValueError('Confirmed graph bytes differ')
    tasks=[PointerExample(**json.loads(line)) for line in Path(settings['graphs']).read_text().splitlines()]
    graph_rows=list(csv.DictReader((source_results/'graphs.csv').open()))
    predictions=torch.tensor([[SYMBOLS.index(c) for c in row['predictions']] for row in graph_rows])
    output.mkdir(parents=True)
    freeze={'checkpoint':settings['model'],'inference_sha256':before,
        'threshold':source_summary['config']['threshold'],'safety_cap':cap,
        'protocol_sha256':sha256_file(args.config),'source_quality_summary_sha256':sha256_file(source_results/'summary.json')}
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    torch.set_num_threads(4)
    device=settings.get('device','cuda')
    model,tokenizer,spec=load_recurrent_checkpoint(Path(settings['model']),device=device)
    model.eval().requires_grad_(False)
    if model.router is None or model.completion_head.kind!='shared_number':
        raise ValueError('Need frozen isolated scalar controller')
    logits,stops,counts=controller_panel(model,tokenizer,spec['token_ids'],tasks,manifest['graph_metadata'],maximum,cap)
    original_counts=list(csv.DictReader((source_results/'controller_counts.csv').open()))
    if stops!=[int(row['first_stop']) if row['first_stop'] else None for row in original_counts]:
        raise ValueError('Stop timing differs; cannot inherit graph quality results')
    write_csv(output/'controller_counts.csv',counts)
    config={**source_summary['config'],'model':settings['model'],'native_counts':settings['native_counts']}
    native=native_fidelity(config,output,tasks,manifest['graph_metadata'],predictions,logits,device)
    subprocess.run([sys.executable,'-m','scripts.eval.number_reader_diagnostic','--model',settings['model'],
        '--output',str(output/'numeric'),'--rollout-max',str(settings['numeric_rollout_max']),
        '--reader-max',str(settings['numeric_reader_max']),'--device',device,
        '--data',str(Path(source_summary['config']['existing_data'])/'train.jsonl')],check=True)
    numeric=json.loads((output/'numeric/summary.json').read_text())
    if checkpoint_hashes(Path(settings['model']))!=before:raise ValueError('Frozen files changed')
    passed=(source_summary['acceptance']['passed'] and native['passed']
            and numeric['exact_stop_questions']==settings['numeric_rollout_max'])
    result={'status':'complete','passed':passed,'checkpoint':settings['model'],'inference_sha256':before,
        'config':settings,'config_sha256':sha256_file(args.config),'source_results':str(source_results),
        'unchanged_non_cell_tensors':len(preserved),'matching_count_timings':maximum,'native_fidelity':native,
        'numeric':numeric,'inherited_graph_quality':source_summary['cohorts'],
        'inherited_acceptance':source_summary['acceptance'],
        'scope':'Component composition on an opened but verified graph panel; unchanged R and reader; new numerical rollouts through 8192. Not new independent graph confirmation.',
        'inference_files_unchanged':True}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    Console().print(f'Final composition · passed {passed} · native {native["questions"]}/{native["questions"]} · countdown {numeric["exact_stop_questions"]}/{settings["numeric_rollout_max"]}')
    if not passed:raise ValueError('Precision repair failed; preserve negative result')


if __name__=='__main__':main()
