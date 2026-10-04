"""Paired requested-count diagnostic using the shared full-loop evaluator."""
import argparse
from collections import defaultdict
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.dataset.pointer import PointerExample, render_prompt, validate_example


DEPTHS = [6, 7, 8, 9, 10, 12, 16]


def variants(tasks: list[PointerExample], count: int, depths: list[int] | None = None) -> list[PointerExample]:
    """Same table/start, only Steps changes; all variants remain in development."""
    depths = DEPTHS if depths is None else depths
    if not depths or depths != sorted(set(depths)) or min(depths) < 1:
        raise ValueError('Depths must be sorted unique positive counts')
    source_depth = max(depths)
    selected = [x for x in tasks if x.task_depth == source_depth][:count]
    if len(selected) != count:
        raise ValueError(f"Need {count} depth-{source_depth} source tasks")
    result = []
    for task in selected:
        for depth in depths:
            variant = replace(task, example_id=f"{task.example_id}-steps-{depth}", task_depth=depth,
                              intermediate_states=task.intermediate_states[:depth], final_state=task.intermediate_states[depth-1],
                              prompt=render_prompt(task.mapping, task.initial_state, depth))
            validate_example(variant)
            result.append(variant)
    return result


def paired_summary(path: Path) -> list[dict]:
    groups = defaultdict(dict)
    for row in csv.DictReader(path.open()):
        source, depth = row['example_id'].rsplit('-steps-', 1)
        groups[source].setdefault(int(depth), []).append(row)
    output = []
    for source, variants_ in groups.items():
        for rows in variants_.values():
            rows.sort(key=lambda r: int(r['loop']))
        anchor = variants_[max(variants_)]
        for depth, rows in variants_.items():
            rows.sort(key=lambda r: int(r['loop']))
            stop = next((int(r['loop']) for r in rows if float(r['stop_logit']) >= 0), None)
            output.append({'source_id':source, 'requested_depth':depth,
                           'common_prefix_prediction_agreement':sum(a['prediction']==b['prediction'] for a,b in zip(rows[:depth],anchor[:depth]))/depth,
                           'trajectory_correct':all(r['intermediate_correct']=='True' for r in rows[:depth]),
                           'first_stop_loop':stop or '', 'exact_stop':stop==depth,
                           'early_stop':stop is not None and stop<depth,
                           'cap_fallback':stop is None})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/pointer/seed-17/depth_test.jsonl'))
    parser.add_argument('--depths', type=int, nargs='+', default=DEPTHS)
    parser.add_argument('--loops', type=int, default=20)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--device', choices=['cpu','cuda','mps'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    from scripts.eval.tracking import add_tracking_arguments, report_saved_run
    add_tracking_arguments(parser)
    args=parser.parse_args()
    if args.count<1 or args.batch_size<1:parser.error('count and batch-size must be positive')
    from rich.console import Console
    from rich.progress import Progress
    from scripts.training.data import read_tasks, encode_tasks
    from scripts.training.evaluation import evaluate
    from scripts.eval.loop_metrics import write_csv, summarize_trajectories
    from scripts.eval.pointer_task import sha256_file
    from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
    if args.loops < max(args.depths):parser.error('loops must cover all requested counts')
    tasks=variants(read_tasks(args.data,'depth_test'),args.count,args.depths)
    console=Console()
    console.print(f'Paired Steps: {args.count} rule tables × {len(args.depths)} counts = {len(tasks)} prompts; {args.loops} loops each')
    if args.dry_run:
        console.print('No model loading or writes. Requested counts:',args.depths)
        return
    output=args.output or Path('eval/pointer_probes')/f'paired-steps-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}'
    if output.exists():parser.error('Output already exists')
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    model,tokenizer,spec=load_recurrent_checkpoint(args.model,device=args.device)
    if spec.get('recurrence_mode')!='fixed_prompt' or model.completion_head is None:
        parser.error('This probe requires a fixed-prompt completion checkpoint')
    encoded=encode_tasks(tasks,tokenizer,dict(zip(spec['symbols'],spec['token_ids'])),256)
    output.mkdir(parents=True)
    (output/'tasks.jsonl').write_text(''.join(json.dumps(task.to_dict(), sort_keys=True)+'\n' for task in tasks))
    with Progress() as progress:
        task=progress.add_task('Paired Steps',total=len(tasks))
        metrics=evaluate(model,encoded,spec['token_ids'],tokenizer.pad_token_id,batch_size=args.batch_size,
                         loops=args.loops,output=output/'trajectories.csv',record_states=True,
                         progress=lambda done,total:progress.update(task,completed=done))
    pairs=paired_summary(output/'trajectories.csv')
    write_csv(output/'pairs.csv',pairs)
    examples,diagnostics=summarize_trajectories(output/'trajectories.csv')
    write_csv(output/'examples.csv',examples)
    sources=[*Path('scripts/recurrent_qwen').glob('*.py'),Path(__file__),Path('scripts/training/evaluation.py')]
    summary={'model':str(args.model),'data_sha256':sha256_file(args.data),'checkpoint_sha256':sha256_file(args.model/'adapter_model.pt'),
             'recurrent_config_sha256':sha256_file(args.model/'recurrent_config.json'),
             'tasks_sha256':sha256_file(output/'tasks.jsonl'),
             'source_sha256':{str(p):sha256_file(p) for p in sources},
             'status':'complete','depths':args.depths,'rule_tables':args.count,'loops':args.loops,'device':args.device,'batch_size':args.batch_size,'dtype':'float32','seed':17,'deterministic_algorithms':True,'threshold':0.5,
             'stopping_semantics':'offline first crossing from full traces; not measured stopped latency',
             'metrics':metrics,'diagnostics':diagnostics}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    console.print(f'Saved paired predictions, stop logits and working-state norms/deltas: {output}')

    report_saved_run(output, args)


if __name__=='__main__':main()
