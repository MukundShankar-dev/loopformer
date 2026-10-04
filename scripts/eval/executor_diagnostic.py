"""One matched development panel for executor, controller and precision failure modes."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.dataset.pointer import PointerExample, execute, render_prompt, validate_example, orbit_structure


def matched_tasks(tasks: list[PointerExample], count: int, depths: list[int]) -> list[PointerExample]:
    selected, seen = [], set()
    for task in tasks:
        if task.mapping_sha256 not in seen:
            selected.append(task)
            seen.add(task.mapping_sha256)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'Need {count} unique development graphs')
    variants = []
    for task in selected:
        for depth in depths:
            states = execute(dict(task.mapping), task.initial_state, depth)
            variant = replace(task, schema_version=2, example_id=f'{task.example_id}-steps-{depth}',
                task_depth=depth, intermediate_states=states, final_state=states[-1],
                prompt=render_prompt(task.mapping, task.initial_state, depth))
            validate_example(variant)
            variants.append(variant)
    return variants


def analyze_panel(path: Path, tasks: list[PointerExample]) -> dict:
    """Strict trajectories, local transitions, R/C disagreements and cycle strata."""
    by_id = {task.example_id: task for task in tasks}
    histories = {}
    for row in csv.DictReader(path.open()):
        histories.setdefault(row['example_id'], []).append(row)
    per_loop, by_orbit, pairs = {}, {}, {}
    first_errors = {}
    for example_id, rows in histories.items():
        task = by_id[example_id]
        rows.sort(key=lambda row: int(row['loop']))
        mapping = dict(task.mapping)
        orbit = orbit_structure(mapping, task.initial_state)
        group = f"transient-{orbit['transient_length']}_period-{orbit['cycle_period']}"
        cohort = by_orbit.setdefault(group, {'examples': 0, 'complete': 0})
        cohort['examples'] += 1
        nominal = rows[:task.task_depth]
        first = next((int(row['loop']) for row in nominal if row['intermediate_correct'] != 'True'), None)
        cohort['complete'] += int(first is None)
        first_errors[str(first) if first else 'none'] = first_errors.get(str(first) if first else 'none', 0) + 1
        previous = task.initial_state
        for row in nominal:
            t = int(row['loop'])
            entry = per_loop.setdefault(str(t), {'count': 0, 'correct': 0, 'prefix_at_risk': 0,
                'first_errors': 0, 'follows_predicted_previous': 0, 'r_correct_c_wrong': 0, 'c_correct_r_wrong': 0})
            entry['count'] += 1
            correct = row['intermediate_correct'] == 'True'
            entry['correct'] += int(correct)
            entry['prefix_at_risk'] += int(first is None or first >= t)
            entry['first_errors'] += int(first == t)
            entry['follows_predicted_previous'] += int(row['prediction'] == mapping[previous])
            if 'r_intermediate_correct' in row:
                r_correct = row['r_intermediate_correct'] == 'True'
                entry['r_correct_c_wrong'] += int(r_correct and not correct)
                entry['c_correct_r_wrong'] += int(correct and not r_correct)
            previous = row['prediction']
        source, depth = example_id.rsplit('-steps-', 1)
        pairs.setdefault(source, {})[int(depth)] = rows
    comparisons = []
    for source, variants in pairs.items():
        anchor = variants[max(variants)]
        for depth, rows in variants.items():
            comparisons.append({'source_id': source, 'requested_depth': depth,
                'common_prefix_disagreements': sum(a['prediction'] != b['prediction'] for a, b in zip(rows[:depth], anchor[:depth])),
                'compared_loops': depth})
    for entry in per_loop.values():
        entry['conditional_transition_accuracy'] = (1 - entry['first_errors'] / entry['prefix_at_risk']
                                                    if entry['prefix_at_risk'] else None)
    failures = sum(count for t, count in first_errors.items() if t != 'none')
    mean_failure = (sum(int(t) * count for t, count in first_errors.items() if t != 'none') / failures
                    if failures else None)
    return {'first_error_mean_among_failures': mean_failure, 'failed_queries': failures,
            'per_loop': per_loop, 'first_error_counts': first_errors, 'by_orbit': by_orbit,
            'matched_counts': comparisons,
            'interpretation': 'R/C disagreement localizes readout disagreement, not causal fault; local consistency is not gold-trajectory correctness'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/pointer/seed-61-independent/validation.jsonl'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--depths', type=int, nargs='+', default=[6, 8, 12, 16, 24, 32, 64])
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--precision-check', action='store_true', help='Compare FP32/eager, FP32/SDPA and BF16/SDPA on identical cases')
    from scripts.eval.tracking import add_tracking_arguments, report_saved_run
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1 or not args.depths or min(args.depths) < 1 or max(args.depths) > 256:
        parser.error('Invalid panel sizes/depths')
    if args.precision_check and args.device != 'cuda':
        parser.error('BF16 precision check requires CUDA')
    from scripts.training.data import read_tasks, encode_tasks
    from scripts.training.evaluation import evaluate
    from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint
    from scripts.eval.pointer_task import sha256_file, synchronize
    from scripts.eval.loop_metrics import summarize_trajectories, write_csv
    from rich.progress import Progress
    import torch
    from time import perf_counter
    torch.set_num_threads(4)
    torch.manual_seed(17)
    tasks = matched_tasks(read_tasks(args.data, 'validation'), args.count, sorted(set(args.depths)))
    output = args.output or Path('eval/pointer_diagnostics') / f'executor-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}'
    if output.exists():
        parser.error('Output already exists')
    model, tokenizer, spec = load_recurrent_checkpoint(args.model, device=args.device)
    items = encode_tasks(tasks, tokenizer, dict(zip(spec['symbols'], spec['token_ids'])), 512)
    output.mkdir(parents=True)
    (output / 'tasks.jsonl').write_text(''.join(json.dumps(t.to_dict()) + '\n' for t in tasks))
    modes = [('float32', 'eager')]
    if args.precision_check:
        modes += [('float32', 'sdpa'), ('bf16', 'sdpa')]
    summaries, predictions, all_rows, loop_rows, pair_rows = {}, {}, [], [], []
    for precision, attention in modes:
        name = f'{precision}-{attention}'
        model.config._attn_implementation = attention
        path = output / name
        path.mkdir()
        synchronize(torch.device(args.device))
        began = perf_counter()
        with Progress() as progress:
            task = progress.add_task(name, total=len(items))
            metrics = evaluate(model, items, spec['token_ids'], tokenizer.pad_token_id,
                batch_size=args.batch_size, output=path / 'trajectories.csv', record_states=True,
                precision=precision, progress=lambda done, total: progress.update(task, completed=done))
        synchronize(torch.device(args.device))
        seconds = perf_counter() - began
        examples, errors = summarize_trajectories(path / 'trajectories.csv')
        write_csv(path / 'examples.csv', examples)
        panel = analyze_panel(path / 'trajectories.csv', tasks)
        pairs = panel.pop('matched_counts')
        write_csv(path / 'pairs.csv', pairs)
        pair_rows.extend({'condition': name, **row} for row in pairs)
        loop_rows.extend({'checkpoint': name, 'task_depth': 'pooled', 'loop': t, **row} for t, row in panel['per_loop'].items())
        summary = {'status': 'complete', 'precision': precision, 'attention': attention, 'metrics': metrics,
                   'diagnostics': errors, 'panel': panel, 'evaluation_seconds': seconds}
        (path / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        summaries[name] = summary
        predictions[name] = {(r['example_id'], r['loop']): r for r in csv.DictReader((path / 'trajectories.csv').open())
                             if r['post_completion'] == 'False'}
        all_rows.extend({'condition': name, **row} for row in csv.DictReader((path / 'trajectories.csv').open()))
    write_csv(output / 'trajectories.csv', all_rows)
    write_csv(output / 'loop_risk.csv', loop_rows)
    write_csv(output / 'pairs.csv', pair_rows)
    reference = predictions['float32-eager']
    precision_comparison = {}
    for name, values in predictions.items():
        if set(values) != set(reference):
            raise ValueError('Precision runs did not evaluate identical targets')
        precision_comparison[name] = {
            'nominal_rows': len(reference),
            'prediction_disagreements': sum(values[k]['prediction'] != row['prediction'] for k, row in reference.items()),
            'max_abs_target_margin_change': max(abs(float(values[k]['intermediate_margin']) - float(row['intermediate_margin'])) for k, row in reference.items())}
    summary = {'status': 'complete', 'model': str(args.model), 'spec': spec, 'device': args.device,
        'batch_size': args.batch_size, 'count': args.count, 'depths': args.depths,
        'data_sha256': sha256_file(args.data), 'checkpoint_sha256': sha256_file(args.model / 'adapter_model.pt'),
        'recurrent_config_sha256': sha256_file(args.model / 'recurrent_config.json'),
        'tasks_sha256': sha256_file(output / 'tasks.jsonl'),
        'source_sha256': {str(p): sha256_file(p) for pattern in ('scripts/recurrent_qwen/*.py', 'scripts/training/*.py', 'scripts/eval/*.py') for p in Path('.').glob(pattern)},
        'precision_comparison': precision_comparison, 'runs': summaries,
        'timing_scope': 'includes forward, metrics and CSV; no inference speedup claim'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report_saved_run(output, args)
    print(f'Saved matched-count, orbit, first-error, R/C and precision diagnostics: {output}')


if __name__ == '__main__':
    main()
