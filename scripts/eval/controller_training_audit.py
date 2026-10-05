"""Audit trained controllers without updates, Qwen loading, or feature extraction."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from rich.console import Console
from rich.table import Table

from scripts.eval.controller_audit_io import load_inputs, load_heads, read_json
from scripts.eval.controller_audit_metrics import cohort, fit_audit, gradient_audit, gradient_indices
from scripts.eval.controller_diagnostic import progress_bar
from scripts.eval.controller_features import replay
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file, synchronize
from scripts.eval.tracking import add_tracking_arguments, report_saved_run
from scripts.recurrent_qwen.interfaces import RecurrentController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('models/stage1_pointer/controller-remaining-seed61'))
    parser.add_argument('--features-cache', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    parser.add_argument('--checkpoints', nargs='+', choices=('best', 'last'), default=['best', 'last'])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gradient-batch-size', type=int, default=64)
    parser.add_argument('--gradient-graphs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=107)
    parser.add_argument('--dry-run', action='store_true')
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if min(args.batch_size, args.gradient_batch_size, args.gradient_graphs) < 1 or len(set(args.checkpoints)) != len(args.checkpoints):
        parser.error('Batch/graph counts must be positive; checkpoint choices must be unique')
    output = args.output or Path('eval/pointer_diagnostics') / f'controller-training-audit-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}'
    console = Console()
    console.print('[bold]Controller training audit[/bold] · no parameter updates · cached inputs only')
    console.print(f'Run: {args.run}\nCheckpoints: {", ".join(args.checkpoints)} · {args.device} · float32\nOutput: {output}')
    console.print(f'Full train/validation fit; gradients on {args.gradient_graphs} graphs per split · seed {args.seed}')
    if args.dry_run:
        console.print('Preview only: no cache/weights loaded, no downloads, no writes.')
        return
    if output.exists():
        raise ValueError(f'Output exists; choose a new --output: {output}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; specify --device cpu or mps explicitly')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable; specify --device cpu explicitly')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    began = perf_counter()
    cache, tasks, entries, provenance = load_inputs(args.run, args.features_cache, args.checkpoints)
    trained = provenance['trained_depths']
    indices = {split: gradient_indices(panel, args.gradient_graphs, args.seed) for split, panel in tasks.items()}
    # Confirm the inherited controller/cache contract before analyzing new heads.
    baseline = RecurrentController(cache['width'], cache['controller_width'])
    baseline.load_state_dict(cache['head'])
    baseline.to(args.device).eval()
    with torch.no_grad():
        for f in cache['features'].values():
            logits, _, _ = replay(baseline, f['context'][:16].to(args.device), f['working'][:16].to(args.device))
            torch.testing.assert_close(logits.cpu(), f['logits'][:16], atol=2e-5, rtol=1e-4)
    del baseline
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'running', 'kind': 'controller_training_audit', 'command': [sys.executable, '-m', 'scripts.eval.controller_training_audit', *sys.argv[1:]],
        'device': args.device, 'dtype': 'float32', 'batch_size': args.batch_size, 'gradient_batch_size': args.gradient_batch_size,
        'gradient_graphs': args.gradient_graphs, 'gradient_seed': args.seed, 'checkpoints': args.checkpoints,
        **provenance, 'source_sha256': {str(p): sha256_file(p) for pattern in ('scripts/eval/controller*.py', 'scripts/training/controller*.py', 'scripts/training/objective.py', 'scripts/recurrent_qwen/interfaces.py', 'scripts/eval/pointer_task.py', 'scripts/eval/tracking.py', 'audit_controller.sh') for p in Path('.').glob(pattern)},
        'target_semantics': 'N-t only through N; initialization t=0; exact stopping requires first stop at N',
        'limitations': ['Raw gradients, not Adam updates; no optimizer moments applied.',
            'Held-out-count gradients are retrospective diagnostics, never training or checkpoint selection.',
            'Counterfactual auxiliary controller gradients in weight-zero controls are labeled inactive.',
            'Numerical decodability and gradient conflict do not establish causal use or a unique failure mechanism.',
            'Cached replay timing is not end-to-end model inference speed.']}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    panel_rows = [{'split': split, 'example_id': panel[i].example_id, 'mapping_sha256': panel[i].mapping_sha256,
                   'depth': panel[i].task_depth, 'cohort': cohort(panel[i].task_depth, trained)}
                  for split, panel in tasks.items() for i in indices[split]]
    write_csv(output / 'gradient_panel.csv', panel_rows)
    fit_rows, cross_rows, norm_rows, cosine_rows, loss_rows, results = [], [], [], [], [], {}
    total = len(entries) * len(args.checkpoints) * 2
    with (output / 'predictions.csv').open('w', newline='') as handle, progress_bar() as progress:
        writer = None
        bar = progress.add_task('Audit checkpoint panels', total=total)
        for entry in entries:
            for checkpoint in args.checkpoints:
                head, readout, step, config = load_heads(entry, checkpoint, cache, args.device)
                frozen = [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for module in (head, readout)]
                label = f'{entry["name"]}/{checkpoint}'
                results[label] = {}
                for split, panel in tasks.items():
                    prefix = {'run': entry['name'], 'checkpoint': checkpoint, 'step': step,
                              'arm': entry['arm'], 'optimizer_seed': entry['optimizer_seed'], 'split': split}
                    progress.update(bar, description=f'{label} · {split} fit')
                    metrics, rows, cross = fit_audit(head, readout, cache['features'][split], panel, trained,
                                                  args.device, config.remaining_scale, args.batch_size)
                    # Reloaded selected validation decisions must reproduce the saved run.
                    if checkpoint == 'best' and split == 'validation':
                        expected = read_json(Path(entry['path']) / 'summary.json')['validation']
                        for name, values in metrics.items():
                            for key in ('exact_stop', 'remaining_initial_mae', 'remaining_decrement_mae'):
                                if abs(values[key] - expected[name][key]) > 2e-5:
                                    raise ValueError(f'Reloaded selected metrics disagree: {label}/{name}/{key}')
                    results[label][split] = metrics
                    fit_rows.extend({**prefix, 'cohort': name, **m} for name, m in metrics.items())
                    cross_rows.extend({**prefix, **r} for r in cross)
                    for row in rows:
                        record = {**prefix, **row}
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(record), lineterminator='\n'); writer.writeheader()
                        writer.writerow(record)
                    handle.flush()
                    for group in ('trained', 'interpolation', 'extrapolation'):
                        selected = [i for i in indices[split] if cohort(panel[i].task_depth, trained) == group]
                        if not selected:
                            continue
                        progress.update(bar, description=f'{label} · {split}/{group} gradients')
                        f = cache['features'][split]
                        gradient_bar = progress.add_task('Gradient batches', total=len(selected))
                        norms, cosines, losses = gradient_audit(head, readout,
                            f['context'][selected].to(args.device), f['working'][selected].to(args.device),
                            torch.tensor([panel[i].task_depth for i in selected], device=args.device),
                            config.remaining_scale, config.remaining_loss_weight, args.gradient_batch_size,
                            progress=lambda done: progress.update(gradient_bar, completed=done))
                        progress.remove_task(gradient_bar)
                        tag = {**prefix, 'cohort': group, 'held_out_count_diagnostic': group != 'trained'}
                        norm_rows.extend({**tag, **r} for r in norms)
                        cosine_rows.extend({**tag, **r} for r in cosines)
                        loss_rows.extend({**tag, **r} for r in losses)
                    progress.advance(bar)
                for module, state in zip((head, readout), frozen, strict=True):
                    for key, value in module.state_dict().items():
                        torch.testing.assert_close(value.cpu(), state[key], atol=0, rtol=0)
                    if any(p.grad is not None for p in module.parameters()):
                        raise RuntimeError('Audit unexpectedly populated parameter .grad')
                del head, readout
    for name, rows in [('fit_metrics', fit_rows), ('countdown_stop_crosstab', cross_rows),
                       ('gradient_norms', norm_rows), ('gradient_cosines', cosine_rows), ('loss_components', loss_rows)]:
        write_csv(output / f'{name}.csv', rows)
    # Detect concurrent input replacement; input files are never written by this code.
    for path, digest in provenance['input_sha256'].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError(f'Audit input changed during execution: {path}')
    synchronize(torch.device(args.device))
    summary = {**metadata, 'status': 'complete', 'weights_unchanged': True, 'parameter_grad_buffers_unchanged': True,
               'baseline_cache_replay_passed': True, 'wall_seconds': perf_counter() - began, 'fit': results}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'run.json').write_text(json.dumps({**metadata, 'status': 'complete'}, indent=2) + '\n')
    table = Table(title='Exact stopping · full cached panels')
    for name in ('Checkpoint', 'Train', 'Validation trained', 'Interpolation', 'Extrapolation'):
        table.add_column(name)
    for label, result in results.items():
        val = result['validation']
        table.add_row(label, f'{result["train"]["trained"]["exact_stop"]:.1%}',
                      *(f'{val[c]["exact_stop"]:.1%}' if c in val else '—' for c in ('trained', 'interpolation', 'extrapolation')))
    console.print(table)
    console.print(f'[green]Audit complete; weights unchanged.[/green] Results: {output}')
    report_saved_run(output, args)


if __name__ == '__main__':
    main()
