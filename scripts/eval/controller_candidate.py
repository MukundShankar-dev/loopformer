"""Evaluate one controller repair against immutable cached development observations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import load_examples, sha256_file
from scripts.eval.tracking import add_tracking_arguments, report_saved_run
from scripts.recurrent_qwen.interfaces import RecurrentController
from scripts.training.controller import ControllerConfig, evaluate_controller
from scripts.training.data import read_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--deep-features-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    config = ControllerConfig(**json.loads((args.run / 'config.json').read_text()))
    config.validate()
    if not config.remaining_readout:
        raise ValueError('Candidate evaluation requires an aligned numerical readout')
    metadata = json.loads((args.run / 'run.json').read_text())
    summary = json.loads((args.run / 'summary.json').read_text())
    if metadata['status'] != 'complete' or summary['status'] != 'complete':
        raise ValueError('Need a completed controller training run')
    cache_path = Path(metadata['features_cache'])
    if sha256_file(cache_path) != metadata['features_sha256']:
        raise ValueError('Training cache hash differs')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    deep = torch.load(args.deep_features_cache, map_location='cpu', weights_only=True)
    if (cache['format'] != 'controller-training-features-v1' or
            deep['identity']['source_sha256'] != cache['identity']['source_sha256'] or
            deep['identity']['source_config_sha256'] != cache['identity']['source_config_sha256'] or
            metadata['source_checkpoint_sha256'] != cache['identity']['source_sha256']):
        raise ValueError('Feature caches/source differ')
    validation = load_examples(args.run / 'validation_tasks.jsonl')
    if ([t.example_id for t in validation] != cache['identity']['validation_ids'] or
            sha256_file(args.run / 'validation_tasks.jsonl') != metadata['validation_tasks_sha256'] or
            any(t.split != 'validation' for t in validation)):
        raise ValueError('Validation task order differs from cache')
    deep_tasks = read_tasks(Path(config.data) / 'depth_test.jsonl', 'depth_test')
    if (sha256_file(Path(config.data) / 'depth_test.jsonl') != deep['identity']['data_sha256'] or
            [t.example_id for t in deep_tasks] != deep['identity']['ids']):
        raise ValueError('Deep tasks differ from cache')
    panels = {'validation': (validation, cache['features']['validation']),
              'depth_test': (deep_tasks, deep['features'])}
    del cache['features']['train']
    output = args.output or Path('eval/pointer_diagnostics') / f'{args.run.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}'
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    console = Console()
    table = Table(title='Exact stopping · unchanged threshold 0.5')
    for name in ('Checkpoint', 'Split', 'Counts', 'Exact stop', 'Initial MAE'):
        table.add_column(name)
    results, rows = {}, []
    for label in ('best', 'last'):
        head = RecurrentController(cache['width'], cache['controller_width']).to(config.device)
        readout = torch.nn.Linear(cache['controller_width'], 1).to(config.device)
        if label == 'best':
            head.load_state_dict(torch.load(args.run / 'best_controller.pt', weights_only=True, map_location=config.device))
            saved = torch.load(args.run / 'best_remaining_readout.pt', weights_only=True, map_location=config.device)
            if saved['step'] != summary['best_step'] or saved['scale'] != config.remaining_scale:
                raise ValueError('Selected head/readout alignment differs')
            readout.load_state_dict(saved['state_dict'])
        else:
            saved = torch.load(args.run / 'last_controller_state.pt', weights_only=True, map_location=config.device)
            head.load_state_dict(saved['head'])
            readout.load_state_dict(saved['remaining_readout'])
        results[label] = {}
        for split, (tasks, features) in panels.items():
            metrics, decisions = evaluate_controller(head, features, tasks, metadata['trained_depths'], config.device,
                remaining_readout=readout, remaining_scale=config.remaining_scale,
                training_loops=config.training_loops)
            results[label][split] = metrics
            write_csv(output / f'{label}_{split}_decisions.csv', decisions)
            rows.extend({'checkpoint': label, 'split': split, 'cohort': cohort, **values}
                        for cohort, values in metrics.items())
            for cohort in ('trained', 'interpolation', 'extrapolation', 'selection', 'seen_count_long_rollout', 'unseen_requested_count'):
                if cohort in metrics:
                    m = metrics[cohort]
                    table.add_row(label, split, cohort, f'{m["exact_stop"]:.2%}', f'{m["remaining_initial_mae"]:.3f}')
    write_csv(output / 'metrics.csv', rows)
    result = {'status': 'complete', 'run': str(args.run), 'metrics': results, 'threshold': .5,
              'scope': 'cached development replay; no measured native latency or confirmation',
              'input_sha256': {str(p): sha256_file(p) for p in [cache_path, args.deep_features_cache,
                  args.run / 'best_controller.pt', args.run / 'best_remaining_readout.pt',
                  args.run / 'last_controller_state.pt', args.run / 'config.json']}}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    console.print(table)
    report_saved_run(output, args)
    console.print(f'Saved evaluation: {output}')


if __name__ == '__main__':
    main()
