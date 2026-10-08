"""Repeat the frozen affine-controller recipe and enforce declared development gates."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from rich.console import Console

from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_task import sha256_file
from scripts.eval.tracking import add_tracking_arguments, report_saved_run
from scripts.training.controller import ControllerConfig


def acceptance(results: dict[int, dict]) -> dict:
    """Exact timing only; letters at cyclic wrong times cannot satisfy this gate."""
    rows, native = [], 0
    for seed, result in results.items():
        if result['status'] != 'complete' or result['threshold'] != .5:
            raise ValueError('Need completed fixed-threshold results')
        checks = result['native_checks']
        if len(checks) != 2 or not all(c['passed'] for c in checks) or sum(c['questions'] for c in checks) != 44:
            raise ValueError('Missing required native panel')
        native += sum(c['questions'] for c in checks)
        counts = set()
        for split, metrics in result['metrics']['best'].items():
            for cohort, values in metrics.items():
                if cohort.startswith('depth_'):
                    count = int(cohort.removeprefix('depth_')); counts.add(count)
                    rows.append({'optimizer_seed': seed, 'split': split, 'requested_count': count, **values})
        if counts != set(range(1, 65)):
            raise ValueError('Need every requested count 1..64')
    if set(results) != {83, 89, 97}:
        raise ValueError('Need all three predeclared optimizer seeds')
    minimum = min(r['exact_stop'] for r in rows)
    exact = sum(r['exact_stop'] * r['questions'] for r in rows) / sum(r['questions'] for r in rows)
    per_seed = {str(seed): sum(r['exact_stop'] * r['questions'] for r in rows if r['optimizer_seed'] == seed)
                / sum(r['questions'] for r in rows if r['optimizer_seed'] == seed) for seed in results}
    return {'passed': min(per_seed.values()) >= .95 and minimum >= .90, 'overall_exact_stop': exact,
            'per_seed_exact_stop': per_seed,
            'minimum_count_exact_stop': minimum, 'native_matched_questions': native,
            'thresholds': {'overall': .95, 'every_count': .90}, 'per_count': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('models/stage1_pointer/controller-affine-seed83'))
    parser.add_argument('--pilot-evaluation', type=Path, required=True)
    parser.add_argument('--deep-features-cache', type=Path, default=Path('models/stage1_pointer/controller-affine-features-deep.pt'))
    parser.add_argument('--output', type=Path, default=Path('eval/pointer_diagnostics/controller-affine-repeats'))
    parser.add_argument('--dry-run', action='store_true')
    add_tracking_arguments(parser)
    args = parser.parse_args()
    config = ControllerConfig(**json.loads((args.pilot / 'config.json').read_text())); config.validate()
    if config.controller_kind != 'affine_suffix' or config.optimizer_seed != 83:
        raise ValueError('Need the completed seed-83 affine pilot')
    metadata = json.loads((args.pilot / 'run.json').read_text())
    pilot_result = json.loads((args.pilot_evaluation / 'summary.json').read_text())
    if (metadata['status'] != 'complete' or pilot_result['run'] != str(args.pilot) or
            sha256_file(Path(metadata['features_cache'])) != metadata['features_sha256']):
        raise ValueError('Pilot/cache provenance differs')
    console = Console(); console.print('Frozen recipe · retain seed 83 · repeat optimizer seeds 89 and 97')
    if args.dry_run:
        console.print('Preview only: no training, model loading or writes.'); return
    args.output.mkdir(parents=True, exist_ok=False)
    results = {83: pilot_result}
    entries = [{'optimizer_seed': 83, 'run': str(args.pilot), 'evaluation': str(args.pilot_evaluation)}]
    for seed in (89, 97):
        run = args.pilot.with_name(f'controller-affine-seed{seed}')
        config_path = args.output / f'seed-{seed}.json'
        config_path.write_text(json.dumps(replace(config, optimizer_seed=seed).to_dict(), indent=2) + '\n')
        subprocess.run([sys.executable, '-u', '-m', 'scripts.training.train_controller', '--config', str(config_path),
                        '--features-cache', metadata['features_cache'], '--output', str(run)], check=True)
        evaluation = args.output / f'seed-{seed}-evaluation'
        subprocess.run([sys.executable, '-u', '-m', 'scripts.eval.controller_candidate', '--run', str(run),
            '--deep-features-cache', str(args.deep_features_cache), '--output', str(evaluation), '--native-check'], check=True)
        results[seed] = json.loads((evaluation / 'summary.json').read_text())
        entries.append({'optimizer_seed': seed, 'run': str(run), 'evaluation': str(evaluation)})
    gate = acceptance(results)
    write_csv(args.output / 'per_count.csv', gate.pop('per_count'))
    summary = {'status': 'complete', 'scope': 'development controller repair through count 64; same data, three optimizer orders',
               'acceptance': gate, 'runs': entries, 'features_sha256': metadata['features_sha256'],
               'source_checkpoint_sha256': metadata['source_checkpoint_sha256'],
               'deep_features_sha256': sha256_file(args.deep_features_cache),
               'reserved_confirmation_read': False, 'metrics': {str(k): v['metrics'] for k, v in results.items()}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report_saved_run(args.output, args)
    console.print('[green]Development gates passed[/green]' if gate['passed'] else '[red]Development gates failed[/red]')
    if not gate['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
