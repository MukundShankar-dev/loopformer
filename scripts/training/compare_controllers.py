"""Run matched controller-only arms on one immutable cache; no best-seed selection."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from rich.console import Console

from scripts.training.controller import ControllerConfig


def comparison_runs(spec: dict) -> list[tuple[str, ControllerConfig]]:
    """Keep graph sampling/initial controller fixed; vary only optimizer order and loss."""
    base = ControllerConfig(**json.loads(Path(spec['training_config']).read_text()))
    seeds = spec['optimizer_seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Need distinct optimizer seeds')
    weight = spec['remaining_loss_weight']
    if weight <= 0:
        raise ValueError('Auxiliary arm needs a positive loss weight')
    runs = []
    for seed in seeds:
        for arm, coefficient in [('stop_only', 0.0), ('remaining', weight)]:
            config = replace(base, optimizer_seed=seed, remaining_readout=True,
                             remaining_loss_weight=coefficient, remaining_scale=spec['remaining_scale'])
            config.validate()
            runs.append((f'seed-{seed}-{arm}', config))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/controller_remaining.json'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    runs = comparison_runs(spec)
    cache, output = Path(spec['features_cache']), Path(spec['output'])
    if args.smoke_test:
        output = output.with_name(output.name + f'-smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}')
        runs = [(name, replace(config, steps=10, warmup_steps=1, eval_every=5, log_every=1)) for name, config in runs[:2]]
    console = Console()
    console.print('[bold]Matched controller comparison[/bold] · frozen executor · shared cache')
    console.print(f'Cache: {cache}\nOutput: {output}')
    for name, config in runs:
        console.print(f'{name}: {config.steps} updates · weight {config.remaining_loss_weight:g} · scale {config.remaining_scale:g}')
    if args.dry_run:
        console.print('Preview only: no model/cache loading, updates or writes.')
        return
    if not cache.is_file():
        raise ValueError(f'Need the original training cache on this device: {cache}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'configs').mkdir()
    manifest = {'status': 'running', 'comparison': spec, 'features_cache': str(cache), 'runs': [],
                'smoke_test': args.smoke_test, 'selection': 'within-run trained-count exact stop then BCE; all seeds reported'}
    for name, config in runs:
        path = output / 'configs' / f'{name}.json'
        path.write_text(json.dumps(config.to_dict(), indent=2) + '\n')
        run = output / name
        manifest['runs'].append({'name': name, 'path': str(run), 'optimizer_seed': config.optimizer_seed,
                                 'arm': 'remaining' if config.remaining_loss_weight else 'stop_only'})
        (output / 'comparison.json').write_text(json.dumps(manifest, indent=2) + '\n')
        console.print(f'[bold cyan]{name}[/bold cyan]')
        subprocess.run([sys.executable, '-u', '-m', 'scripts.training.train_controller',
                        '--config', str(path), '--features-cache', str(cache), '--output', str(run)], check=True)
    # The shared evaluator reuses cached replay/first-crossing semantics. A full
    # run extracts deep observations once; smoke only audits cached validation.
    command = [sys.executable, '-u', '-m', 'scripts.eval.controller_comparison', '--run', str(output)]
    if args.smoke_test:
        command.append('--validation-only')
    subprocess.run(command, check=True)
    manifest['status'] = 'complete'
    (output / 'comparison.json').write_text(json.dumps(manifest, indent=2) + '\n')
    console.print(f'[green]Comparison complete[/green] · {output}')


if __name__ == '__main__':
    main()
