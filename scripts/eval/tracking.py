"""Publish completed local evaluations to W&B without rerunning inference."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from scripts.training.tracking import scalar_metrics


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'),
                        default=os.environ.get('WANDB_MODE', 'disabled'))
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'loopformer'))
    parser.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY'))


def report_saved_run(output: Path, args: argparse.Namespace) -> None:
    """Upload summaries, diagnostic tables and CSV artifacts, never model binaries.

    Runs are retrospective: SDK system statistics are disabled because upload-time
    utilization is not inference utilization. Inference timings come from artifacts.
    """
    if args.wandb_mode == 'disabled':
        return
    import wandb

    summary_path = output / 'summary.json'
    if not summary_path.exists():
        summary_path = output / 'comparison.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('status', 'complete') != 'complete':
        raise ValueError('Only completed evaluations can be published')
    csv_paths = sorted(output.glob('*.csv'))
    files = [summary_path, *csv_paths]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    metadata_keys = ('checkpoint', 'model', 'data', 'data_sha256', 'local_checkpoint_sha256',
                     'source_sha256', 'git_head', 'command', 'device', 'dtype', 'batch_size',
                     'loops', 'stop_policy', 'stop_threshold', 'target_semantics', 'stopping_semantics')
    config = {k: summary[k] for k in metadata_keys if k in summary}
    config.update(artifact_sha256=hashes, timing_scope='Retrospective upload; inference timing is from saved summaries')
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=output.name,
                    job_type='evaluation', mode=args.wandb_mode, dir=str(output.resolve()), config=config,
                    settings=wandb.Settings(quiet=True, console='off', disable_code=True, x_disable_stats=True)) as run:
        excluded = set(metadata_keys) | {'packages', 'python', 'schema_version', 'git_status'}
        run.summary.update(scalar_metrics({k: v for k, v in summary.items() if k not in excluded}, 'eval'))
        artifact = wandb.Artifact(f'evaluation-{run.id}', type='evaluation')
        for path in files:
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        for path in csv_paths:
            # Full CSVs live in the artifact. Bound interactive tables explicitly.
            if path.name in ('trajectories.csv', 'predictions.csv'):
                continue
            with path.open(newline='') as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                columns = reader.fieldnames or []
            def cell(value: str):
                if value == '':
                    return None
                if value in ('True', 'False'):
                    return value == 'True'
                try:
                    return float(value)
                except ValueError:
                    return value
            run.log({f'tables/{path.stem}': wandb.Table(columns=columns,
                     data=[[cell(row[c]) for c in columns] for row in rows[:5000]])})
            run.summary[f'tables/{path.stem}/total_rows'] = len(rows)
            run.summary[f'tables/{path.stem}/shown_rows'] = min(5000, len(rows))
            if path.name == 'depth_comparison.csv':
                labels = list(dict.fromkeys(r['checkpoint'] for r in rows))
                run.log({'plots/trajectory_by_depth': wandb.plot.line_series(
                    xs=[[int(r['task_depth']) for r in rows if r['checkpoint'] == label] for label in labels],
                    ys=[[float(r['trajectory_accuracy']) for r in rows if r['checkpoint'] == label] for label in labels],
                    keys=labels, title='Complete trajectory by requested depth', xname='Requested depth')})
            if path.name == 'conditional_transitions.csv':
                for depth in sorted({int(r['task_depth']) for r in rows}):
                    eligible = [r for r in rows if int(r['task_depth']) == depth and r['conditional_transition_accuracy'] != '']
                    labels = list(dict.fromkeys(r['checkpoint'] for r in eligible))
                    run.log({f'plots/conditional_transition_depth_{depth}': wandb.plot.line_series(
                        xs=[[int(r['loop']) for r in eligible if r['checkpoint'] == label] for label in labels],
                        ys=[[float(r['conditional_transition_accuracy']) for r in eligible if r['checkpoint'] == label] for label in labels],
                        keys=labels, title=f'Correct next transition given correct prefix: depth {depth}', xname='Loop')})
        (output / 'wandb_eval_run.json').write_text(json.dumps(
            {'id': run.id, 'url': run.url, 'project': run.project, 'entity': run.entity,
             'mode': args.wandb_mode, 'sdk_version': wandb.__version__, 'artifact_sha256': hashes}, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--recursive', action='store_true', help='Publish nested completed evaluations as separate runs')
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if args.wandb_mode == 'disabled':
        parser.error('Choose --wandb-mode online or offline to publish saved results')
    roots = [args.results]
    if args.recursive:
        roots = sorted({p.parent for name in ('summary.json', 'comparison.json') for p in args.results.rglob(name)
                        if 'wandb' not in p.relative_to(args.results).parts})
    if not roots:
        parser.error('No saved evaluations found')
    for root in roots:
        report_saved_run(root, args)


if __name__ == '__main__':
    main()
