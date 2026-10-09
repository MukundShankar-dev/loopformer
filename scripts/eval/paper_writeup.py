"""Writeup figures from original saved training and baseline evidence.

No weights are loaded. Training curves retain their original development
populations; they are never relabeled as the new paired benchmark. The final
number-reader/cell fits have scalar summaries, not invented epoch histories.
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import numpy as np

from scripts.eval.benchmark_metrics import wilson_interval
from scripts.eval.pointer_task import sha256_file


RUNS = (
    ('attempt_01', 'Attempt 1 · full-sequence intermediate supervision',
     [('ce_full', 'depth6-fresh30k-seed37-batch4', 2500)]),
    ('attempt_02', 'Attempt 2 · completion head and fixed prompt memory',
     [('joint_full', 'depth6-completion-seed37', 3250),
      ('fixed6', 'depth6-fixed-prompt-seed37', 3250)]),
    ('attempt_03', 'Attempt 3 · deeper training with the same architecture',
     [('fixed12', 'depth12-fixed-prompt-seed47-gaps', 5000)]),
    ('attempt_04', 'Attempt 4 · count-free executor and private controller',
     [('gru', 'executor_r-seed61', 2250)]),
)
BASELINE = Path('eval/pointer_task/20260910-wsl-cuda-instruct-retry')


def baseline_counts(directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Recount every saved prediction and require agreement with the run summary."""
    summary = json.loads((directory / 'summary.json').read_text())
    with (directory / 'predictions.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate ordinary baseline questions')
    depths = np.array(sorted({int(r['task_depth']) for r in rows}))
    totals, correct = [], []
    for n in depths:
        subset = [r for r in rows if int(r['task_depth']) == n]
        for r in subset:
            scored = r['valid_answer'] == 'True' and r['prediction'] == r['target']
            if scored != (r['correct'] == 'True'):
                raise ValueError('Baseline saved correctness differs from parsed answer')
        total, hits = len(subset), sum(r['correct'] == 'True' for r in subset)
        if summary['by_depth'][str(n)]['total'] != total or summary['by_depth'][str(n)]['correct'] != hits:
            raise ValueError('Baseline depth summary differs from individual predictions')
        totals.append(total); correct.append(hits)
    if sum(totals) != summary['total'] or sum(correct) != summary['correct']:
        raise ValueError('Baseline overall summary differs')
    return depths, np.array([correct, totals]), summary


def weighted_depth_metric(validation: dict, depths: list[int], metric: str) -> float:
    """Graph-weighted accuracy over an explicitly named requested-depth cohort."""
    selected = [validation['by_depth'][str(n)] for n in depths]
    denominator = sum(r['examples'] for r in selected)
    if not denominator:
        raise ValueError('Empty development cohort')
    return sum(r[metric] * r['examples'] for r in selected) / denominator


def binned_training(events: list[dict], width: int = 50) -> np.ndarray:
    """Nonoverlapping optimizer-update windows, weighted by minibatch examples."""
    groups: dict[int, list] = {}
    for row in events:
        if row['event'] == 'train':
            groups.setdefault((row['step'] - 1) // width, []).append(row)
    return np.array([[max(r['step'] for r in rows),
                      sum(r['train']['loss'] * r['train']['examples'] for r in rows) /
                      sum(r['train']['examples'] for r in rows)] for rows in groups.values()])


def population_panel(ax, arrays, names: list[str]) -> None:
    from scripts.eval.plot_paper_analysis import NAMES, COLORS
    x = np.arange(1, 257)
    groups = []
    for name in names:
        for metric, style, label in (
            ('nominal_final_correct', '-', 'Forced final letter'),
            ('complete_trajectory', '--', 'Every intermediate letter'),
            ('joint_success', ':', 'Answer + exact first stop')):
            key = f'{name}__{metric}'
            if key in arrays:
                y = 100 * arrays[key].mean(0)
                matching = next((g for g in groups if np.array_equal(g[1], y)), None)
                if matching is None:
                    groups.append([name, y, style, [label]])
                else:
                    matching[3].append(label)
    for i, (name, y, style, labels) in enumerate(groups):
        label = ' / '.join(labels) + (' (identical)' if len(labels) > 1 else '')
        ax.plot(x, y, linestyle=style, color=COLORS[name], marker=['o', 's', '^'][i % 3],
                markevery=(i*5, 37), markersize=3, label=textwrap.fill(label, 36), linewidth=1.6)
    ax.set(xlabel='Requested N · new paired population', ylabel='Graphs correct (%)',
           xlim=(1, 256), ylim=(-2, 102), title='Frozen selected-checkpoint evaluation')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=.2)


def writeup_figures(figures, arrays=None) -> None:
    """Six article figures; optional third column uses complete audited population."""
    import matplotlib.pyplot as plt
    from scripts.eval.plot_paper_analysis import NAMES

    depths, values, summary = baseline_counts(BASELINE)
    correct, total = values; rate = correct / total
    intervals = np.array([wilson_interval(int(k), int(n)) for k, n in zip(correct, total)])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(depths, 100 * rate, yerr=100 * np.array([rate-intervals[:, 0], intervals[:, 1]-rate]),
                fmt='o-', color='#626e7c', capsize=4, label='Qwen2.5-0.5B-Instruct · 3-shot')
    for n, k, count, y in zip(depths, correct, total, rate):
        ax.annotate(f'{k}/{count}', (n, y*100), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
    ax.set(xlabel='Requested transitions N', ylabel='Final-letter accuracy (%)',
           xticks=depths, ylim=(0, max(30, intervals[:, 1].max()*100+6)))
    ax.grid(axis='y', alpha=.2); ax.legend()
    figures.save(fig, '06_writeup', 'baseline', 'Ordinary base checkpoint · 6% final-answer accuracy',
        'Original seed-17 test: 1,000 questions, 125 per depth, greedy 3-shot generation; 95% Wilson intervals.\n'
        'Historical context: this is a different dataset/prompt/inference procedure from the recurrent population comparison.',
        {'depths': depths, 'correct': correct, 'total': total, 'wilson': intervals},
        source_paths=[BASELINE/'predictions.csv', BASELINE/'summary.json'])

    for filename, title, runs in RUNS:
        columns = 3 if arrays is not None else 2
        fig, axes = plt.subplots(len(runs), columns, figsize=(6*columns, 5*len(runs)), squeeze=False)
        numeric, sources = {}, []
        for index, (name, run, selected) in enumerate(runs):
            root = Path('models/stage1_pointer') / run
            events = [json.loads(line) for line in (root/'metrics.jsonl').open()]
            config = json.loads((root/'config.json').read_text())
            trained = config.get('train_depths', list(range(1, config['train_max_depth']+1)))
            validation = [r for r in events if r['event'] == 'validation']
            steps = np.array([r['step'] for r in validation])
            loss = binned_training(events)
            axes[index, 0].plot(loss[:, 0], loss[:, 1], color='#2467af', label='Training symbol CE · 50-update windows')
            axes[index, 0].set(xlabel='Optimizer update', ylabel='Mean symbol cross-entropy', title=NAMES[name])
            cohorts = [('Trained requested counts', trained, '#298476', 'o', '-')]
            other = sorted(set(map(int, validation[0]['validation']['by_depth'])) - set(trained))
            if other:
                cohorts.append(('Untrained requested counts ' + ','.join(map(str, other)), other, '#ce483c', 's', '--'))
            for label, cohort, color, marker, style in cohorts:
                quality = np.array([weighted_depth_metric(r['validation'], cohort, 'trajectory_accuracy') for r in validation])
                sizes = np.array([sum(r['validation']['by_depth'][str(n)]['examples'] for n in cohort) for r in validation])
                if not (sizes == sizes[0]).all():
                    raise ValueError('Original monitoring cohort changed size; do not render as fixed panel')
                axes[index, 1].plot(steps, 100*quality, color=color, marker=marker, linestyle=style, label=f'{label} (n={sizes[0]})')
                numeric[f'{name}__{label}'] = quality
                numeric[f'{name}__{label}__questions'] = sizes
            axes[index, 1].set(xlabel='Optimizer update', ylabel='Strict development trajectory (%)', ylim=(-2, 102), title='Original development monitoring')
            for ax in axes[index, :2]:
                ax.axvline(selected, color='#45505c', linestyle=':', label=f'Checkpoint used: {selected}')
                ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
            if arrays is not None:
                population_panel(axes[index, 2], arrays, [name])
                for metric in ('nominal_final_correct', 'complete_trajectory', 'joint_success'):
                    key = f'{name}__{metric}'
                    if key in arrays: numeric[key] = arrays[key].mean(0)
            numeric[f'{name}__steps'] = steps; numeric[f'{name}__training_ce'] = loss
            sources.extend([root/'metrics.jsonl', root/'config.json'])
        caption = ('Historical training: CE windows weight minibatch examples; validation cohorts weight questions.\n'
                   'Each run retains its original development data. Requested-count holdouts can still be supervised intermediate loops.')
        if filename == 'attempt_04':
            caption = ('Successful R training shown here; later frozen-R GRU fitting is in controller_development.\n'
                       'Population evaluation uses the later GRU controller; the executor training controller was replaced.')
        figures.save(fig, '06_writeup', filename, title, caption, numeric, source_paths=sources)

    # The timer repair uses a controller pilot plus predetermined reader/cell fits.
    root = Path('models/stage1_pointer/controller-affine-seed83')
    with (root/'validation_history.csv').open() as handle: history = list(csv.DictReader(handle))
    reader_path = Path('models/stage1_pointer/controller-shared-number-seed83/summary.json')
    cell_path = Path('models/stage1_pointer/controller-shared-number-precision-seed83/summary.json')
    reader, cell = json.loads(reader_path.read_text()), json.loads(cell_path.read_text())
    fig, axes = plt.subplots(1, 3 if arrays is not None else 2, figsize=(18 if arrays is not None else 12, 5.5))
    numeric = {}
    for cohort, color, style in [('trained', '#2467af', '-'), ('unseen_requested_count', '#ce483c', '--')]:
        rows = sorted([r for r in history if r['cohort'] == cohort], key=lambda r: int(r['step']))
        if not rows: raise ValueError(f'Missing scalar-controller development cohort: {cohort}')
        x = np.array([int(r['step']) for r in rows]); y = np.array([float(r['exact_stop']) for r in rows])
        axes[0].plot(x, y*100, linestyle=style, marker='o' if cohort=='trained' else 's',
                     markevery=(0 if cohort=='trained' else 2, 5), markersize=3,
                     label=cohort.replace('_', ' '), color=color)
        numeric[cohort] = np.array([x, y])
    axes[0].set(xlabel='Scalar-controller pilot optimizer update', ylabel='Exact first stop (%)', ylim=(-2, 102), title='Original scalar-controller development')
    axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=.2)
    fit = cell['fit']; mse = np.array([fit['before_mse'], fit['fitted_mse'], fit['deployed_mse']])
    axes[1].bar(['Before cell fit', 'Fitted FP64', 'Deployed FP32'], mse, color=['#ce6a38', '#298476', '#2467af'])
    axes[1].set(yscale='log', ylabel='Training-label free-running countdown MSE', title='Predetermined cell refinement')
    for i, value in enumerate(mse): axes[1].text(i, value*1.2, f'{value:.2e}', ha='center', fontsize=8)
    axes[1].set_ylim(mse.min()/3, mse.max()*6)
    numeric['cell_training_mse'] = mse
    if arrays is not None:
        population_panel(axes[2], arrays, ['final'])
        for metric in ('nominal_final_correct', 'complete_trajectory', 'joint_success'):
            numeric[f'final__{metric}'] = arrays[f'final__{metric}'].mean(0)
    figures.save(fig, '06_writeup', 'attempt_05', 'Attempt 5 · shared number reader and precise learned countdown',
        'Left: earlier scalar-controller pilot. Middle: later separate fit on the same 58 requested counts and at most 12 labels per count.\n'
        f"Reader was fitted by least squares (maximum training-count error {reader['fit']['training_max_absolute_error']:.2g}); no reader/cell epoch history exists.",
        numeric, source_paths=[root/'validation_history.csv', reader_path, cell_path])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('eval/pointer_analysis/writeup-history-20261009'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    from scripts.eval.plot_paper_analysis import Figures
    figures = Figures(args.output); writeup_figures(figures)
    (figures.output/'manifest.json').write_text(json.dumps({'status': 'historical evidence only', 'figures': figures.entries}, indent=2)+'\n')
    lines = ['# Saved writeup histories', '', 'Actual original baseline/training evidence. Full-population evaluation panels will be included in the final paper bundle after its audit.', '']
    for entry in figures.entries:
        lines.extend([f"## {entry['title']}", '', entry['caption'], '', f"![{entry['title']}]({entry['group']}/{entry['name']}.png)", ''])
    (figures.output/'index.md').write_text('\n'.join(lines))
    print(f'Wrote {len(figures.entries)} saved-history figures; no model inference.')


if __name__ == '__main__':
    main()
