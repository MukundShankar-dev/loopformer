"""Audited trajectory matrices and matched frozen architecture comparison figures."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.dataset.pointer import PointerExample
from scripts.eval.loop_metrics import write_csv
from scripts.eval.pointer_failure_metrics import CATEGORIES, MODES, failure_metrics
from scripts.eval.plot_pointer_benchmark import read_csv, save_figure
from scripts.eval.pointer_task import sha256_file
from scripts.eval.audit_pointer_figures import audit_plot_tables, audit_numeric, audit_failure_exports


def label_figure(fig, title: str, footer: str) -> None:
    fig.suptitle(title, x=.04, y=.985, ha='left', fontsize=16, fontweight='bold')
    fig.text(.04, .02, footer, fontsize=8.5, color='#55616e')
    fig.subplots_adjust(top=.85, bottom=.16, left=.08, right=.97, wspace=.4)


def final_figures(metrics: dict, graph_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    correct = metrics['correct']; maximum = correct.shape[1]
    first = sorted(metrics['first_errors'], key=lambda r: (r['loop'], int(r['graph_index'])))
    indices = [int(r['graph_index']) for r in first]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(correct[indices].astype(int), aspect='auto', interpolation='nearest',
        cmap=ListedColormap(['#bd4d3e', '#dbeee8']), vmin=0, vmax=1, extent=(.5, maximum + .5, len(first) - .5, -.5))
    ax.set_yticks(range(len(first)), [f'{r["graph_index"]} · {r["graph_mode"]}' for r in first], fontsize=7)
    ax.set(xlabel='Recurrent loop', ylabel='All failing graphs, sorted by first error')
    ax.axvline(12, color='#253747', linestyle=':', linewidth=1)
    label_figure(fig, 'Every failing trajectory: errors can persist, alternate, or recover',
        f'Red = wrong C decode; pale green = correct. All {len(first)} failing graphs shown; {len(correct) - len(first):,} fully correct graphs omitted.\n'
        'A recovered letter never repairs the cumulative complete-trajectory score. Dotted line: training unroll 12.')
    fig.subplots_adjust(left=.20, top=.90, bottom=.12)
    save_figure(fig, output, 'failed_trajectory_matrix'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    confusion = metrics['confusion']
    im = axes[0].imshow(confusion, cmap='YlOrRd', vmin=0, vmax=max(1, confusion.max()))
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    axes[0].set_xticks(range(26), letters, fontsize=7); axes[0].set_yticks(range(26), letters, fontsize=7)
    axes[0].set(xlabel='Incorrect predicted letter', ylabel='True next letter', title='First-error confusion · counts')
    for i, j in zip(*np.nonzero(confusion)):
        axes[0].text(j, i, str(confusion[i, j]), ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=axes[0], shrink=.75, label='Independent first failures', ticks=range(int(confusion.max()) + 1))
    targets = metrics['targets']
    # Target frequencies at the maximum horizon; each graph contributes once.
    denominators = np.bincount([ord(t[-1]) - 65 for t in targets], minlength=26)
    errors = np.zeros(26, dtype=int)
    for target, hits in zip(targets, correct, strict=True):
        errors[ord(target[-1]) - 65] += not hits[-1]
    rate = np.divide(errors * 100., denominators, out=np.full(26, np.nan), where=denominators > 0)
    write_csv(output / 'final_error_by_letter.csv', [{'letter': letter, 'graphs': int(denominators[i]),
        'wrong_graphs': int(errors[i]), 'error_rate': float(rate[i]) / 100 if denominators[i] else None}
        for i, letter in enumerate(letters)])
    axes[1].bar(letters, rate, color='#bd4d3e')
    axes[1].set(title=f'Error rate by true letter at loop {maximum}', xlabel='True final letter', ylabel='Graphs wrong (%)')
    axes[1].tick_params(axis='x', labelsize=7); axes[1].grid(axis='y', alpha=.2)
    label_figure(fig, 'Which symbols are confused?',
        f'Left: each of the {len(first)} failing graphs contributes once: its first incorrect intermediate C letter.\n'
        f'Right: per-letter error denominators use all {len(correct):,} graphs at the same horizon.')
    save_figure(fig, output, 'symbol_confusion'); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    matrix = np.array([[r['graphs'] for r in metrics['categories'] if r['graph_mode'] == mode] for mode in MODES])
    axes[0].imshow(matrix, cmap='Blues', vmin=0)
    axes[0].set_xticks(range(4), ['Repeat\nprevious', 'Earlier\nvisited', 'Later\nreachable', 'Outside\nstart orbit'], fontsize=8)
    axes[0].set_yticks(range(3), ['Random function', 'Permutation', 'Full cycle'], fontsize=8)
    for i in range(3):
        for j in range(4):
            axes[0].text(j, i, str(matrix[i, j]), ha='center', va='center')
    axes[0].set(title='First-error type · graph counts')
    first_readouts = metrics['first_readouts']
    labels = ['R and C wrong, same letter', 'R correct / C wrong', 'R and C wrong, different letters']
    bars = axes[1].barh(['Both wrong\nsame decode', 'R correct\nC wrong', 'Both wrong\ndifferent decode'], [first_readouts.get(k, 0) for k in labels], color='#9c6c97')
    axes[1].bar_label(bars, padding=3, fontsize=8)
    axes[1].set_xlim(0, max(first_readouts.values(), default=1) + 4)
    axes[1].tick_params(axis='y', labelsize=8)
    axes[1].set(title='Readouts at the first error', xlabel='Independent failing graphs')
    colors = ['#16826b', '#bd4d3e', '#8a98a5']
    bottom = np.zeros(3)
    for j, kind in enumerate(('recover to reference', 'follow edge from prior decode', 'other wrong transition')):
        rows = [r for r in metrics['dynamics'] if r['category'] == kind]
        proportions = np.array([100 * r['transitions'] / r['prior_wrong_transitions'] if r['prior_wrong_transitions'] else 0 for r in rows])
        axes[2].bar(range(3), proportions, bottom=bottom, color=colors[j], label=kind)
        bottom += proportions
    axes[2].set_xticks(range(3), ['Random\nfunction', 'Permutation', 'Full cycle'])
    axes[2].set(title='After a wrong decoded state', ylabel='Transitions following an error (%)', ylim=(0, 100))
    axes[2].legend(loc='upper center', bbox_to_anchor=(.5, -.19), fontsize=7, frameon=False)
    label_figure(fig, 'Execution failure signatures, without resetting the recurrent state',
        'First-error categories have priority: previous > earlier visited > later reachable > outside orbit.\nDecoded-edge following is an observational readout metric; it does not prove what the hidden state represents.')
    fig.subplots_adjust(bottom=.28, wspace=.55)
    save_figure(fig, output, 'failure_signatures'); plt.close(fig)

    periods = sorted({int(r['cycle_period']) for r in graph_rows})
    rates = []
    sizes = []
    for period in periods:
        idx = [i for i, r in enumerate(graph_rows) if int(r['cycle_period']) == period]
        rates.append(100 * (~correct[idx]).mean(0)); sizes.append(len(idx))
    write_csv(output / 'cycle_period_by_loop.csv', [{'period': period, 'loop': j + 1, 'graphs': n,
        'error_rate': float(rate) / 100} for period, n, values in zip(periods, sizes, rates, strict=True)
        for j, rate in enumerate(values)])
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(rates, aspect='auto', interpolation='nearest', cmap='YlOrRd', vmin=0,
                   extent=(.5, maximum + .5, len(periods) - .5, -.5))
    ax.set_yticks(range(len(periods)), [f'{p} (n={n})' for p, n in zip(periods, sizes, strict=True)], fontsize=8)
    ax.set(xlabel='Recurrent loop', ylabel='Reachable cycle period · graph count')
    fig.colorbar(im, ax=ax, label='Wrong decoded states (%)')
    label_figure(fig, 'Does failure depend on cycle length?',
        'Each cell averages graphs of the same reachable period. Small strata are noisy; horizon cells are dependent.\nAll graph types pooled: structure is descriptive, not a controlled causal manipulation.')
    save_figure(fig, output, 'cycle_period_matrix'); plt.close(fig)


def comparison_figures(path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    summary = json.loads((path / 'summary.json').read_text())
    audit = json.loads((path / 'comparison_reference_audit.json').read_text())
    if summary['status'] != 'complete' or not summary['inference_files_unchanged']:
        raise ValueError('Only completed frozen comparisons may be plotted')
    if not audit['passed'] or any(sha256_file(path / name) != value for name, value in audit['inputs_sha256'].items()):
        raise ValueError('Need a current independent comparison audit')
    arms = summary['config']['models']; counts = summary['config']['counts']
    rows = read_csv(path / 'decisions.csv')
    for arm in arms:
        if not summary['models'][arm['name']]['native_fidelity']['passed']:
            raise ValueError('Unverified native/replay comparison')
    write_csv(output / 'architecture_by_count.csv', [{'model': arm['name'], 'depth': n,
        'graphs': summary['graphs'], **{metric: sum(r[metric] == 'True' for r in rows
            if r['model'] == arm['name'] and int(r['depth']) == n) / summary['graphs']
            for metric in ('nominal_final_correct', 'complete_trajectory', 'exact_stop', 'joint_success')}}
        for arm in arms for n in counts])
    footer = 'Paired diagnostic: same 27 graphs × 30 counts per model; opened panel, not independent confirmation.\nLegacy LoRA differs in data/recipe/capacity. Other four arms have bitwise identical non-controller tensors.'
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    metrics = ('nominal_final_correct', 'complete_trajectory', 'exact_stop', 'joint_success')
    for ax, metric, title in zip(axes, metrics, ('Forced final answer', 'Forced: all nominal intermediate steps', 'Exact stopping loop', 'Actual answer + exact stopping loop'), strict=True):
        rates = np.array([[100 * sum(r[metric] == 'True' for r in rows if r['model'] == arm['name'] and int(r['depth']) == count) / summary['graphs']
                           for count in counts] for arm in arms])
        im = ax.imshow(rates, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        ax.set_yticks(range(len(arms)), [a['label'] for a in arms], fontsize=8)
        ax.set_title(title, fontsize=11, loc='left')
        for i in range(len(arms)):
            for j in range(len(counts)):
                ax.text(j, i, f'{rates[i, j]:.0f}', ha='center', va='center', fontsize=6.5,
                        color='white' if rates[i, j] < 15 or rates[i, j] > 85 else '#253747')
    axes[-1].set_xticks(range(len(counts)), counts, fontsize=8)
    axes[-1].set_xlabel('Requested count · columns are the declared discrete counts, not a linear axis')
    label_figure(fig, 'Separate execution failure from stopping failure across architectures', footer)
    fig.subplots_adjust(left=.21, top=.90, bottom=.11, hspace=.45, right=.91)
    fig.colorbar(im, ax=axes.tolist(), fraction=.025, pad=.02, label='Correct (%)')
    save_figure(fig, output, 'architecture_metric_matrices'); plt.close(fig)

    fig, axes = plt.subplots(1, len(arms), figsize=(17, 7), sharey=True)
    cmap = plt.colormaps['Blues'].copy(); cmap.set_bad('#fafbfc')
    for ax, arm in zip(axes, arms, strict=True):
        matrix = np.zeros((summary['config']['safety_cap'] + 1, len(counts)))
        for r in rows:
            if r['model'] == arm['name']:
                stop = int(r['first_stop']) if r['first_stop'] else 0
                matrix[stop, counts.index(int(r['depth']))] += 1 / summary['graphs']
        im = ax.imshow(np.ma.masked_where(matrix == 0, matrix), aspect='auto', origin='lower',
                       cmap=cmap, norm=LogNorm(vmin=1 / summary['graphs'], vmax=1), interpolation='nearest')
        ax.plot(range(len(counts)), counts, linestyle='none', marker='.', markersize=3, color='#c44132', label='Desired stop')
        ax.set_title(arm['label'].replace(' + ', '\n+ '), fontsize=10)
        ticks = [0, 7, 11, 17, 20, 23, 27, 29]
        ax.set_xticks(ticks, [counts[i] for i in ticks], fontsize=8, rotation=45)
        ax.set_xlabel('Requested count\n(discrete columns)')
    axes[0].set_yticks([0, 12, 64, 128, 192, 256], ['No stop', '12', '64', '128', '192', '256'])
    axes[0].set_ylabel('Actual first stopping loop')
    label_figure(fig, 'When do earlier models actually stop?', footer + '\nBlue = graph fraction on a log color scale; blank = zero; red = desired stop. No-stop row is separate from cap successes.')
    fig.subplots_adjust(top=.81, bottom=.21, right=.91, wspace=.15)
    bar = fig.colorbar(im, ax=axes.tolist(), fraction=.02, pad=.015, label='Fraction of paired graphs · log scale',
                      ticks=[1 / summary['graphs'], .1, .5, 1])
    bar.ax.set_yticklabels([f'1/{summary["graphs"]}', '10%', '50%', '100%'])
    save_figure(fig, output, 'requested_actual_stop_matrices'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for arm in arms:
        selected = [r for r in rows if r['model'] == arm['name']]
        cohorts = [(1, 12), (13, 69), (70, 256)]
        wrongtime = [100 * sum(r['correct_letter_wrong_time'] == 'True' for r in selected if lo <= int(r['depth']) <= hi) /
            sum(lo <= int(r['depth']) <= hi for r in selected) for lo, hi in cohorts]
        axes[0].plot(range(3), wrongtime, marker='o', label=arm['label'])
        late = [100 * sum(r['early_stop'] == 'True' for r in selected if int(r['depth']) == n) / summary['graphs'] for n in counts]
        # Sparse requests are measured points, not observations of intervening counts.
        axes[1].scatter(counts, late, s=14, label=arm['label'])
    axes[0].set_xticks(range(3), ['Counts 1–12', '13–69', '70–256'])
    axes[0].set(title='Correct letter at the wrong time', ylabel='Paired queries (%)', ylim=(0, 102))
    axes[1].set(title='Premature stopping by requested count', xlabel='Requested count', ylabel='Graphs stopping early (%)', ylim=(0, 102))
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    label_figure(fig, 'Final-letter scoring alone conceals control failures', footer + '\nCohorts contain only sampled counts; right panel shows measured points without interpolation.')
    save_figure(fig, output, 'cyclic_coincidence_and_early_stops'); plt.close(fig)


def numeric_figure(paths: list[Path], output: Path, comparison: Path | None) -> dict:
    """Reuse saved controller-only evidence; no claim about full pointer execution."""
    import matplotlib.pyplot as plt
    freeze = json.loads((comparison / 'freeze.json').read_text()) if comparison else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    hashes = {}
    for path in paths:
        audit_numeric(path)
        summary = json.loads((path / 'summary.json').read_text())
        if (summary['status'] != 'complete' or not summary['inference_files_unchanged']
                or summary['executor_executed']):
            raise ValueError('Need frozen controller-only numeric results')
        if freeze and summary['inference_sha256'] not in freeze['inference_sha256'].values():
            raise ValueError('Numeric result belongs to another checkpoint')
        rows = read_csv(path / 'countdown.csv')
        counts = np.array([int(r['requested']) for r in rows])
        exact = np.array([r['exact_stop'] == 'True' for r in rows])
        if (len(rows) != summary['rollout_max'] or counts.tolist() != list(range(1, len(rows) + 1))
                or int(exact.sum()) != summary['exact_stop_questions']):
            raise ValueError('Numeric summary/countdown coverage mismatch')
        final = 'precision' in summary['checkpoint']
        label = 'Final precision repair' if final else 'Shared reader before precision'
        color = '#16826b' if final else '#bd4d3e'
        axes[0].plot(counts, 100 * exact.cumsum() / counts, color=color, label=label)
        residual = np.array([float(r['first_stop']) - int(r['requested']) if r['first_stop'] else np.nan for r in rows])
        axes[1].plot(counts, residual, color=color, label=label)
        for name in ('summary.json', 'countdown.csv'):
            hashes[str(path / name)] = sha256_file(path / name)
    axes[0].set(title='Aggregate over ALL integer requests 1…N', xlabel='Largest included requested count N',
                ylabel='Exact stops among requests 1…N (%)', ylim=(0, 102), yticks=[0, 20, 40, 60, 80, 100])
    axes[1].set(title='Timing drift at each integer request', xlabel='Requested count', ylabel='First stopping loop − requested count')
    for ax in axes:
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)
        ax.axvline(256, color='#8a98a5', linestyle=':', linewidth=1)
    label_figure(fig, 'The precision repair matters beyond the pointer benchmark horizon',
        'Saved numeric-only diagnostics: earlier model tested through 4,096; final through 8,192. No R execution here.\nDotted line: whole-model pointer benchmark limit 256. No pointer-quality claim at the larger counts.')
    save_figure(fig, output, 'numeric_precision_comparison'); plt.close(fig)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('eval/pointer_benchmark/final-full-20261008/independent'))
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--numeric-results', type=Path, nargs='+', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.results / 'summary.json').read_text())
    audit = json.loads((args.results / 'result_reference_audit.json').read_text())
    manifest = json.loads((args.results / 'dataset_manifest.json').read_text())
    if (summary['status'] != 'complete' or not summary['native_fidelity']['passed']
            or not summary['inference_files_unchanged'] or not audit['passed']
            or audit['summary_sha256'] != sha256_file(args.results / 'summary.json')
            or sha256_file(args.graphs) != manifest['graphs_sha256']):
        raise ValueError('Need current audited frozen benchmark and exact raw graph file')
    tasks = [PointerExample(**json.loads(line)) for line in args.graphs.read_text().splitlines()]
    plot_audit = {'benchmark': audit_plot_tables(args.results, args.graphs)}
    if args.comparison:
        from scripts.eval.audit_checkpoint_comparison import audit_comparison
        plot_audit['comparison'] = audit_comparison(args.comparison, args.graphs)
    plot_audit['numeric'] = {str(path): audit_numeric(path) for path in args.numeric_results}
    rows = read_csv(args.results / 'graphs.csv')
    metrics = failure_metrics(tasks, rows, summary['config']['max_depth'])
    if (int(metrics['correct'].sum()) != audit['success_counts']['nominal_final_correct']
            or int(np.logical_and.accumulate(metrics['correct'], axis=1).sum()) != audit['success_counts']['complete_trajectory']):
        raise ValueError('Independent trajectory totals differ from the source reference audit')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('first_errors', 'categories', 'dynamics', 'error_episodes'):
        write_csv(args.output / f'{name}.csv', metrics[name])
    write_csv(args.output / 'first_error_confusion.csv', [{'target': chr(i + 65), **{chr(j + 65): int(n) for j, n in enumerate(row)}}
              for i, row in enumerate(metrics['confusion'])])
    (args.output / 'summary.json').write_text(json.dumps(metrics['summary'], indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                        'axes.spines.right': False, 'svg.fonttype': 'none',
                        'svg.hashsalt': 'loopformer-pointer-figures'})
    final_figures(metrics, rows, args.output)
    plot_audit['failure_exports'] = audit_failure_exports(args.results, args.graphs, args.output)
    if args.comparison:
        comparison_figures(args.comparison, args.output)
    inputs = {'graphs': sha256_file(args.graphs), 'source_summary': sha256_file(args.results / 'summary.json'),
              'source_trajectories': sha256_file(args.results / 'graphs.csv')}
    if args.comparison:
        inputs['comparison_decisions'] = sha256_file(args.comparison / 'decisions.csv')
        inputs['comparison_summary'] = sha256_file(args.comparison / 'summary.json')
    if args.numeric_results:
        inputs.update(numeric_figure(args.numeric_results, args.output, args.comparison))
    (args.output / 'plot_input_audit.json').write_text(json.dumps(plot_audit, indent=2) + '\n')
    provenance = {'matplotlib': matplotlib.__version__, 'inputs_sha256': inputs,
        'sources_sha256': {name: sha256_file(Path(__file__).with_name(name)) for name in
                           ('plot_pointer_failures.py', 'pointer_failure_metrics.py', 'plot_pointer_benchmark.py', 'audit_pointer_figures.py', 'loop_metrics.py', 'pointer_task.py')},
        'exports_sha256': {p.name: sha256_file(p) for p in args.output.iterdir() if p.suffix in ('.png', '.pdf', '.svg')},
        'data_sha256': {p.name: sha256_file(p) for p in args.output.iterdir() if p.suffix in ('.csv', '.json')}}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Saved failure metrics and matrices to {args.output}')


if __name__ == '__main__':
    main()
