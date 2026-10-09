"""Standalone, reference-audited pointer figures; requires optional Matplotlib."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from scripts.eval.pointer_task import sha256_file


COLORS = {'joint_success': '#2563a6', 'complete_trajectory': '#16826b', 'exact_stop': '#3e4349'}
LABELS = {'joint_success': 'Correct answer + exact stopping loop',
          'complete_trajectory': 'Every intermediate step correct', 'exact_stop': 'Exact stopping loop'}
MODES = {'random_function': 'Random functions', 'permutation': 'Permutations',
         'full_cycle': 'Full 26-state cycles'}
READING_COLOR = '#a35339'


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows])


def depth_axis(ax, maximum: int, *, boundaries: bool = True) -> None:
    ax.set(xlabel='Requested steps', xlim=(1, maximum))
    ax.grid(axis='y', color='#dee3e8', linewidth=.7)
    ax.set_axisbelow(True)
    if boundaries:
        for depth, color in ((12, '#16826b'), (63, READING_COLOR), (100, '#9c9da0')):
            if depth <= maximum:
                ax.axvline(depth, color=color, linestyle=':', linewidth=1, alpha=.65)


def quality_lines(ax, rows: list[dict]) -> None:
    x = values(rows, 'depth')
    for key in COLORS:
        ax.plot(x, 100 * values(rows, key), color=COLORS[key], label=LABELS[key],
                linewidth=2, linestyle='--' if key == 'exact_stop' else '-')
        if key != 'exact_stop':
            # Marginal graph-level intervals, not independent graph/count trials.
            ax.fill_between(x, 100 * values(rows, f'{key}_wilson_95_low'),
                100 * values(rows, f'{key}_wilson_95_high'), color=COLORS[key], alpha=.12, linewidth=0)


def zoom_floor(rows: list[dict]) -> float:
    minimum = min(float(r[f'{k}_wilson_95_low']) for r in rows
                  for k in ('joint_success', 'complete_trajectory'))
    return max(0, float(np.floor(100 * minimum - 1)))


def decorate(fig, title: str, subtitle: str, footer: str) -> None:
    fig.suptitle(title, x=.06, y=.98, ha='left', fontsize=17, fontweight='bold', color='#1d2935')
    fig.text(.06, .925, subtitle, ha='left', fontsize=10, color='#55616e')
    fig.text(.06, .025, footer, ha='left', fontsize=8.5, color='#55616e')
    fig.subplots_adjust(left=.07, right=.97, top=.75, bottom=.17, wspace=.28)


def save_figure(fig, output: Path, name: str) -> None:
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=200, facecolor='white')


def make_figures(summary: dict, grouped: list[dict], counts: list[dict],
                 graphs: list[dict], loops: list[dict], output: Path) -> None:
    """Plot only saved measured rates; no smoothing or model execution."""
    import matplotlib.pyplot as plt
    all_rows = [r for r in grouped if r['stratum'] == 'all']
    maximum = summary['config']['max_depth']
    graph_count = summary['graphs']
    if ([int(r['depth']) for r in all_rows] != list(range(1, maximum + 1))
            or len(graphs) != graph_count or len(counts) != maximum):
        raise ValueError('Figure coverage differs from the declared benchmark')
    exposure = 'Dotted lines: R training unroll 12 · largest trained count 63 · three-digit boundary 100.'
    uncertainty = 'Shading: pointwise 95% graph-level Wilson intervals; repeated horizons are paired, not independent.'
    subtitle = f'{graph_count:,} independent graphs · {summary["queries"]:,} paired queries · every count 1–{maximum}'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for ax in axes:
        quality_lines(ax, all_rows)
        depth_axis(ax, maximum)
        ax.set_ylabel('Success (%)')
    axes[0].set(title='Full scale', ylim=(0, 102))
    axes[1].set(title='Zoom of the same measurements', ylim=(zoom_floor(all_rows), 100.4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, .89), ncol=3,
               frameon=False, fontsize=8.5)
    decorate(fig, 'Does the complete model generalize in depth?', subtitle, exposure + '\n' + uncertainty)
    save_figure(fig, output, 'quality_over_depth'); plt.close(fig)

    mode_rows = [[r for r in grouped if r['stratum'] == f'graph_mode={mode}'] for mode in MODES]
    if any(not rows for rows in mode_rows):
        raise ValueError('Missing required graph-mode stratum')
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.3), sharey=True)
    minimum = min(zoom_floor(rows) for rows in mode_rows)
    for ax, (_, label), rows in zip(axes, MODES.items(), mode_rows, strict=True):
        quality_lines(ax, rows); depth_axis(ax, maximum)
        ax.set(title=f'{label}\n{int(rows[0]["graphs"]):,} graphs', ylim=(minimum, 100.4))
    axes[0].set_ylabel('Success (%) · zoomed scale')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, .89), ncol=3,
               frameon=False, fontsize=8.5)
    decorate(fig, 'Graph structure and depth generalization', subtitle, exposure + '\n' + uncertainty)
    save_figure(fig, output, 'quality_by_graph_type'); plt.close(fig)

    first_errors = Counter(int(r['first_error_loop']) for r in graphs if r['first_error_loop'])
    survivors = sum(not r['first_error_loop'] for r in graphs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    axes[0].plot(values(all_rows, 'depth'), 100 * values(all_rows, 'complete_trajectory'),
                 color=COLORS['complete_trajectory'], linewidth=2)
    axes[0].fill_between(values(all_rows, 'depth'),
        100 * values(all_rows, 'complete_trajectory_wilson_95_low'),
        100 * values(all_rows, 'complete_trajectory_wilson_95_high'), color=COLORS['complete_trajectory'], alpha=.13)
    depth_axis(axes[0], maximum)
    axes[0].set(title='Graphs with an entirely correct prefix', ylabel='Graphs surviving (%)',
                ylim=(zoom_floor(all_rows), 100.4))
    last = max(first_errors, default=1)
    x = np.arange(1, last + 1)
    axes[1].bar(x, [first_errors.get(int(i), 0) for i in x], color=READING_COLOR, width=.8)
    axes[1].set(title=f'First error in each failing graph\n{graph_count - survivors:,} fail; {survivors:,} / {graph_count:,} survive through {maximum}', xlabel='First incorrect loop',
                ylabel='Independent graphs', xlim=(.5, last + .5))
    axes[1].grid(axis='y', alpha=.25); axes[1].set_axisbelow(True)
    if 12 <= last:
        axes[1].axvline(12, color=COLORS['complete_trajectory'], linestyle=':', linewidth=1)
    decorate(fig, 'Where does execution first go wrong?', subtitle,
             'Each failing graph contributes once to the histogram; survivors are censored at the maximum tested depth.\n' + uncertainty)
    save_figure(fig, output, 'trajectory_failures'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    x = values(counts, 'depth')
    axes[0].plot(x, values(counts, 'initial_memory') - x, color=READING_COLOR, linewidth=1.6)
    axes[0].set(title='Number-reader residual', ylabel='Initial memory − requested count')
    valid = [r for r in counts if r['first_stop']]
    axes[1].plot(values(valid, 'depth'), values(valid, 'first_stop') - values(valid, 'depth'),
                 color=COLORS['exact_stop'], linewidth=1.8)
    axes[1].set(title='Actual first stopping-loop residual', ylabel='First stopping loop − requested count')
    axes[1].text(.98, .95, f'Missing stops: {len(counts) - len(valid)} / {maximum}',
                 transform=axes[1].transAxes, va='top', ha='right', fontsize=9)
    for ax in axes:
        depth_axis(ax, maximum); ax.axhline(0, color='#aab1b8', linewidth=.8, linestyle='--')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    if valid and all(float(r['first_stop']) == float(r['depth']) for r in valid):
        axes[1].set_ylim(-1, 1)
    decorate(fig, 'Reading the request and deciding when to stop', subtitle,
             exposure + '\nOne result per integer: the controller is graph-independent; graph repetitions add no numeric evidence.')
    save_figure(fig, output, 'controller_diagnostics'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    x = values(loops, 'loop')
    for key, label, color in (('accuracy', 'C answer readout', COLORS['joint_success']),
                              ('r_accuracy', 'Direct R state readout', COLORS['complete_trajectory']),
                              ('r_c_agreement', 'R/C decoded agreement', '#8e638d')):
        if key in loops[0]:
            axes[0].plot(x, 100 * values(loops, key), label=label, color=color, linewidth=1.7)
    conditional = np.asarray([float(r['conditional_transition_accuracy']) if r['conditional_transition_accuracy'] else np.nan for r in loops])
    axes[1].plot(x, 100 * conditional, color=COLORS['complete_trajectory'], linewidth=1.8)
    axes[1].set(title='Next-step accuracy given a correct prefix', ylabel='Conditional transition accuracy (%)')
    axes[0].set(title='Decoded states at each recurrent loop', ylabel='Accuracy / agreement (%)')
    axes[0].legend(loc='lower right', frameon=False, fontsize=8.5)
    observed = [100 * values(loops, key) for key in ('accuracy', 'r_accuracy', 'r_c_agreement') if key in loops[0]]
    finite = np.concatenate([*observed, 100 * conditional[np.isfinite(conditional)]])
    low = max(0, float(np.floor(finite.min() - 1)))
    for ax in axes:
        depth_axis(ax, maximum); ax.set(xlabel='Recurrent loop', ylim=(low, 100.4))
    decorate(fig, 'Intermediate execution readouts', subtitle,
             exposure + '\nConditional accuracy uses only graphs correct at every preceding loop; these are descriptive rates without CI bands.')
    save_figure(fig, output, 'intermediate_readouts'); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Fresh figure directory; default RESULTS/plots')
    args = parser.parse_args()
    summary = json.loads((args.results / 'summary.json').read_text())
    if summary['status'] != 'complete' or not summary['native_fidelity']['passed'] or not summary['inference_files_unchanged']:
        raise ValueError('Plot only a completed benchmark with passed fidelity and unchanged inference files')
    audit_path = args.results / 'result_reference_audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['passed'] or audit['summary_sha256'] != sha256_file(args.results / 'summary.json'):
        raise ValueError('Need a current independent reference audit before plotting')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': '#b2bac3', 'axes.labelcolor': '#34404c',
        'xtick.color': '#55616e', 'ytick.color': '#55616e', 'svg.fonttype': 'none'})
    names = ('summary.json', 'per_count_and_stratum.csv', 'controller_counts.csv', 'graphs.csv', 'per_loop.csv',
             'result_reference_audit.json')
    output = args.output or args.results / 'plots'
    output.mkdir(parents=True, exist_ok=False)
    make_figures(summary, read_csv(args.results / names[1]), read_csv(args.results / names[2]),
                 read_csv(args.results / names[3]), read_csv(args.results / names[4]), output)
    (output / 'provenance.json').write_text(json.dumps({'matplotlib': matplotlib.__version__,
        'script_sha256': sha256_file(Path(__file__)),
        'inputs_sha256': {name: sha256_file(args.results / name) for name in names},
        'vertical_lines': {'12': 'maximum supervised recurrent prefix',
                           '63': 'maximum trained requested count; five excluded', '100': 'three-digit boundary'},
        'uncertainty': 'Per-count marginal graph-level Wilson intervals; no independent count-query sample assumption.',
        'limitations': 'Fixed prompt and 26-state graphs; controller ignores graphs; finite tested range only.',
        'figures_sha256': {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.suffix in ('.png', '.svg', '.pdf')}
    }, indent=2) + '\n')
    print(f'Saved five standalone PNG/SVG/PDF figures to {output}')


if __name__ == '__main__':
    main()
