"""Optional standalone benchmark figures; requires matplotlib, no model execution."""
import argparse
import csv
import json
from pathlib import Path

from scripts.eval.pointer_task import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.results / 'summary.json').read_text())
    if summary['status'] != 'complete' or not summary['native_fidelity']['passed']:
        raise ValueError('Plot only a completed benchmark with passed native fidelity')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader((args.results / 'per_count_and_stratum.csv').open()) if r['stratum'] == 'all']
    counts = list(csv.DictReader((args.results / 'controller_counts.csv').open()))
    x = [int(r['depth']) for r in rows]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for key, label, color, style in (
        ('complete_trajectory', 'Executor: every step correct', '#16845b', '-'),
        ('exact_stop', 'Controller: exact stop', '#333333', '--'),
        ('joint_success', 'Whole model: answer + exact stop', '#2463b4', '-')):
        axes[0].plot(x, [100*float(r[key]) for r in rows], label=label, color=color, linestyle=style, linewidth=1.9)
    axes[0].set(xlabel='Requested steps', ylabel='Accuracy (%)', ylim=(-3, 104), xlim=(1, 256), title='Executor quality and exact stopping')
    axes[0].legend(loc='center right', frameon=False, fontsize=9)
    axes[1].plot(x, x, color='#777777', linestyle='--', linewidth=1.2, label='Correct initialization: memory = N')
    axes[1].plot([int(r['depth']) for r in counts], [float(r['initial_memory']) for r in counts],
                 color='#b14536', linewidth=1.8, label='Frozen learned initializer')
    axes[1].set(xlabel='Requested steps', ylabel='Initial controller memory', xlim=(1, 256), title='Number reading before recurrence')
    axes[1].legend(loc='upper left', frameon=False, fontsize=9)
    for ax in axes:
        ax.axvline(12, color='#bbbbbb', linestyle=':', linewidth=1)
        ax.axvline(63, color='#888888', linestyle=':', linewidth=1)
        ax.axvline(100, color='#bbbbbb', linestyle=':', linewidth=1)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle(f'Frozen pointer benchmark · {summary["graphs"]:,} independent graphs · counts 1–256', fontsize=14)
    out = args.results / 'plots'; out.mkdir(exist_ok=False)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'quality_and_initialization.{suffix}', dpi=180)
    plt.close(fig)
    (out / 'provenance.json').write_text(json.dumps({'matplotlib': matplotlib.__version__,
        'script_sha256': sha256_file(Path(__file__)),
        'inputs_sha256': {name: sha256_file(args.results / name) for name in
                         ('summary.json', 'per_count_and_stratum.csv', 'controller_counts.csv')},
        'vertical_lines': {'12': 'maximum supervised recurrent prefix', '63': 'maximum trained requested value; five values excluded',
                           '100': 'three-digit boundary'},
        'uncertainty': 'Per-count graph-level Wilson intervals are in the CSV; controller graph variability is degenerate at fixed count.'}, indent=2) + '\n')
    print(f'Saved standalone PNG/SVG/PDF figures to {out}')


if __name__ == '__main__':
    main()
