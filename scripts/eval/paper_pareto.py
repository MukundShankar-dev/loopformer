"""Quality versus logical recurrent budget for the frozen observed stop policies.

These are retrospective Pareto points, not threshold tuning or latency benchmarks.
A missing stop costs the full cap and remains a quality failure. Headless CE has
no autonomous policy and is excluded. Both quality axes require exact timing.
"""
import csv
from pathlib import Path
import textwrap

import numpy as np

RANGES = ((1, 6), (7, 12), (13, 32), (33, 64), (65, 128), (129, 256))
METRICS = ('joint_success', 'strict_success')


def pareto_mask(cost: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """Minimize cost, maximize quality; exact duplicate points remain tied."""
    cost, quality = np.asarray(cost), np.asarray(quality)
    if cost.ndim != 1 or cost.shape != quality.shape or not np.isfinite(cost).all() or not np.isfinite(quality).all():
        raise ValueError('Need aligned finite one-dimensional costs and qualities')
    dominated = ((cost[:, None] <= cost[None, :]) & (quality[:, None] >= quality[None, :]) &
                 ((cost[:, None] < cost[None, :]) | (quality[:, None] > quality[None, :]))).any(0)
    return ~dominated


def tradeoff_points(arrays, names: list[str], counts: np.ndarray, cap: int,
                    ranges=RANGES) -> list[dict]:
    """Use native first stopping loops, not the longer forced extraction horizon."""
    rows = []
    for name in names:
        stops = arrays[f'{name}__stops']
        if stops.ndim != 2 or stops.shape[1] != len(counts) or not np.issubdtype(stops.dtype, np.integer):
            raise ValueError('Unaligned stopping records')
        if (stops == -2).all():
            continue  # Undefined stopping, not a zero-cost/zero-quality policy.
        if ((stops != -1) & ((stops < 1) | (stops > cap))).any():
            raise ValueError('Invalid or mixed undefined stopping records')
        cost = np.where(stops > 0, stops, cap)
        for low, high in ranges:
            mask = (counts >= low) & (counts <= high)
            if not mask.any():
                raise ValueError('Empty requested-count cohort')
            for metric in METRICS:
                quality = arrays[f'{name}__{metric}']
                if quality.shape != stops.shape or quality.dtype != np.bool_:
                    raise ValueError('Need audited boolean graph/count quality arrays')
                if (quality & (stops != counts[None, :])).any():
                    raise ValueError('Wrong-loop or missing stops cannot count as success')
                rows.append(dict(model=name, metric=metric, depth_range=f'{low}-{high}',
                    graphs=len(stops), queries=len(stops)*int(mask.sum()),
                    mean_loops=float(cost[:, mask].mean()), success_rate=float(quality[:, mask].mean()),
                    missing_stop_rate=float((stops[:, mask] == -1).mean())))
    for low, high in ranges:
        for metric in METRICS:
            group = [r for r in rows if r['depth_range'] == f'{low}-{high}' and r['metric'] == metric]
            frontier = pareto_mask(np.array([r['mean_loops'] for r in group]), np.array([r['success_rate'] for r in group]))
            for row, value in zip(group, frontier, strict=True):
                row['pareto'] = bool(value)
    return rows


def pareto_figures(figures, config: dict, arrays, names: list[str], metadata: list[dict]) -> None:
    """Two figures, with paired graph-cluster intervals and no coordinate jitter."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.eval.plot_paper_analysis import COLORS, NAMES, STYLES, read

    counts = np.asarray(config['requests'])
    rows = tradeoff_points(arrays, names, counts, config['safety_cap'])
    if not rows:
        raise ValueError('No autonomous stopping policies available for Pareto comparison')
    intervals = {(r['model'], r['metric'], r['depth_range']): r
                 for r in read(figures.root / 'cluster_intervals.csv')}
    rng = np.random.default_rng(config['uncertainty']['bootstrap_seed'])
    groups = {}
    for i, meta in enumerate(metadata):
        groups.setdefault((meta['dataset_seed'], meta['graph_mode']), []).append(i)
    draws = np.concatenate([rng.choice(ids, size=(config['uncertainty']['bootstrap_repeats'], len(ids)), replace=True)
                            for ids in groups.values()], 1)
    cost_intervals = {}
    for row in rows:
        key = row['model'], row['depth_range']
        if key not in cost_intervals:
            low, high = map(int, row['depth_range'].split('-'))
            stops = arrays[f"{row['model']}__stops"]
            cost = np.where(stops > 0, stops, config['safety_cap'])[:, (counts >= low) & (counts <= high)].mean(1)
            cost_intervals[key] = np.quantile(cost[draws].mean(1), [.025, .975])
        row['loops_low'], row['loops_high'] = map(float, cost_intervals[key])
        interval = intervals[row['model'], row['metric'], row['depth_range']]
        if abs(float(interval['rate']) - row['success_rate']) > 1e-12:
            raise ValueError('Pareto quality differs from audited cluster summary')
        row['quality_low'], row['quality_high'] = float(interval['low']), float(interval['high'])
    table_path = figures.root / 'quality_compute.csv'
    with table_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    models = [name for name in names if any(r['model'] == name for r in rows)]
    handles = [Line2D([], [], color=COLORS[n], marker=STYLES[n][1], linestyle='none', label=NAMES[n]) for n in models]
    handles.append(Line2D([], [], color='#45505c', linestyle='--', label='Observed nondominated points (guide)'))
    handles.append(Line2D([], [], color='#45505c', marker='o', markerfacecolor='none', linestyle='none', label='Exact ties (policies named at point)'))
    short = {'joint_full':'Joint', 'fixed6':'Fixed-6', 'fixed12':'Fixed-12', 'gru':'GRU', 'final':'Final'}
    for metric in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        numeric = {}
        for ax, (low, high) in zip(axes.flat, RANGES, strict=True):
            cohort = [r for r in rows if r['metric'] == metric and r['depth_range'] == f'{low}-{high}']
            # Draw coincident coordinates once and explicitly name every policy.
            points = {}
            for row in cohort:
                points.setdefault((row['mean_loops'], row['success_rate']), []).append(row)
            for (x, y), tied in points.items():
                for row in tied:
                    n = row['model']
                    ax.errorbar(x, 100*y,
                        xerr=np.maximum(0, [[x-row['loops_low']], [row['loops_high']-x]]),
                        yerr=100*np.maximum(0, [[y-row['quality_low']], [row['quality_high']-y]]),
                        color=COLORS[n], capsize=3, fmt='none', alpha=.7)
                if len(tied) > 1:
                    ax.plot(x, 100*y, marker='o', markerfacecolor='none', color='#45505c', linestyle='none')
                    label = textwrap.fill(' / '.join(short[r['model']] for r in tied), width=25)
                    right = x < 16
                    ax.annotate(label, (x, 100*y), xytext=(8 if right else -8, 12 if y < .15 else -28),
                                textcoords='offset points', ha='left' if right else 'right', fontsize=7)
                else:
                    n = tied[0]['model']
                    ax.plot(x, 100*y, marker=STYLES[n][1], color=COLORS[n], linestyle='none')
            frontier = sorted({(r['mean_loops'], r['success_rate']) for r in cohort if r['pareto']})
            if frontier:
                ax.plot([p[0] for p in frontier], [100*p[1] for p in frontier], '--', color='#45505c', alpha=.6)
            ax.set(title=f'Requested N = {low}–{high}', xlabel='Mean executed R passes per query (log scale)',
                   ylabel='Answer + exact stop (%)' if metric == 'joint_success' else 'Every step + exact stop (%)',
                   xscale='log', xlim=(.8, config['safety_cap']*1.2), ylim=(-3, 103))
            ax.set_xticks([1, 10, 100, config['safety_cap']], ['1', '10', '100', str(config['safety_cap'])])
            ax.grid(alpha=.2); ax.set_axisbelow(True)
            numeric[f'{low}-{high}__models'] = np.array([r['model'] for r in cohort])
            numeric[f'{low}-{high}__values'] = np.array([[r[k] for k in (
                'mean_loops','success_rate','loops_low','loops_high','quality_low','quality_high','pareto','missing_stop_rate')] for r in cohort])
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.945), ncol=3, fontsize=8)
        fig.subplots_adjust(hspace=.32)
        title = 'Answer-and-stop quality versus recurrent budget' if metric == 'joint_success' else 'Strict execution-and-stop quality versus recurrent budget'
        figures.save(fig, '07_compute', f'pareto_{metric}', title,
            'Frozen policies on the same graphs/counts; 95% paired graph-cluster intervals. Missing stops cost all 272 passes.\n'
            'Loops are a logical budget, not FLOPs/latency; block costs differ. CE-only stopping is undefined and excluded.\n'
            'Exact ties are drawn once with policy names. Frontier is among observed point estimates; no threshold fitting or interpolated policy.',
            numeric, source_paths=[table_path, figures.root/'outcomes.npz', figures.root/'cluster_intervals.csv',
                                   figures.root/'protocol.json', figures.root/'graph_metadata.json', Path(__file__)], layout_top=.86)
