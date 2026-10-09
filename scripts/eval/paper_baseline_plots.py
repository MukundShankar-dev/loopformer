"""Matched-question ordinary generation and final-letter comparison figures."""
import json
from pathlib import Path

import numpy as np

from scripts.eval.pointer_task import sha256_file


def require_audit(root: Path) -> dict:
    """Require the raw audit and the exact matrices it approved before rendering."""
    directory = root / 'ordinary_qwen'
    audit = json.loads((directory/'independent_audit.json').read_text())
    if (not audit['passed'] or audit['outcomes_sha256'] != sha256_file(directory/'outcomes.npz') or
            audit['recurrent_outcomes_sha256'] != sha256_file(root/'outcomes.npz') or
            audit['freeze_sha256'] != sha256_file(directory/'freeze.json')):
        raise ValueError('Ordinary baseline audit or approved outcomes differ')
    for name, digest in audit['analysis_artifacts_sha256'].items():
        if sha256_file(directory/name) != digest:
            raise ValueError('Ordinary baseline approved analysis table differs')
    return audit


def baseline_figures(figures, config: dict, recurrent) -> None:
    """Two labelled families; no fabricated ordinary loop/stopping measurements."""
    import matplotlib.pyplot as plt
    from scripts.eval.plot_paper_analysis import axis, grouped_lines, read, MODES, MODE_NAMES, NAMES

    directory = figures.root/'ordinary_qwen'; audit = require_audit(figures.root)
    metadata = json.loads((figures.root/'graph_metadata.json').read_text())
    rows = read(directory/'quality_by_depth.csv'); x = np.array(config['requests'])
    if [int(r['requested_depth']) for r in rows] != config['requests']:
        raise ValueError('Ordinary plot count order differs')
    with np.load(directory/'outcomes.npz', allow_pickle=False) as data:
        correct = data['correct']; predictions = data['predictions']
        if correct.shape != (len(metadata), len(x)) or not np.array_equal(data['counts'], x):
            raise ValueError('Ordinary plot population differs')
        rate = correct.mean(0); invalid = (~data['valid']).mean(0); budget = data['token_budget'].mean(0)
    if not np.array_equal(rate, np.array([float(r['accuracy']) for r in rows])):
        raise ValueError('Ordinary plot source table disagrees with audited matrix')
    low, high = np.array([[float(r[k]) for r in rows] for k in ('low','high')])
    modes = np.array([correct[np.array([m['graph_mode'] == mode for m in metadata])].mean(0) for mode in MODES])
    frequencies = np.bincount(predictions.ravel(), minlength=27)/predictions.size
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0,0]
    ax.plot(x, 100*rate, color='#333333', label='Ordinary Qwen · greedy 3-shot')
    ax.fill_between(x, 100*low, 100*high, color='#333333', alpha=.14, label='95% pointwise Wilson interval')
    axis(ax, 'Final-letter accuracy (%)'); ax.set_ylim(-.5, min(102, max(20, 100*high.max()+3)))
    ax.set_title(f'Mean accuracy: {100*audit["accuracy"]:.2f}%'); ax.legend(fontsize=8)
    ax = axes[0,1]
    if np.array_equal(invalid,budget):
        ax.plot(x,100*invalid,'--',color='#8060b0',label='Invalid format / 8-token budget (identical rates)')
    else:
        ax.plot(x, 100*invalid, '--', color='#b33a3a', label='Invalid final-letter format')
        ax.plot(x, 100*budget, ':', color='#a57712', label='Reached 8-token budget')
    axis(ax, 'Queries (%)'); ax.set_ylim(-.5,min(102,max(10,100*max(invalid.max(),budget.max())+3)))
    ax.set_title('Format and generation budget'); ax.legend(fontsize=8)
    ax = axes[1,0]
    image = ax.imshow(100*modes, aspect='auto', vmin=0, vmax=min(100,max(20,100*modes.max()+3)), cmap='viridis', extent=(.5,256.5,2.5,-.5))
    ax.set_yticks(range(3), [MODE_NAMES[m] for m in MODES]); ax.set(xlabel='Requested transitions N', title='Accuracy by graph type · 450 graphs/type')
    fig.colorbar(image, ax=ax, label='Final-letter accuracy (%)')
    ax = axes[1,1]
    ax.bar(np.arange(27), 100*frequencies, color='#626e7c', label='Returned letters (all requests)')
    ax.set(xticks=np.arange(27), xticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')+['invalid'],
           ylabel='Queries (%)', title='Response distribution', ylim=(0,max(.1,100*frequencies.max()*1.25)))
    ax.tick_params(axis='x', labelrotation=90); ax.legend(fontsize=8)
    sources = [directory/p for p in ('outcomes.npz','quality_by_depth.csv','independent_audit.json','freeze.json')]
    sources += [figures.root/'graph_metadata.json', Path(__file__)]
    figures.save(fig, '06_writeup', 'matched_baseline', 'Ordinary Qwen · current full graph/count benchmark',
        'Same 1,350 graphs, starts, rule order and requests 1–256 as the recurrent suite; 345,600 unconstrained generations.\n'
        'Existing 3-shot chat prompt; greedy, max 8 new tokens. Invalid responses are failures. Prompt/readout differ from recurrent inference.\n'
        'Pointwise intervals use 1,350 graphs; requests share graph clusters. Generation tokens are not recurrent transitions.',
        dict(counts=x, accuracy=rate, wilson=np.array([low,high]), invalid=invalid, token_budget=budget,
             mode_accuracy=modes, prediction_frequency=frequencies), source_paths=sources)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6)); numeric = dict(counts=x, ordinary_accuracy=rate)
    for ax, metric, title in zip(axes, ('nominal_final_correct','stopped_answer_correct'),
                                 ('Readout at externally requested depth', 'Returned answer after learned stopping'), strict=True):
        series = []
        for arm in config['models']:
            name = arm['name']; key = f'{name}__{metric}'
            if key not in recurrent.files:
                continue
            values = recurrent[key]
            if values.shape != correct.shape:
                raise ValueError('Ordinary/recurrent paired plot shapes differ')
            y = 100*values.mean(0); numeric[key] = y/100; series.append((name,y,NAMES[name]))
        grouped_lines(ax,x,series)
        ax.plot(x,100*rate,color='#111111',linestyle=(0,(6,2,1,2)),linewidth=2.2,
                label='Ordinary Qwen · 3-shot generation',zorder=5)
        axis(ax, 'Final-letter accuracy (%)'); ax.set_title(title); ax.legend(fontsize=7, loc='center right')
    figures.save(fig, '01_quality', 'final_letter_comparison', 'Final answers on identical questions · ordinary and recurrent',
        'All models use the same 345,600 graph/count queries. Ordinary Qwen uses 3-shot chat generation; recurrent models use raw prompts/restricted readout.\n'
        'Left: recurrent readout at N. Right: actual returned letter; correct letters at wrong loops count here, so this is NOT answer + exact-stop success.\n'
        'No trajectory or loop-timing metric is assigned to ordinary generation. CE has no learned stop and is absent from the right panel.',
        numeric, source_paths=sources+[figures.root/'outcomes.npz',directory/'paired_deltas.csv'])
