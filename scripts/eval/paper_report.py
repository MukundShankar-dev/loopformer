"""Write the completion report and handoff status from audited suite artifacts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scripts.eval.paper_plot_contract import validate_manifest


def replace_status(path:Path,message:str)->None:
    start='<!-- paper-run-status:start -->';end='<!-- paper-run-status:end -->'
    text=path.read_text()
    if start not in text or end not in text:raise ValueError(f'{path}: missing owned status region')
    before,rest=text.split(start,1);_,after=rest.split(end,1)
    path.write_text(before+start+'\n'+message+'\n'+end+after)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=Path('eval/pointer_analysis/paper-20261009'));args=parser.parse_args();root=args.input
    audit=json.loads((root/'independent_audit.json').read_text());manifest=json.loads((root/'plots/manifest.json').read_text())
    if not audit['passed'] or manifest['status']!='complete':raise ValueError('Cannot report an incomplete/unverified suite')
    validate_manifest(manifest)
    from scripts.eval.paper_baseline_plots import require_audit
    baseline = require_audit(root)
    config=json.loads((root/'protocol.json').read_text());rows=list(csv.DictReader((root/'quality_by_depth.csv').open()))
    lines=['# Full pointer population analysis','','Generated from the completed frozen suite. No new training or checkpoint/threshold selection.','',
           '## Coverage and outcome','',f"Six documented checkpoint representatives; {audit['graphs']:,} graphs; every requested N=1–256; {audit['scalar_decisions_checked']:,} independently checked model/query decisions.",
           'The already-opened final benchmark is reused after executor-tensor and source identity checks; reuse is not a second confirmation trial.','',
           '| Model | Mean forced final letter | Strict trajectory at N=256 | Mean exact stopping | Mean answer + exact stop |',
           '| --- | ---: | ---: | ---: | ---: |']
    for arm in config['models']:
        name=arm['name'];entries=[]
        for metric in ('nominal_final_correct','complete_trajectory','exact_stop','joint_success'):
            selected=[r for r in rows if r['model']==name and r['metric']==metric and (metric!='complete_trajectory' or int(r['requested_depth'])==256)]
            rates=[float(r['rate']) for r in selected if r['rate']]
            entries.append(f'{np.mean(rates)*100:.2f}%' if rates else 'N/A')
        lines.append('| '+name+' | '+' | '.join(entries)+' |')
    lines.extend(['','Mean scores weight all 256 requested integers equally; they do not replace per-depth curves. Headless CE stopping is undefined. Strict trajectory and final-letter correctness have different meanings.','',
                  '## Ordinary-Qwen baseline','',
                  f"The pinned ordinary Qwen checkpoint ran all {baseline['queries']:,} identical graph/count questions: **{100*baseline['accuracy']:.2f}% exact final-letter accuracy**, {baseline['invalid_answers']:,} invalid answers and {baseline['token_budget_stops']:,} token-budget stops.",
                  'It uses the existing three-shot chat prompt, unconstrained greedy generation and eight new tokens; recurrent models use raw prompts and restricted readout. This is a matched-question baseline, not an isolated architecture/prompt ablation. The historical seed-17 6% result is retained separately.',
                  f"Raw compressed continuations, reconstruction hashes, depth/graph-stratum metrics, graph-cluster intervals and paired recurrent-minus-ordinary final-letter deltas are in `{root}/ordinary_qwen/`. Every response is independently decoded and checked against scalar dictionary execution; 54 predeclared native generation checks must pass.",
                  'Ordinary generation has no recurrent trajectory or exact stopping-loop metric. It is excluded from the logical-R-pass Pareto frontiers. Generated tokens/s is not recurrent transitions/s.', '',
                  '## Reproduction and evidence','', '```bash','bash analyze_pointer.sh','python -m scripts.eval.paper_status','```','',
                  f'Output: `{root}`. The same launcher resumes unchanged atomic extraction chunks. A changed protocol, checkpoint, implementation or batch shape is rejected.',
                  '',f'[Organized figure index](../../{root}/plots/index.md). Five scientific groups, the baseline/five-attempt writeup group and observed-policy Pareto comparisons; PNG/PDF/SVG, exact plotted arrays, captions and hashes.',
                  '',f'[Independent decision audit](../../{root}/independent_audit.json), [reference/chunk coverage](../../{root}/reference_audit.json), [figure manifest](../../{root}/plots/manifest.json).',
                  '', 'The scalar audit reconstructs raw dictionary transitions separately from vectorized scoring. Native batch-one calls verify 54 predeclared graph/count questions per model, including counts 1/6/12/32/128/256 and all nine graph strata. Historical inference uses mathematically equivalent causal-prefix partitioning; each count keeps its own continuous suffix state. No reference state is injected into R.',
                  '', '## Learning and uncertainty','',
                  'Nine retained executor snapshots run on the same 1,350 graphs. Step 1000 has monitoring but no retained weights. Final-timer composition is explicitly diagnostic; old snapshots are not presented as historical autonomous solvers.',
                  '', 'Pointwise Wilson intervals use graph-level binary outcomes. Depth-range summaries and paired deltas use the same 2,000 graph-cluster bootstrap draws within seed/mode strata, seed 239. All requests of a graph remain together. Timer-only integers are not treated as independent graph replications.',
                  '', 'Recovery, wrong episodes/censoring, direct-R/C first-error disagreement, risk-set hazards, signed/absolute timing residuals, structural strata and secondary exposure denominators are saved as tables. Correlated transition exposures are descriptive and receive no false independent-loop interval.',
                  '', 'Pareto figures compare frozen learned-stop policies within six requested-count bands. Cost is the native first stopping loop, or the full cap for a missing stop; quality requires exact stopping. The axes show logical recurrent passes, not matched FLOPs or latency. Headless CE has no autonomous stopping policy and is excluded. Frontiers concern observed point estimates, not statistically proven dominance or tuned thresholds. `quality_compute.csv` and plotted arrays preserve every point and interval.',
                  '', '## Limitations','',
                  'The graph panel is retrospective and opened. Historical architectures differ in data, trainable capacity and recipe; comparison does not isolate architectural causation. Only one successful R training seed exists. Tables have 26 states and long trajectories cycle; this is not evidence for a nonrepeating 50-state chain, unlimited depth, or cross-family transfer. The numeric timer ignores R correctness and does not establish quality-aware adaptive compute.',
                  '', 'Old render exports are retired only after this replacement passes. Raw historical metrics, weights, logs and numeric audits remain; prior render files remain recoverable from Git history.',''])
    Path('docs/experiments/pointer_population_analysis.md').write_text('\n'.join(lines))
    message=(f'The full suite is **complete and independently audited**: six models, {audit["graphs"]:,} graphs, all requested depths 1–256, '
             f'{audit["scalar_decisions_checked"]:,} checked decisions and nine retained successful-executor snapshots. '
             f'Ordinary Qwen additionally has {baseline["queries"]:,} independently checked matched questions. '
             'The clean figure set has seven groups with twenty-five figure families, including historical/matched baselines, final-letter comparisons, five-attempt writeup panels and two logical-compute Pareto comparisons; legends/colorbars, coincident-series labels, denominators and source hashes are included. '
             'See [the population report](experiments/pointer_population_analysis.md) for results and the figure index. Obsolete render exports are retired; raw evidence is retained.')
    for path in (Path('docs/analysis.md'),Path('docs/status.md')):replace_status(path,message)


if __name__=='__main__':main()
