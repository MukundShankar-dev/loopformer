# Full pointer population analysis

Generated from the completed frozen suite. No new training or checkpoint/threshold selection.

## Coverage and outcome

Six documented checkpoint representatives; 1,350 graphs; every requested N=1–256; 2,073,600 independently checked model/query decisions.
The already-opened final benchmark is reused after executor-tensor and source identity checks; reuse is not a second confirmation trial.

| Model | Mean forced final letter | Strict trajectory at N=256 | Mean exact stopping | Mean answer + exact stop |
| --- | ---: | ---: | ---: | ---: |
| ce_full | 7.32% | 0.00% | N/A | N/A |
| joint_full | 6.30% | 0.00% | 2.31% | 2.26% |
| fixed6 | 7.18% | 0.07% | 2.34% | 2.29% |
| fixed12 | 9.08% | 0.00% | 4.46% | 4.36% |
| gru | 97.81% | 96.89% | 4.57% | 4.52% |
| final | 97.81% | 96.89% | 100.00% | 97.81% |

Mean scores weight all 256 requested integers equally; they do not replace per-depth curves. Headless CE stopping is undefined. Strict trajectory and final-letter correctness have different meanings.

## Ordinary-Qwen baseline

The pinned ordinary Qwen checkpoint ran all 345,600 identical graph/count questions: **6.25% exact final-letter accuracy**, 332 invalid answers and 332 token-budget stops.
It uses the existing three-shot chat prompt, unconstrained greedy generation and eight new tokens; recurrent models use raw prompts and restricted readout. This is a matched-question baseline, not an isolated architecture/prompt ablation. The historical seed-17 6% result is retained separately.
Raw compressed continuations, reconstruction hashes, depth/graph-stratum metrics, graph-cluster intervals and paired recurrent-minus-ordinary final-letter deltas are in `eval/pointer_analysis/paper-20261009/ordinary_qwen/`. Every response is independently decoded and checked against scalar dictionary execution; 54 predeclared native generation checks must pass.
Ordinary generation has no recurrent trajectory or exact stopping-loop metric. It is excluded from the logical-R-pass Pareto frontiers. Generated tokens/s is not recurrent transitions/s.

## Reproduction and evidence

```bash
bash analyze_pointer.sh
python -m scripts.eval.paper_status
```

Output: `eval/pointer_analysis/paper-20261009`. The same launcher resumes unchanged atomic extraction chunks. A changed protocol, checkpoint, implementation or batch shape is rejected.

[Organized figure index](../../eval/pointer_analysis/paper-20261009/plots/index.md). Five scientific groups, the baseline/five-attempt writeup group and observed-policy Pareto comparisons; PNG/PDF/SVG, exact plotted arrays, captions and hashes.

[Independent decision audit](../../eval/pointer_analysis/paper-20261009/independent_audit.json), [reference/chunk coverage](../../eval/pointer_analysis/paper-20261009/reference_audit.json), [figure manifest](../../eval/pointer_analysis/paper-20261009/plots/manifest.json).

The scalar audit reconstructs raw dictionary transitions separately from vectorized scoring. Native batch-one calls verify 54 predeclared graph/count questions per model, including counts 1/6/12/32/128/256 and all nine graph strata. Historical inference uses mathematically equivalent causal-prefix partitioning; each count keeps its own continuous suffix state. No reference state is injected into R.

## Learning and uncertainty

Nine retained executor snapshots run on the same 1,350 graphs. Step 1000 has monitoring but no retained weights. Final-timer composition is explicitly diagnostic; old snapshots are not presented as historical autonomous solvers.

Pointwise Wilson intervals use graph-level binary outcomes. Depth-range summaries and paired deltas use the same 2,000 graph-cluster bootstrap draws within seed/mode strata, seed 239. All requests of a graph remain together. Timer-only integers are not treated as independent graph replications.

Recovery, wrong episodes/censoring, direct-R/C first-error disagreement, risk-set hazards, signed/absolute timing residuals, structural strata and secondary exposure denominators are saved as tables. Correlated transition exposures are descriptive and receive no false independent-loop interval.

Pareto figures compare frozen learned-stop policies within six requested-count bands. Cost is the native first stopping loop, or the full cap for a missing stop; quality requires exact stopping. The axes show logical recurrent passes, not matched FLOPs or latency. Headless CE has no autonomous stopping policy and is excluded. Frontiers concern observed point estimates, not statistically proven dominance or tuned thresholds. `quality_compute.csv` and plotted arrays preserve every point and interval.

## Limitations

The graph panel is retrospective and opened. Historical architectures differ in data, trainable capacity and recipe; comparison does not isolate architectural causation. Only one successful R training seed exists. Tables have 26 states and long trajectories cycle; this is not evidence for a nonrepeating 50-state chain, unlimited depth, or cross-family transfer. The numeric timer ignores R correctness and does not establish quality-aware adaptive compute.

Old render exports are retired only after this replacement passes. Raw historical metrics, weights, logs and numeric audits remain; prior render files remain recoverable from Git history.

## Completed review

[Post-transfer verification, visual review and interpretation](pointer_population_review.md).
