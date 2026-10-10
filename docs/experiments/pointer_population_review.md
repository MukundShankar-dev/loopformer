# Completed pointer population review — 2026-10-10

The frozen evaluation and reporting pipeline is complete. This review adds no
training, checkpoint selection, threshold fitting or model inference. The
[generated population report](pointer_population_analysis.md) owns the complete
score table; the [analysis contract](../analysis.md) owns checkpoint selection,
architecture links, target semantics and the writeup coverage map.

## Evidence and article assets

All six recurrent representatives finished on 1,350 graphs, with each requested
count from 1 to 256. Ordinary Qwen finished the same 345,600 questions. Nine saved
successful-executor snapshots finished on the same graphs. Both independent
scoring audits pass: 2,073,600 recurrent decisions and 345,600 ordinary responses.
The recurrent audit also checks 15,360 aggregate cells and cross-checks every
reused final-model decision against the original complete benchmark.

Outputs are on both the Mac and desktop at
`eval/pointer_analysis/paper-20261009/`. No checkpoint weights were copied to the
Mac. The [figure index](../../eval/pointer_analysis/paper-20261009/plots/index.md)
contains historical and matched baselines, five-attempt training/evaluation
panels, aggregate comparisons, failure/structure/stopping matrices, successful-R
learning progression and two logical-compute Pareto comparisons. There are 25
figure families with PNG, PDF and SVG exports. A hosted demo remains unimplemented.
The pipeline has no remaining inference or reporting job.

## Findings

Ordinary Qwen scores **21,614/345,600 = 6.25%** final-letter accuracy with the
existing three-shot chat prompt. It produces 332 invalid responses, all reaching
the eight-token generation budget. Z and T account for about 61.33% of all
responses. These are observed output biases; they do not identify whether the
few-shot prompt causes them. Recurrent models use different prompts and
restricted symbol readouts, so this is a matched-question comparison, not an
isolated causal ablation of recurrence.

The early designs learn useful execution within their trained horizons, but
lose it beyond those horizons. At requested depth 12, strict complete-trajectory
rates are 1.11% for full-sequence CE, 0.37% for joint completion and 2.15% for
fixed memory trained through six. Fixed memory trained through 12 reaches
93.33% there, but only 0.07% at 32 and zero at 256. Increasing training depth
moves its failure boundary; it does not establish long-range extrapolation.

The separated count-free executor is qualitatively different: the GRU and final
numeric-timer models have identical R predictions, **97.81% mean forced final-letter
accuracy**, and **1,308/1,350 = 96.89% strict trajectories through 256**. The GRU's
mean exact-stop rate is 4.57%, versus 100% for the final timer. Its strong forced
execution therefore does not make it a working autonomous solver. The final
model reaches **97.81% mean correct-answer-plus-exact-stop success**; every
remaining failure on this panel is an execution failure, rather than wrong timing.
At N=256 specifically, final-letter/joint success is 97.78%; 97.81% averages all
256 requested counts.

At fixed Steps=256, the final executor's 42 first errors occur at loops 2–30:
18/450 full cycles, 12/450 permutations and 12/450 random functions. Those mode
rates have overlapping uncertainty intervals; the small transient-length strata
also have wide intervals. The evidence does not support a sharply identified
single structural cause. Later correct letters on a failed trajectory can arise
from recovery or cyclic coincidence. They never erase an earlier strict-prefix
error, and correct letters at wrong stopping loops never count as joint success.

Successful R learning improves sharply between retained updates 750 and 1250;
update 1000 has development monitoring but no retained weights. Do not locate
an exact population transition at 1000 from an interpolated snapshot. The last
saved update is not the best on this opened panel: step 2000 has 1,315 strict
trajectories through 256, versus 1,308 at step 2250. The paired change loses
29 previously successful graphs and gains 22. This is a small net late regression,
not evidence that more training necessarily helps. Keep the predeclared
step-2250 final model; selecting 2000 using this benchmark would change the
selection protocol after observing the results.

These comparisons bundle architecture, data distribution, trainable capacity and
optimization changes. They support a practical successful recipe and distinct
execution/control failure modes, not proof that one particular change caused the
improvement. Only one successful R training seed exists. All tables have 26
states and long trajectories cycle. Depth-256 success is finite cyclic-task depth
generalization, not a nonrepeating 50-state-chain result, unlimited depth or
cross-task generality. This panel is retrospective and already opened; later
tuning needs fresh confirmation.

## Post-transfer and visual verification

The Mac copy independently matches all 1,396 recurrent raw-chunk hashes and
1,350 ordinary continuation-chunk hashes. Every figure's plotted numeric file,
all 75 render exports, declared source files and global input hashes match the
manifest. Desktop absolute provenance paths are mapped to the same repository
files for verification without rewriting recorded provenance. All 72 snapshot
plot cells were independently recomputed from raw prediction chunks, including
coverage and the paired step-2000/2250 change. The final exact-stop matrix is true
for all 345,600 queries. Machine-readable checks are saved in
[completed_review.json](../../eval/pointer_analysis/paper-20261009/completed_review.json).

All 25 actual figure families were visually inspected, including axes, legends,
metric definitions, denominators and shared/overlapping executor labels. Discrete
heatmaps now explicitly use nearest-cell rendering: earlier default smoothing
blurred row boundaries and could suggest unmeasured intermediate hazard values.
The execution/timing decomposition legend now has a white background so its
color swatches remain visible over filled regions. The full figure set was
regenerated locally from audited cached outputs, with unchanged scores, and the
changed panels were rechecked. No GPU inference was needed. All 16 focused reporting/export/writeup/Pareto/status
tests passed in 15.69 seconds; 433 local Markdown targets exist. Source and
documentation whitespace checks pass; original CSV line endings and raw terminal
progress logs are preserved. Old 69 historical
render exports were retired on each host after replacement passed; raw evidence
and Git history remain available.

Reproduction uses `bash analyze_pointer.sh` on the configured desktop; identical
completed chunks resume without redoing inference. For rendering only, run:

```bash
python -m scripts.eval.plot_paper_analysis --replace
python -m scripts.eval.paper_report
```

The bounded next step is the project writeup using the completed article assets.
No further training or benchmark is queued. Hosting the recurrent solver is a
separate task requiring its custom executor/controller inference path.
