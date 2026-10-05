# Remaining-work supervision: matched seed-61 comparison

Audited 2026-10-05 after pulling artifact commit `400b1a3`; run source `44c6417`.
This is development evidence from one graph split, not independent confirmation.

## Protocol and artifacts

The desktop ran `bash train_controller_remaining.sh`, using
[configs/controller_remaining.json](../../configs/controller_remaining.json).
All six runs initialize the original step-2,250 controller and keep R frozen.
They reuse the same training cache: 2,048 training graphs paired with requested
counts 1–6, 8, 10, 12. Each has 3,000 updates, batch 256, the same LR schedule,
and paired optimizer seeds 83/89/97. Seed changes affect minibatch ordering and
numerical-readout initialization, not the graph panel or original controller.

The stop-only arm trains a detached numerical measurement readout. The auxiliary
arm also backpropagates weight-1 remaining-work regression into controller memory.
Targets N−t cover t=0..N, with fixed normalization scale 12. Targets and predicted
numbers are never forward inputs or stopping rules. Selection uses trained-count
validation exact stops, then BCE, independently within each run.

- [Training artifacts](../../models/stage1_pointer/controller-remaining-seed61/).
- [Comparison and evaluation](../../eval/pointer_diagnostics/controller-remaining-seed61/).
- [W&B evaluation](https://wandb.ai/mukunds/loopformer/runs/v1i06r1b).

Validation pairs 128 graphs with counts 1–16. Deep evaluation pairs 32 other
graphs with counts 13–64. All controllers replay the same frozen observations;
both seed-83 arms also run 16 native validation and 16 native deep checks.
The separate readout remains outside the portable checkpoint.

## Stopping did not materially improve

Mean exact stopping across the three optimization seeds:

| Cohort | Stop only | Remaining-work loss | Paired change |
| --- | ---: | ---: | ---: |
| Trained requests, validation | 95.54% | 95.31% | −0.23 pp |
| Held-out 7/9/11, validation | 46.61% | 46.70% | +0.09 pp |
| Count 7, validation | 88.54% | 85.94% | −2.60 pp |
| Count 9, validation | 0% | 0% | 0 pp |
| Count 11, validation | 51.30% | 54.17% | +2.86 pp |
| Counts 13–64, deep | 1.703% | 1.723% | +0.020 pp |

Paired seed differences in overall interpolation are −1.56, +1.82 and 0 percentage
points (sample SD 1.69 pp). There is no consistent overall improvement. Count
seven worsens in all three auxiliary runs. Joint exact-stop-and-answer accuracy
on interpolation is identical in the mean: 46.35% in both arms.

Every run has zero exact stops at every deep count 14–64. All deep successes
occur at count 13: stop-only seeds get 28/32, 28/32, 29/32; auxiliary seeds get
31/32, 27/32, 28/32. Do not select seed 83 because of that deep score.

Selected updates are 2,700/3,000/3,000 for stop-only and 2,900/2,800/3,000 for the
auxiliary arm, in seed order 83/89/97. Seed 83 stop-only exactly reproduces the
previous baseline decisions and its recorded complete exported weight hash:
`d5987abb52e5d620213d07d36450407e3a06e0f63e818abaf2912a883b312c5b`.
The detached measurement readout did not alter that observed baseline result.

## The numerical readout improved

Mean numerical diagnostics on trained-count validation queries:

| Metric, units of steps unless a percentage | Stop only | Auxiliary |
| --- | ---: | ---: |
| Initial remaining-work MAE | 0.575 | 0.281 |
| Deviation from a one-step decrement | 0.511 | 0.146 |
| Diagnostic zero crossing at requested loop | 50.95% | 85.24% |
| First zero crossing agrees with first head stop | 49.91% | 85.07% |

The new objective affected the learned representation; this is not a result in
which an inactive auxiliary loss simply did nothing. But more accurate numerical
readout within the trained range did not yield better unfamiliar-count stopping.

The strongest new clue is error **before the first recurrent update**. Across
all 128 validation graphs, the auxiliary readout's mean initial values are:

| Requested count | Seed 83 | Seed 89 | Seed 97 |
| --- | ---: | ---: | ---: |
| 7 | 6.84 | 6.92 | 7.00 |
| 9 | 7.59 | 7.61 | 7.73 |
| 10 | 10.14 | 10.09 | 10.09 |
| 11 | 11.60 | 11.58 | 11.44 |
| 12 | 11.62 | 11.61 | 11.49 |
| 13 | 12.44 | 12.48 | 12.32 |
| 14 | 12.12 | 12.13 | 11.99 |
| 15 | 11.25 | 11.32 | 11.25 |
| 16 | 11.59 | 11.51 | 11.47 |

All 384 auxiliary count-nine initial estimates (128 graphs × three seeds) are
below 8.45, so none is even within half a step of nine. Seed 83 then stops at
seven on 41 graphs and eight on 87, exactly matching its stop-only control.
The error is already present in the learned prompt→initial-memory→numerical-readout
path; later memory decay alone cannot account for it.

For count 64, numerical traces are available on the predeclared eight-graph
sample, not all 32 deep graphs. Initial means are 12.41, 12.34, 12.46 across the
three auxiliary seeds; mean stopping loops are 13.88, 13.62, 13.38. On those
actually executed prefixes, mean absolute deviation from a unit decrement is
0.174, 0.139, 0.179 steps; predicted remaining work at the head stop averages
0.237, 0.077, 0.163. This is consistent with a rough countdown from a badly
underestimated initial value. It is not evidence of successful 64-loop counting:
those native policies stop after roughly fourteen loops.

## Interpretation and next bounded question

The regression readout is observational. Its errors do not prove that every
coordinate of memory represents the wrong number, that the frozen prelude cannot
distinguish digits, or that a scalar countdown causally drives the stop head.
The previous 99.48% prompt-count classifier was fitted on **all counts 1–16**;
it established count identity decodability, not zero-shot numerical magnitude
understanding for requested counts excluded from controller training.

The supported conclusion is narrower: the new objective improves familiar-range
numerical state and update regularity, while the learned conversion of unfamiliar
prompt counts into a useful initial numerical state remains defective. The data
do not yet isolate initializer versus numerical-readout geometry, and cannot
show whether recurrence would work for 64 steps from a correct initial state.

Keep R frozen and retain both arms as evidence. Audit optimization and objective
weighting before selecting an architectural change or broader numeric coverage.
A subsequent proposal could isolate prompt-to-initial-state numerical interpretation from subsequent recurrence.
One possible controlled follow-up is broader **initial-count-only** supervision,
without longer recurrent rollouts or clock inputs. That would explicitly change
numeric coverage: familiar count values at unfamiliar rollout lengths are a
different claim from generalizing to unseen count values. This is a proposal,
not a selected or implemented new run. Repeating the same auxiliary recipe or
adding R training is not supported by this comparison.

## Training-recipe audit

The implementation audit found no detached controller-memory recurrence, missing
auxiliary backward pass, or off-by-one target in the checked paths. The prompt
and executor observations are deliberately detached; controller initialization
and recurrent updates remain trainable. The numerical readout does not determine
the native stop decision: it and the stop classifier are separate heads sharing
controller memory. Good numerical decoding therefore need not imply good stopping.

The recipe nevertheless has unresolved weaknesses:

- Each run makes 3,000 updates of 256 examples over 18,432 tasks: 41 full passes
  plus two thirds of another. Auxiliary runs average 95.0–96.2% minibatch exact
  stopping over the final 500 updates. These are sampled pre-update minibatches,
  not a full training-set evaluation. We have neither perfect training fit nor
  the matched full-train/validation readouts needed to separate underfitting
  from overfitting precisely.
- Pre-clipping controller gradient norms exceed the threshold of one on
  94.35%, 97.01%, and 94.35% of logged auxiliary updates (301 observations per
  run). Mid-run median norms are about 11–12, with late spikes too. Trained-count
  validation remains variable: auxiliary seed 89 moves from 94.18% at update
  2,800 to 89.41% at 3,000. Frequent clipping and threshold-sensitive accuracy
  are reasons to inspect optimization, not proof that clipping caused failure
  or that increasing its threshold would help.
- The remaining-work loss is `mean_examples(mean_t<=N(((prediction-(N-t))/12)^2))`.
  Its direct initialization term receives only `1/(N+1)` of the temporal weight.
  At N=12, a one-step initialization error contributes 1/1872 (about 0.000534)
  to that example's loss if all other readouts are exact. Later losses also
  backpropagate through initialization, so this is not its total gradient.
  Over the final 500 updates, mean auxiliary losses are 0.00075–0.00079 versus
  stop losses 0.0164–0.0209. Scalar magnitudes alone cannot establish gradient
  dominance: per-objective parameter-gradient norms and alignment are not logged.
- Training requests cover only 1–6, 8, 10, and 12. Intermediate remaining-work
  targets include nine, but no training prompt requests nine. The loss permits
  fitting familiar requested counts without learning a reusable numerical
  interpretation of unfamiliar prompts. This is a deliberate generalization
  challenge, not mislabeled data.

Before another training recipe, the bounded diagnostic should evaluate a fixed
training panel alongside validation and separate initial, recurrent, terminal,
and stop-loss gradients by controller module. Saved feature vectors and weights
on the desktop permit this without updating parameters or rerunning the executor.
Those gradient measurements are not available from the pushed scalar logs;
the [audit is now implemented](../diagnostics_and_performance.md#controller-training-audit--current-desktop-command), but has not run on these pretrained artifacts. The observed initial-readout
errors locate a symptom, not a unique architectural cause.

## Audit and runtime

- All 66 distinct recorded source hashes match the checkout. Every run uses the
  same source-checkpoint and feature-cache identities. Panel/task hashes match;
  each has 2,048 distinct training graphs, with no overlap with validation tables.
- Recomputed each run's selected update from its logged validation events and
  checked all 72 paired cohort/depth aggregate rows.
- Audited all 22,272 cached decisions for exact/early/late/missing and joint
  semantics. Recreated the deep task tables from their seeds and independently
  executed all validation/deep reference paths. Targets match; every deep
  stopped prediction is correct for its *executed* loop, despite wrong timing.
  Each run has 22 validation stopped-state prediction errors on the 2,048-query
  expanded panel, separate from its timing errors.
- Audited 384,192 saved numerical trace rows across training and evaluation:
  exact N−t targets only through N, blank post-request targets, sampled first
  stop crossings, and full-panel initial/trajectory/terminal/decrement error
  aggregates and zero-crossing accuracy. Float32 aggregates match within 2e−6.
- All four native checks match replay for 16 queries each. These are bounded
  fidelity checks, not full native evaluations or latency comparisons.
- Deep correct letters at wrong times remain excluded: 216/217/217 such cases
  in the three stop-only runs and 213/213/215 in auxiliary runs.
- Six training invocations together took 475.2 seconds (7.9 minutes), including
  their cache loading, evaluation and export. Shared deep extraction/loading took
  296.2 seconds (4.9 minutes). These do not include all orchestration, native
  checks or comparison upload time and are not an end-to-end speedup measurement.
- Raw checkpoint weights and cached vectors remain on the desktop. Recorded
  weight/cache hashes are cross-checked between artifacts, not independently
  recomputed from absent tensors. No pretrained model ran during this audit.
