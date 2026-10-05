# Seed-61 controller-only training and native stopping

Reviewed artifact commit `5b1cad2` (run source `4877638`). The desktop completed
`bash train_controller.sh && bash eval_controller.sh`. These are development
results; the reserved test split and seed 29 remain untouched.

## Configuration and evidence

The unchanged executor is `executor_r-seed61/step-002250`. Only the existing
332,417-parameter recurrent controller is optimized, initialized from that
checkpoint. Training pairs 2,048 training graphs with counts 1–6, 8, 10, 12
(18,432 queries); validation pairs 128 different graphs with counts 1–16.
The recipe is [controller_seed61.json](../../configs/controller_seed61.json):
3,000 AdamW updates, batch 256, peak LR 0.0003, warmup/cosine decay, per-loop
continue/stop BCE. Count and loop labels enter the objective, not the forward.
Selection uses trained-count validation exact stopping, with BCE as tie-breaker.

- [Training artifacts](../../models/stage1_pointer/controller-seed61/).
- [Native stopping and diagnostics](../../eval/pointer_diagnostics/controller-eval-20261005T024736Z-791/).
- Source weights SHA-256: `5a53372639924222266e457e1b1c586aeaa8db7340192c449cb7f8fd288c4e95`.
- Exported best weights SHA-256: `d5987abb52e5d620213d07d36450407e3a06e0f63e818abaf2912a883b312c5b`.

Update 2,700 wins the specified selection rule. Cached feature extraction took
858.0 seconds; controller updates plus export took 49.3 seconds; total 908.4
seconds (15.1 minutes). The training-phase peak allocation was 932.1 MiB; this
excludes the frozen-model extraction peak. The cache contains 986,611,712 tensor
bytes. Native batch-one inference took 387.0 seconds on validation and 743.8 on
the deep split, with another 128.2 seconds for diagnostics before upload.

## Exact timing improved, count generalization remains uneven

These numbers come from actual native stopped inference, threshold 0.5, cap 64.
Success requires both the requested stopping time and the correct letter.

| Requested counts / split | Exact stops | Correct answer at exact stop |
| --- | ---: | ---: |
| Trained counts, validation | 1,098/1,152 (95.31%) | 1,089/1,152 (94.53%) |
| Held-out 7, 9, 11, validation | 176/384 (45.83%) | 175/384 (45.57%) |
| All 1–12, validation | 1,274/1,536 (82.94%) | 1,264/1,536 (82.29%) |
| 13–64, deep split | 28/1,664 (1.68%) | 28/1,664 (1.68%) |

The matched cached baseline scored 14.32% exact stopping on trained counts;
it now scores 95.31%. Native validation exact stopping rises from the prior
10.74% to 82.94% across all counts 1–12. This improvement follows controller-only
training; it does not require changing the executor.

The aggregate interpolation number hides substantial count-specific failures:

| Count | Validation exact stops | Observed errors |
| --- | ---: | --- |
| 7 | 109/128 (85.16%) | 16 stop at 6; three at 8 |
| 9 | 0/128 | 87 stop at 8; 41 at 7 |
| 11 | 67/128 (52.34%) | 60 stop at 12; one at 10 |
| 13 | 96/128 (75.00%), cached extension | 31 stop early; one late |
| 14–16 | 0/384, cached extension | All early |

On the separate 32-graph deep cohort, count 13 succeeds on 28/32 (87.5%).
Every count 14–64 scores zero exact stops. Across all deep requests, the head
stops between loops 10 and 15. This is limited one-step extension, not a general
counting or stopping algorithm. The difference between 75% and 87.5% at count 13
comes from different graph cohorts, not two measurements of the same examples.

Cyclic coincidences still matter: native deep letter accuracy is 244/1,664
(14.66%), but 216 of those correct letters occur at the wrong loop and are
excluded from joint success. Validation similarly excludes 35 wrong-time correct
letters. There were no cap fallbacks in either native evaluation.

## Memory probes changed materially

The same diagnostic panel and probe procedure compare the old and new controller.
These classifiers are fitted on all counts 1–16, with separate probe graph
partitions. Their scores measure decodability, not the stopping head's zero-shot
numeric generalization or causal use of the decoded representation.

| Probe, held-out probe graphs | Before | After |
| --- | ---: | ---: |
| Prompt count, linear | 99.48% | 99.48% |
| Initial memory count, linear | 95.57% | 90.62% |
| Memory count after loop 1, linear | 51.30% | 80.47% |
| Memory count after loop 8, linear | 7.55% | 73.96% |
| Memory count after loop 12, linear | 6.25% | 67.71% |
| Memory count after loop 16, linear | 6.25% | 53.12% |
| Elapsed loop from memory, MLP | 43.65% | 83.07% |

Count information no longer immediately becomes nearly undecodable. On the
24 probe-holdout graphs, count 9 is linearly decoded correctly in 24/24 prompt
features and 14/24 loop-eight memories, while exact count-nine stopping remains
zero across the full validation cohort. This argues against missing initial
count information as a sufficient explanation. It does not prove a particular
internal counter or identify which controller operation fails.

Continuing the diagnostic tiny fit is not the next production candidate: it
reduces trained-count probe-holdout exact stopping from 93.06% to 74.07%, despite
perfect fitting on its eight training graphs. Fresh tiny fitting gets 62.50%.
These are separate diagnostic copies, not changes to the selected checkpoint.
All probe graphs belong to the production validation pool, so they are not
independent confirmation of a validation-selected production controller.

## Audit and limits

- All 70 distinct recorded source hashes match this checkout. Training-panel and
  validation-task byte hashes match; the panel has 2,048 unique tables with all
  nine trained counts and no overlap with the 128 validation tables.
- Recomputed the selection rule from all 31 validation events: update 2,700 is
  correct. Final update 3,000 has lower BCE but slightly worse exact timing
  (94.97% versus 95.31%). Unseen-count metrics did not select the checkpoint.
- Recreated native validation/deep tasks deterministically in memory and executed
  their dictionaries independently. Audited all 3,200 first-threshold decisions,
  31,518 actual loop rows, gold states, timing flags, and aggregate exact/joint
  scores. All 1,536 native validation stops and letters match the cached best.
- On deep queries all 21,669 executed pointer predictions are correct for their
  actual loop. Their poor joint accuracy is stopping failure, not transition
  failure on these observed prefixes. Native validation has 9,784/9,849 correct
  continuing-reference predictions; unlike joint success, this is loop-weighted.
- Audited all 3,072 diagnostic decisions and 49,152 readouts. All diagnostic
  pointer predictions match the previous frozen-executor diagnostic, including
  its separate controller conditions. All 51 probe aggregate accuracies match
  exported confusion counts to floating-point precision.
- Local pretrained weights and cached vectors are absent. The export code and
  tests preserve every non-controller tensor, and artifact identities/observed
  predictions agree; the audit cannot independently compare the desktop tensors.
- The deep set still repeats 32 graphs at 52 horizons. It does not test 1,664
  independent graphs or nonrepeating 50-step paths. No new model run occurred here.

## Interpretation and next decision

Separate optimization and broader graph coverage fixed much of the trained-count
stopping problem while leaving R intact. They did not establish a reusable
counting rule: failing at nine while succeeding at ten and twelve rules out a
simple maximum-recurrence-length explanation by itself. Count-dependent learned
timing is consistent with the behavior; the exact internal computation is not
identified by these observations.

Retain the selected controller and frozen executor. Further executor training is
not indicated by these stopping results. Before another run, specify a controller
learning change that tests composition of requested count and internally tracked
progress, with the same held-out counts and no externally supplied clock. More
unchanged updates are not established as the solution: the final checkpoint's
lower trained BCE does not improve exact stopping or extrapolation. This report
selects no new architecture, training run, or confirmation evaluation.
