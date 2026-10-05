# Frozen controller diagnosis — seed-61 executor

Reviewed 2026-10-04 (America/New_York). Desktop artifacts use the UTC timestamp
2026-10-05T01:54:05Z. Artifact commit `bdd4c7c`; diagnostic implementation `9282d01`.

## Run and audit scope

Command: `bash diagnose_controller.sh`, using the selected executor checkpoint
`models/stage1_pointer/executor_r-seed61/step-002250`. Exact CLI arguments and
settings are in the saved summary. The full diagnostic took **88.87 seconds**
before retrospective W&B upload.

Artifacts: [`controller-executor_r-seed61-20261005T015405Z-542/`](../../eval/pointer_diagnostics/controller-executor_r-seed61-20261005T015405Z-542/).
W&B: [completed diagnostic](https://wandb.ai/mukunds/loopformer/runs/uc10o0qt).
The local artifact files were audited; the W&B API was not separately queried.

The 64 development graphs were partitioned into 32 probe-fit, eight probe-selection
and 24 held-out diagnostic graphs. Each appears at every requested count 1–16;
all queries execute 16 loops. Controller fits use only eight fit graphs at counts
1–6, 8, 10, 12: **72 questions**. Probe classifiers see all 16 count classes;
controller fits do not see held-out counts 7/9/11 or 13–16 as training requests.
The reserved confirmation/test split remains unused.

Audit checks passed:

- All 68 recorded source hashes and the tasks file hash match locally.
- Selected checkpoint and data hashes match the preceding executor evaluation.
- Graph partitions are disjoint and contain 64 unique rule tables.
- All 3,072 stopping decisions agree with the first >=0.5 probability crossing
  in 49,152 exported loop rows. Correct-answer and exact-stop flags, joint-success
  flags and summary cohort counts/rates were recomputed independently.
- All nominal targets match direct execution of the saved rule tables.
- All 51 probe aggregate rows agree with their exported confusion counts.
- The desktop recorded zero maximum live-versus-cached controller logit error.

Weights and raw feature vectors remain on the desktop. The local audit validates
saved evidence; it does not rerun pretrained feature extraction or the probes.

## Count information is initially accessible, then becomes poorly decodable

Held-out-graph accuracy of the **linear** classifier predicting requested count:

| Representation from the original controller | Correct count / 384 | Accuracy |
| --- | ---: | ---: |
| Prelude vector supplied to controller | 382 | 99.48% |
| Initialized controller memory | 367 | 95.57% |
| Memory after loop 1 | 197 | 51.30% |
| Memory after loop 4 | 45 | 11.72% |
| Memory after loop 8 | 29 | 7.55% |
| Memory after loop 12 | 24 | 6.25% |
| Memory after loop 16 | 24 | 6.25% |

Chance is 6.25% for 16 balanced count classes. The shuffled-fit-label prompt
probe scores 4.95%. Separate MLP probes corroborate the decline: 96.09% from
initialized memory, 46.09% after loop 1, 13.80% after loop 4, and 6.25% by loop 16.
Elapsed-loop decoding reaches 35.68% with a linear probe and 43.65% with an MLP;
progress information exists but this does not establish a reliable internal clock.

The stop output also becomes insensitive to requested count. For each graph,
take the max-minus-min stop probability over the 16 requested counts at a fixed
loop, then average across graphs. For the original controller this range falls
from 0.1325 at loop 1 to 0.0366 at loop 4, 0.00470 at loop 8, 0.000690 at loop 12
and 0.000104 at loop 16.

This rules against the simple explanation that the frozen prompt representation
never provides accessible count information. It supports investigating loss of
usable request information during the learned controller updates. It does **not**
prove that all count information is erased: probes can miss representations, and
these balanced trajectories include post-request loops the stop loss did not
supervise. Probe success is not evidence of causal use. The memory probes examine
the original controller, not the two newly fitted copies.

## The existing controller architecture can learn exact timing on a small set

Both 1,000-update controller-only fits reach **72/72 exact stops** on their fit
questions, with the executor frozen. Fit BCE ends at 0.000206 for continued
weights and 0.000119 for fresh weights. Both select update 1,000 by fit loss.
Continued fitting is unstable mid-run: exact stopping first reaches 100% at a
logged update 75, later drops, and is only 27.78% at update 600 before recovering.
Do not treat this diagnostic learning rate as a validated production recipe.

| Exact stopping cohort | Original | Continued tiny fit | Fresh tiny fit |
| --- | ---: | ---: | ---: |
| Fit questions: 8 graphs × 9 trained counts | 11/72 (15.28%) | 72/72 (100%) | 72/72 (100%) |
| Held-out graphs, trained counts | 31/216 (14.35%) | 144/216 (66.67%) | 135/216 (62.50%) |
| Held-out graphs, unseen counts 7/9/11 | 0/72 | 24/72 (33.33%) | 27/72 (37.50%) |
| Held-out graphs, unseen counts 13–16 | 0/96 | 11/96 (11.46%) | 16/96 (16.67%) |

The held-out cohort is the same 24 graphs in every row. Exact stopping and joint
success coincide in these cohorts; they do not coincide everywhere. Neither tiny
fit is a successful extrapolating controller. For example, continued fitting gets
24/24 held-out graphs right at requested counts 1 and 2, but only 9/24 at count 12,
zero at 15, and 1/24 at 16. Aggregated trained-count accuracy hides that decline.

Cyclic answer coincidences remain excluded from timing success. Across all 1,024
questions, correct answers at the wrong time occur 148 times for the original
controller, 72 for the continued fit, and 71 for the fresh fit. None is counted
as exact stopping or joint success. A missing stop at the safety cap is likewise
not exact success.

The fits establish that the architecture, feature interface and existing stop
labels can support learning these 72 examples without an explicit count/progress
input. They do not isolate the original recipe's failure: the diagnostic changes
learning rate (0.001 versus the original scheduled 0.00002 peak), data repetition,
optimizer state, weight decay, and supervision scale, while freezing the executor
features that previously changed during joint training. It is therefore too strong
to blame just the learning rate, GRU design, or training-set size.

## Next bounded recommendation

Keep the successful executor frozen. Train the existing controller separately on
a broader set of independent **training** graphs with cached frozen features and
its own optimizer/checkpoint selection. First retain the original trained counts
so we can distinguish improved trained-count generalization from merely expanding
the count range. Select using graph-disjoint development exact stopping and stop
loss, retaining the original controller baseline. Monitor per-count timing errors
and count accessibility in the newly trained memory, not just aggregate BCE.

This is a recommendation, not an implemented production controller-training
launcher. No R retraining, parsed numeric counter, new architecture or additional
pretrained sweep was performed in this review. Larger requested-count training
and changes such as persistent prompt conditioning remain subsequent choices;
these results do not yet establish that either is necessary. Keep the diagnostic
head binaries as evidence rather than promoting them automatically.
