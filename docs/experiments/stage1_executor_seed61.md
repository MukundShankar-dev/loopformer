# Isolated executor, seed 61: first pretrained result

Audited 2026-10-04 after pulling artifact commit `86619b0`; training/evaluation
source commit `d2eae3a`. This is development evidence, not independent confirmation.

## Configuration and artifacts

The desktop ran `bash prepare_executor.sh`, `bash train_executor.sh`, and
`bash eval_executor.sh`. Training used `configs/executor_r.json`: pretrained
Qwen2.5-0.5B-Instruct, all R weights trainable, isolated recurrent controller,
normalized re-entry bridge, auxiliary direct R supervision, fixed prompt memory,
BF16/SDPA, batch 4 with accumulation 4, and one epoch (2,250 updates). Training
counts were 1–6, 8, 10, 12. Selection used trained-count validation loss, not deep
results. The selected checkpoint was the final step 2,250.

- Training: [`models/stage1_pointer/executor_r-seed61/`](../../models/stage1_pointer/executor_r-seed61/).
- Evaluation: [`eval/pointer_diagnostics/executor_r-seed61-20261004T231253Z-799/`](../../eval/pointer_diagnostics/executor_r-seed61-20261004T231253Z-799/).
- Frozen checkpoint SHA-256 recorded by the evaluator: `5a53372639924222266e457e1b1c586aeaa8db7340192c449cb7f8fd288c4e95`.

No pretrained weights are available in this Mac checkout. The audit checks saved
outputs, source, reference execution and reproducibility; it does not independently
rerun model inference. The omitted 124 MB training terminal log is unnecessary for
these checks: structured training metrics and all five completed evaluations exist.

## What the numbers measure

| Evaluation | Unique graph/start pairs | Complete nominal trajectories | Final answer at requested loop |
| --- | ---: | ---: | ---: |
| Validation, requested depths 1–12 | 128 | 1,522/1,536 (99.09%) | 99.09% |
| Deep split, requested depths 13–64 | 32 | 1,664/1,664 (100%) | 100% |
| Matched validation panel, depth 12 | 32 | 32/32 (100%) | 100% |
| Same panel, depth 64 | Same 32 | 31/32 (96.875%) | 96.875% |

The deep split repeats the same 32 graph/start pairs at 52 requested counts.
There are 1,664 queries, not 1,664 independent graph trials. Validation similarly
repeats 128 graphs at 12 counts. Two validation graphs fail within 12 loops:
one first fails at loop 3 (ten failing requested horizons), another at loop 9
(four failing horizons). That accounts for all 14 nominal validation failures.
The smaller, different deep cohort contains neither graph. Perfect accuracy there
therefore does not imply that greater depth improves execution.

Existing fixed-budget validation traces already contain 64 loops for every
query. Independently extending each rule table's reference trajectory finds
**125/128 validation graphs correct throughout all 64 loops**. Three graphs
fail, first at loops 3, 9 and 14 in the batch-16 BF16 evaluation. This extension
measures continuing pointer execution, not retention of the requested final answer.

The matched panel includes the third graph and detects a first C-readout error
at loop 13 in all three precision modes; its direct R head is still correct at
that loop. Its longer requested horizons all fail. The panel uses batch 4 and
longer padded inputs than the full validation evaluation, so this one-loop
first-error difference is a remaining batching/input-shape sensitivity, not a
controlled attribution to precision or requested count. Within the matched
panel, changing requested count produces zero common-prefix prediction changes.

## Audit of the perfect deep result

The audit regenerated train, validation and depth-test records in memory using
the exact `prepare_executor.sh` configuration and the locally cached pinned
tokenizer. All three byte hashes match the saved run metadata:

| Split | SHA-256 |
| --- | --- |
| Train | `ea70dfb3e4b8822a7c889a949ba82a22466cbd303af14d8f0be9294bd31667fb` |
| Validation | `5ab3c4ad9dbd0390af9e2597b8dbebbf777543668207e2a93b3b08da12605c4c` |
| Depth test | `d7161f0b19dfbf0b48576b69a2486fecceefd900594afca8a97b032b8bf8279e` |

There are 36,000 unique training tables and **zero table overlaps** between those
three splits, ignoring rule order, start and depth. The reserved test split was
not examined. All 65 distinct source files covered by the saved training/eval
hashes match the current checkout.

A separate audit reparsed Rules and Start from the prompts, executed ordinary
Python dictionary transitions, and compared each CSV prediction and correctness
flag against that reference. On the deep split:

- All **64,064 nominal intermediate predictions** are correct, both from C and
  from the direct R head. This is unconditional accuracy, not a risk-set metric
  that excludes already-failed examples.
- All **106,496 executed C readouts**, including loops beyond shorter requests,
  also match the continuing reference trajectory. These repetitions are correlated.
- The smallest nominal C target margin is 4.625; ties do not explain accuracy.
- The 32 graphs include 11 full 26-state cycles, 15 random permutations and six
  random functions. Only two start-state orbits have period one. Constant-output
  fixed points cannot explain the full result.
- In the matched panel, all 10,496 C predictions agree across FP32/eager,
  FP32/SDPA and BF16/SDPA. Margins differ; nominal maximum BF16 margin change is
  3.4204. This precision result applies to that panel, not every possible input.

Source inspection confirms that the evaluator passes prompt tokens, attention
masks and an external loop budget to the model. Targets are used afterward for
scoring; reference states and decoded symbols are never recurrent inputs.
The recurrent state remains continuous. No target-leakage or perfect-score
aggregation error was found within this audit's scope.

## Stopping remains unsuccessful

| Actual learned stopping | Validation | Deep split |
| --- | ---: | ---: |
| Correct stopped answer | 376/1,536 (24.48%) | 232/1,664 (13.94%) |
| Exact requested stopping time | 165/1,536 (10.74%) | 0/1,664 |
| Correct answer and exact stop | 164/1,536 (10.68%) | 0/1,664 |

Deep examples stop at loop 3 (100 queries) or 4 (1,564 queries), despite requests
for 13–64 transitions. Validation stops at 1 (30), 3 (377), or 4 (1,129). Correct
stopped letters can arise from cyclic coincidences, so they do not imply counting.
The stopped evaluator's `complete_trajectory_accuracy` checks whether the full
nominal prefix was produced; it can count an overlong run and is not exact-stop
accuracy. Use `completion.joint_success_rate` for the joint task.

This is an executor improvement, **not a working end-to-end autonomous solver**.
The full-loop evaluator supplies a 64-loop budget and reads the answer at the
requested loop. The controller still fails even within the training range.
Its exact-stop validation signal remains poor throughout training while pointer
accuracy improves sharply. The result does not establish whether count information
is insufficient in the frozen prelude feature, lost by the GRU, or poorly learned
under the controller objective/optimization. Those remain diagnostic alternatives.

## Training and resources

Startup gates pass: first-pass logit error zero; reused/dense prefix max logit
error 3.81e-6; both forbidden cross-component gradient paths absent. There are
180,108,059 trainable parameters. Validation trajectory accuracy on the trainer's
768-query subset rises from 3.26% at initialization to 41.02% at update 1,000,
94.53% at 1,250 and 99.22% at 2,250. This learning progression is inconsistent
with an evaluator that simply always reports gold predictions.

Saved wall time is 7,673 seconds (127.9 minutes), of which 7,334 seconds is training
and 287 seconds validation. Peak allocated CUDA memory is 4.43 GiB; median
per-update peak is 4.42 GiB. This is allocation evidence, not GPU utilization or
proof of a particular speedup opportunity. Checkpoint selection is based on executor
loss; it does not promise the best controller.

## Interpretation and next bounded work

The evidence supports a substantial improvement in held-out pointer execution,
including beyond the maximum trained loop count. It does not prove arbitrary-depth
correctness: these are 26-state graphs, long paths repeat, and the independent
deep cohort contains only 32 graphs. Multiple architecture, capacity, objective,
data and optimization changes were enabled together; none can receive sole causal
credit without ablations.

Retain this checkpoint. Before changing or retraining R, strengthen development
evidence with more independent graphs at fixed long horizons, while retaining
strict trajectories and cycle strata. Separately diagnose the controller's access
to and retention of the requested count using the frozen executor. Keep the public
prompt-only interface and no explicit numeric loop/count features. The current
artifacts justify isolating the remaining stopping problem rather than assuming R
still has the previous catastrophic depth failure. Independent confirmation and
multi-seed training remain unrun.
