# Controller training audit — 2026-10-08

## Run and provenance

Executed `bash audit_controller.sh` in a detached `controller-audit` tmux session
on the desktop over SSH, from `/home/mukund/loopformer`, at commit `d79af01`.
The command completed with exit status zero. Python 3.11.8, PyTorch 2.14.0+cu130,
RTX 5070 Ti; the audit used float32 CUDA and no optimizer updates. Recorded audit
wall time is **31.19 seconds**, excluding the subsequent W&B upload.

Artifacts: `eval/pointer_diagnostics/controller-training-audit-20261008T221058.031281Z/`.
The sibling launcher log is
`eval/pointer_diagnostics/controller-training-audit-20261008T221055Z-2362.log`.
[W&B run](https://wandb.ai/mukunds/loopformer/runs/lsjjgxfd).
Summaries, gradient tables and the log are committed. The 85 MiB full
`predictions.csv` remains on the desktop and in the W&B evaluation artifact;
it is not included in the Git commit. No weights were copied into Git.

All six runs were inspected at best and final checkpoints. Fit covers all 18,432
training and 2,048 validation queries. Gradient measurements use seed 107,
32 graphs per split with all count variants, batch 64. Validation extrapolation
here means requested counts **13–16**, not the separate deep 13–64 panel.
The [audit guide](../diagnostics_and_performance.md#controller-training-audit--current-desktop-command)
defines loss normalization, counterfactual control gradients and scoring.

## Full-panel fitting

Selected checkpoints, exact stopping:

| Seed | Arm | Training | Validation, trained counts |
| --- | --- | ---: | ---: |
| 83 | Stop only | 97.06% | 95.31% |
| 83 | Remaining work | 96.71% | 95.31% |
| 89 | Stop only | 97.48% | 95.57% |
| 89 | Remaining work | 95.65% | 94.18% |
| 97 | Stop only | 97.94% | 95.75% |
| 97 | Remaining work | 97.44% | 96.44% |

The graph train/validation gap is modest (about 1.0–2.2 percentage points), while
neither training fit nor count generalization is solved. This does not support a
large graph-memorization gap as the main explanation for the count failures.
It does not rule out learning count-specific behavior on familiar counts.

Seed 89's auxiliary run falls from 95.65% training / 94.18% validation at its
selected checkpoint to 90.60% / 89.41% at the final checkpoint. Both panels get
worse; this is not merely validation-only deterioration. The cause of the update
sensitivity remains unresolved; no learning-rate or clipping change was made.

## Objective gradients

For the selected auxiliary checkpoints, on the fixed training gradient panel:

| Seed | Stop gradient norm | Remaining gradient norm | Stop/remaining cosine |
| --- | ---: | ---: | ---: |
| 83 | 0.4643 | 0.1068 | +0.326 |
| 89 | 0.2914 | 0.0714 | −0.328 |
| 97 | 0.6288 | 0.0566 | +0.708 |

These are norms/cosines of the **panel-mean controller parameter gradients**,
not mean minibatch norms or Adam updates. Remaining-work gradients are about
9–24% of stop-gradient magnitude at these checkpoints. They are present, and
are not consistently opposed to stopping across seeds. This weakens a blanket
claim that the auxiliary signal is absent or always in conflict. It does not
establish ideal loss weights, historical gradient behavior, or a causal account
of optimization failure. Module and per-batch measurements are preserved in CSV.

## Numerical readout versus stopping

On held-out counts 7/9/11, auxiliary exact stopping conditioned on an initial
estimate within half a step is 89.94% (159 eligible examples), 89.82% (167), and
85.71% (189). These subsets are easier cases selected by readout correctness;
this association is not a causal intervention or an all-count success rate.

At requested count nine, **none of the 128 validation initial estimates in any
auxiliary seed is within half a step**, and all three heads still have zero exact
stops. Numerical zero crossing is correctly timed on only 3, 2, and 0 examples,
respectively. The poor count-nine result therefore cannot be explained solely
by an otherwise correct numerical countdown feeding an incorrect stop decision.
The readout itself remains a measurement, not the deployed controller mechanism.

## Checks and consequence

The audit verifies saved best/readout alignment, bounded baseline cache replay,
selected validation metric agreement, unchanged in-memory weights, untouched
parameter gradient buffers and unchanged input-file hashes. Local review matched
all recorded source hashes and checked that each of the 1,044 numerical/stop
cross-tabs contains four cells summing to its stated cohort size.

There is no evidence here of a missing auxiliary backward pass or a large
train-versus-validation graph gap. There is evidence of residual fitting error,
late optimization sensitivity, and count-specific numerical-readout failure.
These are different issues; the data do not justify declaring one universal
cause. Preserve R and the current checkpoints. No new training recipe, broader
numeric coverage or confirmation experiment was launched from this audit.
