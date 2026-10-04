# Fixed-prompt paired Steps probe

Reviewed 2026-10-04 while the user runs the new depth-12 experiment. No training configuration or model code changed in this review.

## Protocol and audit

Command: `bash probe_steps.sh`. Checkpoint: `models/stage1_pointer/depth6-fixed-prompt-seed37/step-003250`. Artifacts: `eval/pointer_probes/paired-steps-20261004T150542Z-407/` (`summary.json`, `pairs.csv`, `trajectories.csv`, `examples.csv`). Source data are the first 32 depth-16 tasks in seed-17 development depth_test. Each mapping/start is reused with Steps 6, 7, 8, 9, 10, 12, and 16, and evaluated for 20 forced loops. Stop timing is replayed at threshold 0.5, not measured stopped inference.

All 4,480 rows cover exactly 32 maps × seven requested counts × 20 loops. Nominal targets and correctness were checked against source data, all 224 paired summaries were recomputed, and dataset/source hashes match local files (remapping the desktop absolute script path to this repository). All 640 Steps-16 predictions match their corresponding previous full evaluation. The checkpoint hash agrees with the prior evaluation; adapter binaries remain absent locally and were not independently hashed. Paired comparisons share graphs, starts, seeds, target prefixes, and checkpoint weights; only the requested-count text/length and associated scoring horizon change.

## Stopping fails to track larger requested counts

| Requested Steps | Complete nominal trajectories / 32 | First stop loops |
| --- | ---: | --- |
| 6 | 32 | all 32 at 6 |
| 7 | 31 | all 32 at 6 |
| 8 | 29 | all 32 at 6 |
| 9 | 21 | all 32 at 6 |
| 10 | 2 | 7 at 5; 25 at 6 |
| 12 | 0 | 16 at 5; 16 at 6 |
| 16 | 0 | 13 at 5; 19 at 6 |

This directly controls graph composition and supports failure to extrapolate stopping beyond six. Larger requested values sometimes cause earlier stopping. It does not show that the model ignores Steps entirely: head logits and predictions change with the text. The supervision-support explanation from the [earlier investigation](stage1_fixed_prompt_review.md) remains plausible: no training continue labels existed at loop six or beyond.

## Requested count causally affects late pointer predictions

At loop 9, the reference pointer is identical for each paired mapping/start:

| Requested Steps | Correct loop-nine predictions / 32 |
| --- | ---: |
| 9 | 21 |
| 10 | 16 |
| 12 | 15 |
| 16 | 11 |

Changing only Steps 9→16 loses ten correct predictions and gains none. Steps 9→10 loses five and gains none. Steps 10→16 retains the same two-digit suffix width yet reduces aggregate accuracy from 16/32 to 11/32; therefore suffix token count alone cannot account for the whole difference. These small-sample results identify a causal effect of the requested-count text on model execution, not which representation, attention path, or numerical feature mediates it.

At loop 6 all 32 Steps-9 and Steps-16 variants are correct. Sensitivity grows later. The original all-depth comparison was confounded by changing graphs; this paired result removes that confound. H remains a side readout during forced execution, so the decision to stop cannot itself cause these forced-run errors.

## Working-state evolution continues after decoded failure

For the Steps-16 variants, medians across the 32 maps are:

| Loop | Working-state norm | Update norm | Cosine with preceding state |
| --- | ---: | ---: | ---: |
| 6 | 35.0 | 21.7 | 0.808 |
| 8 | 39.4 | 15.5 | 0.924 |
| 9 | 43.2 | 10.6 | 0.973 |
| 10 | 47.9 | 8.1 | 0.990 |
| 16 | 86.6 | 7.8 | 0.999 |

The raw state is not stationary: its norm and absolute updates remain substantial. Increasing alignment and declining relative update size are consistent with drift toward a dominant direction. These scalars cannot establish an attractor, norm-induced failure, or loss of active-rule access; increasing norm itself contributes to high adjacent-state cosine. Full vectors/interventions would be needed for those claims. Similar trends appear under the shorter requested counts, so larger-number conditioning is not the only plausible contributor to the execution horizon.

## Consequence for the running experiment

Let the already-running depth-12 experiment finish. It addresses the measured supervision gap with continue labels beyond six and includes two-digit requested counts. Held-out counts 7/9/11 test whether stopping depends on more than memorized requested values; depths 13–20 remain true extrapolation. No outcome is guaranteed, and BF16/SDPA/bucketing changes make this a combined recipe change rather than an architecture-only causal ablation.

After `bash eval_depth12.sh`, inspect actual joint exact-stop-and-answer success for 7/9/11 alongside forced trajectories. If those improve but depths 13–20 fail, the new recipe has achieved useful interpolation while farther extrapolation remains open. If forced execution succeeds but stopping does not, the remaining gap is localized more strongly to progress representation/readout. Do not interrupt or alter the current run based on this probe.
