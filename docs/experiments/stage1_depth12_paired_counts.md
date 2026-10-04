# Depth-12 paired-count probe — 2026-10-04

The user ran `bash probe_depth12.sh`. Artifacts: `eval/pointer_probes/depth12-paired-20261004T194923Z-5303/`. Steps 5,000 and 7,500 each evaluated the same 32 depth-20 source mappings with requested counts 12/14/16/18/20, batch 8, 24 forced loops, float32, fixed prompt memory. Only the displayed count and corresponding reference horizon change. The stop head cannot truncate execution.

W&B API verification found both runs finished: [step 5,000](https://wandb.ai/mukunds/loopformer/runs/p7ttylit) and [step 7,500](https://wandb.ai/mukunds/loopformer/runs/9owhtbxq). Aggregate trajectory accuracy matches local summaries (35% and 20.625%, respectively, across the five equally sized count groups). Those aggregates mix horizons; use the depth-specific results below. Evaluation artifacts and example/pair tables are present remotely.

## Controlled evidence

Correct readout at loop 14 on identical maps (32 per count):

| Checkpoint | Steps 14 | Steps 16 | Steps 18 | Steps 20 |
| --- | ---: | ---: | ---: | ---: |
| 5,000 | 24 | 25 | 30 | 8 |
| 7,500 | 3 | 3 | 6 | 2 |

For checkpoint 5,000, changing Steps 18→20 loses 22 correct loop-14 outputs and gains none. Steps 14→20 loses 16 and gains none. Mapping, start, loop index and exact intermediate target are unchanged; both counts have two digits. This directly demonstrates requested-count sensitivity of execution on this cohort, not merely differing map difficulty or digit count. It does not identify which internal path mediates that sensitivity or establish a monotonic relationship with numerical depth.

Complete nominal trajectories:

| Checkpoint | Steps 12 | Steps 14 | Steps 16 | Steps 18 | Steps 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 32/32 | 23/32 | 1/32 | 0/32 | 0/32 |
| 7,500 | 30/32 | 3/32 | 0/32 | 0/32 | 0/32 |

Count sensitivity is not the whole failure: even Steps 18 falls from 30/32 correct loop-14 readouts to 14/32 at loop 15 and 1/32 at loop 16. Later training also degrades execution across all tested counts. Correct readout at a later loop does not imply an entirely correct prefix.

Working-state changes accompany the failures. At checkpoint 5,000 and loop 14, median norm/update norm/adjacent cosine are 45.88/16.11/0.937 for Steps 18 versus 50.54/10.14/0.980 for Steps 20. At checkpoint 7,500, Steps 18 is 53.98/11.07/0.980. These are correlations consistent with increasingly aligned, growing states; they do not prove a stationary attractor or establish norm growth as the cause.

## Audit and next decision

All 7,680 trace rows were checked for complete 24-loop coverage; nominal targets and correctness match exact reference execution. The 160 exact task variants match across checkpoints and their hashes match the summaries. Source hashes match the current implementation; W&B-upload file hashes match local files. Source dataset and checkpoint binaries are absent locally and were not independently rehashed.

Do not continue unchanged training or describe R as a count-independent transition function. A targeted candidate is paired-count training/consistency on common nominal readouts: the same mapping/start should yield the same intermediate symbols for different requested horizons. Keep the original prompt and exact intermediate targets; do not force identical hidden states, because the completion head still needs count information. This is a proposal, not an implemented or selected recipe, and it does not by itself constrain states beyond the training horizon. Isolating the completion loss's gradient effect remains another unresolved question. No new run is launched or confirmation split opened by this review.
