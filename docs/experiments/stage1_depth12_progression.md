# Depth-12 checkpoint progression — 2026-10-04

## Configuration and evidence

Training: `models/stage1_pointer/depth12-fixed-prompt-seed47-gaps`, fresh fixed-prompt recurrence with joint completion loss, requested training counts 1–6/8/10/12, 30k examples, BF16/SDPA, batch 4, one epoch/7,500 updates, constant learning rate 0.0002 after ten-step warmup. Selection on trained-count validation pointer CE chose step 5,000. Counts 7/9/11 were excluded from selection. Seed 29 and the confirmation/test split remain untouched.

The user ran `bash compare_depth12.sh`. Results are in `eval/pointer_loops/depth12-checkpoint-comparison-20261004T191905Z-1322/`: four full-loop evaluations on the same 1,000 seed-47 depth-13–20 examples, float32/eager, batch 16, 24 forced loops. Learned stopping cannot truncate these traces. Each evaluation took approximately 112–114 seconds excluding loading.

## Results

Complete nominal trajectories (all intermediate predictions correct):

| Checkpoint | Depth 13 | 14 | 15 | 16 | 17 | 18–20 | Total / 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,500 | 90.4% | 61.6% | 8.8% | 0.8% | 0% | 0% | 202 |
| 5,000 | 94.4% | 75.2% | 16.8% | 3.2% | 0.8% | 0% | 238 |
| 5,500 | 91.2% | 58.4% | 5.6% | 0.8% | 0.8% | 0% | 196 |
| 7,500 | 47.2% | 3.2% | 0% | 0% | 0% | 0% | 63 |

Relative to step 5,000, step 5,500 gains 22 complete trajectories but loses 64; step 7,500 gains two and loses 177. Mean correct-prefix length falls from 13.397 to 12.297. The earlier 4,500 checkpoint has a slightly longer mean prefix (13.437) despite fewer complete trajectories, so these two metrics must not be conflated.

Among examples correct at every preceding loop, step 5,000 has 896/948 correct transitions at loop 13 (94.5%) and 586/778 at loop 14 (75.3%). Step 7,500 has 502/920 (54.6%) and 38/443 (8.6%). These pooled risk sets include only requested depths at least the loop index; they are not fixed cohorts across loops. The exported depth-specific denominators should accompany interpretation.

Training's 32-example depth-12 validation subset also deteriorates from 100% at step 5,000 to 84.4% at 7,500. Trained-count selection CE increases from 0.0291 to 0.0911. The learning rate does not decay. This supports regression under the current recipe, not a diagnosis of overfitting versus optimization instability or an assertion that lower learning rate will repair extrapolation.

## Audit and limitations

The shared comparison aggregator reproduces matched identities/targets, 4,000 paired records, depth accuracy and conditional-risk metrics. All recorded evaluator sources match the comparison implementation at commit `2b6247f`; the training-file differences against the current checkout come from the later W&B instrumentation. All 24,000 step-5,000 predictions and IDs match its previous evaluation. Dataset files and weight binaries are not present locally, so their contents were not independently rehashed. This is one training seed and development analysis, not a confirmation gate.

## Next bounded diagnostic

Do not extend the same training recipe by another epoch on this evidence. Preserve step 5,000 as the selected reference. Run `bash probe_depth12.sh` to compare steps 5,000 and 7,500 on 32 identical depth-20 source mappings with requested counts 12, 14, 16, 18, 20 and 24 forced loops. All displayed counts have two digits. No oracle symbols, refreshed prompts or progress counters enter recurrence. This is an evaluation intervention, not a proposed inference procedure.

Compare predictions on common nominal prefixes: changing Steps alone can then be assigned responsibility for any paired change, without differences in mapping/start or digit count. Inspect late conditional errors alongside working-state norm/update/cosine traces. Hidden-state scalar correlations cannot prove that drift causes failure. Similar failure across count variants would weaken the count-specific explanation on this cohort but would not uniquely identify the alternative. Only after this comparison choose a controlled training intervention; more data, more epochs, a different learning rate and head isolation remain hypotheses rather than settled fixes.
