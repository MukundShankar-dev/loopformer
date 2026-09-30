# Baseline step-2500 diagnostics and CUDA profiles

Artifacts dated 2026-09-30 UTC (2026-09-29 locally). These are exploratory development measurements, not confirmation or a new training run. All four run summaries report completion. The probe's dataset hash and recorded source hashes match this checkout. Checkpoint tensors were not independently inspected on the Mac.

## Paired execution

The first 32 seed-17 depth-test examples contain four tasks at each depth 9–16. Each was evaluated unchanged, restarted from its reference state after six transitions with the remaining Steps, and with only its requested depth reduced by one. Baseline step 2500, float32, eager attention, batch 1; seed 29 remains untouched.

| Variant | Complete trajectories |
| --- | ---: |
| Original | 6/32 |
| Reference suffix restart | 30/32 |
| Requested depth reduced by one | 9/32 |

The original has 26 first failures. Of these, 22 occur after loop six; every one of those aligned transitions is correct in the suffix run (22/22). The other four fail before the restart boundary. In the shortened-depth comparison, 22 original first failures still have an aligned transition, and only 2/22 become correct; four original first failures lie at the removed final step. These are different eligible cohorts. The suffix's two failures occur at its final transitions at local depths nine and ten.

This supports investigating sensitivity to uninterrupted recurrent history. It does not prove a hidden-state cause: the suffix changes Start, Steps, and input encoding, and receives an oracle-correct intermediate state. The shortened task also has one fewer opportunity for failure, so 9/32 versus 6/32 alone is not a depth-cue effect estimate.

On the same four depth-16 original examples, median answer-position relative update norm falls from 0.741 at loop 6 to 0.152 at loop 12 and 0.097 at loop 16. Median answer RMS changes from 1.136 to 1.329, and sequence RMS from 4.981 to 5.449. This is shrinking relative motion with moderate scale growth, not evidence of exploding norms. It does not establish representational collapse or locate lost information.

## Performance

Two warmups and five synchronized unprofiled measurements; medians below. Memory is peak allocated CUDA tensors, not all device usage.

| Measurement | Median | Peak allocation |
| --- | ---: | ---: |
| Evaluation: 128 examples, 16 loops, batch 16 | 9.083 s | 2.34 GiB |
| One eight-example update: batch 4, accumulation 2 | 0.633 s | 7.01 GiB |
| Same update group: batch 8, accumulation 1 | 0.658 s | 12.33 GiB |

The training timing ranges overlap substantially; no batch-8 speedup is established. Both groups supervise 24 transitions, but batch 4 executes 40 sample-transitions and batch 8 executes 48 because shorter examples continue to the microbatch maximum. Batch 8 performs 20% more masked work. These are disposable first updates with fresh optimizer allocation, not full-training throughput. Neither update clips gradients; gradient norms are about 0.975.

Operator tables show substantial float32 dense-matrix work in R and C. The evaluation trace attributes about 19.5 ms to the LM head in a roughly 1.1 s batch; metrics/rows take about 25 ms and export about 1.7 ms. These observations deprioritize selected-vocabulary projection and CSV cleanup as major speedups for this workload. Inclusive ranges overlap; CPU timings can include GPU waits. Duplicate CPU/CUDA range names and incomplete backward device attribution prevent treating range totals as exclusive GPU utilization. Single trace-overhead ratios below one are timing noise, not evidence profiling accelerates execution.

## Next bounded decision

Keep training batch 4/accumulation 2 and evaluation batch 16 as measured references. Before another training recipe, design an intervention that distinguishes recurrent-history degradation from changed Start/Steps and fresh prompt encoding. No particular internal reset is yet validated or implemented. A separate explicit precision benchmark with paired prediction, loss and gradient checks is better motivated by these profiles than increasing training batch again. Do not silently change the float32 equivalence gate. No halting head or new loss follows from these results.

## Artifacts and exact invocations

Commands below are reconstructed from the saved command arrays (using portable `python` in place of the desktop interpreter path). The named output directories already exist; reruns need new output paths. Summaries contain checkpoint/data/source hashes, runtime versions, settings and individual timing samples. Full profiler traces are ignored by Git.

- [eval/pointer_probes/baseline2500-20260930T005754Z-2318/summary.json](../../eval/pointer_probes/baseline2500-20260930T005754Z-2318/summary.json)

```bash
python -m scripts.eval.probe_pointer --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 --data data/pointer/seed-17/depth_test.jsonl --device cuda --limit 32 --restart-after 6 --output eval/pointer_probes/baseline2500-20260930T005754Z-2318
```

- [eval/pointer_profiles/baseline2500-eval-b16/summary.json](../../eval/pointer_profiles/baseline2500-eval-b16/summary.json)

```bash
python -m scripts.eval.profile_pointer --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 --data data/pointer/seed-17/depth_test.jsonl --device cuda --mode eval --batch-size 16 --loops 16 --limit 128 --output eval/pointer_profiles/baseline2500-eval-b16
```

- [eval/pointer_profiles/baseline2500-20260930T005754Z-2318/train-b4/summary.json](../../eval/pointer_profiles/baseline2500-20260930T005754Z-2318/train-b4/summary.json)

```bash
python -m scripts.eval.profile_pointer --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 --data data/pointer/seed-37-depth6-30k/train.jsonl --device cuda --mode train --batch-size 4 --effective-batch 8 --train-max-depth 6 --output eval/pointer_profiles/baseline2500-20260930T005754Z-2318/train-b4
```

- [eval/pointer_profiles/baseline2500-20260930T005754Z-2318/train-b8/summary.json](../../eval/pointer_profiles/baseline2500-20260930T005754Z-2318/train-b8/summary.json)

```bash
python -m scripts.eval.profile_pointer --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 --data data/pointer/seed-37-depth6-30k/train.jsonl --device cuda --mode train --batch-size 8 --effective-batch 8 --train-max-depth 6 --output eval/pointer_profiles/baseline2500-20260930T005754Z-2318/train-b8
```
