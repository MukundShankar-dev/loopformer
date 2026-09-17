# Fresh 30k pointer baseline

## Configuration and execution

Completed on the RTX 5070 Ti desktop with fresh recurrent adapters on pinned Qwen2.5-0.5B-Instruct; no resume or adapter initialization. Configuration: [batch-4 baseline](../../configs/stage1_pointer_depth6_fresh30k_batch4.json). Dataset seed 37, 30,000 mappings, depths 1–6; training seed 17, one epoch, 3,750 updates, batch 4/accumulation 2, float32, rank-8 q/v LoRA, constant learning rate 0.0002 after ten warmup updates. Per-example mean nominal CE was the training objective. Validation used 32 examples per depth at 1–8 from seed 17; checkpoint selection used example-mean CE at trained depths only.

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_fresh30k_batch4.json --device cuda \
  --output models/stage1_pointer/depth6-fresh30k-seed37-batch4
```

Artifacts: `models/stage1_pointer/depth6-fresh30k-seed37-batch4/`. Wall time 3,654.26 seconds (60m54s), training 3,346.66 seconds, validation 302.59 seconds. Mean training throughput 8.96 examples/s. Peak CUDA tensor allocations 7.01 GiB, not total device usage. The startup gate reported zero T=1 logit error. Earlier batch-1/2 attempts were discarded by the user; this run started fresh.

## Full evaluation

The actual saved evaluation commands used batch 1, despite the earlier suggested batch-4 commands:

```bash
for step in 002500 003250 003750; do
  python -m scripts.eval.loop_test \
    --model "models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-$step" \
    --data data/pointer/seed-17/validation.jsonl --device cuda --loops 8 || break
  python -m scripts.eval.loop_test \
    --model "models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-$step" \
    --data data/pointer/seed-17/depth_test.jsonl --device cuda --loops 16 || break
done
```

Complete trajectories require every nominal intermediate state to be correct. Individual depths have 125 examples; depths 1–6 aggregate 750.

| Depth | 2500 | 3250 | 3750 |
| --- | ---: | ---: | ---: |
| 1–6 | 98.0% | 99.2% | 99.33% |
| 7 | 94.4% | 93.6% | 94.4% |
| 8 | 94.4% | 86.4% | 87.2% |
| 9 | 73.6% | 56.0% | 60.8% |
| 10 | 41.6% | 15.2% | 27.2% |
| 11 | 14.4% | 2.4% | 8.8% |
| 12 | 1.6% | 0% | 0% |
| 13–16 | 0% | 0% | 0% |

The six run directories under `eval/pointer_loops/` begin with `20260911T201538.726082Z`, `20260911T201847.679922Z`, `20260911T202436.195959Z`, `20260911T202743.750447Z`, `20260911T203340.701127Z`, and `20260911T203644.858894Z`, respectively. Each ends in the baseline run name and checkpoint step. They contain CSV trajectories, per-example diagnostics, depth-by-loop tables, and provenance summaries.

Step 2500 is the working depth-generalization checkpoint among three fully evaluated candidates. It wins on validation depths 7–8 (94.4% combined); step 3250 remains the original trained-loss-selected reference. Neither is established as best among every saved checkpoint. Later training improved trained-range accuracy while reducing extension in these evaluations. This is not proof of memorization or a particular internal failure mechanism.

Compared with the preceding depth-6 curriculum checkpoint 500, step 2500 improves complete trajectories at depths 8/9/10 from 70.4/33.6/5.6% to 94.4/73.6/41.6%. Data volume, training history, and batching differ; this does not isolate the cause. Seed-17 sets are now development diagnostics. Seed 29 is reserved and untouched.

## Audit and retention

The training audit checked 34,816 diagnostic rows, 3,750 update records, fixed validation cohorts, reference targets, scoring, aggregate losses, and source hashes. The evaluation audit checked all 72,000 rows, matching data/source hashes, checkpoint identities across paired files, and exact nominal prediction agreement with the training monitor. The seed-37 training file and model weights are absent on the Mac, so their full contents were not independently verified here.

At the user's request, pre-30k recurrent artifacts and redundant checkpoint directories were removed on 2026-09-17. Historical tracked evidence is available at Git revision `1edacda0d2199d2e0db2580824618375cc27170a`. The 30k metrics, six evaluations, and checkpoints 2250/2500/2750/3000/3250/3750 are retained. Desktop ignored weights require the [cleanup command](../training_pointer.md#artifact-cleanup).

The next authorized experiment is [loop-balanced CE](../training_pointer.md#loop-balanced-loss-experiment), keeping data, prompt, initialization seed, architecture, learning rate, and budget fixed. It has not been run on the pretrained model. Compare matching update numbers and validation trajectories as well as each objective's selected checkpoint; differing selection rules alone must not be presented as a training improvement.
