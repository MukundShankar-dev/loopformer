# Loop-balanced CE ablation — 2026-09-29

The user completed the fresh equal-loop run and six full evaluations on CUDA. It did not improve the observed depth-generalization frontier. The next step is diagnosis and performance measurement, not another training sweep. This is one training seed and development data, not confirmation.

## Configuration and results

Config: `configs/stage1_pointer_depth6_loopbalanced.json`; fresh adapters, seed-37 30k mappings at depths 1–6, training seed 17, batch 4/accumulation 2, LR 0.0002, 3,750 updates. Relative to the fresh 30k baseline, the config changes only loss reduction. Selection uses trained-range equal-loop CE, so the selected checkpoints' loss values are not directly comparable across objectives. Training source revision: `2ea98c6d6afef61f44c30fca35bddd7db1eadaec`.

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_loopbalanced.json --device cuda \
  --output models/stage1_pointer/depth6-loopbalanced-seed37

for step in 002500 003250 003750; do
  python -m scripts.eval.loop_test \
    --model "models/stage1_pointer/depth6-loopbalanced-seed37/step-$step" \
    --data data/pointer/seed-17/validation.jsonl --device cuda --loops 8 || break
  python -m scripts.eval.loop_test \
    --model "models/stage1_pointer/depth6-loopbalanced-seed37/step-$step" \
    --data data/pointer/seed-17/depth_test.jsonl --device cuda --loops 16 || break
done
```

Training completed in 3,277.06 seconds (54m37s), with 7.01 GiB peak CUDA tensor allocation. Startup T=1 maximum logit error was zero. Step 3250 was selected by trained-range validation loss. This wall-time comparison alone does not isolate a performance improvement.

Complete-trajectory accuracy, requiring every nominal step correct (125 examples per individual depth):

| Depth | Baseline 2500 | Equal-loop 2500 | Baseline 3250 | Equal-loop 3250 | Baseline 3750 | Equal-loop 3750 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–6 | 98.0% | 98.4% | 99.2% | 99.33% | 99.33% | 98.93% |
| 7 | 94.4% | 91.2% | 93.6% | 95.2% | 94.4% | 96.0% |
| 8 | 94.4% | 66.4% | 86.4% | 92.8% | 87.2% | 89.6% |
| 9 | 73.6% | 32.0% | 56.0% | 63.2% | 60.8% | 68.8% |
| 10 | 41.6% | 8.0% | 15.2% | 15.2% | 27.2% | 24.8% |
| 11 | 14.4% | 0% | 2.4% | 2.4% | 8.8% | 8.8% |
| 12 | 1.6% | 0% | 0% | 0% | 0% | 0% |
| 13–16 | 0% | 0% | 0% | 0% | 0% | 0% |

Some matched-update nearby-depth scores improve, but no evaluated equal-loop checkpoint beats the earlier baseline 2500 on farther depth extension. Baseline 2500 remains the working checkpoint. This weakens inadequate late-loop direct weighting as a sufficient explanation, without identifying a causal mechanism. At equal-loop 3750 on depths 9–16, 341/576 examples with nine correct preceding steps fail at loop 10; 107 of those failures repeat the previous prediction. Repeated decoded symbols explain only part of the failure pattern.

## Audit and retirement

Read-only checks covered 3,750 update records, 32,768 training-validation rows, and all 72,000 full-evaluation rows. Checks compared labels with saved reference targets, recomputed correctness and trajectory counts, checked coverage, dataset hashes, and matching device/dtype/batch/loop/scoring settings across baseline pairs. Full logits and adapter binaries were not re-evaluated locally; no pretrained inference was run.

The user requested deletion of this ablation's artifacts. Removed the training directory and six named full evaluations locally (24.08 MiB). Their tracked contents remain in Git history before this deletion. Evaluation names begin `20260929T232347.929385Z`, `20260929T232646.364715Z`, `20260929T233210.473739Z`, `20260929T233501.570440Z`, `20260929T234030.258696Z`, and `20260929T234322.963417Z`; suffixes identify the run and step. The baseline, datasets, configs, and objective implementation remain.

On the desktop, with no affected process active, `python -m scripts.training.cleanup_pointer_runs --loopbalanced-only` previews these exact directories; add `--apply` to remove them including ignored checkpoint binaries. This has not been executed remotely. Git pull alone cannot remove ignored binaries. Cleanup tests: three focused cases passed, including exact target selection, preview preservation, and symlink rejection.
