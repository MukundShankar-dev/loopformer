# CUDA pointer training on 5,000 mappings — 2026-09-10

Read-only artifact audits of user-run training and two final-answer evaluations. No pretrained run was launched during these audits or full-loop CLI implementation.

## Configuration and artifacts

Training run: `models/stage1_pointer/20260910T220130.926886Z/`, code revision `c8038374980c0e20ec8f50a3fe5393f0f3df4e19`. Recorded Git status contains only the new run directory. Audited source hashes match the code used for both final-answer runs; data hashes match the seed-17 files.

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer.json --device cuda
```

Pinned Qwen2.5-0.5B-Instruct base, shared rank-8 q/v LoRA, seed 17, 5,000 distinct depth-1–4 mappings for one epoch, batch 1, accumulation 8, learning rate 0.0002, and 10 warmup updates. Completed 625 updates / 5,000 example presentations without resume. Monitoring used 16 training examples and 64 validation examples, eight per depth 1–8. CUDA startup checks passed. Training used intermediate 26-symbol CE and raw prompts.

Saved `summary.json` records 868.52 seconds (14m29s), including 785.96 training seconds and 81.43 validation seconds. Peak CUDA tensor allocation: 2,974,977,536 bytes (2.77 GiB); this is not total GPU memory usage.

## Training monitoring

| Measurement | Update 500 | Update 625 |
| --- | ---: | ---: |
| Validation loss on trained depths | 0.07774 | 0.24326 |
| Complete validation trajectories, depths 1–4 | 29/32 | 29/32 |
| Complete validation trajectories, depth 5 | 0/8 | 7/8 |
| Complete validation trajectories, depths 6–8 | 0/24 | 0/24 |
| Fixed training-probe complete trajectories | 14/16 | 14/16 |

Update 500 minimizes trained-depth validation loss; update 625 is the final checkpoint and an explicitly secondary comparison. On the same 16 validation examples used by the earlier overfit experiment, both achieved 15 complete trajectories, versus four at the overfit run's end. This is not an isolated data-size ablation: accumulation, update budget, warmup, and monitoring also differ.

Audited all 16 monitoring CSVs (4,416 rows): exact intermediate/final targets, masks, finite margins, counts, and accuracy summaries. Also verified update sequence, selection, and source/data hashes. Loss summaries were read from logs; full logits were unavailable to independently recompute CE.

## Full final-answer evaluations

```bash
python -m scripts.eval.naive_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000500 --device cuda
python -m scripts.eval.naive_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000625 --device cuda
```

| Depth | Update 500 correct / 125 | Update 625 correct / 125 |
| --- | ---: | ---: |
| 1 | 123 | 125 |
| 2 | 120 | 124 |
| 3 | 111 | 117 |
| 4 | 86 | 99 |
| 5 | 15 | 53 |
| 6 | 3 | 11 |
| 7 | 6 | 5 |
| 8 | 6 | 3 |
| Total / 1,000 | 470 (47.0%) | 537 (53.7%) |

Trained-depth final accuracy: 88.0% versus 93.0%. Evaluation times: 91.92 and 91.95 seconds. Both used float32, CUDA, batch 1, raw prompts, and A–Z argmax after exactly d loops. These are different conditions from the ordinary-Qwen three-shot, unrestricted-generation 6% baseline.

Artifacts, each containing `predictions.csv` and `summary.json`:

- `eval/pointer_task/20260910T222304.339030Z-models-stage1_pointer-20260910T220130.926886Z-step-000500/`
- `eval/pointer_task/20260910T222727.202173Z-models-stage1_pointer-20260910T220130.926886Z-step-000625/`

Both 1,000-row files passed independent audits of unique IDs, reference targets, scores, input prompts, symbol token IDs, requested/executed loops, source hashes, and data hashes. Adapter hashes are recorded in the summaries; absent weights were not independently rehashed on the Mac.

## Interpretation before full-loop evaluation (historical)

The larger run supports learned final-answer execution on unseen mappings within trained depths and some extension to depth 5. At update 500, 62/125 depth-5 answers equal the fourth reference state; at update 625 this falls to 12/125, while 46/125 depth-6 answers equal the fifth state. These final-output patterns suggest stalled progression but do not reconstruct internal trajectories.

Keep update 500 as the validation-selected primary result and update 625 as secondary. The 7/8 depth-5 validation observation did not translate to comparable full-test accuracy. Do not silently change checkpoint selection based on test outcomes.

The next measurement at that point was full-loop evaluation; it is now complete and audited below. Gate 1 criteria remained unresolved. The uninterrupted run did not resolve the [WSL exact-resume issue](wsl_cuda_baseline.md).

## Full-loop evaluations and audit

The user subsequently ran the full-loop evaluator on both saved checkpoints:

```bash
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000500 \
  --device cuda --loops 8
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000625 \
  --device cuda --loops 8
```

Each run contains 1,000 examples and 8,000 readouts. Primary update 500 took 152.55 seconds; secondary update 625 took 160.51 seconds, excluding model loading and final diagnostic exports.

| Depth | Update 500 complete trajectories / 125 | Update 625 complete trajectories / 125 |
| --- | ---: | ---: |
| 1 | 123 | 125 |
| 2 | 120 | 124 |
| 3 | 108 | 117 |
| 4 | 80 | 96 |
| 5 | 11 | 50 |
| 6 | 0 | 4 |
| 7 | 0 | 0 |
| 8 | 0 | 0 |
| Overall / 1,000 | 442 (44.2%) | 516 (51.6%) |

Complete-trajectory accuracy at trained depths 1–4 is **431/500 (86.2%)** versus **462/500 (92.4%)**. Nominal final accuracy remains 47.0% versus 53.7% overall. Every nominal final prediction matches the preceding corresponding `naive_test` run. At update 625, 462 of 465 correct trained-depth final answers also have every preceding step correct.

Conditional first failures localize the extension breakdown. At update 500, 257/318 examples requiring loop 5 fail there despite every preceding step being correct; 197 of those failures repeat the preceding decoded symbol. At update 625, 155/170 examples requiring loop 6 fail after five correct steps; 103 of those failures repeat the preceding symbol. This is evidence of stalled decoded progression, not proof of frozen hidden states.

Overall equal-example nominal CE is 0.83299 at update 500 and 0.97028 at update 625. Despite higher accuracy, update 625 has larger losses at loops 6–8. Accuracy and probability quality therefore do not improve uniformly with more updates.

Artifacts:

- Primary: `eval/pointer_loops/20260910T225645.650884Z-20260910T220130.926886Z-step-000500/`
- Secondary: `eval/pointer_loops/20260910T224545.524555Z-20260910T220130.926886Z-step-000625/`
- Secondary three-example preview: `eval/pointer_loops/20260910T224510.343269Z-20260910T220130.926886Z-step-000625/`

The read-only audits independently executed the dataset mappings, checked trajectory coverage, nominal/final labels, correctness and first-error counts, and aggregated exported losses. Source and dataset hashes were consistent with each run at audit time. Full logits and adapter binaries were unavailable locally, so the audits did not independently recompute CE from logits, rehash model tensors, or rerun inference. Later source edits do not retroactively alter the recorded run provenance.

## Current interpretation and next experiment

These runs support **instance generalization and meaningful stepwise execution at trained depths**, with limited extension to depth 5 and weak execution beyond it. They do not support general depth extrapolation, cross-family transfer, or natural-language reasoning. Both checkpoints are from a single training seed, and Gate 1 quantitative acceptance criteria were not set in advance; no formal gate is declared passed after inspecting these results.

The user has prioritized further Stage 1 training before running overscaling experiments. A [three-total-epoch continuation](../training_pointer.md#continue-the-current-desktop-run) keeps training depths at 1–4 and examines whether extension improves on validation. Retain update 500 as the historical primary and 625 as secondary, even when resuming the latest optimizer state at 625. The seed-17 test results already inspected are diagnostic evidence, not untouched final confirmation for future tuned runs.

[Absorbing-terminal overscaling tools](../overscaling_eval.md) are prepared but execution is deferred. The original continuing-pointer tasks still cannot establish post-completion damage by a changed decoded answer alone. The [WSL exact-resume precision limitation](wsl_cuda_baseline.md) remains unresolved; it is separate from the correctness of these completed evaluation artifacts.

## Three-epoch continuation and full validation

The user completed the documented continuation from update 625, using `configs/stage1_pointer_continue.json` and `--resume models/stage1_pointer/20260910T220130.926886Z/step-000625`. New run: `models/stage1_pointer/20260911T003442.178029Z/`, source revision `04179d37ff714531a8707fe1b4d673565d7d1e43`. All 1,250 additional optimizer updates (626–1875) completed, for three total epochs. Wall time was 1,992.25 seconds (33m12s), with the same 2.77 GiB peak CUDA tensor allocation. Source/data hashes and parent training identity matched. Initial resumed monitoring predictions matched the parent checkpoint; this does not establish bitwise optimizer resume equivalence on CUDA.

Update 1875 was selected by the existing trained-depth validation loss rule: 0.00001757054 on the small monitoring subset. That subset had 32/32 complete trajectories at depths 1–4, 8/8 at depth 5, 3/8 at depth 6, and none at 7–8. Across continuation checkpoints, deeper monitoring fluctuated; update 1700 reached 6/8 at depth 6 but was not substituted for the selected checkpoint.

The user then ran:

```bash
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/20260911T003442.178029Z/step-001875 \
  --data data/pointer/seed-17/validation.jsonl --device cuda --loops 8
```

Artifacts: `eval/pointer_loops/20260911T011247.420716Z-20260911T003442.178029Z-step-001875/`. Full validation took 177.91 seconds for 1,000 examples / 8,000 readouts.

| Depth | Complete trajectories / 125 | Correct final / 125 |
| --- | ---: | ---: |
| 1 | 125 | 125 |
| 2 | 124 | 124 |
| 3 | 125 | 125 |
| 4 | 123 | 123 |
| 5 | 103 | 103 |
| 6 | 26 | 29 |
| 7 | 2 | 7 |
| 8 | 0 | 9 |
| Overall / 1,000 | 628 | 645 |

Trained-depth complete-trajectory accuracy is **497/500 (99.4%)**. Depth-5 accuracy is 82.4%, depth-6 accuracy 20.8%, depth-7 accuracy 1.6%, and depth-8 accuracy 0%. Among 320 examples with five correct preceding steps that require a sixth, 230 first fail at loop 6; 172 of those failures repeat the preceding decoded symbol. This supports reliable nominal execution plus limited depth extension, not arbitrary-depth execution or frozen hidden states.

The read-only audits covered all 30 continuation monitoring CSVs (8,280 rows) and all 8,000 full-validation rows: exact interpreter targets, masks, correctness, aggregate exported CE, source/data hashes, and update sequence. The 512 corresponding monitoring readouts matched the full validation sweep. Adapter tensors and full logits were not available locally; no pretrained inference was repeated by the assistant. The full validation set is different from the earlier checkpoint-500/625 test set, so do not describe their score differences as a paired comparison.

Next is the [depth-6 training and outward OOD experiment](../depth_generalization.md), not additional unchanged depth-4 epochs. Keep update 1875 as the depth-4 reference, run paired evaluation on the same depth-1–16 development examples, and reserve seed 29 for later confirmation. Increasing training depth without moving the OOD range would weaken the extrapolation test. Overscaling and knowledge-retention evaluation remain deferred.
