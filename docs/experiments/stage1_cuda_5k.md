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

## Interpretation and next measurement

The larger run supports learned final-answer execution on unseen mappings within trained depths and some extension to depth 5. At update 500, 62/125 depth-5 answers equal the fourth reference state; at update 625 this falls to 12/125, while 46/125 depth-6 answers equal the fifth state. These final-output patterns suggest stalled progression but do not reconstruct internal trajectories.

Keep update 500 as the validation-selected primary result and update 625 as secondary. The 7/8 depth-5 validation observation did not translate to comparable full-test accuracy. Do not silently change checkpoint selection based on test outcomes.

Next: run the [full-loop evaluator](../loop_pointer_eval.md) on both checkpoints to measure complete trajectories, first errors, and depth-by-loop behavior on full datasets. Gate 1 criteria remain unresolved; final-answer accuracy alone does not establish the mechanism gate. The [WSL exact-resume issue](wsl_cuda_baseline.md) remains unresolved by this uninterrupted run.
