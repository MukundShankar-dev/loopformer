# Offline adaptive-analysis pipeline check on retained nominal traces — 2026-09-29

## Purpose and provenance

This is a **development diagnostic of the offline analyzer**, not an adaptive-inference or terminal-overscaling experiment. The source is the completed fresh 30k depth-6 checkpoint **step 2500** and its two existing seed-17 full-loop evaluations. Source model training and evaluation are documented in [the fresh-run report](stage1_fresh30k.md). The original pointer mapping continues after the requested depth `d`; a changed answer after `d` may be a valid additional transition. Consequently, every derived trajectory category is prefixed `observational_`, and no repair/damage, safe stopping, or learned-policy conclusion is drawn.

The analyzer ran from the current working tree after the adaptive-compute implementation, with code revision based on `3d4ab9d` plus this change. Each derived `summary.json` records source-run, trajectory, source-code, checkpoint, and data hashes. It consumes the existing CSVs without loading a model, modifying the historical results, or rerunning inference. The original 30k run used one training seed; these seed-17 sets are development data. Seed 29 remains untouched.

```bash
.venv/bin/python -m scripts.eval.adaptive_analyze \
  eval/pointer_loops/20260911T201538.726082Z-depth6-fresh30k-seed37-batch4-step-002500 \
  --output eval/pointer_adaptive_analysis/20260929-step2500-depth1to8-observational \
  --allow-observational

.venv/bin/python -m scripts.eval.adaptive_analyze \
  eval/pointer_loops/20260911T201847.679922Z-depth6-fresh30k-seed37-batch4-step-002500 \
  --output eval/pointer_adaptive_analysis/20260929-step2500-depth9to16-observational \
  --allow-observational
```

Both derived directories contain `examples.csv` and `summary.json`. Each cohort has 1,000 examples. The first has a common eight-loop trace budget; the second has sixteen loops. All reported accuracy uses the fixed final target at the indicated stopping loop. “Loop savings” below is **counterfactual replay accounting** against that cohort's common trace budget; the full source inference already ran and saved every loop. It is not measured latency or energy savings.

| Cohort | Replay rule | Fixed-final accuracy | Mean loops | Counterfactual loop savings |
| --- | --- | ---: | ---: | ---: |
| Depths 1–8, budget 8 | Read at budget 8 | 22.7% | 8.00 | 0% |
| Depths 1–8, budget 8 | Read at requested depth `d` | 97.2% | 4.50 | 43.75% |
| Depths 1–8, budget 8 | Labeled first-correct oracle, `t≥d` | 97.8% | 4.56 | 43.03% |
| Depths 1–8, budget 8 | Prediction stability, `k=2`, `t≥d` | 23.8% | 7.80 | 2.46% |
| Depths 9–16, budget 16 | Read at budget 16 | 6.0% | 16.00 | 0% |
| Depths 9–16, budget 16 | Read at requested depth `d` | 21.0% | 12.50 | 21.88% |
| Depths 9–16, budget 16 | Labeled first-correct oracle, `t≥d` | 24.6% | 14.70 | 8.12% |
| Depths 9–16, budget 16 | Prediction stability, `k=2`, `t≥d` | 8.0% | 13.34 | 16.62% |

The contrast between reading at `d` and at a common larger budget reflects the continuing task's reference semantics. The suffix oracle, which requires correctness through the observed horizon, reached only 1.11% and 0.76% loop savings in these two cohorts. That is **not** evidence about terminal-state safety. The first-correct oracle uses true answers and future traces and is unavailable as an inference policy. Prediction stability is causal but was not tuned or confirmed here.

## Interpretation and next measurement

The derived files verify full-trajectory parsing, category accounting, fixed/requested/oracle/heuristic replay, provenance, and loop-count aggregation on real retained traces. They do not establish adaptive headroom on a valid completed task, a pretrained stopped-inference accuracy result, or latency savings. The pretrained adapter tensors and seed-37 training data are absent in this checkout, so no pretrained adaptive forward was run. Follow the [canonical adaptive-compute guide](../adaptive_compute.md) for the terminal-task analysis and measured stopping protocol after the current loop-balanced experiment.
