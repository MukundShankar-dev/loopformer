# Pointer architecture history: paired reconstruction

Date: 2026-10-09. Protocol declared before new inference. The user requested the
architectures preceding executor freezing, rather than another comparison of
number-reader/controller variants. This extends the existing checkpoint evaluator;
no training, checkpoint search, threshold fit or new research stage is authorized.

## Fixed comparison

`configs/pointer_architecture_history.json` specifies six retained checkpoints:

| Name | Checkpoint | Role |
| --- | --- | --- |
| Full sequence, CE only | `depth6-fresh30k-seed37-batch4/step-003250` | Original full-sequence recurrence; no learned stopping |
| Full sequence, joint stop | `depth6-completion-seed37/step-003250` | Adds jointly trained hidden-state completion MLP |
| Fixed prompt, depth 6 | `depth6-fixed-prompt-seed37/step-003250` | Restricts recurrence to the answer workspace with fixed layer-specific prompt memory |
| Fixed prompt, depth 12 | `depth12-fixed-prompt-seed47-gaps/step-005000` | Same architecture, broader sparse-count training recipe; not another architecture |
| Isolated full R + GRU | `controller-prefix-seed83/best` | Successful executor upgrade, retained unsuccessful recurrent controller |
| Isolated full R + learned timer | `controller-shared-number-precision-seed83/best` | Same frozen executor, final repaired controller |

All paths are under `models/stage1_pointer/`. The first three use the same selected
update 3,250, seed-37 30k training tables, requested training depths 1–6, and seed-17
trained-depth pointer-CE selection. The earlier step-2,500 CE model remains a
historically stronger extrapolation checkpoint, but using it here would mix a
checkpoint-selection difference into the matched early comparison. No checkpoint
is selected using the newly evaluated panel.

Inputs remain the first three graphs per seed/type stratum from seeds
307/311/313: **27 identical graphs × 30 declared counts = 810 queries per model**.
The panel has already been opened. This is diagnosis, not fresh confirmation.
CUDA/FP32/SDPA, forced batch 16; learned-stop threshold 0.5, cap 272, actual
stopped calls batch one. No parsed clock or target state is supplied to any model.

Original CE-only stopping is **not applicable**, represented by empty CSV cells
and JSON nulls. Its evaluator forces N loops to test execution, then checks a
predeclared batch-one subset (one graph per seed/type, six native counts).
It does not pretend that forcing N demonstrates learned completion or treat an
absent head as a never-stopping head. Joint-stop and fixed-prompt depth-6 models
run all 810 actual stopped calls, independently audited from raw rules, and
separate per-count forced traces. Their early native prefixes must agree exactly
with batched forced predictions.

Depth-12 and the two isolated arms reuse the original audited observations only
when graph selection, counts, policy, weights, source benchmark and recurrent
implementation hashes match. Their real stopped traces remain part of the
reconstructed artifact. This saves inference without substituting another model's
outputs. All arms are independently rescored together after assembly.

The two isolated checkpoints must share every non-controller tensor. Historical
R sees Steps, so each count changes its actual input and may change its common
intermediate prefix. Its strict-prefix curve is not survival along a single
count-invariant rollout; each column is an independently executed prompt variant. Historical
models additionally differ in objectives, writable memory, data support, capacity,
bridge, routing and optimizer recipe. Only the first three offer the narrower
matched early comparison; even that does not isolate a unique hidden-state cause.

## Reproduction

From the CUDA desktop repository with its environment active:

```bash
python -m scripts.eval.checkpoint_comparison \
  --config configs/pointer_architecture_history.json --dry-run
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u -m scripts.eval.checkpoint_comparison \
  --config configs/pointer_architecture_history.json
python -m scripts.eval.audit_checkpoint_comparison \
  --results eval/pointer_benchmark/architecture-history-20261009 \
  --graphs data/pointer/benchmark-seeds307-311-313/graphs.jsonl
python -m scripts.eval.plot_pointer_failures \
  --graphs data/pointer/benchmark-seeds307-311-313/graphs.jsonl \
  --comparison eval/pointer_benchmark/architecture-history-20261009 \
  --output eval/pointer_benchmark/architecture-history-20261009/plots
```

Fresh output paths are required. Original comparison/benchmark evidence is retained.
Plots include execution-depth points, architecture/count outcome matrices,
actual-stop matrices and cyclic wrong-time/early-stop comparisons. CE-only stop
cells are gray N/A and the stop-only panels exclude that model. The plots also
retain the full final-model failure diagnostics, whose population is 1,350 graphs
rather than this 27-graph comparison.

Status: completed. The three newly executed historical arms ran on the desktop
at Git `a3c0462`; the other three reuse frozen, audited calls with passing identity checks. [Model evolution](../pointer_model_evolution.md)
provides separate technical and plain-language explanations of each architecture,
its training and inference, and the motivation and evidence for each change.


## Completed results

Each percentage averages **810 queries on the same 27 graphs**, with equal weight
for the 30 declared counts. Many requests are beyond the early models' training
range. This average is not a training-range score, 810 independent graph trials,
or the final model's larger benchmark population estimate.

| Checkpoint | Forced final letter | Every nominal step correct | Exact first stop | Correct answer + exact first stop |
| --- | ---: | ---: | ---: | ---: |
| Full sequence · CE only | 29.26% | 24.57% | N/A | N/A |
| Full sequence · joint stop | 22.35% | 20.37% | 16.42% | 16.30% |
| Fixed prompt · depth 6 | 29.26% | 25.31% | 16.67% | 16.67% |
| Fixed prompt · depth 12 | 44.07% | 38.64% | 35.93% | 35.31% |
| Isolated full R + GRU | 94.81% | 94.44% | 23.46% | 22.96% |
| Isolated full R + learned timer | 94.81% | 94.44% | 100.00% | 94.81% |

Depth-specific complete trajectories (every intermediate C letter correct):

| Checkpoint | N=6 | N=9 | N=12 | N=16 | N=64 | N=256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full sequence · CE only | 27/27 | 13/27 | 0/27 | 0/27 | 0/27 | 0/27 |
| Full sequence · joint stop | 26/27 | 0/27 | 0/27 | 0/27 | 0/27 | 0/27 |
| Fixed prompt · depth 6 | 27/27 | 16/27 | 0/27 | 0/27 | 0/27 | 0/27 |
| Fixed prompt · depth 12 | 26/27 | 26/27 | 25/27 | 1/27 | 0/27 | 0/27 |
| Isolated full R + GRU | 26/27 | 26/27 | 26/27 | 25/27 | 25/27 | 25/27 |
| Isolated full R + learned timer | 26/27 | 26/27 | 26/27 | 25/27 | 25/27 | 25/27 |

Joint completion worsens forced execution against matched CE-only training;
premature stopping is therefore not the whole cause. Fixed memory improves the
joint variant at N=8/9, but does not consistently beat CE-only: at N=10 it has
2/27 complete trajectories versus CE-only's 4/27. Extending the same architecture's
training depth moves its failure boundary, rather than establishing robust
extrapolation. Both depth-6 heads stop early on all sampled requests above six;
the depth-12 head stops early on all sampled requests above twelve.

The isolated executor has identical forced outputs under GRU and the final timer.
Their two execution series overlap exactly in the plot. The GRU has no exact
stops at N=64/128/256, whereas the final timer is exact throughout the declared
panel. This separates the executor upgrade's result from the later controller
repair; it does not assign sole causal credit to one upgrade component.

One prose mistake in the original comparison report was corrected: the depth-12
model has **one** complete trajectory at count 17, not zero. That graph starts
at a C fixed point; at all sampled counts from 24 onward the count is zero.
The original CSV and heatmap already contained the correct 1/27 value. No stored
metrics were changed. Fixed-point success alone does not demonstrate multi-edge
long-range execution.

## Figures and verification

Four architecture figures are regenerated under
`eval/pointer_benchmark/architecture-history-20261009/plots/`, in PNG/PDF/SVG:

- [Execution versus requested depth](../../eval/pointer_benchmark/architecture-history-20261009/plots/architecture_execution_depth.png)
- [Four separate execution/control metrics by model and count](../../eval/pointer_benchmark/architecture-history-20261009/plots/architecture_metric_matrices.png)
- [Requested versus actual first stops](../../eval/pointer_benchmark/architecture-history-20261009/plots/requested_actual_stop_matrices.png)
- [Correct letters at the wrong time and premature stopping](../../eval/pointer_benchmark/architecture-history-20261009/plots/cyclic_coincidence_and_early_stops.png)

The same directory contains four regenerated final-model failure figures. Their
PNG bytes match the previously reviewed full-panel exports. Those figures retain
the 1,350-graph population; they are not recomputed from the 27-graph historical
subset. Number-reader precision plots are omitted from this history comparison.
All four changed architecture figures were visually inspected, including the
final consistent-color revision. Provenance records exact source and export hashes.

The independent audit checks **all 4,860 combined decisions** against raw mapping
traversal, coverage and aggregates. An additional export audit checks **720**
model/count/metric cells, including **60 undefined CE-only stopping cells**.
The native re-audit passes **2,538 actual stopped calls / 20,502 transition rows**.
CE's **54 batch-one forced calls / 4,032 transitions** agree exactly with batched
predictions and their independently audited targets. Thus 2,592 native calls are
covered in total; forced and self-stopped calls are reported distinctly.

The three new inference-arm scopes total **796.41 seconds** (13m16s), including
loading and verification; copied arms' older `seconds` fields are historical,
not newly measured runtime. All checkpoint inference hashes remain unchanged.
The log is `eval/pointer_benchmark/architecture-history-launch-20261009/run.log`.
Decisions, per-count exports, freeze, reuse receipts, independent reference audit
and `native_records_audit.json` live in the comparison root. Model binaries stay
on the desktop and no training ran.

The 23 focused benchmark/comparison/failure/audit tests pass, including undefined
stopping, rejection of fabricated zero-valued plot cells, cyclic wrong-time
failure, exact targets and changed-executor rejection. Final focused comparison
checks, local documentation links, compilation, plot/export hashes and whitespace
checks pass. The 27-graph panel is small and opened; historical graph distribution,
training depth and capacity confounds remain. These results do not replace the
full final-model benchmark or establish arbitrary-depth/cross-family generality.
