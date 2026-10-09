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

Status: implementation and evaluation in progress. Results and validation will be
recorded here after completion. Detailed architecture explanations will accompany
this report, with separate technical and plain-language accounts.
