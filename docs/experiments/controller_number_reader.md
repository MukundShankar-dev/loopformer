# Shared number-reader repair — 2026-10-08

Status: fitting completed; fresh graph confirmation is running. The
[protocol](../number_reader_repair.md) was committed before fitting in `03cbc13`.
This is a reader-only intervention, with no training-depth extension.

## Model and training exposure

Source: `models/stage1_pointer/controller-affine-seed83/best`, adapter SHA-256
`8dfdaee726c0f6413ed2f15478abad68574eb63d8401bb3463bbdc09ab27ff7a`.
Export: `models/stage1_pointer/controller-shared-number-seed83/best`.
All 152 non-reader tensors are bitwise unchanged, including the executor,
bridge, countdown and stop readout. The old source inference files are unchanged.

The same 58 numerical labels are used: 1–63 excluding 9/17/29/41/53. Training
graph/count IDs are checked against the original 256-graph panel, and its dataset
bytes/configuration must match. One graph's variants suffice because routed count
features are exactly graph invariant; the final graph's variants are checked.
No validation/test graph, count or target enters fitting. The prior 12-loop
controller supervision and original executor training remain unchanged.

Replace independent suffix-slot weights with a single shared projection and
learned accumulation gain. The fitter infers the gain from existing single-token
and two-token count labels, then least-squares fits the projection on those same
58 labels. It adds no digit-value labels, installs no radix and does not parse a
number at inference. The fit learns gain **10** with maximum training count error
**0.000003814697265625**. This is a task-specific counting bias, not a general
confidence/repair evaluator.

## Commands and artifacts

Desktop: Ubuntu/WSL, RTX 5070 Ti 16 GB, Python 3.11.8, torch 2.14.0+cu130.

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash repair_number_reader.sh
```

The first run was launched in tmux session `number-reader`, with all output
captured in `eval/pointer_benchmark/shared-number-20261008/launch.log` and the
exit code saved alongside it. W&B is disabled. The run refuses existing outputs.

Fit metadata and a portable checkpoint are under
`models/stage1_pointer/controller-shared-number-seed83/`. Frozen identities,
confirmation metrics, full compressed decisions, raw-symbol graph trajectories
and native fidelity traces are under
`eval/pointer_benchmark/shared-number-20261008/`. Graph data live under
`data/pointer/benchmark-seeds281-283-293/`, ignored by Git. No weights are pushed.

## Declared confirmation and diagnostics

Fresh seeds 281/283/293, 150 graphs per seed/mode: **1,350 independent graphs**
and **345,600 graph/count queries**, every integer count 1–256. Same graph modes,
stop threshold 0.5, safety cap 272 and quality thresholds as the failed benchmark.
Reject every old seed-61 table and the opened seeds211/223/227 tables.
The native panel uses one graph per seed/mode at 21 declared counts, **189 actual
stopped calls**. Reused executor trajectories do not constitute an adaptive
latency benchmark. Numbers through 256 were opened on the old architecture;
fresh graph confirmation does not make those numbers wholly new observations.

Controller replay already gives exact stopping at **all 256 counts**, including
69/70 and 99/100, with no early/late/missing stop. End-to-end quality and native
fidelity are still pending; timing alone does not establish a successful repair.

The separate diagnostic reads all integer requests through 10,000 and free-runs
the unchanged timer through 4,096. No R transition executes in that diagnostic.
It separates reader error from accumulated countdown drift and cannot establish
pointer quality at those depths. Results are pending. No tuning follows those
outcomes within this declared experiment.

## Implementation checks

Sixteen initial reader/affine/benchmark tests passed. Thirty-six additional
architecture/prefix/controller-training/remaining-work checks passed. Five reader
checks subsequently passed, including the actual composing fit/export CLI,
unchanged non-reader tensors and rejection of a modified training-count panel.
The shared benchmark supports both old/new scalar controllers and its tiny-model
native/reuse tests include intentionally failed stopping. These are implementation
checks; pretrained confirmation and independent saved-result audits remain pending.
