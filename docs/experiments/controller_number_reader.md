# Shared number-reader repair — 2026-10-08

Status: both repairs and their declared checks completed successfully. The
[protocol](../number_reader_repair.md) was committed before fitting in `03cbc13`.
This is a reader-only intervention, with no training-depth extension.

## Model and training exposure

Source: `models/stage1_pointer/controller-affine-seed83/best`, adapter SHA-256
`8dfdaee726c0f6413ed2f15478abad68574eb63d8401bb3463bbdc09ab27ff7a`.
First export: `models/stage1_pointer/controller-shared-number-seed83/best`.
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

The fresh graph benchmark **passes every predeclared quality criterion**. All
189 actual native stopped calls match reuse, and the independent scalar reference
audit passes all 345,600 decisions and per-count/overall aggregates. There is no
overlap with the 37,638 excluded tables. Frozen inference files are unchanged.

| Metric | Result |
| --- | ---: |
| Overall answer plus exact first stop | 338,764 / 345,600 = **98.02%** |
| Overall complete nominal trajectories | 336,884 / 345,600 = **97.48%** |
| Exact first stop, every count 1–256 | **100%** |
| Complete trajectory through 256 | 1,314 / 1,350 = **97.33%** |
| Correct final answer at 256 | 1,322 / 1,350 = **97.93%** |
| Early / late / missing stops | **0 / 0 / 0** |
| Correct letter at wrong loop | **0** |

The graph-cluster bootstrap 95% interval is 97.30–98.69% for overall joint
success and 96.66–98.25% for complete trajectories. Controller outcomes are
graph invariant at a fixed count; its degenerate graph interval is not a claim
of certainty over untested integers. Minimum per-count joint success is 97.70%;
minimum trajectory success is 97.33%. All 36 failing graphs first err at loops
4–30; there are no additional first failures from 31 through 256.

Forced extraction executes 367,200 example-loops in **799.23 seconds**. The 189
native calls execute 17,325 transitions in **446.36 synchronized inference
seconds**. These have different compute scopes and are not a speedup comparison.
Graphs remain 26-state cyclic systems; this does not test 256 distinct states.

The separate diagnostic reads all integer requests through 10,000 and free-runs
the unchanged timer through 4,096. No R transition executes in that diagnostic.
It separates reader error from accumulated countdown drift and cannot establish
pointer quality at those depths. Maximum initial-reading error is **0.001953125**
through 10,000 (mean 0.0000647860). The copied timer stops correctly through
1,037, then first stops early at **1,038**; only 1,037/4,096 timings are exact.
Its gain 0.99999940395 and offset −0.99998086691 accumulate long-horizon error.
This preserves a negative result outside the graph benchmark and motivates the
separate precision intervention below. The first checkpoint remains untouched.

## Two-parameter precision repair

Protocol/configs were committed before fitting in **`704bfb8`**. Run:

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash refine_countdown.sh
```

The run used tmux session `number-precision`; logs and exit 0 are under
`eval/pointer_benchmark/shared-number-precision-20261008-launch/`.
Float64 L-BFGS fits only the scalar cell's weight/bias on the **original** 58
count labels and nominal min(N,12) free-running numerical targets. Initial memory
comes from the frozen learned reader. No gold memory, parsed number, elapsed-loop
feature or programmed update enters the forward. Reader and stop head stay
unchanged; no new count/depth training or R update occurs.

Ten objective evaluations reduce numerical training MSE from 4.163e−9 to
2.003e−12; after ordinary float32 export it is 6.308e−12. The learned weight is
**1.0** and bias **−0.9999999403953552**. Those values are optimization results,
not assigned constants. All **153 non-cell tensors** remain bitwise unchanged.

Final checkpoint:
`models/stage1_pointer/controller-shared-number-precision-seed83/best`.
Adapter SHA-256:
`0f8a3f2572ec2caf4fafbcdd074b5283bd8c610602502763b3c04487ce1f123d`.
It loads through the existing `loop_test` and `naive_test` interfaces. The latter
still forces the requested depth and does not test stopping.

Results are under `eval/pointer_benchmark/shared-number-precision-20261008/`:

- **8,192/8,192 exact numeric first stops**, including new rollout values
  4,097–8,192. No early, late or missing stops. No R execution in this diagnostic.
- Reader errors through 10,000 unchanged; maximum 0.001953125.
- All **256** graph-panel count timings match the first confirmed model.
- All **36/36** actual native stopped calls at counts 12/70/100/256 match the
  confirmed R trajectories on the nine declared graph strata.
- Frozen inference files remain unchanged; the composition check passes.

The final checkpoint inherits the **98.02% joint and 97.33% through-256
trajectory results** because R, bridge, reader, coda and stop readout are unchanged,
all count timings match, and native calls verify composition. This is component
composition on an opened, audited panel, **not another independent graph
confirmation**. It does not establish R or end-to-end pointer quality at 8,192.
The new numerical panel is deliberately separate from graph quality.

Reuse also requires matching inference architecture/base/options, tokenizer
bytes, and every recurrent-model source module from the original confirmation
commit. These checks guard against interpreting equal tensor payloads as equal
execution after a routing, configuration or code change. The identity audit is
saved beside the final composition result.

The measured generalization failures are resolved without expanding training
depth/count exposure. Scope remains fixed-format pointer execution through 256,
and a requested-count controller tested separately through 8,192. Eight input
characters and finite floating-point precision still bound the architecture;
no arbitrary-integer, cross-family, repair/damage or generic reasoning claim is made.

## Implementation checks

Sixteen initial reader/affine/benchmark tests passed. Thirty-six additional
architecture/prefix/controller-training/remaining-work checks passed. Five reader
checks subsequently passed, including the actual composing fit/export CLI,
unchanged non-reader tensors and rejection of a modified training-count panel.
The shared benchmark supports both old/new scalar controllers and its tiny-model
native/reuse tests include intentionally failed stopping. These are implementation
checks. The precision test exercises free-running supervision at the same
training ceiling and stopping at 70/256/1,038/4,096/8,192. The tiny composing
evaluator preserves negative results and rejects changed R tensors. Independent
audits also reproduce the old failed benchmark's 40,714 cyclic wrong-time cases.
Shell syntax, whitespace and local documentation links pass. The current runtime
uses float32; other inference dtypes are not verified by this experiment.
