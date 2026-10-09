# Paired checkpoint failure diagnostics

## Declared protocol, 2026-10-08

The user requested failure-mode matrices and earlier architecture comparisons
following the [full frozen benchmark](pointer_final_benchmark.md). This is an
outcome-independent subset of an **already opened** panel for diagnosis, not
independent confirmation or checkpoint selection. No fitting or threshold tuning.

`configs/pointer_checkpoint_comparison.json` declares five retained checkpoints:
the historical depth-12 coupled LoRA checkpoint at update 5,000 (selected in its
original experiment), broader-count GRU, positional affine controller, shared
number reader before precision repair, and final precision checkpoint. Use the
first three graphs of every seed/mode stratum from seeds 307/311/313: 27 identical
graphs and 30 explicitly listed counts, 810 queries per checkpoint. All graphs
receive all counts. Threshold 0.5; safety cap 272 for every model. Record inference
hashes before testing and recheck afterwards. CUDA/FP32/SDPA, forced batch 16;
native learned-stop calls use batch one.

The four isolated models must have bitwise identical non-controller tensors,
non-controller inference specification, tokenizer bytes and recurrent code as
the audited source benchmark. Only then reuse count-free executor trajectories.
The GRU receives actual captured R vectors and separately computed context for
**each full raw prompt**, never zero observations or broadcast initialization.
Scalar replay verifies graph-invariant raw prompt reading. Each isolated model
additionally runs 54 real stopped calls: first graph per stratum at counts
1/9/12/70/100/256. Replay must match positive and negative outcomes, including
cap fallbacks; native traces undergo independent raw-rule auditing.

Legacy R sees Steps: run all 810 actual stopped calls and separately compute
forced nominal trajectories for each requested count. Do not share trajectories
across counts. Batched forced and native prefixes must agree wherever both
exist. All native decisions and trajectories receive independent auditing.

Measure forced final-answer accuracy, complete nominal trajectory, exact/early/
late/missing stop, actual stopped-answer accuracy, joint answer/exact-stop success,
strict complete-trajectory/exact-stop success, and correct letter at the wrong
time. Missing stops remain distinct from stopping at the cap. Save per-query
forced traces/targets, per-count rates, identity/fidelity audits and hashes.

The legacy LoRA model differs in architecture, capacity, data and training recipe:
its comparison is descriptive, **not a causal ablation**. Isolated models provide
a controlled frozen-executor controller comparison. No model selection uses this
panel; its 27-graph rates must not replace the 1,350-graph benchmark.

## Commands

On the desktop, from the repo with its CUDA environment:

```bash
python -m scripts.eval.checkpoint_comparison --dry-run
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u -m scripts.eval.checkpoint_comparison
```

Outputs: `eval/pointer_benchmark/checkpoint-comparison-20261008/`. Existing paths
are refused. Full-benchmark failure matrices are computed offline from audited
trajectories and raw tables, without further inference. The run is now complete.

## Full-panel execution failure signatures

These offline measurements use **all 1,350** final-benchmark graphs, independently
traversed from their raw rule tables. They do not use the smaller model-comparison
panel. The cumulative green complete-trajectory curve is flat after loop 30
because all 42 graphs that ever first fail have already failed. Remaining graphs
have no first error through 256. Later recoveries do not restore strict-prefix
success, and later errors on already-failed graphs do not decrease it again.

The decoded trajectories remain active: there are **234 wrong-to-correct
transitions** and **222 subsequent entries into another error episode**. Twelve
of the 42 failing graphs have a correct final letter at 256. Thus the plateau
must not be interpreted as constant decoded states or an absence of later errors.

At the first failure, direct R and C decode the same wrong symbol on **37 graphs**;
on **five**, the direct R readout is correct while C is wrong. No first failure
simply repeats the immediately previous reference state. Mutually exclusive
categories, with the declared priority, are: 16 earlier visited states, 17 later
reachable states, nine states outside the start-reachable orbit. Thirteen first
wrong symbols equal the next reference state's successor, consistent with a
one-step phase displacement at that readout, without proving an extra internal
transition. The most frequent first wrong symbol is K (5/42); errors span 18
letters. This small table cannot establish or exclude a general symbol bias.

Among **7,544 transitions immediately following a wrong C decode**, 6,912
(**91.62%**) follow the table's edge from that prior decoded symbol while remaining
wrong against the original reference path. There are 234 recoveries (**3.10%**)
and 398 other wrong transitions (**5.28%**). These observations support coherent
propagation of many decoded errors, rather than wholly random later outputs.
They are not a causal test of what R's hidden state represents. Repeated graph
loops are dependent; these percentages are not independent-trial estimates.

Cycle-period strata and per-letter final-error rates retain their graph
counts/denominators. Small strata are noisy; pooling graph types confounds cycle
structure with other properties. No causal conclusion is drawn from the heatmap.


## Completed paired comparison

All five arms completed on the RTX 5070 Ti desktop. The measured arm scopes
sum to 736.93 seconds (about 12.3 minutes), including model loading, identity
checks, forced execution/replay and native checks; this is not matched inference
latency across models. Inference ran at Git `b674d70`; final aggregation was
repaired at `6cd1c68` after a CSV column mismatch between legacy and replay
schemas. Every arm had completed and passed native checks before that export
error. `--assemble-only` validated the frozen hashes and rebuilt the combined
CSV from saved arms, without rerunning any inference.

Each row below averages **810 paired queries on 27 graphs**, with equal weight
for the 30 declared count columns. These are descriptive panel rates, not the
full 345,600-query benchmark rates; the smaller panel happens to include two
failing executor graphs. Repeated count variants are dependent.

| Frozen checkpoint | Forced final | Complete nominal trajectory | Exact stop | Answer + exact stop | Correct letter, wrong time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coupled LoRA | 44.07% | 38.64% | 35.93% | 35.31% | 9.88% |
| Isolated R + GRU | 94.81% | 94.44% | 23.46% | 22.96% | 13.46% |
| Positional reader + affine | 94.81% | 94.44% | 73.33% | 70.00% | 3.95% |
| Shared reader before precision | 94.81% | 94.44% | 100.00% | 94.81% | 0.00% |
| Final shared reader + precision | 94.81% | 94.44% | 100.00% | 94.81% | 0.00% |

The old LoRA model stops early for all tested requests above 12, typically near
10–11 at large counts. Its forced final accuracy is 4/27 at count 16; complete
trajectories are zero at every tested count from 17 onward. Both execution and
control fail. Differences from the new executor are confounded by data,
capacity, bridge, routing and training recipe; this is not a LoRA-only ablation.

All four isolated arms preserve exactly **148 non-controller tensors**, the
non-controller specification, tokenizer bytes and recurrent code. Their forced
R/C predictions are identical. The GRU has poor exact timing even at many short
counts despite the strong frozen executor. The positional affine controller is
exact at all sampled counts through 69 and early for every sampled count from
70 onward. Shared reading eliminates this format-dependent failure through 256.
Precision repair changes no pointer outcome on this panel: its benefit is at
larger numerical requests, not an additional executor improvement.

All **1,026 native stopped calls** pass their applicable fidelity checks:
810 legacy calls plus 54 for each isolated arm. Independent native audits check
**18,482 transitions**. All 7,289 legacy native decoded steps match the overlapping
batched forced prefixes. GRU live/replay maximum logit difference is recorded in
`gru/feature_fidelity.json`; its actual R vectors produce exactly the confirmed
source C predictions. All model inference hashes remain unchanged. The additional
independent audit passes **all 4,050 combined decisions**, coverage and aggregates,
including wrong-time letters, missing-stop semantics and strict-prefix failures.

The final/native fidelity subset scores 49/54 (90.74%): it is only nine graphs,
one of which is wrong at all five selected horizons above one. This is a fidelity
panel, not a replacement population estimate or contradiction of the full result.
The full frozen model's reported 97.81% joint success remains unchanged.

## Eight new figures and machine-readable evidence

All figures are unsmoothed and saved as PNG/PDF/SVG under
`eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/`.
The provenance records input, source and export hashes. CSVs retain first-error
categories, first-error confusion, post-error dynamics, final-letter denominators,
cycle-period/loop rates, and architecture/count rates.

- [Every failing graph × recurrent loop](../../eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/failed_trajectory_matrix.png)
- [First-error symbol confusion and final-letter error rates](../../eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/symbol_confusion.png)
- [First-error categories, R/C agreement and propagation/recovery](../../eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/failure_signatures.png)
- [Cycle period × loop error rates](../../eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/cycle_period_matrix.png)
- [Architecture × count matrices for four quality metrics](../../eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/architecture_metric_matrices.png)
- [Requested count × actual stopping-loop matrices](../../eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/requested_actual_stop_matrices.png)
- [Wrong-time letters and premature stops](../../eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/cyclic_coincidence_and_early_stops.png)
- [Saved controller-only numerical precision comparison](../../eval/pointer_benchmark/final-full-20261008/independent/failure_matrices/numeric_precision_comparison.png)

The numeric figure reuses previously measured results, with checkpoint hashes
matched to the current freeze: the pre-precision controller is exact for
1,037/4,096 requests and first fails at 1,038; the final is exact for 8,192/8,192.
No R is executed in those numerical diagnostics. Pointer quality above 256 remains
untested. The stopping matrix uses a labeled logarithmic color scale for nonzero
probabilities so dispersed GRU failures remain visible; blank cells mean zero.

Combined decisions, per-arm traces, audit files, freeze and the original launch
log are in `eval/pointer_benchmark/checkpoint-comparison-20261008/`. Checkpoint
binaries remain on the desktop. The launch log preserves the export error;
`export_recovery.json` records the successful assembly/audit without inference.

To audit and plot a fresh reproduction after the GPU comparison completes:

```bash
python -m scripts.eval.audit_checkpoint_comparison \
  --graphs data/pointer/benchmark-seeds307-311-313/graphs.jsonl

python -m scripts.eval.plot_pointer_failures \
  --graphs data/pointer/benchmark-seeds307-311-313/graphs.jsonl \
  --comparison eval/pointer_benchmark/checkpoint-comparison-20261008 \
  --numeric-results eval/pointer_benchmark/shared-number-20261008/numeric \
                    eval/pointer_benchmark/shared-number-precision-20261008/numeric \
  --output eval/pointer_benchmark/final-full-20261008/independent/failure_matrices
```

Optional plotting dependencies are in `requirements-plots.txt`. Mac rendering
used `MPLCONFIGDIR=/tmp/loopformer-mpl-cache` and an identical raw graph file
copied to `/tmp/loopformer-benchmark-graphs/graphs.jsonl`. Existing output/audit
paths are refused; choose fresh paths or update a copied protocol for reproduction.
For a completed-arm export failure only, `python -m scripts.eval.checkpoint_comparison
--assemble-only` verifies hashes and assembles without loading models.

Validation: eight benchmark tests and nine comparison/failure-metric tests passed, including
changed-executor rejection, cyclic wrong-time failure, recovery versus complete
trajectory, mutually exclusive cyclic categories, post-error denominators and
corrupted-audit rejection, mixed export schemas and assembly rejection after a
checkpoint changes. The existing comparison tests remain preserved.
Independent reference audits, figure layout inspection, export/provenance hashes,
local links, compilation and whitespace checks pass. No training or new stage is
selected by these descriptive diagnostics. Seed 29 stays closed.
