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
trajectories and raw tables, without further inference. Results follow execution.

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
