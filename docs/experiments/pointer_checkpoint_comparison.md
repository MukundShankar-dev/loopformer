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
