# Diagnose recurrent failures and measure execution cost

Status: proposed next bounded work after the [negative loop-balanced ablation](experiments/stage1_loopbalanced.md). The source inspection below is complete; new diagnostic logging, GPU profiling, and optimizations are not implemented or benchmarked. Do not start another training sweep based on this proposal.

## Questions before another training change

We have reliable execution at trained depths and substantial nearby extension, followed by a steep failure boundary. Aggregate CE cannot distinguish transition errors, state deterioration, readout errors, or dependence on the requested depth. Equal-loop weighting did not solve that boundary. Use baseline step 2500, fixed development examples, and exact reference trajectories; keep seed 29 untouched.

| Question | Measurement | Decision it informs |
| --- | --- | --- |
| Where does correct execution first break? | Task-depth × loop conditional failure rate given a completely correct prefix, counts, and paired per-example outcomes across checkpoints | Separate accumulated earlier mistakes from a newly unreliable transition |
| What replaces the correct next state? | First-error categories: previous state, another earlier path state, a future nominal state, or off-path symbol; target rank, predicted margin, entropy, and top competing symbols before/at failure | Distinguish repeats, jumps, and competing lookups; high confidence is not correctness |
| Can the same remaining computation work from a fresh start? | Paired suffix tasks with the same rule table, start at a known reference state, and adjusted remaining Steps; compare with reaching that transition after a longer recurrent prefix | If suffix execution works while the original fails, investigate recurrent-history sensitivity; changed prompt/start makes this a diagnostic, not a causal proof |
| Does the requested depth alter early execution? | Paired same-table/start tasks with different valid requested depths, comparing shared-prefix predictions | Diagnose dependence on the depth cue while keeping prefix targets identical |
| Does state scale or direction change around failures? | On a small fixed probe set, per-loop hidden RMS, relative update norm and cosine change at the answer position plus sequence aggregates; compare successful/failed and shallow/deep cases | Localize state drift; norms do not establish what is represented or prove hidden collapse |
| Is optimization healthy? | Existing pre-clip gradient norms plus clipping frequency; sampled per-layer LoRA gradient/update norms and effective adapter contribution relative to base projection | Diagnose saturation, disproportionate updates, or inactive adapters before changing LR/rank; raw A/B norms alone are scale-ambiguous |

First-error categories and conditional rates can largely be derived from existing nominal traces. Save examples and cohort denominators, not just averages. Hidden-state and adapter probes should run on a small separate diagnostic batch at checkpoints, not retain every training graph or dump all hidden tensors. Any extra gradient-attribution experiment belongs in a separately measured diagnostic pass. Decoded correctness is not a certificate of a fully valid hidden state, and a readout probe must use held-out probe data to support representation claims.

Keep the terminal dashboard compact: step/ETA, recent loss, last validation with its update number, throughput and memory. Persist detailed metrics as JSONL/CSV, render curves after checkpoints, and record diagnostic overhead. An experiment tracker would improve presentation, but cannot supply the missing scientific measurements by itself. No new tracking dependency is selected yet.

## Concrete overhead found in the current code

- `loop_test.py` defaults to batch 1; all six recent full evaluations used it. Fixed-depth evaluation supports larger batches already. Adaptive stopped inference currently requires batch 1 and must be benchmarked separately.
- Training and checkpoint loading explicitly use eager attention; training and the full-loop CLI use float32. CUDA fast-path alternatives need explicit configuration, equivalence/gradient checks, and paired quality measurements.
- `RecurrentQwen.forward` projects the answer vector to all 151,936 vocabulary logits every loop, then `symbolic_scores` retains 26. A task-specific selected-row projection is mathematically sufficient for symbolic CE, but must be opt-in, preserve full-vocabulary Stage 0/ordinary behavior, and verify logits and adapter gradients under numerical tolerance. Its actual share of runtime is unmeasured.
- `evaluate` calls `batch_metrics` for the batch and again for each example, causing repeated `.item()`/`.tolist()` transfers. It also reads individual GPU losses with `.item()` inside the row loop. Consolidate detached statistics and transfer once per batch. CPU entropy/margin calculation and CSV construction need separate timing.
- Training extracts metrics and refreshes the dashboard each microbatch; input validation also evaluates GPU boolean reductions in Python. Preserve boundary validation and finite checks while measuring synchronization overhead and moving static checks to validated CPU inputs where appropriate.
- Mixed-depth training unrolls every row to the deepest item in a microbatch. Shorter rows have masked losses but still consume recurrent/coda compute. Log executed versus supervised transitions. Repartitioning within the existing eight-example optimizer group can be tested without changing sample groups; globally sorting by depth changes training order and is a separate intervention.
- The six-layer coda reads every loop and its operations must remain differentiable during training. Frozen weights do not make this path free. Do not detach it or remove intermediate readouts to claim a speedup while silently changing supervision.

These are code-level candidates, not evidence of GPU utilization percentages or a measured bottleneck ranking. VRAM occupancy alone is not compute utilization.

## Next bounded implementation and benchmark

1. Add existing-trace failure summaries and an opt-in short profiling mode. Separate input preparation/transfer, prelude, recurrent block, coda/readout, loss/backward, optimizer, metrics/serialization, validation, and checkpoint time. Use CUDA events or profiler ranges for device work and synchronized end-to-end intervals; asynchronous CPU launch time is not GPU duration. Warm up first; report repeated median/p95 timings, allocated/reserved/device memory, hardware/runtime, and profiler overhead separately.
2. Benchmark fixed-depth evaluation batches 1/4/8/16 on identical examples at 8 and 16 loops, subject to measured memory. Check predictions, logits/loss tolerances and any near-tie changes against batch 1. Record questions/s, example-loops/s and full wall time. Reuse a loaded checkpoint during the benchmark so repeated loading does not obscure inference cost.
3. Remove redundant metric synchronization and test restricted-symbol projection, each independently. Preserve full-depth trajectories and checkpoint compatibility. Verify paired correctness and gradients before benchmarking speed.
4. Profile training on disposable short runs with matched eight-example update groups: compare batch 4/accumulation 2 with batch 8/accumulation 1 if it fits. Batch 16 changes the effective batch and is not the same training experiment. Record valid versus executed transitions/s as well as optimizer updates/s.
5. Only then assess SDPA, BF16/TF32 or compilation as separate explicit execution choices. Do not loosen the float32 architecture gate or silently change old checkpoint defaults. Numerical and determinism effects must be reported alongside speed.

Use the results to choose one justified model/training intervention. No cache change, new loop counter, adaptive-head training, or longer training run is implied by this plan. The immediate need is diagnosis plus measured execution efficiency; the wider [adaptive roadmap](adaptive_compute.md) remains conditional on its research gates.
