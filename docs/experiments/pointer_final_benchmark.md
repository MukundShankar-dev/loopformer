# Final checkpoint full benchmark — 2026-10-08

Status: declared before execution; results pending.

The user requested the full benchmark after the shared-reader/countdown repair.
Freeze `models/stage1_pointer/controller-shared-number-precision-seed83/best`
before any new evaluation. Adapter SHA-256 is
`0f8a3f2572ec2caf4fafbcdd074b5283bd8c610602502763b3c04487ce1f123d`.
No fitting, threshold tuning, checkpoint selection, or changed training exposure.

## Protocol

Run all 1,536 seed-61 test queries on 128 graphs at depths 1–12 using native
learned-stop inference. This previously opened split is a regression check.
Then generate 1,350 independent graphs: fresh seeds 307/311/313, 150 per seed
and graph mode (random function, permutation, full cycle). Reject tables from
every seed-61 split and both previously opened benchmark panels (seeds
211/223/227 and 281/283/293). Evaluate every integer count 1–256, giving 345,600
paired graph/count queries. All 26-state graphs eventually repeat states.

Use the existing frozen evaluator: count-free executor trajectories reused over
requested counts, independent controller replay, threshold 0.5, safety cap 272,
and the original quality criteria. Execute the original 21 reader-repair native
counts on the first graph in each of nine strata (189 actual stopped calls).
Require every native decision to match replay, independently audit every saved
decision against raw graph traversal, and verify inference file hashes before
and after. Batch size 64 changes execution scheduling only; native batch-one
checks guard reuse/floating-point decision fidelity. Bootstrap 2,000 times
within seed/mode strata with seed 239; graphs are the independent sampling unit.

This evaluates the final checkpoint directly on fresh graphs, rather than
inheriting the earlier reader-only checkpoint's executor results. Pointer quality
above depth 256, nonrepeating paths longer than 26 states, prompt variation,
cross-family transfer and adaptive latency remain outside this benchmark.
Seed 29 and later research stages remain closed. W&B is disabled.

## Commands and artifacts

```bash
bash benchmark_pointer.sh --full --dry-run
bash benchmark_pointer.sh --full
```

Config: `configs/pointer_full_benchmark.json`. All tracked results go under
`eval/pointer_benchmark/final-full-20261008/`; generated graph JSONL/manifest
go under `data/pointer/benchmark-seeds307-311-313/` on the desktop.
Weights remain on the desktop. The launcher preserves the older benchmark's
default command and refuses overwriting completed independent outputs.

Plots use the optional pinned Matplotlib dependency, without model loading:

```bash
python -m scripts.eval.plot_pointer_benchmark \
  --results eval/pointer_benchmark/final-full-20261008/independent
```

Results, measured timing, validations and figure interpretation will be recorded
here after the run. No performance or acceptance outcome is established yet.
