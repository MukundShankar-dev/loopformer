# Frozen pointer benchmark — 2026-10-08

The user authorized opening the current reserved test and a larger independent
pointer benchmark after the controller repair. Freeze the first predeclared
controller, `controller-affine-seed83/best`, including R, prelude/coda, bridge,
initializer, recurrence and stop readout. No training, seed/checkpoint selection,
threshold fitting or repair based on these outcomes is permitted within this run.
The stop threshold is 0.5; the common safety cap is 272 for every request.
Inference-file hashes and the protocol/config hash are recorded before execution
and checked afterwards. Seed 29 remains unopened; this new authorization covers
the existing seed-61 test and seeds 211/223/227 only.

## Panels declared before results

1. **Existing test:** all 1,536 reserved queries on 128 graphs, depths 1–12,
   executed through the existing native learned-stop evaluator. No filtering.
2. **Independent benchmark:** seeds 211/223/227, each with 150 new graphs in
   each of three modes (random function, permutation, full cycle): 1,350 distinct
   graphs total. Evaluate every count 1–256, yielding 345,600 graph/count queries.
   Reject overlap with every old dataset split and with other benchmark graphs.
   Graph/start/rule order are sampled without consulting requested count.
3. **Native fidelity panel:** first graph in each seed/mode stratum, counts
   1/9/12/17/29/41/53/64/80/99/100/101/128/256: 126 actual learned-stop queries.
   This is fixed before outcomes and checks all replay scoring, including failures.

The controller saw most numeric requests through 63 during fitting, but recurrent
supervision ended at 12. Report 1–12, 13–63, 64–99 and 100–256 separately, plus
held-out values 9/17/29/41/53. Three-digit counts change token positions in its
fixed eight-token suffix and deliberately test that boundary; do not fix the model
if this fails. Graph kinds and orbit transient/cycle lengths are separate strata.
All graphs still have 26 states: long execution necessarily repeats states. This
benchmark does not establish nonrepeating paths with 50 or more distinct states,
cross-family transfer, arbitrary prompt wording or arbitrary integers.

## Efficient scoring and fidelity

Reuse shared model loading, `forward_symbols`, controller initialization/advance,
reference execution and `stop_rows`. The executor is structurally count-free and
the affine controller ignores executor observations. Run each new graph once
through 272 forced loops, batching graphs, then replay the frozen native controller
on each raw prompt's suffix for all 256 counts. This supplies exact nominal R/C
trajectories even if the controller stops early. No gold symbol or numeric target
enters inference. Check native panel predictions/timing/cap semantics against
these results and reject the benchmark on disagreement. This reuse measures
quality efficiently; it is not an adaptive latency benchmark or 345,600 native
independent graph executions. Record actual compute and no claimed speedup.

Measure final accuracy at the requested loop, complete nominal trajectory,
per-loop/conditional transition accuracy, first error, R/C agreement, exact first
stop, early/late/missing stop, stopped-answer accuracy, joint answer/timing, strict
trajectory-and-timing, and correct-letter/wrong-time coincidences. Missing stops
remain failures even if safety-cap execution lands on the correct letter.
Use graph-cluster bootstrap confidence intervals (2,000 repeats, seed 239) within
balanced seed/mode strata; repeated horizons do not multiply the independent
sample size. Per-count rates also carry marginal graph-level Wilson intervals,
which remain nondegenerate at zero/all successes; these are not simultaneous
confidence bounds over all depths. Report all counts and seeds without selecting a favorable range.

`configs/pointer_benchmark.json` declares descriptive acceptance thresholds:
95% overall joint success/complete trajectory, 99% exact stopping, with at least
90% joint/trajectory and 95% timing at every depth. Report point estimates and
confidence intervals; these are task-specific finite benchmark criteria, not a
universal claim. Existing test and each numeric range are reported independently.
Broader research stage gates, terminal repair/damage and confidence-based adaptive
allocation remain separate.

## Commands and implementation

```bash
bash benchmark_pointer.sh --dry-run
bash benchmark_pointer.sh
```

The launcher records/verifies the freeze, runs the existing test unless its
completed output already exists, then runs the independent benchmark. It refuses
to proceed past an incomplete test or overwrite independent benchmark outputs.
Use the complete checkpoint and old dataset on the CUDA desktop. All parameters
have gradients disabled for benchmark extraction, with inference mode and disk
hashes checked before/after. The existing native evaluator uses inference mode.
W&B is disabled for this reserved benchmark; results remain local/repository
artifacts. The benchmark is evaluative only and never fits an initializer/head.

Model mechanics stay in `scripts/recurrent_qwen/`; graph generation/count variants
are in `scripts/dataset/benchmark.py`, scoring/bootstrap in
`scripts/eval/benchmark_metrics.py`, and composing CLI in
`scripts/eval/frozen_pointer_benchmark.py`. Native evaluation still uses `loop_test`.
All 345,600 decision rows are compressed into `decisions.csv.gz`; per-graph
predictions, per-count/stratum/loop rates, controller counts and native traces are
saved as CSV. The graph JSONL remains under `data/pointer/benchmark-seeds211-223-227/`;
its hashes, seeds, exclusions and orbit metadata are in the tracked output manifest.

Six focused tests cover, including seeded generation and overlap rejection,
cyclic wrong-time letters, missing-cap failures, recovery without a perfect
trajectory, clustered horizons, unchanged exported weights, and a tiny end-to-end
native/reuse check with deliberately failed stopping. The actual pretrained run
is in progress; success and performance remain unmeasured until its report.
