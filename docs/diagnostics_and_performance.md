# Diagnose recurrent failures and measure execution cost

Status: initial diagnostics and short profiler implemented after the [negative loop-balanced ablation](experiments/stage1_loopbalanced.md). Offline first-failure analysis has run on retained traces. Paired model probes and pretrained CUDA profiling completed on the desktop; see the [results and limitations](experiments/baseline2500_diagnostics.md). The broader measurements below distinguish implemented interfaces from future work; no optimization or new training sweep has been run.

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

- `loop_test.py` defaults to batch 1; the full scientific evaluations used it. A subsequent 128-example desktop benchmark measured 40.93/17.13/12.62/9.75 seconds at batches 1/4/8/16, with identical predictions across all 2,048 loop readouts. Batch 16 was 4.20× faster in that single 16-loop comparison; memory and timing variance were not recorded. Fixed-depth evaluation supports larger batches already. Adaptive stopped inference currently requires batch 1 and must be benchmarked separately.
- Training and checkpoint loading explicitly use eager attention; training and the full-loop CLI use float32. CUDA fast-path alternatives need explicit configuration, equivalence/gradient checks, and paired quality measurements.
- `RecurrentQwen.forward` projects the answer vector to all 151,936 vocabulary logits every loop, then `symbolic_scores` retains 26. A task-specific selected-row projection is mathematically sufficient for symbolic CE, but must be opt-in, preserve full-vocabulary Stage 0/ordinary behavior, and verify logits and adapter gradients under numerical tolerance. The measured evaluation trace assigns about 19.5 ms of a roughly 1.1 s batch to this head; it is not the leading optimization target for that workload.
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

## Implemented diagnostic commands

The first implementation adds three opt-in CLIs. It does not change the optimizer objective, recurrent architecture, attention implementation, precision or evaluator batch default. New ordinary full-loop exports add `target_rank` (descending logits, A–Z tie break) and `top_symbols` (top three) alongside the existing confidence fields. Historical exports remain readable, with unavailable fields blank.

### Existing traces: no GPU required

```bash
python -m scripts.eval.diagnose_pointer \
  eval/pointer_loops/20260911T201847.679922Z-depth6-fresh30k-seed37-batch4-step-002500 \
  --data data/pointer/seed-17/depth_test.jsonl \
  --output eval/pointer_diagnostics/baseline2500-depth9to16
```

This command has already run locally. It checks the source summary/data hash, unique complete trace coverage, reference targets and correctness before reporting first failures. `first_failures.csv` contains classifications and available confidence before/at failure. `conditional_failures.csv` contains per-depth/per-loop correct-prefix denominators, first-failure counts and rates; an empty risk set has an undefined rate. `summary.json` records provenance and totals. No model loading or inference occurs. All-correct cohorts have no `first_failures.csv` and an explicit zero failure count.

For the retained baseline's 1,000 depth-9–16 development examples, 164 have complete trajectories. Among 836 first failures, 191 repeat the preceding reference state, 317 predict another earlier path state, 163 predict a future nominal state, and 165 predict an off-path symbol. These mutually exclusive categories depend on the non-repeating nominal reference. They describe decoded errors, not hidden-state causes. The exact results are saved under the output path above.

### Paired probes: small real checkpoint run

Run on the desktop with the complete baseline checkpoint:

```bash
python -m scripts.eval.probe_pointer \
  --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 \
  --data data/pointer/seed-17/depth_test.jsonl --device cuda \
  --limit 32 --restart-after 6 \
  --output eval/pointer_probes/baseline2500
```

The fixed prefix is selected before observing failures (four examples at each depth 9–16 with this dataset). Each task gets three forwards: original; suffix starting at the known reference state after step six with the remaining requested depth; and the same original start/table with requested depth reduced by one. Exact transformed tasks are saved in `tasks.jsonl`. `pairs.csv` compares aligned transition predictions and flags original first errors. `states.csv` records target rank, top symbols, confidence, hidden RMS, update RMS, relative update norm and cosine change at the answer position and over the whole unpadded sequence. Zero-norm ratios/cosines are undefined, not fabricated zeros. Hidden tensors are retained only for one example's forward at a time and never exported. `h0` is used only for state-change measurement and has no supervised start-state readout assertion.

Changed prompts are a confound: successful suffix execution suggests history sensitivity, but cannot by itself prove hidden-state drift. Depth-cue probes test common-prefix invariance on shortened prompts, not all possible depth variations. The baseline's original correct-prefix condition is recorded explicitly; subsequent wrong-prefix outcomes must not be described as first-transition failures. This probe is not a throughput benchmark.

### Short profiler: warmed evaluation and disposable training

```bash
python -m scripts.eval.profile_pointer \
  --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 \
  --data data/pointer/seed-17/depth_test.jsonl --device cuda \
  --mode eval --batch-size 16 --loops 16 --limit 128 \
  --output eval/pointer_profiles/baseline2500-eval-b16

for batch in 4 8; do
  python -m scripts.eval.profile_pointer \
    --model models/stage1_pointer/depth6-fresh30k-seed37-batch4/step-002500 \
    --data data/pointer/seed-37-depth6-30k/train.jsonl --device cuda \
    --mode train --batch-size "$batch" --effective-batch 8 --train-max-depth 6 \
    --output "eval/pointer_profiles/baseline2500-train-b$batch" || break
done
```

All output directories must be new. The scripts default to cached models, eager float32, strict deterministic algorithms, seed 17, four CPU threads; set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the CUDA shell. `--download` is explicit. CPU mode exists for tiny implementation checks; GPU attribution requires a CUDA run. No fallback device or precision changes occur.

The profiler loads once, warms up twice, and measures five synchronized repeats (`--warmup`/`--repeats` configurable). It reports individual samples, median, nearest-rank p95 (with five samples this is the maximum), examples/s, supervised/executed transitions/s, and CUDA allocated/reserved peaks plus device free/total memory. CPU memory/device utilization sampling is not implemented. Evaluation timing includes collation, forward, scoring, aggregation and CSV output but excludes loading/encoding. A separate one-batch trace compares the same interval with/without profiler; profiler teardown/export are excluded from that ratio. Do not add nested inclusive CPU/GPU range times together.

Training mode selects one seeded balanced eight-example group, resets adapters outside each measurement and constructs a fresh AdamW optimizer (LR 0.0002, no weight decay, clipping 1). It measures one disposable example-mean update, including first-step optimizer allocation; it is not a resume, a training experiment, or steady-state trainer throughput. Both microbatch partitions use identical IDs and effective batch. A separate diagnostic update saves adapter parameter gradient/update norms and its overhead; these factor norms do not measure effective LoRA contribution relative to the base projection. Original checkpoint files remain untouched, no new checkpoint is saved, and no validation set is trained on.

Outputs: `summary.json`, `operators.txt`, `ranges.json`, and optional-to-share `trace.json.gz`. The trace contains CPU/CUDA operator events and ranges for input preparation/transfer, prelude, recurrent block, coda, LM head, loss, backward/optimizer in training, and evaluation metrics/export. Large traces under `eval/pointer_profiles/` are ignored by Git; summaries and tables remain trackable. Warmed throughput uses the unprofiled repeats. Training validation, dashboard, and checkpoint serialization costs are outside this first profiler's scope; no end-to-end training speedup claim follows from it.

The paired probes and CUDA profiler have not been run on pretrained weights locally. Effective adapter contribution ratios, per-loop gradient attribution, optimizations, and full training-phase timing remain future work contingent on these measurements.

Implementation validation: the full local suite passed 119 tests in 115.18 seconds, including offline CLI integrations with a tiny saved Qwen checkpoint. Tests verify reference/risk-set accounting, paired target semantics, hook removal and forward equivalence, frozen parameters, and unchanged checkpoint bytes. These checks do not certify CUDA performance or pretrained probe behavior.

### Desktop shell wrapper

Run `bash profile_pointer.sh` from the repository (or invoke the script by path). The wrapper changes to its own repository directory, activates `.venv`, sets the CUDA determinism environment, checks the baseline checkpoint metadata/weights, runs train profiles at batches 4 and 8, then the paired probe. Each invocation uses timestamp/PID output directories so earlier results are preserved. Combined stdout/stderr also goes to `profile-pointer-debug.txt` (replaced on each invocation). Any failed command stops the remaining work and returns a nonzero status through `tee`. The checkpoint variable and `--output` spelling are corrected. Shell syntax and stubbed success/failure execution were checked; no pretrained run was launched.

## Saved-artifact follow-up

The [offline review](experiments/baseline2500_offline_review.md) records failure-aligned confidence measurements, checkpoint pairing, the supervision audit, and proposed controlled interventions. Rich confidence/state fields are available only for the 32-example probe; later historical checkpoints contain target margins alone. The artifact analysis reuses the existing reference validator and runs without model inference.

## Controlled restart and rule-context experiments

Implemented, awaiting pretrained desktop results. Use the same step-2500 checkpoint; no training or checkpoint mutation occurs. On the desktop, from the repository:

```bash
git pull --ff-only
bash probe_controls.sh
```

The wrapper activates `.venv`, checks checkpoint files, sets the CUDA determinism environment, and runs the first 32 examples of both seed-17 validation and depth_test. Validation covers depths 1–8; depth_test covers 9–16. It uses float32/eager batch-1 inference and does not download missing models implicitly. Combined terminal output is saved in `eval/pointer_probes/controls-small-<UTC timestamp>-<PID>/run.log`; results go into `validation/` and `depth_test/` beneath that directory. Any failure stops the wrapper with a nonzero exit status. No output directory is reused.

Review the small outputs before expanding. To run all 1,000 examples in each split later:

```bash
bash probe_controls.sh full
```

This remains development evaluation; neither command accesses seed 29. Both modes use the same implementation and variant definitions. Neither launches training or performance profiling.

The existing probe CLI enables controls with `--controls`. Without that flag, its original three-variant behavior remains available. With controls, every task receives original, rule-refresh and rule-no-op runs. Tasks deeper than `--restart-after` additionally receive the original suffix and shortened-depth probes and the new `suffix_original_steps` run.

- **suffix_original_steps:** same oracle-reference suffix start/table/targets as `suffix`, but the displayed Steps remains the original task depth. Execute exactly the remaining number of loops. This deliberately inconsistent instruction/horizon is a diagnostic: it separates two suffix prompts, not a normal task score. `tasks.jsonl` preserves a valid reference task separately from the actual `model_prompt`, input token IDs/count, answer position, rule-prefix boundary, displayed Steps and scoring horizon. Compare `suffix_steps_cue` directly to isolate this text change; token-length differences remain visible rather than silently padded away.
- **rule_refresh:** run the original prompt uninterrupted. Before each R call after loop six, replace only the contiguous Rules prefix with its frozen prelude output h0. Start, Steps and answer positions retain their current recurrent state. The prefix includes the rule-line newline and ends immediately before Start (Qwen merges the closing parenthesis and newline into one token); a tokenizer token crossing it causes an error. R and C then operate normally on the resulting sequence. No reference intermediate answer enters this intervention.
- **rule_noop:** the same hook timing and concatenation, copying the current rule prefix instead of h0. Exact equality of all exported observations with the untouched original is required; refresh must also match every untouched initial loop. A mismatch aborts the run. Tasks of depth six or less never activate refresh and are negative controls; validation depths seven/eight test nearby execution with refresh active.

`states.csv` saves all loop readouts/confidence/hidden scalars; `examples.csv` saves complete-trajectory outcomes and first-error/prefix lengths. `pairs.csv` aligns equal reference transitions, labels its baseline variant, and retains the baseline correct-prefix condition. Summary counts include changed predictions, correct-to-wrong/wrong-to-correct steps and gained/lost fully correct aligned segments; `quality_by_depth` includes denominators. Suffix completeness describes a shorter, oracle-started task and must not be equated with original full-task completeness. For `suffix_steps_cue`, the baseline is `suffix`, not `original`.

The hooks are scoped to one eval/no-grad forward and removed on exceptions. Production model/training interfaces are unchanged. Fresh rule context is an out-of-distribution intervention; a negative result does not rule out persistent-memory architectures. Positive results would justify further controls, not establish a new research gate. State-update summaries around refresh include the intervention's effect and must not be interpreted as unmodified recurrent dynamics or as direct cross-variant hidden distances.

Local validation: 122 tests passed before the final tokenizer boundary adjustment; the final 9-test diagnostic suite passed afterward, including prefix-only intervention, no-op equivalence, exception cleanup, altered-prompt scoring, checkpoint preservation and shell logging/failure propagation. All 2,000 real development prompts passed boundary checks using the cached Qwen tokenizer. Pretrained CUDA behavior and runtime remain unmeasured for these controls.
