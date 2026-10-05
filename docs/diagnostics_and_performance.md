# Diagnose recurrent failures and measure execution cost

## Controller training audit — current desktop command

After the [remaining-work comparison](experiments/controller_remaining_seed61.md),
inspect fitting and supervision before changing the model or training recipe.
This audit performs **zero optimizer updates** and does not load Qwen, regenerate
features, change inference, or touch confirmation splits.

```bash
git pull --ff-only
bash audit_controller.sh
```

The launcher activates `.venv`, defaults to CUDA, displays Rich progress/ETA and
saves stdout/stderr in a timestamped sibling log under `eval/pointer_diagnostics/`.
It exits on failure. Results go into a fresh `controller-training-audit-<timestamp>/`
directory. W&B defaults to online project `loopformer`, using the existing
retrospective uploader; all local results are saved before upload. Use
`WANDB_MODE=disabled bash audit_controller.sh` to disable tracking.

### Required local artifacts and scope

Keep `data/pointer/seed-61-independent/`, the original
`models/stage1_pointer/controller-seed61/features.pt`, and all six runs under
`models/stage1_pointer/controller-remaining-seed61/`. Each run needs its existing
metadata, `best_controller.pt`, `best_remaining_readout.pt`, and
`last_controller_state.pt`. These ignored tensors already exist on the training
desktop; pulled metadata on the Mac is insufficient. Missing inputs fail instead
of triggering downloads, extraction or retraining. The exported full Qwen/R
weights are not loaded or required by this audit.

All three seeds and both arms are included. Both **best and final** saved heads
are evaluated, without picking a new winner. Full fit evaluation covers all
18,432 cached training questions (2,048 graphs × nine counts) and 2,048 validation
questions (128 graphs × counts 1–16). Training and validation graphs are disjoint.
The deep 13–64 extraction is not repeated; this audit targets training mechanics
and the nearby generalization failure.

The gradient panel uses seed **107** to select 32 graphs independently from each
split, retaining all count variants. Every checkpoint sees the same panel. Report
training, validation trained counts, validation interpolation and validation
extrapolation separately. Held-out-count gradients use labels only for retrospective
measurement; they never drive updates, selection, or threshold calibration.

### Measurements and interpretation

- Full-panel fit reuses the existing first-crossing evaluator: exact/early/late/
  missing stops, answer accuracy and joint correctness by count/cohort, plus initial,
  trajectory, terminal and decrement errors for the numerical readout. A correct
  letter at the wrong loop remains a failure, including cyclic coincidences.
- Per-example records identify the first remaining-work error, with loop zero
  distinguished from recurrent loops. Numerical correctness uses absolute error
  <=0.5; observational zero crossing is the first predicted value <=0.5 after a
  loop. Neither criterion changes the deployed stop head or its 0.5 threshold.
- Cross-tabs pair exact stopping with initial-count accuracy, complete nominal
  numerical-trajectory accuracy, and exact zero-crossing timing. Summaries include
  conditional stop accuracy with explicit denominator counts and the mean first
  numerical-error loop among failures. Correct numerical
  decoding with incorrect stopping is visible instead of hidden in separate means.
- Gradient measurements decompose the **existing** loss into stop BCE, initial
  remaining-work error, interior remaining-work error, and terminal error. The
  latter three retain the original per-example `1/(N+1)` and `1/12²` scaling and
  sum to the original auxiliary loss. Gradients retain full controller BPTT.
- Record raw norms and pairwise cosine similarities for initialization, observation,
  GRU, stop readout, numerical readout, and the combined controller. Save both each
  fixed minibatch and the example-weighted panel-mean gradient; a norm of the mean
  is different from an average of norms. Zero-norm cosines are blank, not zero.
- The actual weighted controller objective and its hypothetical norm-1 clipping
  factor are included. In stop-only controls the numerical gradients into memory
  are explicitly **counterfactual** and receive zero weight in that objective.
  Their passive numerical readout was fitted separately during the original run.

These are raw gradients at saved checkpoints, **not Adam parameter updates** or a
reconstruction of historical optimizer batches. Gradient batch size defaults to
64, versus original training batch 256. Conflicting gradients can motivate a
controlled experiment but do not prove that conflict caused failure; decodability
does not prove that a scalar countdown drives the native stop policy.

### Outputs and verification

- `summary.json`, `run.json`: per-checkpoint fit summaries, settings, limitations,
  source/input hashes and explicit no-mutation checks.
- `fit_metrics.csv`, `predictions.csv`: full train/validation metrics and example
  records, including first numerical error and wrong-time correct letters.
- `countdown_stop_crosstab.csv`: paired numerical correctness versus stop timing.
- `gradient_panel.csv`: exact sampled graph/count identities.
- `loss_components.csv`, `gradient_norms.csv`, `gradient_cosines.csv`: per-batch and
  panel-mean objective contributions, norms, clipping factors and alignment.

The audit validates cache/data/panel identity and graph separation, matches the
original controller against cached logits on bounded samples, and verifies best
checkpoint/readout selection steps and saved validation metrics. It asserts
bitwise unchanged in-memory weights and untouched parameter gradient buffers,
and rehashes input files after execution. Existing outputs cannot be overwritten.

Preview with no artifacts, writes or model loading:

```bash
bash audit_controller.sh --dry-run
```

The launcher forwards CLI options, for example `--device cpu`, `--checkpoints best`,
`--gradient-graphs 16`, or `--output eval/pointer_diagnostics/controller-audit-recheck`.
Defaults constitute the intended full audit; reducing graph counts changes the
measurement panel and is recorded. No additional dependencies are required.

## Controller diagnostic — current desktop command

The [seed-61 audit](experiments/stage1_executor_seed61.md) shows strong forced
execution and unsuccessful stopping. The user authorized a bounded diagnosis
with the executor frozen, before another production training run. That diagnostic
has now [completed and been audited](experiments/controller_seed61_diagnostic.md).
The command below reproduces it; another identical run is not the next recommendation.

On the CUDA desktop, with the existing data and full checkpoint:

```bash
git pull --ff-only
bash diagnose_controller.sh
```

The wrapper selects `executor_r-seed61/best_checkpoint.json`, requires the local
weights, and runs `scripts.eval.controller_diagnostic`. It logs stdout/stderr to
`eval/pointer_diagnostics/controller-<run>-<timestamp>-<pid>.log`, propagates failures,
and shows Rich phase progress/ETA refreshed once per second. The log contains
terminal redraw sequences at this bounded rate. W&B defaults to online project
`loopformer`; `WANDB_MODE=disabled bash diagnose_controller.sh` disables upload.
W&B receives the completed metrics and CSVs retrospectively, not live fit curves
or upload-time GPU statistics presented as experiment utilization. Local results
are saved before upload, so a tracking failure does not discard the experiment.

### Panel and isolation

Using seed 71, choose 64 distinct **validation** rule tables and partition them
before making count variants: 32 fit graphs, eight probe-selection graphs, and
24 held-out diagnostic graphs. Every graph/start gets every requested count 1–16
and runs all 16 loops, including after a requested stop. This balances both count
and elapsed-loop labels instead of conditioning on examples that have not stopped.
All three groups remain development data. The reserved test split is untouched.

Extraction uses the existing recurrent model, selected-symbol readout, FP32/SDPA
and batch 16. Scoped hooks observe the actual prelude input to the controller,
initialized memory, R observation at each loop, controller memory and stop logits.
They cannot replace activations. A separate controller copy must replay the saved
inputs and match the live logits/memories (absolute tolerance 2e-5, relative 1e-4).
The source checkpoint hash is checked before/after extraction. All Qwen weights
stay frozen; no source checkpoint is overwritten or promoted.

### Checks and metrics

1. **Count access:** linear and small MLP probes decode requested count from the
   full-prompt prelude vector and initialized controller memory. Standardize using
   fit graphs only; select probe weights by dev CE; report graph-held-out accuracy
   and confusion matrices. A shuffled-fit-label linear probe is a negative control.
2. **Memory:** separate linear/MLP count probes at loops 1, 4, 8, 12 and 16, plus
   elapsed-loop probes on pooled memories. All counts occur at every loop. Separate
   per-loop count probes allow the representation to change with recurrence.
3. **Tiny-set learning:** compare the original controller to copies trained from
   its existing weights and from fresh initialization. Each fit uses eight of the
   fit graphs at the nine original training counts (72 questions). Full-batch AdamW,
   LR 0.001, no weight decay, norm clip 1, 1,000 updates. Use the existing per-example
   class-balanced completion BCE: continue before N, stop at N, ignore t>N. Train
   on cached frozen inputs with full controller BPTT and select by tiny-fit loss.
   The controller never receives numeric N/t/countdown, reference states, or probe
   predictions. The BCE has unit weight in this standalone diagnostic optimizer;
   this is not an unchanged continuation of the original joint optimizer.

Report first stop at probability >=0.5, early/late/missing stop, stopped-answer
accuracy, and correct-answer-at-exact-stop. The threshold is fixed, not calibrated
on held-out graphs. Distinguish original trained counts, interpolation counts
7/9/11, and extrapolation counts 13–16, and distinguish tiny-fit versus new graphs.
A missing stop at the budget is never an exact-stop success. Stop curves include
forced continuation beyond the first stop; their later crossings are not actual
inference stopping decisions. Cached replay supports first-crossing simulation
because this controller cannot change the executor and never feeds back its decision.
No inference-speedup claim is made from cached fits.

Each diagnostic probe runs 400 updates, LR 0.003, and checks dev CE every 25.
The linear/MLP classifiers see all count classes, including 13–16; they measure
count accessibility across graphs, **not unseen-count generalization**. Only the
controller-fit count split measures that. Probe success does not establish causal
use; failure does not prove the information is absent. Failure of a 1,000-update
tiny fit does not prove architectural impossibility. These controls narrow the
next hypothesis without selecting a production controller or retraining R.

### Outputs and reuse

The output directory contains:

- `summary.json`: provenance, settings, probe and stopping aggregates, limitations.
- `panel.csv`, `tasks.jsonl`: exact graph partitions and prompts.
- `probes.csv`, `probe_confusion.csv`: selected probe results and confusion counts.
- `controller_fit.csv`: fit loss, exact stopping, learning rate and gradient norms.
- `decisions.csv`, `stopping_by_depth.csv`, `trajectories.csv`: original/continued/fresh
  controller outcomes and stop probabilities by requested count and loop.
- `features.pt`: frozen vectors and original controller weights for reuse.
- `continued_tiny.pt`, `fresh_tiny.pt`: diagnostic-only controller copies, **not**
  complete recurrent checkpoints accepted by the ordinary evaluator.
- `wandb_eval_run.json`: upload identity and artifact checksums when tracking is enabled.

Push the JSON/JSONL/CSV files and sibling log. `.pt` caches and weights are already
Git-ignored; no large binary upload is needed for the initial review. No additional
requirements are needed. The desktop diagnostic completed in 88.87 seconds before upload; see the linked
report for pretrained outcomes. Local implementation tests use a tiny random Qwen model.

The extraction can be reused without loading Qwen or its checkpoint:

```bash
python -m scripts.eval.controller_diagnostic \
  --features-cache eval/pointer_diagnostics/<completed-run>/features.pt \
  --device cuda
```

Cache reuse retains its original graph partitions, count range and extraction
provenance; it does not resample using extraction CLI flags. Fit settings are
recorded separately. This is an optional follow-up, not needed for the first run.

## Historical diagnostics

The sections below retain the earlier executor-failure investigation. The current
controller command above supersedes their suggested next action.

The current implementation is the [executor upgrade](pipeline_upgrade.md), with
[matched diagnostics](evaluation.md#executor-upgrade-diagnostic-bundle). It adds
R/C readouts, count controls, cycle strata and precision comparisons to the shared
evaluator, and implements tested prefix reuse/selected projection. Earlier
proposal and profiling sections below describe the historical pipeline.

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

Implemented and run on the 32-example validation/depth-test prefixes. See the [controlled restart report](experiments/stage1_controlled_restarts.md). The exact direct rule-refresh intervention was harmful and should not be expanded to the full development set. The commands below remain the reproduction interface; no training or checkpoint mutation occurs.

```bash
git pull --ff-only
bash probe_controls.sh
```

The wrapper activates `.venv`, checks checkpoint files, sets the CUDA determinism environment, and runs the first 32 examples of both seed-17 validation and depth_test. Validation covers depths 1–8; depth_test covers 9–16. It uses float32/eager batch-1 inference and does not download missing models implicitly. Combined terminal output is saved in `eval/pointer_probes/controls-small-<UTC timestamp>-<PID>/run.log`; results go into `validation/` and `depth_test/` beneath that directory. Any failure stops the wrapper with a nonzero exit status. No output directory is reused.

The completed small result makes expansion of the harmful rule-refresh intervention unnecessary. The full-mode command remains available for reproduction, not the recommended next run:

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

## Frozen rule-edit probe

The next diagnostic uses the retained step-2500 checkpoint and the first 32 seed-17 `depth_test` examples. It runs original full questions to locate cases whose **first** wrong transition is later than loop six. For each eligible case, it probes loop six (normally correct) and the observed first-error loop. Each probe changes exactly one destination letter in the complete input table, keeping Start, Steps, rule order, and loop count fixed:

- **Relevant:** change the outgoing edge of the exact reference state entering that loop to an unused symbol. The reference target changes only at that loop; all earlier targets are checked unchanged.
- **Irrelevant:** make the same kind of one-letter edit to a source absent from the whole reference path. The target through that loop remains unchanged.

The model processes each complete edited prompt internally. No loop receives an externally decoded answer or a reconstructed question. `pairs.csv` records original and edited predictions, targets, target rank/margin, whether earlier predictions changed, and whether the edited prefix remains correct. Interpret transition following only on the retained-prefix cohort; `summary.json` reports its denominator and a matched cohort where both edit types retain correct prefixes. An edited prompt can change hidden states from loop one, so a relevant-rule effect narrows the mechanism but cannot alone prove a particular internal lookup circuit. If most late counterfactuals lose the prefix, this probe is inconclusive and should lead to a different controlled measurement, not another training run.

On the CUDA desktop, from the repository root:

```bash
git pull --ff-only
bash probe_rule_edits.sh
```

The wrapper checks local checkpoint files, activates `.venv`, sets the deterministic CUDA environment, and writes `eval/pointer_rule_edits/baseline2500-<UTC timestamp>-<PID>/run.log` plus `results/summary.json`, `pairs.csv`, `states.csv`, and `inputs.jsonl`. `states.csv` contains C-readout symbols/confidence and scalar R-state summaries for every observed loop. `inputs.jsonl` records exact edited prompts, token IDs, edge changes, and reference targets through the probed loop. Full hidden vectors, internal per-layer attention, and gradients are not saved. The script is inference-only, never alters the checkpoint, and does not use the seed-29 confirmation split. The pretrained desktop result and its interpretive limits are in the [rule-edit report](experiments/stage1_rule_edit_probe.md); rerunning the command is not the next recommended step.

Local checks: the diagnostic tests pass with a tiny saved model, including one full CLI run and byte-identical checkpoint files before/after. The complete repository suite passed 125 tests before the final added CLI test, which then passed separately. The cached pinned Qwen tokenizer validated all 88 edited prompts across the 22 eligible late-error cases in the first 32 development examples; their token counts match the originals. The checkpoint directory on this Mac lacks adapter weights and its saved tokenizer, so only the desktop can execute the pretrained diagnostic.

## Integrated frozen mechanism diagnostic

The [completed rule-edit result](experiments/stage1_rule_edit_probe.md) is not a sufficient explanation of late failure. The integrated suite runs one predeclared development design against the **same saved checkpoint**, without training or changing inference behavior. It checks several distinct ways the model could appear to perform short pointer chains while failing to reuse the transition process:

| Question | Controlled measurement | What it can and cannot establish |
| --- | --- | --- |
| Where does execution first break? | Frozen-C readout of `h_0` before recurrence, then original per-loop trajectory, first-error category, correct-prefix risk set, confidence and state-update scalars | Checks whether P+C already predicts Start/step one and locates the later boundary; a decoded answer is not a direct hidden-state label. |
| Does the trained adapter matter? | Temporarily disable recurrent LoRA for the same complete prompt and loop count, then restore it | Shows contribution to this task; neither outcome alone proves an algorithm. |
| Are outputs tied to rule position, symbol names, or requested horizon? | Per-example rule-order shuffle, full A–Z bijection, `Steps=d+1` with the same scored prefix, and an off-path edge edit | Measures exact paired prediction equivariance; prompt changes can also affect tokenization or distribution. Token counts are saved. |
| Does a changed rule propagate through a chain? | At a correct early loop and a late first error, test **two distinct**, seeded off-path destinations excluding the original predicted letter; pair each with an irrelevant-edge edit to the same destination. Early edits are scored through the whole changed path, including the next three transitions. | Directly tests the specified input rule and downstream compositional execution. Score only cases retaining a correct edited prefix; prompt perturbations begin at loop one, so this is not a hidden-state causal intervention. |
| Is the reference state linearly readable before/after failure? | Fit fixed-alpha ridge readouts on R's answer-position `h_t` and frozen C's normalized answer vector using validation instances at trained loops 1–6; evaluate on different validation instances and deeper tasks | Separates **linear decodability** from the installed frozen head only when a probe's held-out early accuracy reaches `max(80%, model early accuracy − 5 percentage points)`. It does not prove what R functionally uses; an inadequate early probe makes its late comparison inconclusive. |
| Does answer-token attention visibly select the active rule? | During original held-out/deep runs, save per-layer active source/destination attention ranks, top-1 head counts, and masses from Answer to the 26 table pairs | Describes attention allocation; attention weight alone is not causal attribution, and an indirect lookup may not attend to the named destination token. |

The deterministic cohort has 24 fit examples at each validation depth 1–6 (144 mappings), eight held-out validation examples at each depth 1–8 (64), and eight deep development examples at each depth 9–16 (64). The fit cohort trains **only** the offline diagnostic readout; Qwen and its adapters remain frozen. The two evaluation cohorts use disjoint rule tables and are chosen before examining outputs. The deep set is still seed-17 development data, not seed-29 confirmation. Relevant-edge failure probes are selected from original first errors after the trained-depth boundary; this conditioning is reported, not disguised as a random sample. Each replacement target excludes the original predicted letter, avoiding the previous accidental-success issue. A model that passes these controls only supports the tested behavioral claims; finite behavioral tests cannot prove a unique internal algorithm.

### Matched horizon and checkpoint comparison

The same selected 64 validation and 64 deep mappings also run on steps **2500, 3250, and 3750** from the fresh depth-6 run. The wrapper loads these checkpoints sequentially and requires identical model configuration and tokenizer. Each deep mapping keeps Rules and Start identical while varying only the displayed `Steps`: its native depth `d`, the trained ceiling `6`, and `17` (one beyond the maximum selected deep depth). Each prompt is executed both for six actual loops and through its full nominal depth `d`. The first-six predictions must agree between short and full unrolls of the *same prompt*; this is an implementation/prefix-invariance check, not a separate generalization score. Native and `Steps: 17` are both valid through all scored `d` loops, so their paired late differences isolate the displayed horizon on the same instance. `Steps: 6` is valid only through loop six; later outputs are explicitly labeled an **out-of-instruction stress test**, not normal accuracy. All cue comparisons retain exact input token counts to expose tokenization differences. A cached-tokenizer preflight of all 256 selected comparison prompts passed: `Steps: 17` changes token count for eight native depth-9 examples, while `Steps: 6` changes it for 56 depth-10–16 examples. The summary therefore reports paired effects for the equal-token-length subset as well as the complete cohort; length-mismatched results cannot isolate the depth cue from a changed answer-token position.

For each checkpoint, save native per-depth complete trajectories, loop accuracy, first-error/correct-prefix cases, and the same-mapping cue comparisons. Against step 2500, count cases whose correct prefix gets longer, shorter, or stays the same, and report paired mean step-accuracy and prefix differences. Fixed-long cue effects are separated into loops 1–6 and 7+. The reported 95% intervals are deterministic stratified **case-level** bootstraps (2,000 resamples within task depth), not intervals treating correlated loop rows as independent observations. With only eight deep mappings per depth these are discovery diagnostics; depth-specific estimates remain imprecise. The matched comparisons can identify whether the displayed horizon and additional training correlate with the failure boundary. They cannot alone assign a unique cause inside the network.

Run once on the CUDA desktop from the repository root. The suite directly uses NumPy for portable `.npz` state artifacts; it is now pinned in `requirements.txt`:

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements.txt
bash diagnose_mechanism.sh
```

The wrapper activates `.venv`, checks all three checkpoints' weights/tokenizers, sets CUDA determinism, and saves combined stdout/stderr to `eval/pointer_mechanism/matched-2500-3250-3750-<UTC timestamp>-<PID>/run.log`. The `results/` directory contains `summary.json`, original/first-error/paired/directed/probe/attention CSVs, exact `inputs.jsonl`, and `answer_states.npz`. The new `comparison.csv` records each matched checkpoint/cue/loop, `comparison_cases.csv` records correct prefixes and whether first errors occur inside the displayed horizon, `comparison_prefix.csv` records the six/full unroll agreement checks, and `comparison_inputs.jsonl` preserves the exact prompts and token IDs. `cases.csv` includes the pre-recurrence C(`h_0`) symbol and whether it matches Start or the first target. The NPZ stores float32 answer-position R states `h_0..h_T`, normalized frozen-C answer vectors, exact reference target indices, cohort labels, and lengths for the **step-2500** probe cohort; it is diagnostic data, not a model checkpoint. Attention CSVs likewise apply to step 2500 only. The summary records source/data/checkpoint hashes, cohort IDs, paired case denominators, bootstrap intervals, and artifact hashes. `status: complete` is the completion marker; a failed or interrupted run is not evidence. The CLI remains configurable as `python -m scripts.eval.mechanism_diagnostic --help`; its `--comparison-model` option accepts additional compatible checkpoints.

Interpret the suite jointly. Strong order/relabel and distractor invariance plus counterfactual branch continuation beyond the trained horizon would support reusable table execution, though not prove it. A steep first-error boundary with reliable early edits, weak late two-target following, and poor continuation would support a finite-horizon or state-propagation limitation. If a held-out R-state probe remains accurate where the installed C readout fails, inspect readout alignment; if both fail, the latent state or its encoding may have changed, but a linear probe cannot rule out nonlinear information. An attention-rank change is a lead for later causal tests, not a diagnosis. Do not choose new training or architecture from one aggregate score or one uncalibrated probe.

Local validation uses tiny saved Qwen models and focused metric/variant tests; no pretrained weights are present on this Mac. The complete repository suite passed **133 tests in 136.89 seconds**, including a three-checkpoint tiny-model integration and matched-metric checks. A preflight with the cached pinned Qwen tokenizer validated 1,296 original/transformed prompts and the 256 new matched-horizon prompts. Eight `Steps=d+1` depth-nine prompts change token count; those differences are recorded in `paired.csv` and should be excluded from a strict same-token-count interpretation. The new horizon comparison records its token counts separately. The completed pretrained run and its interpretation are in the [integrated mechanism report](experiments/stage1_integrated_mechanism.md); this wrapper is now a reproduction command, not a request for another run.

## Training observability through W&B

The trainer now exposes its persisted train/validation events to an optional W&B logger; see [setup, metrics and modes](training_pointer.md#wb-experiment-logging). Numeric per-loop and per-depth metrics are logged separately, so a batch-depth change should not be mistaken for progress or regression. CUDA peaks are reset per optimizer update and reported alongside batch depth, while a separate lifetime maximum preserves startup/validation peaks. This corrects the misleading interpretation of the old post-backward ~2 GiB allocation display. It does not establish GPU saturation or a safe larger batch. Benchmark throughput and deepest-batch memory before changing the recipe.
