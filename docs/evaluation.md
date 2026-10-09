# Evaluation and reproducibility

The current [analysis contract](analysis.md) specifies a full 1,350-graph selected
architecture comparison, execution/structural/stopping diagnostics and successful
model progression. Its [inventory](experiments/pointer_evidence_inventory.md)
separates available records from missing population measurements. Collection and
offline final-reference checks are complete. Run `bash analyze_pointer.sh` for
the implemented resumable population suite; its full run and figure replacement
are in progress. Read `python -m scripts.eval.paper_status` and the
[population report](experiments/pointer_population_analysis.md) for actual completion. The historical 27-graph comparison below remains valid exploratory
evidence and is not the main population report.

The [architecture-history protocol](experiments/pointer_architecture_history.md)
uses the same paired evaluator for the earlier full-sequence and fixed-prompt
models. `forced_only` arms have no learned controller: their stopping CSV cells
are empty and JSON metrics null; plots show N/A rather than zero. Batch-one
forced checks still verify their executor predictions. Previously audited arms
can be reused only with identical weights, inputs, policy and inference code.

The [complete figure audit](experiments/pointer_figure_audit.md) rescored and
regenerated all thirteen then-current benchmark/comparison figures. At requested
count N, **joint success** requires the correct final C letter and an actual
first stop at N; an earlier intermediate error does not automatically fail it.
**Complete trajectory** requires every C letter at loops 1 through N to be
correct. Its denominator is the same graph panel at every N, not the number of
loops. The first-error histogram concerns intermediate C-letter mismatches;
it is not a histogram of final-answer/timing failures.

The [paired checkpoint diagnostic](experiments/pointer_checkpoint_comparison.md)
compares retained architectures on identical opened benchmark inputs. Run
`python -m scripts.eval.checkpoint_comparison --dry-run` to inspect the fixed
protocol. It composes the existing actual-stop evaluator, replay and independent
audit; GRU observations remain actual R vectors, and legacy count-dependent R is
executed per count. No fitting or threshold selection occurs.

`python -m scripts.eval.audit_checkpoint_comparison --graphs <benchmark-graphs.jsonl>`
independently checks every matched decision against the raw rule tables.
`python -m scripts.eval.plot_pointer_failures --graphs <benchmark-graphs.jsonl>
--comparison eval/pointer_benchmark/checkpoint-comparison-20261008 --output <fresh-directory>`
adds trajectory, symbol-confusion, cycle-period, architecture-quality and stopping
matrices. Supply `--numeric-results <numeric-dir> <numeric-dir>` to compare existing
controller-only countdown diagnostics; the figure explicitly separates that scope
from full pointer inference. Plotting requires `requirements-plots.txt`.

The [final full benchmark](experiments/pointer_final_benchmark.md) directly
evaluates the frozen repaired checkpoint on the entire existing test and a new
independent graph panel. `bash benchmark_pointer.sh --full` reuses the standard
benchmark and native evaluator, with fresh graph seeds and no training changes.

The [shared number-reader repair](number_reader_repair.md) reuses the frozen
benchmark and normal checkpoint loader. Its fresh panel excludes opened graph
tables as well as old dataset splits. It measures exact stop and joint success
at every count 1–256; the separate numeric stress diagnostic does not execute R.
Checkpoint/config hashes are fixed before confirmation, and native calls must
match reuse. No numeric stress result establishes pointer execution at that depth.

`python -m scripts.eval.audit_pointer_benchmark --results <result-directory>
--graphs <graphs.jsonl>` independently traverses raw graph records and checks
every compressed decision, overall/per-count aggregates, first-error annotations,
data bytes and exclusions. It does not call the inference scorer or fit any model.
The number-reader launcher runs this after inference and numeric diagnostics.

The same auditor accepts `--native-results <native-output> --native-data <jsonl>`
to independently check every native intermediate target, first threshold crossing,
stop/cap semantics, final verdict and aggregate. It does not run inference or
reuse its scorer. The full launcher audits both native panels as well as all
compressed graph/count decisions.

`plot_pointer_benchmark` uses optional `requirements-plots.txt` and requires a
completed result with passed native fidelity and a current independent audit.
It now also requires `--graphs <original-graphs.jsonl>` and independently
recomputes all compressed decision flags, stratum/depth rates, pointwise Wilson
bands and per-loop denominators before rendering. Failure plotting additionally
checks its exported taxonomy, confusion, recovery and cycle-matrix cells against
raw traversal. Both plotters save `plot_input_audit.json` with their provenance.
It saves five figures as PNG/SVG/PDF, plus input/script/render hashes: full-scale
and zoomed quality, graph-type strata, first-error survival/histogram, numeric
initialization/stop residuals, and intermediate R/C readouts. Use `--output` for
a fresh alternative figure directory. Rates are unsmoothed; graph-level Wilson
bands are pointwise, not simultaneous depth guarantees. Exact controller timing
has one outcome per integer because its input ignores graph content.

For a standalone audit of all source tables, use:

```bash
python -m scripts.eval.audit_pointer_figures \
  --results eval/pointer_benchmark/final-full-20261008/independent \
  --graphs data/pointer/benchmark-seeds307-311-313/graphs.jsonl \
  --comparison eval/pointer_benchmark/checkpoint-comparison-20261008 \
  --numeric-results eval/pointer_benchmark/shared-number-20261008/numeric \
                    eval/pointer_benchmark/shared-number-precision-20261008/numeric \
  --output /tmp/pointer-figure-audit.json
```

Use a fresh output path. This rescoring never loads or executes model weights.

Status: nominal per-loop evaluation is implemented through the [full-loop checkpoint CLI](evaluation.md) and training monitoring. Repair/damage and censored survival are implemented for the separate [absorbing-terminal diagnostic](evaluation.md), whose pretrained execution is deferred. The conventions derive from project plan sections 6–8 and 23–27. Allowed-token raw-logit margins are implemented and tested in [outputs](../scripts/recurrent_qwen/outputs.py). [Stage 0](experiments/stage0_validation.md) records architecture measurements only.

The [ordinary-model final-answer baseline](evaluation.md) is implemented separately with unconstrained greedy generation, strict decoded-symbol accuracy, per-depth summaries, and CSV/JSON artifacts. Its toy-model tests verify evaluation mechanics; the [first full pretrained run](experiments/naive_pointer_baseline.md) achieved 60/1,000 correct and has been audited. The restricted-logit and recurrent-transition conventions below apply to the later recurrent evaluation, not automatically to this generation baseline.

Original baseline and per-attempt learning figures for the article are rendered
without model work using `python -m scripts.eval.paper_writeup`. The
[writeup coverage map](analysis.md#writeup-coverage--user-outline-2026-10-09)
records source populations and planned article placement. The completed paper
renderer joins those histories to audited population curves in `06_writeup/`;
reader/cell fits have saved summaries rather than invented epoch histories.

## Targets and loop indexing

`h_0` is the prelude output; `h_t` follows `t` recurrent passes. Task depth and recurrent depth are separate quantities.

Record both step-specific intermediate targets and the fixed final answer. Intermediate accuracy measures execution at the corresponding step. Final-answer correctness supports repair, damage, and solution-survival analysis. Label the target basis explicitly rather than using an ambiguous `is_correct` field alone.

Stage 1 data uses explicit requested steps and a non-repeating nominal path, with targets only at steps 1..d. `scripts.dataset.pointer.check_predictions` checks decoded symbols against that nominal trajectory; training monitoring records recurrent per-loop metrics. Original continuing-task post-completion behavior and general cyclic-task semantics remain open in [decisions](decisions.md). The separate terminal diagnostic resolves completion only for its own transformed inputs. Report nominal execution separately from post-completion trajectories; a valid additional pointer lookup must not automatically count as damage.

## Answer margins

Restrict symbolic evaluation to the validated allowed answer tokens. With scores from the frozen coda readout:

```text
margin_t = score(correct) - max(score(other allowed answer))
wrong:           margin_t <= 0
fragile correct: 0 < margin_t < gamma
robust correct:  margin_t >= gamma
```

Use a documented score convention consistently; the implemented margin helper uses raw logits. Record `gamma`, answer vocabulary, and target basis. For margin-based correctness, ties are not correct; if also reporting argmax accuracy, document its tie convention separately.

## Transition metrics

Using a fixed final target, let `A_t` be the fraction correct at loop `t`:

```text
R_t = count(wrong -> right) / count(wrong at t)
H_t = count(right -> wrong) / count(right at t)
G_t = (1 - A_t) * R_t - A_t * H_t
```

Report all four transition counts and the conditional denominators. An empty denominator makes its conditional rate undefined, not a measured zero. Net gain can still be computed directly as `(wrong_to_right - right_to_wrong) / N`.

For the same fully observed cohort and correctness definition, `G_t = A_(t+1) - A_t`; use this as a metric consistency check. Do not silently change the cohort across adjacent steps.

## First solution and survival

Define `T*` as the first loop correct against the final answer. Report examples that never solve separately. Continuous survival at offset `k` means every prediction from `T*` through `T* + k` remains correct; recovery after damage does not restore continuous survival.

Finite trajectories may end before an offset is observed. Document the censoring/eligibility convention and report counts at each offset; do not count unobserved tails as successful survival. Pointwise correctness after first solution may be reported separately.

## Required views

- Task-depth × recurrent-depth final-accuracy heatmap, including held-out task and loop depths.
- Intermediate-step accuracy and task depth versus first-correct loop.
- Repair, damage, and net-gain trajectories with supporting counts.
- Continuous solution survival and answer-margin trajectories.
- Per-family and holdout results when applicable; ordinary-Qwen degradation for preservation experiments.
- Diagnostics for no-op behavior, oscillation, collapse, and excessive confidence sharpening.

Keep all adjacent-loop observations even when figures display only selected budgets. State evaluation sample sizes, seeds, checkpoint-selection criteria, and uncertainty estimates used in comparisons. High final accuracy alone does not establish iterative execution.

## Trajectory records

Preserve the project plan's minimum fields:

```text
example_id, seed, family, task_depth, nominal_final_state,
loop_index, current_prediction, correct_target_for_this_loop,
final_answer, margin, is_correct, next_prediction, transition_type
```

Also distinguish intermediate versus final correctness/margins and whether nominal execution has finished. Specify field meanings in the implemented schema, including whether `final_answer` is a label or prediction; avoid two names with silently different meanings. Mark absent next predictions at the last observed loop explicitly rather than inventing a transition.

## Run and report metadata

Save code revision/worktree identification, base-model revision, checkpoint hash, tokenizer/symbol configuration, task-generator and split configuration, recurrent depths, LoRA/bridge settings, `gamma`, optimizer/training configuration, seeds, package versions, device, and dtype. Record commands and artifact paths so a report can be reconstructed from its configuration and trajectories.

Write human-readable reports in `docs/experiments/` as Markdown. Keep machine-readable logs, checkpoints, and figure outputs in an explicitly configured artifact location, linked from the report. Do not commit large artifacts by default. Record deterministic settings and any reproducibility limitations observed on the actual device.

## Implemented Stage 1 monitoring

[Training usage](training_pointer.md#progress-and-saved-signals) documents the current per-loop loss/accuracy, whole-trajectory accuracy, fixed train/validation subsets, depth-by-loop final readouts, and CSV margins. Only nominal steps contribute to supervised metrics. Post-completion predictions are recorded without retention labels or repair/damage claims. The best checkpoint minimizes validation loss on trained depths; deeper monitored depths are excluded from selection.

`naive_test --model <recurrent step directory>` evaluates the final allowed-symbol readout at each task's requested loop count. It uses raw dataset prompts and reports example-loops/s; ordinary model paths retain three-shot unconstrained generation and tokens/s. These are distinct evaluation conditions, even though both write final-answer CSV/JSON under `eval/pointer_task/`.

`python -m scripts.eval.loop_test --model <step directory>` evaluates every example through a common loop budget, exporting nominal losses/accuracies, full trajectories, first-error and correct-prefix diagnostics, and the depth-by-loop matrix under `eval/pointer_loops/`. See [commands and metric definitions](evaluation.md). Its original-task final-target sweep remains observational. The separate `python -m scripts.eval.overscaling_test` CLI transforms inputs into absorbing-terminal tasks and computes explicitly scoped dynamics; it must not relabel the original runs as terminal experiments.

The [integrated failure diagnostic](diagnostics_and_performance.md#integrated-frozen-mechanism-diagnostic) uses a smaller fixed development cohort for same-mapping prompt interventions and matched checkpoints. It records complete nominal trajectories separately from outputs produced after a prompt's displayed `Steps`; the latter are stress observations, not ordinary task accuracy. Checkpoint and cue comparisons use per-mapping pairs and case-level uncertainty rather than treating multiple loops from one mapping as independent trials.


## Implemented terminal dynamics and future knowledge retention

The terminal evaluator exports every adjacent transition, per-depth and aggregate rates, raw final margins, continuous survival, and a hold-at-d decoded-answer baseline. Primary dynamics include only transitions starting at t >= d; all-loop counts remain observational. Survival is anchored to the first final-correct loop at or after d, and reports observed/censored denominators. Earlier final matches are separately counted as potential shortcuts. Post-completion CE is not introduced. Figures and hidden-state diagnostics are not implemented by these exports.

A later [knowledge-retention regression check](project_plan.md#future-knowledge-retention-regression-check) will compare original versus adapted one-pass ordinary-model performance on a fixed small benchmark, potentially a seeded MMLU subset. Benchmark choice, scoring protocol, and implementation remain deferred. It will not reuse the pointer A–Z-only readout as a general knowledge evaluator or silently introduce anchor training.


## Outward depth evaluation

The [depth-6 experiment](training_pointer.md) evaluates both the depth-4 reference and new depth-6 checkpoint on identical depth-1–8 validation and depth-9–16 development examples. Reports include `task_depth`, `train_max_depth`, and `steps_beyond_training`. In-range improvements at depths 5–6 are not extrapolation; compare nearby and farther OOD ranges separately and inspect equal positive offsets. Equal offsets use different task depths, and the curriculum comparison is not controlled for total training compute. Seed 29 is reserved for confirmation after choices are frozen; it has not been generated or evaluated. Neither broad depth generalization nor a gate pass follows from merely shifting the failure boundary.

## Adaptive-compute evaluation

The [canonical adaptive-compute guide](adaptive_compute.md) defines the implemented offline oracle/heuristic replay, opt-in stopped inference, latency scopes, and remaining terminal and confirmation gates. New full-loop exports include top-two predicted-answer margin and allowed-answer entropy; historical true-target margins remain diagnostic labels and cannot drive a stopping policy. Original continuing-task post-depth changes remain observational. Loop savings from full-trace replay are counterfactual; actual skipped work and synchronized batch-1 latency are reported only for stopped inference.

---

## Stage 2 research gate (deferred)

Status: evaluation scripts are prepared; pretrained execution is **deferred** while the user pursues further Stage 1 training and depth generalization. No repair/damage result or Gate 2 pass is established. Source: project plan sections 6–8 and 24; the user-directed priority update is recorded in the [plan](project_plan.md) and [status](status.md).

### Purpose

Establish whether an untreated learned recurrence can both repair wrong answers and damage solved answers when run too long. This is an evaluation stage with a fixed Stage 1 checkpoint. Useful unseen-mapping execution and depth extension are reported in [the fresh 30k experiment](experiments/stage1_fresh30k.md); terminal dynamics remain unmeasured.

### Completion semantics

Original Stage 1 mappings continue after the requested depth, so an extra valid lookup must not automatically count as damage. The prepared experiment uses a **separate absorbing-terminal variant**, replacing the final state's outgoing edge with a self-loop. All nominal targets stay the same, but the prompt distribution changes. Inspect nominal execution and early-final shortcuts before interpreting post-completion dynamics.

Only transitions starting at t >= d count as post-nominal repair/damage. All-loop final-target counts are separately observational. Survival starts at the first final-correct readout at/after d and requires continuous correctness, with explicit censoring. General cyclic-task semantics remain unresolved. See [decision record](decisions.md) and [metric definitions](evaluation.md#metric-conventions).

### Deferred execution

1. Complete the bounded Stage 1 continuation and review validation depth extension before launching this experiment.
2. Freeze a validation-selected checkpoint and record selection criteria, configuration, and evaluation seeds.
3. Preview transformed inputs without model loading or writes; then run a small 16-loop checkpoint test when ready.
4. Sweep every example through a common budget such as 32 loops. Every intermediate readout is retained; 64-loop resource use is unmeasured and should be checked before scaling.
5. Inspect nominal execution separately from post-nominal recovery, damage, margins, continuous survival, and a hold-at-d decoded-answer baseline.
6. Diagnose absent recovery/damage or shortcut behavior before adding any retention objective.

Commands and artifact definitions live in [overscaling usage](evaluation.md), avoiding duplicate run instructions here. `scripts/eval/overscaling_test.py` composes the existing checkpoint loader and full-loop evaluator; `scripts/dataset/terminal.py` owns the transform; `scripts/eval/overscaling_metrics.py` owns conditional dynamics. Training objectives, recurrent weights, and original datasets are unchanged.

### Deliverables and gate

Implemented exports provide exact transformed tasks, trajectory and adjacent-transition records, a depth-by-loop matrix, rates with counts, margins, and survival with censoring. Checkpoint/data/source hashes and transform version make runs traceable. CSVs support future plots, but plotting, hidden-state collapse/no-op diagnostics, and confidence/stability stopping has separate, unconfirmed interfaces in the [adaptive guide](adaptive_compute.md). No pretrained terminal trajectories or figures exist yet.

The gate requires both useful wrong→right repair and right→wrong damage under overscaling. A changed answer in an original continuing-pointer task is insufficient. If either phenomenon is absent, record that result and diagnose the mechanism or task distribution. Implementation tests cannot pass this empirical gate.

---

## Ordinary-model pointer baseline

Status: implemented and tested with local random toy models. The user completed a full three-shot pretrained run with **6.00% accuracy (60/1,000)**; the saved results were audited without rerunning inference. See the [baseline report](experiments/naive_pointer_baseline.md) for evidence and limitations. This is a final-answer baseline, separate from recurrent execution and training.

A subsequent [WSL CUDA run](experiments/wsl_cuda_baseline.md) completed all 1,000 examples in 84.27 seconds with the same 6.00% accuracy and the same predictions-CSV hash as the historical report. Its setup checks and separate training-resume test limitation are recorded there.

### Usage

From the repository root with the configured environment and generated dataset:

```bash
# Inspect three actual model inputs, responses, and right/wrong decisions.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --test

# Inspect two instead.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --test 2

# All 1,000 same-depth test examples, default CPU / float32 / batch 1.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct

# Short first run on Apple Silicon, if MPS is available.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --device mps --limit 8

# Held-out depths 9–16.
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --data data/pointer/seed-17/depth_test.jsonl

# A directory written with Hugging Face save_pretrained, including its tokenizer.
.venv/bin/python -m scripts.eval.naive_test --model path/to/saved-model

# A full PyTorch state dict, using the specified base for architecture/tokenizer.
.venv/bin/python -m scripts.eval.naive_test --model path/to/weights.pt --base-model Qwen/Qwen2.5-0.5B-Instruct

.venv/bin/python -m scripts.eval.naive_test --help
```

These commands are for user execution; the completed default full-test run is documented in the [baseline report](experiments/naive_pointer_baseline.md). Other configurations shown here are usage examples, not measured results. `--limit N` selects the first N rows; the generated file cycles through depths, so a tiny prefix need not cover every depth. The summary reports the depths actually evaluated.

`--test` runs only the first three examples; `--test 2` runs two (or all available if the file is smaller). It prints the exact input passed to the tokenizer, including chat formatting, the raw decoded response as a Python `repr` so whitespace is visible, the expected symbol, parsed symbol or format failure, RIGHT/WRONG, stop reason, and generated token IDs. It uses the same generation/scoring/output path as the full evaluator and writes the normal artifacts with `test_mode: true`; default output directory names include `test-`. `--test` and `--limit` are mutually exclusive. Test mode is an actual model run, not a dry run.

`--model` supports a Hugging Face repository ID, a local full-model directory, or a standalone `.safetensors`, `.pt`, `.pth`, or `.bin` file. PyTorch files must contain a tensor state dict or `{"state_dict": tensor_dict}`; full pickled model objects are not loaded. A standalone file requires `--base-model` unless its directory contains `config.json` and a usable tokenizer. `--tokenizer` can override the tokenizer source. Missing/unexpected model weights fail; tied parameter aliases may be omitted in standalone files when their shared tensor is present. A single shard is not a complete checkpoint. Adapter-only checkpoints and the custom recurrent wrapper are not supported by this ordinary-model evaluator.

Loading uses cached/local files unless `--download` is explicitly supplied. The default revision for the project's Qwen model is `7ae557604adf67be50417f59c2c2f167def9a775`; other Hub IDs default to `main`. `--revision` selects a different model/base revision. The resolved model commit is saved. An overridden tokenizer uses its own source's default revision; use a local tokenizer snapshot to pin a separate tokenizer exactly.

`--device cpu|mps|cuda` and `--dtype float32|float16|bfloat16` are explicit; there is no fallback. Defaults are CPU and float32. Set `--batch-size` to change the default batch of one. Left padding is used, and tokenization uses the evaluated model's tokenizer rather than the token IDs stored in the Qwen dataset. Unsupported device/dtype operations fail rather than silently switching settings.

### Prompt and score

The shared [prompt template](../prompts/pointer_task.txt) explains pointer pairs, asks for exactly the specified number of transitions, and requests only a final uppercase letter. It includes three hand-written demonstrations at depths 1, 2, and 3, each with an independent shuffled six-symbol rule table and a single-letter final answer. Their exact paths are A→F, A→D→E, and E→B→A→C; only final answers appear in the demonstrations. They are not selected from the test set. The actual task follows a clear boundary and uses its own rules.

Placeholders are limited to `{rules}`, `{start}`, and `{steps}`; the evaluated example's final/intermediate targets are never supplied to the template. `--prompt FILE` selects another template with that same interface. This three-shot prompt replaces the initial zero-shot instructions after the user's 0/3 report; compare runs using the saved prompt text/hash, since they represent different prompting conditions. The full three-shot run reached 6.00%; no paired full zero-shot run was collected, so an improvement from demonstrations has not been established.

Default `--prompt-format chat` wraps the task text in one user message with the model tokenizer's chat template and generation prefix. For a model without a chat template, explicitly choose `--prompt-format raw`. The task text stays shared; the chat wrapper is model-specific. This adds natural-language task instructions to the dataset's bare rule table and is not a silent change to the future recurrent training prompt.

Generation is greedy, one beam, with no answer-vocabulary restriction, `use_cache=False`, strict deterministic algorithms, seed 17, four CPU threads, and an eight-token output budget by default. `--max-new-tokens` changes the budget. Token outputs are ordinary autoregressive generation, not recurrent loops. Inputs are never truncated; a prompt plus output budget exceeding the model's declared context limit fails explicitly.

Decode only newly generated tokens, skip special tokens, and strip surrounding whitespace. A prediction must equal the expected single uppercase symbol to count as correct. For target D, `" D\n"` passes; `"d"`, `"D."`, `"Answer: D"`, empty text, and prose fail. Invalid answers stay in the accuracy denominator. The CSV preserves the response, and the summary separately counts invalid answers and token-budget stops. A valid answer at the token limit can still be correct; the stop reason remains visible.

This strict score measures both task correctness and requested-format compliance. It does not implement lenient answer extraction or compare only A–Z logits. Comparisons with a later restricted-logit recurrent readout must label that distinction.

### Outputs and timing

Default output: `eval/pointer_task/<UTC timestamp>-<model>/`, rooted at the repository even when invoked from another working directory. `--output DIRECTORY` selects another new directory. Existing output directories are refused. Generated evaluation artifacts are excluded from Git.

- `predictions.csv`: example ID, split, seed, task depth, target, raw response, parsed prediction, valid/correct flags, prompt/generated token counts, stop reason, generated token IDs, rendered task prompt, and complete model input.
- `summary.json`: run configuration, prompt/data/source hashes, model and tokenizer identity, local checkpoint hashes when applicable, package versions, overall and per-depth accuracy, invalid-answer/token-limit counts, timings, and throughput.

CSV rows flush after each batch. The summary initially has `status: running` and changes to `complete` only after all selected examples finish. If interrupted or failed, partial CSV rows and a running summary are not a completed evaluation; start a new output directory to rerun. There is no resume mode.

Rich displays a progress bar, completed questions, elapsed/remaining time, running accuracy, generated tokens/s, and questions/s. Normal runs save individual prompts and responses only to CSV; `--test` also prints them for inspection.

Generated tokens/s divides generated token count by synchronized `model.generate` time, including prompt prefill. Generated token counts include the first EOS but exclude subsequent batch padding. Questions/s uses evaluation-loop wall time including tokenization, decoding, CSV writes, and progress updates, but excludes loading and initial metadata hashing. These are measured batch-run rates, not standalone decode latency. Memory use remains unmeasured; the first CPU pretrained throughput measurements are in the [baseline report](experiments/naive_pointer_baseline.md).

### Stepwise testing later

The dataset already stores the state after every pointer transition. A recurrent evaluator can read the frozen coda after loop t and compare its decoded symbol against `intermediate_states[t-1]`. That measures whether one recurrent loop corresponds to one transition and belongs to Stage 1's recurrent evaluator.

Ordinary Qwen could instead be prompted to print a sequence of intermediate symbols, but that would measure a generated textual trace, not internal recurrent depth. Neither stepwise baseline is implemented in this change. Final-answer accuracy alone does not pass the Stage 1 mechanism gate or establish repair/damage dynamics.

### Implementation and checks

- [naive_test.py](../scripts/eval/naive_test.py): CLI, explicit configuration, progress, CSV and JSON output.
- [pointer_task.py](../scripts/eval/pointer_task.py): dataset verification, input-only prompting, generation batches, scoring, and metrics.
- [loading.py](../scripts/eval/loading.py): Hub/directory/state-file model loading with explicit tokenizer, device, and dtype.
- [Focused tests](../tests/test_naive_pointer_eval.py): strict scoring, prompt field restrictions, target validation, EOS/padding token counts, per-depth denominators, batch continuation slicing, saved-directory/PyTorch/Safetensors weight fidelity, partial-state rejection, and CLI output/overwrite behavior using local random toy models.

Focused validation: `.venv/bin/python -m pytest tests/test_naive_pointer_eval.py -q` passed all 17 cases. CLI tests include normal mode and both `--test` sizes, asserting that the printed RIGHT/WRONG counts agree with the persisted summary. No pretrained checkpoint was loaded during these checks.

Final regression check: `.venv/bin/python -m pytest -q` passed all 80 tests in 19.33 seconds, with no skips.

No new dependencies are required. The [first full-run report](experiments/naive_pointer_baseline.md) supersedes the initial 0/3 inspection as the aggregate baseline; its source summary/CSV are retained under `eval/pointer_task/`.

### Saved recurrent training checkpoints

The same CLI also accepts the step directories produced by [Stage 1 training](training_pointer.md#artifacts-and-checkpoint-evaluation):

```bash
python -m scripts.eval.naive_test --model models/stage1_pointer/<run>/step-000160 --test
```

The checkpoint's `recurrent_config.json` selects recurrent loading. The evaluator uses its frozen base revision, saved tokenizer, raw dataset prompt, and A–Z argmax at loop equal to requested depth. It prints and records this scoring convention separately from ordinary-model three-shot generation. In a mixed-depth batch, each example is scored at its own depth even though the batch executes its maximum depth. Throughput is example-loops/s and questions/s. Omit generation-budget, prompt, base-model, revision, and tokenizer overrides for this format. Use the whole step directory, not its adapter tensor file; the recorded base must be cached or explicitly downloaded with `--download`.

---

## Full-loop checkpoint evaluation

Evaluate a saved recurrent checkpoint without training or loading optimizer state. The CLI reuses the trainer's per-loop evaluation and exact nominal targets. It records every frozen-coda readout; predictions are never fed back as input tokens. This measures latent recurrent execution, not a generated explanation. Fixed-depth behavior remains the default. The checkpoint selects full-sequence or fixed-prompt recurrence automatically. Historical adaptive policies are documented in the [adaptive-compute guide](adaptive_compute.md); the separate prompt-only completion policy is below.

### Prompt-only completion stopping

`--stop-policy completion --stop-threshold 0.5 --loops 20 --batch-size 1` uses the checkpoint's own hidden-state completion head. An explicit positive loop cap and threshold in (0,1) are required; the cap is not derived from task depth. The head receives no parsed depth or loop number and may stop too early. `--head` selects a different, historical adaptive policy and is not accepted here. A cap below requested depth is allowed and reported as truncated execution/fallback, rather than silently increasing it.

The evaluator saves actual `trajectories.csv`, per-example `decisions.csv`, and `summary.json`. Report stopped-answer accuracy, exact/early/late head stops, cap fallbacks, joint exact-stop-and-answer success, and by-depth rates separately. Reaching the cap at the correct depth does not count as a learned exact stop. Timing synchronizes CUDA/MPS and includes input-memory preparation, every executed R/C/head pass and threshold check; it excludes loading, tokenization and CSV export. Batch-1 stopped latency and batch-16 full-sweep throughput are different measurements and cannot establish a matched batching speedup.

For the new fixed-prompt training run, `bash eval_ckpts.sh` selects its pointer-CE-selected checkpoint and runs both full sweeps and actual stopping on validation/depth-test. The wrapper's threshold 0.5 is diagnostic; confirmation still requires a threshold frozen on development. See [training and output paths](training_pointer.md#fixed-prompt-memory-completed-depth-6-run).

### Run

From the repository root on the configured desktop:

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Inspect three examples through eight loops first.
python -m scripts.eval.loop_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000500 \
  --device cuda --loops 8 --test

# Full test set, both previously identified checkpoints; separate outputs.
for step in 000500 000625; do
  python -m scripts.eval.loop_test \
    --model "models/stage1_pointer/20260910T220130.926886Z/step-$step" \
    --device cuda --loops 8
done
```

Use the actual **step directory**, including `adapter_model.pt`, `recurrent_config.json`, and saved tokenizer files. The metadata-only Git checkout cannot reconstruct the trained model. Run on the desktop where the complete checkpoints remain, or copy them separately. The pinned frozen base loads from the local cache; `--download` explicitly permits a missing base download. `training_state.pt` is not required. No checkpoint is modified.

Default input is all 1,000 examples in `data/pointer/seed-17/test.jsonl`. `--data` selects another file, including `depth_test.jsonl`. `--loops` defaults to the deepest selected task and applies equally to every example. It must cover every selected nominal target: an insufficient budget is rejected, not truncated. `--limit N` selects the first N examples; `--test [2|3]` selects a small prefix and prints prompts and per-loop RIGHT/WRONG decisions. Both use the same scoring path and write normal result files. To compare a preview with the full run, specify the same loop budget explicitly.

Defaults are CPU, float32, batch 1, seed 17, four CPU threads, eager attention, and strict deterministic algorithms. Use `--device cuda` or `--device mps` explicitly; there is no fallback. A compact Rich progress bar shows ETA, questions/s, and executed example-loops/s. Timing includes evaluation, scoring, and trajectory CSV writing, but excludes checkpoint loading, input encoding, and final diagnostic exports. There is no text generation or tokens/s measurement.

### Artifacts

Each run writes a new directory under `eval/pointer_loops/<timestamp>-<training-run>-<step>/`. `--output` selects another new directory; existing directories are rejected. The repository's current ignore rules allow these small CSV/JSON artifacts to be committed.

| File | Contents |
| --- | --- |
| `trajectories.csv` | One row per example per loop: ID, split, seed, family, depth, initial state, prediction, intermediate and final targets/correctness, intermediate CE, intermediate/final raw-logit margins, target-free predicted-answer margin and entropy in new runs, and post-completion flag |
| `examples.csv` | One row per example: target and predicted sequences, complete-trajectory correctness, nominal final correctness, first erroneous loop, correct-prefix length, first loop matching the final answer, and repeated adjacent nominal predictions |
| `depth_by_loop.csv` | Task depth × loop matrix in long form: counts, final accuracy, and nominal intermediate accuracy/loss |
| `summary.json` | Overall and per-depth/per-loop metrics, diagnostic counts, model/checkpoint/data/source hashes, command, settings, package versions, Git provenance, timing, and completion status |

`summary.json` begins with `status: running` and is marked `complete` only after all exports finish. Interrupted runs are incomplete and cannot be resumed; choose a new output directory. Trajectory CSV writing occurs after the evaluation sweep. No model weights, optimizer state, hidden-state tensors, or full vocabulary logits are exported.

### Interpretation

- Loop t is scored against the exact state after t transitions for **t <= task depth**. Complete-trajectory accuracy requires every nominal step to be correct, even if the final answer is right after an earlier error.
- First-error indices are 1-based and blank for perfect trajectories. Correct-prefix length ends at the first error; recovery cannot lengthen it. First-final-correct indices search the full observed sweep and are blank if never correct; an early final match does not prove correct execution.
- Loss is 26-symbol CE, averaged across nominal loops within each example, then across examples. Per-loop loss/accuracy use only examples whose depth reaches that loop; their denominators differ. These are the same definitions as training validation.
- A margin is the target symbol's raw logit minus the largest other allowed-symbol logit. A–Z argmax uses the first symbol for ties; a correct argmax can therefore have zero margin.
- All examples are swept through the same loop count, so the final-target depth × loop matrix includes post-completion observations. Intermediate target/loss/margin/correctness are blank after completion and excluded from supervised metrics. This does not define a retention target or classify valid continuing pointer moves as damage.
- Repeated adjacent predictions can indicate a stalled decoded answer, but do not establish frozen hidden states. The count considers adjacent nominal loops only, where the reference path does not repeat.

For the original one-epoch experiment, update 500 remains the validation-loss-selected primary checkpoint. Update 625 is its already inspected secondary comparison; do not silently reselect a primary checkpoint using test performance. Separate depths 1–4 (trained), 5–8 (untrained depths already monitored during training), and the deeper test file. Both historical checkpoints have now been evaluated and audited in the [run report](experiments/stage1_cuda_5k.md). Gate 1 still requires explicit empirical criteria; the CLI does not declare a gate passed. The next priority is [further training](training_pointer.md#continue-the-current-desktop-run), not repeating these completed sweeps.

### Validation

Tests use prescribed logits and a random tiny Qwen checkpoint with a local synthetic tokenizer. They cover first-error/recovery semantics, nominal masking, margins, mixed depths/padding, checkpoint loading without optimizer state, agreement with `naive_test` final readouts, exports, inspection mode, and rejection of insufficient loops or existing output paths. The user subsequently completed pretrained full-loop execution on CUDA for both checkpoints; see the [audited report](experiments/stage1_cuda_5k.md). Implementation tests alone make no performance or gate claim.

Validation on the Mac: `.venv/bin/python -m pytest tests/test_loop_eval.py tests/test_pointer_training.py -q` passed all nine focused cases; `.venv/bin/python -m pytest -q` passed all 89 cases in 45.95 seconds. The CLI help command passed. No pretrained checkpoint evaluation was launched by the assistant; the user-run evidence is recorded separately.


For later overscaling experiments with a well-defined terminal state, use the separate [terminal evaluator](evaluation.md). The ordinary `loop_test` command and existing dataset retain their nominal-only semantics. Terminal execution is currently deferred in favor of further Stage 1 training.

For the current outward-depth experiment, update 1875 from the completed depth-4 continuation is the reference. The [paired depth evaluator](training_pointer.md#evaluate-both-models-through-depth-16) evaluates it and the new depth-6 checkpoint on identical files and exports absolute and relative depth metrics.

---

## Absorbing-terminal overscaling protocol (prepared; pretrained sweep deferred)

Status: implemented as a deferred experiment; no pretrained terminal sweep has been run. The user's current priority is further Stage 1 training and depth generalization. Preparing these scripts does not establish Gate 1 or authorize launching Stage 2 now.

### Task and interpretation

`scripts/dataset/terminal.py` deterministically transforms each validated source task by replacing the final state's outgoing edge with a self-loop. Rule order, initial state, requested depth, seed, source ID, and nominal targets are retained. The prompt and mapping hash are recomputed and validated. Source files and checkpoints are never changed.

For `A -> C -> E`, the terminal version contains `E -> E`. Reference execution is `C, E, E, E, ...`. Because the nominal path has no repeated states, replacing that outgoing edge cannot alter any target through depth d. The source dataset's seeds and file hash plus `absorbing-terminal-v1` fully determine the transformed tasks; there is no additional sampling RNG.

This is a **different input distribution** from training. The terminal self-loop can also provide an answer shortcut: early final matches must be inspected alongside complete nominal trajectories. A nominal-performance drop must be diagnosed before drawing stability conclusions. High terminal-task accuracy alone does not demonstrate stepwise execution or transfer. Cyclic tasks in general remain outside this experiment.

For t >= d, the fixed final target is also the valid continuing reference state. Wrong→right is post-nominal recovery; right→wrong is damage. For t < d, movement toward the final answer is ordinary execution, so those transition counts are explicitly observational. No retention training, detached rollouts, anchor loss, hidden-state probes, or model architecture changes are introduced.

### Preview now; inference later

From the repository root with the environment activated:

```bash
python -m scripts.eval.overscaling_test --dry-run
python -m scripts.eval.overscaling_test --dry-run 10 --loops 16
```

These show the first five or ten transformed tasks, changed terminal rules, and exact reference targets. No checkpoint is required, no model weights are loaded, and nothing is written. Unlike the main dataset generator's depth-spanning preview, this preview selects the input file's prefix and is not a distribution estimate.

When Stage 1 is ready for review, use a complete saved checkpoint directory on the desktop. The following are **unrun future commands**, not the current training instructions:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m scripts.eval.overscaling_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000500 \
  --device cuda --loops 16 --test

python -m scripts.eval.overscaling_test \
  --model models/stage1_pointer/20260910T220130.926886Z/step-000500 \
  --device cuda --loops 32
```

Update 500 is the historical validation-selected checkpoint, not necessarily the checkpoint for the future experiment. Select that future checkpoint using validation before test sweeps. Use 625 only as the already identified secondary historical comparison. Do not choose the best overscaling result after looking at test scores.

The default is all 1,000 source test examples and 32 loops. `--loops` must exceed every selected task depth. `--test [2|3]` performs real inference and prints per-loop predictions and correct targets, including terminal targets after d. `--limit N` runs a prefix; these subset flags and `--dry-run` are mutually exclusive. `--data` changes the source file. `--output` must be a new directory. CPU/float32, batch 1, seed 17, and strict determinism match `loop_test`; explicitly select CUDA/MPS. Downloads are opt-in with `--download`.

Each sweep observes every loop up to its budget; a 32-loop run supplies the 1/2/3/4/6/8/16/32 prefixes. A 64-loop run is supported but memory and timing at these budgets are unmeasured on pretrained checkpoints. Start small when these experiments are eventually launched.

### Artifacts

Outputs go under `eval/pointer_overscaling/<timestamp>-<run>-<step>/`:

| File | Purpose |
| --- | --- |
| `tasks.jsonl` | Exact transformed prompts, mappings, source IDs/seeds, and nominal targets |
| `trajectories.csv` | Every prediction and final margin; nominal CE/intermediate targets only through d |
| `examples.csv`, `depth_by_loop.csv` | Shared full-loop execution diagnostics and final-accuracy matrix |
| `transitions.csv` | Every adjacent prediction pair, target, margins, correctness, transition type, and post-nominal flag |
| `transition_rates.csv` | Per-depth and aggregate counts, conditional denominators, repair, damage, net gain, and adjacent accuracies |
| `solutions.csv` | First correct loop at/after d, early-final-match flag, nominal and last correctness |
| `survival.csv` | Continuous solution survival with observed, censored, and never-solved counts |
| `summary.json` | Run completion, settings, source/checkpoint/data hashes, transform version, transformed-task hash, timing, nominal metrics, and dynamics summaries |

Only `status: complete` runs have finished all exports. Run time covers inference and trajectory export; derived dynamics/export time is excluded from the inference throughput, as are loading and encoding. No hidden states or full logits are saved. CSV exports support downstream figures; plotting is not implemented in this change.

### Metric conventions

`transition_rates.csv` separates `all_loops_observational` from `post_nominal`. For the latter, include an example in transition t→t+1 only when t >= its requested depth. This aggregate cohort can grow with t. Both adjacent accuracies on any row use that row's same cohort, so `net_gain = accuracy_at_next - accuracy_at_t`. Compare depth-specific rows to avoid mixing depths. Empty conditional denominators are blank in CSV, never reported as zero measured repair/damage.

Survival starts at the first final-correct readout at or after d. An early final match does not start survival. At offset k, include only examples observed through first-correct+k. Success requires all readouts in that interval to remain correct; later recovery does not restore continuous survival. Never-solved examples and censored tails are explicit. This is an observed-cohort proportion, not a censoring-adjusted population estimate.

The summary includes a **hold-the-nominal-prediction baseline**: stop changing the decoded answer at d. Its later accuracy equals nominal final accuracy and it cannot repair wrong answers. This is a readout-halting comparison, not evidence from an implemented hidden-state no-op ablation. Confidence/stability stopping has separate, unconfirmed interfaces in the [adaptive guide](adaptive_compute.md); hidden-state diagnostics remain future work.

Nominal CE is unchanged and excludes t>d. Final-target margins remain available after completion. A positive raw-logit margin is stricter than the A–Z argmax convention in a tie. These metrics do not automatically declare a research gate passed.

### Validation

Focused tests cover terminal execution through 64 steps, original-record preservation, deterministic transformation, conditional counts, the net-gain identity, censoring, recovery after damage, malformed trajectories, a no-write preview, and offline tiny-Qwen checkpoint inference. The integration case checks nominal agreement with `naive_test` on the transformed tasks and inference without optimizer state. The Mac full suite passed 92 tests in 54.83 seconds; the five focused cases passed in 19.82 seconds. The continuation data/tokenizer preview and CLI help checks also passed. See [current status](status.md). No pretrained inference, training, or performance claims are made by these tests.

## Failure and performance diagnostics

Use the opt-in [diagnostic commands](diagnostics_and_performance.md#implemented-diagnostic-commands) for reference-checked first-error categories and conditional risk sets, paired suffix/depth-cue probes, and warmed performance attribution. New full-loop rows include A–Z-tie-broken target rank and top-three symbols. Old traces remain valid with absent confidence/rank fields left unknown. This does not change accuracy, loss, readout or stopping semantics.

## Paired requested-count diagnostic and depth-12 cohorts

`bash probe_steps.sh` evaluates the existing fixed-prompt step-3250 checkpoint on 32 depth-16 development mappings, each with requested Steps 6, 7, 8, 9, 10, 12, and 16. Only Steps and the corresponding reference targets change; the rule table/start stay identical. `--dry-run` validates variants without loading weights or writing files. The shared evaluator records 20-loop symbol/stop traces and answer-position hidden norm, update norm and cosine to the previous state. `pairs.csv` measures common-prefix prediction agreement against the depth-16 variant, nominal correctness and first threshold crossing. These are offline stop decisions, not actual stopped latency. This controls map identity while testing requested-count/suffix-length effects; hidden-state scalar changes alone do not identify a mechanism.

`bash eval_depth12.sh` evaluates the new selected sparse-count checkpoint via `eval_ckpts.sh`, then aggregates `cohorts.json`: trained requested counts 1–6/8/10/12; held-out requested counts 7/9/11; and unseen recurrent depths 13–20. Full and genuinely stopped evaluation remain separate. Threshold 0.5 and cap 24 are explicit, independent of individual task depth. The test split and seed-29 confirmation are not used. See the [run instructions](training_pointer.md#depth-12-with-held-out-counts-current-desktop-run).

## Depth-12 checkpoint progression

Run `bash compare_depth12.sh` on the CUDA desktop after pulling. It evaluates steps 4500, 5000, 5500 and 7500 of `models/stage1_pointer/depth12-fixed-prompt-seed47-gaps` on the same 1,000 seed-47 depth-13–20 development examples. It uses the existing full-loop evaluator, float32/eager, batch 16, and 24 forced loops. Learned stopping is bypassed. Step 5000 is deliberately rerun under the same invocation settings. No training or confirmation evaluation occurs.

The timestamped `eval/pointer_loops/depth12-checkpoint-comparison-*/` directory contains each full evaluation, a combined `run.log`, and:

- `depth_comparison.csv`: complete-trajectory accuracy, mean correct-prefix length and first-error counts by requested depth; `none` means no nominal error.
- `conditional_transitions.csv`: probability of the next transition being correct given every earlier transition was correct, by requested depth and loop, with risk-set denominators. Empty rates mean no surviving examples.
- `paired_examples.csv`: each example's trajectory outcome and correct-prefix change relative to step 5000.
- `comparison.json`: input artifact hashes and evaluation provenance.

Only loops through the requested depth contribute to these diagnostics. The aggregator rejects mismatched datasets, example identities/targets and core evaluation settings. This comparison tests whether additional optimization extends or shrinks R's reliable horizon; it does not identify the internal cause or establish that another epoch will help. Pretrained execution of this launcher remains pending. All four checkpoint binaries and the dataset must be present; missing files or evaluation failures stop the launcher. Existing artifacts are preserved.

## W&B coverage for standalone evaluations

`naive_test`, `loop_test` (including learned stopping), `overscaling_test`, `paired_steps`, and `compare_checkpoints` accept `--wandb-mode online|offline|disabled`, `--wandb-project` (default `loopformer`) and `--wandb-entity`. They also honor `WANDB_MODE`, `WANDB_PROJECT`, and `WANDB_ENTITY`. Direct CLIs default to disabled. The current `eval_depth12.sh`, `compare_depth12.sh` and `probe_depth12.sh` wrappers default to online; use `WANDB_MODE=disabled` to retain local-only behavior. Dry runs never initialize W&B.

A completed evaluation is published after local files are saved. Each evaluation gets a separate W&B run with saved scalar metrics, provenance and checksums, diagnostic CSV tables, and a versioned artifact containing the summary and all top-level CSVs. Comparison runs additionally plot complete trajectories by requested depth and conditional transition accuracy by loop for each task depth. Interactive tables show at most 5,000 rows and record both shown/total counts; full rows remain in the artifact. Large trajectory/prediction CSVs are artifact-only. Model tensors and checkpoint directories are never uploaded. These CSVs can contain task text/predictions; they are experimental data, not model weights.

`wandb_eval_run.json` stores tracking identity locally. Uploads are retrospective, so SDK system monitoring is disabled and inference timings remain those in the saved summary. This does not add live evaluation progress charts or kernel profiling. If upload fails, inference artifacts remain available and the command fails visibly. Retry the uploader rather than rerunning inference. Repeated publication creates a new tracking run; it does not resume or silently replace an earlier upload.

Publish the already-completed comparison and its four nested evaluations without model loading:

```bash
python -m scripts.eval.tracking \
  --results eval/pointer_loops/depth12-checkpoint-comparison-20261004T191905Z-1322 \
  --recursive --wandb-mode online
```

The same uploader can publish other completed diagnostic directories containing `summary.json` and CSVs. Their existing metric semantics are preserved; specialized plots are currently provided for checkpoint comparisons. No historical training run is imported by this evaluator utility.

## Matched requested counts after depth-12 training

Run `bash probe_depth12.sh --dry-run`, then `bash probe_depth12.sh`. The existing paired-Steps evaluator now accepts `--depths` and `--loops`; historical defaults remain unchanged. The new launcher fixes 32 depth-20 seed-47 development mappings, counts 12/14/16/18/20, 24 loops, batch 8, and checkpoints 5,000/7,500. All counts are two-digit inputs. It saves exact task variants, checkpoint/data/source hashes, per-loop predictions/stop logits and working-state norms/update norms/cosines under `eval/pointer_probes/depth12-paired-*/`. Each checkpoint gets W&B coverage; stdout/stderr share one local log. Missing checkpoints and failures stop the launcher.

Interpret comparisons only within both prompts' nominal horizons; extra loops after a smaller requested count are stress observations. The largest requested count is the paired-summary anchor. No new training or stopping-policy calibration is performed. See the [checkpoint progression report](experiments/stage1_depth12_progression.md) for the reason to run this diagnostic.

## Executor upgrade diagnostic bundle

`bash eval_executor.sh [training-run-directory]` loads the selected checkpoint
from the new run (default `models/stage1_pointer/executor_r-seed61`) and records
all output beneath `eval/pointer_diagnostics/`. It runs:

1. A **matched development panel**: 32 validation graphs/start states, requested
   counts 6/8/12/16/24/32/64, exact reference targets including cycles. Compare
   the same weights/cases under FP32/eager, FP32/SDPA, and BF16/SDPA.
2. Full validation and depth-development execution through 64 loops, using
   BF16/SDPA, batch 16, direct R/C diagnostics and working-state summaries.
3. Actual batch-one stopping on both splits at threshold 0.5 and safety cap 64,
   using FP32/SDPA. These latency results have a different precision/batching
   scope from forced execution and are not a matched-compute speedup claim.

The shared evaluator now exports optional `r_prediction`,
`r_intermediate_correct`, and `r_c_agree` alongside existing C scores, exact targets,
stop logits, confidence and state norms. No reference answer is supplied to the
forward pass. `--precision`, `--attention`, and `--record-states` make forced-run
compute explicit; defaults remain FP32/eager. Stopped timing currently requires
FP32. Naive evaluation still reads C at externally requested depth; actual learned
halting remains a separate measurement.

The integrated panel adds first-error histograms, transition accuracy conditional
on a correct prefix, local consistency with the model's previous prediction,
R-correct/C-wrong and C-correct/R-wrong counts, matched-count disagreements and
transient-length/cycle-period strata. Precision comparisons report prediction
flips and maximum target-margin change on exactly matched rows. R/C disagreements
localize observations; they do not prove which block caused an error. Repeated
queries share graphs and should not be treated as independent statistical trials.

Run the panel directly with `python -m scripts.eval.executor_diagnostic --model
 <checkpoint> --device cuda --precision-check`. Raw trajectories, per-example
summaries, pairs, loop risk tables, source/data/checkpoint hashes and JSON summaries
are saved and published through the existing optional W&B integration. A missing
checkpoint or failed stage stops the shell bundle. Confirmation data stays unused.

The v2 cyclic dataset uses strict **exact-step execution**. Reaching the same symbol
on an earlier cycle is not an exact stop. Do not reinterpret ordinary post-request
lookups as terminal damage. Absorbing-terminal transformation rejects cases where
the final state appeared earlier, since editing that edge would change nominal
execution. Terminal repair/damage and cross-family claims remain deferred.


## Matched controller remaining-work comparison

The [current runbook](training_pointer.md#remaining-work-comparison--current-desktop-run)
extends the existing `evaluate_controller`/first-crossing metrics across six
controller-only runs. Validation uses the shared 16-loop cache; deep replay uses
one shared 64-loop extraction. Every seed, exact-stop/joint outcome, numerical
error and paired difference is reported. Numerical diagnostic zero crossings
never replace the stop head. Readout traces sample eight predefined graphs per
panel; complete decisions and aggregates cover all queries. Both arms of the
lowest predeclared seed additionally execute bounded native learned-stop checks
through `scripts.eval.loop_test`. These guard checkpoint/replay fidelity, not
statistical confirmation. Cached replay cannot support adaptive latency claims.


## Completed affine-controller repair

The [repair protocol](controller_repair.md) uses the existing exact first-crossing
scorer and `controller_candidate --native-check`; it does not create another
stopping definition. The source executor remains frozen. Both selected and final
heads are scored at threshold 0.5, with independent rows for early/late/missing
stops, joint answer/timing and complete nominal execution. A cap fallback never
counts as an exact stop, even when a cycle repeats the correct letter.

The [three-seed report](experiments/controller_affine_seed61.md) records 100% timing
through 64 and all 132 native/replay matches. Validation counts 1–16 share 128
graphs; deep counts 13–64 share 32 different graphs. These repeated horizons and
optimizer seeds are not independent graph samples. Numeric requests through 63
were mostly seen during initialization/prefix training; recurrent supervision
unrolled only 12 loops. Report both distinctions, plus held-out values
9/17/29/41/53 and the one-value extrapolation to 64.

For native learned stopping use `loop_test --stop-policy completion
--stop-threshold 0.5 --loops 64` with the exported `best/` checkpoint; see the
[full command](controller_repair.md#completed-repair-and-current-use). Ordinary
`naive_test` forces requested-depth inference for recurrent checkpoints and cannot
validate stop timing. Cached replay supports quality/fidelity checks, not measured
adaptive inference speedups. Historical GRU-specific fitting/audit tools remain
historical interfaces; use the candidate/native path for affine checkpoints.


## Frozen reserved-test and larger benchmark

The user authorized opening the existing reserved test and new independent pointer
seeds after freezing the model. `bash benchmark_pointer.sh` composes the existing
native evaluator and a count-invariant trajectory benchmark; `--dry-run` writes
nothing. The [predeclared protocol](pointer_benchmark.md) owns sample sizes, seeds,
checkpoint/threshold, numeric cohorts, native fidelity and graph-cluster confidence
intervals. No tuning on these results is allowed within this run. The seed-29
split and unrelated confirmation stages remain unopened. A larger range must
pass every declared count rather than hiding failure behind a pooled mean.


The [completed report](experiments/pointer_frozen_benchmark.md) records a perfect
reserved test, 96.96% full executor trajectories through 256, and zero exact timing
at counts 70–256. All 126 native calls match reuse. All 345,600 decision rows and
aggregates pass independent reference checks. The wider end-to-end quality gate
fails; correct cyclic letters at wrong times remain failures. These evaluated
panels are now opened and cannot become new confirmation data after tuning.
