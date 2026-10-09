# Pointer analysis and agent handoff

Updated 2026-10-09. This is the active analysis contract. Start here before
expanding the figures or running another benchmark. The [research plan](project_plan.md)
still owns scientific stage gates; [model evolution](pointer_model_evolution.md)
owns the detailed technical/plain-language architecture explanations.

## Current objective and scope

The user wants a paper-style account of the working pointer system and its failure
modes, with the strongest documented checkpoint from each earlier architecture.
The 27-graph historical comparison is exploratory context, not the main population
comparison. Use **every one of the 1,350 graphs** in the existing seeds
307/311/313 panel. Show training/checkpoint progression only for the final executor
and its controller development, not sweeps across every old architecture.

This turn completed the evidence inventory and analysis design, plus an offline
reference check of all final-model graph trajectories. It did **not** run the
new population comparison, train a model, change thresholds, or regenerate the
paper figures. The collection script is implemented; the additional aggregation,
population inference and plotting work below is pending. The machine-readable
[protocol](../configs/pointer_analysis_protocol.json) is a design specification,
not an argument to an existing evaluation CLI.

The central questions are:

1. Does each R checkpoint emit the correct state after each transition on new graphs?
2. Where does that execution first fail, and what happens afterward?
3. Which graph/prompt properties are associated with failure?
4. Does the complete model stop at the requested transition and return its answer?
5. How did the successful executor learn, and which controller repairs explain
   the difference between strong execution and a working complete solver?

Confidence scores and hidden-state norms are not required main-paper metrics.
They do not identify a failure mechanism by themselves. No new internal probes
are selected just to produce more plots.

## Checkpoint registry: freeze these choices before population inference

All paths below are relative to `models/stage1_pointer/`. Desktop weights were
verified present by inventory. Config hashes and selection reasons are also in
the [machine-readable registry](../eval/pointer_analysis/evidence-20261009/checkpoint_registry.json).
"Selected" does not mean a demonstrated global optimum over every saved step.

| ID | Checkpoint | Architecture and selection | Owning evidence |
| --- | --- | --- | --- |
| `ce_full` | `depth6-fresh30k-seed37-batch4/step-002500` | Full-sequence recurrent LoRA, intermediate CE only, no stop head. Documented best depth-extension candidate among three full evaluations, chosen from opened development data, including validation depths 7–8. | [Fresh 30k report](experiments/stage1_fresh30k.md) |
| `joint_full` | `depth6-completion-seed37/step-003250` | Same writable full sequence, jointly trained completion MLP. Original minimum trained-depth validation CE. | [Joint-completion report](experiments/stage1_learned_completion.md) |
| `fixed6` | `depth6-fixed-prompt-seed37/step-003250` | Fixed prompt memory, one recurrent working position, LoRA and coupled completion MLP. Original minimum trained-depth validation CE. | [Fixed-prompt report](experiments/stage1_fixed_prompt_review.md) |
| `fixed12` | `depth12-fixed-prompt-seed47-gaps/step-005000` | Same architecture as `fixed6`, different data/depth/precision/optimizer exposure. Original minimum trained-count validation CE. | [Depth-12 report](experiments/stage1_depth12_progression.md) |
| `gru` | `controller-prefix-seed83/best` | Count-free full-R executor and private GRU controller; controller-development selection at update 5,800. | [Controller repair](controller_repair.md) |
| `final` | `controller-shared-number-precision-seed83/best` | Identical frozen executor to `gru`; shared number reader and learned scalar timer. Predetermined reader/cell fits, not population-based checkpoint selection. | [Reader/cell report](experiments/controller_number_reader.md) |

Only `gru` and `final` represent the regime after freezing R. Do not add every
initializer repair to the main architecture comparison. `fixed12` is explicitly
a recipe variant. CE step 3,250 remains the matched-update historical control in
[the earlier report](experiments/pointer_architecture_history.md), but it is not
the CE checkpoint for the new selected-candidate comparison. That difference
must appear in the caption. No new sweep to select old architectures is planned.

The final executor originates at `executor_r-seed61/step-002250`. The complete
final checkpoint must be used for autonomous inference; the executor training
checkpoint contains an unsuccessful earlier controller. Its readout, bridge and
R tensors are retained through the later controller exports.

### Architecture and implementation map

All use pinned Qwen2.5-0.5B-Instruct, layers P=0–5/R=6–17/C=18–23, shared R,
continuous recurrent working states, and intermediate symbolic supervision.
C's decoded letter is never re-entered into R. See [architecture](architecture.md)
and [training](training_pointer.md) for tensor/gradient contracts.

| Responsibility | Files |
| --- | --- |
| Shared backbone, recurrence and working-state flow | [`model.py`](../scripts/recurrent_qwen/model.py), [`memory.py`](../scripts/recurrent_qwen/memory.py), [`prefix.py`](../scripts/recurrent_qwen/prefix.py) |
| Historical rank-8 q/v LoRA and completion MLP | [`lora_utils.py`](../scripts/recurrent_qwen/lora_utils.py), [`completion.py`](../scripts/recurrent_qwen/completion.py) |
| Count routing, normalized bridge, private GRU/affine/shared-number controller | [`interfaces.py`](../scripts/recurrent_qwen/interfaces.py), [`number_reader.py`](../scripts/recurrent_qwen/number_reader.py) |
| Portable checkpoint configuration/loading | [`checkpoint.py`](../scripts/recurrent_qwen/checkpoint.py); each checkpoint's `recurrent_config.json` |
| Supervision, training and selection | [`objective.py`](../scripts/training/objective.py), [`runner.py`](../scripts/training/runner.py), [`controller.py`](../scripts/training/controller.py) |
| Graph generation and reference execution | [`pointer.py`](../scripts/dataset/pointer.py), [`benchmark.py`](../scripts/dataset/benchmark.py) |
| Population/paired inference | [`frozen_pointer_benchmark.py`](../scripts/eval/frozen_pointer_benchmark.py), [`checkpoint_comparison.py`](../scripts/eval/checkpoint_comparison.py), [`loop_test.py`](../scripts/eval/loop_test.py) |
| Aggregation and failure taxonomy | [`benchmark_metrics.py`](../scripts/eval/benchmark_metrics.py), [`pointer_failure_metrics.py`](../scripts/eval/pointer_failure_metrics.py) |
| Independent raw-reference audits | [`audit_pointer_benchmark.py`](../scripts/eval/audit_pointer_benchmark.py), [`audit_checkpoint_comparison.py`](../scripts/eval/audit_checkpoint_comparison.py) |

The successful executor combines fixed memory, count-free routing, normalized
re-entry, full R adaptation, direct R supervision, cyclic sampling and a new
recipe. It has about 180 million trainable parameters during executor fitting,
not the earlier 270k LoRA budget. The historical comparison cannot isolate which
change caused the improvement. The final timer has task-specific numerical
structure/supervision and ignores R correctness; it is not uncertainty-based
adaptive computation.

## Evidence inventory and availability

The [inventory report](experiments/pointer_evidence_inventory.md) explains all
evidence families, limitations and desktop-only records. Raw inventory artifacts
are under `eval/pointer_analysis/evidence-20261009/`:

- `mac-final.json.gz`, `desktop-final.json.gz`: complete present-file inventories,
  metadata, CSV schemas/row coverage, JSONL event coverage and checkpoint availability.
- `catalog.json.gz`: grouped run/dataset/table catalog.
- `collector_at_capture.py.gz`: exact collector source matching both snapshot hashes;
  the current collector additionally reports orphan weight directories explicitly.
- `coverage.json`: host differences and captured code revision.
- `checkpoint_registry.json`: the six selected candidates above.
- `full_final_observations.json`: independently recomputed reference targets,
  first errors and exploratory graph strata for all 1,350 final-model graphs.

Both hosts were at `b49030e` when inventoried. The desktop has 83 checkpoint
configs with inference weights; the Mac has 78 configs and no inference weights.
An extra controller-audit `predictions.csv` has 245,760 desktop-only rows; its
schema/coverage is inventoried. W&B filesystem caches are inventoried, but the
remote W&B API was not queried. Duplicate exports/replays are not independent trials.
The seed-17 manifest difference is provenance only: split counts/hashes match.
The benchmark graph file was also restored locally after inventory, at its normal
ignored `data/pointer/benchmark-seeds307-311-313/graphs.jsonl` path; its hash is recorded.

Key existing populations:

| Evidence | Coverage and proper use |
| --- | --- |
| Final full benchmark | 1,350 graphs, all requests 1–256; 345,600 paired decisions, full C/direct-R trajectories, graph metadata, exact stop records and native fidelity. Main final-result source. |
| Two earlier population benchmarks | Separate 1,350-graph panels for positional affine and shared-reader stages. Preserve negative results; different graph seeds prevent calling raw score changes a paired improvement. |
| Architecture history | 27 graphs, 30 counts per selected model; CE uses step 3,250. Context/audited evaluator evidence only, not the new population result. |
| Original architecture development evaluations | Typically 1,000 queries per validation/deep split, plus stopped traces where implemented. Different datasets/generation distributions; do not pool as a paired architecture benchmark. |
| Executor learning history | Fixed train-probe/768-query validation monitoring at updates 0–2,250, 250 apart, with per-loop losses/predictions and direct R readouts. Main learning-curve source. |
| Controller fitting/repair history | Saved development histories and numerical traces; reader/cell repairs are separate fits, not epochs of one uninterrupted run. |
| Causal input probes | Same-map requested-count changes, rule edits and prefix/restart interventions; small development cohorts. Explain the measured intervention, not a universal internal cause. |

Closed datasets remain closed. The inventory did not read unopened test records;
seed-61 test is already opened. No population tuning/threshold fit is allowed here.

## Population protocol: graphs, horizons and reuse

Use the **entire 1,350-graph panel**, 450 graphs per mode and 150 per seed/mode
stratum. Reuse its original graph/start/rule order and exact bytes. These are
opened data: this is retrospective characterization, not fresh confirmation.
Do not filter failures, fixed points, difficult examples or numeric requests.

For historical architecture comparison, declare **every integer request 1–256**:
345,600 paired queries per model, with every intermediate prediction through its
requested horizon retained. Reuse the final model's existing complete coverage
instead of rerunning it. Long integer coverage matters for exposing count/format
failures as well as execution; do not substitute another sparse request panel
and describe it as comprehensive. Jobs may finish in request blocks, but label
unfinished coverage explicitly rather than presenting it as the completed suite.

Older R sees Steps: re-encode and execute **each request independently**. It is
invalid to execute only `Steps: 256` and reuse that trajectory for shorter
requests. Its accuracy across requests need not be monotonic. A true survival
curve is computed within one fixed requested prompt, not by connecting different
Steps values. Use N=6/12/32/256 as declared fixed-prompt first-failure panels.

The isolated executor ignores Steps. Reuse its audited full-panel trajectories
only after verifying graph, tokenizer, routing, checkpoint/tensor and inference
code identity. GRU timing must replay **actual R observations**, not zeros or
gold states. Its observations may require one full extraction because saved final
letter strings cannot reconstruct continuous working vectors. The final timer
can reuse its existing audited population result; identical executor tensors
do not imply identical controller outcomes.

Hold threshold 0.5, cap 272, CUDA/FP32/SDPA scoring and allowed symbols fixed.
Record batch/device/source hashes. Verify actual batch-one stopped inference
against replay at boundary counts on a fixed fidelity panel, including available
failures. The fidelity panel may be small; **population accuracy cannot use it**.
Compare batch results and independently traverse raw mappings before plotting.
No new weights or thresholds are fitted.

The old dense/count-dependent paths are substantially more expensive than the
new count-free executor: fully forced coverage requires 44,409,600 example-loops
per historical candidate. Record measured throughput and projected elapsed time
before launching the full suite; split it into recoverable per-model/per-request
jobs. Any exact caching optimization needs native/dense equivalence checks;
there is no authorization to alter inference precision or model behavior silently.
Do not repeat work because plots are incomplete. Full selected-candidate population
coverage is currently missing; the existing comparison CLI does not yet provide
all planned reuse/structural summaries. Extend the shared infrastructure rather
than create another model loader/evaluator.

## Metrics and denominator contracts

For graph g/request N, let x_t be the reference state after t transitions,
p_t the C letter after loop t, F the first t with p_t != x_t, and s the first
learned stop signal. Force trajectories for executor diagnosis separately from
actual stopping. Missing stop is a failure even if the cap produces a correct letter.

| Metric | Definition and reason |
| --- | --- |
| Forced final accuracy | Fraction with p_N=x_N, ignoring controller. Distinguishes execution from timing; cycles can hide earlier errors. |
| Complete nominal trajectory | Fraction with p_t=x_t for every t=1..N. Primary execution metric for one-loop/one-transition behavior. |
| First-error distribution and survival | F per graph within fixed N; unfailed graphs censored at N. Plot fraction whose entire prefix remains correct through t, plus first-error histogram. Never restore prefix success after recovery. |
| Conditional first-error hazard | Number first failing at t / number correct through t−1, with explicit risk-set counts. Quantifies when previously successful execution breaks. Zero eligible graphs means undefined. |
| Restricted correct-prefix length | Mean min(F−1,N), treating unfailed cases as N. Report horizon N; do not average only failures or assign infinity. |
| Error persistence/recovery | Wrong-loop fraction and wrong-episode lengths; wrong→right rate against the continuing reference, with preceding-wrong denominator and censored tail counts. This is recovery of ongoing execution, not terminal solution repair. |
| First-error taxonomy | Mutually exclusive previous state, earlier visited state, later reachable state, outside start orbit, using established precedence. Count each graph once per fixed request and display exposure. |
| Decoded transition consistency | Fraction p_t=f(p_(t−1)), using Start at t=0; show separately before/after first error. Can distinguish wrong state with edge-consistent advancement from inconsistent readouts; does not prove the hidden state follows that letter. |
| Direct R versus C readouts | Correctness/agreement at first C failure, only where a trained direct head exists. N/A for historical models. A correct direct readout with wrong C is a localized discrepancy, not proof R is internally perfect. |
| Exact first stop | s=N. Checks counting, irrespective of answer. No head means N/A, not success or missing stop. |
| End-to-end success | s=N and p_s=x_N. Primary autonomous task metric. |
| Strict autonomous success | Exact first stop plus complete nominal trajectory. Separates a clean execution from a recovered/cyclically coincident final answer. |
| Stop residual/category | Signed s−N, absolute error and early/exact/late/missing rates. Missing stays a separate category; cap is not invented stop time. |
| Correct letter at wrong time | Returned letter equals x_N while s!=N (include explicitly marked cap-only coincidences). Never count as exact completion. |
| Failure decomposition | Exact+correct, exact+wrong, nonexact+correct, nonexact+wrong; mark missing-stop share separately within nonexact. Explains whether end-to-end losses belong to execution, timing, or both. |

For scalar timing independent of graph content, each integer has **one effective
controller outcome**, not 1,350 independent timing observations. A graph bootstrap
cannot establish uncertainty over untested integers. Keep the separate 1–8,192
controller-only stress result; it does not execute R or extend pointer quality.

### Which problems break it?

Primary strata: graph mode, data seed, start-orbit transient length, cycle period,
orbit size, and requested count relative to training exposure. Show success/failure
counts and population rates, not just how the failed subset is composed.
Use cycle bins 1/2–4/5–8/9–16/17–26 and transient bins 0/1–4/5–8/9–25;
also retain exact-value tables. Stratify period comparisons within graph mode,
because full cycles all have period 26. Requested count, recurrent age and
cycle-entry/revisit time are different axes; do not conflate them.

Secondary exploratory features from raw rules: maximum in-degree, active-rule
display position, target letter, and whether failure occurs near first revisit.
Letter/rule-position error rates need denominators among eligible target exposures
at comparable loop/request/mode. A confusion matrix of raw failure counts alone
does not establish letter bias. Do not use mean out-degree/mean in-degree as
graph difficulty: every table has one outgoing edge per state and mean in-degree one.
Multiple properties overlap; associations are not causal shortcuts. Controlled
renaming/reordering or training ablations would be additional experiments, not
conclusions from these strata.

## Final-model progression only

Plot the saved successful executor run's fixed train-probe and validation
per-loop CE/accuracy and complete trajectories at every recorded update. Keep
trained requested values 1–6/8/10/12 separate from held-out requested 7/9/11;
the latter still occur as supervised intermediate positions inside longer tasks.
These curves measure learning on development monitoring, not a fresh test.

Retained R snapshots are updates 0/250/500/750/1250/1500/1750/2000/2250.
Update 1,000 monitoring exists, but its weights were not found; do not invent a
population result there. If population depth-learning curves are needed, evaluate
these retained executor snapshots on all 1,350 graphs through 256 once each,
holding the final timer fixed as an explicitly labeled composed diagnostic.
Do not call that the historical controller attached at each training step.
No old-architecture checkpoint sweeps are planned.

The final solver was assembled in stages: full-R SFT, frozen-R controller fitting,
reader-only repair, then two-cell-parameter refinement. Show native controller
development histories where present and numerical reader/countdown before/after
results on aligned requests. Do not draw a single fictitious epoch curve through
different stages or compare different population panels as paired progress.
The final reader/cell fits do not have exported checkpoint sequences; their
recorded objective before/after is the available fitting evidence.

## Proposed paper figures and statistical presentation

1. **Architecture/data table and main quality figure:** selected candidates,
   trainability/training exposure/data support; final-letter, complete-trajectory,
   exact-stop and joint curves on the full paired panel. Separate undefined metrics.
2. **Execution failure figure:** fixed-prompt first-error survival/hazard and
   counts, first-error taxonomy, and recovery/persistence. Show all architectures'
   selected candidates, not training sweeps.
3. **Problem structure figure:** mode/orbit-period/transient failure rates with
   denominators and uncertainty. Include complete successes, not only a sorted
   image of failed graphs. Exploratory symbol confusion can go in the appendix.
4. **Stopping figure:** requested versus actual first stop, early/late/missing
   rates, signed residuals, and end-to-end failure decomposition. Show cyclic
   right-letter/wrong-time coincidences explicitly.
5. **Successful model learning figure:** executor training/development progression,
   optional retained-snapshot depth generalization, and clearly separated controller
   fitting/repair stages. No checkpoint progression for every historical architecture.

Bootstrap graph clusters within balanced seed/mode strata (2,000 draws, seed 239),
using the same resampled graphs for paired model differences. All horizons/start
readouts of a graph stay together. Pointwise graph-level Wilson intervals may
accompany per-count rates, labeled pointwise rather than simultaneous guarantees.
Report paired deltas and counts, including sparse/empty structural bins, without
claiming population-wide causal mechanisms or cherry-picking significant cells.
Only one successful executor training seed exists: graph uncertainty does not
measure optimizer-seed robustness. Figures export PNG/PDF/SVG and their numeric
tables/source hashes; every displayed cell must trace to independently audited rows.
Per-depth curves and explicit trained/interpolation/extrapolation ranges are primary;
an overall average must state its equal-integer weighting and cannot replace them.

## Findings already established versus missing measurements

The final panel has 42/1,350 graphs with a C error by loop 256; all first errors
occur at loops 2–30. Full-cycle failures are 18/450 (4%), random-function and
permutation failures each 12/450 (2.67%). Their pointwise intervals overlap;
this is not compelling evidence that graph mode alone explains failure.
Seed counts are 15/450, 15/450 and 12/450.

At first C error, direct R is correct on 5/42 graphs; the two readouts agree on
the same wrong letter for the other 37. Thirteen first errors are one reference
transition ahead; none repeats the immediately previous reference state. Twelve
of the 42 failed graphs nevertheless return the correct letter at loop 256.
These independently recomputed observations motivate separating prefix correctness,
state/readout discrepancies, and cyclic final-answer recovery; none proves an
internal mechanism. See the raw [observation artifact](../eval/pointer_analysis/evidence-20261009/full_final_observations.json).

Missing: full-panel selected-candidate historical trajectories/stopping, aligned
structural failure summaries for those candidates, and population progression of
retained successful R snapshots. Saved scalar confidence/norm observations do
not fill those gaps. Missing scientific controls also include isolated component
ablations, multiple successful executor seeds, larger state spaces/nonrepeating
50-state paths and cross-family transfer. Those are not authorized by this analysis.

## Handoff checklist

Local validation: 11 focused inventory/comparison/failure tests passed. Collection
completed on both hosts; the archived collector matches both snapshot hashes.
Compilation, 319 local file/directory links and whitespace checks pass. These
checks validate this handoff/inventory; they do not pass the pending population suite.

- [x] Inventory both hosts without model execution or opening closed tests.
- [x] Record checkpoint availability, selection provenance and architecture/source map.
- [x] Independently reconstruct all full-final first-error/structural observations.
- [x] Define the population, requests, target semantics, uncertainty and figure questions.
- [ ] Extend existing extraction/reuse to the full selected-candidate panel; freeze weights.
- [ ] Extend shared failure aggregation with optional direct R fields, fixed-N risk sets,
  exposure-normalized structural tables, and mutually exclusive stop failures.
- [ ] Audit raw targets, missing/undefined metrics, replay/native fidelity and plot cells.
- [ ] Run recoverable population jobs on the desktop; retain every result, including failures.
- [ ] Assemble figures/tables and update this document with actual results and exact commands.

Next bounded work is **analysis implementation/population evaluation**, not
another training run. Read host Git status before syncing; preserve untracked
desktop weights/metrics. Do not overwrite existing experiment directories or
silently relabel the 27-graph plots as full coverage. Other agents should record
completed work and unresolved gaps here, with evidence in `docs/experiments/`.
