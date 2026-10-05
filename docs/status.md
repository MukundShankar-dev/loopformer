# Project status

Last updated: 2026-10-05.

## Current priority

The no-update [controller training audit](diagnostics_and_performance.md#controller-training-audit--current-desktop-command)
is implemented. Next desktop command: **`bash audit_controller.sh`**. It checks all
six remaining-work comparison runs at their selected and final checkpoints using
full cached train/validation panels, fixed graph panels for per-objective/module
gradients, and paired numerical-readout/stopping cross-tabs. No Qwen extraction,
optimizer updates, new checkpoint selection or confirmation data. Desktop results
and CUDA runtime remain unmeasured. Validation: 28 audit/remaining-work/controller
training tests passed in 68.54 seconds; after the final conditional summaries and
manifest checks, all 10 audit tests passed again in 8.63 seconds. Synthetic CLI
coverage verifies both arms and best/final checkpoints, objective-gradient
reconstruction, readout alignment, no mutation, no-overwrite behavior and launcher
failure propagation. Dry-run, shell syntax, 242 local file links and whitespace
checks passed. No pretrained audit ran locally.


The [remaining-work comparison has completed](experiments/controller_remaining_seed61.md)
on three matched optimizer seeds. Auxiliary supervision improves numerical
readout and familiar-range decrement accuracy but leaves exact stopping essentially
unchanged: trained 95.54%→95.31%, held-out 7/9/11 46.61%→46.70%, deep
1.703%→1.723%. Every run fails at count nine and every count 14–64. Before any
loop, auxiliary count-nine estimates average 7.59–7.73; sampled count-64 estimates
average about 12.4, followed by stops around fourteen. Audit the training recipe
before attributing those readouts to an architectural limitation: gradients are
clipped on 94–97% of logged auxiliary updates, and initial-count supervision is
diluted by the trajectory average. Neither observation proves the cause. Correct
long-range recurrence from a correct initial state is not yet established. Keep R frozen;
no new training recipe or confirmation run is selected. All 22,272 decisions,
384,192 numerical trace rows, 72 paired aggregates and bounded native checks were
audited; see the report for limits and exact cohort definitions.

Implementation remains `bash train_controller_remaining.sh`: its commands now
reproduce a completed protocol and must not overwrite existing outputs.
Validation: 28 focused remaining-work, controller-training and controller-diagnostic
tests passed in 81.11 seconds. Contracts include exact masked targets, full memory
gradients with detached executor inputs, bitwise CPU stop-only equivalence with a
passive readout, unchanged non-controller checkpoint tensors, native/replay
agreement, deep-cache reuse, all-seed reporting and launcher failure propagation.
Shell syntax, 285 local Markdown links and whitespace checks passed. No new
pretrained run or CUDA benchmark was performed locally.


The [controller-only run has completed](experiments/controller_seed61_training.md).
Update 2,700 improves trained-count validation exact stopping from 14.32% to
95.31%; held-out counts 7/9/11 score 45.83%. Native deep exact success is 28/1,664,
all at count 13; every count 14–64 fails, stopping at loops 10–15. Memory count
decoding at loop 12 improves from 6.25% to 67.71%, so immediate forgetting alone
no longer explains the failure. Retain that checkpoint as evidence; the authorized
comparison above has now completed. No confirmation evaluation is selected. The report audits all native stopping
decisions, cyclic coincidences, cache/native agreement, and diagnostic readouts.
The paragraphs below retain the preceding experiment context.

The user authorized the [pipeline upgrade](pipeline_upgrade.md). Implementation
now includes an isolated recurrent controller, count-free executor view, normalized
re-entry, optional direct R supervision, full-R/full-model training controls,
depth-independent cyclic data, differentiable prefix reuse, selected-symbol
projection, cosine decay and a matched diagnostic bundle. The existing depth-12
run remains a historical reference; no artifacts were deleted or overwritten.

The first full executor run and evaluation have completed on the desktop. The
[audited seed-61 report](experiments/stage1_executor_seed61.md) finds 99.09% complete
validation trajectories and 100% deep trajectories, with important sampling limits:
the deep split contains only 32 independent graphs repeated at 52 horizons. All
saved deep transitions pass independent reference execution; data hashes replay,
splits have no graph overlap, and the matched precision panel agrees. Existing
64-loop validation traces are fully correct on 125/128 graphs. Learned stopping
still fails: 10.74% exact stops on validation and none on deep queries, which all
stop at loop 3 or 4. This is strong executor development evidence, not end-to-end
completion or independent confirmation.

Retain step 2,250. The [controller diagnostic has completed](experiments/controller_seed61_diagnostic.md).
Count is 99.48% decodable from the prompt feature and 95.57% from initialized
memory on held-out graphs, falling to chance by loop 12 in the original controller.
Both continued and fresh controller-only tiny fits achieve 72/72 exact fit stops;
held-out trained-count accuracy improves to 66.67% and 62.50%, but extrapolation
remains weak. The user approved a separate controller-training phase on more independent
training graphs with R frozen and the same trained counts. The
[training and evaluation launchers](training_pointer.md#separate-controller-training--current-desktop-run)
are now implemented: `bash train_controller.sh && bash eval_controller.sh`.
The new pretrained run has not been launched locally. Selection uses trained-count
development exact stopping, then BCE; the exported best checkpoint preserves all
non-controller tensors and loads through the existing evaluators.
The audit checked all exported stopping decisions, probe confusion counts and
source hashes; the raw vectors and weights remain on the desktop.
The reserved test split and seed 29 remain untouched. No pretrained inference was
rerun on the Mac; this review audited saved artifacts and reproduced dataset bytes.
The full run took 127.9 minutes and peaked at 4.43 GiB allocated CUDA memory.
Implementation validation remains 196 tests with one CUDA-only skip; final focused
checks passed 96 tests with one skip, followed by 31 training/executor tests.
Controller diagnostic validation: 35 focused tests passed, including the offline
W&B uploader with local IPC permitted. After the final scoring/logging additions,
10 diagnostic tests passed, covering real cyclic early/late answer coincidences,
missing-stop semantics, replay equivalence, hook cleanup, graph-disjoint probe
selection, tiny-fit gradients, extraction/cache-reuse CLI and launcher failures.
Local file/heading links and shell syntax passed. No pretrained controller
features or fits have run on this Mac.

Controller-training validation: 30 focused training/diagnostic/executor tests passed.
After final launcher and telemetry changes, all nine controller-training tests
passed, including frozen-tensor preservation, cache reuse, portable checkpoint
reload, native learned-stop evaluation, selection isolation and launcher failures.
These use tiny local models; the new pretrained CUDA run remains unrun.

The following sections retain historical evidence.

The [depth-12 paired-count probe](experiments/stage1_depth12_paired_counts.md) is complete and its two W&B uploads were verified through the API. At checkpoint 5,000, identical maps yield 30/32 correct loop-14 readouts with Steps 18 versus 8/32 with Steps 20 (22 losses, no gains). All counts are two-digit. R is therefore sensitive to requested count even on the same nominal prefix; this does not uniquely identify the internal mechanism. Later-loop failure persists even for the better count variant, and step 7,500 is worse across all tested counts. The prior [checkpoint progression](experiments/stage1_depth12_progression.md) also argues against unchanged-epoch continuation. A count-invariance objective on common intermediate readouts is a candidate to design, not an implemented/selected training recipe. No new training or confirmation run is authorized by these results alone. The following paragraphs preserve historical context.

The fixed-prompt desktop run and both evaluation modes have completed. The [failure investigation](experiments/stage1_fixed_prompt_review.md) finds stronger forced execution at depths 7–9 than the previous joint run, a sharp first-error rise at loops 9–10, and persistent early stopping: all deep examples stop at five or six while 98.5% still have the correct intermediate pointer. Threshold calibration cannot solve it. The loss has no continue labels at loop six or later and no requested training depths above six. The user approved the broader-depth experiment. Sparse training counts, paired-Steps diagnostics, BF16/SDPA, depth-grouped updates and desktop launchers are now implemented; see the [runbook](training_pointer.md#depth-12-with-held-out-counts-current-desktop-run). CUDA fit/speed and the new scientific results remain unverified. The historical context below is retained.

Stage 1 failure diagnosis. The [integrated frozen mechanism suite](diagnostics_and_performance.md#integrated-frozen-mechanism-diagnostic) completed on pretrained steps 2500, 3250, and 3750; see the [audited report](experiments/stage1_integrated_mechanism.md). Early relevant-edge edits are followed on 106/106 eligible trials, with 53/53 eligible cases following two distinct replacement targets. At late first errors, only 17/70 relevant edits follow the replacement, 19/70 irrelevant edits change the answer, and only 6/31 eligible cases follow both replacements. Same-mapping valid `Steps: 17` comparisons have negligible late effect. Checkpoints 3250 and 3750 have better early readouts but fewer complete deep trajectories than step 2500 on matched examples. This narrows the failure to recurrent-history-dependent behavior without locating a unique internal cause. A same-edge, variable-recurrent-age control remains unrun. Seed 29 remains reserved; seed-17 evaluations are development diagnostics.

The [prompt-only learned completion ablation](experiments/stage1_learned_completion.md) has now run on pretrained Qwen. It retains exact intermediate pointer targets and adds a joint hidden-state stopping loss without an explicit loop/depth feature. The selected step-3250 checkpoint matches CE-only execution through trained depths 1–6 (742/750 complete trajectories versus 744/750), but falls to 27.2% at depth 8 and 0.3% across depths 9–16, versus 86.4% and 9.2% for the matched CE-only checkpoint. At the diagnostic 0.5 threshold, the head stops early on every depth-9–16 example. This negative result does not localize why the auxiliary objective degraded forced-depth behavior. Actual self-stopped inference subsequently ran for the fixed-prompt architecture; see the current review above. The subsequently authorized architecture comparison is described below.

The [architecture cross-check](architecture.md#literature-cross-check-2026-09-30) finds that the code implements a valid shared-middle-block depth recurrence. It lacks the explicit loop-time signal of the Universal Transformer and the per-loop input injection studied by Yang et al.; these are differences to investigate, not diagnosed defects. Earlier probes decoded C's answer-position A–Z output each loop and saved scalar R-state changes; the integrated suite now records answer-position R/C vectors, but neither yields a direct symbolic pointer from R.

The integrated suite saves answer-position R/C vectors and per-layer active-rule attention summaries for offline inspection. Its expanded implementation passed **133 local tests**, including the sequential three-checkpoint tiny-model run and matched-metric checks. The 1,296-prompt original/variant preflight and 256-prompt horizon preflight passed with the cached pinned tokenizer. The pretrained desktop run completed in about 7m49s; all saved artifact, data, and source hashes match locally, and all 3,264 native predictions/targets match the earlier six full evaluations. Adapter binaries and saved tokenizers remain on the CUDA desktop, so their contents have not been independently hashed here.

## Historical evidence and implementation milestones

W&B logging to `loopformer` and per-update CUDA memory peaks are implemented; see [training observability](training_pointer.md#wb-experiment-logging). The depth-12 launcher enables online tracking, while dry runs remain side-effect free. Validation: 20 tracking/training tests passed, including the real offline SDK lifecycle, tiny-model training, resume equivalence and CLI checks; five launcher tests passed. Desktop online/CUDA verification is pending; no new training recipe is selected.

The [paired-Steps probe](experiments/stage1_paired_steps.md) completed on 32 matched maps. Changing Steps 9→16 loses ten correct loop-nine predictions and gains none; all Steps 7/8/9 prompts still stop at six. Raw working-state norms grow while adjacent states become more aligned, without proving a stationary attractor or causal norm failure. All 4,480 trace rows and 224 paired summaries were audited; Steps-16 predictions match the prior full evaluation. The user reports the new depth-12 training is running. Keep that run unchanged and evaluate with `bash eval_depth12.sh` when it finishes.

At the user’s request, [fixed prompt memory and recurrent working state](learned_loop_completion.md#fixed-prompt-memory-and-recurrent-working-state) are implemented with a fresh-run launcher and actual learned-stop evaluation. The final answer position recurs while layer-specific prompt context stays fixed within each forward, with no externally supplied progress. Tiny-model equivalence, gradients, training/resume and evaluation pass. Those launchers have now completed on the desktop; the reviewed run took 64.5 minutes and peaked at 7.02 GiB CUDA allocations; research gates and seed-29 confirmation remain unchanged.

Local implementation validation: **163 tests passed in 142.65 seconds**, including tiny-model training, resume, checkpoint reload, ordinary and full-loop evaluation, actual stopping, and launcher failure propagation. After tightening startup checks to measure gradients specifically at the recurrent working position, **36 focused tests passed**. Shell syntax and whitespace checks pass. The Linux terminal recorder was tested with stubs; the actual WSL launcher and pretrained CUDA run remain unrun. This dense correctness implementation fixes prefix activations but still recomputes prefix layer operations; it makes no throughput-improvement claim.

The fresh 30k baseline completed on CUDA in 60m54s, with 7.01 GiB peak CUDA tensor allocations. It started fresh adapters, not a resumed curriculum checkpoint. Full evaluation of steps 2500, 3250, and 3750 supports substantially stronger depth extension than the earlier curriculum. Step 2500 scores 98.0% complete trajectories at depths 1–6, 94.4% at both depths 7 and 8, 73.6% at 9, 41.6% at 10, 14.4% at 11, 1.6% at 12, and zero at 13–16. It is the working depth-generalization checkpoint among the three fully evaluated candidates; step 3250 is the trained-loss-selected reference. See the [fresh-run report](experiments/stage1_fresh30k.md).

The baseline's 34,816 training diagnostic rows and six full evaluations (72,000 loop rows) were audited against reference execution, metric aggregates, and source hashes. The new training dataset and model binaries are not on the Mac; full training-data hash and weight contents were not independently rechecked here. These are single-run development results, not a passed confirmatory gate.

## Completed loss ablation

`loss_reduction: loop_mean` uses selected-dataset loop counts to give every trained loop equal total direct weight. Microbatch losses are weighted estimates of that dataset objective; accumulation preserves the same effective-batch gradient. The default `example_mean` behavior and historical checkpoint identities remain compatible. Changing the objective requires a fresh run or explicit adapter-only initialization, not resume. The completed ablation used fresh adapters; no repeat run is currently planned.

Training logs retain legacy example-mean CE and add optimized objective loss and dataset loop weights. New-run checkpoint selection averages per-loop CE over trained-depth validation examples only. Evaluation outputs and their existing loss semantics remain unchanged. The user completed the pretrained run and six full evaluations. See the [negative ablation report](experiments/stage1_loopbalanced.md) for matched-update results and audit scope. All three new candidates still fail beyond depth 11; none improves the previous working checkpoint’s farther-depth accuracy.

Validation: **107 tests pass in 91.15 seconds** on the Mac, including exact dataset-objective and gradient comparisons across microbatch partitions, masked targets, OOD exclusion from selection, legacy resume compatibility, a loop-balanced CLI update/resume, and cleanup retention/symlink checks. Config comparison confirms only `loss_reduction` differs from the baseline. Local Markdown file links and whitespace checks pass. The seed-37 dataset is absent locally, so the real-data preview must run on the desktop.

## Artifact retention

The user subsequently requested deletion of the completed loop-balanced run and its six full evaluations. Removed 24.08 MiB locally; retained the negative Markdown report and baseline. The desktop cleanup option `--loopbalanced-only --apply` removes the same named directories, including ignored checkpoint binaries; it has not been run remotely.

At the user's request, superseded pre-30k recurrent runs/evaluations and redundant 30k checkpoint directories were pruned locally (about 29.6 MiB). Historical tracked evidence remains in Git history; historical report paths may refer to pruned artifacts. The current 30k logs and six full evaluations remain. Checkpoints 2250, 2500, 2750, 3000, 3250, and 3750 are retained for comparison. Ordinary-Qwen baseline results, Stage 0 evidence, all datasets, and new/unknown runs are preserved. [Desktop cleanup](training_pointer.md#artifact-cleanup) also removes ignored weights and optimizer files that Git cannot remove on pull; it has not been executed remotely.

## Historical constraints and deferred work

The historical runs used a frozen pretrained base and shared q/v LoRA. The authorized executor upgrade changes trainability explicitly; intermediate supervision and disabled generation caching remain required. The updated startup gate passed on the desktop with zero reported T=1 logit error. Batch changes can preserve optimizer/cursor state via explicit same-effective-batch resume, but bitwise numerical equivalence is not claimed. See [architecture](architecture.md) and [training](training_pointer.md).

Overscaling execution, shortcut diagnostics, asymmetric training, multi-family transfer, and knowledge-retention benchmarks remain deferred. Gate 1 lacks predeclared thresholds and independent confirmation; Gate 2 lacks pretrained terminal-dynamics evidence. Supporting historical evidence lives in [the original CUDA report](experiments/stage1_cuda_5k.md) and the [documentation index](README.md).

## Research direction after the current bounded experiment

Failure diagnosis and performance profiling now precede another pretrained experiment. The next major research direction is [adaptive inference compute](adaptive_compute.md): analyze existing depth trajectories, establish offline oracle allocation, then evaluate heuristic stopping before a learned halting head. Offline oracle/heuristic summaries, opt-in stopped inference, synchronized latency recording, and a gated lightweight head trainer are implemented. They have no pretrained terminal evaluation, fitted pretrained head, measured adaptive latency gain, or confirmation result.

This extends the dynamics question: if extra recurrence helps some examples but is unnecessary or damaging for others, can inference allocate it per example? Reuse the full-loop evaluator, trajectory diagnostics, terminal repair/damage metrics, and synchronized timing. First establish completion semantics and useful recurrent dynamics, including the deferred terminal evidence needed for damage claims. Adaptive allocation does not replace asymmetric-dynamics or transfer gates, and thresholds must be frozen on development data before confirmation.

## Documentation update validation — 2026-09-29

Reframed the entry point and added the planned adaptive-compute roadmap without changing model code, historical reports, or experiment artifacts. Full suite: `python -m pytest -q` using `.venv` — **107 passed in 87.54 seconds**. A local file/directory and heading-link check passed for all 162 links in the nine edited Markdown files; `git diff --check` passed. No repository-provided documentation/link checker was found. No metrics utility, stopping policy, or new pretrained experiment was added.

## Adaptive-compute implementation update — 2026-09-29

The user expanded implementation scope while retaining the research gates. The canonical [adaptive-compute guide](adaptive_compute.md) documents the interfaces, exact commands, causal features, oracle semantics, headroom gate, and unverified parts. New code extends the existing loop evaluator and shared model path; fixed-depth behavior remains the default. Target-free margin/entropy fields are added only to new trajectory exports. A [read-only nominal-trace analysis](experiments/adaptive_nominal_analysis.md) checks the offline pipeline but is explicitly observational because original pointer rules continue after `d`. The local checkout has no pretrained adapter tensors, so real stopped latency and terminal repair/damage remain unmeasured. This records the adaptive implementation state before the later completed loss ablation.

Validation for this adaptive-compute change: `.venv/bin/python -m pytest -q` — **114 passed in 97.96 seconds**. Local Markdown file/directory/heading links passed for all 194 links across the edited Markdown files; `git diff --check` passed. The two derived 1,000-example observational exports passed coverage/category checks. No pretrained inference or training ran locally.

## Documentation organization — 2026-09-29

Active guidance is consolidated in the [documentation index](README.md): the research plan, status, architecture, pointer training, evaluation, adaptive compute, setup, desktop setup, and decisions. Earlier phase and single-command guides were merged into those owners; dated experiment reports remain separate evidence. This was a documentation-only reorganization. It did not run new pretrained experiments, change model behavior, or change historical results. That was the priority at reorganization time; the current priority above supersedes it.

## Diagnosis and cleanup update — 2026-09-29

Inspected training/evaluation code for metric synchronization, full-vocabulary projection, batching, attention and dtype choices. These are performance candidates, not profiled CUDA bottlenecks. New diagnostic logging and optimizations are proposed, not implemented. Cleanup’s three focused tests passed; no model execution was performed.

## Diagnostic implementation

Added reference-checked offline first-failure/risk-set analysis, small paired suffix/depth-cue probes with hidden-state scalar summaries, and warmed eval/disposable-update profiling with separate trace overhead. New full-loop exports include target rank and top-three symbols. Fixed-depth defaults, float32/eager execution, training objectives and model mechanics remain unchanged. The [canonical guide](diagnostics_and_performance.md#implemented-diagnostic-commands) owns commands, timing scope and limitations.

The baseline step-2500 depth-9–16 trace analysis completed locally: 164/1,000 complete trajectories; first failures split into 191 previous-state repeats, 317 other earlier-path predictions, 163 future-path predictions and 165 off-path predictions. These observations do not establish a causal failure mechanism. Paired pretrained probes and CUDA profiling require the desktop weights and have not run locally.

Validation: full local suite passed **119 tests in 115.18 seconds**. New tests exercise reference/risk-set semantics, undefined denominators, paired target alignment, state summaries, unchanged forwards under profiling hooks, frozen-parameter preservation, all diagnostic CLIs, and byte-identical source checkpoint files before/after profiling. CLI tests use random tiny local Qwen models, not pretrained weights.

## Offline failure review

The [saved-artifact review](experiments/baseline2500_offline_review.md) validates all six baseline evaluations and matches all 32 original probe trajectories to historical predictions. Late failures show weakening target margins and smaller answer-state updates; reference suffix restarts restore confident execution at 22/22 aligned first errors. Later checkpoints improve trained-depth accuracy but shorten the correct prefix on hundreds of the same deep examples and increasingly repeat the previous reference state. These are descriptive results, not proof of hidden-state collapse or a missing supervision signal. The code audit confirms full intermediate supervision and differentiable recurrence; all prompt representations evolve each loop. Proposed next controls separate requested-depth changes from recurrence history and rule-context evolution. No model/training implementation changed.

## Controlled probe implementation

Added `--controls` to the existing probe: suffix restart retaining original displayed Steps, and rule-prefix refresh/no-op after six original loops. `bash probe_controls.sh` runs 32 validation and 32 deep examples, logs all output, and checks no-op/prefix invariance. See the [exact runbook](diagnostics_and_performance.md#controlled-restart-and-rule-context-experiments). Per-example first errors and per-depth/paired outcomes are persisted.

Validation for controls: full local suite passed **122 tests in 116.76 seconds** before the final tokenizer-boundary correction. After that correction and the shell-wrapper test, all **9 focused diagnostic tests passed in 18.65 seconds**. The cached real Qwen tokenizer was checked against all 2,000 seed-17 validation/depth-test prompts: every refresh prefix ends before Start. Stubbed shell runs verified both-split success, stderr capture, and stopping after the first failed command with its exit code. Documentation links, shell syntax and whitespace checks passed. The pretrained CUDA outcome is recorded below.

## Controlled probe result — 2026-09-30

The 32-example validation and depth-test controls completed on CUDA. Retaining the original displayed Steps in suffix prompts produces the same correctness as the ordinary suffix restart: 30/32 complete deep trajectories and 206/208 correct transitions. Directly refreshing only the Rules prefix after loop six is strongly harmful: deep complete trajectories fall from 6/32 to 0/32 and correct transitions from 276/400 to 186/400. At loop seven, accuracy falls from 28/32 to 5/32 and median target margin from 9.46 to -2.13. The no-op control is exactly invariant.

This rules against the displayed Steps reduction as the source of restart success in the sampled cohort and rejects the direct mixed-age Rules splice for this checkpoint. It does not rule out persistent-memory designs because the model was not trained on a fresh-prefix/old-state mixture. Do not expand this exact intervention to 1,000 examples. External decoded-symbol re-entry is not an acceptable next result because the controller would perform the state transfer. The [report](experiments/stage1_controlled_restarts.md) records the narrower diagnostic needed before selecting an architecture. No new training is authorized by this result.


## Sparse-count implementation validation — 2026-10-04

Implemented the depth-12 dataset/config/launchers, explicit sparse-count checkpoint selection, paired-Steps diagnostic, BF16 autocast, SDPA selection, depth-grouped updates, separate validation batching and batched CPU metric transfers. The full local suite ran with 168 passes, one CUDA-only skip and one failure in a test model's optional recording interface. After fixing that compatibility issue, all five focused recording/selection/precision tests passed. Two additional sparse CLI-preview/cohort aggregation tests passed; the launcher tests and sparse dataset seed-replay checks also passed. The actual pinned-tokenizer paired preview and five-example depth-12 data preview passed without dataset writes. CUDA compute fit, BF16 kernel execution, speedup, full new data generation and pretrained outcomes remain unverified; the desktop deepest-batch probe and smoke run are the next checks. No weights or existing artifacts were deleted.


The depth-12 batch-4 smoke completed all ten updates with finite metrics and passing gates; trained-depth validation CE fell from 3.48 to 3.08. The deepest-batch backward probe passed, with 16.42 GiB peak CUDA allocation. The user explicitly selected batch 4 despite limited memory headroom. Config and launcher now use batch 4/accumulation 1 (7,500 updates per full epoch); full training remains user-launched.
