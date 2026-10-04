# Project status

Last updated: 2026-10-03.

## Current priority

Stage 1 failure diagnosis. The [integrated frozen mechanism suite](diagnostics_and_performance.md#integrated-frozen-mechanism-diagnostic) completed on pretrained steps 2500, 3250, and 3750; see the [audited report](experiments/stage1_integrated_mechanism.md). Early relevant-edge edits are followed on 106/106 eligible trials, with 53/53 eligible cases following two distinct replacement targets. At late first errors, only 17/70 relevant edits follow the replacement, 19/70 irrelevant edits change the answer, and only 6/31 eligible cases follow both replacements. Same-mapping valid `Steps: 17` comparisons have negligible late effect. Checkpoints 3250 and 3750 have better early readouts but fewer complete deep trajectories than step 2500 on matched examples. This narrows the failure to recurrent-history-dependent behavior without locating a unique internal cause. The next bounded question is whether the *same rule-table edge* fails more often when reached after a longer recurrent history; predeclare a matched-edge age control before another training or architecture change. Seed 29 remains reserved; seed-17 evaluations are development diagnostics.

The [architecture cross-check](architecture.md#literature-cross-check-2026-09-30) finds that the code implements a valid shared-middle-block depth recurrence. It lacks the explicit loop-time signal of the Universal Transformer and the per-loop input injection studied by Yang et al.; these are differences to investigate, not diagnosed defects. Earlier probes decoded C's answer-position A–Z output each loop and saved scalar R-state changes; the integrated suite now records answer-position R/C vectors, but neither yields a direct symbolic pointer from R.

The integrated suite saves answer-position R/C vectors and per-layer active-rule attention summaries for offline inspection. Its expanded implementation passed **133 local tests**, including the sequential three-checkpoint tiny-model run and matched-metric checks. The 1,296-prompt original/variant preflight and 256-prompt horizon preflight passed with the cached pinned tokenizer. The pretrained desktop run completed in about 7m49s; all saved artifact, data, and source hashes match locally, and all 3,264 native predictions/targets match the earlier six full evaluations. Adapter binaries and saved tokenizers remain on the CUDA desktop, so their contents have not been independently hashed here.

## Latest evidence

The fresh 30k baseline completed on CUDA in 60m54s, with 7.01 GiB peak CUDA tensor allocations. It started fresh adapters, not a resumed curriculum checkpoint. Full evaluation of steps 2500, 3250, and 3750 supports substantially stronger depth extension than the earlier curriculum. Step 2500 scores 98.0% complete trajectories at depths 1–6, 94.4% at both depths 7 and 8, 73.6% at 9, 41.6% at 10, 14.4% at 11, 1.6% at 12, and zero at 13–16. It is the working depth-generalization checkpoint among the three fully evaluated candidates; step 3250 is the trained-loss-selected reference. See the [fresh-run report](experiments/stage1_fresh30k.md).

The baseline's 34,816 training diagnostic rows and six full evaluations (72,000 loop rows) were audited against reference execution, metric aggregates, and source hashes. The new training dataset and model binaries are not on the Mac; full training-data hash and weight contents were not independently rechecked here. These are single-run development results, not a passed confirmatory gate.

## Completed loss ablation

`loss_reduction: loop_mean` uses selected-dataset loop counts to give every trained loop equal total direct weight. Microbatch losses are weighted estimates of that dataset objective; accumulation preserves the same effective-batch gradient. The default `example_mean` behavior and historical checkpoint identities remain compatible. Changing the objective requires a fresh run or explicit adapter-only initialization, not resume. The completed ablation used fresh adapters; no repeat run is currently planned.

Training logs retain legacy example-mean CE and add optimized objective loss and dataset loop weights. New-run checkpoint selection averages per-loop CE over trained-depth validation examples only. Evaluation outputs and their existing loss semantics remain unchanged. The user completed the pretrained run and six full evaluations. See the [negative ablation report](experiments/stage1_loopbalanced.md) for matched-update results and audit scope. All three new candidates still fail beyond depth 11; none improves the previous working checkpoint’s farther-depth accuracy.

Validation: **107 tests pass in 91.15 seconds** on the Mac, including exact dataset-objective and gradient comparisons across microbatch partitions, masked targets, OOD exclusion from selection, legacy resume compatibility, a loop-balanced CLI update/resume, and cleanup retention/symlink checks. Config comparison confirms only `loss_reduction` differs from the baseline. Local Markdown file links and whitespace checks pass. The seed-37 dataset is absent locally, so the real-data preview must run on the desktop.

## Artifact retention

The user subsequently requested deletion of the completed loop-balanced run and its six full evaluations. Removed 24.08 MiB locally; retained the negative Markdown report and baseline. The desktop cleanup option `--loopbalanced-only --apply` removes the same named directories, including ignored checkpoint binaries; it has not been run remotely.

At the user's request, superseded pre-30k recurrent runs/evaluations and redundant 30k checkpoint directories were pruned locally (about 29.6 MiB). Historical tracked evidence remains in Git history; historical report paths may refer to pruned artifacts. The current 30k logs and six full evaluations remain. Checkpoints 2250, 2500, 2750, 3000, 3250, and 3750 are retained for comparison. Ordinary-Qwen baseline results, Stage 0 evidence, all datasets, and new/unknown runs are preserved. [Desktop cleanup](training_pointer.md#artifact-cleanup) also removes ignored weights and optimizer files that Git cannot remove on pull; it has not been executed remotely.

## Constraints and deferred work

Frozen pretrained base, shared recurrent q/v LoRA, intermediate supervision, and `use_cache=False` remain unchanged. The updated startup gate passed on the desktop with zero reported T=1 logit error. Batch changes can preserve optimizer/cursor state via explicit same-effective-batch resume, but bitwise numerical equivalence is not claimed. See [architecture](architecture.md) and [training](training_pointer.md).

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
