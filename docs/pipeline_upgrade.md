# Executor/controller pipeline upgrade

This is the implementation checklist authorized on 2026-10-04 after the
[pipeline reassessment](experiments/pipeline_reassessment.md). Existing runs remain
historical controls. New results must not be attributed to an individual change
when several changes are enabled together.

## Contracts and scientific consequences

The public input remains one raw `Rules / Start / Steps / Answer` prompt. An
explicit, deterministic text router removes the Steps line from the executor's
view. The controller sees the full prompt through the prelude, initializes its
own learned recurrent memory, and updates once per executor transition. It never
receives a numeric loop index, parsed requested count, reference state, or
remaining-step feature. Its output cannot modify R, and its BCE gradients cannot
modify R or the prelude. This structural separation deliberately changes the
architecture; merely moving a classifier outside R did not provide it before.

R still unrolls its own continuous working state. Exact state-after-transition
labels supervise C's symbolic output on every nominal loop. An optional direct
linear R readout supplies an additional local state loss; its prediction is
never fed back. A normalized learned re-entry bridge acts only before loop two
and later, preserving first-pass equivalence on the executor view. Normalization
is an intervention to test, not evidence that drift caused previous failures.

Trainability is explicit: legacy LoRA, all recurrent-block weights, or all Qwen
weights. The first new configuration trains all R weights and the bridge/readout;
P/C remain frozen to isolate recurrence adaptation. A full-model configuration
is a separate capacity control. Both are SFT from pretrained weights, not random
initialization. Neither is assumed to fit the desktop before a measured smoke run.

Graph sampling must not depend on requested depth. A new versioned generator
allows cycles and longer queries while preserving the old generator byte replay.
Strict trajectories and exact stopping remain primary: a repeated correct final
letter is insufficient evidence of correct execution or correct stopping.

## Checklist

- [x] Separate prompt routing, controller memory, and gradient paths; retain legacy loading.
- [x] Configurable normalized re-entry and full R/full model training.
- [x] Direct R intermediate supervision alongside existing C per-loop loss.
- [x] Depth-independent graph generator, cyclic targets and split checks.
- [x] Differentiable fixed-prefix reuse and selected-vocabulary projection with equivalence tests.
- [x] Explicit decay schedule, objective/gradient/resource logging and reproducible checkpoints.
- [x] Integrated matched-count, R/C readout and precision diagnostics through the shared evaluator.
- [x] Learned task-native positive control, clearly separate from the Qwen claim.
- [x] Fresh-run configs and fail-fast launchers, including dry run and disposable smoke.
- [x] Tiny-model contracts, documentation consistency and runnable handoff.

## What the linked paper contributes

[Liu et al., Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/html/2610.02185v1)
proposes training-free contrastive decoding of early/late representations or
logits for the **same next-token target**. It is not intermediate-step training.
Our loop t and loop t+1 generally have different correct pointer symbols;
contrasting them as though they predict the same target would alter the intended
semantics. We therefore do not add LoopCD as a pointer-supervision objective.

No finite empirical suite proves arbitrary-depth correctness. The enforceable
claims here are input/gradient isolation, parameter sharing, exact labels,
checkpoint fidelity, and equivalent optimized computation. Generalization still
requires held-out graphs, depths, independent seeds, and measured outcomes.

## Validation and handoff

The full local suite passed 196 tests with one CUDA-only skip during integration;
final focused validation passed 96 tests (one CUDA-only skip), then 31 training/executor tests after the final changes. The pinned-tokenizer, five-example
v2 preview passed without writing data. Tiny-model tests exercise a real CLI
training run, checkpoint reload, full-loop evaluation, actual stopping and the
integrated panel. Full R/full model and optimized/dense gradient contracts are
covered. The offline W&B SDK tests pass when local IPC is allowed.

The first pretrained desktop run and diagnostic bundle have now completed; see
[the audited result](experiments/stage1_executor_seed61.md). Forced execution is
strong, including on the small deep cohort, while learned stopping still fails.
The dataset replay and saved CUDA resource measurements were checked. The native
control comparison, multi-seed reproducibility and independent confirmation remain
unrun. Neither the root launchers nor the diagnostic panel uses the reserved test
split. Old artifacts were preserved.
