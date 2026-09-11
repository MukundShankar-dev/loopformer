# Stage 2: untreated overthinking

Status: evaluation scripts are prepared; pretrained execution is **deferred** while the user pursues further Stage 1 training and depth generalization. No repair/damage result or Gate 2 pass is established. Source: project plan sections 6–8 and 24; the user-directed priority update is recorded in the [plan](../project_plan.md) and [status](../status.md).

## Purpose

Establish whether an untreated learned recurrence can both repair wrong answers and damage solved answers when run too long. This is an evaluation stage with a fixed Stage 1 checkpoint. Useful unseen-mapping execution already exists at trained depths, but depth extension remains weak; see the [full-loop evidence](../experiments/stage1_cuda_5k.md).

## Completion semantics

Original Stage 1 mappings continue after the requested depth, so an extra valid lookup must not automatically count as damage. The prepared experiment uses a **separate absorbing-terminal variant**, replacing the final state's outgoing edge with a self-loop. All nominal targets stay the same, but the prompt distribution changes. Inspect nominal execution and early-final shortcuts before interpreting post-completion dynamics.

Only transitions starting at t >= d count as post-nominal repair/damage. All-loop final-target counts are separately observational. Survival starts at the first final-correct readout at/after d and requires continuous correctness, with explicit censoring. General cyclic-task semantics remain unresolved. See [decision record](../decisions.md) and [metric definitions](../overscaling_eval.md#metric-conventions).

## Deferred execution

1. Complete the bounded Stage 1 continuation and review validation depth extension before launching this experiment.
2. Freeze a validation-selected checkpoint and record selection criteria, configuration, and evaluation seeds.
3. Preview transformed inputs without model loading or writes; then run a small 16-loop checkpoint test when ready.
4. Sweep every example through a common budget such as 32 loops. Every intermediate readout is retained; 64-loop resource use is unmeasured and should be checked before scaling.
5. Inspect nominal execution separately from post-nominal recovery, damage, margins, continuous survival, and a hold-at-d decoded-answer baseline.
6. Diagnose absent recovery/damage or shortcut behavior before adding any retention objective.

Commands and artifact definitions live in [overscaling usage](../overscaling_eval.md), avoiding duplicate run instructions here. `scripts/eval/overscaling_test.py` composes the existing checkpoint loader and full-loop evaluator; `scripts/dataset/terminal.py` owns the transform; `scripts/eval/overscaling_metrics.py` owns conditional dynamics. Training objectives, recurrent weights, and original datasets are unchanged.

## Deliverables and gate

Implemented exports provide exact transformed tasks, trajectory and adjacent-transition records, a depth-by-loop matrix, rates with counts, margins, and survival with censoring. Checkpoint/data/source hashes and transform version make runs traceable. CSVs support future plots, but plotting, hidden-state collapse/no-op diagnostics, and confidence/stability halting are not implemented in this change. No pretrained terminal trajectories or figures exist yet.

The gate requires both useful wrong→right repair and right→wrong damage under overscaling. A changed answer in an original continuing-pointer task is insufficient. If either phenomenon is absent, record that result and diagnose the mechanism or task distribution. Implementation tests cannot pass this empirical gate.
