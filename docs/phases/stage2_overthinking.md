# Stage 2: untreated overthinking

Status: planned; requires meaningful Stage 1 execution. Source: project plan sections 6–8 and 24.

## Purpose

Establish that the untreated learned recurrence can both repair wrong answers and damage solved answers when run too long. This is an evaluation stage with a fixed Stage 1 checkpoint.

## Execution

1. Freeze the selected checkpoint and record its selection criteria, configuration, and evaluation seeds.
2. Evaluate problem depths independently from recurrent depth. Include loop budgets `1, 2, 3, 4, 6, 8, 16, 32, 64` and held-out problem depths.
3. Record every adjacent transition along each trajectory, including loops between the displayed budgets. A 64-loop run can supply shorter prefixes under the same deterministic evaluation settings.
4. Compute final-answer correctness, repair, damage, net gain, margins, first-correct loop, and solution survival using [evaluation](../evaluation.md).
5. Report intermediate execution separately, and separate nominal execution from post-completion behavior.
6. Inspect failures for oscillation, confidence growth, representation collapse, or inert recurrence.

Completion semantics must be settled before interpreting a changed answer as damage. A valid extra pointer transition in a continuing program is not sufficient evidence of a stability failure.

## Deliverables and gate

Produce a task-depth × recurrent-depth heatmap, transition-rate curves with counts, margin trajectories, and survival curves. Save trajectory records and a Markdown experiment report with exact commands and artifact locations.

The gate requires observable useful wrong→right repair and right→wrong damage under overscaling. If either is absent, record the result and diagnose the mechanism or task semantics. Do not add retention just because evaluation code runs.

## Implementation record

No untreated checkpoint, trajectories, figures, or measured failure mode exists yet. Record results here and link the detailed experiment report when available.
