# Evaluation and reproducibility

Status: research metric and logging conventions remain planned, derived from project plan sections 6–8 and 23–27. Allowed-token raw-logit margins are implemented and tested in [outputs](../scripts/recurrent_qwen/outputs.py). [Stage 0](experiments/stage0_validation.md) records architecture measurements only; no task-dynamics experiment has run.

## Targets and loop indexing

`h_0` is the prelude output; `h_t` follows `t` recurrent passes. Task depth and recurrent depth are separate quantities.

Record both step-specific intermediate targets and the fixed final answer. Intermediate accuracy measures execution at the corresponding step. Final-answer correctness supports repair, damage, and solution-survival analysis. Label the target basis explicitly rather than using an ambiguous `is_correct` field alone.

Stage 1 data uses explicit requested steps and a non-repeating nominal path, with targets only at steps 1..d. `scripts.dataset.pointer.check_predictions` checks decoded symbols against that nominal trajectory; no model metrics have been measured. Post-completion behavior and future cyclic-task semantics remain open in [decisions](decisions.md). Report nominal execution separately from post-completion trajectories; a valid additional pointer lookup must not automatically count as damage.

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
