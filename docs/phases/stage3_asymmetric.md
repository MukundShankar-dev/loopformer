# Stage 3: asymmetric recurrent dynamics

Status: planned; requires demonstrated repair and overscaling damage in Stage 2. Source: project plan sections 9–13 and 31.

## Purpose

Train transitions on states the model actually visits, reducing damage without losing repair or stepwise execution. Correct hidden states may continue moving as long as their answers remain safe.

## Implementation approach

1. Sample a rollout length and generate `h_t` with gradients disabled using the current recurrent model.
2. Detach `h_t`, then compute one additional shared recurrent transition and coda readout with gradients enabled.
3. Classify the current state using its labeled answer margin and configured `gamma > 0`.
4. Apply the corresponding next-step objective and update only the recurrent adapters/optional bridge.

For wrong states before nominal completion, use cross-entropy toward the known next intermediate target. For wrong states after completion, use the final answer as the recovery target. Fragile and robust correct cases both use `max(0, gamma - next_margin)`; the robust case has zero retention loss while the next state stays inside the safe region.

Before coding the objective, settle the target basis and branch precedence for unfinished programs, the exact completion boundary, and early/repeated final-state visits in [decisions](../decisions.md). The project plan does not fully specify these cases. Do not silently freeze valid intermediate progress or substitute a different training objective.

Document rollout-length sampling, case weights, batch reduction, CE vocabulary, and treatment of empty cases. Validate off-by-one targets, gradient detachment, margin boundaries, and the zero-loss safe region.

## Controlled comparisons

| Variant | State generation | Step/repair objective | Retention |
| --- | --- | --- | --- |
| A | Differentiable unroll | Yes | No |
| B | Detached model rollout | Yes | No |
| C | Detached model rollout | Ordinary task loss, definition to settle | Yes |
| D | Detached model rollout | Yes | Yes |

Use the same starting Stage 1 checkpoint, data splits, and documented budget accounting. Report compute differences between differentiable unrolls and detached updates. Define variant C before experiments so it is distinguishable from D.

Include fixed-budget, confidence-stopping, prediction-stability-stopping, and trivial no-op baselines. Practical halting must use observable predictions, not the labeled true-answer margin; label any oracle comparison explicitly. Probe-based stopping is optional later work.

## Acceptance gate

Compared especially with variant B, damage should fall substantially at 16–64 loops while repair remains useful, intermediate execution persists, and net recurrent gain stays useful farther beyond training depth. Compare survival and the depth-by-loop heatmap using the same evaluation examples.

A near-zero damage rate with collapsed repair does not pass. Hidden-state motion is diagnostic evidence, not by itself proof of useful computation. Report negative outcomes and comparisons where halting performs as well as or better than transition training.

## Implementation record

No detached-rollout trainer, loss implementation, baselines, or results exist yet. Add concrete interfaces, validation commands, configurations, reports, and gate evidence as implemented.
