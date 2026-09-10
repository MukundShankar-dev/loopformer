# Design decisions and unresolved questions

This document records choices that affect implementation or interpretation. The [project plan](project_plan.md) supplies the established research constraints. Open entries below are questions to settle before their dependent work, not approved changes to the experiment.

## Established constraints

- Study single-family recurrent dynamics before generality.
- Start with the specified pretrained Qwen, shared recurrence, frozen base weights, recurrent LoRA, and an optional bridge.
- Gate training on one-loop equivalence, weight sharing, and gradient scope.
- Preserve intermediate supervision and measure damage together with repair.
- Keep documentation in Markdown under `docs/`, with `AGENTS.md` as the root agent guide.

## Initial dependency baseline

Use Python 3.11 with a local `venv` and pinned direct dependencies in [requirements.txt](../requirements.txt). Stage 0 adds PEFT, pytest, and Rich to the original PyTorch/Transformers/Hub baseline. Direct pins are not a transitive lock; validation JSON records every installed distribution. See [setup validation](setup.md#version-and-validation-status).

## Stage 0 resolutions — 2026-09-10

- **D01, resolved for the pinned environment.** Preserve Transformers 5.17.0 decoder arguments, full causal attention, default RoPE, and ordinary-forward positions, with caching disabled. The CLI uses float32, eager attention, a fixed model revision, and `atol=rtol=1e-5` selected before pretrained validation. CPU and MPS pass. See [architecture](architecture.md) and the [report](experiments/stage0_validation.md).
- **D02, resolved for Stage 0; revisit only with evidence.** No bridge. R returns the same shape it accepts; direct shared recurrence preserves one-loop behavior and gradient flow. This does not prove that untrained recurrent states support pointer execution. A future bridge would change the transition and needs evidence and renewed validation.
- **Initial LoRA.** Adapt q/v projections in layers 6–17, rank 8, alpha 16, zero dropout, no bias training. This is a small starting configuration, not an optimized rank or target choice. With PyTorch linear-weight orientation, the added weight is `(alpha/r) B A`, where A is `[r,in]` and B is `[out,r]`. B maps back out of the rank space; it is not a bridge or the coda. Random A and zero B preserve the initial forward. A can have zero gradient until B changes. Injection follows the [PEFT low-level API](https://huggingface.co/docs/peft/en/developer_guides/low_level_api).
- **Readout.** Default to last unmasked input position and answer-position logits. Full logits remain available for equivalence and full hidden states are optional. Labels explicitly contain one token ID per loop. This settles an API default, not D05's task-specific prompt/symbol design.
- **MPS determinism.** Keep strict deterministic algorithms enabled. Advanced-index and gather answer readouts failed backward on unsupported deterministic MPS scatter operations. Equivalent integer slices and stacking pass without changing numerical tolerances or research semantics. This incurs one answer-index device synchronization per forward, acceptable for tiny batches. MPS margin-loss backward is not yet validated.
- **Output and layout.** Rich renders terminal configuration and checks; JSON saves exact inputs, revision, packages, source hashes, parameter/gradient details, and measurements. Downloads are opt-in in the Stage 0 CLI. At the user's explicit request, implementation lives in `scripts/recurrent_qwen/` and the composing CLI in `scripts/validate_stage0.py`; tests stay in `tests/`. The project plan's layout illustration was updated to match, without changing the research design.

## Open implementation and research choices

| ID | Resolve before | Question and consequence |
| --- | --- | --- |
| D03 | Pointer generator labels | How does the task specify requested depth and completion? Absorbing terminal states, explicit completion information, and a continuing mapping imply different learning problems. A valid extra transition must not be mistaken for damage. |
| D04 | Pointer generator labels and Stage 3 loss | How are cycles and early/repeated visits to the final state handled? Specify the correctness target before completion, branch precedence, and the exact completion boundary; otherwise retention can inhibit valid execution. |
| D05 | Stage 1 data/readout | Which answer position, token contexts, and prompt format expose the state? `h_0` need not decode to the start symbol through the frozen coda; do not assume this is an architectural guarantee. |
| D06 | Each training stage | What CE vocabulary, masking, loop/case weighting, and reduction apply? These choices affect confidence, gradients, and comparisons. |
| D07 | Stage 3 comparisons | What precisely is variant C's ordinary task loss, and how are budgets compared across all four variants? Ambiguity makes the retention ablation uninterpretable. |
| D08 | Stage 4 training | Which family is held out, how is input progress represented, and which data may inform model selection? Decide before training to preserve the zero-shot claim. |
| D09 | Each empirical gate | What sample sizes, seeds, thresholds, and uncertainty reporting establish meaningful execution, useful repair, and substantial damage reduction? Select criteria before judging the experiment. |

## Recording a resolution

For each resolved entry, record the date, status, chosen behavior, rationale, alternatives that affect interpretation, supporting inspection or experiment evidence, and affected code/docs. Update the owning phase guide and current status. If a technical constraint changes the research design, explain its scientific consequence before implementing it.

Routine implementation details can be resolved within the authorized scope. Do not create a new approval requirement merely because a choice appears here. Preserve genuinely unresolved research choices explicitly.
