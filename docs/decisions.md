# Design decisions and unresolved questions

This document records choices that affect implementation or interpretation. The [project plan](project_plan.md) supplies the established research constraints. Open entries below are questions to settle before their dependent work, not approved changes to the experiment.

## Established constraints

- Study single-family recurrent dynamics before generality.
- Start with the specified pretrained Qwen, shared recurrence, frozen base weights, recurrent LoRA, and an optional bridge.
- Gate training on one-loop equivalence, weight sharing, and gradient scope.
- Preserve intermediate supervision and measure damage together with repair.
- Keep documentation in Markdown under `docs/`, with `AGENTS.md` as the root agent guide.

## Initial dependency baseline

For basic inference, use Python 3.11 with a local `venv` and pin PyTorch, Transformers, and Hugging Face Hub in [requirements.txt](../requirements.txt). Keep PEFT and training/test dependencies deferred until their implementation needs them. Direct pins provide a shared starting point; they are not a transitive lock or evidence of runtime compatibility. Package metadata has been checked, but installation and inference remain unverified. See [setup](setup.md). This setup choice does not change the research design or resolve D01.

## Open implementation and research choices

| ID | Resolve before | Question and consequence |
| --- | --- | --- |
| D01 | Stage 0 wrapper | What decoder interfaces, position/mask behavior, device/dtype, and numerical tolerances apply to the actual installed environment? Incorrect assumptions can break base equivalence. |
| D02 | Stage 0 wrapper; revisit only with evidence | Is a bridge needed, and where does it act? Its initialization must preserve one-pass equivalence and its reuse must preserve the shared-transition interpretation. Start by evaluating the simplest bridge-free design. |
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
