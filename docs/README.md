# Documentation index

## Reading order

1. [Project plan](project_plan.md): research thesis, full stage sequence, constraints, and supported claims.
2. [Status](status.md): what exists, what has been verified, and the next implementation milestone.
3. [Architecture](architecture.md): implemented Stage 0 modules, recurrent execution, and gradient boundaries.
4. The relevant phase guide below, together with [evaluation](evaluation.md) and [decisions](decisions.md).

[Agent instructions](../AGENTS.md) define coding and documentation conventions. The project plan is the research source of truth; these guides translate it into bounded implementation work. Stage 0 and Stage 1 data are implemented and validated; Stage 1 training and later stages remain planned.

## Phase guides

For collaborator onboarding, see [Python environment setup](setup.md) and the pinned Stage 0 requirements.

| Phase | Purpose | Guide |
| --- | --- | --- |
| 0 | Verify the recurrent architecture before research training | [Architecture validation](phases/stage0_architecture.md) |
| 1, data milestone | Validate symbols, mappings, intermediate targets, and splits before training | [Pointer execution](phases/stage1_pointer.md) |
| 1, training milestone | Establish one-loop/one-transition execution | [Pointer execution](phases/stage1_pointer.md) |
| 2 | Measure repair and damage in untreated recurrent trajectories | [Overthinking](phases/stage2_overthinking.md) |
| 3 | Reduce damage while retaining useful computation | [Asymmetric dynamics](phases/stage3_asymmetric.md) |
| 4 and 4b | Train shared multi-family execution; optionally preserve ordinary Qwen behavior | [Transfer](phases/stages4_6_transfer.md) |
| 5 | Evaluate a wholly held-out task family | [Transfer](phases/stages4_6_transfer.md) |
| 6 | Explore natural-language and real-task transfer | [Transfer](phases/stages4_6_transfer.md) |

## Keeping documentation current

Recorded validation: [Stage 0 architecture on pretrained Qwen, CPU and MPS](experiments/stage0_validation.md).

Recorded data validation: [Stage 1 seeded pointer dataset](experiments/stage1_data_validation.md). Preview and exact seed-17 reproduction commands are in the [root README](../README.md#pointer-dataset-preview-and-reproduce). Generator commands, schema, and programmatic prediction checking live in the [Stage 1 guide](phases/stage1_pointer.md). All current dataset code is packaged under `scripts/dataset/`.

Put implementation explanations in the owning phase guide and shared architecture document. Put metric definitions in the evaluation guide, design decisions in the decision record, and current progress in status. Add verified setup and execution commands when the corresponding code exists.

Future experiment reports belong in `docs/experiments/` as Markdown. Each should identify its hypothesis, code and checkpoint versions, configuration, exact commands, validation, artifact locations, observations, and supported conclusions. Link new reports here and from status; do not rely on conversation history for evidence.
