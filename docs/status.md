# Project status

Last updated: 2026-09-10.

## Current state

**Stage 0 is implemented and its architecture gate passes on pretrained Qwen2.5-0.5B-Instruct on CPU and MPS.** The [validation report](experiments/stage0_validation.md) records exact commands, model revision, configuration, evidence, failed MPS attempts and their fix, artifact locations, and limitations.

Implementation lives in [scripts/recurrent_qwen](../scripts/recurrent_qwen/) with a composing [validation CLI](../scripts/validate_stage0.py). The configurable default split is frozen P (0–5), shared R (6–17), and frozen C (18–23), with all original weights frozen, no bridge, and q/v LoRA in R only. The 270,336 trainable adapter parameters are shared across loops. Outputs expose per-loop logits, optional hidden states, and explicit-label allowed-token margins.

The local Python 3.11.8 venv has all six pinned direct dependencies, including PEFT, Rich, and pytest; requirements installation and `pip check` pass. Rich terminal tables show the surgery and validation results; JSON saves reproducible evidence. [Setup](setup.md) documents commands and dependency versions.

The focused suite has 36 passing tests. Pretrained CPU and MPS runs have zero T=1 full-logit error before/after LoRA, verified module/parameter sharing at 1/2/4 loops, and a passing two-loop gradient-scope check. Original weights receive no gradients; both recurrent states receive gradients through frozen C. MPS uses a deterministic integer-slice answer readout after advanced indexing and gather failed under strict determinism. No pretrained optimizer step or research training has run.

The older [ordinary-Qwen generation smoke test](../scripts/smoke_test_qwen.py) and its recorded CPU example remain available. It was not rerun during Stage 0; the new CPU/MPS evidence concerns ordinary forward versus recurrent forward and backward, not generation.

## Milestones

| Milestone | State | Evidence |
| --- | --- | --- |
| Agent guide and execution documentation | Updated | [Agent guide](../AGENTS.md), including Rich preference and implementation layout |
| Virtual environment and Stage 0 requirements | Installed; consistent | [Setup](setup.md); requirements installation and `pip check` |
| Ordinary-Qwen inference smoke test | Implemented; historical CPU generation example | [Script](../scripts/smoke_test_qwen.py) and [evidence limits](setup.md#version-and-validation-status) |
| Stage 0 architecture and tests | **Verified; Gate 0 passes on CPU and MPS** | [Report](experiments/stage0_validation.md); 36 tests and pretrained checks |
| Pointer symbols/data and generator tests | Not started | Next bounded milestone |
| Stage 1 stepwise training | Not started | Stop again after data validation |
| Stage 2 untreated dynamics | Not started | No repair/damage measurements |
| Stage 3 asymmetric dynamics | Not started | Depends on earlier gates |
| Stages 4–5 multi-family execution and holdout | Not started | Depends on earlier gates |
| Stage 6 natural-language transfer | Exploratory, not started | No transfer claims |

## Next bounded milestone and unresolved issues

Implement the [single-token symbol vocabulary and pointer generator](phases/stage1_pointer.md), then validate symbol contexts, mapping validity, exact intermediate targets, and train/test depth splits. Stop before training.

Resolve D03–D05 in [decisions](decisions.md) as required for that data milestone: completion/terminal semantics, cycles and repeated final-state visits, prompt format, and task-specific answer readout. D01 (pinned decoder interface) and D02 (bridge-free Stage 0) are resolved. No new implementation blocker is established.

Stage 0 proves architecture correctness on small probes, not useful recurrent computation or one-loop/one-transition behavior. Long-unroll training memory, mixed precision, MPS margin-loss backward, and adapter checkpoint persistence remain unverified or unimplemented. Actual Stage 1 resource use must be measured before scaling; no future phase is authorized by this status entry.

## Updating this file

Record implemented behavior, gate outcome, evidence links, unresolved issues, and the next bounded task. Keep detailed validation and experiment history in linked reports under `docs/experiments/`.
