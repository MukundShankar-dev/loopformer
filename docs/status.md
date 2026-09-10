# Project status

Last updated: 2026-09-10.

## Current state

**Stage 0 is implemented and its architecture gate passes on pretrained Qwen2.5-0.5B-Instruct on CPU and MPS.** The [validation report](experiments/stage0_validation.md) records exact commands, model revision, configuration, evidence, failed MPS attempts and their fix, artifact locations, and limitations.

**Stage 1 data is implemented and validated. The trainer is implemented; pretrained training has not started.** The seeded generator produced 13,000 examples under `data/pointer/seed-17/`, with exact per-step targets, balanced depths, disjoint whole rule tables, actual single-token answer IDs, and a reproducibility manifest. Dataset code is packaged in [scripts/dataset](../scripts/dataset/), invoked with `python -m scripts.dataset`. Rich dry runs print the configured depth distribution and five or ten examples spanning shallow through deep tasks without dataset writes. All files passed read-back checks and seed replay. After relocation, regeneration in memory matched all four original JSONL files byte for byte; the original data and manifest were preserved. See the [data report](experiments/stage1_data_validation.md), [Stage 1 commands/schema](phases/stage1_pointer.md), and [README reproduction instructions](../README.md#pointer-dataset-preview-and-reproduce).

Implementation lives in [scripts/recurrent_qwen](../scripts/recurrent_qwen/) with a composing [validation CLI](../scripts/validate_stage0.py). The configurable default split is frozen P (0–5), shared R (6–17), and frozen C (18–23), with all original weights frozen, no bridge, and q/v LoRA in R only. The 270,336 trainable adapter parameters are shared across loops. Outputs expose per-loop logits, optional hidden states, and explicit-label allowed-token margins.

The [ordinary-model pointer baseline](naive_pointer_eval.md) is implemented under `scripts/eval/`, with a shared task prompt in `prompts/pointer_task.txt`, strict final-answer accuracy, per-depth summaries, CSV/JSON artifacts under `eval/pointer_task/`, and Rich progress/throughput. `--test` runs three examples (or `--test 2` for two) and prints exact model inputs, responses, and RIGHT/WRONG decisions using the same scoring path as full evaluation. The user completed the full three-shot test run: **60/1,000 (6.00%)**, including 17.60% at depth 1 and 4.34% across depths 2–8. A read-only audit verified the saved results and found start-copying on 23.2% of examples, first-lookup matches on 18.5%, and substantial symbol bias. The [baseline report](experiments/naive_pointer_baseline.md) owns the configuration, artifacts, diagnostics, and limitations. The assistant did not rerun inference; automated tests load only random toy checkpoints.

The local Python 3.11.8 venv matches all seven explicitly pinned dependencies, including PEFT, Rich, pytest, and tokenizers 0.23.2. The tokenizer version already used for the verified dataset is now pinned directly for reproducibility. `pip check` passes. Rich terminal tables show the surgery and validation results; JSON saves reproducible evidence. [Setup](setup.md) documents commands and dependency versions.

The suite now has **87 passing tests**: 36 Stage 0 tests, 27 pointer-data cases, 17 baseline cases, and seven training/checkpoint cases. Baseline cases cover loading, prompting, scoring, metrics, output artifacts, and normal/two-/three-example CLI modes using random toy models. Recorded pretrained CPU and MPS runs have zero T=1 full-logit error before/after LoRA, verified module/parameter sharing at 1/2/4 loops, and a passing two-loop gradient-scope check. Original weights receive no gradients; both recurrent states receive gradients through frozen C. MPS uses a deterministic integer-slice answer readout after advanced indexing and gather failed under strict determinism. No pretrained optimizer step or research training has run.

The [training implementation](training_pointer.md) uses JSON configs under `configs/`, modules under `scripts/training/`, and adapter checkpoints/logs under `models/`. It preserves the frozen base and differentiable intermediate supervision. Rich displays a compact dashboard; JSONL and per-loop CSV retain detailed metrics. Both shipped configs passed data/tokenizer-only dry runs without model loading or output writes. New toy tests cover CE weighting/masking, frozen gradients, checkpoint readout round trips, exact interrupted-versus-uninterrupted updates, per-loop metrics, CLI training/resume, and checkpoint loading through `naive_test`. The [implementation validation report](experiments/stage1_training_implementation.md) records commands and limitations. These checks are implementation evidence, not learned pointer execution.

The older [ordinary-Qwen generation smoke test](../scripts/smoke_test_qwen.py) and its recorded CPU example remain available. It was not rerun during Stage 0; the new CPU/MPS evidence concerns ordinary forward versus recurrent forward and backward, not generation.

The [Windows/WSL2 CUDA handoff guide](windows_cuda_setup.md) covers installation from a fresh desktop, VS Code's WSL connection and Python interpreter selection, Codex CLI setup and project handoff, matching Python/direct dependency/model versions, deterministic CUDA checks, and data/history migration for the RTX 5070 Ti desktop. The official Python 3.11 Linux torch 2.14.0+cu130 wheel was verified in the PyTorch index. Desktop installation and pretrained CUDA execution remain unverified; the standalone Stage 0 CLI still accepts CPU/MPS only, while the trainer performs its own gate on the selected CUDA device.

## Milestones

| Milestone | State | Evidence |
| --- | --- | --- |
| Agent guide and execution documentation | Updated | [Agent guide](../AGENTS.md), including Rich preference and implementation layout |
| Virtual environment and Stage 0 requirements | Installed; consistent | [Setup](setup.md); requirements installation and `pip check` |
| Ordinary-Qwen inference smoke test | Implemented; historical CPU generation example | [Script](../scripts/smoke_test_qwen.py) and [evidence limits](setup.md#version-and-validation-status) |
| Stage 0 architecture and tests | **Verified; Gate 0 passes on CPU and MPS** | [Report](experiments/stage0_validation.md); 36 tests and pretrained checks |
| Pointer symbols/data and generator tests | **Implemented and validated** | [Report](experiments/stage1_data_validation.md); 13,000 examples, exact targets, token contexts, seed replay, disjoint tables |
| Ordinary-model pointer final-answer baseline | **Full three-shot run completed and audited** | [Report](experiments/naive_pointer_baseline.md); 60/1,000 correct, per-depth and output diagnostics |
| Stage 1 stepwise training | **Implemented; pretrained run pending** | [Training guide](training_pointer.md); intermediate loss, compact logging, per-loop validation, adapter checkpoints and resume, toy-model checks |
| Stage 2 untreated dynamics | Not started | No repair/damage measurements |
| Stage 3 asymmetric dynamics | Not started | Depends on earlier gates |
| Stages 4–5 multi-family execution and holdout | Not started | Depends on earlier gates |
| Stage 6 natural-language transfer | Exploratory, not started | No transfer claims |

## Next bounded milestone and unresolved issues

The data milestone and first full ordinary-model baseline are complete. Preserve the recorded three-shot result. The next user-run milestone is the [32-example Stage 1 overfit experiment](training_pointer.md#preview-then-run): inspect per-loop losses, fixed training-trajectory accuracy, validation behavior, and actual memory before the larger depth-1–4 epoch. The trainer fixes 26-symbol CE, equal-example loss weighting, nominal-depth masking, and seeded monitoring subsets. Empirical Gate 1 thresholds remain unresolved and must be selected before judging success. No pretrained training has been launched by the assistant.

D03–D05 in [decisions](decisions.md) are resolved for nominal Stage 1 data: explicit requested steps, no repeated state along the nominal path, and raw pair prompts with a validated space-prefixed symbol readout. D03/D04 still require decisions for post-completion/cyclic behavior before later dynamics work. A valid extra pointer lookup is not automatically damage. D01 (pinned decoder interface) and D02 (bridge-free Stage 0) remain resolved. No new implementation blocker is established.

Stage 0 proves architecture correctness on small probes, not useful recurrent computation or one-loop/one-transition behavior. Long-unroll training memory, mixed precision, and MPS margin-loss backward remain unverified or unimplemented. Adapter persistence and exact CPU resume are verified with toy models; actual pretrained training and training-device resource use remain unverified. Actual Stage 1 resource use must be measured before scaling; no future phase is authorized by this status entry.

## Updating this file

Record implemented behavior, gate outcome, evidence links, unresolved issues, and the next bounded task. Keep detailed validation and experiment history in linked reports under `docs/experiments/`.
