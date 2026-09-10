# Project status

Last updated: 2026-09-10.

## Current state

The repository is at the documentation/setup stage. The research plan, agent conventions, documentation index, planned architecture, phase guides, evaluation conventions, and decision record are present.

Basic inference dependencies are pinned in [requirements.txt](../requirements.txt), with [virtual environment setup instructions](setup.md) and ignore rules for `.venv/` and `venv/`. Package metadata was inspected; installation and inference have not been run.

An [ordinary-Qwen inference smoke test](../scripts/smoke_test_qwen.py) is implemented, with CPU/MPS selection, the instruct checkpoint's chat template, and a short greedy response. Python syntax, CLI help, invalid arguments, and local documentation links were checked. The current Python has no PyTorch or Transformers installation; pretrained execution remains unverified.

There is no model wrapper, task generator, training/evaluation implementation, or test suite yet. The local ML environment and pretrained model interfaces have not been validated. No research training or experiments have run, and no stage gate has passed.

## Milestones

| Milestone | State | Evidence |
| --- | --- | --- |
| Agent guide and execution documentation | Written | [Agent guide](../AGENTS.md) and [documentation index](README.md) |
| Virtual environment and inference requirements | Written; runtime unverified | [Setup](setup.md), pinned requirements, and virtual environment ignore rules |
| Ordinary-Qwen inference smoke test | Implemented; runtime unverified | [Script](../scripts/smoke_test_qwen.py); syntax and CLI checks |
| Stage 0 architecture and tests | Not started | None |
| Pointer data and generator tests | Not started | None |
| Stage 1 stepwise training | Not started | None |
| Stage 2 untreated dynamics | Not started | None |
| Stage 3 asymmetric dynamics | Not started | None |
| Stages 4–5 multi-family execution and holdout | Not started | None |
| Stage 6 natural-language transfer | Exploratory, not started | None |

## Next bounded implementation milestone

Follow [Stage 0](phases/stage0_architecture.md): inspect the environment and actual Qwen implementation, finalize the recurrent wrapper design, implement the wrapper and architecture tests, run validation, then stop and report. Pointer data and training follow as separate milestones after their prerequisites.

No implementation blocker has been established yet. Track unresolved choices in [decisions](decisions.md); settle only those needed for the active milestone.

## Updating this file

After each milestone, record implemented behavior, exact validation evidence or a link to its report, gate outcome, unresolved issues, and the next bounded task. Keep historical experiment details in linked Markdown reports rather than growing this file into a chronological work log.
