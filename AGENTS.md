# Agent guide

## Start here

This project studies recurrent depth dynamics with Qwen2.5-0.5B first, then transfer across in-context algorithms. Prioritize scientific interpretability, correctness, reproducibility, then optimization.

Before working, read [the documentation index](docs/README.md), [the research plan](docs/project_plan.md), and [current status](docs/status.md). Then read the documents relevant to the task below. Explicit user instructions determine the current scope; a documented future phase is not an instruction to implement it now.

| Working on | Read |
| --- | --- |
| Research objectives, constraints, and claim hierarchy | [Project plan](docs/project_plan.md) |
| Module boundaries, model flow, and gradient behavior | [Architecture](docs/architecture.md) |
| Recurrent wrapper and architecture tests | [Stage 0](docs/phases/stage0_architecture.md) |
| Pretrained Stage 0 validation evidence and device limits | [Validation report](docs/experiments/stage0_validation.md) |
| Symbol vocabulary, pointer data, and intermediate training | [Stage 1](docs/phases/stage1_pointer.md) |
| Depth-6 training, initialization, and outward OOD evaluation | [Depth generalization setup](docs/depth_generalization.md) |
| Training commands, logging, checkpoints, and resume | [Pointer training](docs/training_pointer.md) |
| Windows desktop setup, CUDA dependencies, and state migration | [WSL2/CUDA guide](docs/windows_cuda_setup.md) |
| Untreated overscaling and failure analysis | [Stage 2](docs/phases/stage2_overthinking.md) and [terminal evaluation usage](docs/overscaling_eval.md) |
| Detached rollouts, asymmetric losses, and baselines | [Stage 3](docs/phases/stage3_asymmetric.md) |
| Multi-family execution, preservation, and transfer | [Stages 4–6](docs/phases/stages4_6_transfer.md) |
| Metrics, trajectories, figures, and reproducibility | [Evaluation](docs/evaluation.md) |
| Ordinary-model final-answer pointer baseline | [Naive pointer evaluation](docs/naive_pointer_eval.md) |
| Saved-checkpoint per-loop pointer evaluation | [Full-loop evaluation](docs/loop_pointer_eval.md) |
| Latest pointer training and full-loop evidence | [CUDA experiment report](docs/experiments/stage1_cuda_5k.md) |
| Fresh 30k baseline and loop-balanced follow-up | [Fresh-run report](docs/experiments/stage1_fresh30k.md) and [training commands](docs/training_pointer.md#loop-balanced-loss-experiment) |
| Unresolved choices and recorded design decisions | [Decisions](docs/decisions.md) |
| Implemented work, validation evidence, and next milestone | [Status](docs/status.md) |

## Code conventions

- Write modular, minimal, readable code. Give each module and function a clear responsibility; favor direct control flow and descriptive names.
- Keep model mechanics, task generation, training objectives, evaluation, and command-line entry points separate. Scripts should compose library functions rather than contain the implementation.
- Keep the current recurrent model implementation in `scripts/recurrent_qwen/`, with its composing CLI in `scripts/validate_stage0.py` and contract tests in `tests/`.
- Keep dataset generation, symbol validation, reference execution, persistence, and the dataset CLI in `scripts/dataset/`; invoke it with `python -m scripts.dataset`. Dataset tests stay in `tests/`.
- Implement only what the current phase needs. Introduce abstractions when they clarify a real shared interface; avoid speculative frameworks, unnecessary wrappers, and duplicated logic.
- Make dependencies and configuration explicit. Avoid hidden global state, import-time model loading, implicit downloads, and silent device or dtype changes.
- Use type hints on public interfaces. Document tensor shapes, loop indexing, target semantics, and non-obvious gradient behavior. Comments should explain reasons and constraints rather than restate code.
- Validate assumptions at boundaries and fail with useful errors. Do not catch broad exceptions to hide incorrect model behavior or failed experiments.
- Follow existing formatting and tooling once established. Add dependencies only when needed for the current work.
- Use Rich for readable terminal output: label configuration, measurements, and pass/fail results clearly. Save machine-readable validation evidence separately when applicable.
- Test scientific and implementation contracts: equivalence, parameter sharing, gradient scope, exact targets, and metric correctness. Use small focused tests and relevant integration checks; do not add tests that merely repeat implementation details.
- Keep changes focused, preserve unrelated user work, and report what was checked and what remains unverified.

## Scientific constraints

- Follow the stage gates in the project plan. The initial milestones are Stage 0 implementation and validation, then pointer data and validation, then Stage 1 training and evaluation. Preserve the plan's stop-and-report boundaries unless the user explicitly changes the scope.
- Reuse one recurrent module and parameter set across all loops. Initially freeze all original Qwen parameters; train only recurrent LoRA and an optional bridge.
- Require one-loop equivalence before training. Keep `use_cache=False` until recurrence is correct.
- Preserve intermediate supervision and the interpretation of one loop as one transition. Do not silently substitute final-answer-only training.
- Establish useful repair and overscaling damage before adding asymmetric retention. Validate single-family dynamics before multi-family training.
- Do not introduce the ordinary-Qwen anchor loss in the first pointer experiment unless a concrete failure justifies it. Keep shared adapters across families.
- Measure repair as well as damage; stability from inert recurrence is not success. Stable decoded answers do not require frozen hidden states.
- Keep instance, depth, cross-family, and natural-language transfer claims separate. Record negative results and unresolved limitations.
- Explain the scientific consequence of a necessary design change before making it, and record the decision. Unresolved proposals in the docs are not settled requirements.
- Design for the target 32 GB Apple Silicon machine: short prompts, tiny batches, gradient accumulation, and short initial training unrolls. Measure actual resource use before scaling.

## Documentation conventions

- Store all Codex planning, design, implementation, validation, experiment-report, and handoff documentation as Markdown files under `docs/`. `AGENTS.md` is the root instruction and navigation exception. Keep a root README, if used, to an entry-point description and links into `docs/`.
- Use [docs/project_plan.md](docs/project_plan.md) as the research source of truth. Supporting docs explain execution; do not silently rewrite the research design or treat an implementation proposal as an amendment.
- Update the relevant phase and architecture documentation in the same change as implementation. Describe the purpose, actual implementation and interfaces, commands, validation results, and limitations as they become known.
- Distinguish planned, implemented, and verified behavior. Never present unrun commands, unmeasured performance, or unpassed gates as established results.
- Maintain [docs/status.md](docs/status.md) with the current phase, completed work, evidence, unresolved issues, and next bounded milestone. Record design choices and their rationale in [docs/decisions.md](docs/decisions.md).
- Keep durable information in its owning document and link to it elsewhere. Add new documents to [docs/README.md](docs/README.md), and update this reading map when they provide an important new entry point.
- When experiments begin, write Markdown reports under `docs/experiments/` with configuration, exact commands, results, artifact locations, and interpretation. Machine-readable data, checkpoints, logs, and plots are artifacts, not Markdown documentation; record their locations in the report rather than embedding large dumps.
- Before finishing a documentation change, check local links and consistency with the current code and research plan.
