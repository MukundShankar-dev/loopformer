# Project status

Last updated: 2026-09-10.

## Current priority

**Continue Stage 1 training to improve depth generalization before running overscaling experiments.** The user requested that overscaling scripts be prepared and pushed now, with execution deferred. A future knowledge-retention regression evaluation is planning only; no MMLU evaluator or anchor loss is implemented.

We have evidence of generalization to unseen pointer mappings within trained depths, not general depth extrapolation. The first 5,000-mapping run trained only depths 1–4. Full-test complete-trajectory accuracy is **86.2%** for validation-selected update 500 and **92.4%** for secondary update 625 at those depths. At depth 5 it is 8.8% versus 40.0%; at depth 6, 0% versus 3.2%; neither has a fully correct depth-7/8 trajectory. See the [audited run report](experiments/stage1_cuda_5k.md).

The [continuation config and commands](training_pointer.md#continue-the-current-desktop-run) extend the existing desktop run from one to three total epochs, preserving training depths, data, optimizer settings, and intermediate supervision. No pretrained continuation has been launched by the assistant. This is a bounded pilot, not a declared Gate 1 pass or a guarantee that more training fixes depth extension.

## Implemented and verified

| Area | Current evidence |
| --- | --- |
| Stage 0 architecture | Pretrained Qwen CPU/MPS T=1 equivalence, shared recurrence, and gradient-scope checks pass; [report](experiments/stage0_validation.md) |
| Pointer dataset | 13,000 seed-17 examples, exact reference targets, symbol-token checks, disjoint rule tables, and byte-for-byte replay; [report](experiments/stage1_data_validation.md) |
| Ordinary instruct baseline | Mac and WSL full tests both score 60/1,000 (6%); different runtimes, matching desktop dataset; [Mac](experiments/naive_pointer_baseline.md) and [WSL](experiments/wsl_cuda_baseline.md) reports |
| Stage 1 training | 32-example overfit and 5,000-mapping CUDA runs complete; shared rank-8 q/v LoRA, frozen base, nominal per-loop CE; [overfit](experiments/stage1_cuda_overfit.md) and [larger run](experiments/stage1_cuda_5k.md) |
| Saved-checkpoint evaluation | Full final-answer and all-loop tests audited for updates 500 and 625; each full-loop run has 1,000 examples × eight readouts; [usage](loop_pointer_eval.md) |
| Overscaling preparation | Separate deterministic terminal transform, shared checkpoint sweep, conditional repair/damage, margins, censored survival, and readout-halting comparison implemented; [usage](overscaling_eval.md). Pretrained execution deferred |
| Stage 3 onward | Asymmetric training, multi-family training, cross-family transfer, and natural-language transfer not implemented or established |
| Knowledge retention | Future small fixed benchmark comparison described in the [plan](project_plan.md#future-knowledge-retention-regression-check); no implementation |

The 5,000-mapping run took 868.52 seconds and peaked at 2.77 GiB of CUDA tensor allocations (not total GPU memory). Full nominal final accuracy is 47.0% at update 500 and 53.7% at 625. Full-loop evaluation took 152.55 and 160.51 seconds respectively. All nominal predictions agree with their corresponding previous final-answer evaluation. See the owning report for exact commands, CSV paths, loss diagnostics, and audit limits.

The initial recurrent model remains in `scripts/recurrent_qwen/`: frozen prelude layers 0–5, shared recurrent layers 6–17 with 270,336 trainable adapter parameters, frozen coda layers 18–23, no bridge, and no KV cache. Training and inference use raw pointer prompts and restricted A–Z readouts. The ordinary baseline uses three-shot chat prompting and unconstrained generation; the two evaluation conditions must not be conflated.

## Validation and environment

The current Mac suite passes **92 tests in 54.83 seconds**. The overscaling-focused tests passed (5 cases in 19.82 seconds), including offline tiny-Qwen checkpoint integration and a no-write terminal preview. The continuation data/tokenizer-only preview passed: 5,000 examples, 64 validation items, 16 training probes, and 1,875 total planned updates, with no weights loaded or files written. CLI help, local Markdown links, and whitespace checks also pass. These are implementation checks, not pretrained overscaling or continuation evidence.

The WSL desktop uses an RTX 5070 Ti with 16 GB VRAM, 32 GB host RAM, Python 3.11.8, and CUDA Torch. The pinned model is cached, deterministic CUDA operations and trainer startup checks passed, and regenerated data matches the Mac hashes. A toy exact-resume test has a small reproducible precision discrepancy on WSL; **bitwise CUDA resume equivalence remains unverified**. The uninterrupted training and successful checkpoint inference do not resolve that issue. See [desktop evidence](experiments/wsl_cuda_baseline.md) and [setup](windows_cuda_setup.md).

## Next bounded milestone

1. Preview the three-total-epoch config, then let the user launch continuation from update 625 on the desktop. Keep the original validation-selected update 500 as the historical primary.
2. Review trained-depth validation trajectories and untrained depths 5–8 separately; use full validation evaluation beyond the small monitoring subset. Record failure if more updates do not improve depth extension.
3. Set measurable gate criteria and a confirmation seed before a future confirmatory evaluation. The already inspected seed-17 test set cannot serve as untouched confirmation for changes informed by it.
4. After that review, decide whether to run the prepared terminal overscaling pilot. First verify nominal execution on the changed terminal distribution and check early-final shortcuts. Observe useful repair and damage before considering asymmetric retention.

Gate 1 is not formally declared passed: thresholds were not set in advance, evidence is from one training seed, and extension remains weak. Gate 2 has no pretrained terminal evidence. D03's original task semantics remain unchanged; the terminal diagnostic has its own explicit convention. General cyclic-task loss semantics, long-unroll resource use, mixed precision, confidence/stability halting, hidden-state no-op ablations, plots, and future preservation benchmarks remain unverified or unimplemented. See [decisions](decisions.md).
