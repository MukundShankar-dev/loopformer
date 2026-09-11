# Project status

Last updated: 2026-09-11.

## Latest experiment update

The depth-6 stage and checkpoint evaluations are complete. Full-validation complete-trajectory accuracy at trained depths 1–6 is 99.1% for step 500, 97.9% for 800, and 97.3% for final 938. At depths 7–8 combined these score 79.2%, 78.8%, and 64.8%. Step 500 is the provisional working checkpoint; its complete trajectories fall to 33.6% at depth 9, 5.6% at depth 10, and zero at 11–16. Six runs under `eval/pointer_loops/*depth6-seed17*` contain 72,000 audited loop rows; reference targets, scoring, aggregates, and dataset/source hashes agree.

The next authorized run is [fresh depth-6 training on 30,000 mappings](training_pointer.md#fresh-depth-6-run-with-30000-mappings), using new adapters on pretrained Qwen rather than initialization from a pointer checkpoint. The config is implemented and schema-validated; the user reports dataset generation on the desktop. The dataset is absent on the Mac, so the full dry-run and training remain unverified here. Training stays at depths 1–6, with seed-17 development evaluation through depth 16 and seed 29 reserved. Overscaling and shortcut diagnostics remain deferred. Earlier setup and validation evidence below describe the preceding stages.

## Current priority

**Continue Stage 1 training to improve depth generalization before running overscaling experiments.** The user requested that overscaling scripts be prepared and pushed now, with execution deferred. A future knowledge-retention regression evaluation is planning only; no MMLU evaluator or anchor loss is implemented.

The depth-4 continuation has completed three total epochs / update 1875. Its full validation run has **99.4% complete trajectories at depths 1–4**, 82.4% at depth 5, 20.8% at depth 6, 1.6% at depth 7, and 0% at depth 8. The [audited report](experiments/stage1_cuda_5k.md) records the continuation, small monitoring subset, and full validation separately from the earlier test runs.

**Next: [train through depth 6 and move OOD evaluation outward through depth 16](depth_generalization.md).** The new config uses one epoch over 7,500 depth-1–6 mappings (938 new updates), initialized from update 1875's adapters with fresh optimizer/RNG/counters. Both the depth-4 reference and new checkpoint will be evaluated on identical development datasets, reporting absolute depth and steps beyond the training maximum. This is additional training, not an equal-compute depth ablation. Seed 29 is reserved for confirmation after choices are frozen; it has not been generated or evaluated. No pretrained depth-6 training or paired sweep has been launched by the assistant.

Dataset coverage rechecked: seed-17 training/validation/test cover depths 1–8, and the existing `depth_test.jsonl` covers 9–16 with 125 examples per depth. All four file hashes match their manifest. No regeneration is required. Training-time validation remains at 1–8; the separate paired evaluator reads both evaluation files through depth 16. See [data coverage and verification](depth_generalization.md#existing-data-no-regeneration-required).

## Implemented and verified

| Area | Current evidence |
| --- | --- |
| Stage 0 architecture | Pretrained Qwen CPU/MPS T=1 equivalence, shared recurrence, and gradient-scope checks pass; [report](experiments/stage0_validation.md) |
| Pointer dataset | 13,000 seed-17 examples, exact reference targets, symbol-token checks, disjoint rule tables, and byte-for-byte replay; [report](experiments/stage1_data_validation.md) |
| Ordinary instruct baseline | Mac and WSL full tests both score 60/1,000 (6%); different runtimes, matching desktop dataset; [Mac](experiments/naive_pointer_baseline.md) and [WSL](experiments/wsl_cuda_baseline.md) reports |
| Stage 1 training | 32-example overfit and 5,000-mapping CUDA runs complete; shared rank-8 q/v LoRA, frozen base, nominal per-loop CE; [overfit](experiments/stage1_cuda_overfit.md) and [larger run](experiments/stage1_cuda_5k.md) |
| Saved-checkpoint evaluation | Full tests audited for updates 500/625, plus full validation for update 1875; each full-loop run has 1,000 examples × eight readouts; [usage](loop_pointer_eval.md) |
| Overscaling preparation | Separate deterministic terminal transform, shared checkpoint sweep, conditional repair/damage, margins, censored survival, and readout-halting comparison implemented; [usage](overscaling_eval.md). Pretrained execution deferred |
| Stage 3 onward | Asymmetric training, multi-family training, cross-family transfer, and natural-language transfer not implemented or established |
| Knowledge retention | Future small fixed benchmark comparison described in the [plan](project_plan.md#future-knowledge-retention-regression-check); no implementation |

The 5,000-mapping run took 868.52 seconds and peaked at 2.77 GiB of CUDA tensor allocations (not total GPU memory). Full nominal final accuracy is 47.0% at update 500 and 53.7% at 625. Full-loop evaluation took 152.55 and 160.51 seconds respectively. All nominal predictions agree with their corresponding previous final-answer evaluation. See the owning report for exact commands, CSV paths, loss diagnostics, and audit limits.

The initial recurrent model remains in `scripts/recurrent_qwen/`: frozen prelude layers 0–5, shared recurrent layers 6–17 with 270,336 trainable adapter parameters, frozen coda layers 18–23, no bridge, and no KV cache. Training and inference use raw pointer prompts and restricted A–Z readouts. The ordinary baseline uses three-shot chat prompting and unconstrained generation; the two evaluation conditions must not be conflated.

## Validation and environment

The current Mac suite passes **94 tests in 64.94 seconds**. The nine focused initialization/training/comparison cases passed in 27.96 seconds. The real-data depth-6 preview passed with 7,500 training examples, 64 validation examples, 24 probes, and 938 planned updates, including source metadata validation; no weights loaded or files written. CLI help, the plan-only paired-evaluation test, local Markdown links, and whitespace checks pass. These are implementation checks, not pretrained depth-6 or OOD evaluation evidence. The preceding overscaling implementation had 92 passing tests; its pretrained runs remain deferred.


The WSL desktop uses an RTX 5070 Ti with 16 GB VRAM, 32 GB host RAM, Python 3.11.8, and CUDA Torch. The pinned model is cached, deterministic CUDA operations and trainer startup checks passed, and regenerated data matches the Mac hashes. A toy exact-resume test has a small reproducible precision discrepancy on WSL; **bitwise CUDA resume equivalence remains unverified**. The uninterrupted training and successful checkpoint inference do not resolve that issue. See [desktop evidence](experiments/wsl_cuda_baseline.md) and [setup](windows_cuda_setup.md).

## Previous depth-stage milestone (completed; superseded above)

1. Preview [depth-6 adapter-only initialization](depth_generalization.md), then let the user launch its bounded one-epoch stage on the desktop. The three-epoch depth-4 continuation is already complete.
2. Use the new stage's validation-selected checkpoint and reference update 1875 in the paired evaluator on depths 1–8 and 9–16. Keep in-range, +1/+2, and farther OOD results separate; do not select checkpoints by test scores.
3. Review whether the failure boundary merely moves or extrapolation improves. Fix quantitative claims, checkpoints, and metrics before generating reserved seed-29 confirmation data.
4. Only after that review decide whether to run terminal overscaling. Knowledge retention, asymmetric objectives, and multi-family work remain deferred.

Gate 1 is not formally declared passed: thresholds were not set in advance, evidence is from one training seed, and extension remains weak. Gate 2 has no pretrained terminal evidence. D03's original task semantics remain unchanged; the terminal diagnostic has its own explicit convention. General cyclic-task loss semantics, long-unroll resource use, mixed precision, confidence/stability halting, hidden-state no-op ablations, plots, and future preservation benchmarks remain unverified or unimplemented. See [decisions](decisions.md).
