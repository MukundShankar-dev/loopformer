# Project status

Last updated: 2026-09-17.

## Current priority

Stage 1: improve pointer depth generalization. The next authorized experiment is a fresh **loop-balanced cross-entropy** run, using the same 30,000 seed-37 mappings at depths 1–6, training seed 17, batch 4 with accumulation 2, learning rate 0.0002, and 3,750 updates as the completed baseline. See [exact commands](training_pointer.md#loop-balanced-loss-experiment). No loop counter, prompt change, longer training depth, or retention objective is introduced. Seed 29 remains reserved for confirmation; existing seed-17 evaluation sets are development diagnostics.

## Latest evidence

The fresh 30k baseline completed on CUDA in 60m54s, with 7.01 GiB peak CUDA tensor allocations. It started fresh adapters, not a resumed curriculum checkpoint. Full evaluation of steps 2500, 3250, and 3750 supports substantially stronger depth extension than the earlier curriculum. Step 2500 scores 98.0% complete trajectories at depths 1–6, 94.4% at both depths 7 and 8, 73.6% at 9, 41.6% at 10, 14.4% at 11, 1.6% at 12, and zero at 13–16. It is the working depth-generalization checkpoint among the three fully evaluated candidates; step 3250 is the trained-loss-selected reference. See the [fresh-run report](experiments/stage1_fresh30k.md).

The baseline's 34,816 training diagnostic rows and six full evaluations (72,000 loop rows) were audited against reference execution, metric aggregates, and source hashes. The new training dataset and model binaries are not on the Mac; full training-data hash and weight contents were not independently rechecked here. These are single-run development results, not a passed confirmatory gate.

## Implemented next experiment

`loss_reduction: loop_mean` uses selected-dataset loop counts to give every trained loop equal total direct weight. Microbatch losses are weighted estimates of that dataset objective; accumulation preserves the same effective-batch gradient. The default `example_mean` behavior and historical checkpoint identities remain compatible. Changing the objective requires a fresh run or explicit adapter-only initialization, not resume. Current instructions call for fresh adapters.

Training logs retain legacy example-mean CE and add optimized objective loss and dataset loop weights. New-run checkpoint selection averages per-loop CE over trained-depth validation examples only. Evaluation outputs and their existing loss semantics remain unchanged. The new pretrained experiment has not been run by the assistant.

Validation: **107 tests pass in 91.15 seconds** on the Mac, including exact dataset-objective and gradient comparisons across microbatch partitions, masked targets, OOD exclusion from selection, legacy resume compatibility, a loop-balanced CLI update/resume, and cleanup retention/symlink checks. Config comparison confirms only `loss_reduction` differs from the baseline. Local Markdown file links and whitespace checks pass. The seed-37 dataset is absent locally, so the real-data preview must run on the desktop.

## Artifact retention

At the user's request, superseded pre-30k recurrent runs/evaluations and redundant 30k checkpoint directories were pruned locally (about 29.6 MiB). Historical tracked evidence remains in Git history; historical report paths may refer to pruned artifacts. The current 30k logs and six full evaluations remain. Checkpoints 2250, 2500, 2750, 3000, 3250, and 3750 are retained for comparison. Ordinary-Qwen baseline results, Stage 0 evidence, all datasets, and new/unknown runs are preserved. [Desktop cleanup](training_pointer.md#artifact-cleanup) also removes ignored weights and optimizer files that Git cannot remove on pull; it has not been executed remotely.

## Constraints and deferred work

Frozen pretrained base, shared recurrent q/v LoRA, intermediate supervision, and `use_cache=False` remain unchanged. The updated startup gate passed on the desktop with zero reported T=1 logit error. Batch changes can preserve optimizer/cursor state via explicit same-effective-batch resume, but bitwise numerical equivalence is not claimed. See [architecture](architecture.md) and [training](training_pointer.md).

Overscaling execution, shortcut diagnostics, asymmetric training, multi-family transfer, and knowledge-retention benchmarks remain deferred. Gate 1 lacks predeclared thresholds and independent confirmation; Gate 2 lacks pretrained terminal-dynamics evidence. Supporting historical evidence lives in [the original CUDA report](experiments/stage1_cuda_5k.md) and the [documentation index](README.md).
