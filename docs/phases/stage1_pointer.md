# Stage 1: pointer data and stepwise execution

Status: planned; requires Stage 0 validation. Source: project plan sections 5, 7, and 32.

## Purpose

Teach one recurrent loop to execute one pointer transition using rules supplied in the prompt. Separate data validation from training so target errors cannot masquerade as model failures.

## Milestone 1: data only

1. Define task records with family, depth, initial state, exact intermediate states, final state, example ID, seed, and rendered prompt.
2. Validate a fixed symbolic vocabulary with the actual tokenizer, including prompt and answer contexts. Store the answer token IDs explicitly.
3. Generate a fresh random mapping and start state per example. Compute targets with an exact reference interpreter independent of model predictions.
4. Define prompt rendering and the answer readout position. Resolve completion, cycles, and repeated-state semantics in [decisions](../decisions.md) before finalizing labels.
5. Build reproducible instance and depth splits. Start with training depths at most eight and reserve deeper compositions for evaluation.
6. Test single-token symbols, valid mappings, exact trajectories, deterministic regeneration, and the depth/split boundaries.
7. Stop and report data validation before training.

Randomize mappings so the weights cannot solve examples by memorizing a global symbol-to-symbol lookup. Document split construction and any example-overlap checks.

## Milestone 2: training and evaluation

Implement the Stage 1 training loop using differentiable unrolls and intermediate supervision. For a trajectory `A -> F -> C -> Q`, target `F` after loop 1, `C` after loop 2, and `Q` after loop 3. Do not replace these with the final answer at every loop.

Specify masking for mixed-depth batches and loss reduction explicitly. Choose the cross-entropy vocabulary convention and record it; symbolic evaluation is restricted to the validated answer set regardless. Document any supervision beyond nominal completion separately.

Begin with tiny batches, short prompts, gradient accumulation, and recurrent depths around 4–8. Save reproducible configurations and checkpoints. Build the depth-by-loop evaluator alongside training using [evaluation conventions](../evaluation.md).

## Acceptance gate

- Meaningful intermediate accuracy on unseen random mappings.
- Deeper tasks generally require more recurrent computation.
- Additional loops can extend execution, including evaluation beyond trained depths.
- Final-answer accuracy does not conceal shallow shortcuts or broken intermediate execution.

Use the depth-by-loop heatmap and intermediate trajectories to assess the gate. Establish quantitative thresholds and evaluation sizes before judging success; none have been measured yet.

## Implementation record

No task generators, tokenizer checks, training code, checkpoints, or results exist yet. Add actual interfaces, commands, hyperparameters, validation, and observations here as each milestone is completed.
