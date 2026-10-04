# End-to-end research reassessment — 2026-10-04

This is an audit and proposal prompted by the user's request to reconsider the whole pipeline. This report records the pre-change review; the user subsequently authorized the [implementation checklist](../pipeline_upgrade.md). It supersedes treating paired-count consistency as the obvious next intervention.

## What the existing evidence establishes

The reference interpreter, intermediate labels, masking, weight sharing, frozen-parameter gradient scope, one-pass equivalence, checkpoint reload and matched prediction reproduction have substantial checks. Training unrolls its own continuous states with backpropagation through earlier loops; there is no teacher-forced symbolic state re-entry. Correctness at unseen mappings and limited extension beyond training depth are real. At step 5,000, full trajectories are 94.4% at depth 13, 75.2% at 14, then 16.8% at 15. Later training sharply degrades that extension. The matched-count probe demonstrates a causal effect of the displayed count on late output correctness on its cohort. It does not identify the internal mediator.

The observable is the output of R followed by six frozen C layers with access to fixed prompt memory. Calling every readout failure an R-state failure is too strong. Frozen C remains a substantial computation, not an inert classifier. A correct decoded letter also does not prove the working vector is a reusable representation of that pointer. Halting cannot cause early termination in forced evaluations, but its training gradient can shape R.

## Architecture and adaptation

Qwen layer-17 output is returned directly to layer 6, with no learned re-entry bridge or normalization of the whole returned residual stream. Internal Qwen RMSNorm does not bound that stream. Only the final answer-position vector evolves; it must support pointer execution and count/progress information for H. Fixed prompt activations are reused at each layer. This provides repeated access to the task and should not be described as complete absence of input access/injection.

Trainability is rank-8 q/v LoRA (270,336 parameters) plus the completion head (116,737). Keys, output projections, MLPs, prelude and coda remain frozen. Parameter count alone cannot establish adequacy: the allowed update directions may constrain adaptation of a feedforward representation into a repeatedly usable state. One-loop equivalence verifies surgery, not multi-loop stability or learnability. Rising state norms and increasing adjacent cosine are consistent with a problematic recurrent interface, but are not a causal diagnosis.

Useful missing controls: a matched fixed-prompt executor-only gradient condition; a modest re-entry/normalization or adapter-target ablation if that fails; direct diagnostic decoding of R states trained on early depths and tested on later depths; gradient decomposition for pointer versus completion loss; perturbation/JVP sensitivity as a diagnostic rather than a global contraction objective. Probe decodability alone is not causal evidence. No full fine-tuning or architecture redesign is selected here.

## Objective and optimization

Exact per-loop CE is an appropriate dense signal. RL is not a default remedy for missing algorithmic generalization. However, readout supervision allows many hidden representations for the same symbol and does not enforce closure under the next recurrent update.

H's BCE backpropagates into R. Its weight 0.1 does not bound its gradient contribution at 10%. A clean control is to train the same head on detached R features while retaining pointer gradients through the full recurrent unroll. This keeps the prompt and diagnostic head, while removing its influence on executor learning. It does not require an external counter.

The current equal-example loss gives each transition weight 1/d. Across the balanced nine counts, loop-1 direct CE mass is approximately 33.1 times loop-12 mass. This is not the total BPTT gradient ratio. A prior loop-balanced experiment failed on another setup; repeating it without a specific hypothesis is unwarranted. Log/decompose contributions before assuming equal-example low loss certifies the deep update.

The recipe has batch 4, depth-homogeneous update groups, learning rate 0.0002 after ten warmup steps, no decay and zero weight decay. This can produce temporally different gradient distributions and late regression. The fixed train probe is only four cases per trained count. The validation selection subset has 32 cases per count. Existing evidence cannot separate overfitting from optimization drift. More epochs at unchanged learning rate are not justified; lower-rate/decay continuation is a controlled optimization hypothesis, not a promised cure. Compare changed objectives at the same optimizer recipe rather than bundling changes.

## Data and evaluation

The sampler forces a non-repeating path of d+1 distinct symbols; d edges are planted and the rest of the 26-edge table is random. Changing d therefore changes graph distribution as well as requested number and recurrent age. No training/evaluation nominal path revisits a symbol, and the data contract limits depth to 25. This is useful for unambiguous trajectories but cannot establish indefinite execution or behavior on cycles.

A clean diagnostic suite should separate graph structure, requested count and recurrence length. Reuse fixed graph pools with matched shorter/longer queries and isolated development/confirmation maps. A controlled mixture of cycle structures would support longer execution with 26 symbols, but is a deliberate data-contract change. Repeated final symbols then make strict per-loop trajectories and exact stop timing especially necessary. Broader training support is legitimate if the evaluation frontier moves outward; it is not evidence of arbitrary extrapolation.

Other missing controls: a simple task-native learned recurrent executor as a positive learnability/control baseline, a second training seed before general claims, and same-example BF16/SDPA versus float32/eager checks to isolate export/precision effects. A symbolic interpreter validates labels but is not a learned positive control. Such controls do not replace the desired prompt-only Qwen architecture or insert oracle states into its inference.

## Compute

The fixed-memory implementation still recomputes full-prefix R/C layer operations each loop and discards/replaces prefix results. It projects every loop to the full approximately 152k vocabulary before selecting 26 symbols for loss. These are concrete redundant computations. Shared, differentiable prefix K/V reuse and selected-row LM projection could preserve the objective, but must pass forward and gradient equivalence. Frozen parameters still require activation gradients, so parameter memory alone does not size training. Per-update memory peaks are newly implemented; actual utilization/speed requires desktop measurements. Global contraction or state freezing would not be an appropriate optimization target for a moving pointer.

## Recommended order

1. Freeze the current reference and evaluation definitions. Define a clean graph-controlled development panel and a small same-weight precision check. Do not tune on confirmation.
2. Prioritize a matched test of completion-gradient coupling in the current fixed-prompt architecture, with a common reasonable learning-rate schedule and equal example/transition budgets. Treat the optimizer change separately from the objective comparison. Preserve all current per-loop targets and prompt-only inference.
3. If executor-only training still has the same failure, investigate the recurrent interface/adaptation capacity before adding a count-consistency penalty. A small task-native learned control can distinguish a pipeline/training issue from the cost of adapting pretrained Qwen.
4. Validate an exact compute optimization before spending more on large comparisons. Repeat a promising result with another seed, then revisit stopping and the broader research gates.

This is a decision framework, not a demand to run every listed technique. Paired-count augmentation is a reasonable candidate only after resolving these more basic controls; it may suppress an observed symptom without fixing recurrent-state reuse.

## Literature grounding

- [Fan et al., Looped Transformers for Length Generalization](https://arxiv.org/html/2409.15647v3): task-specific small models, input injection and step-dependent final supervision. Their positive results do not validate our frozen-Qwen, one-position, per-loop pointer implementation.
- [Geiping et al., recurrent-depth latent reasoning](https://arxiv.org/html/2502.05171v1): explicit input/state mixing, recurrent normalization and sampled recurrence. These differences motivate testing our re-entry assumptions, not copying their much larger pretraining recipe.
- [McLeish et al., Retrofitted Recurrence](https://arxiv.org/html/2511.07384v1): adapters, recurrence curricula and adaptation/healing after pretrained surgery. Their scale and supervision differ substantially. We should not infer that LoRA surgery alone inherits their generalization.
