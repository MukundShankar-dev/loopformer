# Stages 4–6: execution generality and transfer

Status: planned future work; requires validated single-family recurrent dynamics. Source: project plan sections 14–22 and 25.

## Stage 4: shared multi-family execution

Purpose: test whether one recurrent parameter set can execute several trained transition families.

Extend the task interface with pointer lookup, random permutations, finite-state machines, symbolic conditional transitions, graph/grid navigation, and modular updates as appropriate to the chosen experiment. Every family must provide exact intermediate targets, a prompt containing its rules, and reproducible generation.

For families with input or action sequences, specify how the current instruction/progress is represented. Validate reference execution and tokenization before mixing tasks. Randomize symbols, rules, structures, start states, depths, and surface forms without changing the intended transition semantics.

Train one shared LoRA configuration and optional bridge across the configured family mixture. Do not add task-specific adapters. Preserve intermediate supervision and report results per family, alongside aggregate dynamics and compute use.

Gate: several trained families retain interpretable one-loop/one-transition execution with the same parameters. Investigate family-specific specialization and loss of pointer behavior before claiming a reusable mechanism.

## Stage 4b: optional ordinary-Qwen preservation

Purpose: measure and, if useful, limit changes to ordinary pretrained behavior.

Evaluate frozen original Qwen and adapted Qwen at one-pass behavior on ordinary text/prompts. Optionally train with `KL(p_base || p_adapted)` on these inputs. Document prompt selection, token masking, loss weighting, and additional compute.

Track preservation separately from synthetic recurrent performance and compare with/without the anchor. This loss is not part of the first pointer experiment or a requirement for the core dynamics result.

## Stage 5: whole-family holdout

Purpose: test zero-shot recurrent progression on an entire family excluded from recurrent training.

Choose the holdout and evaluation protocol before multi-family training. Keep the held-out family out of training and model-selection decisions. For example, train pointer, modular, symbolic, and FSM tasks, then evaluate graph navigation.

Measure each intermediate transition, final accuracy across loop counts, depth extrapolation, and overscaling behavior. Compare against relevant unadapted/single-family baselines and document shared encodings or structural similarities that limit the interpretation.

Gate: meaningful zero-shot progression on a wholly held-out family supports cross-algorithm transfer. Several trained families or unseen mappings alone do not establish that claim. Failed transfer remains an informative result.

## Stage 6: exploratory natural-language transfer

Purpose: determine whether recurrence survives changes in rule representation and eventually helps structured real tasks.

Progress from formal rules to natural-language versions of the same rules, then algorithm instructions, structured reasoning, and only later real benchmarks. Change one source of difficulty at a time and document target/decoding conventions when intermediate states become ambiguous.

Distinguish failures of execution from language understanding, arithmetic, and answer decoding. Natural-language or benchmark transfer is exploratory and is not required for the core project to succeed.

## Optional probes

After relevant trajectories exist, use entropy/margins, linear probes, or small MLPs to study current correctness, next-step help/harm, and task-family separability. Use separate probe training/evaluation data. Treat probe findings as diagnostics; do not force a proposed representation during the core training experiment.

## Implementation record

No multi-family generators, training mixture, anchor implementation, holdout results, or natural-language experiments exist yet. Split this guide into more detailed phase documents only when implementation warrants it, and update the documentation index and agent reading map.


## Future preservation regression evaluation

At the user's request, the [project plan](../project_plan.md#future-knowledge-retention-regression-check) now includes a small fixed ordinary-knowledge regression check before/after pointer adaptation, with a seeded MMLU subset as one candidate. It is a future evaluation, not an implemented benchmark or a requirement to introduce the anchor loss. Keep one-pass preservation distinct from cross-family and natural-language transfer. Current work remains Stage 1 depth generalization; these later stages are not activated by preparing that plan.
