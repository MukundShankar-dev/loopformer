# Design decisions and unresolved questions

This document records choices that affect implementation or interpretation. The [project plan](project_plan.md) supplies the established research constraints. Open entries below are questions to settle before their dependent work, not approved changes to the experiment.

## Established constraints

- Study single-family recurrent dynamics before generality.
- Start with the specified pretrained Qwen, shared recurrence, frozen base weights, recurrent LoRA, and an optional bridge.
- Gate training on one-loop equivalence, weight sharing, and gradient scope.
- Preserve intermediate supervision and measure damage together with repair.
- Keep documentation in Markdown under `docs/`, with `AGENTS.md` as the root agent guide.

## Initial dependency baseline

Use Python 3.11 with a local `venv` and pinned direct dependencies in [requirements.txt](../requirements.txt). Stage 0 adds PEFT, pytest, and Rich to the original PyTorch/Transformers/Hub baseline. Direct pins are not a transitive lock; validation JSON records every installed distribution. See [setup validation](setup.md#version-and-validation-status).

## Stage 0 resolutions — 2026-09-10

- **D01, resolved for the pinned environment.** Preserve Transformers 5.17.0 decoder arguments, full causal attention, default RoPE, and ordinary-forward positions, with caching disabled. The CLI uses float32, eager attention, a fixed model revision, and `atol=rtol=1e-5` selected before pretrained validation. CPU and MPS pass. See [architecture](architecture.md) and the [report](experiments/stage0_validation.md).
- **D02, resolved for Stage 0; revisit only with evidence.** No bridge. R returns the same shape it accepts; direct shared recurrence preserves one-loop behavior and gradient flow. This does not prove that untrained recurrent states support pointer execution. A future bridge would change the transition and needs evidence and renewed validation.
- **Initial LoRA.** Adapt q/v projections in layers 6–17, rank 8, alpha 16, zero dropout, no bias training. This is a small starting configuration, not an optimized rank or target choice. With PyTorch linear-weight orientation, the added weight is `(alpha/r) B A`, where A is `[r,in]` and B is `[out,r]`. B maps back out of the rank space; it is not a bridge or the coda. Random A and zero B preserve the initial forward. A can have zero gradient until B changes. Injection follows the [PEFT low-level API](https://huggingface.co/docs/peft/en/developer_guides/low_level_api).
- **Readout.** Default to last unmasked input position and answer-position logits. Full logits remain available for equivalence and full hidden states are optional. Labels explicitly contain one token ID per loop. This settles an API default, not D05's task-specific prompt/symbol design.
- **MPS determinism.** Keep strict deterministic algorithms enabled. Advanced-index and gather answer readouts failed backward on unsupported deterministic MPS scatter operations. Equivalent integer slices and stacking pass without changing numerical tolerances or research semantics. This incurs one answer-index device synchronization per forward, acceptable for tiny batches. MPS margin-loss backward is not yet validated.
- **Output and layout.** Rich renders terminal configuration and checks; JSON saves exact inputs, revision, packages, source hashes, parameter/gradient details, and measurements. Downloads are opt-in in the Stage 0 CLI. At the user's explicit request, implementation lives in `scripts/recurrent_qwen/` and the composing CLI in `scripts/validate_stage0.py`; tests stay in `tests/`. The project plan's layout illustration was updated to match, without changing the research design.

## Stage 1 data resolutions — 2026-09-10

- **D03, resolved for nominal Stage 1 data only.** Follow the user's explicit-step format: a random rule table, `Start`, and `Steps: d`, with targets exactly at steps 1 through d. The final target is the d-th lookup; the mapping can continue afterward. There are no post-completion training labels or absorbing-state assumptions. Consequence: valid continued execution cannot be called overscaling damage without defining separate completion behavior for Stage 2/3. Absorbing terminals would be a different task distribution and are not imposed by this dataset.
- **D04, resolved for the initial sampler; later loss semantics remain open.** Condition the nominal path on d+1 distinct states. This prevents early/repeated visits to the final state and keeps first arrival at the final symbol at depth d. Fill the other edges randomly, allowing cycles outside that path. This restricts the distribution and limits unique-path depth to 25 with 26 symbols. It does not settle Stage 3 branch precedence for future cyclic tasks.
- **D05, resolved for this prompt version.** Fixed A–Z logical symbols, encoded as space-prefixed single tokens. Render pairs as `( A, C)`, use raw text without a chat template/special tokens, and read the next token at the final `Answer:` colon. Compact `(A,C)` was inspected and can merge symbols with punctuation; the chosen format passes exhaustive symbol-context checks and all 13,000 generated prompt checks. `h_0` has no start-symbol readout guarantee. See [Stage 1 interfaces](phases/stage1_pointer.md#prompt-tokens-and-record-schema).
- **Initial size and splits.** Seed 17; 10,000 training examples and 1,000 each for validation, same-depth test, and deeper test. Balance depths 1–8 in the first three splits and 9–16 in the last; shuffle rule order and use 26 rules at every depth. These are configurable starting budgets, not power calculations or evidence of adequate training. Exclude exact whole-table overlap across all records; shared edges/subpaths are allowed. Reproduction uses per-example derived seeds plus recorded source/runtime metadata and file checksums.
- **Dataset packaging and preview.** At the user's request, all current dataset implementation and CLI code lives in `scripts/dataset/`, invoked with `python -m scripts.dataset`; tests remain in `tests/`. This changes the earlier layout proposal, not the sampling or research design. The original prefix preview disproportionately showed depth 1 because each split starts at its lowest depth. Preview selection now spans available depths and prints the complete configured distribution. Full-dataset sampling and serialization are unchanged: all four seed-17 JSONL files were reproduced byte for byte in memory after relocation. Preview first; writing a dataset remains a separate explicit CLI invocation. The original manifest is historical provenance and is not rewritten when source files move.

## Ordinary-model baseline — 2026-09-10

- At the user's request, add an ordinary-model final-answer evaluation before training. This is an evaluation-only baseline, not a change to intermediate supervision or the stage gates. The user will launch pretrained evaluation; implementation checks use local random toy models only.
- Keep reusable task instructions in `prompts/pointer_task.txt`; only rules, start, and requested depth are interpolated. Default to the model's chat template for instruct checkpoints, with explicit raw mode for other models. Do not change the existing dataset prompts or future recurrent-training targets.
- Use unconstrained greedy generation and strict single-uppercase-symbol matching after whitespace stripping. Record invalid-format answers and budget stops instead of hiding them through answer extraction. This differs from the later restricted-symbol-logit recurrent readout and must be labeled in comparisons.
- Keep `use_cache=False` under the existing project constraint. Generation length is not recurrent depth. Rich reports progress/throughput; CSV and JSON under `eval/pointer_task/` retain detailed evidence. See the [baseline guide](naive_pointer_eval.md).
- **Prompt revision after user inspection.** Following the user's report of three incorrect answers, add three independent hand-written demonstrations at depths 1, 2, and 3. Keep answers as single symbols and do not include evaluated-example labels. This changes the ordinary baseline from zero-shot to three-shot prompting; saved prompt hashes distinguish the runs. The [full three-shot run](experiments/naive_pointer_baseline.md) reached 6.00%; demonstration benefit remains unestablished without a paired full zero-shot run.

## Stage 1 training resolutions — 2026-09-10

- **D06, resolved for the first pointer experiment.** Cross-entropy uses only the 26 validated symbol logits. At loop t, supervise the exact t-th transition for t <= d. Mean valid loops within each example, then mean examples; gradient accumulation preserves this weighting, including short tail updates. Later losses backpropagate through earlier states; no detached states, final-answer retention, or ordinary-Qwen anchor loss. Consequence: CE and symbol accuracy measure discrimination within A–Z, not unrestricted language generation, and longer examples do not receive greater total weight solely from their depth.
- **Initial budgets.** Start with 32 balanced depth-1–4 examples over 20 epochs (160 updates) to inspect fitting and resource use. The larger config uses 5,000 depth-1–4 examples for one epoch (625 updates). Batch size 1, gradient accumulation, float32, eager attention, strict determinism, and the existing bridge-free q/v LoRA remain explicit. These are starting budgets, not evidence that one epoch or these hyperparameters suffice.
- **Monitoring and selection.** Fixed seeded training probes and held-out validation subsets; validation sweeps through the configured maximum depth. Select checkpoints using mean validation loss on trained depths only. Monitored deeper validation depths are not untouched test data. Post-completion final readouts are observations, not a settled damage definition. D09 empirical success criteria remain open.
- **Checkpoint contract.** At the user's request, JSON configs live in `configs/`, the training entry point in `scripts/training/`, and actual checkpoints/logs in `models/`. Save recurrent adapters, the tokenizer, explicit base revision and architecture, and optimizer/RNG/cursor state. The existing naive-test CLI recognizes these directories and reconstructs recurrent inference; it labels raw-prompt, restricted-symbol readout separately from the ordinary three-shot generation baseline. See [training usage](training_pointer.md).

## Remaining implementation and research choices

| ID | Resolve before | Question and consequence |
| --- | --- | --- |
| D03 | Stage 2/3 post-completion evaluation/training | Nominal Stage 1 labels are settled above. Define required behavior beyond requested depth before interpreting continued pointer moves as damage. |
| D04 | Stage 3 loss or a future cyclic dataset | Initial paths do not repeat. Future cyclic tasks still need early/repeated-final correctness semantics and loss branch precedence. |
| D06 | Later training stages | Stage 1 is settled above; specify vocabulary, masking, and weighting again when later objectives change. |
| D07 | Stage 3 comparisons | What precisely is variant C's ordinary task loss, and how are budgets compared across all four variants? Ambiguity makes the retention ablation uninterpretable. |
| D08 | Stage 4 training | Which family is held out, how is input progress represented, and which data may inform model selection? Decide before training to preserve the zero-shot claim. |
| D09 | Each empirical gate | What sample sizes, seeds, thresholds, and uncertainty reporting establish meaningful execution, useful repair, and substantial damage reduction? Select criteria before judging the experiment. |

## Recording a resolution

For each resolved entry, record the date, status, chosen behavior, rationale, alternatives that affect interpretation, supporting inspection or experiment evidence, and affected code/docs. Update the owning phase guide and current status. If a technical constraint changes the research design, explain its scientific consequence before implementing it.

Routine implementation details can be resolved within the authorized scope. Do not create a new approval requirement merely because a choice appears here. Preserve genuinely unresolved research choices explicitly.
