# Learned completion of an explicit-step pointer task

**Status: both the historical joint run and fixed-prompt training/evaluation have completed on pretrained Qwen.** See the [fixed-prompt failure investigation](experiments/stage1_fixed_prompt_review.md): forced execution improves near the training horizon, but learned stopping still fails beyond trained depths. The [experiment report](experiments/stage1_learned_completion.md) records a negative result: at the 0.5 diagnostic threshold, the head stops early on every depth-9–16 example, and the jointly trained checkpoint has worse forced-depth pointer trajectories than the matched CE-only checkpoint. This is a Stage 1 ablation of the existing [pointer training](training_pointer.md), not the terminal-damage controller in [adaptive inference](adaptive_compute.md).

## Task and model contract

The **only task input** is the existing raw dataset prompt:

```text
Rules: ( A, C) ( C, D) ( D, E) ...
Start: A
Steps: 2
Answer:
```

Tokenization, attention mask, and the answer position are mechanical representations of that prompt. The rule table, start symbol, and requested depth appear in text; neither the recurrent block nor the stopping head receives a separately parsed depth, loop index, remaining-step count, reference state, or previous decoded answer. No new prompt fields, special progress tokens, teacher-forced state insertions, or per-loop time embeddings are added.

Keep the current frozen prelude P, shared recurrent block R with trainable LoRA, and frozen coda C. For each pass, `h_t = R(h_{t-1})`. The ordinary symbol readout is `C(h_t)` at the existing answer position, restricted to the validated A–Z token set for training and evaluation. Add a small trainable halting head `H` that reads **only the answer-position vector from `h_t`** and emits one stop logit. The implemented head uses LayerNorm, a hidden linear layer, GELU, and a scalar linear output; its width and weights are saved with the adapter. The head's inputs contain no explicit `t`, `d`, confidence-history feature, or output from the reference interpreter. Its gradient flows into R so the auxiliary objective can change the recurrent representation. C's readout and H's stop decision do not feed back into R.

**Implemented self-stopped inference:** supply the prompt and a fixed safety cap `B` that is independent of its requested depth. Repeat R until the stop probability crosses a threshold or the cap is reached. Decode the **single final A–Z symbol at the stopped pass**. Diagnostic exports should also retain stop probability, optional per-pass symbol predictions, executed loops, threshold, and whether the cap forced a stop. The outer loop still has to enforce a finite cap; it must not parse `Steps` to choose the stop pass. The existing evaluator enables this with `--stop-policy completion --loops 20 --stop-threshold 0.5 --batch-size 1`. This policy uses the checkpoint head and can stop before the requested depth; it never receives a parsed depth. Omitting the policy retains full-loop exports and forced-depth scoring. The explicit loop cap is required and is never derived from task depth for this policy. Actual stopped inference for fixed-prompt recurrence is recorded in the linked review; no matched-compute speedup claim is established.

## Supervision

For a training record, the dataset interpreter knows the path `s_0 = Start`, `s_t = mapping[s_{t-1}]`, and requested depth `d`. The model receives none of those computed states. Training unrolls through `d` even if its provisional head would stop early, so every nominal target receives a gradient. For a mixed-depth batch, unroll to the batch maximum and mask every example's losses after its own `d`.

| Pass | Symbol target for C | Stop target for H |
| --- | --- | --- |
| `1 <= t < d` | `s_t`, the result of exactly `t` lookups | Continue (`0`) |
| `t = d` | `s_d`, the final task answer | Stop (`1`) |
| `t > d` in a padded batch | No target | No target |

Thus the existing masked 26-symbol cross-entropy continues to teach **one lookup per pass**, including every intermediate result. The new binary loss teaches **when the requested number of lookups has been completed**; it does not replace intermediate supervision. A wrong symbol at pass `d` is still penalized even if the head correctly says stop, and a correct symbol reached at an earlier pass does not receive an early-stop label.

For each example, use the current mean of valid per-pass symbol CE as `L_pointer`. Define `L_halt` as the mean continue BCE over passes `1..d-1` and the stop BCE at pass `d`, with those two components equally weighted when `d > 1`; for `d = 1`, use the stop BCE alone. This avoids the single stop label vanishing among many continue labels. Optimize `mean_examples(L_pointer + lambda_halt * L_halt)`. Keep the current recurrent adapters trainable, the pretrained Qwen weights frozen, and the coda differentiable but frozen. The supplied first-run config fixes `lambda_halt=0.1` before any pretrained run. To keep checkpoint selection comparable to the CE-only baseline, the primary best-checkpoint criterion remains trained-depth validation pointer CE. Logs also record trained-depth halt loss and their joint objective. Per-pass symbol metrics and stop classification use probability 0.5 as a diagnostic threshold. The trainer uses `t` and `d` only to construct and mask labels. That bookkeeping is not an input feature to R or H.

This is intentionally **joint** training. Fitting H on a frozen recurrent checkpoint would test whether pass progress is already decodable from its states, but cannot teach R a better process. That cheap head-only probe is a useful control before spending on a fresh joint run. The existing five-feature [adaptive head](../scripts/eval/adaptive_policy.py), which is given normalized loop and depth and cannot stop before requested depth, is not a valid implementation of this experiment.

## Controlled comparison and failure interpretation

Reuse the fresh depth-6 run's data selection, prompt format, base revision, recurrent split, optimizer budget, and intermediate pointer loss as the control. Train the joint variant from the same initial adapter state, not from a selected trained checkpoint. Only H and its auxiliary loss differ. Keep mapping-disjoint development and untouched confirmation splits; fit any stop threshold on development and freeze it before confirmation. Choose a safety cap above the deepest evaluated depth (for example, 20 for depths through 16), and report cap fallbacks rather than counting them as learned stops.

Evaluate two modes on **the same joint checkpoint**:

1. **Forced depth:** read C after exactly `d` passes. Compare per-pass accuracy, correct-prefix length, first-error position, complete trajectory, and final-answer accuracy with the CE-only control. Any change here concerns learned recurrent dynamics, not the inference stopping rule.
2. **Self-stop:** let H choose the pass under the depth-independent cap. Record exact-stop rate `P(stop=d)`, early and late stop rates, cap-fallback rate, mean loops, final-answer accuracy, and joint success `P(stop=d and answer=s_d)`, all by requested depth. A right final letter at the wrong pass is not evidence that it learned the requested process.

Use held-out rule tables at trained depths and depth-extended tasks separately. On selected rule tables and starts, vary only the prompt's `Steps` value across valid depths: stopping should track that text, while the first `min(d_1,d_2)` pointer targets remain checkable. Also retain rule-order and symbol-renaming controls. Compare both models at equal forced depth before attributing any gain to the halt head. If self-stop improves but forced-depth trajectories do not, the benefit is stopping; if forced-depth trajectories improve, the auxiliary gradient may have changed execution. If neither improves, lack of a learned stopping decision was not the demonstrated bottleneck. A head that merely predicts the common training depth fails exact-stop and held-out-depth tests.

## Boundaries and implementation requirements

The prompt's `Steps` field remains necessary: without it, the same rules and start have multiple valid final answers. The runtime cap remains necessary to prevent unbounded execution. Neither is an explicit per-pass clock supplied to R or H. Conversely, removing only the caller's `num_loops=d` stop with **unchanged weights** cannot alter logits from earlier passes: the integrated diagnostic observed identical first-six predictions in all 3,456 matched short/full unroll comparisons. Improvement in forced-depth execution is therefore a hypothesis about the *additional joint training signal*, not a direct consequence of changing inference control.

The checkpoint format is versioned: `loopformer-stage1-completion-v1` stores H and the recurrent adapters; historical `loopformer-stage1-v1` files still load. The existing `stop_policy` still calls a depth-aware policy and is **not** used for completion training. The implementation extends the shared model/trainer/validation/evaluator interfaces; no parallel pointer dataset was created. Focused tests cover head-only hidden-state input, exact 1-based stop targets, masking after `d`, H and LoRA gradient flow with frozen Qwen weights, microbatch weighting, checkpoint round-trip, resume, a tiny CLI update, and compatibility with full-loop evaluation and stop-probability export. Early-stop execution and cap fallback are implemented for the checkpoint completion head. Threshold selection remains a development-set decision; the supplied 0.5 is diagnostic.

The original pointer task is not absorbing after `d`. This experiment tests **completion of the prompt's requested count**, not discovery of an intrinsically terminal state or adaptive allocation of useful compute. Because `d` is already given in the prompt, a learned head cannot claim a compute-allocation advantage over an external stop-at-`d` baseline. The scientific question is whether internal progress supervision improves the recurrent computation and whether that learned count extends beyond training depth.

## Fixed prompt memory and recurrent working state

**2026-10-04: implemented and evaluated on the desktop; see the linked failure investigation.** The user approved the architectural proposal that makes R a reusable transition while the model learns completion from prompt-only input. This changes where recurrence can write. It does not add an external clock, reference state, decoded-symbol feedback, or per-loop prompt.

Currently R evolves every prompt position, including the representations of Rules, Start, and Steps. The fixed-prompt mode keeps those input representations available as read-only memory within a forward pass, and carries only a working state between loops. The hypothesis is that maintaining original task information separately from evolving computation makes repeated transitions easier to learn. This is not a finding that rule memory has been proven to be the cause of the current failure.

### Implemented first version

Use the existing final unmasked prompt position (the answer readout position) as one recurrent working vector `w`, initially its P output. All preceding tokens form the read-only prompt memory. Keep Qwen's existing attention layers, original token positions, frozen base weights, shared R q/v LoRA, frozen C, and hidden-state completion MLP. No new scratch tokens or parsed task fields are required.

For each R layer, prepare the prompt-prefix hidden activations that an ordinary single pass would present to that layer. Causal attention makes those prefix activations independent of the later working position. On every recurrent pass, that layer receives its own fixed prefix activations and the current working vector; only its working-position output advances to the next layer and eventually to the next pass. Prefix activations are distinct at different layers but identical across recurrent passes. C likewise reads the current working vector using its layer-appropriate ordinary-pass prefix context. C's output is a side readout and never the next R input.

The computational interface is `w_next = R(w, memory)`, `symbol = C(w_next, memory)`, and `stop_probability = H(w_next)`. Neither R nor H accepts a numeric loop index or parsed requested depth. The original Steps text remains in the prompt memory. Progress must be learned in the evolving working state. The readout letter never overwrites that state, and the software never selects the next rule.

Memory is computed from the original prompt once per forward and reused, not regenerated from the model's predicted answer. Read-only means fixed *within that forward*, not detached from learning: prefix activations depending on trainable LoRA must retain their gradient paths and be recomputed after parameter updates. The correctness path uses `use_cache=False`. It saves differentiable first-pass layer inputs and restores non-working positions before each layer. It still executes dense full-sequence layers and discards their recomputed prefix outputs; answer-only attention and projected-memory reuse are not implemented. No speedup is established.

Tiny-model tests verify one-pass equivalence, left/right padding, fixed per-layer prefix invariance, shared parameters, earlier-pass gradients, and equality of both outputs and adapter gradients against independently recomputed context. The actual pretrained one-pass gate runs before training. Historical checkpoints retain full-sequence recurrence; the new mode is `recurrence_mode: fixed_prompt`, saved as `loopformer-stage1-fixed-prompt-v1`. Missing/contradictory modes fail rather than silently reinterpret adapters. Fresh adapters or same-mode resume are required.

This is a concrete restriction of the recurrent workspace, not a guaranteed implementation of a symbolic transition. One vector may be an inadequate or poorly conditioned workspace; removing distributed writable state is a real risk. Any improvement supports the combined memory/workspace architecture, not a uniquely identified explanation of the old failure. No claim about later task families follows from pointer results.

### Supervision and first comparison

Keep the current per-pass pointer CE and balanced completion BCE, including the existing coefficient 0.1, for the initial architectural comparison. At pass t, supervise the exact reference symbol after t transitions, continue before d, and stop at d. Full differentiable rollout uses the model's own hidden state throughout; a provisional early stop does not truncate training supervision. Mask labels after d exactly as before. There is no numeric-progress target and no supplied countdown.

Do not require full working vectors to match whenever they decode the same pointer: differing requested depths and elapsed histories may require different internal progress information. Correct one-step behavior is the objective; hidden-state equality is not its definition. A future consistency objective would need to identify the task-state portion without erasing progress, and is not part of this proposal.

The supplied first comparison uses fresh adapters, the same depth-1–6 seed-37 data and seed-17 development sets, and the same optimizer, batching, selection rule, and update budget as joint completion. This keeps increased depth exposure or a changed stopping loss from explaining the architectural comparison. No claim that six supervised depths suffice for arbitrary counting is made. If coverage is subsequently expanded, held-out depth must move outward and both architectures need the same exposure.

The implemented inference path actually obeys H, under a depth-independent safety cap, and returns C's symbol at the first threshold crossing. The cap is a runtime safeguard, not an input feature or a parsed-depth stop. Keep full forced-loop traces alongside this path to distinguish execution from timing; compare measured answers and stopped prefixes against offline replay. Freeze a threshold on development data before any confirmation. The initial threshold 0.5 remains a diagnostic, not a guarantee of calibration.

Before a full pretrained run, require one-pass equivalence and the architectural contracts above, then a small training check that pointer and completion losses both receive the intended gradients. The full comparison must report nominal trajectories, correct-prefix length and conditional next-step accuracy by recurrent age, first-stop timing, and actual stopped-answer accuracy at trained and held-out depths. A gain confined to trained depths or stopping accuracy does not establish reusable long-depth transitions. The user authorized implementation and a desktop launcher. Local tiny-model CLI training, resume, checkpoint reload, naive evaluation, full-loop evaluation, and actual stopping pass. No pretrained training was launched locally.

### Commands and saved artifacts

These commands describe the completed depth-6 experiment. For the newly approved sparse-count experiment and paired diagnostic, use the [current runbook](training_pointer.md#depth-12-with-held-out-counts-current-desktop-run).

On the CUDA desktop, run `bash train_fixed_prompt.sh --dry-run`, inspect the budget, then `bash train_fixed_prompt.sh`. The launcher uses `configs/stage1_pointer_depth6_fixed_prompt.json`, preserves the live Rich dashboard in a PTY, and records terminal stdout/stderr to a timestamped launch log beside the run directory. It refuses an existing output directory and propagates training failures. The preview performs no writes.

Checkpoints and machine-readable training metrics go to `models/stage1_pointer/depth6-fixed-prompt-seed37/`. `bash eval_ckpts.sh` resolves this run's pointer-CE-selected checkpoint and runs the two seed-17 development splits in both modes: full 20-loop sweeps at batch 16, and actual head-controlled stopping with cap 20 and threshold 0.5 at batch 1. Its timestamped directory under `eval/pointer_loops/` contains `validation/`, `depth_test/`, `validation-stopped/`, `depth_test-stopped/`, and `run.log`. No model file is deleted or overwritten. An explicit older completion-run directory remains accepted by the evaluator wrapper.

Stopped exports contain per-example decisions and per-loop trajectories. Summary metrics include final accuracy, first head-stop exact/early/late rates, cap-fallback rate, exact-stop-and-answer joint success, executed loops, and synchronized latency, overall and by depth where applicable. A fallback at the requested depth is not counted as a learned exact stop. A stop head that never crosses its threshold remains a fallback even if its forced final readout is right. Full-loop metrics retain their historical nominal semantics.

### Evidence and limits of the motivation

[Yang et al., section 4.1](https://arxiv.org/html/2311.12424v3#S4.SS1) compare weight-tied recurrence with repeated access to the original input and find better extrapolation with input injection on their regression task. Their additive injection is different from the read-only attention memory proposed here, and their fixed-point objective differs from pointer transitions. This supports investigating persistent task information, not predicting success for this implementation.

The earlier [prefix-splice experiment](experiments/stage1_controlled_restarts.md) was harmful. It replaced part of a late state in a model trained for full-sequence recurrence. The proposed model instead learns with fixed memory from its first update, at every layer and loop. That distinction makes it a new architectural hypothesis, not evidence that the failed splice secretly worked. The completed [joint completion ablation](experiments/stage1_learned_completion.md) also leaves auxiliary-gradient interference unresolved; retaining its loss here isolates the architectural change rather than claiming to fix that separate possibility.
