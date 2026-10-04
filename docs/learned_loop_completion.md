# Learned completion of an explicit-step pointer task

**Status: joint training and full-loop offline stop diagnostics completed on pretrained Qwen; actual self-stopped inference has not run.** The [experiment report](experiments/stage1_learned_completion.md) records a negative result: at the 0.5 diagnostic threshold, the head stops early on every depth-9–16 example, and the jointly trained checkpoint has worse forced-depth pointer trajectories than the matched CE-only checkpoint. This is a Stage 1 ablation of the existing [pointer training](training_pointer.md), not the terminal-damage controller in [adaptive inference](adaptive_compute.md).

## Task and model contract

The **only task input** is the existing raw dataset prompt:

```text
Rules: ( A, C) ( C, D) ( D, E) ...
Start: A
Steps: 2
Answer:
```

Tokenization, attention mask, and the answer position are mechanical representations of that prompt. The rule table, start symbol, and requested depth appear in text; neither the recurrent block nor the proposed stopping head receives a separately parsed depth, loop index, remaining-step count, reference state, or previous decoded answer. No new prompt fields, special progress tokens, teacher-forced state insertions, or per-loop time embeddings are added.

Keep the current frozen prelude P, shared recurrent block R with trainable LoRA, and frozen coda C. For each pass, `h_t = R(h_{t-1})`. The ordinary symbol readout is `C(h_t)` at the existing answer position, restricted to the validated A–Z token set for training and evaluation. Add a small trainable halting head `H` that reads **only the answer-position vector from `h_t`** and emits one stop logit. The implemented head uses LayerNorm, a hidden linear layer, GELU, and a scalar linear output; its width and weights are saved with the adapter. The head's inputs contain no explicit `t`, `d`, confidence-history feature, or output from the reference interpreter. Its gradient flows into R so the auxiliary objective can change the recurrent representation. C's readout and H's stop decision do not feed back into R.

**Planned self-stopped inference:** supply the prompt and a fixed safety cap `B` that is independent of its requested depth. Repeat R until the stop probability crosses a threshold or the cap is reached. Decode the **single final A–Z symbol at the stopped pass**. Diagnostic exports should also retain stop probability, optional per-pass symbol predictions, executed loops, threshold, and whether the cap forced a stop. The outer loop still has to enforce a finite cap; it must not parse `Steps` to choose the stop pass. This inference mode is not yet implemented. The existing full-loop evaluator loads new checkpoints and records every stop probability for offline timing analysis, while retaining forced-depth pointer scoring. It does not skip work when the head crosses a threshold.

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

The checkpoint format is versioned: `loopformer-stage1-completion-v1` stores H and the recurrent adapters; historical `loopformer-stage1-v1` files still load. The existing `stop_policy` still calls a depth-aware policy and is **not** used for completion training. The implementation extends the shared model/trainer/validation/evaluator interfaces; no parallel pointer dataset was created. Focused tests cover head-only hidden-state input, exact 1-based stop targets, masking after `d`, H and LoRA gradient flow with frozen Qwen weights, microbatch weighting, checkpoint round-trip, resume, a tiny CLI update, and compatibility with full-loop evaluation and stop-probability export. Early-stop execution, cap fallback, and threshold selection remain inference work.

The original pointer task is not absorbing after `d`. This experiment tests **completion of the prompt's requested count**, not discovery of an intrinsically terminal state or adaptive allocation of useful compute. Because `d` is already given in the prompt, a learned head cannot claim a compute-allocation advantage over an external stop-at-`d` baseline. The scientific question is whether internal progress supervision improves the recurrent computation and whether that learned count extends beyond training depth.
