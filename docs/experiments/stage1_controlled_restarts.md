# Controlled restart and rule-context probes

Run completed 2026-09-30 on CUDA using the retained step-2500 checkpoint. This is a seed-17 development diagnostic, not confirmation. It performed no training and did not modify the checkpoint.

## Configuration and artifacts

The command was:

```bash
bash probe_controls.sh
```

The wrapper evaluated the first 32 examples from both `validation.jsonl` and `depth_test.jsonl`, with `restart_after=6`, float32 eager inference, batch size one, and strict deterministic execution. The complete artifacts and log are under [`eval/pointer_probes/controls-small-20260930T112605Z-473/`](../../eval/pointer_probes/controls-small-20260930T112605Z-473/). Both summaries report `status: complete`. The adapter, checkpoint metadata, tokenizer, and dataset hashes match the preceding step-2500 probe.

The unchanged original and identical-copy `rule_noop` match on every exported observation. This verifies that the hook and concatenation path alone did not alter execution.

## Results

| Split and variant | Complete trajectories | Correct nominal steps |
| --- | ---: | ---: |
| Validation original | 30/32 | 138/144 |
| Validation rule refresh | 25/32 | 130/144 |
| Depth-test original | 6/32 | 276/400 |
| Depth-test rule refresh | 0/32 | 186/400 |
| Depth-test suffix restart | 30/32 | 206/208 |
| Depth-test suffix with original displayed Steps | 30/32 | 206/208 |

The displayed Steps control is nearly invariant. Across 208 aligned depth-test suffix transitions, changing the displayed value from remaining depth to original depth changes one decoded prediction, and both versions are wrong on that transition. It creates no correctness changes. The two suffix variants also match on all 12 validation transitions. Therefore the strong suffix-restart result is not explained by reducing the displayed Steps value in this sample.

Refreshing only the Rules prefix is harmful. In the deep probe it changes 195/400 predictions, causes 96 correct-to-wrong transitions, and repairs six wrong transitions. It loses all six originally complete trajectories and gains none. Validation loses five complete trajectories and gains none.

The damage begins exactly when the intervention activates:

| Split | Loop | Original correct | Rule-refresh correct | Median target margin, original | Median target margin, refresh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 6 | 11/12 | 11/12 | 10.80 | 10.80 |
| Validation | 7 | 7/8 | 2/8 | 9.61 | -4.08 |
| Depth test | 6 | 28/32 | 28/32 | 9.90 | 9.90 |
| Depth test | 7 | 28/32 | 5/32 | 9.46 | -2.13 |

At deep loop eight, refresh scores 3/32 versus 28/32 original; at loop nine it scores 0/32 versus 25/32. The immediate margin reversal is inconsistent with a simple account where stale Rules representations are the principal cause and replacing them with clean prelude representations should repair execution.

## Interpretation

The restart benefit survives the Steps-text control. For the 22 previously identified late first errors, execution through loop six is still on the correct reference path, so the suffix Start is also the model's correctly decoded state at that boundary. Restarting changes the Start symbol, reruns P, resets all recurrent positions, and shortens the executed loop sequence together. This result does not isolate which change helps.

The rule-refresh failure does not show that persistent rule memory is useless. It replaces one token region with loop-age-zero representations while leaving the rest at loop age six. R was never trained on that mixed representation. The result instead shows that this direct splice is invalid for the trained checkpoint and suggests that token-region representations are coupled across recurrent age. It provides no reason to run the 1,000-example version of this exact harmful intervention.

The evidence weakens the displayed depth-cue explanation but does not distinguish a wrong rule lookup, loss of the current state, failure of C to read the state, or a broader recurrent dynamics problem. Choosing a bridge, static memory, or a new loss from these probes would be premature.

## Next bounded diagnostic

Do not run `bash probe_controls.sh full`; the small paired result is sufficient to reject the direct rule-prefix splice and the Steps-cue explanation.

An inference procedure that decodes the current letter, rebuilds the prompt, and invokes the model again is explicitly rejected: the external controller would perform the state transfer, so success would not show that the recurrent model learned to carry the process internally.

Keep the checkpoint frozen. On a small set of examples with a correct prefix and a late first error, make a counterfactual copy of the original question that changes only the outgoing rule for the *reference* current state at that loop. Keep the same Start, Steps, rule order, and number of recurrent loops. Compare whether the model's prediction at that loop follows the changed edge. An irrelevant-rule edit is the negative control; apply the same relevant edit at early loops the model normally solves as a positive control. Record the exact edited rule, expected target, prediction, target rank, margin, and whether earlier predictions changed.

This tests whether the failing loop still responds to the rule it should use. A failure to respond would narrow the mechanism but would not by itself distinguish lost state from lost access to the rule; changing a rule can also perturb earlier hidden representations. Analyze only cases where the counterfactual retains the correct prefix, and report that denominator. The model receives each complete question once and must solve it internally. No decoding, prompt reconstruction between loops, training update, or architecture change is part of this diagnostic.
