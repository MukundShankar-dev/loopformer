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

The restart benefit survives the Steps-text control. For the 22 previously identified late first errors, execution through loop six is still on the correct reference path, so the suffix Start is also the model's correctly decoded state at that boundary. The useful ingredients of suffix restart are therefore re-encoding the current symbol and rebuilding a coherent prompt/state, not correcting an already-wrong symbol or merely requesting fewer Steps.

The rule-refresh failure does not show that persistent rule memory is useless. It replaces one token region with loop-age-zero representations while leaving the rest at loop age six. R was never trained on that mixed representation. The result instead shows that this direct splice is invalid for the trained checkpoint and suggests that token-region representations are coupled across recurrent age. It provides no reason to run the 1,000-example version of this exact harmful intervention.

The evidence now favors a bounded continuous-state or re-entry problem over a prompt depth-cue problem. It does not yet distinguish whether success requires explicit symbolic re-encoding, a learned bridge, normalization, or jointly refreshed context. Introducing any of those into training would change the architecture and must be tested as a fresh controlled run.

## Next bounded experiment

Do not run `bash probe_controls.sh full`; the small paired result is sufficient to reject the direct rule-prefix splice and the Steps-cue explanation.

The next diagnostic should test re-entry without oracle correction: after loop six, decode the model's current symbol, rebuild the prompt with that symbol as Start, preserve the original rule table, and execute the remaining loop budget. Report separately the cohort whose loop-six symbol is reference-correct and the cohort whose symbol is wrong. On the correct cohort this tests whether explicit state re-encoding and coherent context reset reproduce the suffix benefit without consulting the reference state. On the wrong cohort it measures propagation from the model's own error and must not be described as repair.

This remains an inference diagnostic. Its outcome should determine whether a learned continuous re-entry/bridge experiment is justified. It should not silently become teacher forcing or a new training objective.

