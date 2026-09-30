# Frozen rule-edit probe of late pointer failures

Run completed 2026-09-30 on CUDA. This is an exploratory seed-17 development diagnostic of the retained step-2500 checkpoint, not a training run or confirmation result. The checkpoint was not modified.

## Configuration and audit

From the repository root, the desktop ran:

```bash
bash probe_rule_edits.sh
```

The wrapper executed `scripts.eval.rule_edit_probe` with `--device cuda --limit 32 --early-loop 6` on [`data/pointer/seed-17/depth_test.jsonl`](../../data/pointer/seed-17/depth_test.jsonl). Original prompts determined first errors; 22 of the fixed 32 examples had a correct original prefix through loop six and a later first error (loops 9: four, 10: eight, 11: seven, 12: three). For each eligible example, the script reran complete prompts with one outgoing edge changed at loop six and at the first-error loop, paired with an edge whose source is absent from the whole reference path. Start, Steps, rule order, token count, and inference loop count stayed fixed. The changed prompt entered the model from loop one; no decoded answer was fed back between loops.

Artifacts: [`summary.json`](../../eval/pointer_rule_edits/baseline2500-20260930T122401Z-2350/results/summary.json), [`pairs.csv`](../../eval/pointer_rule_edits/baseline2500-20260930T122401Z-2350/results/pairs.csv), [`states.csv`](../../eval/pointer_rule_edits/baseline2500-20260930T122401Z-2350/results/states.csv), [`inputs.jsonl`](../../eval/pointer_rule_edits/baseline2500-20260930T122401Z-2350/results/inputs.jsonl), and [`run.log`](../../eval/pointer_rule_edits/baseline2500-20260930T122401Z-2350/run.log). The saved summary reports `status: complete`. Its data, inputs, and source hashes match the local files. Adapter, checkpoint metadata, tokenizer, and data hashes match the earlier controlled depth-test probe; all 400 original per-loop rows match that run exactly for prediction, target, correctness, rank, margin, and answer-state update RMS.

## Results

Analyze transition following only when the edited prompt's prior decoded predictions all match the unchanged reference prefix. The most comparable cohort also requires **both** relevant and irrelevant edits to retain that prefix for the same example.

| Probe loop | Matched correct-prefix cases | Relevant prediction changes | Relevant changes to new target | Irrelevant prediction changes |
| --- | ---: | ---: | ---: | ---: |
| Early loop 6 | 21 | 21 | 21 | 0 |
| Late first error | 19 | 9 | 6 | 4 |

At the early loop, all 21 matched relevant edits switch the answer to the new edge destination, while the irrelevant edits leave the answer unchanged and correct. No earlier decoded prediction changes in these matched cases.

At the late first error, the relevant edit makes seven of 19 matched outputs equal the new destination, but **one of those seven was already predicting that letter before the edit**. Thus six of 19 actually *change to* the new target. Two of the 19 baseline predictions already equal the chosen replacement letter; one stays there and one changes away. The relevant edit changes nine predictions overall, versus four for the irrelevant edit. In a paired shift count, five cases change only under the relevant edit, four change under both, and ten under neither. Of the six actual changes to the new target, four have an unchanged prediction under the irrelevant edit; in the other two, the irrelevant edit also changes the prediction, but not to the new target. Earlier decoded predictions remain unchanged in all 19 matched cases for both edits.

The relevant replacement is the first symbol absent from each complete reference path, so it is concentrated on A–E (B in eight of the 19 late matched cases). Four of the six late changes to the new target are to B. This is a limitation for claims about general rule following: a single selected destination per task does not eliminate target-frequency or prompt-perturbation effects.

The original target margin drops from a median 5.49 at the preceding correct loop to -2.75 at the first-error loop for the 19 retained relevant cases. The median answer-position update RMS in their original run drops from 0.621 to 0.399 across the same two loops. These are descriptive, failure-selected comparisons; they do not establish hidden-state collapse or prove where the transition went wrong. The existing `states.csv` holds scalar answer/sequence state summaries and C-decoded A–Z readouts, not full hidden vectors or direct pointer-state labels inside R.

## Interpretation and next decision

The model can use an edited rule reliably at loop six, and some late outputs still respond to an edited relevant rule. A claim that it simply cannot read the table after six loops is therefore too strong. Late responses are much less reliable, and irrelevant edits also perturb some late outputs. The result does not distinguish an incorrect latent current state, degraded rule access, an R update error, or C readout error. Because every edit changes the input from the start, a preserved **decoded** prefix does not guarantee the same hidden prefix.

Do not infer that adding input injection, a new head, or more training would fix this. Before an architectural choice, a small follow-up should test multiple distinct replacement destinations per case, exclude the original predicted letter, and retain the matched irrelevant control. This would establish whether late output tracks the *specified edge destination* rather than a favored letter or generic prompt sensitivity. If that test still shows selective tracking, the next question is where the intended current state and rule are represented; any hidden-state probe should use held-out instances and distinguish R state from the frozen C readout. Seed 29 remains untouched.
