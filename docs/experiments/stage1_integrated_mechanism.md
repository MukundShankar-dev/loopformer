# Integrated pointer mechanism diagnostic

Run reviewed 2026-10-03 Eastern (executed 2026-10-04 00:18–00:25 UTC). This is a frozen-checkpoint **development** diagnostic, not a new training run or a confirmation test.

## Configuration and artifacts

The desktop ran [the documented wrapper](../diagnostics_and_performance.md#integrated-frozen-mechanism-diagnostic) from code revision `9d22c86`:

```bash
bash diagnose_mechanism.sh
```

It used pinned Qwen2.5-0.5B-Instruct, float32/eager CUDA, deterministic algorithms, batch one, and three checkpoints from the fresh depth-6 run: steps 2500, 3250, and 3750. The main internal and rule-edit probes use step 2500; the matched checkpoint/horizon comparisons use all three. The fixed seed-17 cohorts contain 144 validation mappings for fitting only the offline linear probes, 64 separate validation mappings (depths 1–8), and 64 deep development mappings (depths 9–16), eight per depth. Seed 29 remains untouched.

Artifacts: [`eval/pointer_mechanism/matched-2500-3250-3750-20261004T001803Z-552/`](../../eval/pointer_mechanism/matched-2500-3250-3750-20261004T001803Z-552/), including [`results/summary.json`](../../eval/pointer_mechanism/matched-2500-3250-3750-20261004T001803Z-552/results/summary.json), per-loop/case CSVs, exact prompts and token IDs, answer-position state vectors, and `run.log`. The run ended with `status: complete` in about 7m49s. All 13 saved artifact hashes match the summary; both dataset hashes and all 12 recorded source hashes match the local checkout. The 3,264 native comparison predictions and targets agree exactly with the matching rows of the six earlier full checkpoint evaluations. Checkpoint adapter binaries are absent locally, so their saved SHA-256 values (`25c428d2…`, `8a088eee…`, `96f7ede7…`) could not be independently recomputed on this machine.

## Matched checkpoint behavior

| Checkpoint | Validation complete / 64 | Deep complete / 64 | Deep complete at depths 9, 10, 11, 12–16 |
| --- | ---: | ---: | --- |
| 2500 | 60 | 12 | 6, 5, 1, 0 |
| 3250 | 60 | 3 | 2, 1, 0, 0 |
| 3750 | 63 | 8 | 5, 3, 0, 0 |

Compared with step 2500 on the *same 64 deep mappings*, step 3250 gains one complete trajectory and loses ten; step 3750 gains one and loses five. Yet mean per-case step accuracy rises by 2.17 and 3.34 percentage points, respectively, with mapping-level stratified-bootstrap 95% intervals of −3.93 to +8.16 and −2.65 to +9.05 points. More later-loop predictions can become correct after an earlier mistake while fewer trajectories remain correct throughout. Aggregate step accuracy alone would miss this.

Later checkpoints reduce early errors but become less reliable around loops 9–10. Among deep cases reaching loop 9 with a correct prefix, first failures are 8/50 at step 2500, 18/55 at 3250, and 17/54 at 3750; at loop 10 they are 13/36, 19/35, and 18/32. The risk sets differ across checkpoints, so these are descriptive rates. Restricting loop 9 to mappings both checkpoints got right through loop 8, step 2500 alone is correct in 13 cases versus five for step 3250 (44 eligible); against 3750 the corresponding counts are 11 versus four (42 eligible). These small paired counts support the direction but are not decisive by themselves. The earlier 1,000-example [full evaluation](stage1_fresh30k.md#full-evaluation) provides stronger evidence for the nearby-depth complete-trajectory decline.

The correct-prefix comparison is heterogeneous: relative to step 2500, step 3250 lengthens 22 prefixes, shortens 31, and ties 11; step 3750 lengthens 18, shortens 24, and ties 22. Mean prefix differences are slightly positive because some early failures move much later, and their bootstrap intervals include zero. Thus “every case degrades” and “mean correct prefix falls” are both unsupported.

## Depth cue and structural controls

Changing the same deep prompt from native `Steps: d` to `Steps: 17` leaves 771/800, 780/800, and 788/800 predictions unchanged for steps 2500/3250/3750. The mean **late** per-case accuracy changes on the 56 mappings with equal token counts are only +0.4, −0.4, and −0.4 percentage points. At step 2500 a `Steps: 6` cue improves first-six accuracy from 339/384 to 352/384, but it changes token length for 56/64 mappings; the eight equal-token-count cases have no accuracy change. That cue is only a valid instruction through loop six, so its later outputs are stress observations. All 3,456 six-loop outputs agree with the first six outputs of the corresponding full unrolls. The displayed requested horizon is therefore not a sufficient explanation of the late failure on this cohort.

The trained adapters matter: disabling recurrent LoRA at step 2500 reduces correct symbolic readouts from 277/288 to 6/288 on validation and from 529/800 to 21/800 on deep examples. An inert adapter cannot explain the measured early behavior.

Rule order and symbol names are imperfect invariances. On deep examples, shuffled rule order preserves the exact prediction at 528/800 loops and full A–Z renaming preserves the correspondingly renamed prediction at 502/800 loops. At early loops 1–6, nearly all *originally correct* predictions remain correct (338/339 for rule order, 335/339 for renaming), while many original early mistakes change or recover. At loops 9+, exact equivariance drops to 80/288 and 68/288. These are prompt-distribution sensitivities, especially after the model has gone wrong; they do not establish a simple fixed-position shortcut. An irrelevant-edge edit leaves 709/800 deep predictions unchanged overall but can also disturb late failures.

## Necessary-rule edits and internal observations

At an early correct transition, all 106/106 matched relevant edits follow the new target, while none of the 106 matched irrelevant edits changes the answer. All 53 cases with two eligible replacement destinations follow **both** targets. For the next one, two, and three transitions, the changed branch stays entirely correct in 104/106, 97/106, and 82/106 trials; matched irrelevant branches score 105/106, 99/106, and 80/106. This is strong evidence of local rule-dependent execution and short compositional continuation on this distribution, not proof of a unique internal algorithm.

At a later original first error, only 17/70 eligible relevant edits follow their new target; only 6/31 cases with both eligible replacements follow both. Relevant edits change the prediction in 29/70 matched trials, while *irrelevant* edits change it in 19/70. Eight irrelevant edits happen to correct the original target; because the original readout at that loop was wrong by selection, this is perturbation-induced recovery, not stability. The late result is therefore a mix of weak target-specific following and general prompt sensitivity. Both numerator and prefix-retention denominators matter: 70 matched pairs were retained from 84 attempted pairs; those retained pairs cover 39 cases.

The held-out early linear probes decode the target from R's answer-position state and the normalized frozen-C output at 255/264 steps each, close to the installed readout's 256/264. On deep loops 9+, the model itself is correct at 85/288 steps, while the early-trained R and C probes score 60/288 and 71/288. At the **38 late first errors**, the correct target is linearly decoded by R in one case and by C in two. This offers no evidence for a simple failure of the installed frozen readout while a readily linearly decodable correct answer persists upstream. A shallow-trained linear probe can fail on a changed representation even when nonlinear information remains.

At those 38 late first errors, answer-position relative update magnitude falls versus the preceding loop in all 38 cases (mean 0.476 to 0.302); its active-rule source top-1 attention-head count, averaged across recurrent layers, falls from 2.26 to 1.09. Successful late transitions also often have smaller updates than their predecessors (65/70), so shrinking updates alone is not the diagnosis. Full-sequence relative update magnitude remains around 0.03 rather than freezing. Attention correlation cannot distinguish wrong-state selection from inability to retrieve the needed rule.

## Interpretation and next gate

The evidence rules against three simple explanations: the trained adapters are inert; the model only repeats a fixed letter or ignores arbitrary early rule replacements; or the late boundary is primarily caused by the displayed `Steps` value. It supports a narrower failure class: reliable local transitions and short branch continuation do **not** remain stable as recurrent history grows; later training improves early execution while worsening the nearby depth-extension frontier. The precise locus—state encoding, rule selection/access, recurrent update, or their interaction—remains unresolved. Prior [suffix-restart](stage1_controlled_restarts.md) and [rule-edit](stage1_rule_edit_probe.md) results are consistent with history sensitivity but have prompt/intervention confounds.

Do not select an architecture or launch another SFT recipe from the aggregate metrics alone. A next bounded diagnostic should compare the **same table edge** when encountered at early versus late recurrent ages, with matched displayed horizon and exact reference execution, then use a controlled internal intervention only if that behavioral age effect persists. This would test the remaining history-versus-edge-difficulty distinction without supplying an oracle state during normal inference. Freeze the design and decision rule before evaluating the reserved confirmation seed.
