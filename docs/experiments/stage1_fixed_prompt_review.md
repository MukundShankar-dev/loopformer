# Fixed-prompt recurrence: results and failure investigation

Reviewed 2026-10-04. This is a development-data investigation, not confirmation or evidence of a unique internal mechanism. No pretrained inference or training was run locally.

## Run and reproduction

The desktop ran `bash train_fixed_prompt.sh` followed by `bash eval_ckpts.sh`, using `configs/stage1_pointer_depth6_fixed_prompt.json`. The run completed 3,750 updates on 30,000 seed-37 examples with requested depths 1–6, microbatch 4 and accumulation 2, float32, rank-8 q/v LoRA, and joint pointer CE plus 0.1 balanced completion BCE. The frozen-base/gradient gate passed with zero one-pass logit error and 387,073 trainable parameters. Training took 3,516 seconds; validation 348 seconds; total wall time 3,870 seconds (64.5 minutes), peak CUDA allocation 7.02 GiB. This does not measure GPU utilization.

Training artifacts: `models/stage1_pointer/depth6-fixed-prompt-seed37/`. Trained-depth pointer validation CE selected step 3250. Evaluation artifacts: `eval/pointer_loops/depth6-fixed-prompt-seed37-best-20261004T130128Z-1212/`, containing validation/depth_test full traces and actual stopped counterparts. Both development splits contain 1,000 examples, 125 per depth. Full traces use 20 loops; stopping uses cap 20 and threshold 0.5.

Reproduce the additional read-only analysis from the repository root:

```bash
python eval/pointer_diagnostics/fixed-prompt-review/analyze.py
```

This writes `analysis.json` beside the script; the original terminal capture is `analysis.txt`. It audits targets, computes first-failure risk sets, stop-logit separability, and a development-fitted upper bound for any shared threshold. No policy is selected by that bound. Input trace hashes are recorded. The review additionally checked both evaluation dataset hashes and all recorded source hashes against local files, recomputed trajectory accuracy from CSV rows, and matched actual stopped loop counts and predictions to full-trace replay. Pretrained adapter binaries are absent locally and not independently verified.

## Execution improved, but only to a limited horizon

Complete nominal trajectories, requiring every intermediate prediction to be correct:

| Depth | Previous joint completion step 3250 | Fixed prompt step 3250 | CE-only step 2500 | CE-only step 3250 |
| --- | ---: | ---: | ---: | ---: |
| 7 | 76.0% | 97.6% | 94.4% | 93.6% |
| 8 | 27.2% | 96.0% | 94.4% | 86.4% |
| 9 | 2.4% | 72.8% | 73.6% | 56.0% |
| 10 | 0% | 3.2% | 41.6% | 15.2% |
| 11 | 0% | 0% | 14.4% | 2.4% |
| 12 | 0% | 0% | 1.6% | 0% |
| 13–16 | 0% | 0% | 0% | 0% |
| 9–16 pooled | 0.3% | 9.5% | 16.4% | 9.2% |

The architectural comparison supports improved execution relative to the preceding joint run. It does not exceed the older CE-only frontier. Freezing prompt memory and restricting the writable workspace changed together, so improvement cannot uniquely be attributed to preserving rule representations.

## Finding 1: stopping has an explicit training-support gap

The loss in `scripts/training/objective.py` gives continue at t<d, stop at t=d, and no loss at t>d. With d restricted to 1–6:

- Every supervised label at loop 6 is stop; there are no continue examples there.
- No supervised labels exist at later loops.
- Requested Steps values 7 and above are absent from the task training distribution.

The loss is implemented correctly, but multiple solutions fit these labels, including a six-depth classifier plus finite-duration internal progression. It does not uniquely identify a reusable count-comparison algorithm. A learned state must retain enough progress information to distinguish the same pointer symbol reached at different elapsed times; a pure current-symbol transition alone cannot decide completion for arbitrary requested N. That information must be learned internally under the current contract, not provided by the controller.

On all 750 trained-depth evaluation examples, stopping is exact. On depth 7 it is exact for 3/125, on depth 8 for 0/125. Every depth-9–16 example stops at loop 5 (490 cases) or loop 6 (510). At that premature stop, 985/1,000 still decode the correct intermediate symbol. This separates premature stopping from an already incorrect pointer trajectory; it does not prove why H or its input representation fails.

Saved 32-per-depth training-validation traces show zero exact stops at depths 7 and 8 at every checkpoint from update 0 through 3750, while complete pointer trajectories improve to 32/32 at both depths at steps 3250 and 3750. Thus the observed stopping failure is persistent, not confined to the selected checkpoint. These small cohorts do not establish full deep performance for other checkpoints.

## Finding 2: a threshold change is insufficient

Use raw logits rather than saturated float32 probabilities. For a given example, a shared logit threshold c stops exactly at d only if max(z_1,...,z_(d-1)) < c <= z_d. Sweep all interval boundaries to obtain an optimistic, label-fitted maximum count:

| Cohort | Maximum exact stops under any single threshold |
| --- | ---: |
| Depths 1–6 | 750/750 |
| Depths 7–8 | 135/250 |
| Depths 9–10 | 106/250 |
| Depths 9–16 | 106/1,000 |
| Depths 1–16 | 763/2,000 |

Each row optimizes its own threshold on that cohort and is an upper bound, not an independent result. At depths 13–16, none of the examples even has a terminal logit greater than all preceding logits, so even example-specific thresholds cannot yield exact first-crossing timing. Calibration alone cannot recover useful deep stopping.

## Finding 3: execution develops a sharp recurrent-age failure

On the 1,000 depth-9–16 examples, compute accuracy only among trajectories correct at every preceding loop:

| Loop | Correct next / eligible | Conditional accuracy |
| --- | ---: | ---: |
| 6 | 971/982 | 98.9% |
| 7 | 956/971 | 98.5% |
| 8 | 889/956 | 93.0% |
| 9 | 482/889 | 54.2% |
| 10 | 46/391 | 11.8% |
| 11 | 1/42 | 2.4% |

Risk sets exclude tasks already completed, so their composition changes after loop 9. This is not merely declining complete-trajectory accuracy caused by constant small per-step errors. Of 905 failing deep examples, 487 first errors repeat the immediately preceding reference state, 194 return another earlier path symbol, 110 predict a later path symbol, and 114 are off-path. Reference nominal paths never repeat, so those repeats are erroneous rather than valid cycles.

The observations are consistent with recurrence becoming unable to advance reliably, but decoded repeats do not prove a hidden-state fixed point. This checkpoint's full evaluations saved logits, not recurrent hidden vectors or attention. We cannot establish a norm explosion, attractor, loss of rule access, or insufficient LoRA rank from these files. H is a readout, so crossing its threshold cannot itself modify forced-run R; shared training gradients could still couple their representations.

## Finding 4: requested-count representation is also out of distribution

The cached pinned Qwen tokenizer encodes `Steps: 6` with one digit token and `Steps: 10` with two (`1`, `0`). Thus depth 10 changes both the requested number and the length/positions of the prompt suffix relative to every training example. This is a verified representation change, not proof it causes failure. Even at the same recurrent age (loop 9), conditional correctness is 91/115 = 79.1% for requested depth 9, versus 391/774 = 50.5% for requested depths 10–16. The maps and path conditioning differ between cohorts, so this observational contrast cannot isolate the numeral or positional effect. Paired same-map Steps controls are needed. A depth-8-only extension would still omit all two-digit requests.

## What relevant literature does and does not establish

- [Newman et al., The EOS Decision and Length Extrapolation (2020)](https://aclanthology.org/2020.blackboxnlp-1.26/) report that learning termination can degrade length extrapolation and identify position-stratified states and attractor behavior in their models. This motivates inspecting recurrent states and separating execution from termination. Our separate head and latent-loop task differ; matching behavioral symptoms are not a diagnosis of their mechanism here.
- [Fan et al., Looped Transformers for Length Generalization (2025 version)](https://arxiv.org/html/2409.15647v3) use shared recurrent blocks, input injection, varied training iteration counts, and oracle or confidence-based inference. Their results support repeated access to input and diverse training depth. They do not establish that a frozen pretrained Qwen with q/v-only LoRA and a hidden-state stop head learns arbitrary counting from six requested values. Their confidence rule also cannot directly solve our task: a confident intermediate letter need not be the requested final letter.
- [Kuo et al., Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping (2026 preprint)](https://arxiv.org/html/2606.29983v1) separate oracle-over-iterations quality from actual policy quality and study stochastic training depths. Their results show substantial variability despite strong in-range performance. Their sampled-depth final-output supervision and RL stopping objective differ from our exact one-transition-per-loop target. Applying their schedule blindly would change our task semantics; this is not a reason to switch to RL now.

## Recommended bounded next experiment — proposed, not implemented

The strongest actionable evidence concerns training coverage and persistent stopping failure, rather than proof of inadequate rank. Keep fixed prompt memory, exact intermediate CE, shared R, and prompt-only learned stopping. Optimize throughput before spending another hour. Then use a fresh, broader-depth dataset with a diagnostic split that separates count interpolation from extrapolation:

- Train requested depths 1–6, 8, 10, and 12, with every intermediate transition supervised as before. Hold out requested depths 7, 9, and 11 entirely from training prompts. Their intermediate positions still receive supervision within longer training tasks; label this **held-out requested-count interpolation**, not unseen recurrent-age generalization.
- Evaluate genuinely unseen recurrent depths 13–20 separately, with new held-out mappings. Keep seed 29 reserved until a policy and acceptance thresholds are frozen. The generator supports depths through 25; its no-repeat constraint means deeper maps are also conditioned differently, so paired controls remain important.
- Include paired prompts with the same mapping/start and different Steps, using mappings whose paths support every tested depth. Evaluate agreement of intermediate predictions over their common prefix and how stopping moves. Keep all related mappings inside one split. This controls the otherwise confounded change in graph distribution with depth.
- Report forced trajectories and actual stopped success independently. Propose at least 90% joint exact-stop-and-correct-answer success on held-out requested counts as a useful engineering milestone; report depth-13+ curves honestly even if they fail. This is a proposed development target, not a retrospectively passed research gate.

The choice of depth 12 gives supervised continue labels at the current execution failure region (loops 9–10), whereas extending only to 8 leaves that region unsupervised. It also trains stopping beyond single-digit requested values. It does not guarantee extrapolation past 12. If held-out counts fail while forced trajectories succeed, focus the next intervention on learning/representing progress; if both fail at experienced recurrent ages, investigate R capacity/dynamics before another horizon extension.

Before that run, a small paired-Steps evaluation of the existing checkpoint can distinguish sensitivity to requested count from recurrent age/map confounds without retraining. It should save working-state norms/deltas and head logits through the first failure, while scoring rule lookup and stop timing separately. Fixed-prompt-specific recording must be verified before using older full-sequence splice interventions. The local machine has no adapter binaries, so that causal comparison cannot be executed from the current artifacts. No external counter, fresh intermediate prompt, decoded-state reinsertion, numeric-progress target, or architecture change is proposed here.
