# Joint learned-completion ablation

## Setup and execution

The CUDA desktop completed a fresh depth-6 run using [the completion config](../../configs/stage1_pointer_depth6_completion.json) and the same 30,000-example seed-37 training set, seed-17 validation set, pinned Qwen revision, recurrent split, rank-8 q/v LoRA, batch 4 with accumulation 2, and 3,750-update optimizer budget as the [fresh 30k CE-only baseline](stage1_fresh30k.md). The new terms are a 128-wide hidden-state completion head and weight 0.1 on balanced continue/stop BCE. Intermediate pointer CE remains active at every nominal pass; validation pointer CE on depths 1–6 still selects the checkpoint. No explicit loop or parsed-depth feature enters the head or recurrent block. See the [design and implementation guide](../learned_loop_completion.md).

The training command was:

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_depth6_completion.json --device cuda \
  --output models/stage1_pointer/depth6-completion-seed37
```

The desktop then ran `bash eval_ckpts.sh`, which resolved the selected checkpoint and evaluated the complete seed-17 validation and depth-test sets for 20 forced loops with batch 16. Training completed at step 3750; the pointer-CE-selected checkpoint was step 3250. The startup gate passed with zero reported one-loop logit error. Training used 3,648.48 wall seconds and 7,527,191,552 peak allocated CUDA bytes, close to the CE-only run's 3,654.26 seconds and 7,525,116,928 bytes. These are tensor allocations, not total VRAM use.

Artifacts: `models/stage1_pointer/depth6-completion-seed37/` and `eval/pointer_loops/completion-best-20261004T040328Z-1574/`. The selected checkpoint's adapter binary remains on the desktop; its hash is recorded in the evaluation metadata but its contents were not independently checked on this Mac.

## Forced-depth pointer execution

The comparison below uses the **same checkpoint update 3250** and the same seed-17 examples. Each depth has 125 examples. "Trajectory" requires every nominal intermediate prediction through the requested depth to be correct. The CE-only evaluations used batch 1 and 8/16 forced loops; the completion evaluations used batch 16 and 20 forced loops. Both runs score the same nominal prefix.

| Requested depth | CE-only step 3250 trajectory | Joint completion step 3250 trajectory |
| --- | ---: | ---: |
| 1–6 combined | 99.2% (744/750) | 98.93% (742/750) |
| 7 | 93.6% (117/125) | 76.0% (95/125) |
| 8 | 86.4% (108/125) | 27.2% (34/125) |
| 9 | 56.0% (70/125) | 2.4% (3/125) |
| 10 | 15.2% (19/125) | 0% |
| 11 | 2.4% (3/125) | 0% |
| 12–16 | 0% | 0% |

Across all validation depths 1–8, completion reaches 87.1% complete trajectories and 87.8% nominal final-answer accuracy, versus 96.9% trajectories and 97.3% final-answer accuracy for the matched CE-only checkpoint. Across depths 9–16, completion reaches 0.3% complete trajectories and 6.4% nominal final answers, versus 9.2% and 14.1%. The working CE-only step-2500 checkpoint is stronger still at extended depths, with 16.4% complete deep trajectories. Thus the joint variant's loss of depth extension appears even when the caller forces the correct number of loops; early stopping alone does not explain it.

## Completion-head timing

At the prespecified diagnostic probability threshold 0.5, the head's **first** stop signal is exactly at the requested pass for 745/1000 validation examples and for 0/1000 depth-test examples. It fires early for the other 255 validation examples and **all** 1,000 depth-test examples. On validation, all depth-1–5 examples stop at the correct pass, 120/125 depth-6 examples do, and every depth-7–8 example stops early. On depth-test examples the first signal occurs at loop 3 (213 cases), 4 (760), or 5 (27). Replaying the answer readout at that first crossing gives 740/1000 correct final symbols on validation and 0/1000 on depth test. This is offline replay from full traces, **not** executed early-stop inference or measured compute savings.

The saved `stop_at_depth_accuracy` is 100% on both splits because the head also signals stop at the requested pass. It does not mean correct timing: earlier threshold crossings make it unsuitable as a standalone stop metric. Later requested-depth logits are often extremely positive, so the failure is specifically a missing *continue-until-depth* behavior. The 0.5 threshold is diagnostic and was not fit on depth-test data. A different threshold could change timing; no development threshold or actual self-stopped policy has been evaluated.

## Audit and interpretation

Both evaluations completed and recorded the same validation and depth-test file hashes as the baseline (`d5925d830c330808bc5df636018ef0728e39b376ca63fa2df2b8b6fce1355b74` and `20c92ce8159e895cd98e7cee2a2da46e01e0ecdf07457e58d91ec31134ead4c9`). The 20,000 trajectory rows per completion split cover 1,000 examples × 20 loops; nominal targets and correctness were checked against reference execution. The 2,048 saved training-monitor rows for selected step 3250 overlap the full validation sweep: all pointer predictions, targets, and stop labels agree across training-monitor and batch-16 evaluation, with maximum stop-logit difference below 0.001. This reduces concern that evaluation batch size created the observed collapse. The pretrained weights and full training file were not independently rehashed here.

This single configured run shows that the joint completion objective learned timing within the trained range but **did not extrapolate** the stopping rule or improve pointer execution beyond it. The forced-depth comparison rules out a merely premature external stop as the whole explanation. The auxiliary objective may have altered recurrent states in an unhelpful way, but these outcomes do not isolate gradient interference, representation loss, or another internal cause. This is a development-set ablation, not an independent confirmation. A new training run or threshold search is not justified by these results alone; any next experiment should distinguish a decodable progress signal from effects of jointly changing the recurrent dynamics.
