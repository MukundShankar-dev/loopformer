# First CUDA pointer overfit run — 2026-09-10

Audited an existing completed run; no training was launched during this documentation change. Run: `models/stage1_pointer/20260910T214347.326698Z/`, code revision `04552d2c7dab8dda173c17c83ec1663cd43a48df` (clean at launch). The saved `run.json` records the command, source hashes, data identity, packages, and startup gate.

```bash
.venv/bin/python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_overfit.json --device cuda
```

Configuration: pinned Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, seed 17, 32 training examples and 16 validation examples at depths 1–4, batch 1, accumulation 4, learning rate 0.0002, 20 epochs / 160 updates. The fixed training probe covers all 32 training examples. Training uses 26-symbol intermediate CE and shared recurrent LoRA (270,336 trainable parameters). The saved CUDA startup gate passed: T=1 maximum logit error `1.5139579772949219e-05`, shared weights verified, no frozen-parameter gradients, and nonzero gradients at both checked recurrent states.

| Measurement | Initial | Best validation loss, step 32 | Final, step 160 |
| --- | ---: | ---: | ---: |
| Train-probe loss | 3.1665 | 1.2665 | 0.1301 |
| Train-probe trajectory accuracy | 3.125% | 28.125% | 84.375% |
| Validation loss | 3.4673 | 1.4956 | 4.4640 |
| Validation trajectory accuracy | 0% | 31.25% | 25% |
| Validation final-answer accuracy | 0% | 37.5% | 37.5% |

Final validation intermediate accuracy was 45%; nominal loop accuracies were 100%, 0%, 0%, and 50% (denominators 16, 12, 8, and 4). These results show learning on the tiny training set but weak multi-step validation execution and worsening validation loss after the selected checkpoint. Gate 1 is not established. This restricted-symbol recurrent readout is not directly comparable to the ordinary-model three-shot generation baseline.

Recorded wall time: 388.51 seconds; peak CUDA tensor allocation: 2,974,977,536 bytes (2.77 GiB). This is one small run, not a long-unroll resource estimate. The separate WSL exact-resume precision failure remains documented in the [setup report](wsl_cuda_baseline.md).

Small records are eligible for Git: `config.json`, `run.json`, `metrics.jsonl`, `summary.json`, `train-probe-step-*.csv`, `validation-step-*.csv`, `best_checkpoint.json`, `last_checkpoint.json`, and per-checkpoint configuration files. Best points to `step-000032`; last points to `step-000160`. Binary adapter/optimizer state and repeated tokenizer payloads remain ignored. The tracked pointers describe selection; they do not provide a restorable checkpoint without the separately retained binary files and tokenizer.

Validation for this report: read saved configuration, gate, summary, and initial/selected/final metric events; checked pointer consistency and CSV scores. No model inference or optimizer updates were rerun. Diagnose multi-step generalization and the validation-loss regression before scaling the experiment.
