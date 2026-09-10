# WSL CUDA setup and instruct baseline — 2026-09-10

## Scope and environment

User requested completing the WSL environment, reproducing the README dataset, downloading the required model, and running the ordinary instruct-model baseline on all 1,000 test examples. No pretrained recurrent training is part of this run.

Code revision: `8ff477a9bade8f6a83106da5fb635b48176d4595`. Ubuntu runs under WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`, with an NVIDIA GeForce RTX 5070 Ti (16 GB), NVIDIA-SMI driver `610.43.02` / Windows KMD `610.47`. The repository-local environment uses Python 3.11.8 and all seven direct requirement pins, including `torch==2.14.0+cu130`. Torch reports CUDA 13.0 and compiled `sm_120` support.

Executed installation commands:

```bash
.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
```

`pip check` passed. The pinned model/tokenizer are cached under `/home/mukund/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`. This instruct checkpoint is the only model required by this task.

A float32 CUDA matrix forward/backward check with seed 17, strict deterministic algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` passed finite-output and finite-gradient assertions. Machine-readable evidence and the complete resolved package list are in `artifacts/wsl_setup_20260910/cuda.json` and `requirements-resolved.txt`.

## Dataset reproduction

```bash
.venv/bin/python -m scripts.dataset --seed 17 \
  --train-count 10000 --validation-count 1000 --test-count 1000 \
  --depth-test-count 1000 --min-depth 1 --max-train-depth 8 \
  --max-eval-depth 16 --output data/pointer/seed-17
```

Generation and integrated read-back verification passed: 13,000 examples, exact targets, tokenizer contexts, disjoint whole rule tables, split boundaries, and seed replay. All four JSONL SHA-256 hashes match the [original data report](stage1_data_validation.md). The manifest records this new run's provenance. Terminal evidence: `artifacts/wsl_setup_20260910/dataset-generation.txt`.

## Validation and setup issues

`.venv/bin/python -m pytest -q` completed with **86 passed, one failed** in 33.59 seconds. All ordinary-model baseline tests passed. The failure is `test_resume_matches_uninterrupted_updates_including_short_tail`: an exact-equality assertion found a maximum absolute parameter difference of `1.4901161193847656e-08`. This is a training-resume reproducibility limitation on the new environment; no tolerance was relaxed. Full output: `artifacts/wsl_setup_20260910/pytest.txt`.

The same test also failed in isolation (`.venv/bin/python -m pytest tests/test_pointer_training.py::test_resume_matches_uninterrupted_updates_including_short_tail -q`); evidence: `artifacts/wsl_setup_20260910/resume-recheck.txt`.

The sandbox blocks GPU access and package-download networking. Installation, model download, and CUDA checks needed execution outside it. The first full evaluation attempt loaded the model but failed before answering any examples because Triton could not find a C compiler. Its incomplete output is preserved at `eval/pointer_task/20260910-wsl-cuda-instruct/`; it is not a baseline result. Traceback: `artifacts/wsl_setup_20260910/naive-evaluation.txt`. The user then installed Ubuntu `build-essential` (12.10ubuntu1); GCC 13.3.0 was confirmed before retrying. The initial automated sudo attempt could not proceed because it required the user's local password.

## Evaluation result

The fresh full evaluation completed successfully with **60/1,000 correct (6.00%)**, matching the [historical baseline](naive_pointer_baseline.md) in overall and per-depth accuracy. The new predictions CSV also matches the historical report's recorded SHA-256 exactly, providing evidence of byte-identical saved predictions. The original CSV itself was not present in this checkout for a direct file comparison.

Exact successful command (terminal output redirected to `artifacts/wsl_setup_20260910/naive-evaluation-retry.txt`):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.eval.naive_test \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cuda \
  --output eval/pointer_task/20260910-wsl-cuda-instruct-retry
```

Settings: pinned revision above, float32, batch 1, seed 17, four CPU threads, the existing three-shot chat prompt, greedy decoding, eight-token budget, strict deterministic algorithms, `use_cache=False`, and exact uppercase-symbol scoring. No evaluation code, prompt, or dependency pin was changed.

| Depth | Correct / 125 | Accuracy |
| --- | ---: | ---: |
| 1 | 22 | 17.60% |
| 2 | 6 | 4.80% |
| 3 | 5 | 4.00% |
| 4 | 5 | 4.00% |
| 5 | 5 | 4.00% |
| 6 | 7 | 5.60% |
| 7 | 5 | 4.00% |
| 8 | 5 | 4.00% |

There were three invalid answers and three token-budget stops. Evaluation took 84.27 seconds (83.05 seconds in generation), at 11.87 questions/s and 24.30 generated tokens/s. Loading took 1.80 seconds. These timings exclude installation/downloads; peak GPU memory was not captured. Transformers emitted the existing warning about ignored sampling flags; the evaluator explicitly uses greedy decoding.

Results: `eval/pointer_task/20260910-wsl-cuda-instruct-retry/predictions.csv` and `summary.json`. An independent read-back audit verified complete status, 1,000 unique examples, 125 at every depth, each row's strict score, total correct, and all per-depth counts. Audit evidence is `artifacts/wsl_setup_20260910/baseline-audit.json`.

- Summary SHA-256: `4d030b2620af0ce6444cb3b3235130bc251148064f3aa1e79067b6d7b6c8b6ff`.
- Predictions SHA-256: `4d82eecd681273b2129da9e3dd0b2697fd5546acadeb15c694a4100cde3e9eb1`.

This establishes functioning pretrained CUDA generation and reproduces the weak ordinary-model pointer baseline. It does not establish recurrent CUDA equivalence, exact training resume on WSL, learned one-loop transitions, or any training result. The requested setup/data/baseline work is complete; the exact-resume test remains a documented issue before relying on training reproducibility.
