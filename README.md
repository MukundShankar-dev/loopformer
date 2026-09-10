# LoopFormer

This project studies recurrent dynamics in `Qwen/Qwen2.5-0.5B-Instruct`:

*   Can looped transformers learn to repair wrong answers while preserving correct ones?
*   If so, how task-specific is this capability?
*   Additionally, is there a scale at which the looping mechanism stops being useful?

---

## Setup

Use Python 3.11, a local virtual environment, PyTorch, and Hugging Face Transformers. Run these commands from the repository root. See [environment setup](docs/setup.md) for dependency details and platform requirements.

For a Windows desktop with an NVIDIA GPU, follow the [Windows / WSL2 / CUDA setup guide](docs/windows_cuda_setup.md), including dataset and experiment-state migration.

### Create virtual environment

Create and activate the Python venv:

```
python3.11 -m venv .venv
source .venv/bin/activate
```

Then install the required packages:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The environment is gitignored. In each new terminal, run `source .venv/bin/activate` again; use `deactivate` to leave it.

### Download Qwen2.5-0.5B-Instruct

Download the model and tokenizer revision used by the dataset and evaluation scripts:

```
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
```

### Test model inference (smoke test)

Run the smoke test on CPU:

```
python scripts/smoke_test_qwen.py
```

It asks “What color is the sky on a clear day?” and prints the generated answer. If on Apple silicon, use this to test with \`mps\`:

```
python scripts/smoke_test_qwen.py --device mps
```

This checks ordinary model loading and generation. A valid run will output something like this:
```
Loading Qwen/Qwen2.5-0.5B-Instruct (revision=main)
Device: cpu; dtype: float32; use_cache: False
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 659/659 [00:00<00:00, 3.69MB/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.30k/7.30k [00:00<00:00, 23.0MB/s]
vocab.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.78M/2.78M [00:00<00:00, 57.4MB/s]
merges.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.67M/1.67M [00:00<00:00, 54.5MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.03M/7.03M [00:00<00:00, 133MB/s]
model.safetensors: downloading bytes: █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████|  854MB, 31.7MB/s  
model.safetensors: reconstructing file: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████|  988MB /  988MB, 65.7MB/s  
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 612.85it/s]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 242/242 [00:00<00:00, 1.89MB/s]
[transformers] The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Prompt: What color is the sky on a clear day?
Answer: The sky appears blue on a clear day because it reflects sunlight directly into our eyes.
```

## Pointer dataset: preview and reproduce

All dataset code is packaged in [`scripts/dataset`](scripts/dataset/). Start with a Rich terminal preview; these commands write no dataset files:

```bash
.venv/bin/python -m scripts.dataset --seed 17 --dry-run
.venv/bin/python -m scripts.dataset --seed 17 --dry-run 10
```

The five-example preview spans **1, 4, 8, 12, and 16 steps** and prints the full configured distribution. The previous preview took the first rows of each split, which overrepresented one-step examples; the saved dataset itself is balanced.

The existing dataset is `data/pointer/seed-17/`, generated with **master seed 17**:

| Split | Total examples | Steps | Examples at each step count |
| --- | ---: | --- | ---: |
| Training | 10,000 | 1–8 | 1,250 |
| Validation | 1,000 | 1–8 | 125 |
| Test | 1,000 | 1–8 | 125 |
| Deeper test | 1,000 | 9–16 | 125 |

There are 13,000 examples in total: **1,500 at each depth 1–8**, and **125 at each depth 9–16**. Every example has 26 shuffled rules over A–Z, a start symbol, a requested step count, exact per-step answers, answer-token IDs, and its own derived seed. For example, `(A,C) (C,D) (D,E)` from A with two steps has targets C, D and final answer D. Nominal paths have no repeated state.

Generated data is not included in Git. After reviewing the preview, generate it at the location used by evaluation:

```bash
.venv/bin/python -m scripts.dataset \
  --seed 17 \
  --train-count 10000 \
  --validation-count 1000 \
  --test-count 1000 \
  --depth-test-count 1000 \
  --min-depth 1 \
  --max-train-depth 8 \
  --max-eval-depth 16 \
  --output data/pointer/seed-17
```

## Test a model on pointer tasks

First inspect three examples, including the exact model input, response, expected answer, and right/wrong score:

```bash
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct --test
```

Use `--test 2` for two examples. To run the final-answer baseline on the full 1,000-example test set:

```bash
.venv/bin/python -m scripts.eval.naive_test --model Qwen/Qwen2.5-0.5B-Instruct
```

Add `--device mps` for Apple Silicon, or `--limit 8` for a short first run. Models load from local files/cache by default; add `--download` to allow downloads. A saved Hugging Face model directory also works with `--model path/to/model`.

For the configured WSL/NVIDIA environment, run the full test with:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.eval.naive_test \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cuda
```

The [WSL run report](docs/experiments/wsl_cuda_baseline.md) records desktop setup and results. Ubuntu `build-essential` must be installed for the CUDA runtime's Triton compilation.

The evaluation uses a **three-shot prompt**, with solved examples at depths 1, 2, and 3 before each question. Prompt templates live in [`prompts/`](prompts/); edit [pointer_task.txt](prompts/pointer_task.txt) to change the shared task instructions and examples.

The terminal shows progress, accuracy, tokens/s, and questions/s. Results are saved under `eval/pointer_task/<run>/` as `predictions.csv` and `summary.json`. See [evaluation usage](docs/naive_pointer_eval.md) for standalone weight files, the deeper test set, and scoring details.

## Train recurrent pointer execution

Training configs live in [`configs/`](configs/). Preview the small overfit run first, then run it:

```bash
python -m scripts.training.train_pointer --config configs/stage1_pointer_overfit.json --dry-run
python -m scripts.training.train_pointer --config configs/stage1_pointer_overfit.json
```

Add `--device mps` for Apple Silicon. The preview validates data and tokenization without loading model weights or writing files. The overfit config uses 32 examples at depths 1–4; `configs/stage1_pointer.json` configures the larger one-epoch run. Training uses raw dataset prompts and per-loop targets.

A compact Rich dashboard shows progress, ETA, losses, accuracy, gradients, throughput, and memory. Checkpoints and detailed logs go under `models/stage1_pointer/<run>/`. Use a saved step directory with the existing evaluator:

```bash
python -m scripts.eval.naive_test --model models/stage1_pointer/<run>/step-000160 --test
```

See [training usage](docs/training_pointer.md) for configuration, metrics, checkpoint selection, and resume commands. Recurrent evaluation reads the A–Z prediction after the requested number of loops; the ordinary-model three-shot baseline still uses generation.

To evaluate every loop of a saved recurrent checkpoint:

```bash
python -m scripts.eval.loop_test --model models/stage1_pointer/<run>/step-000625 --device cuda --loops 8
```

Add `--test` to inspect three examples first. Results go to `eval/pointer_loops/`. See [full-loop evaluation](docs/loop_pointer_eval.md) for exact checkpoint commands, per-loop losses, trajectory scoring, and output files.

## Project documentation

*   [Documentation index](docs/README.md)
*   [Research plan](docs/project_plan.md)
*   [Architecture and stage gates](docs/phases/stage0_architecture.md)
