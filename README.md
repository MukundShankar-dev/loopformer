# LoopFormer

This project studies recurrent dynamics in `Qwen/Qwen2.5-0.5B-Instruct`:

*   Can looped transformers learn to repair wrong answers while preserving correct ones?
*   If so, how task-specific is this capability?
*   Additionally, is there a scale at which the looping mechanism stops being useful?

---

## Setup

Use Python 3.11, a local virtual environment, PyTorch, and Hugging Face Transformers. Run these commands from the repository root. See [environment setup](docs/setup.md) for dependency details and platform requirements.

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

The smoke test (next step) downloads the model and tokenizer automatically on its first run. To download them separately first:

```
hf download Qwen/Qwen2.5-0.5B-Instruct
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

After reviewing the preview, this fully explicit command reproduces the **exact four JSONL files** in a new directory:

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
  --output data/pointer/seed-17-reproduced
```

## Project documentation

*   [Documentation index](docs/README.md)
*   [Research plan](docs/project_plan.md)
*   [Architecture and stage gates](docs/phases/stage0_architecture.md)
