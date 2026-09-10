# Python environment

Use Python 3.11 and a repository-local virtual environment. From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Use `deactivate` to leave the environment, and `source .venv/bin/activate` to return. Both `.venv/` and `venv/` are gitignored. Commit dependency changes to [requirements.txt](../requirements.txt), not the environment directory.

## Inference dependencies

The initial target is [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

| Package | Purpose |
| --- | --- |
| `torch` | Tensor operations and model execution on CPU or Apple Silicon MPS |
| `transformers` | Qwen model classes, tokenizer, and generation |
| `huggingface-hub` | Model/tokenizer downloads and explicit Hub access |

Transformers installs its tokenizer and Safetensors dependencies automatically. PEFT will be added when implementing recurrent LoRA. Accelerate is unnecessary for basic inference with explicit device placement; `device_map="auto"` is outside this minimal setup.

## Download and inference

After activating the environment and installing requirements, run:

```sh
python scripts/smoke_test_qwen.py
```

The [smoke test](../scripts/smoke_test_qwen.py) downloads any missing model/tokenizer files through `from_pretrained`, loads the instruct-tuned checkpoint, and asks “What color is the sky on a clear day?” using the tokenizer's chat template. It prints the model, requested revision, device, dtype, prompt, and generated answer. Expect a short answer describing a blue sky; exact wording is not asserted.

The default is CPU with float32 weights. Generation uses evaluation mode, disabled gradients, greedy decoding, at most 32 new tokens, and `use_cache=False`. It fails if no visible answer is generated. This is a basic inference check, not recurrent equivalence or a research accuracy result.

To select Apple Silicon MPS explicitly, or change the prompt:

```sh
python scripts/smoke_test_qwen.py --device mps
python scripts/smoke_test_qwen.py --prompt "What is 2 + 2?" --max-new-tokens 16
```

The script reports an error if MPS is requested but unavailable; it does not select another device automatically. See [architecture](architecture.md) for subsequent recurrent work.

You can download the checkpoint separately using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
hf download Qwen/Qwen2.5-0.5B-Instruct
python scripts/smoke_test_qwen.py --local-files-only
```

The model is public and ungated, so credentials are optional. If you want authenticated downloads, run `hf auth login`. Files are cached under `~/.cache/huggingface/hub` by default (environment overrides such as `HF_HOME` can change this); model weights do not belong in Git. The script reuses cached files, and `--local-files-only` prevents model/tokenizer downloads and fails if required files are missing.

The default revision is `main`, which can change. For repeatable runs, pass the same model commit hash to `hf download Qwen/Qwen2.5-0.5B-Instruct --revision COMMIT_HASH` and the script's `--revision COMMIT_HASH` option. Record resolved package versions as well.

## Version and validation status

Direct dependencies are pinned to give collaborators a common starting point. This is not a full environment lock: transitive dependencies can vary. Record the resolved package versions and model revision for experiments.

Package versions and Python requirements were checked against PyPI metadata for [PyTorch](https://pypi.org/project/torch/2.14.0/), [Transformers](https://pypi.org/project/transformers/5.17.0/), and [Hugging Face Hub](https://pypi.org/project/huggingface-hub/1.31.0/). The pinned PyTorch release provides a Python 3.11 Apple Silicon wheel requiring macOS 14 or later.

The script's Python syntax, CLI help, and invalid-argument handling have been checked without loading ML dependencies. Setup and inference commands above are instructions, not completed runtime validation. Dependencies have not been installed for this project, and model download, inference, MPS behavior, and recurrent equivalence remain unverified. See [status](status.md) for the next milestone.
