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

## Inference, Stage 0, and Stage 1 data dependencies

The initial target is [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

| Package | Purpose |
| --- | --- |
| `torch` | Tensor operations and model execution on CPU or Apple Silicon MPS |
| `transformers` | Qwen model classes, tokenizer, and generation |
| `tokenizers` | Explicitly pinned to 0.23.2 to reproduce the verified pointer dataset's tokenization |
| `huggingface-hub` | Model/tokenizer downloads and explicit Hub access |
| `peft` | LoRA injection into the shared recurrent layers |
| `rich` | Readable terminal configuration, measurements, and pass/fail tables |
| `pytest` | Focused architecture and metric contract tests |

Transformers uses tokenizers; its version is now explicitly pinned in requirements rather than left to transitive resolution. Safetensors remains transitive. PEFT installs Accelerate transitively; model placement remains explicit and does not use `device_map="auto"`.

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

Stage 1 data reproduction also pins `tokenizers==0.23.2`, the version already installed when seed 17 was generated and verified. All seven requirement pins match the local Python 3.11.8 environment and `pip check` passes; adding this pin requires no local package changes. See the [README reproduction instructions](../README.md#pointer-dataset-preview-and-reproduce).

Package versions and Python requirements were checked against PyPI metadata for [PyTorch](https://pypi.org/project/torch/2.14.0/), [Transformers](https://pypi.org/project/transformers/5.17.0/), and [Hugging Face Hub](https://pypi.org/project/huggingface-hub/1.31.0/). The pinned PyTorch release provides a Python 3.11 Apple Silicon wheel requiring macOS 14 or later.

Environment audit on 2026-09-10 confirmed `.venv` uses Python 3.11.8 with torch 2.14.0, transformers 5.17.0, and huggingface-hub 1.31.0, matching the direct pins. Running `.venv/bin/python -m pip check` returned `No broken requirements found.` The virtual environment is ignored and contains no Git-tracked files.

The script's Python syntax, CLI help, and invalid-argument handling were checked previously. The [README](../README.md#test-model-inference-smoke-test) now includes CPU generation output showing model loading and a nonempty answer. This is recorded output, not an independently repeated inference run during this audit; the transcript does not capture an immutable model revision or complete environment. It also contains a warning about ignored generation settings, which remains undiagnosed. Treat the answer as a loading/generation example, not a factual-accuracy benchmark.

Stage 0 added peft 0.20.0, rich 15.0.0, and pytest 9.1.1 to this venv and the direct pins. Installation from `requirements.txt` succeeds and `pip check` passes. The recurrent suite has 36 passing tests. Pretrained Stage 0 validation now passes on CPU and MPS; the ordinary generation smoke test itself was not rerun. See the [Stage 0 report](experiments/stage0_validation.md) for exact evidence, including the MPS deterministic-readout fix.

## Recurrent surgery and validation

From the repository root with the venv active:

```sh
python -m pytest -q
python -m scripts.validate_stage0 --device cpu --output artifacts/stage0/cpu.json
python -m scripts.validate_stage0 --device mps --output artifacts/stage0/mps.json
```

The [CLI](../scripts/validate_stage0.py) composes the implementation in `scripts/recurrent_qwen/`. It creates the configurable P/R/C split, freezes original parameters, attaches recurrent LoRA, and verifies the architecture. It prints Rich tables and saves JSON with exact inputs, all package versions, source hashes, model revision, gradient evidence, timing, and peak process RSS.

Unlike the older smoke test, this command defaults to **local files only** at model revision `7ae557604adf67be50417f59c2c2f167def9a775`. Add `--download` explicitly when files are missing. No model was downloaded during the recorded Stage 0 validation. MPS may be hidden by an execution sandbox; the recorded MPS checks ran outside it. Device fallback is never automatic.

See [Stage 0](phases/stage0_architecture.md) for options, contracts, and the stop boundary before pointer data and training.
