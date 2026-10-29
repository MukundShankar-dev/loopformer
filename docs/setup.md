# Setup

Use Python 3.11 (3.11.8 is recorded) in a repository-local `.venv`. Both `.venv/` and `venv/` are gitignored. Commands are in [usage](usage.md).

## Mac (Apple Silicon, CPU/MPS)

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

[requirements.txt](../requirements.txt) pins the direct dependencies: torch 2.14.0, transformers 5.17.0, tokenizers 0.23.2 (this version reproduces the dataset tokenization), huggingface-hub 1.31.0, peft 0.20.0, numpy 2.4.6, rich 15.0.0, pytest 9.1.1, wandb 0.30.0 and matplotlib 3.11.2. Transitive dependencies are not locked. The torch 2.14.0 wheel for macOS needs macOS 14 or later.

Most CUDA launchers (`train_*`, `benchmark_pointer.sh`, `mmlu_retention.sh`, …) assume Linux. On the Mac, use the Python CLIs with `--device cpu` or `--device mps`. A device is never chosen for you, and requesting an unavailable one is an error.

## Desktop: WSL2 + CUDA (RTX 5070 Ti, 16 GB VRAM)

Run Ubuntu (24.04) under WSL2, keep the clone on the Linux filesystem (for example `~/src/loopformer`, not `/mnt/c`), and install the Windows NVIDIA driver only. Do not install a Linux display driver or a CUDA toolkit. Then run:

```bash
sudo apt install -y git curl ca-certificates tmux build-essential   # GCC is required: Triton compiles a Qwen RoPE kernel
nvidia-smi
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
uv python install 3.11.8 && uv venv --python 3.11.8 --seed .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
```

The `+cu130` build satisfies the `torch==2.14.0` pin. Blackwell GPUs need CUDA 13 wheels and a current Windows driver. Launchers that write logs through a PTY need util-linux `script` (installed by default on Ubuntu). `mmlu_retention.sh` and `trace_baselines.sh` also need `flock`.

WSL limits its VM to about half of host RAM, roughly 16 GB on the 32 GB desktop. If you need more, edit `%UserProfile%\.wslconfig` and then run `wsl --shutdown` from PowerShell. Shutdown stops every WSL session, so do not run it during a job.

```ini
[wsl2]
memory=24GB
swap=8GB
```

Copy checkpoints between machines yourself: `*.pt` files and repeated tokenizer payloads under `models/` are gitignored. Regenerate `data/` with `python -m scripts.dataset` (see [usage](usage.md#dataset)). Results from CPU, MPS and CUDA are not bitwise identical.

## Optional requirement files

Install each one with `--constraint requirements.txt` so the research pins are kept:

| File | Contents | Needed for |
| --- | --- | --- |
| [requirements-mmlu.txt](../requirements-mmlu.txt) | lm_eval 0.4.13, datasets 5.1.0 | `scripts.eval.mmlu`, `mmlu_analysis` |
| [requirements-demo.txt](../requirements-demo.txt) | gradio 6.30.0, gradio-client 2.7.2, fastapi 0.143.0, uvicorn 0.54.0 | demo proxy and verification, `tests/test_demo_proxy.py` |
| [demo/requirements-inference.txt](../demo/requirements-inference.txt) | the research inference pins | published model repo (`requirements.txt` there) |
| [demo/hf_space/requirements.txt](../demo/hf_space/requirements.txt) | **torch 2.13.0** (the ZeroGPU limit), plus gradio, `spaces`, websockets | Hugging Face Space (Python 3.12) |

The v2 headroom probe uses its own environment, `.venv-probe` (gitignored), not the research one, because EvalPlus and LiveCodeBench pin their own torch/vLLM stacks. Install steps are at the top of [requirements-probe.txt](../requirements-probe.txt). Pin exact versions there on first install.

## Model files

The base is `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`. The model is public, so no login is needed. Download it once:

```sh
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
python scripts/smoke_test_qwen.py --local-files-only --revision 7ae557604adf67be50417f59c2c2f167def9a775   # optional generation check
```

Research CLIs load with `local_files_only=True` and fail if files are missing. Pass `--download` explicitly where a CLI supports it (`train_pointer`, `validate_stage0`, `loop_test`, `scripts.demo`). The dataset CLI, MMLU and controller tools are always local-only. Files are cached in `~/.cache/huggingface/hub`, or under `HF_HOME` if set.

## Determinism and environment variables

| Setting | Where |
| --- | --- |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Exported by the launchers. `scripts.demo` and the Space set it with `setdefault`. It is required for deterministic CUDA matmuls when `torch.use_deterministic_algorithms(True)` is active. |
| `torch.use_deterministic_algorithms(config.deterministic)` | `train_pointer`. `executor_r.json` sets `deterministic: false` for BF16/SDPA speed. |
| `torch.use_deterministic_algorithms(True)` | `frozen_pointer_benchmark`, `scripts.demo`, the Space |
| TF32 disabled (`allow_tf32 = False`) | `scripts.demo`, `verify_checkpoint`, `mmlu`, `paper_baseline`, the Space |
| `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=4` | `trace_baselines.sh` |
| `MPLCONFIGDIR` | `analyze_pointer.sh` (`$TMPDIR/loopformer-paper-mpl`) |
| `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_NAME` | Launchers default `WANDB_MODE=online` and the CLIs default to `disabled`. Use `WANDB_MODE=disabled bash scripts/launch/…` to run offline. |
| `EXECUTOR_CONFIG`, `EXECUTOR_RUN` | `train_executor.sh` overrides |
| `LOOPFORMER_MODEL_ID`, `LOOPFORMER_MODEL_REVISION`, `LOOPFORMER_MAX_STEPS`, `LOOPFORMER_PROXY_TOKEN` | Space runtime (set by `scripts.demo.deploy`, except the proxy token) |
