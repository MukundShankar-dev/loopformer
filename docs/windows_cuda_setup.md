# Windows desktop setup: RTX 5070 Ti

Use Windows with Ubuntu in **WSL2**, and run the repository with Linux Python and CUDA inside Ubuntu. This guide targets the desktop RTX 5070 Ti with 16 GB VRAM and 32 GB system RAM. The current scripts use Unix resource logging, so native Windows Python is not the supported route here.

This guide was checked against the repository and official installation documentation on 2026-09-10. It has not been executed on the desktop. The Mac's 87 passing tests and CPU/MPS results do not establish CUDA correctness on another device.

## 1. What matching the Mac means

| Component | State to reproduce |
| --- | --- |
| Repository | Clone the pushed `main` branch, including `configs/`, `prompts/`, training/evaluation scripts, tests, and documentation |
| Python | 3.11.8, in a new Linux `.venv` |
| Dependencies | The seven pins in `requirements.txt`; PyTorch uses its CUDA build instead of the Mac build |
| Base model/tokenizer | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| Data | The seed-17 pointer dataset: 10,000 train, 1,000 validation, 1,000 test, 1,000 deeper-test examples |
| Baseline | Preserve the Mac's completed three-shot run: 60/1,000 correct, or 6.00% |
| Training | Trainer implemented; no pretrained training has started and no trained research checkpoint needs migration |

Git brings the code and Markdown reports. `data/`, `artifacts/`, `eval/pointer_task/`, `models/`, and `.venv/` are ignored. Data can be copied or reproduced; historical measurements must be copied to retain the original evidence. Create a fresh environment and download the pinned model on Linux. Do not copy the Mac `.venv` or install its entire transitive package list as a Linux lockfile.

Matching means the same code, inputs, model revision, prompt, and experimental settings. CPU/MPS/CUDA floating-point results and timings need not be bitwise identical. PyTorch also limits reproducibility guarantees across platforms and releases. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

## 2. Prepare Windows and install WSL2

Use an up-to-date Windows 11 installation. Install the current compatible **Windows NVIDIA driver** for the 5070 Ti through the NVIDIA App or [NVIDIA driver downloads](https://www.nvidia.com/en-us/drivers/), then reboot if requested. NVIDIA's WSL support exposes the Windows driver to Linux; do not install a separate Linux NVIDIA display driver inside Ubuntu. This suite uses packaged PyTorch CUDA libraries and does not require a separate CUDA Toolkit or Docker installation. [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows when requested. Open Ubuntu from the Start menu and create its Linux username/password; the password is used for `sudo` and need not match your Windows password.

Back in **PowerShell**:

```powershell
wsl --update
wsl --list --verbose
```

Expect `Ubuntu-24.04` with `VERSION 2`. If it is version 1:

```powershell
wsl --set-version Ubuntu-24.04 2
```

If Ubuntu is already installed, inspect the distribution list first and use its actual name in subsequent WSL commands. The instructions below assume `Ubuntu-24.04`. [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install)

Allow roughly 40–60 GB of free SSD space as working headroom for Ubuntu, Python/CUDA packages, model downloads, and future checkpoints. This is a planning allowance, not the current repository size.

### Optional WSL memory ceiling

WSL normally caps its VM at half of Windows RAM, approximately 16 GB here. That should be enough for the initial setup. To allow more host-memory headroom while leaving 8 GB for Windows, open this file in PowerShell:

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

Create or merge these settings; preserve any unrelated existing settings:

```ini
[wsl2]
memory=24GB
swap=8GB
```

Before training starts, apply changes with:

```powershell
wsl --shutdown
```

Then reopen Ubuntu. Shutdown stops all WSL sessions, so do not do it during a run. The setting controls system RAM; it does not increase the card's separate 16 GB VRAM. [Microsoft WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

## 3. Prepare Ubuntu and clone the repository

All remaining commands are **Ubuntu Bash**, unless marked otherwise.

```bash
sudo apt update
sudo apt install -y git curl ca-certificates tmux
nvidia-smi
```

`nvidia-smi` should show the RTX 5070 Ti and a driver version. If it is not on PATH, try `/usr/lib/wsl/lib/nvidia-smi`. If neither works, fix the Windows driver/WSL setup before installing or running models.

Clone into the Linux filesystem:

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/MukundShankar-dev/loopformer.git
cd loopformer
git rev-parse HEAD
git status --short
```

Compare `HEAD` with the commit reported by the Mac after the push. A fresh clone should have a clean status. For later updates, stop work on the checkout, inspect `git status`, then use `git pull --ff-only`; do not overwrite uncommitted work to force an update.

Keep the working repository at `~/src/loopformer`, rather than running it from `/mnt/c/...`. Microsoft recommends Linux filesystem storage for workloads run in WSL. To view it in Windows Explorer, run `explorer.exe .` from Ubuntu. [WSL filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)

## 4. Install Python 3.11.8 and the CUDA environment

Ubuntu's default Python may differ from the project version. Use uv only to install the matching interpreter and create the venv; the project's dependencies remain managed by pip and `requirements.txt`.

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/loopformer-uv-install.sh
sh /tmp/loopformer-uv-install.sh
source "$HOME/.local/bin/env"
uv python install 3.11.8
uv venv --python 3.11.8 --seed .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
```

Expect Python 3.11.8. The uv installer prints its actual installation location if your shell uses a customized location. [uv installation](https://docs.astral.sh/uv/getting-started/installation/), [managed Python versions](https://docs.astral.sh/uv/guides/install-python/)

Install the explicit CUDA 13.0 PyTorch wheel first, then the remaining repository requirements:

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
```

The [official CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/) lists `torch-2.14.0+cu130` for Python 3.11 Linux x86-64. The `+cu130` suffix is expected; it satisfies the repository's `torch==2.14.0` requirement. Do not replace repository pins with arbitrary latest versions. A recent compatible Windows driver is required; PyTorch's Blackwell guidance specifies CUDA 13.0+ wheels and Windows driver 580.88 or newer for that runtime. Prefer a current supported driver over installing that old minimum. [PyTorch CUDA/Blackwell guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/)

The seven direct versions should match the Mac: torch 2.14.0 (CUDA build), transformers 5.17.0, tokenizers 0.23.2, huggingface-hub 1.31.0, peft 0.20.0, rich 15.0.0, pytest 9.1.1. CUDA-specific dependencies will differ. No torchvision, torchaudio, or development CUDA toolkit is needed by this suite.

## 5. Verify real GPU computation and determinism

The scripts enable strict deterministic algorithms. Set cuBLAS's workspace configuration **before launching Python**, including evaluation and training:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Repeat this in each new training terminal, or add that exact export line once to your Ubuntu `~/.bashrc`. This is the cuBLAS setting required by certain CUDA matrix operations under PyTorch's deterministic mode. [PyTorch deterministic algorithms](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html)

Check both device discovery and an actual forward/backward calculation:

```bash
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is unavailable; check the driver and PyTorch wheel'
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
torch.manual_seed(17)
torch.use_deterministic_algorithms(True)
x = torch.randn(256, 256, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('PASS: deterministic CUDA matrix forward/backward')
PY
```

Expect `2.14.0+cu130`, runtime `13.0`, the RTX 5070 Ti, and approximately 16 GiB VRAM. A successful `nvidia-smi` alone does not establish that the installed PyTorch can execute kernels on this card.

## 6. Download the exact base model and tokenizer

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
```

The public model does not require credentials. Download to Ubuntu's default Hugging Face cache; subsequent project commands use cached/local files by default. Keep this revision fixed for the first experiment. No trained checkpoint needs downloading from the Mac because pretrained training has not begun.

## 7. Bring across the dataset and historical evidence

There are two routes. Use **A** to preserve the original data manifest and experiment files exactly. Use **B** if you want to regenerate the dataset and only need the committed experiment reports for now.

### A. Copy the Mac artifact bundle

The setup handoff includes a Git-ignored `artifacts/handoff/mac-artifacts-20260910.tar.gz` on the Mac, with a companion `.sha256` file. It contains:

- `data/pointer/seed-17/`, including the original manifest;
- `artifacts/stage0/`, with the original CPU/MPS evidence;
- the completed three-shot run under `eval/pointer_task/`;
- `artifacts/handoff/mac-environment.json`, recording the Mac's Python and installed package versions for comparison.

Move both bundle files to your Windows Downloads directory using your preferred file-transfer method. No Git release upload is required. From Ubuntu, replace `YOUR_WINDOWS_USER` with the Windows profile directory name:

```bash
cd /mnt/c/Users/YOUR_WINDOWS_USER/Downloads
sha256sum -c mac-artifacts-20260910.tar.gz.sha256
cd ~/src/loopformer
tar -xzf /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/mac-artifacts-20260910.tar.gz
```

Extract into a fresh clone before generating data or creating same-named results. This archive restores the original relative paths. The old absolute Mac paths in historical JSON are provenance; leave them unchanged. Verify the restored dataset:

```bash
python -m scripts.dataset --verify data/pointer/seed-17
```

Original baseline directory:

```text
eval/pointer_task/20260910T172432.074426Z-Qwen-Qwen2.5-0.5B-Instruct/
```

It contains the 60/1,000 result, not a Windows measurement. The [baseline report](experiments/naive_pointer_baseline.md) records its artifact checksums. No `models/` checkpoints or Mac virtual environment are included in this handoff.

### B. Reproduce seed 17 locally

Start with a terminal preview:

```bash
python -m scripts.dataset --seed 17 --dry-run
```

Then generate into the location expected by training and evaluation:

```bash
python -m scripts.dataset \
  --seed 17 \
  --train-count 10000 \
  --validation-count 1000 \
  --test-count 1000 \
  --depth-test-count 1000 \
  --min-depth 1 \
  --max-train-depth 8 \
  --max-eval-depth 16 \
  --output data/pointer/seed-17
python -m scripts.dataset --verify data/pointer/seed-17
sha256sum data/pointer/seed-17/*.jsonl
```

Compare the four JSONL checksums with the table in the [data validation report](experiments/stage1_data_validation.md). Matching hashes establish byte-identical dataset files. The new manifest correctly records this Linux generation's command, environment, and Git commit, so it will differ from the original Mac manifest. Do not regenerate over a copied dataset: existing directories are deliberately rejected.

## 8. Run the suite and inspect evaluation

With the pinned tokenizer cached:

```bash
python -m pytest -q
```

The current expected result is **87 passing tests**, with no cache-related skips. These are predominantly CPU/tiny-model implementation checks; they do not certify pretrained CUDA training. Save the actual desktop result if it differs.

The standalone architecture CLI currently accepts **CPU and MPS only**. On WSL you can independently run its CPU gate:

```bash
python -m scripts.validate_stage0 --device cpu
```

Its default timestamped output preserves the historical copied CPU JSON. Do not pass `--device cuda` to this CLI. The training command has its own checks on the selected CUDA device: fresh T=1 equivalence, shared weights, and a two-loop gradient path, before its first optimizer update.

Inspect three ordinary-model answers on the GPU:

```bash
python -m scripts.eval.naive_test \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --test
```

This uses the same three-shot prompt in `prompts/pointer_task.txt`. Poor pointer accuracy is not an installation failure; the first three Mac examples were also wrong. Successful loading, finite computation, scoring, and artifact creation are the immediate checks. A full rerun is optional:

```bash
python -m scripts.eval.naive_test \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cuda
```

New outputs go into a new timestamped directory under `eval/pointer_task/`. Preserve the original 6.00% result separately; cross-device decoding need not match exactly.

## 9. Preview and start the first CUDA training run

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_overfit.json --device cuda --dry-run
```

Expect 32 training examples at depths 1–4, 16 validation examples, a 32-example fixed training probe, and 160 planned optimizer updates. This validates data/tokenization and prints the budget; it does not test GPU memory or execute training.

For a run that survives closing the terminal, start a tmux session:

```bash
tmux new -s pointer
cd ~/src/loopformer
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_overfit.json --device cuda
```

Detach with **Ctrl-B, then D**. Return with `tmux attach -t pointer`. tmux does not survive a Windows restart, WSL shutdown, or keep the machine training during sleep; set Windows power options so the desktop stays awake while plugged in.

Look for the architecture/loss-path checks passing, initial validation progressing, then optimizer updates. Rich shows ETA and current metrics; full events and per-loop trajectories are saved alongside checkpoints under `models/stage1_pointer/<run>/`. Inspect fixed train-probe trajectory accuracy and per-loop losses to see whether the small set is being learned. Check actual GPU memory and utilization from another Ubuntu terminal with:

```bash
nvidia-smi -l 2
```

Keep the first run at float32, batch 1, and the configured short unrolls. Measure it before increasing depth or batch size. Once the overfit behavior and resource use are understood, start a fresh larger run:

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer.json --device cuda
```

That configuration selects 5,000 depth-1–4 training examples for one epoch, or 625 optimizer updates. It is not one epoch over all 10,000 dataset rows. Success is judged from per-loop and held-out behavior, not merely finishing the budget. See [training usage](training_pointer.md) for loss, selection, and logging definitions.

## 10. Evaluate, resume, and bring results back

Inspect the actual run's `last_checkpoint.json` or `best_checkpoint.json`, then supply the named **step directory**:

```bash
python -m scripts.eval.naive_test \
  --model models/stage1_pointer/RUN_NAME/step-000160 --device cuda --test
```

`RUN_NAME` and the step number are placeholders; use the saved path. Omit `--test` for the full test set. Recurrent checkpoints use raw prompts and A–Z readout at the requested loop count, distinct from ordinary three-shot generation.

To resume an interrupted desktop overfit run, use a completed checkpoint and keep `--device cuda`:

```bash
python -m scripts.training.train_pointer \
  --config configs/stage1_pointer_overfit.json --device cuda \
  --resume models/stage1_pointer/RUN_NAME/step-000080
```

A new output directory is created. The configured budget is the total budget, not additional epochs. Same-device resume checks model/data/tokenizer/optimization identity. Exact CPU/MPS-to-CUDA optimizer resume is intentionally rejected; evaluation of a checkpoint on another device is supported. There is currently no Mac-trained state to resume.

To bring a run back to the Mac, copy the entire run directory under `models/stage1_pointer/` and relevant `eval/pointer_task/` outputs. The Mac needs the same code and cached pinned base model. Git carries code/config/docs changes; it does not carry generated weights or results. Avoid concurrent edits to the same files on both machines; commit and push a coherent change from one, then pull it on the other.

## 11. Troubleshooting and completion check

| Symptom | Action |
| --- | --- |
| `nvidia-smi` fails in Ubuntu | Check the Windows NVIDIA driver, WSL2 version, and `wsl --update`; use `/usr/lib/wsl/lib/nvidia-smi` if only PATH is missing |
| `torch.cuda.is_available()` is false | Confirm Ubuntu `.venv` is active and the installed torch is the CUDA wheel, not a Windows/Mac/CPU environment |
| `no kernel image` / unsupported `sm_120` | Recheck the official cu130 wheel and current compatible Windows driver; do not ignore the warning |
| cuBLAS deterministic-mode error | Export `CUBLAS_WORKSPACE_CONFIG=:4096:8` in that terminal before starting Python |
| Missing cached tokenizer/model | Run the pinned `hf download` command inside Ubuntu, with the same user/cache used for the project |
| Dataset directory already exists | Verify the existing copied/generated data; use one data setup route rather than overwriting it |
| `resource` import failure | Run Linux Python inside WSL, rather than native Windows Python |
| CUDA memory exhaustion | Keep batch 1 and depth 4, close other GPU-heavy work, inspect `nvidia-smi`; do not assume system RAM extends VRAM |
| Gate failure / non-finite logits or gradients | Preserve the error and config and investigate before training; do not silently relax tolerances or disable determinism |

The desktop is caught up when the pushed code is present, pinned packages/model are installed, dataset verification and all 87 tests pass, historical artifacts are restored if wanted, and CUDA matrix forward/backward works. The first pretrained CUDA training run is still a new experiment, with its own startup gate and measurements.
