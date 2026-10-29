# LoopFormer

Looped small language models: reuse a block of a pretrained model's middle
layers as one shared recurrent step, and post-train it.

- **v1 (finished):** Qwen2.5-0.5B-Instruct with layers 6–17 as the shared step,
  trained so one loop executes one transition of an in-context pointer-chasing
  task, with a stopping controller. There's a public demo. Results and
  limitations: [docs/history.md](docs/history.md).
- **v2 (planned):** a small looped coding agent, comparing three ways to spend
  test-time compute: depth (loops), breadth (samples) and interaction (running
  code and revising). See [docs/plan.md](docs/plan.md).

## Quick start

```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Docs

| File | Contents |
| --- | --- |
| [docs/plan.md](docs/plan.md) | Current goals, milestones and open decisions |
| [docs/architecture.md](docs/architecture.md) | How the looped model and its training work, from the code |
| [docs/usage.md](docs/usage.md) | Commands for every workflow |
| [docs/setup.md](docs/setup.md) | Environments, the CUDA desktop, dependency files |
| [docs/history.md](docs/history.md) | Log of everything done so far: results, decisions, dead ends |
| [AGENTS.md](AGENTS.md) | Rules for coding agents working here |

Layout: model code in `scripts/recurrent_qwen/`, data in `scripts/dataset/`,
training in `scripts/training/`, evaluation in `scripts/eval/`, demo in
`scripts/demo/` and `demo/`, shell launchers in `scripts/launch/`, configs in
`configs/`, results under `eval/` and `models/`.
