---
language: en
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
  - loopformer
  - recurrent-transformer
  - pointer-chasing
  - custom-code
---

# LoopFormer: recurrent pointer execution

Frozen final checkpoint: `controller-shared-number-precision-seed83`.
This is a custom recurrent Qwen model for following a supplied A–Z rule table.
It requires the included Python inference code; ordinary `AutoModelForCausalLM`
generation does not implement its recurrence.

## Download and run

Use Python 3.11 or 3.12. Install a PyTorch build suitable for your device.
The requirements record the validated research dependency versions; the Space
adds Gradio and `spaces` for hosting. Export a graph request from the playground,
or create JSON with `mapping` (26 ordered `[source, destination]` pairs), `start`
(one A–Z letter), and `steps` (a positive integer).

```bash
hf download MODEL_REPO_ID --local-dir loopformer-model
cd loopformer-model
python -m pip install -r requirements.txt
python -m scripts.demo --checkpoint . --request graph.json --device cuda --download
```

The explicit `--download` option permits downloading the pinned Qwen base
revision. It is needed on the first run. `--device cpu` and `--device mps` are
also available, but their latency and final-checkpoint fidelity must be measured
on the selected device. Each output line is a cumulative JSON snapshot of actual
model predictions; the final line includes separate correctness/stopping metrics.

The default requested-step ceiling is 256 and the safety cap is 272. To explore
longer rollouts, explicitly increase both `--max-steps` and `--loop-cap`. This
does not establish graph-execution accuracy beyond the evaluated 256 steps.
The symbol vocabulary remains exactly A–Z; 256 steps does not mean 256 nodes.

## Architecture and inference

The base is `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Six prelude layers prepare the prompt;
twelve shared middle layers update a recurrent working state; six coda layers
read out a distribution over the 26 letters after every loop. A trained bridge
re-enters the shared middle block. Fixed prefix memory is reused within a run.
The final executor adapts all R weights and freezes P/C.

The Steps field is routed away from the executor. A shared learned number reader
initializes a separate scalar timer, whose learned affine recurrence and stop
readout determine completion. Stop probability is a timing signal, not a
correctness estimate. Decoded predictions and reference states are never fed
back into recurrence. Default inference is FP32 with SDPA and threshold 0.5.

The downloadable runtime uses research PyTorch 2.14.0; HF ZeroGPU requires
PyTorch 2.13.0 for the live Space. All 280 checked intermediate predictions,
stopping summaries and stop probabilities matched the audited saved runs on
that hosted runtime. This bounded check is not full-population equivalence.

`adapter_model.pt` contains all adapted executor/controller tensors, not merely
LoRA weights. SHA-256:
`0f8a3f2572ec2caf4fafbcdd074b5283bd8c610602502763b3c04487ce1f123d`.
Optimizer state and training data are excluded from this inference release.

## Evidence and limitations

On the frozen final panel of 1,350 new graphs and requests 1–256:

- Final letter plus exact stopping: 97.81% over 345,600 graph/count queries.
- Complete trajectories through depth 256: 96.89% (1,308/1,350 graphs).
- Exact stopping: 100% across that finite requested-count panel.

Executor supervision used depths 1–6, 8, 10, 12. Controller fitting used
counts 1–63 excluding 9, 17, 29, 41, 53, with supervised unrolls capped at 12.
Controller-only numerical results at longer horizons are distinct from executing
graphs correctly. This is one task family, not general reasoning or transfer
evidence. Arbitrary hand-edited graphs can fail. Changes to precision, runtime,
hardware or kernels need validation against native intermediate readouts.

See the [source and full evidence](https://github.com/MukundShankar-dev/loopformer)
and the [interactive playground](https://mukundshankar-dev.github.io/projects/loopformer/demo/).
