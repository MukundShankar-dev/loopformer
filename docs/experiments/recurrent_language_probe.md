# Recurrent checkpoint language probe — 2026-10-12

The user requested an ordinary question to the published recurrent pointer
checkpoint. This is an exploratory inference probe, with no fitting or broader
capability benchmark. The subsequent user-authorized MMLU comparison is a
separate evaluation.

## Configuration and execution

Host: SSH `desktop`, `/home/mukund/loopformer`, RTX 5070 Ti. Reuse
`models/stage1_pointer/controller-shared-number-precision-seed83/best`.
Its adapter SHA-256 matches the [published model card](../demo_model_card.md):
`0f8a3f2572ec2caf4fafbcdd074b5283bd8c610602502763b3c04487ce1f123d`.
Pinned base revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
Python 3.11.8, Torch 2.14.0+cu130, Transformers 5.17.0.

Question: exactly `What color is the sky?`, as the sole user message in the
tokenizer's chat template. Use FP32/SDPA, greedy decoding, disabled KV caching,
TF32 disabled, deterministic algorithms, and at most 32 generated tokens.
Compare one, two, and four recurrent passes per next-token prediction. Preserve
the trained recurrent block and bridge. Append predicted token IDs directly;
do not decode and re-encode the growing input. Stop on tokenizer EOS.

The native controller rejects this input because it requires a raw pointer
prompt containing Rules/Start/Steps/Answer. For this probe, explicitly bypass
the router and completion controller in memory. Use full-vocabulary coda
readout. This measures forced-loop decoding by the trained executor, not
native timer-controlled inference. At one loop the bridge is not invoked.
Checkpoint files are unchanged.

An independent original-Qwen instance uses the same pinned initialization,
prompt, tokenizer, precision, attention, token budget, and greedy decoding.
It shares no adapted layers with the recurrent instance.

The completed command was:

```bash
ssh desktop 'cd /home/mukund/loopformer && PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python /tmp/loopformer_loop_compare.py'
```

The script and raw token IDs, decoded outputs, prompt IDs and package/device
metadata are retained on the Mac under `artifacts/language_probe/20261012/`
(`loop_compare.py`, `results.json`). Remote temporary copies are
`/tmp/loopformer_loop_compare.py` and `/tmp/loopformer_loop_compare.json`.

## Observations

| Model | Loops per token | Tokens including EOS | Total recurrent passes | EOS reached | Response |
| --- | ---: | ---: | ---: | --- | --- |
| Trained recurrent | 1 | 6 | 6 | Yes | `The sky is blue.` |
| Trained recurrent | 2 | 32 | 64 | No | Repeats `The sky is is.` |
| Trained recurrent | 4 | 32 | 128 | No | Begins `the is is  system`, then repeats `Q` |
| Original Qwen | N/A | 32 | N/A | No | Begins `The sky is usually a colorless, colorless, and colorless color`; truncated by the budget |

The four-loop result reproduces the earlier exploratory response. The earlier
script reconstructed the growing input by decoding/re-encoding; this comparison
uses direct token append for both model arms.

The trained recurrent executor answers this question correctly at one loop.
Extra forced loops degrade this particular decoding. This observation does
not establish general language preservation, catastrophic forgetting, or a
natural-language meaning for an individual recurrent loop. The original model
also fails the exact prompt here; this one question cannot rank general quality.
