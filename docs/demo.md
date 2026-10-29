# Public graph playground and model hosting

The user authorized a public website demo and downloadable final model on
2026-10-11: **at most 26 graph nodes and 256 traversal steps**. This deploys the
frozen model; it opens no new training or research stage.

## Locations

- Website: `https://mukundshankar-dev.github.io/projects/loopformer/demo/`.
- Website source: `/Users/mukunds/Desktop/MukundShankar-dev.github.io/public/projects/loopformer/demo/`.
- Portable frontend source: [demo/web](../demo/web/).
- Model: `https://huggingface.co/mukunds/loopformer`.
- Free ZeroGPU Space: `https://huggingface.co/spaces/mukunds/loopformer-demo`.
- Space entry point: [app.py](../demo/hf_space/app.py).
- Inference, CLI and release helpers: [scripts/demo](../scripts/demo/).
- Published model-card source: [demo_model_card.md](demo_model_card.md).

The frontend is static HTML/CSS/ES modules. No credential is shipped to the
browser. `config.json` selects the Space or compatible Gradio service URL,
model-download link and public step limit. Changing hosts does not change the
website URL. Keeping Gradio needs only a config change; another API protocol
needs a client adapter in `demo.mjs`.

The initial demo uses `journalUrl: null`, sending back links to the homepage.
The journal was subsequently published separately; that setting can be changed
to `/projects/loopformer/` when integrating navigation. Publish only
`projects/loopformer/demo/` from an isolated checkout of the current `gh-pages`
branch; whole-site deployment would include unrelated unpublished work.

## Model behavior and streaming

The existing loader reconstructs the final shared-number checkpoint. FP32,
SDPA, disabled TF32, threshold 0.5 and a 272-loop safety cap are explicit.
The count-free executor and learned timer retain their native semantics; no
reference state or decoded letter enters recurrence. The cap is not
requested-depth stopping.

`RecurrentQwen.forward(loop_callback=...)` observes each readout, including the
stopping loop, through detached copies of C, stop and direct-R logits. Callback
return values cannot choose stopping. Observation requires eval/no-grad, batch
one, answer logits and no labels. Exceptions propagate.

`stream_inference` runs one native forward in a worker thread and emits cumulative
JSON snapshots. Reference scoring happens separately in the consumer. Closing
the generator cancels and joins the worker before GPU ownership is released.
The loop cap bounds the queue; the final snapshot retains every prediction if
transport coalesces updates. Gradio serializes calls, queues at most 16 and requests
30 GPU seconds. Playback speed is independent of inference arrival. Cancellation
and connection errors preserve explicitly incomplete partial traces.

Input is exactly `mapping`, `start`, `steps`. Mapping is an ordered array of
26 A–Z pairs with one outgoing rule per source; order is preserved in the prompt.
Smaller visible graphs give unused letters self-loops while still supplying all
26 rules. Seeded random functions, permutations, full cycles and manual editing
are validated in browser and server.

The UI distinguishes reference-at-loop correctness, decoded edge consistency,
final-letter correctness, strict requested-prefix correctness and exact stopping.
Stop probability is a timing signal, not correctness confidence. Direct R
predictions are absent in the saved examples and marked accordingly. The
architecture view is a schematic, not measured attention or a human reasoning trace.

## Runtime compatibility

Research and downloadable inference pins remain PyTorch **2.14.0**. The first
ZeroGPU build failed because HF injects `torch<=2.13.0`; only the hosted service
therefore pins **2.13.0**, preserving FP32/SDPA and weights. Hosted prediction
and stopping fidelity must pass before enabling the website; a bounded check
cannot establish full-population numerical equivalence. Space requirements are
self-contained because HF mounts them at `/tmp/requirements.txt` during build.

The Space pins a model commit through `LOOPFORMER_MODEL_REVISION`. The publisher
checks the final weight SHA-256 and copies only inference files, excluding
optimizer state, data and credentials. It requests `zero-a10g` only, never a
paid fallback. The original model commit is
`e29f88cb85afdbf5c3f877e853a20067ce4fb313`.

## Commands

From the research checkout, after `hf auth login`:

```bash
.venv/bin/python -m scripts.demo.export_examples
node --test demo/test_core.mjs
.venv/bin/python -m pytest -q tests/test_demo.py tests/test_executor_upgrade.py tests/test_fixed_prompt.py tests/test_adaptive_inference.py tests/test_number_reader.py
.venv/bin/python -m scripts.demo.deploy \
  --checkpoint models/stage1_pointer/controller-shared-number-precision-seed83/best \
  --model-id mukunds/loopformer --space-id mukunds/loopformer-demo --max-steps 256
```

For app-only fixes, upload only the changed Space files instead of republishing
weights. An accepted upload does not establish working inference: check build,
runtime and a real API request. The publisher's model-card source is linked above.

Export a graph from the UI and run locally:

```bash
.venv/bin/python -m scripts.demo --model mukunds/loopformer \
  --request graph.json --device cuda --download
```

Increase `--max-steps` and `--loop-cap` explicitly for longer exploration. The
alphabet remains A–Z; graph accuracy beyond 256 is unestablished.

## Verified evidence — 2026-10-11

- 58 focused Python tests passed, including 14 demo tests; observers cannot
  change native logits/stops even when they mutate their copies.
- Three Node tests passed: seeded generation, malformed editor inputs and
  independent scoring of saved examples. Python/JS syntax checks passed.
- A local Gradio HTTP call with tiny synthetic Qwen streamed lengths 1–12 and a
  final snapshot. The browser client completed it and correctly reported the
  safety-cap outcome. This verifies transport, not pretrained quality.
- Desktop final-checkpoint CUDA agreed with native and recorded predictions at
  all 280 checked loops (two 12-step examples and one 256-step example), with
  exact native stop-probability agreement. Machine evidence:
  `artifacts/demo/streaming_cuda_validation.json`.
- Warm streaming took 0.489 / 0.493 s for 12 steps and 10.204 s for 256 on the
  RTX 5070 Ti. Includes observation/reference/export overhead; excludes loading
  and network. These are not HF hosting latencies.
- Desktop and 390-pixel mobile layouts were inspected. Failure navigation,
  generation, malformed-rule rejection and 256-step scrubbing passed. Mobile
  document width equaled viewport width. Local streaming had no browser errors.

The initial Space deployment ran on free `zero-a10g`, source commit
`1bf186492dcd92da2f70626d839b5d7f01828da9`. The hosted PyTorch 2.13 service
matched all 280 audited predictions, all summaries and every stop probability
exactly on the bounded three-case panel. It emitted 13, 13 and 257 snapshots;
the first frame contained one prediction in every case. Evidence:
`artifacts/demo/hosted_validation.json`.

Observed authenticated wall times were 8.175 s for the first 12-step request,
0.629 s for a warm 12-step request, and 5.409 s for the 256-step request. First
frames arrived at 7.907 / 0.307 / 0.442 s respectively. These are individual
observations including network/queue overhead, not latency guarantees. Anonymous
browser access also passed a correct 12-step live run with exact stopping and
no browser errors. No paid plan/hardware was selected.

GitHub Pages demo-only deployment commit: `e60a8ea6724ff73301968105b9296fff1b1e3da0`.
The build completed, and the [public playground](https://mukundshankar-dev.github.io/projects/loopformer/demo/)
passed a fresh anonymous live request: eight working nodes, random-function seed
73, start B, 24 requested steps. All 24 predictions were correct, stopping was
exact, and the browser reported no errors. The model card was updated in
`01b4b8a665b1839a1838ec4fbe099b8d7f6bf305`; the service retains its original
pinned inference commit. Only the demo directory was published to Pages; the
local journal draft was not part of this deployment.

## Owner-authenticated proxy — 2026-10-11

After anonymous requests hit HF's runs limit, the user authorized sharing the
owner account's allowance through a server-side proxy. The proxy lives in the
existing ZeroGPU Space's CPU web process; no second hosting service is needed.
`scripts/demo/proxy.py` adds `POST /api/predict` through Gradio’s `app_kwargs`.
The regular `Blocks.launch` path is retained because ZeroGPU registers GPU
workers in its launch hook; standalone Uvicorn startup failed that check.
It uses a private `LOOPFORMER_PROXY_TOKEN` Space secret to call the unchanged
`/predict` endpoint through HF's public gateway. This second authenticated HTTP
hop is necessary: directly calling the GPU function would retain the anonymous
visitor's quota identity. The browser never receives the token.

`demo/web/proxy.mjs` consumes newline-delimited JSON as predictions arrive and
supports cancellation. `config.json` selects `proxyUrl`; removing that field
restores direct Gradio access. The original `/predict` API remains available for
people using their own accounts. The proxy uses the exact same model, request
validation, prompt order, stopping policy and snapshots.

Admission permits one active sponsored run, at least ten seconds between new
runs, and at most twenty new attempts per hour. Requests must have the website's
Origin, contain at most 4 KiB, and pass native A–Z/1–256 validation. Origin/CORS
checks discourage unwanted browser embedding but are not authentication against
non-browser clients. The global limits still apply when Origin is forged.
Only successful complete results are cached: at most 32 entries for one hour,
keyed by the entire ordered request. Cached results are labelled in the UI and
do not spend GPU quota. A quota failure blocks new attempts for five minutes;
HF remains the authority on when the account quota actually resets. Requests
have a 120-second relay timeout and cancellation is forwarded upstream.

These controls and the cache are in memory for one process/replica and reset on
restart. They are usage safeguards, not a financial spending cap. All visitors
share the owner's HF quota; free-account runs limits still apply. No paid plan,
credits, or automatic hardware upgrade is configured by this change. If the
owner later enables paid over-quota credits, review these limits first.

Set the secret through HF's Space settings or `HfApi.add_space_secret` using
`get_token()` locally; never put it in source files, browser configuration, URLs
or command-line arguments. Restart/redeployment retains an existing Space secret.
The publisher includes the proxy module, but a new Space requires its secret to
be configured separately. Gradio remains in direct mode if the secret is absent.

Verified: 20 Python tests (14 native demo and six proxy cases), four Node tests,
and local real HTTP self-relay streaming/cached replay. The corrected Space is
running at `e0ea85bfab6ad5948beb7e0c400f4d31238200bb`. An anonymous hosted relay
request emitted 13 frames (lengths 1–12, then final), matching all audited
predictions, stop probabilities and summary fields for the 12-step example.
First frame: 3.777 s; total: 4.028 s, including startup/network/queue overhead.
A repeated identical request was served from cache. Invalid payloads and missing
Origin were rejected. Evidence: `artifacts/demo/proxy_validation.json`.

Pages commit `02f94ae` changes only `config.json`, `demo.mjs` and `proxy.mjs`,
based on the newer journal deployment `5b55463`. The Pages build completed.
The public browser, without HF sign-in, replayed the 12-step request from cache
and completed a fresh 26-node full-cycle graph (seed 84, start P, 256 steps): all
predictions correct, exact learned stopping, 6.09 s server inference including
observation/export overhead, and no browser errors. No owner credential appears
in frontend assets.

Recheck without authenticating the caller:

```bash
.venv/bin/python -m scripts.demo.verify_proxy \
  --url https://mukunds-loopformer-demo.hf.space/api/predict \
  --output artifacts/demo/proxy_validation.json
```

A cached first response legitimately has one frame; use a fresh service/cache to
measure incremental transport. Validation never creates substitute model outputs.
