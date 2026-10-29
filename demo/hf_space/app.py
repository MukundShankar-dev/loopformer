"""Hugging Face entry point. Explicit startup loading, no inference on import elsewhere."""

import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# ZeroGPU must install its CUDA emulation before importing torch/model code.
import spaces
import torch
from gradio_client import Client
from huggingface_hub import snapshot_download

from scripts.demo.inference import DemoRequest, stream_inference
from scripts.demo.gradio_app import create_app
from scripts.demo.proxy import create_proxy
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint


MODEL_ID = os.environ["LOOPFORMER_MODEL_ID"]
MODEL_REVISION = os.environ["LOOPFORMER_MODEL_REVISION"]
MAX_STEPS = int(os.environ.get("LOOPFORMER_MAX_STEPS", "256"))
if not 1 <= MAX_STEPS <= 256:
    raise ValueError("Public demo maximum must be within 1–256 steps")
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
checkpoint = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                   allow_patterns=["*.json", "*.pt", "*.jinja", "*.txt"]))
model, tokenizer, spec = load_recurrent_checkpoint(checkpoint, device="cuda", download=True)
model.config._attn_implementation = "sdpa"
model.requires_grad_(False)


@spaces.GPU(duration=30)
def predict(request: dict):
    task = DemoRequest.parse(request, max_steps=MAX_STEPS)
    yield from stream_inference(model, tokenizer, spec, task,
                                checkpoint_id=f"{MODEL_ID}@{MODEL_REVISION}")


app = create_app(predict, model_url=f"https://huggingface.co/{MODEL_ID}")
if __name__ == "__main__":
    if os.environ.get("LOOPFORMER_PROXY_TOKEN"):
        # Re-enter through HF's public gateway so ZeroGPU authenticates the owner.
        # A direct Python call would inherit the anonymous visitor's identity.
        def owner_client():
            return Client(os.environ["SPACE_ID"], token=os.environ["LOOPFORMER_PROXY_TOKEN"],
                          verbose=False, analytics_enabled=False, download_files=False,
                          max_workers=1)

        proxy = create_proxy(lambda value: DemoRequest.parse(value, max_steps=MAX_STEPS).payload(),
                             owner_client, origins=("https://mukundshankar-dev.github.io",))
        # Keep Blocks.launch: spaces registers GPU workers in its launch hook.
        # A standalone Uvicorn/mount_gradio_app launch skips that registration.
        app.launch(show_error=True, ssr_mode=False,
                   app_kwargs={"routes": proxy.routes, "middleware": proxy.user_middleware})
    else:
        app.launch(show_error=True)
