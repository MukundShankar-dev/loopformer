"""Download/run the final checkpoint, printing one JSON snapshot per readout."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from huggingface_hub import snapshot_download

from scripts.demo.inference import DemoRequest, stream_inference
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--model", help="Hugging Face model repository ID")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--request", required=True, type=Path, help="JSON exported from the demo")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--download", action="store_true", help="Allow explicit Hub/base downloads")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--loop-cap", type=int, default=272)
    args = parser.parse_args()
    request = DemoRequest.parse(json.loads(args.request.read_text()), max_steps=args.max_steps)
    if args.loop_cap < request.steps:
        parser.error("--loop-cap must cover the requested steps")
    path = args.checkpoint or Path(snapshot_download(args.model, revision=args.revision,
                                                   local_files_only=not args.download))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, spec = load_recurrent_checkpoint(path, device=args.device, download=args.download)
    model.config._attn_implementation = "sdpa"
    model.requires_grad_(False)
    for frame in stream_inference(model, tokenizer, spec, request,
                                   checkpoint_id=args.model or str(path), loop_cap=args.loop_cap):
        print(json.dumps(frame, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
