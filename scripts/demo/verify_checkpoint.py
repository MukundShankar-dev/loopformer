"""Compare streamed outputs with native inference and audited saved examples."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from scripts.dataset.pointer import SYMBOLS
from scripts.demo.inference import DemoRequest, stream_inference
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, spec = load_recurrent_checkpoint(args.checkpoint, device=args.device)
    model.config._attn_implementation = "sdpa"
    model.requires_grad_(False)
    reports = []
    for example in json.loads(args.examples.read_text()):
        request = DemoRequest.parse(example["request"])
        encoded = tokenizer(request.prompt(), add_special_tokens=False, return_tensors="pt")
        with torch.inference_mode():
            native = model(input_ids=encoded["input_ids"].to(args.device),
                           attention_mask=encoded["attention_mask"].to(args.device),
                           num_loops=272, completion_threshold=.5, readout_token_ids=spec["token_ids"])
        started = perf_counter()
        frames = list(stream_inference(model, tokenizer, spec, request, checkpoint_id=str(args.checkpoint)))
        elapsed = perf_counter() - started
        final = frames[-1]
        predictions = [SYMBOLS[int(scores[0].argmax())] for scores in native.loop_logits]
        streamed = [row["prediction"] for row in final["trace"]]
        recorded = [row["prediction"] for row in example["trace"]]
        assert predictions == streamed == recorded, example["label"]
        assert len(frames) == len(predictions) + 1
        assert final["summary"] == example["summary"]
        probabilities = native.stop_logits[0].sigmoid().cpu().tolist()
        assert probabilities == [row["stop_probability"] for row in final["trace"]]
        reports.append({"label": example["label"], "loops": len(predictions),
                        "frames": len(frames), "stream_seconds": elapsed,
                        "native_and_recorded_predictions_match": True,
                        "native_stop_probabilities_match": True})
    with (args.checkpoint / "adapter_model.pt").open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    evidence = {"device": args.device, "torch": torch.__version__, "dtype": "float32",
                "attention": "sdpa", "weights_sha256": checksum, "passed": True,
                "cases": reports, "timing_scope": "Warm batch-one forward plus observation, reference scoring and JSON snapshot construction; excludes model loading and network"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
