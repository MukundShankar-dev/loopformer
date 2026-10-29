"""Bounded hosted-runtime fidelity and incremental transport verification."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from gradio_client import Client
from huggingface_hub import get_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authenticated", action="store_true")
    args = parser.parse_args()
    client = Client(args.space, token=get_token() if args.authenticated else None,
                    verbose=False, analytics_enabled=False)
    cases = []
    for example in json.loads(args.examples.read_text()):
        started = perf_counter()
        job = client.submit(example["request"], api_name="/predict")
        lengths = []
        first_seconds = None
        for frame in job:
            lengths.append(len(frame["trace"]))
            if first_seconds is None:
                first_seconds = perf_counter() - started
        final = job.result()
        assert final["status"] == "complete"
        assert final["request"] == example["request"]
        assert [r["prediction"] for r in final["trace"]] == [r["prediction"] for r in example["trace"]]
        assert final["summary"] == example["summary"]
        assert [(r["expected"], r["correct"], r["edge_valid"]) for r in final["trace"]] == [
            (r["expected"], r["correct"], r["edge_valid"]) for r in example["trace"]]
        assert all(a <= b for a, b in zip(lengths, lengths[1:]))
        assert len(lengths) > 1 and lengths[0] < len(final["trace"])
        probability_delta = max(abs(a["stop_probability"] - b["stop_probability"])
                                for a, b in zip(final["trace"], example["trace"], strict=True))
        cases.append({"label": example["label"], "loops": len(final["trace"]),
                      "frames_received": len(lengths), "first_frame_loops": lengths[0],
                      "first_frame_seconds": first_seconds, "wall_seconds": perf_counter() - started,
                      "server_inference_seconds": final.get("inference_seconds"),
                      "max_stop_probability_difference_from_recorded": probability_delta,
                      "checkpoint": final["checkpoint"], "predictions_and_summary_match": True})
        print(json.dumps(cases[-1]), flush=True)
    evidence = {"passed": True, "space": args.space, "authenticated": args.authenticated,
                "cases": cases, "scope": "Bounded decoded/stop fidelity; not bitwise cross-runtime equivalence or a population benchmark"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
