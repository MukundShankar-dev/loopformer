"""Run agent episodes of a Hugging Face model on a problem set; one JSON line per episode.

    python -m scripts.agent.run --model <snapshot dir> --problems data/agent/mbppplus-v0.2.0 \\
        --condition interaction --max-turns 4 --seeds 0 --limit 20 --output eval/agent/<run>/episodes.jsonl

Conditions share one format: `single` allows only `submit` in one turn;
`interaction` allows every tool for up to --max-turns turns. Up to --rows
episodes decode together (scripts/agent/runner.py); episodes are written as
they finish, not in problem order. run.json records the arguments and, at the
end, throughput.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.agent.decoding import Decoder, Sampling
from scripts.agent.engine import Engine
from scripts.agent.episode import Agent, EpisodeConfig
from scripts.agent.flops import qwen2_flops
from scripts.agent.format import TOOLS
from scripts.agent.runner import Job, run_episodes
from scripts.codetasks.problems import read_problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="local snapshot directory of a pinned revision")
    parser.add_argument("--problems", type=Path, required=True, help="problem-set directory")
    parser.add_argument("--condition", choices=["single", "interaction"], required=True)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-turn-tokens", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--limit", type=int, help="first N problems only")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=32, help="episodes decoded together")
    parser.add_argument("--capacity", type=int, default=4096,
                        help="KV cache tokens per row; a turn that would exceed it ends unfinished")
    parser.add_argument("--workers", type=int, default=8, help="threads running episode code and tools")
    parser.add_argument("--graders", type=int, default=4, help="processes grading submissions")
    parser.add_argument("--grading-workers", type=int, default=1, help="sandbox runs at a time per grade")
    parser.add_argument("--first-failure", action="store_true",
                        help="stop hidden grading at the first failure (training; pass fractions become lower bounds)")
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True,
                        help="replay the decode step as CUDA graphs (CUDA only)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new path")
    if args.condition == "single" and args.max_turns != 1:
        parser.error("the single condition has exactly one turn; pass --max-turns 1")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    engine = Engine(model, args.rows, args.capacity, graphs=args.cuda_graphs and args.device.startswith("cuda"))
    decoder = Decoder(engine, qwen2_flops(model.config), model.config.vocab_size)
    agent = Agent(tokenizer, model.config.vocab_size)
    tools = ("submit",) if args.condition == "single" else TOOLS
    config = EpisodeConfig(tools, args.max_turns, Sampling(args.temperature, args.top_p, args.max_turn_tokens),
                           grading_workers=args.grading_workers, first_failure=args.first_failure)
    problems = read_problems(args.problems)[: args.limit]
    jobs = [Job(problem, config, seed) for seed in args.seeds for problem in problems]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings = {**vars(args), "model": str(args.model), "problems": str(args.problems), "output": str(args.output)}
    (args.output.parent / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    console = Console()
    solved = generated = 0
    started = time.perf_counter()
    with args.output.open("w") as handle:
        for count, episode in enumerate(run_episodes(agent, decoder, jobs, args.workers, args.graders), start=1):
            handle.write(episode.to_json() + "\n")
            handle.flush()
            solved += episode.hidden_outcome == "pass"
            generated += sum(turn.tokens for turn in episode.turns)
            console.print(f"{count}/{len(jobs)} {episode.problem_id} seed {episode.seed}: {episode.stop}, "
                          f"{len(episode.turns)} turns, hidden {episode.hidden_outcome} "
                          f"({episode.hidden_pass_fraction:.2f}), {episode.flops / 1e12:.2f} TFLOPs",
                          markup=False)
    seconds = time.perf_counter() - started
    throughput = {"seconds": seconds, "episodes_per_minute": 60 * len(jobs) / seconds,
                  "generated_tokens_per_second": generated / seconds, **decoder.stats}
    (args.output.parent / "run.json").write_text(json.dumps({**settings, "throughput": throughput}, indent=2) + "\n")
    console.print(f"{len(jobs)} episodes in {seconds:.0f} s ({throughput['episodes_per_minute']:.1f}/min, "
                  f"{throughput['generated_tokens_per_second']:.0f} generated tokens/s)")
    console.print(f"solved {solved} of {len(jobs)} episodes; decoder {dict(decoder.stats)}", markup=False)


if __name__ == "__main__":
    main()
