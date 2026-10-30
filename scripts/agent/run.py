"""Run agent episodes of a Hugging Face model on a problem file; one JSON line per episode.

    python -m scripts.agent.run --model <snapshot dir> --problems data/agent/mbppplus-v0.2.0.jsonl \\
        --condition interaction --max-turns 4 --seeds 0 --limit 20 --output eval/agent/<run>/episodes.jsonl

Conditions share one format: `single` allows only `submit` in one turn;
`interaction` allows every tool for up to --max-turns turns.
"""
import argparse
import json
from pathlib import Path

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.agent.decoding import Sampling
from scripts.agent.episode import Agent, EpisodeConfig
from scripts.agent.flops import qwen2_flops
from scripts.agent.format import TOOLS
from scripts.codetasks.problems import read_problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="local snapshot directory of a pinned revision")
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--condition", choices=["single", "interaction"], required=True)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-turn-tokens", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--limit", type=int, help="first N problems only")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new path")
    if args.condition == "single" and args.max_turns != 1:
        parser.error("the single condition has exactly one turn; pass --max-turns 1")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    agent = Agent(model, tokenizer, qwen2_flops(model.config), model.config.vocab_size)
    tools = ("submit",) if args.condition == "single" else TOOLS
    config = EpisodeConfig(tools, args.max_turns, Sampling(args.temperature, args.top_p, args.max_turn_tokens))
    problems = read_problems(args.problems)[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "run.json").write_text(json.dumps({**vars(args), "model": str(args.model),
                                                             "problems": str(args.problems),
                                                             "output": str(args.output)}, indent=2) + "\n")
    console = Console()
    solved = 0
    with args.output.open("w") as handle:
        for seed in args.seeds:
            for index, problem in enumerate(problems):
                episode = agent.run(problem, config, seed)
                handle.write(episode.to_json() + "\n")
                handle.flush()
                solved += episode.hidden_outcome == "pass"
                console.print(f"seed {seed} {index + 1}/{len(problems)} {problem.id}: {episode.stop}, "
                              f"{len(episode.turns)} turns, hidden {episode.hidden_outcome} "
                              f"({episode.hidden_pass_fraction:.2f}), {episode.flops / 1e12:.2f} TFLOPs",
                              markup=False)
    console.print(f"solved {solved} of {len(problems) * len(args.seeds)} episodes")


if __name__ == "__main__":
    main()
