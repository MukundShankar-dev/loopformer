"""LiveCodeBench's own codegen runner on problems from scripts/codeprobe/livecodebench_data.py.

    python -m scripts.codeprobe.livecodebench_codegen --lcb ~/src/LiveCodeBench --problems <dir> \\
        --output <run>/generations.json -- --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
        --local_model_path <snapshot> --n 16 --temperature 0.8

Arguments after `--` go to the unmodified `lcb_runner.runner.main`, which
builds the prompts, samples with vLLM, extracts the code and writes its usual
output list. Three substitutions:
- the model is registered under its Hub id with the CodeQwenInstruct style,
  which LiveCodeBench uses for its listed Qwen2.5 models;
- problems come from --problems, without hidden tests (codegen does not use
  them), instead of `load_dataset` with the dataset's remote loading script;
- the output goes to --output.
LiveCodeBench's `--evaluate` runs samples on the host, so it is refused;
scripts/codeprobe/evaluate_livecodebench.py grades in the sandbox. vLLM's
GPU share is capped (scripts/codeprobe/vllm_share.py).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REFUSED = ("--evaluate", "--continue_existing_with_eval", "--not_fast", "--scenario", "--use_cache")


def load_problems(directory: Path, problem_class, start_date: str | None, end_date: str | None) -> list:
    """LiveCodeBench problems without hidden tests, filtered by date as its own loader does."""
    problems = []
    for line in (directory / "problems.jsonl").read_text().splitlines():
        problem = problem_class(**json.loads(line), private_test_cases="[]")
        if start_date is not None and problem.contest_date < datetime.strptime(start_date, "%Y-%m-%d"):
            continue
        if end_date is not None and problem.contest_date > datetime.strptime(end_date, "%Y-%m-%d"):
            continue
        problems.append(problem)
    print(f"Loaded {len(problems)} problems")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lcb", type=Path, required=True, help="LiveCodeBench checkout")
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("lcb_args", nargs=argparse.REMAINDER, help="after --: arguments for lcb_runner.runner.main")
    args = parser.parse_args()
    lcb_args = args.lcb_args[1:] if args.lcb_args[:1] == ["--"] else args.lcb_args
    refused = [flag for flag in lcb_args if flag.split("=")[0] in REFUSED]
    if refused:
        parser.error(f"refused LiveCodeBench flags: {refused}")
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument("--model", required=True)
    model_id = model_parser.parse_known_args(lcb_args)[0].model
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    problems, output, lcb = args.problems.absolute(), args.output.absolute(), args.lcb.absolute()

    sys.path.insert(0, str(lcb))
    from scripts.codeprobe import vllm_share  # vLLM loads only once the arguments are valid
    os.chdir(lcb)  # LiveCodeBench opens its prompt files relative to its checkout, at import
    import anthropic
    # Removed from current anthropic (a vLLM dependency); LiveCodeBench imports them only for Claude-style
    # prompts, and its own code_generation.py falls back to None when they are missing.
    anthropic.HUMAN_PROMPT = anthropic.AI_PROMPT = None
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.lm_styles import LanguageModel, LanguageModelStore, LMStyle
    from lcb_runner.runner import main as lcb_main, scenario_router

    LanguageModelStore[model_id] = LanguageModel(model_id, model_id.split("/")[-1], LMStyle.CodeQwenInstruct, None)
    scenario_router.load_code_generation_dataset = (
        lambda release_version, start_date=None, end_date=None:
        load_problems(problems, CodeGenerationProblem, start_date, end_date))
    lcb_main.get_output_path = lambda model_repr, lcb_arguments: str(output)
    vllm_share.cap()
    sys.argv = [sys.argv[0], *lcb_args]
    lcb_main.main()


if __name__ == "__main__":
    main()
