"""Run one agent episode on one problem and record everything measured.

The agent sees the instructions, the statement and the public tests. Each
turn is sampled under the format grammar and its tool is executed in the
sandbox; the observation becomes the next user message. On the last turn the
grammar allows only `submit`, so every episode that is not cut off by the
per-turn token cap ends with a submission (`forced_submit` marks those).
The submission is graded on the hidden tests, whose results are recorded
here and never shown to the agent.

Observation text comes from untrusted programs, so it is tokenized with
special tokens split: a program cannot print `<|im_start|>` to forge a turn.
"""
import json
from dataclasses import asdict, dataclass, field

import torch
import xgrammar

from scripts.agent.decoding import Conversation, Sampling
from scripts.agent.flops import FlopModel
from scripts.agent.format import grammar, instructions, parse_turn, render_public_tests, render_run
from scripts.codetasks.problems import Problem
from scripts.sandbox.grader import Tests, failure, grade
from scripts.sandbox.runner import Limits, run
from scripts.sandbox.values import decode

RUN_LIMITS = Limits(wall_seconds=5, cpu_seconds=5)


@dataclass(frozen=True)
class EpisodeConfig:
    tools: tuple[str, ...]  # ("submit",) is the single-turn condition
    max_turns: int
    sampling: Sampling
    grading_workers: int = 4


@dataclass
class TurnRecord:
    tool: str | None
    thought: str | None
    code_characters: int
    tokens: int
    finished: bool
    model_seconds: float
    sandbox_seconds: float = 0.0
    outcome: str | None = None  # run_python: failure class or "ok"; run_public_tests: the grade's outcome
    public_pass_fraction: float | None = None
    observation: str | None = None
    text: str = ""  # the generated turn, without its stop token


@dataclass
class Episode:
    problem_id: str
    seed: int
    stop: str  # "submit", "forced_submit" or "truncated"
    turns: list[TurnRecord] = field(default_factory=list)
    final_code: str | None = None
    hidden_outcome: str | None = None
    hidden_pass_fraction: float = 0.0
    public_outcome: str | None = None
    flops: float = 0.0
    tokens_processed: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def describe_public_tests(tests: Tests) -> str:
    if tests.kind == "call":
        lines = []
        for case in tests.cases:
            arguments = ", ".join(repr(argument) for argument in decode(json.loads(case.input)))
            lines.append(f"assert {tests.function}({arguments}) == {decode(json.loads(case.expected))!r}")
        return "\n".join(lines)
    return "\n\n".join(f"Input:\n{case.input}\nExpected output:\n{case.expected}" for case in tests.cases)


class Agent:
    """A model, its tokenizer and compiled grammars, shared across episodes."""

    def __init__(self, model, tokenizer, flop_model: FlopModel, vocab_size: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.flop_model = flop_model
        self.vocab_size = vocab_size
        info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        self.compiler = xgrammar.GrammarCompiler(info)
        self.grammars: dict[tuple[str, ...], xgrammar.CompiledGrammar] = {}

    def matcher(self, tools: tuple[str, ...]) -> xgrammar.GrammarMatcher:
        if tools not in self.grammars:
            self.grammars[tools] = self.compiler.compile_grammar(grammar(tools))
        return xgrammar.GrammarMatcher(self.grammars[tools])

    def observation_tokens(self, text: str) -> list[int]:
        """Tokens after the assistant's stop token: a user message with `text`, then the next assistant header."""
        encode = self.tokenizer.encode
        return (encode("\n<|im_start|>user\n", add_special_tokens=False)
                + encode(text, add_special_tokens=False, split_special_tokens=True)
                + encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False))

    def run(self, problem: Problem, config: EpisodeConfig, seed: int) -> Episode:
        generator = torch.Generator().manual_seed(seed)
        conversation = Conversation(self.model, self.flop_model, self.vocab_size, generator)
        messages = [{"role": "system", "content": instructions(config.tools, config.max_turns)},
                    {"role": "user", "content": f"Problem:\n{problem.statement}\n\nPublic tests:\n"
                                                f"{describe_public_tests(problem.public)}"}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=False)
        episode = Episode(problem.id, seed, stop="truncated")
        for number in range(config.max_turns):
            last = number == config.max_turns - 1
            tools = ("submit",) if last else config.tools
            generation = conversation.generate(prompt, self.matcher(tools), config.sampling)
            text = self.tokenizer.decode(generation.tokens[:-1] if generation.finished else generation.tokens)
            record = TurnRecord(None, None, 0, len(generation.tokens), generation.finished, generation.seconds, text=text)
            episode.turns.append(record)
            if not generation.finished:
                break
            turn = parse_turn(text)
            record.tool, record.thought, record.code_characters = turn.tool, turn.thought, len(turn.code)
            if turn.tool == "submit":
                episode.final_code = turn.code
                episode.stop = "forced_submit" if tools != config.tools else "submit"
                break
            if turn.tool == "run_python":
                result = run({"main.py": turn.code}, stdin=turn.stdin or "", limits=RUN_LIMITS)
                record.sandbox_seconds, record.outcome = result.wall_seconds, failure(result) or "ok"
                record.observation = render_run(result)
            else:
                public = grade(turn.code, problem.public, workers=config.grading_workers)
                record.sandbox_seconds = sum(result.wall_seconds for result in public.results)
                record.outcome, record.public_pass_fraction = public.outcome, public.pass_fraction
                record.observation = render_public_tests(problem.public, public)
            prompt = self.observation_tokens(record.observation)
        episode.flops, episode.tokens_processed = conversation.flops, conversation.length
        if episode.final_code is not None:
            hidden = grade(episode.final_code, problem.hidden, workers=config.grading_workers)
            public = grade(episode.final_code, problem.public, workers=config.grading_workers)
            episode.hidden_outcome, episode.hidden_pass_fraction = hidden.outcome, hidden.pass_fraction
            episode.public_outcome = public.outcome
        return episode
