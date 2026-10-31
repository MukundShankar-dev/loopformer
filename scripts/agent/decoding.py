"""Grammar-constrained, batched sampling of turns over an Engine.

`Decoder` holds one conversation per engine row and at most one turn in
progress per row. `start` prefills the row with the turn's prompt (preceded by
the stop token that ended the row's previous turn, which is sampled but not yet
processed) and samples the first token; `step` decodes one token for every
row with a turn in progress, in one batched forward. Each token is processed
exactly once and the cache carries across turns, so the FLOPs counted here
(scripts/agent/flops.py) are the compute performed.

Sampling does not depend on the batch: each row draws one uniform per token
from its own CPU generator and inverts its masked, tempered and top-p
truncated distribution with it, so a row's tokens depend only on its seed and
its own logits.
"""
import time
from collections import Counter
from dataclasses import dataclass, field

import torch
import xgrammar
from torch import Tensor

from scripts.agent.engine import Engine
from scripts.agent.flops import FlopModel


@dataclass(frozen=True)
class Sampling:
    temperature: float  # 0 means greedy
    top_p: float = 1.0
    max_tokens: int = 1024  # per turn


@dataclass(frozen=True)
class Generation:
    tokens: list[int]  # generated, including the stop token if one was sampled
    finished: bool  # the grammar completed and a stop token was sampled
    seconds: float  # wall time from the turn's prefill to its last token, shared with the rest of the batch


@dataclass
class Conversation:
    generator: torch.Generator
    flops: float = 0.0
    processed: int = 0  # tokens processed, which the cache holds
    pending: list[int] = field(default_factory=list)  # sampled but not yet processed: the last stop token


@dataclass
class _Turn:
    matcher: xgrammar.GrammarMatcher
    sampling: Sampling
    started: float
    tokens: list[int] = field(default_factory=list)


def sample(logits: Tensor, samplings: list[Sampling], uniforms: Tensor) -> Tensor:
    """One token per row of masked `logits` [rows, vocab] (disallowed tokens at -inf), by inverse CDF at `uniforms`."""
    greedy = torch.tensor([s.temperature == 0 for s in samplings], device=logits.device)
    temperatures = torch.tensor([s.temperature or 1.0 for s in samplings], device=logits.device)
    top_p = torch.tensor([s.top_p for s in samplings], device=logits.device)
    probabilities = torch.softmax(logits / temperatures[:, None], dim=-1)
    order = None
    if (top_p < 1).any():
        probabilities, order = probabilities.sort(dim=-1, descending=True)
        probabilities = probabilities.masked_fill(probabilities.cumsum(-1) - probabilities > top_p[:, None], 0.0)
    cumulative = probabilities.cumsum(-1)
    targets = (uniforms.to(logits.device) * cumulative[:, -1])[:, None]
    index = torch.searchsorted(cumulative, targets, right=True).clamp_max(logits.shape[-1] - 1)
    # Rounding can push a target past the last allowed token; fall back to the most probable one.
    allowed = probabilities.gather(1, index) > 0
    index = torch.where(allowed, index, probabilities.argmax(-1, keepdim=True))
    chosen = (index if order is None else order.gather(1, index))[:, 0]
    return torch.where(greedy, logits.argmax(-1), chosen)


class Decoder:
    def __init__(self, engine: Engine, flop_model: FlopModel, vocab_size: int) -> None:
        self.engine = engine
        self.flop_model = flop_model
        self.vocab_size = vocab_size  # logits width, which can exceed the tokenizer's size
        self.conversations: dict[int, Conversation] = {}
        self.turns: dict[int, _Turn] = {}
        # Where wall time goes (synchronized at each sampled token): decode and prefill forwards plus sampling.
        self.stats = Counter()
        # One grammar mask buffer, reused every token; pinned so the copy to the GPU is asynchronous. Each token's
        # sampling synchronizes before the next fill, so a fill never overwrites a copy in flight.
        self.bitmask = xgrammar.allocate_token_bitmask(engine.rows, vocab_size)
        if engine.device.type == "cuda":
            self.bitmask = self.bitmask.pin_memory()

    def open(self, row: int, seed: int) -> None:
        if row in self.conversations:
            raise ValueError(f"row {row} is in use")
        self.engine.reset(row)
        self.conversations[row] = Conversation(torch.Generator().manual_seed(seed))

    def close(self, row: int) -> Conversation:
        if row in self.turns:
            raise ValueError(f"row {row} has a turn in progress")
        return self.conversations.pop(row)

    def start(self, row: int, prompt: list[int], matcher: xgrammar.GrammarMatcher,
              sampling: Sampling) -> Generation | None:
        """Prefill `prompt` and sample the turn's first token; the Generation if the turn already ended."""
        conversation = self.conversations[row]
        tokens = conversation.pending + prompt
        started = time.perf_counter()
        if len(tokens) >= self.engine.room(row):  # no room to decode a single token after the prompt
            return Generation([], False, 0.0)
        first = self.engine.lengths[row]
        logits = self.engine.prefill(row, tokens)
        self.stats["prefills"] += 1
        self.stats["prefill_tokens"] += len(tokens)
        conversation.flops += sum(self.flop_model.per_token(first + offset + 1, logits=False)
                                  for offset in range(len(tokens))) + self.flop_model.logits()
        conversation.processed += len(tokens)
        conversation.pending = []
        self.turns[row] = _Turn(matcher, sampling, started)
        ended = self._advance([row], logits[None], self._masks([row])).get(row)
        self.stats["prefill_seconds"] += time.perf_counter() - started
        return ended

    def step(self) -> dict[int, Generation]:
        """Decode one token for every row with a turn in progress; the turns that ended, by row."""
        rows = sorted(self.turns)
        if not rows:
            return {}
        tokens = [self.turns[row].tokens[-1] if row in self.turns else None for row in range(rows[-1] + 1)]
        for row in rows:
            self.conversations[row].flops += (self.flop_model.per_token(self.engine.lengths[row] + 1, logits=False)
                                              + self.flop_model.logits())
            self.conversations[row].processed += 1
        started = time.perf_counter()
        logits = self.engine.decode(tokens)
        masks = self._masks(rows)  # on the CPU while the GPU runs the forward: masks depend only on accepted tokens
        ended = self._advance(rows, logits[rows], masks)
        self.stats["decode_steps"] += 1
        self.stats["decode_rows"] += len(rows)
        self.stats["decode_seconds"] += time.perf_counter() - started
        return ended

    def _masks(self, rows: list[int]) -> Tensor:
        """Next-token grammar masks of these rows' turns, in order: the first len(rows) rows of the buffer."""
        for index, row in enumerate(rows):
            self.turns[row].matcher.fill_next_token_bitmask(self.bitmask, index)
        return self.bitmask[:len(rows)]

    def _advance(self, rows: list[int], logits: Tensor, masks: Tensor) -> dict[int, Generation]:
        turns = [self.turns[row] for row in rows]
        logits = logits.contiguous()
        xgrammar.apply_token_bitmask_inplace(logits, masks.to(logits.device, non_blocking=True))
        uniforms = torch.tensor([0.0 if turn.sampling.temperature == 0 else
                                 float(torch.rand((), generator=self.conversations[row].generator))
                                 for row, turn in zip(rows, turns)])
        tokens = sample(logits, [turn.sampling for turn in turns], uniforms).tolist()
        ended = {}
        for row, turn, token in zip(rows, turns, tokens):
            if not turn.matcher.accept_token(token):
                raise RuntimeError(f"grammar rejected token {token} that its own mask allowed")
            turn.tokens.append(token)
            finished = turn.matcher.is_terminated()
            if finished or len(turn.tokens) == turn.sampling.max_tokens or self.engine.room(row) < 1:
                if finished:
                    self.conversations[row].pending = [token]
                del self.turns[row]
                ended[row] = Generation(turn.tokens, finished, time.perf_counter() - turn.started)
        return ended
