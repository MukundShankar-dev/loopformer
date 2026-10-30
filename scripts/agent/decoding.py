"""Grammar-constrained sampling over one conversation with a persistent KV cache.

`Conversation` holds the token stream of an episode. Each token is processed
exactly once: prompt and observation tokens are prefilled, generated tokens
are fed back one at a time, and the cache carries across turns. FLOPs are
counted for exactly that work (scripts/agent/flops.py), so accounted compute
equals performed compute.

This drives a Hugging Face causal LM (the base model at T=1). The looped model
gets the same interface in M1, once it has a KV cache.
"""
import time
from dataclasses import dataclass

import torch
import xgrammar
from transformers import DynamicCache

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
    seconds: float  # synchronized wall time of this turn's prefill and decoding


class Conversation:
    def __init__(self, model, flop_model: FlopModel, vocab_size: int, generator: torch.Generator) -> None:
        self.model = model
        self.flop_model = flop_model
        self.vocab_size = vocab_size  # logits width, which can exceed the tokenizer's size
        self.generator = generator  # a CPU generator: masking and sampling happen on the CPU
        self.device = next(model.parameters()).device
        self.cache = DynamicCache()
        self.length = 0
        self.flops = 0.0
        self.pending: list[int] = []  # sampled but not yet processed (the last stop token)

    @torch.inference_mode()
    def _forward(self, tokens: list[int]) -> torch.Tensor:
        """Process `tokens` after the current context; logits of the last one, shape [vocab]."""
        ids = torch.tensor([tokens], device=self.device)
        output = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        for offset in range(len(tokens)):
            self.flops += self.flop_model.per_token(context=self.length + offset + 1, logits=False)
        self.flops += self.flop_model.logits()
        self.length += len(tokens)
        return output.logits[0, -1].float()

    def _sample(self, logits: torch.Tensor, sampling: Sampling) -> int:
        if sampling.temperature == 0:
            return int(logits.argmax())
        probabilities = torch.softmax(logits / sampling.temperature, dim=-1)
        if sampling.top_p < 1.0:
            sorted_probabilities, order = probabilities.sort(descending=True)
            outside = sorted_probabilities.cumsum(0) - sorted_probabilities > sampling.top_p
            sorted_probabilities[outside] = 0.0
            probabilities = torch.zeros_like(probabilities).scatter(0, order, sorted_probabilities)
        return int(torch.multinomial(probabilities, 1, generator=self.generator))

    def generate(self, prompt: list[int], matcher: xgrammar.GrammarMatcher, sampling: Sampling) -> Generation:
        """Append `prompt` (any pending token first), then sample a turn the grammar accepts."""
        synchronize(self.device)
        start = time.perf_counter()
        logits = self._forward(self.pending + prompt)
        self.pending = []
        bitmask = xgrammar.allocate_token_bitmask(1, self.vocab_size)
        tokens = []
        while len(tokens) < sampling.max_tokens:
            matcher.fill_next_token_bitmask(bitmask)
            masked = logits.unsqueeze(0).cpu()
            xgrammar.apply_token_bitmask_inplace(masked, bitmask)
            token = self._sample(masked[0], sampling)
            if not matcher.accept_token(token):
                raise RuntimeError(f"grammar rejected token {token} that its own mask allowed")
            tokens.append(token)
            if matcher.is_terminated():
                self.pending = [token]
                break
            logits = self._forward([token])
        synchronize(self.device)
        return Generation(tokens, matcher.is_terminated(), time.perf_counter() - start)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
