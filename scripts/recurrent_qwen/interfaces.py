"""Explicit text routing and learned interfaces for isolated pointer execution."""
from functools import lru_cache
import re
from typing import Any

import torch
from torch import Tensor, nn


class PromptRouter:
    """Remove the Steps line without parsing its value or changing public inputs.

    Preserve the original answer index for existing diagnostic callers. Padding
    inserted before that position is masked and RoPE positions are compacted, so
    neither count digits nor their token length reach the executor. Cached CPU
    tokenization is bounded; no prompt or activation survives in a checkpoint.
    """
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.executor_ids = lru_cache(maxsize=4096)(self._executor_ids)

    def _executor_ids(self, ids: tuple[int, ...]) -> tuple[int, ...]:
        prompt = self.tokenizer.decode(list(ids), skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
        if not re.fullmatch(r"Rules: [^\n]+\nStart: [A-Z]\nSteps: [1-9][0-9]*\nAnswer:", prompt):
            raise ValueError("Isolated execution requires the raw Rules/Start/Steps/Answer prompt")
        routed = re.sub(r"\nSteps: [1-9][0-9]*", "", prompt)
        return tuple(self.tokenizer.encode(routed, add_special_tokens=False))

    def __call__(self, ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        rows = ids.detach().cpu().tolist()
        masks = mask.detach().cpu().tolist()
        routed = torch.full_like(ids, self.tokenizer.pad_token_id)
        attention = torch.zeros_like(mask)
        for row, (tokens, valid) in enumerate(zip(rows, masks, strict=True)):
            positions = [i for i, active in enumerate(valid) if active]
            executor = self.executor_ids(tuple(tokens[i] for i in positions))
            end = positions[-1]
            if not executor or len(executor) > end + 1:
                raise ValueError("Routed executor prompt does not fit original input positions")
            routed[row, :len(executor) - 1] = torch.tensor(executor[:-1], device=ids.device)
            routed[row, end] = executor[-1]
            attention[row, :len(executor) - 1] = 1
            attention[row, end] = 1
        positions = (attention.cumsum(-1) - 1).clamp_min(0)
        return routed, attention, positions


class ReentryBridge(nn.Module):
    """Learned map from R output back to R input, applied only on re-entry.

    Normalize direction, then restore the *initial* working-state RMS scale.
    The scale contains rules/start information only and stays differentiable.
    Identity initialization of the projection does not claim T>1 equivalence.
    """
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.projection.weight)
        self.gain = nn.Parameter(torch.ones(width))

    def forward(self, state: Tensor, initial: Tensor) -> Tensor:
        rms = state.float().square().mean(-1, keepdim=True).add(1e-6).sqrt()
        scale = initial.float().square().mean(-1, keepdim=True).add(1e-6).sqrt()
        normalized = (state.float() / rms * scale).to(state.dtype)
        return self.projection(normalized * self.gain)


class RecurrentController(nn.Module):
    """Own learned memory; detached prompt/executor features, no numeric clock."""
    def __init__(self, width: int, intermediate: int = 128) -> None:
        super().__init__()
        self.intermediate = intermediate
        self.context = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, intermediate), nn.Tanh())
        self.observation = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, intermediate), nn.Tanh())
        self.cell = nn.GRUCell(intermediate, intermediate)
        self.readout = nn.Linear(intermediate, 1)

    def initialize(self, prompt_state: Tensor) -> Tensor:
        return self.context(prompt_state.detach())

    def advance(self, state: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        memory = self.cell(self.observation(state.detach()), memory)
        return self.readout(memory).squeeze(-1), memory
