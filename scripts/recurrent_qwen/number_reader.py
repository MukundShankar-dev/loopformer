"""Shared learned token composition, with no numeric parser in inference."""
import re

import torch
from torch import Tensor, nn


MAX_NUMBER_TOKENS = 8


def number_token_ids(tokenizer, ids: tuple[int, ...]) -> tuple[int, ...]:
    """Route only the raw Steps text; never convert it to a number.

    One token per character is a tokenizer contract, not a digit-value lookup.
    The public prompt and executor routing remain unchanged.
    """
    text = tokenizer.decode(list(ids), skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    match = re.fullmatch(r'Rules: [^\n]+\nStart: [A-Z]\nSteps: ([1-9][0-9]*)\nAnswer:', text)
    if match is None:
        raise ValueError('Number reader requires the raw Rules/Start/Steps/Answer prompt')
    field = match.group(1)
    tokens = tokenizer.encode(field, add_special_tokens=False)
    if not 1 <= len(tokens) <= MAX_NUMBER_TOKENS or len(tokens) != len(field):
        raise ValueError('Number reader requires one token per decimal character and at most eight characters')
    # Length alone does not rule out a tokenizer that groups/reorders characters.
    if any(tokenizer.decode([t], clean_up_tokenization_spaces=False) != c
           for t, c in zip(tokens, field, strict=True)):
        raise ValueError('Number reader tokenizer does not preserve individual decimal characters')
    return tuple(tokens)


class SharedNumberReader(nn.Module):
    """Read padded frozen embeddings [B,8*(H+1)] using the same map at each slot.

    Last channel of each slot is a validity mask. Valid tokens are left aligned.
    The gain starts at identity, not a decimal radix. Training labels fit both
    gain and token projection. Accumulate in float64 to reduce numerical error,
    then return the controller's scalar-memory dtype [B,1].
    """
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.in_features = MAX_NUMBER_TOKENS * (width + 1)
        self.token_value = nn.Linear(width, 1)
        self.gain = nn.Parameter(torch.ones(()))

    def unpack(self, context: Tensor) -> tuple[Tensor, Tensor]:
        if context.ndim != 2 or context.shape[1] != self.in_features:
            raise ValueError('Need padded number embeddings with token-validity masks')
        slots = context.reshape(-1, MAX_NUMBER_TOKENS, self.width + 1)
        return slots[..., :self.width], slots[..., self.width:]

    def forward(self, context: Tensor) -> Tensor:
        embeddings, mask = self.unpack(context.detach())
        values = torch.nn.functional.linear(embeddings.double(), self.token_value.weight.double(),
                                            self.token_value.bias.double())
        state = values.new_zeros(len(values), 1)
        for value, valid in zip(values.unbind(1), mask.unbind(1), strict=True):
            state = torch.where(valid.bool(), self.gain.double() * state + value, state)
        return state.to(self.token_value.weight.dtype)
