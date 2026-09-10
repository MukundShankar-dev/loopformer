"""Validate the actual tokenizer in the exact pointer prompt/answer contexts."""

from typing import Any
import re

from scripts.dataset.pointer import SYMBOLS


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def validate_prompt_tokens(tokenizer: Any, prompt: str, token_ids: dict[str, int]) -> int:
    """Check every space-prefixed symbol span; return the answer readout index.

    Prompts are raw text, with no chat template or added special tokens. The
    answer position is the final ':' token's index (a next-token readout).
    """
    if not prompt.endswith("\nAnswer:"):
        raise ValueError("Prompt must end with '\\nAnswer:' without a trailing space")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    spans = dict(zip(map(tuple, encoded["offset_mapping"]), encoded["input_ids"]))
    for match in re.finditer(r" [A-Z](?=[,)\n])", prompt):
        if spans.get(match.span()) != token_ids[match.group()[1:]]:
            raise ValueError(f"Symbol does not retain its token at {match.span()}: {match.group()!r}")
    return len(encoded["input_ids"]) - 1


def validate_symbols(tokenizer: Any) -> dict[str, int]:
    """Require one unique token per ' '+symbol in rules, starts, and answers."""
    token_ids = {}
    for symbol in SYMBOLS:
        ids = tokenizer.encode(" " + symbol, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != " " + symbol:
            raise ValueError(f"Symbol {symbol!r} is not one reversible space-prefixed token")
        token_ids[symbol] = ids[0]
    if len(set(token_ids.values())) != len(SYMBOLS):
        raise ValueError("Answer tokens must be unique")
    # Exhaustive pair contexts, and all 26 possible answer continuations.
    for source in SYMBOLS:
        prompt = "Rules: " + " ".join(f"( {source}, {target})" for target in SYMBOLS)
        prompt += f"\nStart: {source}\nSteps: 1\nAnswer:"
        validate_prompt_tokens(tokenizer, prompt, token_ids)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        for target in SYMBOLS:
            if tokenizer.encode(prompt + " " + target, add_special_tokens=False) != [*prefix, token_ids[target]]:
                raise ValueError(f"Answer boundary merges for {target!r}")
    return token_ids
