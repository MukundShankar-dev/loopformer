"""Tagged JSON for Python values that cross the sandbox boundary in call tests.

Plain JSON turns tuples and sets into lists, but benchmark checkers compare
with Python `==`, where `(1, 2) != [1, 2]`. Tuples, sets and dicts with
non-string keys are wrapped in single-key tagged objects, so the host can
compare encoded forms structurally without decoding untrusted output.

This file is also copied into the sandbox next to the call harness, so it
uses only the standard library.
"""
import json

TUPLE, SET, DICT, COMPLEX = "__tuple__", "__set__", "__dict__", "__complex__"
TAGS = (TUPLE, SET, DICT, COMPLEX)


def encode(value):
    """A JSON-serializable form of `value`; raises TypeError for unsupported types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {COMPLEX: [value.real, value.imag]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, tuple):
        return {TUPLE: [encode(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        # Sorted by encoded text, so equal sets encode identically whatever their element types.
        return {SET: sorted((encode(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, dict):
        if all(isinstance(key, str) and key not in TAGS for key in value):
            return {key: encode(item) for key, item in value.items()}
        pairs = [[encode(key), encode(item)] for key, item in value.items()]
        return {DICT: sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    raise TypeError(f"cannot encode a value of type {type(value).__name__}")


def decode(value):
    """Inverse of `encode`, for trusted data only (test inputs, reference outputs)."""
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if len(value) == 1 and TUPLE in value:
            return tuple(decode(item) for item in value[TUPLE])
        if len(value) == 1 and SET in value:
            return {decode(item) for item in value[SET]}
        if len(value) == 1 and COMPLEX in value:
            return complex(*value[COMPLEX])
        if len(value) == 1 and DICT in value:
            return {decode(key): decode(item) for key, item in value[DICT]}
        return {key: decode(item) for key, item in value.items()}
    return value
