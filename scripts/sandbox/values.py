"""Tagged JSON for Python values that cross the sandbox boundary in call tests.

Plain JSON turns tuples and sets into lists, but benchmark checkers compare
with Python `==`, where `(1, 2) != [1, 2]`. Tuples, sets and dicts with
non-string keys are wrapped in single-key tagged objects, so the host can
compare encoded forms structurally without decoding untrusted output.
Integers of BIG_INT or more travel as tagged decimal strings: Python refuses
to convert integers over 4,300 digits to or from text by default, and the
host should never parse huge untrusted integers. Equal integers have equal
decimal strings, so comparing the strings matches `==`. Code that encodes or
decodes such integers must lift the limit (sys.set_int_max_str_digits(0)).

This file is also copied into the sandbox next to the call harness, so it
uses only the standard library.
"""
import json

TUPLE, SET, DICT, COMPLEX, INT = "__tuple__", "__set__", "__dict__", "__complex__", "__int__"
TAGS = (TUPLE, SET, DICT, COMPLEX, INT)
BIG_INT = 10 ** 4000  # 4,001 digits, below the default 4,300-digit conversion limit


def encode(value):
    """A JSON-serializable form of `value`; raises TypeError for unsupported types."""
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= BIG_INT:
        return {INT: str(value)}
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


def decode(value, hashable: bool = False):
    """Inverse of `encode`, for trusted data only (test inputs, reference outputs).

    Set elements and dict keys decode with `hashable`, which turns sets into frozensets: a set of frozensets
    encodes like a set of sets, and frozenset({1}) == {1} under Python `==`.
    """
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if len(value) == 1 and TUPLE in value:
            return tuple(decode(item, hashable) for item in value[TUPLE])
        if len(value) == 1 and SET in value:
            items = (decode(item, hashable=True) for item in value[SET])
            return frozenset(items) if hashable else set(items)
        if len(value) == 1 and COMPLEX in value:
            return complex(*value[COMPLEX])
        if len(value) == 1 and INT in value:
            return int(value[INT])
        if len(value) == 1 and DICT in value:
            return {decode(key, hashable=True): decode(item) for key, item in value[DICT]}
        return {key: decode(item) for key, item in value.items()}
    return value
