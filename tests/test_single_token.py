"""General-evaluation token boundaries, padding, and one-loop readout contracts."""

from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.eval.single_token import encode_pair, last_logits, padded_inputs
from scripts.recurrent_qwen.model import RecurrentQwen


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]


def test_single_token_boundary_and_rejection():
    tokenizer = CharacterTokenizer()
    assert encode_pair(tokenizer, "Answer:", "A") == (tuple(map(ord, "Answer:")), ord("A"))
    with pytest.raises(ValueError, match="exactly one"):
        encode_pair(tokenizer, "Answer: ", "A")
    with pytest.raises(ValueError, match="nonempty"):
        encode_pair(tokenizer, "", "A")


def test_retokenized_context_is_rejected():
    class BoundaryTokenizer:
        def encode(self, text, **kwargs):
            return [1] if text == "x" else [2, 3]
    with pytest.raises(ValueError, match="unchanged context"):
        encode_pair(BoundaryTokenizer(), "x", "y")


def test_last_readout_matches_full_logits_and_fresh_one_loop():
    torch.manual_seed(5)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, pad_token_id=0, attn_implementation="sdpa")).eval()
    recurrent = RecurrentQwen(deepcopy(base), 1, 2, recurrence_mode="fixed_prompt").eval()
    ids, mask = padded_inputs([(1, 2, 3), (4, 5)], pad_token_id=0, device=torch.device("cpu"))
    assert ids.tolist() == [[1, 2, 3], [0, 4, 5]]
    with torch.inference_mode():
        full = base(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1]
        compact = last_logits(base, ids, mask, recurrent=False)
        loop_one = last_logits(recurrent, ids, mask, recurrent=True)
    torch.testing.assert_close(compact, full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(loop_one, compact, atol=1e-5, rtol=1e-5)
