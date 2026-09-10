import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


@pytest.fixture
def base():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    config = Qwen2Config(
        vocab_size=47,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture
def inputs():
    return torch.tensor([[3, 5, 7, 9], [11, 13, 17, 19]])
