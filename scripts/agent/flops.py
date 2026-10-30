"""Forward-pass FLOPs per token, for matched-compute comparisons.

Per token at context length n (the token plus everything before it), a dense
decoder costs about 2 FLOPs per non-embedding weight plus attention reads of
2 * layers * n * hidden (Kaplan et al., 2020), and the output projection
costs 2 * hidden * vocab where logits are computed. A looped model with a
shared block of `looped_layers` run T times executes
`layers + (T - 1) * looped_layers` layers per token. Embedding lookups,
norms' and activations' elementwise work, and softmax are ignored.
"""
from dataclasses import dataclass

from transformers import Qwen2Config


@dataclass(frozen=True)
class FlopModel:
    layer_weights: int  # non-embedding weights of one decoder layer
    layers: int  # layers per token at T=1
    hidden: int
    vocab: int
    looped_layers: int = 0  # layers in the shared block, run once per loop

    def per_token(self, context: int, loops: int = 1, logits: bool = True) -> float:
        if context < 1 or loops < 1 or (loops > 1 and not self.looped_layers):
            raise ValueError(f"invalid context {context} or loops {loops}")
        layers = self.layers + (loops - 1) * self.looped_layers
        return 2 * layers * (self.layer_weights + context * self.hidden) + (self.logits() if logits else 0.0)

    def logits(self) -> float:
        return 2 * self.hidden * self.vocab


def qwen2_layer_weights(config: Qwen2Config) -> int:
    """Weights of one Qwen2 decoder layer: biased q/k/v, o, gated MLP and two RMSNorms."""
    hidden, kv = config.hidden_size, config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
    attention = (hidden * hidden + hidden) + 2 * (hidden * kv + kv) + hidden * hidden
    return attention + 3 * hidden * config.intermediate_size + 2 * hidden


def qwen2_flops(config: Qwen2Config, looped_layers: int = 0) -> FlopModel:
    return FlopModel(qwen2_layer_weights(config), config.num_hidden_layers, config.hidden_size, config.vocab_size,
                     looped_layers)
