"""Batched prefill and decode for a Qwen2 causal LM over a static per-row KV cache.

The engine runs the model's own decoder-layer modules (projections, norms,
MLP, rotary embedding) but does attention and caching itself, so the same
code can run any layer schedule: the base model is layers 0..L-1 in order;
a looped model (M1) is P, then R repeated T times, then C. Each position in
the schedule has its own cache slot, which is the exact per-loop cache.

Cache: keys and values of shape [slots, rows, kv_heads, capacity + 1,
head_dim]. Row r holds one conversation; `lengths[r]` tokens of it are
cached. `decode` runs one token for each of the first n rows in one batched
forward; rows with no token (waiting on a tool, or empty) compute on a
placeholder whose key and value land at index `lengths[r]`, past the row's
valid prefix (the extra index keeps a full row's placeholder in bounds), and
are overwritten by that row's next real token. Attention for row r covers
indices 0..lengths[r] only.

With `graphs=True` (CUDA), decode replays a captured CUDA graph, which
removes the per-kernel launch overhead that otherwise dominates a decode
step. Graphs need fixed shapes, so a graphed step runs every row and attends
over a span rounded up to the next of SPANS (or the full capacity); the extra
indices are masked out. Prefill is never graphed.

Grouped-query attention without copying keys: the query heads that share a
key-value head are folded into the query-length axis, so attention runs over
[rows, kv_heads, groups * new_tokens, head_dim] queries against the cache
as stored.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

SPANS = (256, 512, 1024, 2048)  # graphed attention spans below the full capacity


class Engine:
    def __init__(self, model: Qwen2ForCausalLM, rows: int, capacity: int, graphs: bool = False) -> None:
        config = model.config
        if getattr(config, "use_sliding_window", False):
            raise ValueError("sliding-window attention is not supported")
        self.model = model
        self.layers = list(model.model.layers)  # the schedule; slot i caches self.layers[i]
        self.rows, self.capacity = rows, capacity
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.groups = config.num_attention_heads // config.num_key_value_heads
        parameter = next(model.parameters())
        self.device = parameter.device
        shape = (len(self.layers), rows, self.kv_heads, capacity + 1, self.head_dim)
        self.keys = torch.zeros(shape, dtype=parameter.dtype, device=self.device)
        self.values = torch.zeros_like(self.keys)
        self.lengths = [0] * rows
        if graphs and self.device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA model")
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, Tensor]] | None = {} if graphs else None
        # Static inputs of the graphed decode step.
        self._ids = torch.zeros(rows, 1, dtype=torch.long, device=self.device)
        self._starts = torch.zeros(rows, dtype=torch.long, device=self.device)
        self._row_index = torch.arange(rows, device=self.device)
        self._key_index = torch.arange(capacity + 1, device=self.device)

    def room(self, row: int) -> int:
        return self.capacity - self.lengths[row]

    def reset(self, row: int) -> None:
        self.lengths[row] = 0

    @torch.inference_mode()
    def prefill(self, row: int, tokens: list[int]) -> Tensor:
        """Append `tokens` to row `row`; logits of the last one, shape [vocab], float32."""
        start, count = self.lengths[row], len(tokens)
        if not tokens or count > self.room(row):
            raise ValueError(f"cannot prefill {count} tokens into row {row} with room {self.room(row)}")
        positions = torch.arange(start, start + count, device=self.device)
        keys = torch.arange(start + count, device=self.device)
        mask = (keys[None, :] <= positions[:, None])[None, None]  # [1, 1, count, start + count]
        writes = (slice(row, row + 1), slice(start, start + count))
        hidden = self._forward(torch.tensor([tokens], device=self.device), positions[None], mask, writes,
                               (slice(row, row + 1), start + count))
        self.lengths[row] += count
        return hidden[0]

    @torch.inference_mode()
    def decode(self, tokens: list[int | None]) -> Tensor:
        """One token for each of rows 0..len(tokens)-1 (None: no token); logits [len(tokens), vocab], float32.
        Only rows with a token advance."""
        count = len(tokens)
        if any(token is not None and self.room(row) < 1 for row, token in enumerate(tokens)):
            raise ValueError("a row with a token is full")
        ids = [0 if token is None else token for token in tokens]
        if self.graphs is None:
            logits = self._decode_eager(ids)
        else:
            self._ids.copy_(torch.tensor(ids + [0] * (self.rows - count))[:, None])
            self._starts.copy_(torch.tensor(self.lengths))
            graph, output = self._graph(self._span(max(self.lengths) + 1))
            graph.replay()
            logits = output[:count].clone()  # the next replay overwrites `output`
        for row, token in enumerate(tokens):
            self.lengths[row] += token is not None
        return logits

    def _decode_eager(self, ids: list[int]) -> Tensor:
        count = len(ids)
        starts = torch.tensor(self.lengths[:count], device=self.device)
        span = max(self.lengths[:count]) + 1
        mask = (torch.arange(span, device=self.device)[None, :] <= starts[:, None])[:, None, None]  # [n, 1, 1, span]
        writes = (torch.arange(count, device=self.device), starts)
        return self._forward(torch.tensor(ids, device=self.device)[:, None], starts[:, None], mask, writes,
                             (slice(0, count), span))

    def _span(self, needed: int) -> int:
        return next((span for span in SPANS if span >= needed and span <= self.capacity), self.capacity + 1)

    def _decode_static(self, span: int) -> Tensor:
        """The decode step over every row on the static inputs, as captured."""
        mask = (self._key_index[:span][None, :] <= self._starts[:, None])[:, None, None]
        return self._forward(self._ids, self._starts[:, None], mask, (self._row_index, self._starts),
                             (slice(0, self.rows), span))

    def _graph(self, span: int) -> tuple[torch.cuda.CUDAGraph, Tensor]:
        if span not in self.graphs:
            # Warm-up runs execute: with the static inputs at the current lengths, their cache writes land on each
            # row's placeholder index, past its valid prefix.
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._decode_static(span)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._decode_static(span)
            self.graphs[span] = (graph, output)
        return self.graphs[span]

    def _forward(self, ids: Tensor, positions: Tensor, mask: Tensor, writes: tuple, reads: tuple[slice, int]) -> Tensor:
        """`ids` and `positions` [batch, new]; `mask` [batch, 1, new, span] over cache indices 0..span-1. `writes`
        indexes (rows, cache indices) for the new keys; `reads` gives (rows, span) of the cache to attend over."""
        model = self.model.model
        hidden = model.embed_tokens(ids)
        cos, sin = model.rotary_emb(hidden, positions)
        batch, new = ids.shape
        span = mask.shape[-1]  # query row g * new + i is token i seen by the g-th head of its group
        folded = mask.expand(batch, 1, new, span)[:, :, None].expand(batch, 1, self.groups, new, span)
        folded = folded.reshape(batch, 1, self.groups * new, span)
        for slot, layer in enumerate(self.layers):
            hidden = self._layer(slot, layer, hidden, cos, sin, folded, writes, reads)
        last = model.norm(hidden[:, -1])
        return self.model.lm_head(last).float()

    def _layer(self, slot: int, layer, hidden: Tensor, cos: Tensor, sin: Tensor, mask: Tensor, writes: tuple,
               reads: tuple[slice, int]) -> Tensor:
        attention = layer.self_attn
        batch, new, _ = hidden.shape
        normed = layer.input_layernorm(hidden)
        shape = (batch, new, -1, self.head_dim)
        query = attention.q_proj(normed).view(shape).transpose(1, 2)
        key = attention.k_proj(normed).view(shape).transpose(1, 2)
        value = attention.v_proj(normed).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        self._write(slot, writes, key, value)
        rows, span = reads
        keys, values = self.keys[slot, rows, :, :span], self.values[slot, rows, :, :span]
        query = query.reshape(batch, self.kv_heads, self.groups * new, self.head_dim)
        output = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=attention.scaling)
        output = output.reshape(batch, self.kv_heads * self.groups, new, self.head_dim).transpose(1, 2)
        hidden = hidden + attention.o_proj(output.reshape(batch, new, -1))
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _write(self, slot: int, writes: tuple, key: Tensor, value: Tensor) -> None:
        rows, indices = writes
        if isinstance(rows, slice):  # prefill: one row, a contiguous range of indices
            self.keys[slot, rows, :, indices] = key
            self.values[slot, rows, :, indices] = value
        else:  # decode: row i writes its single new token at indices[i]
            self.keys[slot][rows, :, indices] = key[:, :, 0]
            self.values[slot][rows, :, indices] = value[:, :, 0]
