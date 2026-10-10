"""Cached generation through the frozen ordinary evaluator's existing interface.

Keep the historical evaluator byte-identical: its completed baseline freezes
that source hash. This explicit adapter changes only the generation configuration
for new trace experiments, then replaces final-only scoring with trace scoring.
"""
from copy import deepcopy
import json
from typing import Any, Iterator

import torch
from transformers.generation import CompileConfig

from scripts.dataset.pointer import PointerExample
from scripts.eval.pointer_task import evaluate_batches
from scripts.trace_task import score_trace


class CachedGeneration:
    """Read-only model facade; all weights and generation execute in ordinary Qwen."""
    def __init__(self, model: Any, cache: str, compiled: bool, max_cache_len: int | None):
        if cache not in ('dynamic','static'):
            raise ValueError('Expected an explicit dynamic/static cache')
        if compiled and (cache!='static' or model.device.type!='cuda'):
            raise ValueError('Compiled decode requires a static CUDA cache')
        self.model=model
        self.device=model.device
        self.config=model.config
        self.generation_config=model.generation_config
        self.cache=cache;self.compiled=compiled;self.max_cache_len=max_cache_len

    def generate(self, **kwargs: Any) -> torch.Tensor:
        config=deepcopy(kwargs['generation_config'])
        config.use_cache=True;config.cache_implementation=self.cache
        config.disable_compile=not self.compiled
        config.compile_config=CompileConfig(fullgraph=True,mode='reduce-overhead') if self.compiled else None
        config.max_cache_len=self.max_cache_len
        config.temperature=1.;config.top_p=1.;config.top_k=50
        kwargs['generation_config']=config
        return self.model.generate(**kwargs)


def cached_batches(model: Any,tokenizer: Any,tasks: list[PointerExample],template: str,*,
        batch_size: int,max_new_tokens: int,cache: str='dynamic',compiled: bool=False,
        max_cache_len: int|None=None) -> Iterator[tuple[list[dict],float]]:
    """Reuse prompting, padding, timing and raw IDs; strictly decode complete traces."""
    adapter=CachedGeneration(model,cache,compiled,max_cache_len)
    offset=0
    for rows,seconds in evaluate_batches(adapter,tokenizer,tasks,template,batch_size=batch_size,
            max_new_tokens=max_new_tokens,prompt_format='chat'):
        for row,task in zip(rows,tasks[offset:offset+len(rows)],strict=True):
            if row['example_id']!=task.example_id:raise ValueError('Trace batch identity differs')
            ids=json.loads(row['generated_token_ids'])
            response=tokenizer.decode(ids[:-1] if row['stop_reason']=='eos' else ids,skip_special_tokens=False)
            row.update(response=response,**score_trace(response,task,row['stop_reason']))
        offset+=len(rows)
        yield rows,seconds
