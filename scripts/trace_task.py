"""Explicit autoregressive pointer traces: input/target boundaries and metrics.

One emitted uppercase letter represents one transition, not one latent loop.
Gold prefixes occur only in ordinary teacher-forced SFT, never in generation.
"""
from dataclasses import dataclass
import re
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from scripts.dataset.pointer import PointerExample, execute
from scripts.eval.pointer_task import render_task, validate_template

TRACE = re.compile(r'[A-Z](?:\s+[A-Z])*', flags=re.ASCII)


def trace_states(response: str) -> list[str] | None:
    """Accept only whitespace-separated uppercase states; never extract from prose."""
    value = response.strip()
    return value.split() if TRACE.fullmatch(value) else None


def score_trace(response: str, task: PointerExample, stop_reason: str) -> dict:
    """Separate the last returned letter, nominal prefix, and exact EOS termination."""
    states = trace_states(response)
    gold = execute(dict(task.mapping), task.initial_state, task.task_depth)
    valid = states is not None
    states = states or []
    hits = [s == t for s, t in zip(states, gold)]
    n = len(gold)
    complete = valid and len(states) >= n and all(hits[:n])
    exact = valid and len(states) == n and stop_reason == 'eos'
    final = valid and bool(states) and states[-1] == gold[-1]
    first_symbol_error = next((i + 1 for i, hit in enumerate(hits) if not hit), None)
    first_failure = first_symbol_error
    if not valid:
        first_failure = 1
    elif len(states) < n and first_failure is None:
        first_failure = len(states) + 1
    return dict(prediction=states[-1] if states else '', valid_answer=valid,
        correct=final, states=states, emitted_states=len(states),
        nominal_final_correct=valid and len(states) >= n and states[n-1] == gold[-1],
        complete_trajectory=complete, exact_stop=exact,
        joint_success=final and exact, strict_success=complete and exact,
        early_stop=valid and stop_reason == 'eos' and len(states) < n,
        late_stop=valid and stop_reason == 'eos' and len(states) > n,
        missing_stop=stop_reason != 'eos', stopped_answer_correct=final,
        correct_letter_wrong_time=final and not exact,
        first_symbol_error=first_symbol_error, first_failure=first_failure,
        prefix_correct=first_failure-1 if first_failure else n,
        malformed=not valid)


@dataclass
class TraceExample:
    task: PointerExample
    input_ids: list[int]
    response_start: int
    response_ids: list[int]


def encode_trace(task: PointerExample, tokenizer: Any, template: str, max_tokens: int) -> TraceExample:
    """Preserve the identical inference prefix and supervise every state plus EOS."""
    validate_template(template)
    prompt = tokenizer.apply_chat_template([{'role':'user','content':render_task(template, task)}],
                                           tokenize=False, add_generation_prompt=True)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    text = ' '.join(execute(dict(task.mapping), task.initial_state, task.task_depth))
    response = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.encode(prompt + text, add_special_tokens=False) != prefix + response:
        raise ValueError('Prompt/response token boundary changes under concatenation')
    if tokenizer.decode(response) != text or len(response) != task.task_depth:
        raise ValueError('Trace format must have exactly one Qwen token per state')
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError('Trace SFT requires an explicit EOS token')
    response = [*response, eos]
    ids = [*prefix, *response]
    if len(ids) > max_tokens:
        raise ValueError(f'{task.example_id}: {len(ids)} tokens exceeds {max_tokens}; no truncation')
    return TraceExample(task, ids, len(prefix), response)


def trace_collate(items: list[TraceExample], pad_id: int, device: str) -> dict[str, Tensor]:
    """Right-padded input [B,S]; response prediction positions/targets/mask [B,R]."""
    if not items:
        raise ValueError('Empty trace batch')
    b, s, r = len(items), max(len(x.input_ids) for x in items), max(len(x.response_ids) for x in items)
    inputs = torch.full((b,s), pad_id, dtype=torch.long)
    attention = torch.zeros_like(inputs)
    positions = torch.zeros((b,r),dtype=torch.long)
    targets = torch.zeros_like(positions)
    mask = torch.zeros_like(positions,dtype=torch.bool)
    for i,item in enumerate(items):
        inputs[i,:len(item.input_ids)] = torch.tensor(item.input_ids)
        attention[i,:len(item.input_ids)] = 1
        count=len(item.response_ids)
        # Position p predicts token p+1. Prompt tokens never have a target loss.
        positions[i,:count] = torch.arange(item.response_start-1,item.response_start-1+count)
        targets[i,:count] = torch.tensor(item.response_ids)
        mask[i,:count] = True
    return {k:v.to(device) for k,v in dict(input_ids=inputs,attention_mask=attention,
        positions=positions,targets=targets,target_mask=mask).items()}


def trace_loss(model: Any, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    """Standard full-vocabulary response CE, normalized per example.

    Project only supervised response positions, avoiding an unnecessary [B,S,V]
    prompt-logit tensor. Gradients equal full causal-LM CE with masked prompt loss.
    No vocabulary restriction or target-dependent decoding is introduced.
    """
    states = model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                         use_cache=False).last_hidden_state
    positions=batch['positions']
    selected=states.gather(1,positions[:,:,None].expand(-1,-1,states.shape[-1]))
    logits=model.lm_head(selected)
    targets,mask=batch['targets'],batch['target_mask']
    losses=F.cross_entropy(logits.float().flatten(0,1),targets.flatten(),reduction='none').view_as(targets)
    losses=losses*mask
    objective=(losses.sum(1)/mask.sum(1)).mean()
    predictions=logits.detach().argmax(-1)
    return objective,dict(correct=((predictions==targets)&mask).sum(),tokens=mask.sum(),
        trajectory=((predictions==targets)|~mask).all(1).sum(),examples=torch.tensor(len(targets),device=targets.device))


def set_trainable(model: Any, scope: str, start: int, end: int) -> list[str]:
    """Ordinary Qwen stays nonrecurrent; full SFT includes embeddings and LM head."""
    if scope not in ('middle','full') or not 0 <= start < end <= len(model.model.layers):
        raise ValueError('Invalid ordinary trainability scope/layer range')
    model.requires_grad_(scope=='full')
    if scope=='middle':
        for layer in model.model.layers[start:end]:layer.requires_grad_(True)
    return [name for name,p in model.named_parameters() if p.requires_grad]
