"""A narrow harness adapter for exact single-token continuation likelihoods."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from lm_eval.api.model import LM
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from scripts.eval.paper_baseline import atomic_json
from scripts.eval.single_token import encode_pair, last_logits, padded_inputs


class SingleTokenLM(LM):
    """Reuse harness tasks/scoring while avoiding prompt-sized vocabulary logits.

    Only nonempty contexts with exactly one-token continuations are supported.
    Four MMLU choices sharing the encoded context use one model forward. The
    result is a full-vocabulary log probability, not a choice-normalized score.
    """

    def __init__(self, model: Any, tokenizer: Any, *, recurrent: bool,
                 batch_size: int, progress_path: Path | None = None) -> None:
        super().__init__()
        if batch_size < 1 or model.training:
            raise ValueError("Need positive batch size and an eval-mode model")
        self.model, self.tokenizer = model, tokenizer
        self.recurrent, self.batch_size = recurrent, batch_size
        self.progress_path = progress_path
        self._device = next(model.parameters()).device
        self.completed_contexts = 0

    @property
    def tokenizer_name(self) -> str:
        return "qwen2_5_0_5b_pinned"

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        groups = defaultdict(list)
        for index, request in enumerate(requests):
            context, target = encode_pair(self.tokenizer, *request.args)
            if len(context) > self.model.config.max_position_embeddings:
                raise ValueError("MMLU prompt exceeds the model context length; no truncation is allowed")
            groups[context].append((index, target))
        contexts = sorted(groups, key=len, reverse=True)
        results: list[tuple[float, bool] | None] = [None] * len(requests)
        with Progress(TextColumn("{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn()) as progress:
            task = progress.add_task("Recurrent T=1" if self.recurrent else "Original Qwen", total=len(contexts))
            with torch.inference_mode():
                for start in range(0, len(contexts), self.batch_size):
                    batch = contexts[start:start + self.batch_size]
                    ids, mask = padded_inputs(batch, pad_token_id=self.tokenizer.pad_token_id,
                                              device=self.device)
                    logits = last_logits(self.model, ids, mask, recurrent=self.recurrent)
                    log_probs = logits.float().log_softmax(-1).cpu()
                    greedy = logits.argmax(-1).cpu().tolist()
                    for row, context in enumerate(batch):
                        for index, target in groups[context]:
                            result = (float(log_probs[row, target]), greedy[row] == target)
                            results[index] = result
                            self.cache_hook.add_partial("loglikelihood", requests[index].args, result)
                    self.completed_contexts += len(batch)
                    progress.update(task, advance=len(batch))
                    if self.progress_path is not None and (start % (self.batch_size * 25) == 0 or start + len(batch) == len(contexts)):
                        atomic_json(self.progress_path, {"status": "running", "contexts_this_call": len(contexts),
                                    "completed_this_call": start + len(batch),
                                    "completed_contexts": self.completed_contexts,
                                    "remaining_seconds": progress.tasks[task].time_remaining})
        if any(result is None for result in results):
            raise RuntimeError("Missing likelihood results")
        return results

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        raise NotImplementedError("This adapter supports single-token multiple choice only")

    def generate_until(self, requests: list[Any]) -> list[str]:
        raise NotImplementedError("This adapter does not generate responses")
