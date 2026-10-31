"""Cap vLLM's GPU memory share on the shared desktop.

The official runners build `vllm.LLM` without `gpu_memory_utilization`, so
vLLM would reserve 90% of the GPU. `cap()` makes the default the share in
$VLLM_GPU_MEMORY_UTILIZATION (0.3 if unset); an explicit argument still wins.
"""
import os

import vllm


def cap() -> None:
    share = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.3"))
    original = vllm.LLM.__init__

    def capped(self, *args, **kwargs):
        kwargs.setdefault("gpu_memory_utilization", share)
        original(self, *args, **kwargs)

    vllm.LLM.__init__ = capped
