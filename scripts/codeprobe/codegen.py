"""`evalplus.codegen` with vLLM's GPU memory share capped, for the shared desktop.

EvalPlus builds `vllm.LLM` without `gpu_memory_utilization`, so vLLM would
reserve 90% of the GPU. This sets the share from $VLLM_GPU_MEMORY_UTILIZATION
(default 0.3) and otherwise runs the unmodified EvalPlus CLI. Generation
executes no model code, so it runs outside the sandbox.
"""
import os

import vllm
from evalplus.codegen import main

SHARE = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.3"))
_original_init = vllm.LLM.__init__


def _capped_init(self, *args, **kwargs):
    kwargs.setdefault("gpu_memory_utilization", SHARE)
    _original_init(self, *args, **kwargs)


vllm.LLM.__init__ = _capped_init

if __name__ == "__main__":
    main()
