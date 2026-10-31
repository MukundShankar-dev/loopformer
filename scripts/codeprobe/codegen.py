"""`evalplus.codegen` with vLLM's GPU memory share capped (scripts/codeprobe/vllm_share.py).

Otherwise the unmodified EvalPlus CLI. Generation executes no model code, so
it runs outside the sandbox.
"""
from evalplus.codegen import main

from scripts.codeprobe import vllm_share

if __name__ == "__main__":
    vllm_share.cap()
    main()
