import hashlib
import json
import platform
from pathlib import Path
import torch
import transformers
from transformers import Qwen2ForCausalLM
from scripts.recurrent_qwen.checkpoint import load_recurrent_checkpoint

checkpoint = Path('models/stage1_pointer/controller-shared-number-precision-seed83/best')
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model, tokenizer, spec = load_recurrent_checkpoint(checkpoint, device='cuda', download=False)
model.config._attn_implementation = 'sdpa'
model.requires_grad_(False)
model.router = None
model.completion_head = None
question = 'What color is the sky?'
prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)
encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
initial_ids = encoded['input_ids'].to('cuda')
initial_mask = encoded['attention_mask'].to('cuda')
results = []

def generate(label, forward, loops):
    ids, mask = initial_ids.clone(), initial_mask.clone()
    generated = []
    with torch.inference_mode():
        for _ in range(32):
            scores = forward(ids, mask)
            token = int(scores[0].argmax())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
            ids = torch.cat((ids, torch.tensor([[token]], device='cuda')), dim=1)
            mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device='cuda')), dim=1)
    row = {'model': label, 'loops_per_token': loops, 'generated_tokens': len(generated),
           'total_recurrent_loops': loops * len(generated), 'eos_reached': generated[-1] == tokenizer.eos_token_id,
           'token_ids': generated, 'answer': tokenizer.decode(generated, skip_special_tokens=True)}
    results.append(row)
    print(json.dumps(row), flush=True)

for loops in (1, 2, 4):
    generate('recurrent_pointer', lambda ids, mask: model(input_ids=ids, attention_mask=mask, num_loops=loops).loop_logits[-1], loops)

base = Qwen2ForCausalLM.from_pretrained(spec['base_model'], revision=spec['revision'], local_files_only=True,
                                      dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval()
base.requires_grad_(False)
generate('original_qwen', lambda ids, mask: base(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1], 0)
report = {'question': question, 'prompt': prompt, 'prompt_token_ids': initial_ids[0].tolist(),
          'checkpoint': str(checkpoint), 'weights_sha256': hashlib.sha256((checkpoint / 'adapter_model.pt').read_bytes()).hexdigest(),
          'base_model': spec['base_model'], 'base_revision': spec['revision'],
          'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
          'device': torch.cuda.get_device_name(), 'dtype': 'float32', 'attention': 'sdpa',
          'decoding': 'greedy', 'max_new_tokens': 32, 'use_cache': False,
          'router_bypassed': True, 'timer_bypassed': True, 'reentry_bridge_preserved': True,
          'token_feedback': 'append generated token IDs directly; re-encode neither prompt nor response',
          'results': results}
Path('/tmp/loopformer_loop_compare.json').write_text(json.dumps(report, indent=2) + '\n')
