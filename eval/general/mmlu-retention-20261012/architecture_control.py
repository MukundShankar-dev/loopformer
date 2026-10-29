import json
from pathlib import Path
import torch
from scripts.eval.loading import load_model
from scripts.eval.paper_baseline import atomic_json
from scripts.eval.single_token import encode_pair, last_logits, padded_inputs
from scripts.recurrent_qwen.model import RecurrentQwen

root = Path('eval/general/mmlu-retention-20261012')
config = json.loads((root / 'protocol.json').read_text())
spec = json.loads((Path(config['checkpoint']) / 'recurrent_config.json').read_text())
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
base, tokenizer, _ = load_model(config['base_model'], base_model=None,
    tokenizer_source=config['checkpoint'], revision=config['base_revision'],
    device='cuda', dtype='float32', download=False)
base.config._attn_implementation = 'sdpa'
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
rows = [json.loads(path.read_text().splitlines()[0]) for path in sorted((root / 'base').glob('mmlu_*.jsonl'))[:3]]
contexts = [encode_pair(tokenizer, *row['arguments'][0])[0] for row in rows]
ids, mask = padded_inputs(contexts, pad_token_id=tokenizer.pad_token_id, device=torch.device('cuda'))
with torch.inference_mode():
    expected = last_logits(base, ids, mask, recurrent=False)
    wrapper = RecurrentQwen(base, spec['recurrent_start'], spec['recurrent_end'], recurrence_mode='fixed_prompt')
    wrapper.configure_executor(tokenizer, **spec['executor'])
    wrapper.router = None
    wrapper.completion_head = None
    wrapper.eval()
    observed = last_logits(wrapper, ids, mask, recurrent=True)
    torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-5)
result = {'passed': True, 'contexts': len(contexts), 'executor_settings': spec['executor'],
          'max_full_vocabulary_logit_difference': float((observed - expected).abs().max()),
          'router_and_timer_bypassed': True, 'unadapted_R': True,
          'checkpoint_weights_modified': False}
atomic_json(root / 'architecture_control.json', result)
print(json.dumps(result), flush=True)
