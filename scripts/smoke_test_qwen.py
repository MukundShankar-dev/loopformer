"""Load the instruct-tuned Qwen checkpoint and generate one short response."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--prompt", default="What color is the sky on a clear day?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--local-files-only", action="store_true", help="Use cached files only."
    )
    parser.add_argument(
        "--revision", default="main", help="Hugging Face branch, tag, or commit hash."
    )
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must contain text")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    # Keep --help and argument validation available before dependencies are installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this environment; use --device cpu")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_id} (revision={args.revision})", flush=True)
    print(f"Device: {args.device}; dtype: float32; use_cache: False", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=args.revision, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        dtype=torch.float32,
        use_safetensors=True,
    ).to(args.device)
    model.eval()
    model.config.use_cache = False

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
        {"role": "user", "content": args.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(args.device)
    
    # Use greedy decoding without inheriting the checkpoint's sampling settings.
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=False,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation_config)

    # Generated sequences include the prompt; decode only the continuation.
    answer_ids = outputs[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    if not answer:
        raise RuntimeError("The model generated no visible answer.")
    print(f"Prompt: {args.prompt}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
