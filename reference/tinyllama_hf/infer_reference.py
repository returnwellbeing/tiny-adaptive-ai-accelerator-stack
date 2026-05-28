#!/usr/bin/env python3
"""Run a TinyLlama reference inference with Transformers."""

from __future__ import annotations

import argparse
import time


DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_GGUF_MODEL_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
DEFAULT_GGUF_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEFAULT_PROMPT = "Explain adaptive AI accelerators in one short paragraph."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference TinyLlama inference through Hugging Face Transformers."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--use-gguf",
        action="store_true",
        help="Load the GGUF repo/file through Transformers. This can use more RAM while dequantizing.",
    )
    parser.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default="You are a concise technical assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to run on. auto picks CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model dtype passed to Transformers.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"),
        default="eager",
        help="Attention implementation passed to Transformers. Defaults to eager.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use files already present in the Hugging Face cache.",
    )
    return parser.parse_args()


def choose_device(torch, name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    return torch.device(name)


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"<|system|>\n{system_prompt}</s>\n"
        f"<|user|>\n{user_prompt}</s>\n"
        "<|assistant|>\n"
    )


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)

    device = choose_device(torch, args.device)
    model_id = DEFAULT_GGUF_MODEL_ID if args.use_gguf and args.model_id == DEFAULT_MODEL_ID else args.model_id
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"

    tokenizer_kwargs = {"local_files_only": args.local_files_only}
    model_kwargs = {
        "dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    if args.use_gguf:
        tokenizer_kwargs["gguf_file"] = args.gguf_file
        model_kwargs["gguf_file"] = args.gguf_file
        print(f"Loading {model_id} ({args.gguf_file}) on {device}...")
    else:
        print(f"Loading {model_id} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(device)
    model.eval()
    print(f"Attention implementation: {model.config._attn_implementation}")

    prompt = build_prompt(tokenizer, args.system_prompt, args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    do_sample = args.temperature > 0.0
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    elapsed = time.perf_counter() - start

    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, prompt_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\nPrompt:")
    print(args.prompt)
    print("\nGenerated:")
    print(text)
    print(f"\nGenerated {generated_ids.numel()} tokens in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
