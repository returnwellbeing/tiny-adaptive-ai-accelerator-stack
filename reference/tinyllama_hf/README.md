# TinyLlama Reference

TinyLlama reference model inference and analysis notes live under
`reference/tinyllama_hf`.

## Goal

Use Hugging Face Transformers as the reference implementation.

We do not treat this as the accelerator implementation.
Instead, we use it to understand:

- model config
- module hierarchy
- tensor shapes
- valid breakpoint points
- reference behavior of each sublayer

## Accelerator Boundary

Tokenizer and embedding are outside the initial accelerator boundary.

Initial accelerator input:

```text
hidden_states: [batch, seq, hidden]
```

## Run

From the repository root:

```bash
python reference/tinyllama_hf/infer_reference.py
```

Use a custom prompt:

```bash
python reference/tinyllama_hf/infer_reference.py \
  --prompt "Explain TinyLlama attention in one short paragraph." \
  --max-new-tokens 80
```

Run with GGUF loading through Transformers:

```bash
python reference/tinyllama_hf/infer_reference.py --use-gguf
```

GGUF loading may use a lot of RAM because Transformers converts and
de-quantizes GGUF tensors during model load.

## Notes

The script uses Hugging Face Transformers with `AutoModelForCausalLM`.
By default it loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and sets
`attn_implementation="eager"` so the Llama eager attention path is used
instead of SDPA.
