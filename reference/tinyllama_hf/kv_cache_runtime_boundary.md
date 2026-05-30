# KV Cache Runtime Boundary

This note separates framework-level cache abstraction from the explicit
runtime/hardware model this project needs for accelerator analysis.

## Boundary Summary

Hugging Face reference code treats KV cache as a model-serving
convenience:

```text
model(..., past_key_values=cache, cache_position=pos, use_cache=True)
```

The accelerator/runtime model should treat KV cache as memory state:

```text
k_cache_in, v_cache_in
new_k, new_v
cache_position
-> k_cache_out, v_cache_out
```

The key distinction:

```text
Reference framework abstraction:
  hides cache mechanics behind DynamicCache

Explicit accelerator/runtime model:
  exposes cache tensors, positions, layout, reads, and writes
```

## Why Runtime Owns Cache Management

KV cache management belongs at the runtime/hardware boundary because it
is not just math. It involves:

- memory allocation
- static maximum sequence length
- physical layout
- page/block strategy
- read bandwidth
- write granularity
- device residency
- token-by-token decode latency

The model layer produces new K/V tensors. The runtime decides how those
tensors are stored, reused, and presented to the attention kernel.

## Prefill Boundary

During prefill:

```text
input tokens: many
new K/V span: [B, S, num_kv_heads, head_dim]
cache update: write S positions
attention pattern: causal [S x S]
```

Runtime concerns:

- allocate or select cache storage
- write a contiguous prompt span
- track valid sequence length
- expose the resulting cache to later decode steps

Prefill is more throughput oriented than decode because multiple prompt
tokens are available at once.

## Decode Boundary

During decode:

```text
input token: one
new K/V: [B, 1, num_kv_heads, head_dim]
cache update: write one position
attention pattern: [1 x cache_length]
```

Runtime concerns:

- read all previous K/V needed by the token
- write exactly the new token position
- avoid unnecessary cache copies
- keep cache layout friendly for repeated reads
- minimize per-token latency

Decode is often dominated by small batch size, dependency chain latency,
and KV cache bandwidth.

## Proposed Explicit Tensor Contract

When this project implements cache workloads, prefer an explicit JAX
signature like:

```text
attention_decode(
  q,
  new_k,
  new_v,
  k_cache_in,
  v_cache_in,
  cache_position,
  causal_mask_or_valid_length,
) -> (
  attention_output,
  k_cache_out,
  v_cache_out,
)
```

This is not how the Hugging Face reference API looks. That is deliberate.
The explicit signature is better for inspecting:

- memory movement
- cache update mechanics
- StableHLO tensor shapes
- IREE lowering behavior
- hardware bottlenecks

## What Not To Do Yet

Do not implement the full decoder block today.

Do not implement a custom runtime today.

Do not hide cache behavior behind a Python object in the minimal JAX
workload. Keep cache state visible as tensors when cache workloads are
introduced.
