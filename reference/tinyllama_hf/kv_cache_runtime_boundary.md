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

For Llama-style RoPE:

```text
new K is RoPE-applied before the cache write
new V is projected but not RoPE-applied
```

The cache should not store raw `hidden_states`, and it should not store
raw K before RoPE. The runtime boundary should treat cached K as already
position-encoded.

The cache should also remain compact at `num_key_value_heads`. For
grouped-query attention, `repeat_kv` belongs after cache read and before
attention matmul:

```text
compact K/V cache
-> repeat_kv or equivalent shared-head indexing
-> attention computation
```

Storing repeated K/V heads would waste cache capacity and bandwidth.

## Prefill Boundary

During prefill:

```text
input tokens: many
new RoPE-applied K span: [B, num_kv_heads, S, head_dim]
new projected V span:    [B, num_kv_heads, S, head_dim]
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

This project traces prefill as two separate views:

```text
attention_prefill_jax.py
  q, k, v -> context

attention_prefill_cache_update_jax.py
  new_k, new_v, k_cache_in, v_cache_in
  -> k_cache_out, v_cache_out
```

The second view models cache construction with `dynamic_update_slice`.
The initial cache tensor may be zero or dummy-filled; the valid region
after prefill is `[0:S]`.

## Decode Boundary

During decode:

```text
input token: one
new RoPE-applied K: [B, num_kv_heads, 1, head_dim]
new projected V:    [B, num_kv_heads, 1, head_dim]
cache update: write one position
attention pattern: [1 x cache_length]
```

Runtime concerns:

- read all previous K/V needed by the token
- write exactly the new token position
- avoid unnecessary cache copies
- keep cache layout friendly for repeated reads
- minimize per-token latency
- apply RoPE only to the newly generated K at `cache_position`
- do not apply RoPE again to previously cached K

Decode is often dominated by small batch size, dependency chain latency,
and KV cache bandwidth.

This project traces decode as two separate views:

```text
attention_decode_jax.py
  q, k, v -> context

attention_decode_cache_update_jax.py
  new_k, new_v, k_cache_in, v_cache_in, cache_position
  -> k_cache_out, v_cache_out
```

This project uses an update-first decode model:

```text
old cache + new_k/new_v
-> attention_decode_cache_update
-> updated visible cache
-> attention_decode
```

The compute view reads visible cache tensors that already include the
current token position. The update view models a small
`cache_position`-dependent indexed write.

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

For cache-update-only traces, use smaller explicit contracts:

```text
attention_prefill_cache_update(
  new_k,
  new_v,
  k_cache_in,
  v_cache_in,
) -> (k_cache_out, v_cache_out)

attention_decode_cache_update(
  new_k,
  new_v,
  k_cache_in,
  v_cache_in,
  cache_position,
) -> (k_cache_out, v_cache_out)
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
