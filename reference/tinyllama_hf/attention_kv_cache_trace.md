# Attention and KV Cache Trace

This note describes how TinyLlama-style attention should be understood in
this project. It is a reference trace, not an implementation plan for a
full decoder block today.

## Reference Framing

The project narrative is:

```text
Hugging Face reference
-> Minimal JAX reproduction
-> StableHLO
-> IREE
-> Hardware/runtime interpretation
```

Hugging Face provides a high-level framework abstraction. It hides many
runtime details behind objects such as `DynamicCache`. For accelerator
analysis, this project intentionally makes those details explicit in
small JAX workloads.

## Projection Path

For a decoder layer, attention starts from:

```text
hidden_states: [B, S, hidden_size]
```

TinyLlama reference shapes:

```text
hidden_size = 2048
num_attention_heads = 32
num_key_value_heads = 4
head_dim = 64
```

Projection shapes:

```text
q_proj: [B, S, 2048] x [2048, 2048] -> [B, S, 2048]
k_proj: [B, S, 2048] x [2048, 256]  -> [B, S, 256]
v_proj: [B, S, 2048] x [2048, 256]  -> [B, S, 256]
```

After reshape and transpose into the Hugging Face attention layout:

```text
q: [B, 32, S, 64]
k: [B, 4, S, 64]
v: [B, 4, S, 64]
```

The smaller number of KV heads is grouped-query attention. At attention
time, K/V are logically shared across groups of query heads.

## Prefill Attention Path

Prefill processes a prompt span with `S > 1`.

Conceptual flow:

```text
hidden_states
-> q_proj, k_proj, v_proj
-> reshape into heads
-> apply rotary position embedding to q and k
-> write RoPE-applied k and projected v into cache positions for this prompt span
-> compute causal self-attention over prompt tokens
-> o_proj
```

Causal mask use:

```text
token i can attend to tokens <= i
token i cannot attend to tokens > i
```

For prefill, attention score shape is conceptually:

```text
scores: [B, num_heads, S, S]
```

The score computation is:

```text
scores = Q @ K^T / sqrt(head_dim)
```

This is a batched activation x activation matmul. It differs from QKV
projection, where the second operand is a model weight. Causal-mask
application and softmax occur after score computation and are kept as
separate tracing workloads.

The causal mask is visible in the reference model as a framework-level
attention mask. In a minimal accelerator model, this should become an
explicit input or a generated static mask when shapes are fixed.

The current minimal causal-mask workload models it as an explicit
boolean `[S, S]` input shared across batch and heads:

```text
masked_scores = where(causal_mask, scores, -inf)
```

This keeps mask application visible while leaving mask generation and
runtime policy outside the workload.

Softmax then normalizes each query row over the key sequence:

```text
masked_scores [B, num_heads, query_seq, key_seq]
-> attention_weights [B, num_heads, query_seq, key_seq]
```

The reduction width is `key_seq`. During decode, this corresponds to
the visible cache length and grows as tokens are generated.

In this project, prefill is split into two workload views:

```text
attention_prefill_jax.py
  compute-only causal attention

attention_prefill_cache_update_jax.py
  bulk K/V cache construction for positions [0:S]
```

The split is intentional. In a serving runtime, both happen in the
prefill phase, but separating them makes compute behavior and cache
write behavior visible independently in StableHLO.

## Decode Attention Path

Decode processes one new token at a time, usually with:

```text
S = 1
```

Conceptual flow:

```text
new hidden_state
-> q_proj, k_proj, v_proj
-> reshape into heads
-> apply rotary position embedding at cache_position
-> read previous K/V cache
-> append or update current RoPE-applied K and projected V at cache_position
-> attention over [past tokens + current token]
-> o_proj
```

Decode attention score shape is conceptually:

```text
scores: [B, num_heads, 1, cache_length]
```

Decode is latency sensitive because each token depends on the previous
token. KV cache read bandwidth and cache update granularity become
visible bottlenecks.

In this project, decode is split into two workload views:

```text
attention_decode_jax.py
  compute-only decode attention using updated visible K/V cache

attention_decode_cache_update_jax.py
  incremental K/V cache update at cache_position
```

The decode ordering used here is update-first:

```text
attention_decode_cache_update:
  old cache + new_k/new_v -> updated cache

attention_decode:
  q + updated k/v -> context
```

Decode compute reads the compact visible K/V cache, applies `repeat_kv`
for query-head-aligned attention, and produces one-token context. The
visible cache already includes the current token position.

The integrated decoder-layer workload composes this rule with the rest
of the layer:

```text
decoder_layer_prefill_decode_jax.py
  prefill layer compute
  prefill K/V cache update
  decode projection and RoPE
  decode K/V cache update at cache_position
  visible cache slice [0:S+1]
  decode layer compute
  final norm, lm_head, argmax, embedding lookup
```

It still follows the update-first decode ordering. The decode attention
path reads a visible cache that already includes the current token K/V.

The looped decoder-layer workload keeps the same ordering but repeats
the decode part with static loop-carried cache tensors:

```text
decoder_layer_prefill_decode_loop_jax.py
  prefill layer compute
  prefill K/V cache update
  for each decode token:
    decode projection and RoPE
    decode K/V cache update at cache_position + step
    masked decode attention over [B, KVH, T_MAX, D]
    decode layer compute
    final norm, lm_head, argmax, embedding lookup
```

Inside the loop, the visible length is represented as a mask rather than
a smaller slice because each loop iteration must keep the same tensor
shape.

The generation tail is also traced separately as a primitive workload:

```text
generation_tail_jax.py
  decoder output
  -> final RMSNorm
  -> lm_head logits
  -> argmax token id
  -> embedding lookup for next decode hidden state
```

Sampling policies such as temperature, top-k, top-p, and random sampling
are not part of that workload yet.

Important RoPE/cache rule:

```text
Q path: hidden_state -> q_proj -> RoPE -> attention score
K path: hidden_state -> k_proj -> RoPE -> K cache write
V path: hidden_state -> v_proj -> V cache write
```

KV cache should not store raw `hidden_states`.

K cache should not store raw K before RoPE. K cache should store
RoPE-applied key states.

V cache stores projected value states. V is not RoPE-applied.

In decode mode, previously cached K already includes positional
encoding. Only the newly generated K should receive RoPE using the
current `cache_position`; previously cached K should not receive RoPE
again.

Cache update workloads assume the incoming K tensor is already
RoPE-applied. They do not implement RoPE internally.

## Grouped-Query Attention and repeat_kv

TinyLlama has more query heads than KV heads:

```text
num_attention_heads = 32
num_key_value_heads = 4
```

K/V cache should remain compact with 4 KV heads. For attention
computation, each compact KV head is shared by a group of query heads.
The reference `repeat_kv` helper expresses this sharing as an expansion:

```text
compact cached K/V
-> repeat_kv
-> query-head-aligned K/V
-> attention score/value matmul
```

Do not store repeated K/V heads in the cache. That would increase cache
storage and read bandwidth by the repeat factor. A backend may also
avoid materializing the repeated tensors and implement the sharing in
indexing or attention-kernel logic.

## KV Cache Shape

A practical explicit cache tensor shape for this project is:

```text
k_cache: [B, num_key_value_heads, max_seq, head_dim]
v_cache: [B, num_key_value_heads, max_seq, head_dim]
```

For TinyLlama reference sizes:

```text
k_cache: [B, 4, max_seq, 64]
v_cache: [B, 4, max_seq, 64]
```

For current small JAX workloads, equivalent toy shapes might use:

```text
B = 1
max_seq = 16 or 32
num_heads = 8
num_key_value_heads = 2
head_dim = 8
```

## Hugging Face DynamicCache

Hugging Face uses cache abstractions such as `DynamicCache` to manage
past keys and values. That abstraction is convenient for model code:

```text
past_key_values
cache_position
use_cache
```

The framework hides:

- cache allocation policy
- physical memory layout
- append/update mechanics
- cache read bandwidth
- cache write granularity
- static maximum sequence length decisions

That is the right abstraction for a general-purpose model API, but it is
too implicit for accelerator analysis.

## Explicit JAX Cache Model

This project will model KV cache explicitly as tensor inputs and outputs
when we reach cache workloads:

```text
inputs:
  q
  new_k
  new_v
  k_cache_in
  v_cache_in
  cache_position
  causal_mask or valid_length

outputs:
  attention_output
  k_cache_out
  v_cache_out
```

Explicit tensors expose:

- cache memory layout
- cache read bandwidth
- cache update granularity
- cache position
- static max sequence length
- decode latency concerns

This distinction matters:

```text
HF DynamicCache hides cache management.
Explicit JAX cache tensors expose runtime/hardware behavior.
```

Today, we only prepare the projection workload and reference analysis.
Full attention and KV cache update workloads are intentionally deferred.
