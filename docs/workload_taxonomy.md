# Workload Taxonomy

This project traces TinyLlama-style workloads across:

```text
Hugging Face reference
-> Minimal JAX reproduction
-> StableHLO
-> IREE
-> hardware/runtime interpretation
```

The goal is not to build a production LLM implementation. The goal is to
make workload structure visible.

## Normalization Workload

Example:

```text
RMSNorm
```

Representative files:

```text
workloads/tinyllama/rmsnorm_jax.py
ir/tinyllama/rmsnorm/rmsnorm.stablehlo.mlir
```

StableHLO character:

- reduction over hidden dimension
- elementwise square, add, rsqrt, multiply
- no `stablehlo.dot_general`

Hardware/runtime view:

- reduction-heavy
- elementwise-heavy
- bandwidth and reduction sensitive
- benefits from fusing reduction-adjacent elementwise operations

## Projection Workload

Examples:

```text
Linear
QKV projection
```

Representative files:

```text
workloads/tinyllama/linear_jax.py
workloads/tinyllama/qkv_jax.py
ir/tinyllama/linear/linear.stablehlo.mlir
ir/tinyllama/qkv/qkv.stablehlo.mlir
```

StableHLO character:

- core operation is `stablehlo.dot_general`
- QKV adds reshape/layout steps after projection
- source-level Linear and MatMul often lower to the same StableHLO op

Hardware/runtime view:

- GEMM-heavy
- layout-sensitive when reshape/transposes are present
- tiling and compute utilization matter
- systolic array mapping is relevant
- operand provenance still matters: activation x weight differs from
  activation x activation even when both appear as `dot_general`

## Activation / Gated MLP Workload

Examples:

```text
SiLU
SwiGLU
LlamaMLP
```

Representative files:

```text
workloads/tinyllama/swiglu_jax.py
workloads/tinyllama/mlp_jax.py
ir/tinyllama/swiglu/swiglu.stablehlo.mlir
ir/tinyllama/mlp/mlp.stablehlo.mlir
```

StableHLO character:

- SwiGLU is elementwise-heavy
- MLP combines projection GEMMs with elementwise gating
- `silu(x)` lowers to elementwise operations such as negate,
  exponential, add, divide, and multiply

Hardware/runtime view:

- MLP projection matmuls are GEMM-heavy
- SwiGLU gating is bandwidth/fusion sensitive
- fusion matters between activation and elementwise multiply
- data movement between projections and gating can dominate small shapes

## Attention Workload

Attention should be split into smaller workloads rather than implemented
as a full decoder block immediately.

### RoPE Preparation

Representative files:

```text
workloads/tinyllama/rope_jax.py
ir/tinyllama/rope/rope.stablehlo.mlir
```

StableHLO/runtime character:

- applies to Q and K only
- no `stablehlo.dot_general`
- uses slice, negate, concatenate, broadcast, multiply, and add
- layout + elementwise heavy

KV cache rule:

```text
K cache stores RoPE-applied key states.
V cache stores projected value states.
Previously cached K should not receive RoPE again during decode.
```

### Grouped-Query Head Expansion

Representative files:

```text
workloads/tinyllama/repeat_kv_jax.py
ir/tinyllama/repeat_kv/repeat_kv.stablehlo.mlir
```

StableHLO/runtime character:

- expands compact K/V heads to match query head count
- uses broadcast and reshape
- no GEMM, reduction, or elementwise arithmetic
- layout-sensitive and memory-sensitive

Runtime rule:

```text
keep K/V cache compact at num_kv_heads
apply repeat_kv after cache read and before attention matmul
avoid physically materializing repeated K/V when possible
```

### Prefill Attention

Prefill processes multiple prompt tokens at once.

Representative files:

```text
workloads/tinyllama/attention_scores_prefill_jax.py
ir/tinyllama/attention_scores_prefill/attention_scores_prefill.stablehlo.mlir
workloads/tinyllama/causal_mask_jax.py
ir/tinyllama/causal_mask/causal_mask.stablehlo.mlir
workloads/tinyllama/attention_softmax_jax.py
ir/tinyllama/attention_softmax/attention_softmax.stablehlo.mlir
```

StableHLO/runtime character:

- QKV projection is GEMM-heavy
- compact K/V heads are expanded for grouped-query attention
- attention score matmul is activation x activation
- causal mask is applied across `[S, S]`
- softmax introduces reduction and elementwise operations
- KV cache is written for the prompt span

Attention-score matmul:

```text
Q @ K^T / sqrt(head_dim)
[B, heads, S, D] x [B, heads, S, D]
-> [B, heads, S, S]
```

This is activation x activation `dot_general`, unlike projection
workloads that are activation x weight.

Causal-mask application:

```text
scores [B, heads, S, S] + boolean mask [S, S]
-> masked scores [B, heads, S, S]
```

The mask is broadcast across batch and heads. It is an elementwise,
layout-sensitive workload and is a natural fusion candidate for
softmax.

Attention softmax:

```text
masked scores [B, heads, query_seq, key_seq]
-> attention weights [B, heads, query_seq, key_seq]
```

Softmax reduces over `key_seq` for both maximum and sum. It combines
reduction and elementwise operations, and benefits from fusion with
causal-mask application.

Hardware/runtime view:

- more throughput-oriented than decode
- attention score and value matmuls can use GEMM-like machinery
- mask and softmax introduce non-GEMM behavior

### Decode Attention

Decode processes one token at a time.

StableHLO/runtime character:

- QKV projection for one token
- read previous K/V cache
- expand compact cached K/V heads for grouped-query attention
- attention score shape is conceptually `[B, heads, 1, cache_length]`
- update one cache position

Hardware/runtime view:

- latency-sensitive
- KV cache read bandwidth matters
- cache update granularity matters
- small-M matmuls can underutilize large systolic arrays

### KV Cache Behavior

KV cache should be modeled explicitly in future JAX workloads:

```text
k_cache_in, v_cache_in, cache_position
-> k_cache_out, v_cache_out
```

This exposes:

- cache memory layout
- cache read bandwidth
- cache update granularity
- static maximum sequence length
- decode latency concerns

Hugging Face `DynamicCache` hides these details for API convenience.
This project will expose them for workload tracing.
