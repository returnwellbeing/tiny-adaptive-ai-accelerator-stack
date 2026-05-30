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

### Prefill Attention

Prefill processes multiple prompt tokens at once.

StableHLO/runtime character:

- QKV projection is GEMM-heavy
- attention score matmul is activation x activation
- causal mask is applied across `[S, S]`
- softmax introduces reduction and elementwise operations
- KV cache is written for the prompt span

Hardware/runtime view:

- more throughput-oriented than decode
- attention score and value matmuls can use GEMM-like machinery
- mask and softmax introduce non-GEMM behavior

### Decode Attention

Decode processes one token at a time.

StableHLO/runtime character:

- QKV projection for one token
- read previous K/V cache
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
