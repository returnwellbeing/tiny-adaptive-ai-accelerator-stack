# TinyLlama RoPE StableHLO

This directory contains the StableHLO MLIR generated from a minimal
Rotary Position Embedding workload.

RoPE is applied to Q and K, not V.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/rope_jax.py > ir/tinyllama/rope/rope.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/rope_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/rope/rope.stablehlo.mlir
```

Expected workload tags:

```text
elementwise-heavy
layout-sensitive
```

## Tensor Shapes

Toy static shapes:

```text
B = 1
S = 16
num_heads = 8
num_kv_heads = 2
head_dim = 8
```

Inputs:

```text
q:   [1, 16, 8, 8]
k:   [1, 16, 2, 8]
cos: [16, 8]
sin: [16, 8]
```

Outputs:

```text
q_rotated: [1, 16, 8, 8]
k_rotated: [1, 16, 2, 8]
```

## Workload Pattern

The implementation uses:

```text
rotate_half(x) = concat(-x[..., half:], x[..., :half])
rope(x) = x * cos + rotate_half(x) * sin
```

Expected StableHLO operation pattern:

```text
stablehlo.slice
stablehlo.negate
stablehlo.concatenate
stablehlo.broadcast_in_dim
stablehlo.multiply
stablehlo.add
```

There is no `stablehlo.dot_general`; RoPE is not GEMM-heavy.

## Hardware / Runtime Notes

RoPE is layout + elementwise heavy. The important concerns are:

- vectorized elementwise multiply/add
- cost of slicing and concatenating head halves
- broadcasting `cos` and `sin` across heads
- avoiding unnecessary materialization between Q/K projection and RoPE

RoPE should be considered part of the attention preparation path rather
than a projection GEMM.

## KV Cache Note

RoPE is applied to Q and K only:

```text
Q path:
hidden_state -> q_proj -> RoPE -> attention score

K path:
hidden_state -> k_proj -> RoPE -> K cache write

V path:
hidden_state -> v_proj -> V cache write
```

The KV cache should not store raw `hidden_states`.

The K cache should not store raw K before RoPE. It should store
RoPE-applied key states.

The V cache stores projected value states. V does not receive RoPE.

In decode mode:

```text
previously cached K already includes positional encoding
only newly generated K receives RoPE at the current cache_position
previously cached K must not receive RoPE again
```

This workload does not implement full attention, KV cache update, or
`repeat_kv`.
