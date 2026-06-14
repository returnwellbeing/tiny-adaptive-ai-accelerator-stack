# TinyLlama repeat_kv StableHLO

This directory contains the StableHLO MLIR generated from a minimal
grouped-query attention `repeat_kv` workload.

`repeat_kv` expands compact K/V heads so they can be used with a larger
number of query heads.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/repeat_kv_jax.py > ir/tinyllama/repeat_kv/repeat_kv.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/repeat_kv_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/repeat_kv/repeat_kv.stablehlo.mlir
```

Expected workload tag:

```text
layout-sensitive
```

## Tensor Shapes

Toy static shapes:

```text
B = 1
S = 16
num_kv_heads = 2
num_repeats = 4
head_dim = 8
```

Input and output:

```text
hidden_states: [1, 16, 2, 8]
output:        [1, 16, 8, 8]
```

The output head count is:

```text
num_kv_heads * num_repeats = 2 * 4 = 8
```

The same helper can be applied separately to K and V.

## StableHLO Pattern

The generated StableHLO contains:

```text
stablehlo.broadcast_in_dim
stablehlo.broadcast_in_dim
stablehlo.reshape
```

There is no `stablehlo.dot_general`, reduction, or elementwise
arithmetic. This is a pure layout/expansion workload.

## KV Cache and Runtime Note

The KV cache should remain compact:

```text
K cache: [B, cache_length, num_kv_heads, head_dim]
V cache: [B, cache_length, num_kv_heads, head_dim]
```

Do not store repeated K/V heads in the cache. Repeating them would
multiply cache storage and read bandwidth.

Conceptually:

```text
compact cached K/V
-> repeat_kv
-> query-head-aligned K/V
-> attention score/value matmul
```

An optimized backend may avoid physically materializing the repeated
tensor by encoding the head-sharing relationship in indexing or kernel
logic.
