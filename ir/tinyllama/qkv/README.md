# TinyLlama QKV Projection StableHLO

This directory contains the StableHLO MLIR generated from a reduced
TinyLlama-style QKV projection workload.

This is not full attention. It only covers the projection and reshape
path that prepares query, key, and value tensors.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/qkv_jax.py > ir/tinyllama/qkv/qkv.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/qkv_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/qkv/qkv.stablehlo.mlir
```

Expected workload tags:

```text
GEMM-heavy
layout-sensitive
```

## Operation

The workload starts from:

```text
hidden_states: [B, S, H]
```

Test shapes:

```text
B = 1
S = 16
H = 2048
num_heads = 32
num_kv_heads = 4
head_dim = 64
```

Projection weight shapes:

```text
q_weight: [2048, 2048]
k_weight: [2048, 256]
v_weight: [2048, 256]
```

The generated StableHLO contains three `stablehlo.dot_general`
operations:

```text
q projection: M=16 K=2048 N=2048
k projection: M=16 K=2048 N=256
v projection: M=16 K=2048 N=256
```

Then the projection outputs are reshaped and transposed into the
Hugging Face attention layout `[B, heads, S, head_dim]`:

```text
q: [1, 32, 16, 64]
k: [1, 4, 16, 64]
v: [1, 4, 16, 64]
```

The smaller K/V head count mirrors grouped-query attention in TinyLlama.
This workload uses TinyLlama's reference hidden/head dimensions while
keeping sequence length small for fast tracing.
