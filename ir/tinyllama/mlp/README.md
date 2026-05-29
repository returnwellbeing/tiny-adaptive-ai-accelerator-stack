# TinyLlama MLP StableHLO

This directory contains the StableHLO MLIR generated from a reduced
TinyLlama-style MLP workload.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/mlp_jax.py > ir/tinyllama/mlp/mlp.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/mlp_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/mlp/mlp.stablehlo.mlir
```

Expected workload tags:

```text
GEMM-heavy
elementwise-heavy
```

## Operation

The workload implements the core `LlamaMLP` structure:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

For this reduced test case:

```text
x:           [1, 16, 64]
gate_weight: [64, 256]
up_weight:   [64, 256]
down_weight: [256, 64]
output:      [1, 16, 64]
```

The generated StableHLO contains three `stablehlo.dot_general`
operations:

```text
gate projection: M=16 K=64  N=256
up projection:   M=16 K=64  N=256
down projection: M=16 K=256 N=64
```

The internal SwiGLU primitive is isolated separately in:

```text
workloads/tinyllama/swiglu_jax.py
ir/tinyllama/swiglu/swiglu.stablehlo.mlir
```
