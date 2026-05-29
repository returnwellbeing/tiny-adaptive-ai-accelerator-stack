# TinyLlama Linear StableHLO

This directory contains the StableHLO MLIR generated from a TinyLlama-style
linear projection workload.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/linear_jax.py > ir/tinyllama/linear/linear.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/linear_jax.py
```

## Summarize

Use the StableHLO summary tool to inspect the `dot_general` shape and
GEMM interpretation:

```bash
python tools/summarize_stablehlo.py ir/tinyllama/linear/linear.stablehlo.mlir
```

Expected interpretation:

```text
lhs = tensor<1x16x64xf32>
rhs = tensor<64x256xf32>
output = tensor<1x16x256xf32>
M = 16
K = 64
N = 256
```

Compared with RMSNorm, this workload is compute-heavy and GEMM-centric
because its core operation is `stablehlo.dot_general`.
