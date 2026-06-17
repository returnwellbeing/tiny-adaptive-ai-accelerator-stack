# TinyLlama Residual Add StableHLO

This directory contains the StableHLO MLIR generated from a
TinyLlama-style residual addition workload.

Residual additions appear after attention output projection and after
the MLP:

```text
hidden_states + sublayer_output -> updated hidden_states
```

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/residual_add_jax.py > ir/tinyllama/residual_add/residual_add.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/residual_add_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/residual_add/residual_add.stablehlo.mlir
```

## IREE

CPU compile:

```bash
iree-compile \
  ir/tinyllama/residual_add/residual_add.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/residual_add_cpu.vmfb
```

CPU run:

```bash
iree-run-module \
  --module=/tmp/residual_add_cpu.vmfb \
  --function=main \
  --input=1x16x64xf32=1 \
  --input=1x16x64xf32=1
```

## Tensor Shapes

Toy static shapes:

```text
B = 1
S = 16
hidden = 64
```

Inputs:

```text
hidden_states: [1, 16, 64]
residual:      [1, 16, 64]
```

Output:

```text
output: [1, 16, 64]
```

## StableHLO Observations

Expected operation pattern:

```text
stablehlo.add
```

This workload is elementwise-heavy and bandwidth-sensitive. It is small
by itself, but residual boundaries matter when reasoning about fusion
between attention, normalization, and MLP subgraphs.
