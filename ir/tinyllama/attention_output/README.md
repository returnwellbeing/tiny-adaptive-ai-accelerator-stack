# TinyLlama Attention Output StableHLO

This directory contains the StableHLO MLIR generated from the
TinyLlama-style attention output projection workload.

The workload starts from attention context in attention-ready layout:

```text
[B, H, S, D]
```

It converts the context back to projection layout and applies `o_proj`:

```text
context [B, H, S, D]
-> transpose [B, S, H, D]
-> reshape [B, S, H * D]
-> o_proj [H * D, hidden_size]
-> output [B, S, hidden_size]
```

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/attention_output_jax.py > ir/tinyllama/attention_output/attention_output.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/attention_output_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/attention_output/attention_output.stablehlo.mlir
```

## IREE

CPU compile:

```bash
iree-compile \
  ir/tinyllama/attention_output/attention_output.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/attention_output_cpu.vmfb
```

CPU run:

```bash
iree-run-module \
  --module=/tmp/attention_output_cpu.vmfb \
  --function=main \
  --input=1x8x16x8xf32=1 \
  --input=64x64xf32=1
```

## Tensor Shapes

Toy static shapes:

```text
B = 1
S = 16
num_heads = 8
head_dim = 8
hidden = 64
```

Inputs:

```text
context:  [1, 8, 16, 8]
o_weight: [64, 64]
```

Output:

```text
output: [1, 16, 64]
```

## StableHLO Observations

Expected operation patterns include:

```text
stablehlo.transpose
stablehlo.reshape
stablehlo.dot_general
```

This workload is layout-sensitive and GEMM-heavy. It differs from
attention score/value matmuls because one operand is a model weight.
