# TinyLlama RMSNorm StableHLO

This directory contains the StableHLO MLIR generated from the TinyLlama
RMSNorm JAX workload.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/rmsnorm_jax.py > ir/tinyllama/rmsnorm/rmsnorm.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/rmsnorm_jax.py
```

## Summarize

Use the StableHLO summary tool to inspect operation counts, tensor type
hints, and workload-level interpretation:

```bash
python tools/summarize_stablehlo.py ir/tinyllama/rmsnorm/rmsnorm.stablehlo.mlir
```

The summary includes static estimates for FLOP-equivalent work, minimum
DRAM bytes, arithmetic intensity, and approximate systolic array usage.
For this RMSNorm workload, no `dot_general` is present, so systolic array
utilization is reported as not applicable / approximately zero.

## Run with IREE

Compile the StableHLO MLIR to an IREE VM flatbuffer:

```bash
iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  ir/tinyllama/rmsnorm/rmsnorm.stablehlo.mlir \
  -o /tmp/rmsnorm.vmfb
```

Run the compiled module with simple all-ones inputs:

```bash
iree-run-module \
  --module=/tmp/rmsnorm.vmfb \
  --function=main \
  --input=1x16x64xf32=1 \
  --input=64xf32=1
```

The first input is `hidden_states`; the second input is the RMSNorm
`weight`. With both inputs set to ones, the output should remain close to
ones because `rsqrt(mean(square(1)) + 1e-6)` is approximately `1`.

## StableHLO Walkthrough

The generated IR implements:

```text
y = x * rsqrt(mean(square(x), axis=-1, keepdims=True) + eps) * weight
```

For this test case:

- `x`: `tensor<1x16x64xf32>`
- `weight`: `tensor<64xf32>`
- output: `tensor<1x16x64xf32>`
- reduction axis: hidden dimension, axis `2`

Line-by-line meaning:

```text
module @jit_rms_norm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
```

Defines the compiled JAX module. The attributes say this IR is for one
partition and one replica.

```text
func.func public @main(%arg0: tensor<1x16x64xf32>, %arg1: tensor<64xf32>) -> (tensor<1x16x64xf32> {jax.result_info = "result"}) {
```

Defines the entry function. `%arg0` is `hidden_states` with shape
`[batch=1, seq=16, hidden=64]`. `%arg1` is the RMSNorm weight with shape
`[hidden=64]`. The function returns normalized hidden states with the
same shape as `%arg0`.

```text
%0 = chlo.square %arg0 : tensor<1x16x64xf32> -> tensor<1x16x64xf32>
```

Squares every element of `hidden_states`.

```text
%cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
```

Creates the scalar zero initializer for the reduction.

```text
%1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x16x64xf32>, tensor<f32>) -> tensor<1x16xf32>
```

Sums the squared values across hidden dimension `2`. This produces one
sum per `[batch, seq]` position, with shape `[1, 16]`.

```text
%2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<1x16x1xf32>
```

Adds back a singleton hidden dimension, changing `[1, 16]` into
`[1, 16, 1]`.

```text
%cst_0 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
```

Creates the scalar divisor `64.0`, the hidden size.

```text
%3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
```

Broadcasts the scalar hidden size to shape `[1, 16, 1]`.

```text
%4 = stablehlo.divide %2, %3 : tensor<1x16x1xf32>
```

Computes the mean of squared hidden values:
`sum(square(x)) / hidden_size`.

```text
%cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
```

Creates the RMSNorm epsilon value, approximately `1e-6`.

```text
%5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x16x1xf32>
```

Broadcasts epsilon to shape `[1, 16, 1]`.

```text
%6 = stablehlo.add %4, %5 : tensor<1x16x1xf32>
```

Adds epsilon to the variance estimate.

```text
%7 = stablehlo.rsqrt %6 : tensor<1x16x1xf32>
```

Computes reciprocal square root: `1 / sqrt(mean(square(x)) + eps)`.

```text
%8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x16x1xf32>) -> tensor<1x16x64xf32>
```

Broadcasts the scale factor across the hidden dimension.

```text
%9 = stablehlo.multiply %arg0, %8 : tensor<1x16x64xf32>
```

Applies RMS normalization to `hidden_states`.

```text
%10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
```

Reshapes the weight vector so it lines up with the hidden dimension.

```text
%11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x1x64xf32>) -> tensor<1x16x64xf32>
```

Broadcasts the weight across batch and sequence positions.

```text
%12 = stablehlo.multiply %9, %11 : tensor<1x16x64xf32>
```

Applies the learned RMSNorm weight.

```text
return %12 : tensor<1x16x64xf32>
```

Returns the final RMSNorm output.
