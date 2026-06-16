# TinyLlama Attention Prefill Cache Update StableHLO

This directory contains the StableHLO MLIR generated from a minimal
prefill KV-cache construction workload.

Prefill starts from an empty or zero-initialized cache buffer and writes
the whole prompt K/V span into positions `[0:S]`.

## Purpose

This workload traces cache construction only. It does not compute
attention scores, softmax, or context.

The K input is assumed to already be RoPE-applied:

```text
K path: hidden_state -> k_proj -> RoPE -> K cache write
V path: hidden_state -> v_proj -> V cache write
```

The cache should not store raw hidden states or raw K before RoPE.

## Regenerate

```bash
python workloads/tinyllama/attention_prefill_cache_update_jax.py > ir/tinyllama/attention_prefill_cache_update/attention_prefill_cache_update.stablehlo.mlir
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/attention_prefill_cache_update/attention_prefill_cache_update.stablehlo.mlir
```

## IREE

CPU:

```bash
iree-compile \
  ir/tinyllama/attention_prefill_cache_update/attention_prefill_cache_update.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/attention_prefill_cache_update_cpu.vmfb
```

```bash
iree-run-module \
  --module=/tmp/attention_prefill_cache_update_cpu.vmfb \
  --function=main \
  --input=1x2x16x8xf32=1 \
  --input=1x2x16x8xf32=1 \
  --input=1x2x32x8xf32=0 \
  --input=1x2x32x8xf32=0
```

CUDA:

```bash
iree-compile \
  ir/tinyllama/attention_prefill_cache_update/attention_prefill_cache_update.stablehlo.mlir \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_80 \
  -o /tmp/attention_prefill_cache_update_cuda.vmfb
```

## Tensor Shapes

```text
new_k:   [1, 2, 16, 8]
new_v:   [1, 2, 16, 8]
k_cache: [1, 2, 32, 8]
v_cache: [1, 2, 32, 8]

updated_k_cache: [1, 2, 32, 8]
updated_v_cache: [1, 2, 32, 8]
```

## Workload Flow

```text
k_cache[:, :, 0:S, :] = new_k
v_cache[:, :, 0:S, :] = new_v
```

## Expected StableHLO

```text
stablehlo.dynamic_update_slice
stablehlo.constant
```

## Hardware / Runtime Interpretation

Prefill cache update is a bulk cache write. It is bandwidth and layout
sensitive, and it is separate from the prefill attention compute path.

StableHLO models this as a functional tensor update:

```text
k_cache_in, new_k -> k_cache_out
```

Later bufferization may lower this to an in-place slice write if
input/output aliasing is legal.
