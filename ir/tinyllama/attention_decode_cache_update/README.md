# TinyLlama Attention Decode Cache Update StableHLO

This directory contains the StableHLO MLIR generated from a minimal
decode KV-cache update workload.

Decode writes exactly one new K/V position at `cache_position`.

## Purpose

This workload traces cache update only. It does not compute decode
attention.

The new K input is assumed to already be RoPE-applied:

```text
K path: hidden_state -> k_proj -> RoPE -> K cache write
V path: hidden_state -> v_proj -> V cache write
```

Previously cached K already includes positional encoding and must not
receive RoPE again.

## Regenerate

```bash
python workloads/tinyllama/attention_decode_cache_update_jax.py > ir/tinyllama/attention_decode_cache_update/attention_decode_cache_update.stablehlo.mlir
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/attention_decode_cache_update/attention_decode_cache_update.stablehlo.mlir
```

## IREE

CPU:

```bash
iree-compile \
  ir/tinyllama/attention_decode_cache_update/attention_decode_cache_update.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/attention_decode_cache_update_cpu.vmfb
```

```bash
iree-run-module \
  --module=/tmp/attention_decode_cache_update_cpu.vmfb \
  --function=main \
  --input=1x2x1x8xf32=1 \
  --input=1x2x1x8xf32=1 \
  --input=1x2x32x8xf32=0 \
  --input=1x2x32x8xf32=0 \
  --input=i32=16
```

CUDA:

```bash
iree-compile \
  ir/tinyllama/attention_decode_cache_update/attention_decode_cache_update.stablehlo.mlir \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_80 \
  -o /tmp/attention_decode_cache_update_cuda.vmfb
```

## Tensor Shapes

```text
new_k:          [1, 2, 1, 8]
new_v:          [1, 2, 1, 8]
k_cache:        [1, 2, 32, 8]
v_cache:        [1, 2, 32, 8]
cache_position: scalar i32

updated_k_cache: [1, 2, 32, 8]
updated_v_cache: [1, 2, 32, 8]
```

## Workload Flow

```text
k_cache[:, :, cache_position:cache_position + 1, :] = new_k
v_cache[:, :, cache_position:cache_position + 1, :] = new_v
```

## Expected StableHLO

```text
stablehlo.dynamic_update_slice
stablehlo.compare
stablehlo.select
```

## Hardware / Runtime Interpretation

Decode cache update is a small position-dependent indexed write. It
exposes address generation, `cache_position` handling, write granularity,
and runtime-managed cache state. It is different from prefill cache
construction, which writes a contiguous prompt span.
