# TinyLlama Attention Decode StableHLO

This directory contains the StableHLO MLIR generated from a minimal
decode attention compute workload.

Decode processes one new query token and reads visible K/V cache
tensors. It does not update the cache.

## Purpose

This workload traces compute-only decode attention:

```text
q @ k^T
softmax
attention_probs @ v
```

This workload follows the update-first decode model:

```text
attention_decode_cache_update:
  old cache + new_k/new_v -> updated cache

attention_decode:
  q + updated k/v -> context
```

Therefore the `k` and `v` inputs are visible cache tensors: they already
include the current token position. Cached K is assumed to already
include positional encoding. Previously cached K must not receive RoPE
again.

## Regenerate

```bash
python workloads/tinyllama/attention_decode_jax.py > ir/tinyllama/attention_decode/attention_decode.stablehlo.mlir
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/attention_decode/attention_decode.stablehlo.mlir
```

## IREE

CPU:

```bash
iree-compile \
  ir/tinyllama/attention_decode/attention_decode.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/attention_decode_cpu.vmfb
```

```bash
iree-run-module \
  --module=/tmp/attention_decode_cpu.vmfb \
  --function=main \
  --input=1x8x1x8xf32=1 \
  --input=1x2x16x8xf32=1 \
  --input=1x2x16x8xf32=1
```

CUDA:

```bash
iree-compile \
  ir/tinyllama/attention_decode/attention_decode.stablehlo.mlir \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_80 \
  -o /tmp/attention_decode_cuda.vmfb
```

## Tensor Shapes

```text
q: [1, 8, 1, 8]
k: [1, 2, 16, 8]
v: [1, 2, 16, 8]

context: [1, 8, 1, 8]
```

After `repeat_kv`:

```text
k: [1, 8, 16, 8]
v: [1, 8, 16, 8]
```

## Workload Flow

```text
repeat_kv(k)
repeat_kv(v)
scores = q @ k^T / sqrt(head_dim)
attention_probs = softmax(scores)
context = attention_probs @ v
```

## Expected StableHLO

```text
stablehlo.dot_general
stablehlo.broadcast_in_dim
stablehlo.reshape
stablehlo.reduce
stablehlo.exponential
stablehlo.divide
```

## Hardware / Runtime Interpretation

Decode compute is vector-matrix attention. It has `query_seq = 1`, so
large systolic arrays may be shape-limited. It reads the visible K/V
cache for every generated token, making cache bandwidth and latency
central runtime concerns.
