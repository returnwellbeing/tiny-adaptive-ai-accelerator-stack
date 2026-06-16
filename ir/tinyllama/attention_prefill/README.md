# TinyLlama Attention Prefill StableHLO

This directory contains the first complete prefill attention workload
trace in this project.

It starts after QKV projection, reshape/transpose, and RoPE. The inputs
are already in attention-ready layout:

```text
[B, H, S, D]
```

not projection layout:

```text
[B, S, H, D]
```

RoPE is assumed to have already been applied to Q and K. RoPE is not
part of this workload.

## Workload Flow

```text
QKV Projection
-> reshape/transpose
-> RoPE
-> repeat_kv
-> attention score
-> causal mask
-> softmax
-> context
```

This workload implements:

```text
Q, K, V
-> repeat_kv(K), repeat_kv(V)
-> scores = Q @ K^T / sqrt(head_dim)
-> apply causal mask
-> attention_probs = softmax(scores)
-> context = attention_probs @ V
```

It does not implement decode attention, KV cache update, DynamicCache,
paged attention, or continuous batching.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/attention_prefill_jax.py > ir/tinyllama/attention_prefill/attention_prefill.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/attention_prefill_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/attention_prefill/attention_prefill.stablehlo.mlir
```

## IREE

CPU compile:

```bash
iree-compile \
  ir/tinyllama/attention_prefill/attention_prefill.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/attention_prefill_cpu.vmfb
```

CPU run:

```bash
iree-run-module \
  --module=/tmp/attention_prefill_cpu.vmfb \
  --function=main \
  --input=1x8x16x8xf32=1 \
  --input=1x2x16x8xf32=1 \
  --input=1x2x16x8xf32=1
```

CUDA compile:

```bash
iree-compile \
  ir/tinyllama/attention_prefill/attention_prefill.stablehlo.mlir \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_80 \
  -o /tmp/attention_prefill_cuda.vmfb
```

CUDA run:

```bash
iree-run-module \
  --module=/tmp/attention_prefill_cuda.vmfb \
  --function=main \
  --input=1x8x16x8xf32=1 \
  --input=1x2x16x8xf32=1 \
  --input=1x2x16x8xf32=1
```

## Tensor Shapes

Toy static shapes:

```text
B = 1
S = 16
num_heads = 8
num_kv_heads = 2
head_dim = 8
num_repeats = 4
```

Inputs:

```text
q: [1, 8, 16, 8]
k: [1, 2, 16, 8]
v: [1, 2, 16, 8]
```

After `repeat_kv`:

```text
k_repeated: [1, 8, 16, 8]
v_repeated: [1, 8, 16, 8]
```

Attention score:

```text
q @ k_repeated^T -> scores
[1, 8, 16, 8] x [1, 8, 8, 16] -> [1, 8, 16, 16]
```

Softmax and context:

```text
attention_probs: [1, 8, 16, 16]
context:         [1, 8, 16, 8]
```

## StableHLO Observations

Expected workload classes:

```text
GEMM-heavy
reduction-heavy
elementwise-heavy
layout-sensitive
```

Expected operation patterns include:

```text
stablehlo.dot_general
stablehlo.broadcast_in_dim
stablehlo.reshape
stablehlo.compare
stablehlo.select
stablehlo.reduce
stablehlo.exponential
stablehlo.divide
```

Two batched `dot_general` operations are expected:

```text
scores  = Q @ K^T
context = attention_probs @ V
```

Both are activation x activation matmuls, unlike projection workloads
where one operand is a model weight.

## Hardware Interpretation

Prefill attention is dominated by matrix-matrix attention. The
attention score tensor grows as `S * S`, and memory traffic grows with
sequence length.

Softmax introduces reduction-heavy behavior over the key sequence
dimension. Causal masking and softmax are natural fusion candidates.

Grouped-query attention keeps K/V compact before attention, then
`repeat_kv` expands or logically shares K/V heads for query-head-aligned
matmuls. Optimized runtimes may avoid materializing repeated K/V.

Layout matters: this workload starts in `[B, H, S, D]` attention layout,
while projection workloads begin from `[B, S, hidden]` and reshape into
head-aware attention tensors.
