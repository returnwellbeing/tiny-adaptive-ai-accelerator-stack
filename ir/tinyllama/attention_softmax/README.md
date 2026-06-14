# TinyLlama Attention Softmax StableHLO

This directory contains the StableHLO MLIR generated from a minimal
attention softmax workload.

Softmax converts masked attention scores into probabilities along the
final `key_seq` dimension:

```text
attention_weights = softmax(masked_scores, axis=-1)
```

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/attention_softmax_jax.py > ir/tinyllama/attention_softmax/attention_softmax.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/attention_softmax_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/attention_softmax/attention_softmax.stablehlo.mlir
```

## Tensor Shapes

Toy prefill shape:

```text
B = 1
num_heads = 8
query_seq = 16
key_seq = 16
```

Input and output:

```text
masked_scores:     [1, 8, 16, 16]
attention_weights: [1, 8, 16, 16]
```

Each `[key_seq]` row is normalized independently. Masked `-inf` values
become zero probability.

## StableHLO Pattern

Numerically stable softmax follows:

```text
max_value = reduce_max(masked_scores, axis=-1)
shifted = masked_scores - max_value
exp_values = exp(shifted)
sum_value = reduce_sum(exp_values, axis=-1)
attention_weights = exp_values / sum_value
```

Expected StableHLO operations:

```text
stablehlo.reduce
stablehlo.maximum
stablehlo.subtract
stablehlo.exponential
stablehlo.add
stablehlo.divide
stablehlo.broadcast_in_dim
```

There is no `stablehlo.dot_general`. Softmax combines reductions and
elementwise operations.

## Hardware / Runtime Notes

- Reduction occurs independently for every batch, head, and query row.
- Reduction latency and synchronization matter.
- Intermediate tensors can create substantial memory traffic.
- Fusing causal-mask application with softmax can avoid materializing
  masked scores.
- The reduction width is `key_seq`, which grows during decode as the KV
  cache length grows.

The next workload is the attention value matmul:

```text
attention_weights @ V
```
