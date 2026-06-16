# TinyLlama Causal Mask StableHLO

This directory contains the StableHLO MLIR generated from a minimal
prefill causal-mask application workload.

It applies a boolean lower-triangular mask to attention scores:

```text
masked_scores = where(causal_mask, scores, -inf)
```

Mask generation is intentionally outside the jitted workload. This
trace focuses on applying a runtime- or framework-provided mask.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/causal_mask_jax.py > ir/tinyllama/causal_mask/causal_mask.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/causal_mask_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/causal_mask/causal_mask.stablehlo.mlir
```

## Tensor Shapes

Toy prefill shapes:

```text
B = 1
num_heads = 8
query_seq = 16
key_seq = 16
```

Inputs and output:

```text
scores:        [1, 8, 16, 16]
causal_mask:   [16, 16]
masked_scores: [1, 8, 16, 16]
```

The mask is shared across batch and head dimensions. A `true` entry
keeps the score, while a `false` entry replaces it with negative
infinity so softmax assigns zero probability.

## StableHLO Pattern

Expected StableHLO operations:

```text
stablehlo.broadcast_in_dim
stablehlo.constant
stablehlo.select
```

There is no `stablehlo.dot_general` or reduction. Causal-mask
application is elementwise and layout-sensitive.

## Hardware / Runtime Notes

- The mask must broadcast across batch and heads.
- Applying the mask is bandwidth-sensitive unless fused into softmax.
- Materializing a full `[S, S]` mask costs memory and bandwidth.
- A specialized attention kernel may encode causality in indexing
  instead of reading a materialized mask.

Prefill uses a causal `[S, S]` relationship. Decode commonly has
`query_seq = 1` and masks or valid-length logic over the current cache
length.

The next workload is softmax over the final key dimension:

```text
workloads/tinyllama/softmax_jax.py
```
