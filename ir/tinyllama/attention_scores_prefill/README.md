# TinyLlama Prefill Attention Scores StableHLO

This directory contains the StableHLO MLIR generated from a minimal
prefill attention-score workload.

It computes:

```text
scores = Q @ K^T / sqrt(head_dim)
```

This workload does not apply the causal mask or softmax yet.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/attention_scores_prefill_jax.py > ir/tinyllama/attention_scores_prefill/attention_scores_prefill.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/attention_scores_prefill_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/attention_scores_prefill/attention_scores_prefill.stablehlo.mlir
```

## Tensor Shapes

Toy prefill shapes:

```text
B = 1
S = 16
num_heads = 8
head_dim = 8
```

Inputs:

```text
q: [1, 8, 16, 8]
k: [1, 8, 16, 8]
```

Output:

```text
scores: [1, 8, 16, 16]
```

The score matrix contains one `[query_seq, key_seq]` matrix per batch
and attention head. Q and K use the Hugging Face attention layout
`[B, heads, S, head_dim]`.

## StableHLO Pattern

The generated StableHLO contains:

```text
stablehlo.dot_general
stablehlo.constant
stablehlo.broadcast_in_dim
stablehlo.divide
```

The `dot_general` is a batched activation x activation matmul:

```text
batch = B * num_heads = 8
M = query_seq = 16
K = head_dim = 8
N = key_seq = 16
```

This differs from projection workloads, where one operand is a model
weight. Both appear as `stablehlo.dot_general`, but operand provenance
and tensor shapes reveal their different roles.

## Hardware / Runtime Notes

Prefill attention scores are GEMM-heavy, but their shape differs from
linear projection GEMMs:

- batch dimension represents independent attention heads
- M and N scale with sequence length
- K is only `head_dim`
- score tensor storage grows as `S * S`

The divide by `sqrt(head_dim)` may be fused with the score matmul or the
next attention stage.

The next preparation workloads are causal-mask application and softmax.
