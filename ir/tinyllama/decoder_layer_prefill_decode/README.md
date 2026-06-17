# TinyLlama Decoder Layer Prefill + Decode StableHLO

This directory contains a single TinyLlama-style decoder layer workload
that performs one prefill pass, updates the K/V cache, then performs one
decode-token pass and updates the K/V cache for that token.

This is still a workload trace, not a production LLM implementation.

## Workload Flow

```text
prefill hidden_states
-> input_layernorm
-> q_proj, k_proj, v_proj
-> reshape/transpose into [B, H, S, D]
-> RoPE on Q and K
-> prefill attention
-> o_proj
-> residual add
-> post_attention_layernorm
-> MLP
-> residual add
-> prefill K/V cache update

decode hidden_state
-> input_layernorm
-> q_proj, k_proj, v_proj
-> reshape/transpose into [B, H, 1, D]
-> RoPE on Q and K
-> decode K/V cache update at cache_position
-> slice visible cache [all B, all KVH, 0:S+1, all D]
-> decode attention
-> o_proj
-> residual add
-> post_attention_layernorm
-> MLP
-> residual add
-> final RMSNorm
-> lm_head
-> argmax token id
-> embedding lookup for next decode hidden state
```

The decode path follows the update-first model used by the smaller
decode workloads in this project: the new token K/V is written to cache
before decode attention reads the visible cache.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/decoder_layer_prefill_decode_jax.py > ir/tinyllama/decoder_layer_prefill_decode/decoder_layer_prefill_decode.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/decoder_layer_prefill_decode_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/decoder_layer_prefill_decode/decoder_layer_prefill_decode.stablehlo.mlir
```

## IREE

CPU compile:

```bash
iree-compile \
  ir/tinyllama/decoder_layer_prefill_decode/decoder_layer_prefill_decode.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/decoder_layer_prefill_decode_cpu.vmfb
```

For Intel macOS machines, run IREE compile/runtime commands in the
Linux devcontainer or another Linux environment. The current macOS PyPI
IREE wheels may provide arm64-only binaries even when tagged as
universal2.

## Tensor Shapes

Toy static shapes:

```text
B = 1
prefill_seq = 16
decode_seq = 1
max_seq = 32
hidden = 64
intermediate = 256
vocab = 128
num_heads = 8
num_kv_heads = 2
head_dim = 8
```

Inputs:

```text
prefill_hidden_states: [1, 16, 64]
decode_hidden_state:   [1, 1, 64]

input_norm_weight: [64]
post_norm_weight:  [64]
final_norm_weight: [64]

q_weight: [64, 64]
k_weight: [64, 16]
v_weight: [64, 16]
o_weight: [64, 64]
lm_head_weight: [64, 128]
embedding_table: [128, 64]

gate_weight: [64, 256]
up_weight:   [64, 256]
down_weight: [256, 64]

prefill_cos/prefill_sin: [16, 8]
decode_cos/decode_sin:   [1, 8]

k_cache/v_cache: [1, 2, 32, 8]
cache_position:  scalar i32
```

Outputs:

```text
prefill_output:           [1, 16, 64]
decode_output:            [1, 1, 64]
logits:                   [1, 1, 128]
next_token_id:            [1, 1]
next_decode_hidden_state: [1, 1, 64]
k_cache_out:              [1, 2, 32, 8]
v_cache_out:              [1, 2, 32, 8]
```

Decode attention reads a visible cache slice after the decode token has
been written:

```text
k_cache/v_cache [B, KVH, T_MAX, D]
-> visible cache [B, KVH, S + 1, D]
```

The batch, KV-head, and head-dimension axes are taken from the cache
shape rather than hard-coded to the toy values.

## StableHLO Observations

Expected workload classes:

```text
GEMM-heavy
reduction-heavy
elementwise-heavy
layout-sensitive
cache-update
```

Expected operation patterns include:

```text
stablehlo.dot_general
stablehlo.transpose
stablehlo.reshape
stablehlo.broadcast_in_dim
stablehlo.dynamic_update_slice
stablehlo.slice
stablehlo.reduce
stablehlo.exponential
stablehlo.divide
stablehlo.rsqrt
stablehlo.gather
```

The workload contains both activation x weight projection matmuls and
activation x activation attention matmuls. It also exposes K/V cache
write behavior with `dynamic_update_slice`, vocabulary reduction for
argmax, and embedding-table lookup.

## Hardware Interpretation

This trace combines the major decoder-layer behaviors that were
previously isolated:

- RMSNorm reductions over hidden dimension
- Q/K/V/O and MLP projection GEMMs
- RoPE layout-sensitive elementwise operations
- prefill matrix-matrix attention
- decode vector-matrix attention over visible cache
- bulk and single-token K/V cache updates
- final norm, LM head, deterministic argmax, and embedding lookup

Prefill is throughput-oriented and includes an `S x S` attention score
tensor. Decode is latency-sensitive and uses a one-token query with
visible cache length `S + 1`. The generation tail is deterministic:
sampling policies such as temperature, top-k, top-p, and random sampling
are not included.
