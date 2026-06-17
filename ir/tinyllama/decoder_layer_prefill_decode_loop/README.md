# TinyLlama Decoder Layer Prefill + Decode Loop StableHLO

This directory contains a TinyLlama-style decoder layer workload that
performs one prefill pass, constructs the K/V cache, then runs a fixed
number of decode steps with a loop.

This is a workload trace. It is not a production LLM implementation.

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

for each decode token:
  decode hidden_state
  -> input_layernorm
  -> q_proj, k_proj, v_proj
  -> reshape/transpose into [B, H, 1, D]
  -> RoPE on Q and K for this token position
  -> decode K/V cache update at cache_position + step
  -> masked decode attention over full cache buffer
  -> o_proj
  -> residual add
  -> post_attention_layernorm
  -> MLP
  -> residual add
  -> final RMSNorm
  -> lm_head
  -> argmax token id
  -> embedding lookup
  -> next decode hidden_state for the following step
```

The decode loop uses `jax.lax.scan`. StableHLO is expected to contain a
loop-like operation such as `stablehlo.while`.

Unlike the one-step workload, this loop models autoregressive hidden
state feedback: only the initial decode hidden state is an input. Later
decode hidden states come from the previous step's selected token
embedding.

## Why Full-Cache Masking?

In a loop, the visible cache length grows each step:

```text
S + 1, S + 2, ...
```

JAX loop-carried values need static shapes, so this workload keeps the
decode attention K/V tensors at full cache shape:

```text
[B, KVH, T_MAX, D]
```

and masks invalid key positions with:

```text
key_position < valid_length
```

This differs from the one-step decoder-layer workload, where a static
visible cache slice `[0:S+1]` can be used.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/decoder_layer_prefill_decode_loop_jax.py > ir/tinyllama/decoder_layer_prefill_decode_loop/decoder_layer_prefill_decode_loop.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/decoder_layer_prefill_decode_loop_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/decoder_layer_prefill_decode_loop/decoder_layer_prefill_decode_loop.stablehlo.mlir
```

## IREE

CPU compile:

```bash
iree-compile \
  ir/tinyllama/decoder_layer_prefill_decode_loop/decoder_layer_prefill_decode_loop.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/decoder_layer_prefill_decode_loop_cpu.vmfb
```

For Intel macOS machines, run IREE compile/runtime commands in the
Linux devcontainer or another Linux environment.

## Tensor Shapes

Toy static shapes:

```text
B = 1
prefill_seq = 16
decode_steps = 4
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
initial_decode_hidden_state: [1, 1, 64]

input_norm_weight: [64]
post_norm_weight:  [64]
final_norm_weight: [64]

q_weight: [64, 64]
k_weight: [64, 16]
v_weight: [64, 16]
o_weight: [64, 64]
lm_head_weight:  [64, 128]
embedding_table: [128, 64]

gate_weight: [64, 256]
up_weight:   [64, 256]
down_weight: [256, 64]

prefill_cos/prefill_sin: [16, 8]
decode_cos/decode_sin:   [4, 8]

k_cache/v_cache: [1, 2, 32, 8]
cache_position:  scalar i32
```

Outputs:

```text
prefill_output: [1, 16, 64]
decode_outputs: [1, 4, 64]
logits:         [1, 4, 128]
next_token_ids: [1, 4]
k_cache_out:    [1, 2, 32, 8]
v_cache_out:    [1, 2, 32, 8]
```

## StableHLO Observations

Expected operation patterns include:

```text
stablehlo.while
stablehlo.dynamic_update_slice
stablehlo.dot_general
stablehlo.reduce
stablehlo.compare
stablehlo.select
stablehlo.gather
stablehlo.transpose
stablehlo.reshape
```

The loop body exposes the recurrent decode structure: one-token
projection, RoPE, cache update, masked attention over cache, output
projection, residual add, post-attention RMSNorm, MLP, final norm, LM
head, deterministic argmax, and embedding lookup.
