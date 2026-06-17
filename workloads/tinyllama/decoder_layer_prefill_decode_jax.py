import math

import jax
import jax.numpy as jnp


def rms_norm(hidden_states, weight, eps=1e-6):
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + eps)
    return hidden_states * weight


def rotate_half(x):
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return jnp.concatenate((-second, first), axis=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


def qkv_projection(hidden_states, q_weight, k_weight, v_weight, head_dim):
    batch, seq, _ = hidden_states.shape
    q_out = q_weight.shape[-1]
    k_out = k_weight.shape[-1]
    v_out = v_weight.shape[-1]

    if k_out != v_out:
        raise ValueError("k_weight and v_weight must have the same output dimension")
    if q_out % head_dim != 0 or k_out % head_dim != 0:
        raise ValueError("projection output dimensions must be divisible by head_dim")

    num_heads = q_out // head_dim
    num_kv_heads = k_out // head_dim

    q = jnp.matmul(hidden_states, q_weight)
    k = jnp.matmul(hidden_states, k_weight)
    v = jnp.matmul(hidden_states, v_weight)

    q = jnp.reshape(q, (batch, seq, num_heads, head_dim))
    k = jnp.reshape(k, (batch, seq, num_kv_heads, head_dim))
    v = jnp.reshape(v, (batch, seq, num_kv_heads, head_dim))
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    return q, k, v


def repeat_kv(hidden_states, num_repeats):
    batch, num_kv_heads, seq, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :]
    repeated = jnp.broadcast_to(
        expanded,
        (batch, num_kv_heads, num_repeats, seq, head_dim),
    )
    return jnp.reshape(
        repeated,
        (batch, num_kv_heads * num_repeats, seq, head_dim),
    )


def attention_prefill(q, k, v):
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    seq = q.shape[2]
    head_dim = q.shape[3]
    num_repeats = num_heads // num_kv_heads

    k = repeat_kv(k, num_repeats)
    v = repeat_kv(v, num_repeats)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)
    scores = scores / math.sqrt(head_dim)

    query_positions = jnp.arange(seq)[:, None]
    key_positions = jnp.arange(seq)[None, :]
    causal_mask = query_positions >= key_positions
    scores = jnp.where(causal_mask[None, None, :, :], scores, -jnp.inf)

    attention_probs = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", attention_probs, v)


def attention_decode(q, k, v):
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[3]
    num_repeats = num_heads // num_kv_heads

    k = repeat_kv(k, num_repeats)
    v = repeat_kv(v, num_repeats)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)
    scores = scores / math.sqrt(head_dim)
    attention_probs = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", attention_probs, v)


def attention_output(context, o_weight):
    batch, num_heads, seq, head_dim = context.shape
    hidden = num_heads * head_dim

    context = jnp.transpose(context, (0, 2, 1, 3))
    context = jnp.reshape(context, (batch, seq, hidden))
    return jnp.matmul(context, o_weight)


def mlp(hidden_states, gate_weight, up_weight, down_weight):
    gate = jnp.matmul(hidden_states, gate_weight)
    up = jnp.matmul(hidden_states, up_weight)
    hidden_states = jax.nn.silu(gate) * up
    return jnp.matmul(hidden_states, down_weight)


def prefill_cache_update(new_k, new_v, k_cache, v_cache):
    k_cache = jax.lax.dynamic_update_slice(k_cache, new_k, (0, 0, 0, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, new_v, (0, 0, 0, 0))
    return k_cache, v_cache


def decode_cache_update(new_k, new_v, k_cache, v_cache, cache_position):
    start_indices = (0, 0, cache_position, 0)
    k_cache = jax.lax.dynamic_update_slice(k_cache, new_k, start_indices)
    v_cache = jax.lax.dynamic_update_slice(v_cache, new_v, start_indices)
    return k_cache, v_cache


def generation_tail(
    decoder_output,
    final_norm_weight,
    lm_head_weight,
    embedding_table,
):
    normalized = rms_norm(decoder_output, final_norm_weight)
    logits = jnp.matmul(normalized, lm_head_weight)
    next_token_id = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    next_decode_hidden_state = jnp.take(embedding_table, next_token_id, axis=0)
    return logits, next_token_id, next_decode_hidden_state


def decoder_layer_prefill_decode(
    prefill_hidden_states,
    decode_hidden_state,
    input_norm_weight,
    post_norm_weight,
    final_norm_weight,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    lm_head_weight,
    embedding_table,
    gate_weight,
    up_weight,
    down_weight,
    prefill_cos,
    prefill_sin,
    decode_cos,
    decode_sin,
    k_cache,
    v_cache,
    cache_position,
):
    head_dim = prefill_cos.shape[-1]
    prefill_seq = prefill_hidden_states.shape[1]

    prefill_norm = rms_norm(prefill_hidden_states, input_norm_weight)
    prefill_q, prefill_k, prefill_v = qkv_projection(
        prefill_norm,
        q_weight,
        k_weight,
        v_weight,
        head_dim,
    )
    prefill_q, prefill_k = apply_rope(
        prefill_q,
        prefill_k,
        prefill_cos,
        prefill_sin,
    )

    prefill_context = attention_prefill(prefill_q, prefill_k, prefill_v)
    prefill_attention = attention_output(prefill_context, o_weight)
    prefill_residual = prefill_hidden_states + prefill_attention
    prefill_mlp_input = rms_norm(prefill_residual, post_norm_weight)
    prefill_output = prefill_residual + mlp(
        prefill_mlp_input,
        gate_weight,
        up_weight,
        down_weight,
    )

    k_cache, v_cache = prefill_cache_update(prefill_k, prefill_v, k_cache, v_cache)

    decode_norm = rms_norm(decode_hidden_state, input_norm_weight)
    decode_q, decode_k, decode_v = qkv_projection(
        decode_norm,
        q_weight,
        k_weight,
        v_weight,
        head_dim,
    )
    decode_q, decode_k = apply_rope(decode_q, decode_k, decode_cos, decode_sin)

    k_cache, v_cache = decode_cache_update(
        decode_k,
        decode_v,
        k_cache,
        v_cache,
        cache_position,
    )
    batch, num_kv_heads, _, cache_head_dim = k_cache.shape
    visible_limit = (batch, num_kv_heads, prefill_seq + 1, cache_head_dim)
    visible_k = jax.lax.slice(k_cache, (0, 0, 0, 0), visible_limit)
    visible_v = jax.lax.slice(v_cache, (0, 0, 0, 0), visible_limit)

    decode_context = attention_decode(decode_q, visible_k, visible_v)
    decode_attention = attention_output(decode_context, o_weight)
    decode_residual = decode_hidden_state + decode_attention
    decode_mlp_input = rms_norm(decode_residual, post_norm_weight)
    decode_output = decode_residual + mlp(
        decode_mlp_input,
        gate_weight,
        up_weight,
        down_weight,
    )
    logits, next_token_id, next_decode_hidden_state = generation_tail(
        decode_output,
        final_norm_weight,
        lm_head_weight,
        embedding_table,
    )

    return (
        prefill_output,
        decode_output,
        logits,
        next_token_id,
        next_decode_hidden_state,
        k_cache,
        v_cache,
    )


def main():
    batch = 1
    prefill_seq = 16
    decode_seq = 1
    max_seq = 32
    hidden = 64
    intermediate = 256
    vocab = 128
    num_heads = 8
    num_kv_heads = 2
    head_dim = 8

    prefill_hidden_states = jnp.ones(
        (batch, prefill_seq, hidden),
        dtype=jnp.float32,
    )
    decode_hidden_state = jnp.ones((batch, decode_seq, hidden), dtype=jnp.float32)

    input_norm_weight = jnp.ones((hidden,), dtype=jnp.float32)
    post_norm_weight = jnp.ones((hidden,), dtype=jnp.float32)
    final_norm_weight = jnp.ones((hidden,), dtype=jnp.float32)

    q_weight = jnp.ones((hidden, num_heads * head_dim), dtype=jnp.float32)
    k_weight = jnp.ones((hidden, num_kv_heads * head_dim), dtype=jnp.float32)
    v_weight = jnp.ones((hidden, num_kv_heads * head_dim), dtype=jnp.float32)
    o_weight = jnp.ones((num_heads * head_dim, hidden), dtype=jnp.float32)
    lm_head_weight = jnp.ones((hidden, vocab), dtype=jnp.float32)
    embedding_table = jnp.ones((vocab, hidden), dtype=jnp.float32)

    gate_weight = jnp.ones((hidden, intermediate), dtype=jnp.float32)
    up_weight = jnp.ones((hidden, intermediate), dtype=jnp.float32)
    down_weight = jnp.ones((intermediate, hidden), dtype=jnp.float32)

    prefill_cos = jnp.ones((prefill_seq, head_dim), dtype=jnp.float32)
    prefill_sin = jnp.ones((prefill_seq, head_dim), dtype=jnp.float32)
    decode_cos = jnp.ones((decode_seq, head_dim), dtype=jnp.float32)
    decode_sin = jnp.ones((decode_seq, head_dim), dtype=jnp.float32)

    k_cache = jnp.zeros((batch, num_kv_heads, max_seq, head_dim), dtype=jnp.float32)
    v_cache = jnp.zeros((batch, num_kv_heads, max_seq, head_dim), dtype=jnp.float32)
    cache_position = jnp.array(prefill_seq, dtype=jnp.int32)

    lowered = jax.jit(decoder_layer_prefill_decode).lower(
        prefill_hidden_states,
        decode_hidden_state,
        input_norm_weight,
        post_norm_weight,
        final_norm_weight,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        lm_head_weight,
        embedding_table,
        gate_weight,
        up_weight,
        down_weight,
        prefill_cos,
        prefill_sin,
        decode_cos,
        decode_sin,
        k_cache,
        v_cache,
        cache_position,
    )
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
