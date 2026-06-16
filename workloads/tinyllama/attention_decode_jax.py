import math

import jax
import jax.numpy as jnp


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


def main():
    batch = 1
    cache_length = 16
    num_heads = 8
    num_kv_heads = 2
    head_dim = 8

    q = jnp.ones((batch, num_heads, 1, head_dim), dtype=jnp.float32)
    k = jnp.ones((batch, num_kv_heads, cache_length, head_dim), dtype=jnp.float32)
    v = jnp.ones((batch, num_kv_heads, cache_length, head_dim), dtype=jnp.float32)

    lowered = jax.jit(attention_decode).lower(q, k, v)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
