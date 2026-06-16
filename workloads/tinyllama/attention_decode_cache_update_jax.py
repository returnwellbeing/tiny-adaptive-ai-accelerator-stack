import jax
import jax.numpy as jnp


def attention_decode_cache_update(new_k, new_v, k_cache, v_cache, cache_position):
    start_indices = (0, 0, cache_position, 0)
    k_cache = jax.lax.dynamic_update_slice(k_cache, new_k, start_indices)
    v_cache = jax.lax.dynamic_update_slice(v_cache, new_v, start_indices)
    return k_cache, v_cache


def main():
    batch = 1
    max_seq = 32
    num_kv_heads = 2
    head_dim = 8

    new_k = jnp.ones((batch, num_kv_heads, 1, head_dim), dtype=jnp.float32)
    new_v = jnp.ones((batch, num_kv_heads, 1, head_dim), dtype=jnp.float32)
    k_cache = jnp.zeros((batch, num_kv_heads, max_seq, head_dim), dtype=jnp.float32)
    v_cache = jnp.zeros((batch, num_kv_heads, max_seq, head_dim), dtype=jnp.float32)
    cache_position = jnp.array(16, dtype=jnp.int32)

    lowered = jax.jit(attention_decode_cache_update).lower(
        new_k,
        new_v,
        k_cache,
        v_cache,
        cache_position,
    )
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
