import jax
import jax.numpy as jnp


def attention_prefill_cache_update(new_k, new_v, k_cache, v_cache):
    k_cache = jax.lax.dynamic_update_slice(k_cache, new_k, (0, 0, 0, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, new_v, (0, 0, 0, 0))
    return k_cache, v_cache


def main():
    batch = 1
    seq = 16
    max_seq = 32
    num_kv_heads = 2
    head_dim = 8

    new_k = jnp.ones((batch, num_kv_heads, seq, head_dim), dtype=jnp.float32)
    new_v = jnp.ones((batch, num_kv_heads, seq, head_dim), dtype=jnp.float32)
    k_cache = jnp.zeros((batch, num_kv_heads, max_seq, head_dim), dtype=jnp.float32)
    v_cache = jnp.zeros((batch, num_kv_heads, max_seq, head_dim), dtype=jnp.float32)

    lowered = jax.jit(attention_prefill_cache_update).lower(
        new_k,
        new_v,
        k_cache,
        v_cache,
    )
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
