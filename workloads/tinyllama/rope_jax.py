import jax
import jax.numpy as jnp


def rotate_half(x):
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return jnp.concatenate((-second, first), axis=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


def main():
    batch = 1
    seq = 16
    num_heads = 8
    num_kv_heads = 2
    head_dim = 8

    q = jnp.ones((batch, seq, num_heads, head_dim), dtype=jnp.float32)
    k = jnp.ones((batch, seq, num_kv_heads, head_dim), dtype=jnp.float32)
    cos = jnp.ones((seq, head_dim), dtype=jnp.float32)
    sin = jnp.ones((seq, head_dim), dtype=jnp.float32)

    lowered = jax.jit(apply_rope).lower(q, k, cos, sin)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
