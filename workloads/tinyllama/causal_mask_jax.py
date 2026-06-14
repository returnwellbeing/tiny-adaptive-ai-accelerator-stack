import jax
import jax.numpy as jnp


def apply_causal_mask(scores, causal_mask):
    causal_mask = causal_mask[None, None, :, :]
    return jnp.where(causal_mask, scores, -jnp.inf)


def main():
    batch = 1
    seq = 16
    num_heads = 8

    scores = jnp.ones((batch, num_heads, seq, seq), dtype=jnp.float32)
    causal_mask = jnp.tril(jnp.ones((seq, seq), dtype=jnp.bool_))

    lowered = jax.jit(apply_causal_mask).lower(scores, causal_mask)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
