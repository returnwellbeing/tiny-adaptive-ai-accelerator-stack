import jax
import jax.numpy as jnp


def softmax(masked_scores):
    return jax.nn.softmax(masked_scores, axis=-1)


def main():
    batch = 1
    seq = 16
    num_heads = 8

    masked_scores = jnp.ones(
        (batch, num_heads, seq, seq),
        dtype=jnp.float32,
    )

    lowered = jax.jit(softmax).lower(masked_scores)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
