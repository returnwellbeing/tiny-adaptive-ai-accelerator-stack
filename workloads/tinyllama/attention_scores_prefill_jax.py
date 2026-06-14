import math

import jax
import jax.numpy as jnp


def attention_scores_prefill(q, k):
    head_dim = q.shape[-1]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)
    return scores / math.sqrt(head_dim)


def main():
    batch = 1
    seq = 16
    num_heads = 8
    head_dim = 8

    q = jnp.ones((batch, num_heads, seq, head_dim), dtype=jnp.float32)
    k = jnp.ones((batch, num_heads, seq, head_dim), dtype=jnp.float32)

    lowered = jax.jit(attention_scores_prefill).lower(q, k)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
