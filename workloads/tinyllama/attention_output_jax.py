import jax
import jax.numpy as jnp


def attention_output(context, o_weight):
    batch, num_heads, seq, head_dim = context.shape
    hidden = num_heads * head_dim

    context = jnp.transpose(context, (0, 2, 1, 3))
    context = jnp.reshape(context, (batch, seq, hidden))
    return jnp.matmul(context, o_weight)


def main():
    batch = 1
    seq = 16
    num_heads = 8
    head_dim = 8
    hidden = num_heads * head_dim

    context = jnp.ones((batch, num_heads, seq, head_dim), dtype=jnp.float32)
    o_weight = jnp.ones((hidden, hidden), dtype=jnp.float32)

    lowered = jax.jit(attention_output).lower(context, o_weight)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
