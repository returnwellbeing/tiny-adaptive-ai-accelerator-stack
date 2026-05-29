import jax
import jax.numpy as jnp


def linear(hidden_states, weight):
    return jnp.matmul(hidden_states, weight)


def main():
    batch = 1
    seq = 16
    hidden = 64
    out = 256

    x = jnp.ones((batch, seq, hidden), dtype=jnp.float32)
    w = jnp.ones((hidden, out), dtype=jnp.float32)

    lowered = jax.jit(linear).lower(x, w)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
