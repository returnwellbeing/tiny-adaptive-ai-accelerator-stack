import jax
import jax.numpy as jnp


def residual_add(hidden_states, residual):
    return hidden_states + residual


def main():
    batch = 1
    seq = 16
    hidden = 64

    hidden_states = jnp.ones((batch, seq, hidden), dtype=jnp.float32)
    residual = jnp.ones((batch, seq, hidden), dtype=jnp.float32)

    lowered = jax.jit(residual_add).lower(hidden_states, residual)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
