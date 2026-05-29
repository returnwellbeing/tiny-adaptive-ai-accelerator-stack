import jax
import jax.numpy as jnp


def mlp(hidden_states, gate_weight, up_weight, down_weight):
    gate = jnp.matmul(hidden_states, gate_weight)
    up = jnp.matmul(hidden_states, up_weight)
    hidden_states = jax.nn.silu(gate) * up
    return jnp.matmul(hidden_states, down_weight)


def main():
    batch = 1
    seq = 16
    hidden = 64
    intermediate = 256

    x = jnp.ones((batch, seq, hidden), dtype=jnp.float32)
    gate_w = jnp.ones((hidden, intermediate), dtype=jnp.float32)
    up_w = jnp.ones((hidden, intermediate), dtype=jnp.float32)
    down_w = jnp.ones((intermediate, hidden), dtype=jnp.float32)

    lowered = jax.jit(mlp).lower(x, gate_w, up_w, down_w)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
