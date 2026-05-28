import jax
import jax.numpy as jnp


def rms_norm(hidden_states, weight, eps=1e-6):
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + eps)
    return hidden_states * weight


def main():
    batch = 1
    seq = 16
    hidden = 64

    x = jnp.ones((batch, seq, hidden), dtype=jnp.float32)
    w = jnp.ones((hidden,), dtype=jnp.float32)

    lowered = jax.jit(rms_norm).lower(x, w)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
