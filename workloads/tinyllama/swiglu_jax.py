import jax
import jax.numpy as jnp


def swiglu(gate, up):
    return jax.nn.silu(gate) * up


def main():
    batch = 1
    seq = 16
    intermediate = 256

    gate = jnp.ones((batch, seq, intermediate), dtype=jnp.float32)
    up = jnp.ones((batch, seq, intermediate), dtype=jnp.float32)

    lowered = jax.jit(swiglu).lower(gate, up)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
