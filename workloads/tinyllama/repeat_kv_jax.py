import jax
import jax.numpy as jnp


def repeat_kv(hidden_states, num_repeats=4):
    batch, seq, num_kv_heads, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, :, None, :]
    repeated = jnp.broadcast_to(
        expanded,
        (batch, seq, num_kv_heads, num_repeats, head_dim),
    )
    return jnp.reshape(
        repeated,
        (batch, seq, num_kv_heads * num_repeats, head_dim),
    )


def main():
    batch = 1
    seq = 16
    num_kv_heads = 2
    head_dim = 8
    num_repeats = 4

    hidden_states = jnp.ones(
        (batch, seq, num_kv_heads, head_dim),
        dtype=jnp.float32,
    )

    lowered = jax.jit(
        repeat_kv,
        static_argnames=("num_repeats",),
    ).lower(hidden_states, num_repeats=num_repeats)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
