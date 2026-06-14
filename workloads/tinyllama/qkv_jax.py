import jax
import jax.numpy as jnp


def qkv_projection(hidden_states, q_weight, k_weight, v_weight, head_dim=64):
    batch, seq, _ = hidden_states.shape
    q_out = q_weight.shape[-1]
    k_out = k_weight.shape[-1]
    v_out = v_weight.shape[-1]

    if k_out != v_out:
        raise ValueError("k_weight and v_weight must have the same output dimension")
    if q_out % head_dim != 0 or k_out % head_dim != 0:
        raise ValueError("projection output dimensions must be divisible by head_dim")

    num_heads = q_out // head_dim
    num_kv_heads = k_out // head_dim

    q = jnp.matmul(hidden_states, q_weight)
    k = jnp.matmul(hidden_states, k_weight)
    v = jnp.matmul(hidden_states, v_weight)

    q = jnp.reshape(q, (batch, seq, num_heads, head_dim))
    k = jnp.reshape(k, (batch, seq, num_kv_heads, head_dim))
    v = jnp.reshape(v, (batch, seq, num_kv_heads, head_dim))
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    return q, k, v


def main():
    batch = 1
    seq = 16
    hidden = 2048
    num_heads = 32
    num_kv_heads = 4
    head_dim = 64

    x = jnp.ones((batch, seq, hidden), dtype=jnp.float32)
    q_weight = jnp.ones((hidden, num_heads * head_dim), dtype=jnp.float32)
    k_weight = jnp.ones((hidden, num_kv_heads * head_dim), dtype=jnp.float32)
    v_weight = jnp.ones((hidden, num_kv_heads * head_dim), dtype=jnp.float32)

    lowered = jax.jit(qkv_projection).lower(x, q_weight, k_weight, v_weight)
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
