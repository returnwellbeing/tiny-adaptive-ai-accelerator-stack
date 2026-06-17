import jax
import jax.numpy as jnp


def rms_norm(hidden_states, weight, eps=1e-6):
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + eps)
    return hidden_states * weight


def generation_tail(
    decoder_output,
    final_norm_weight,
    lm_head_weight,
    embedding_table,
):
    normalized = rms_norm(decoder_output, final_norm_weight)
    logits = jnp.matmul(normalized, lm_head_weight)
    next_token_id = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    next_decode_hidden_state = jnp.take(embedding_table, next_token_id, axis=0)
    return logits, next_token_id, next_decode_hidden_state


def main():
    batch = 1
    seq = 1
    hidden = 64
    vocab = 128

    decoder_output = jnp.ones((batch, seq, hidden), dtype=jnp.float32)
    final_norm_weight = jnp.ones((hidden,), dtype=jnp.float32)
    lm_head_weight = jnp.ones((hidden, vocab), dtype=jnp.float32)
    embedding_table = jnp.ones((vocab, hidden), dtype=jnp.float32)

    lowered = jax.jit(generation_tail).lower(
        decoder_output,
        final_norm_weight,
        lm_head_weight,
        embedding_table,
    )
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
