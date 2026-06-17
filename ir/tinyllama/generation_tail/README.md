# TinyLlama Generation Tail StableHLO

This directory contains the StableHLO MLIR generated from the decode
tail that is intentionally separate from the decoder-layer workload.

It models:

```text
decoder output [B, 1, hidden]
-> final RMSNorm
-> lm_head [hidden, vocab]
-> logits [B, 1, vocab]
-> argmax
-> next token id [B, 1]
-> embedding lookup
-> next decode hidden state [B, 1, hidden]
```

This workload uses deterministic `argmax`. Sampling policies such as
temperature, top-k, top-p, and random sampling are not included here and
should be traced separately.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/generation_tail_jax.py > ir/tinyllama/generation_tail/generation_tail.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/generation_tail_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/generation_tail/generation_tail.stablehlo.mlir
```

## IREE

CPU compile:

```bash
iree-compile \
  ir/tinyllama/generation_tail/generation_tail.stablehlo.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o /tmp/generation_tail_cpu.vmfb
```

CPU run:

```bash
iree-run-module \
  --module=/tmp/generation_tail_cpu.vmfb \
  --function=main \
  --input=1x1x64xf32=1 \
  --input=64xf32=1 \
  --input=64x128xf32=1 \
  --input=128x64xf32=1
```

For Intel macOS machines, run IREE compile/runtime commands in the
Linux devcontainer or another Linux environment.

## Tensor Shapes

Toy static shapes:

```text
B = 1
S = 1
hidden = 64
vocab = 128
```

Inputs:

```text
decoder_output:    [1, 1, 64]
final_norm_weight: [64]
lm_head_weight:    [64, 128]
embedding_table:   [128, 64]
```

Outputs:

```text
logits:                   [1, 1, 128]
next_token_id:            [1, 1]
next_decode_hidden_state: [1, 1, 64]
```

## StableHLO Observations

Expected operation patterns include:

```text
stablehlo.reduce
stablehlo.rsqrt
stablehlo.multiply
stablehlo.dot_general
stablehlo.compare
stablehlo.select
stablehlo.gather
```

The final norm is reduction-heavy and elementwise-heavy. The LM head is
an activation x model-weight projection. Argmax introduces a reduction
over the vocabulary dimension, and embedding lookup is an indexed gather
from the token embedding table.
