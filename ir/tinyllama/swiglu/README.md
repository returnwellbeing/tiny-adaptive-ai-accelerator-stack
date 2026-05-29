# TinyLlama SwiGLU StableHLO

This directory contains the StableHLO MLIR generated from the SwiGLU
activation/gating primitive used inside TinyLlama's `LlamaMLP`.

## Regenerate

From the repository root:

```bash
python workloads/tinyllama/swiglu_jax.py > ir/tinyllama/swiglu/swiglu.stablehlo.mlir
```

Source workload:

```text
workloads/tinyllama/swiglu_jax.py
```

## Summarize

```bash
python tools/summarize_stablehlo.py ir/tinyllama/swiglu/swiglu.stablehlo.mlir
```

Expected workload tags:

```text
elementwise-heavy
```

## Operation

The workload implements:

```text
silu(gate) * up
```

For this reduced test case:

```text
gate:   [1, 16, 256]
up:     [1, 16, 256]
output: [1, 16, 256]
```

In StableHLO, `silu(x)` is lowered into elementwise operations:

```text
x * (1 / (1 + exp(-x)))
```

This workload has no `stablehlo.dot_general`; it isolates the MLP
elementwise gating path.
