# axonml-autograd Documentation

> Automatic differentiation engine for the AxonML ML framework.

## Overview

`axonml-autograd` implements reverse-mode automatic differentiation (backprop)
over a dynamic computation graph. The `Variable` type wraps a `Tensor<f32>`
with a `GradAccumulator` and an optional `GradFn` recording the op that
produced it. Calling `.backward()` walks the graph and accumulates gradients
into the leaves.

## Core Concepts

### Computational Graph

Every op on a `Variable` with gradient tracking enabled appends a node to a
directed acyclic graph. `backward()` topologically sorts the reachable graph
and calls each node's `GradientFunction::backward()` in reverse order.

```
Forward:  x -> [Linear] -> h -> [ReLU] -> y -> [Loss] -> L
Backward: dL/dx <- [dLinear] <- dL/dh <- [dReLU] <- dL/dy <- 1.0
```

### Variables

```rust
pub struct Variable {
    data: Arc<RwLock<Tensor<f32>>>,
    // grad accumulator + optional GradFn recording producing op
}
```

## Modules

### `variable`

`Variable` — tensor with automatic gradient tracking (1577 lines, 67 public
methods). Differentiable versions of all tensor ops:

- Arithmetic: `add_var`, `sub_var`, `mul_var`, `div_var`, `neg`,
  `add_scalar`, `mul_scalar`, `pow`
- Activations: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `elu`,
  `leaky_relu`, `softmax`, `log_softmax`
- Reductions: `sum`, `mean`, `sum_dim`, `mean_dim`, `var_dim`
- Shape: `reshape`, `transpose`, `t`, `narrow`, `select`, `unsqueeze`,
  `expand`, `cat`
- Linear algebra: `matmul`
- Misc: `exp`, `log`, `sqrt`, `clamp`

Control: `backward()`, `detach()`, `requires_grad_()`, `data()` (read guard),
`zero_grad()`, `from_operation(op, inputs)` for custom `GradFn` attachment.

### `backward`

Topological-sort driven backward pass. The top-level `backward(var)` function
entry point. Gradients accumulate into leaf variables; non-leaf gradients are
freed unless retained explicitly.

### `grad_fn`

`GradFn` / `GradientFunction` traits and the `GradAccumulator` used by leaf
`Variable`s to sum contributions. Also `AccumulateGrad` (leaf terminator node).

### `graph`

`ComputationGraph` and `GraphNode` — the global dynamic graph scaffolding.
`with_graph(|g| ...)` provides scoped access.

### `no_grad`

Gradient-disable context.

```rust
use axonml_autograd::no_grad;

no_grad(|| {
    let y = model.forward(&x);
    // no graph built
});
```

`NoGradGuard` RAII guard, plus `is_grad_enabled()` and `enable_grad()`
(inverse scope). For a stricter guarantee — variables created inside the
scope can never be upgraded to require gradients — use `inference_mode()`
or `InferenceModeGuard::new()` / `is_inference_mode()`.

### `amp`

Automatic Mixed Precision (F16 autocast) — 321 lines.

Thread-local autocast state (enabled flag + target `DType` + nesting depth).

```rust
use axonml_autograd::amp::{autocast, AutocastPolicy};
use axonml_core::DType;

autocast(DType::F16, || {
    let h = linear.forward(&x);  // matmul/conv run in f16
    let y = loss.forward(&h);
});
```

API: `AutocastGuard`, `AutocastPolicy`, `autocast`, `disable_autocast`,
`is_autocast_enabled`, `autocast_dtype`. Pairs with `GradScaler` in
`axonml-optim` for loss scaling.

### `checkpoint`

Gradient checkpointing (428 lines) — trade compute for memory.

```rust
use axonml_autograd::{checkpoint, checkpoint_sequential};

// Single block: no activations kept; recomputed on backward
let y = checkpoint(|x| block.forward(x), x);

// Sequence with N segments:
let y = checkpoint_sequential(&[block1, block2, block3], 2, x);
```

Cuts peak memory from O(layers) to O(sqrt(layers)) at the cost of one extra
forward pass. `checkpoint_rng_seed()` returns the deterministic RNG seed
during recompute so dropout / other stochastic ops replay identically.

### `inspect`

Native graph inspection and DOT export — no external `torchviz`-style tool
required.

```rust
use axonml_autograd::inspect::{trace_backward, to_dot, depth, node_count};

let snapshot = trace_backward(&loss);
snapshot.node_count();
snapshot.depth();
snapshot.leaf_count();
snapshot.operation_names();
let summary = snapshot.gradient_flow_summary();

let dot = to_dot(&snapshot);
std::fs::write("graph.dot", &dot).unwrap();
```

Exports `GraphSnapshot`, `SnapshotNode`, `trace_backward`, `to_dot`, `depth`,
`node_count`.

### `functions/`

Seven submodules of `*Backward` structs implementing `GradientFunction`.

| Submodule    | Covers                                                                 |
|--------------|------------------------------------------------------------------------|
| `basic`      | add, sub, mul, div, neg, scalar ops, reshape, transpose, narrow, select, unsqueeze, expand, cat, clamp, exp, log, sqrt, pow, sum, mean, sum_dim, mean_dim, var_dim |
| `activation` | relu, sigmoid, tanh, gelu, silu, elu, leaky_relu, softmax, log_softmax |
| `linalg`     | matmul                                                                 |
| `loss`       | mse, cross_entropy, bce, bce_with_logits, l1, smooth_l1, nll           |
| `conv`       | conv1d, conv2d (BLAS-accelerated backward)                             |
| `rnn`        | lstm, gru, rnn cell backward (`LstmGatesBackward`, `GruGatesBackward`) |
| `attention`  | multi-head attention (`FusedAttentionBackward`)                        |

## Usage Examples

### Basic gradient

```rust
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

let x = Variable::new(
    Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(),
    true,
);

let y = x.mul_var(&x);   // y = x^2
let loss = y.sum();
loss.backward();

// d(x^2)/dx = 2x  =>  [4, 6]
println!("{:?}", x.grad());
```

### Training loop

```rust
use axonml::prelude::*;

let model = Linear::new(10, 1);
let mut optimizer = SGD::new(model.parameters(), 0.01);

for batch in dataloader.iter() {
    let output = model.forward(&batch.data);
    let loss = mse_loss(&output, &batch.targets);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

### Inference

```rust
use axonml_autograd::no_grad;

no_grad(|| {
    for batch in test_loader.iter() {
        let output = model.forward(&batch.data);
        let predictions = output.argmax(1);
    }
});
```

## Implementation Notes

- Gradients accumulate into leaves; call `optimizer.zero_grad()` each step.
- Intermediate activations are freed after backward.
- `Variable` is `Send`; the graph itself is single-threaded per scope.
- Softmax/log-softmax use the log-sum-exp trick for numerical stability.

## Feature Flags

- `std` (default) — standard library

## Related Modules

- [Tensor](../tensor/README.md) — underlying data structure
- [Neural Networks](../nn/README.md) — modules built on Variable
- [Optimizers](../optim/README.md) — gradient-based training

## Last updated

0.6.1 (2026-04-16)
