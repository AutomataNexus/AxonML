# axonml-autograd

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/rust-1.85%2B-orange.svg" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/version-0.6.1-green.svg" alt="Version 0.6.1">
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

**axonml-autograd** provides reverse-mode automatic differentiation for the AxonML framework. `Variable` wraps `Tensor<f32>` with gradient tracking; `backward()` walks the dynamic computation graph and accumulates gradients via the `GradFn` / `GradientFunction` trait. The crate also ships gradient checkpointing, F16 autocast (AMP), a thread-local no-grad scope, and a built-in graph inspection / DOT export API.

## Features

- **Dynamic Computational Graph** - Built as the forward pass runs; no static graph compilation.

- **Reverse-Mode Autodiff** - `backward()` traverses the graph in reverse topological order and accumulates gradients on leaf variables.

- **Gradient Accumulation** - Parameters used multiple times automatically sum incoming gradients.

- **No-Grad Scope** - Thread-local `GRAD_ENABLED` flag controlled by `no_grad(closure)` and the `NoGradGuard` RAII scope. Re-entrant via `enable_grad` inside a no-grad scope (useful for second-order gradients).

- **Inference Mode** - `inference_mode(closure)` / `InferenceModeGuard` — stricter than `no_grad`: variables created inside the scope cannot later be upgraded to require gradients, catching accidental misuse at the boundary rather than silently falling through.

- **Gradient Checkpointing** - `checkpoint(func, input)` and `checkpoint_sequential(funcs, segments, input)` trade one extra forward pass for O(sqrt(L)) peak memory. `checkpoint_rng_seed` handles deterministic dropout replay across the recompute pass.

- **Automatic Mixed Precision (AMP)** - `autocast(dtype, closure)` and `AutocastGuard` enable F16 downcasting for matmul/conv ops with a nesting-depth counter. Pairs with `GradScaler` in `axonml-optim` for loss scaling.

- **Gradient Checking** - `backward::numerical_gradient` + `backward::gradcheck` verify analytical gradients against finite differences for debugging custom ops.

- **BLAS-Accelerated Conv Backward** - Convolution backward pass uses im2col + GEMM for weight and input gradients instead of naive nested loops.

- **Native Graph Inspection** - `trace_backward` captures a `GraphSnapshot`, `to_dot` exports Graphviz DOT, and `node_count` / `depth` / gradient-flow summary give structural analysis without torchviz / tensorboard.

## Modules

| Module | Description |
|--------|-------------|
| `variable` | `Variable` wrapping `Tensor<f32>` with `requires_grad`, grad accumulator, and differentiable ops |
| `grad_fn` | `GradFn`, `GradFnId`, and the `GradientFunction` trait that defines the backward op interface |
| `graph` | `ComputationGraph`, `GraphNode`, and topological ordering for the backward pass |
| `backward` | Top-level `backward(root)` entry point, plus `numerical_gradient` / `gradcheck` utilities |
| `no_grad` | `NoGradGuard`, `no_grad(closure)`, `is_grad_enabled`, `enable_grad` for second-order grads; `InferenceModeGuard` / `inference_mode(closure)` / `is_inference_mode` for the stronger guarantee that variables created inside cannot later require gradients |
| `checkpoint` | `checkpoint`, `checkpoint_sequential`, `checkpoint_rng_seed`, memory/segment heuristics |
| `amp` | `AutocastGuard`, `autocast`, `autocast_dtype`, `disable_autocast`, `AutocastPolicy` |
| `inspect` | `trace_backward`, `to_dot`, `GraphSnapshot`, `SnapshotNode`, `node_count`, `depth` |
| `functions` | Gradient ops grouped into `basic`, `activation`, `linalg`, `loss`, `conv`, `rnn`, `attention` |

## Cargo Features

| Feature | Purpose |
|---------|---------|
| `cuda` | Forwards CUDA support to `axonml-tensor` and `axonml-core` for GPU gradient ops |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-autograd = "0.6.1"
```

### Basic Example

```rust
use axonml_autograd::{Variable, no_grad};
use axonml_tensor::Tensor;

// Create variables with gradient tracking
let x = Variable::new(
    Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(),
    true  // requires_grad = true
);

// Forward pass builds computational graph
let y = x.pow(2.0);  // y = x^2
let loss = y.sum();  // scalar loss

// Backward pass computes gradients
loss.backward();

// Access gradients: dy/dx = 2x = [4.0, 6.0]
let grad = x.grad().unwrap();
println!("Gradient: {:?}", grad.to_vec());
```

### Chained Operations

```rust
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

let a = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
let b = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);

// Build complex computation
let c = &a * &b;      // c = a * b
let d = c.pow(2.0);   // d = c^2 = (a*b)^2
let loss = d.sum();

loss.backward();

// dc/da = b = 3.0, dd/dc = 2c = 12.0, dL/da = 36.0
println!("dL/da = {:?}", a.grad().unwrap().to_vec());
// dc/db = a = 2.0, dd/dc = 2c = 12.0, dL/db = 24.0
println!("dL/db = {:?}", b.grad().unwrap().to_vec());
```

### No-Grad Scope

```rust
use axonml_autograd::{Variable, no_grad, NoGradGuard};
use axonml_tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(), true);

// Using closure
let output = no_grad(|| {
    // No gradient tracking here
    x.relu()
});

// Using guard
{
    let _guard = NoGradGuard::new();
    // No gradient tracking in this scope
    let y = x.sigmoid();
}
// Gradient tracking restored here
```

### Loss Functions

```rust
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

let predictions = Variable::new(
    Tensor::from_vec(vec![0.5, 1.5, 2.5], &[3]).unwrap(),
    true
);
let targets = Variable::new(
    Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
    false
);

// MSE Loss
let loss = predictions.mse_loss(&targets);
loss.backward();

// Binary Cross Entropy
let probs = Variable::new(Tensor::from_vec(vec![0.7, 0.3], &[2]).unwrap(), true);
let labels = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap(), false);
let bce_loss = probs.binary_cross_entropy(&labels);
```

### Matrix Operations with Gradients

```rust
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// Linear layer: y = xW + b
let x = Variable::new(Tensor::from_vec(vec![1.0; 6], &[2, 3]).unwrap(), false);
let w = Variable::new(Tensor::from_vec(vec![0.1; 12], &[3, 4]).unwrap(), true);
let b = Variable::new(Tensor::from_vec(vec![0.0; 4], &[4]).unwrap(), true);

let y = x.matmul(&w).add_var(&b);
let loss = y.sum();
loss.backward();

// Gradients available for w and b
println!("dL/dW shape: {:?}", w.grad().unwrap().shape());
println!("dL/db shape: {:?}", b.grad().unwrap().shape());
```

### Gradient Checking

```rust
use axonml_autograd::{Variable, backward::{numerical_gradient, gradcheck}};
use axonml_tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(), true);

// Compute numerical gradient
let numerical = numerical_gradient(
    |v| v.pow(2.0).sum(),
    &x,
    1e-5  // epsilon
);

// Compare with analytical gradient
let y = x.pow(2.0).sum();
y.backward();
let analytical = x.grad().unwrap();

// Verify gradients match
assert!(gradcheck(&analytical, &numerical, 1e-3, 1e-3));
```

### Gradient Checkpointing

```rust
use axonml_autograd::{Variable, checkpoint};
use axonml_tensor::Tensor;

let input = Variable::new(Tensor::from_vec(vec![1.0; 128], &[128]).unwrap(), true);

// Drop intermediate activations; recompute them in backward.
let output = checkpoint(|x| x.relu().pow(2.0), &input);
output.sum().backward();
```

### Mixed Precision (AMP)

```rust
use axonml_autograd::{Variable, autocast};
use axonml_core::DType;
use axonml_tensor::Tensor;

let w = Variable::new(Tensor::from_vec(vec![0.1; 16], &[4, 4]).unwrap(), true);
let x = Variable::new(Tensor::from_vec(vec![1.0; 4], &[1, 4]).unwrap(), false);

let y = autocast(DType::F16, || x.matmul(&w).relu());
y.sum().backward();
```

### Graph Inspection

Trace and visualize the computation graph — built-in, no external tools required.

```rust
use axonml_autograd::{Variable, trace_backward, to_dot, node_count, depth};
use axonml_tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(), true);
let y = x.pow(2.0).relu().sum();
y.backward();

// Capture the computation graph
let snapshot = trace_backward(&y);
println!("Nodes: {}, Depth: {}", node_count(&snapshot), depth(&snapshot));

// Export to Graphviz DOT format
let dot = to_dot(&snapshot);
std::fs::write("graph.dot", &dot).unwrap();
// Then: dot -Tpng graph.dot -o graph.png
```

## Tests

Run the test suite:

```bash
cargo test -p axonml-autograd
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

_Last updated: 2026-04-16 (v0.6.1)_
