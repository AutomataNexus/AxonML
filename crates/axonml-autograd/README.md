# axonml-autograd

<p align="center">
  <!-- Logo placeholder -->
  <img src="../../assets/logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/part_of-AxonML-purple.svg" alt="Part of AxonML">
</p>

## Overview

**axonml-autograd** provides reverse-mode automatic differentiation (backpropagation) for computing gradients of tensor operations. This is the foundation for training neural networks using gradient descent optimization in the AxonML framework.

## Features

- **Dynamic Computational Graph** - Build computational graphs dynamically during the forward pass, enabling flexible model architectures.

- **Reverse-Mode Autodiff** - Efficient backpropagation through the graph to compute gradients for all learnable parameters.

- **Gradient Accumulation** - Automatic gradient accumulation for parameters used multiple times in the computation.

- **No-Grad Context** - Temporarily disable gradient tracking for inference or evaluation with `no_grad()` and `NoGradGuard`.

- **Inference Mode** - Optimized inference mode for production deployment with `inference_mode()`.

- **Gradient Checking** - Built-in numerical gradient verification using finite differences for debugging.

## Modules

| Module | Description |
|--------|-------------|
| `variable` | `Variable` struct wrapping tensors with gradient tracking and differentiable operations |
| `graph` | Computational graph management with topological ordering for backpropagation |
| `backward` | Backward pass implementation traversing the graph in reverse order |
| `grad_fn` | `GradientFunction` trait and implementations for all differentiable operations |
| `no_grad` | Context managers for disabling gradient computation (`NoGradGuard`, `InferenceModeGuard`) |
| `functions` | Gradient implementations for arithmetic, activation, loss, and linear algebra operations |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-autograd = "0.1.0"
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

### No-Grad Context

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
