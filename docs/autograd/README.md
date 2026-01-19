# axonml-autograd Documentation

> Automatic differentiation engine for the Axonml ML framework.

## Overview

`axonml-autograd` provides reverse-mode automatic differentiation (backpropagation) through a dynamic computational graph. This enables gradient computation for any tensor operations, making it the foundation for neural network training.

## Core Concepts

### Computational Graph

When `requires_grad=true`, operations build a directed acyclic graph (DAG) that tracks the computation history. During the backward pass, gradients flow through this graph in reverse order.

```
Forward:  x → [Linear] → h → [ReLU] → y → [Loss] → L
Backward: ∂L/∂x ← [∂Linear] ← ∂L/∂h ← [∂ReLU] ← ∂L/∂y ← [1.0]
```

### Variables

`Variable` wraps a `Tensor` and adds gradient tracking:

```rust
pub struct Variable {
    data: Tensor<f32>,
    grad: Option<Tensor<f32>>,
    grad_fn: Option<Arc<dyn GradFn>>,
    requires_grad: bool,
}
```

## Modules

### variable.rs

The main `Variable` struct for gradient-tracked tensors.

**Creation:**
```rust
use axonml_autograd::Variable;

// Create with gradient tracking
let x = Variable::new(tensor, true);

// Create without gradient tracking
let y = Variable::new(tensor, false);

// From existing variable
let z = Variable::from_tensor(tensor);
```

**Key Methods:**
- `data()` - Get underlying tensor
- `grad()` - Get gradient tensor (if computed)
- `requires_grad()` - Check if tracking gradients
- `backward()` - Compute gradients
- `detach()` - Create non-tracking copy
- `zero_grad()` - Reset gradient to zero

### backward.rs

Backward pass implementation.

**Usage:**
```rust
let x = Variable::new(tensor, true);
let y = some_operation(&x);
let loss = compute_loss(&y);

// Compute all gradients
loss.backward();

// Access gradients
if let Some(grad) = x.grad() {
    println!("Gradient: {:?}", grad);
}
```

**Details:**
- Topological sort ensures correct order
- Gradients accumulate (not overwritten)
- Non-leaf gradients freed by default
- Supports multiple backward passes

### no_grad.rs

Context manager for disabling gradient computation.

```rust
use axonml_autograd::no_grad;

// Inside no_grad, operations don't track gradients
no_grad(|| {
    let y = model.forward(&x);
    // No graph built, faster inference
});
```

### functions/

Differentiable operations with forward and backward implementations.

**Arithmetic (`basic.rs`):**
- `Add` - a + b, gradients: (1, 1)
- `Sub` - a - b, gradients: (1, -1)
- `Mul` - a * b, gradients: (b, a)
- `Div` - a / b, gradients: (1/b, -a/b²)
- `Neg` - -a, gradient: -1
- `MatMul` - a @ b, gradients: (∂L/∂y @ b.T, a.T @ ∂L/∂y)

**Activations (`activation.rs`):**
- `Sigmoid` - σ(x), gradient: σ(x)(1 - σ(x))
- `Tanh` - tanh(x), gradient: 1 - tanh²(x)
- `ReLU` - max(0, x), gradient: x > 0 ? 1 : 0
- `LeakyReLU` - x > 0 ? x : αx
- `GELU` - Gaussian Error Linear Unit
- `Softmax` - exp(x) / Σexp(x)

**Reductions (`reduction.rs`):**
- `Sum` - Σx, gradient: broadcast ones
- `Mean` - Σx/n, gradient: broadcast 1/n
- `Max` - max(x), gradient: one-hot at max

**Shape (`shape.rs`):**
- `Reshape` - gradient: reshape back
- `Transpose` - gradient: transpose back
- `Squeeze/Unsqueeze` - gradient: inverse operation

## Usage Examples

### Basic Gradient Computation

```rust
use axonml::prelude::*;

// Create input with gradient tracking
let x = Variable::new(
    Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(),
    true
);

// Compute y = x^2
let y = x.mul_var(&x);

// Compute sum to get scalar
let loss = y.sum();

// Backward pass
loss.backward();

// Gradient: d(x^2)/dx = 2x
// For x = [2, 3], grad = [4, 6]
println!("Gradient: {:?}", x.grad());
```

### Training Loop

```rust
use axonml::prelude::*;

let model = Linear::new(10, 1);
let mut optimizer = SGD::new(model.parameters(), 0.01);

for batch in dataloader.iter() {
    // Forward pass (builds graph)
    let output = model.forward(&batch.data);
    let loss = mse_loss(&output, &batch.targets);

    // Backward pass (computes gradients)
    loss.backward();

    // Update weights
    optimizer.step();
    optimizer.zero_grad();
}
```

### Inference (No Gradients)

```rust
use axonml::prelude::*;

// Disable gradient tracking for inference
no_grad(|| {
    for batch in test_loader.iter() {
        let output = model.forward(&batch.data);
        let predictions = output.argmax(1);
        // Process predictions...
    }
});
```

## Implementation Details

### Memory Management

- Gradients are stored in leaf variables only (by default)
- Intermediate activations are freed after backward
- Use `retain_grad()` to keep non-leaf gradients

### Thread Safety

- Variables are `Send` but not `Sync`
- Computational graph is single-threaded
- Use `no_grad` for parallel inference

### Numerical Stability

- Log-sum-exp trick for softmax
- Gradient clipping available
- NaN/Inf detection in debug mode

## Feature Flags

- `std` (default) - Enable standard library

## Related Modules

- [Tensor](../tensor/README.md) - Underlying data structure
- [Neural Networks](../nn/README.md) - Modules using autograd
- [Optimizers](../optim/README.md) - Gradient-based optimization

@version 0.1.0
@author AutomataNexus Development Team
