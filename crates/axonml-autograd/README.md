# axonml-autograd

[![Crates.io](https://img.shields.io/crates/v/axonml-autograd.svg)](https://crates.io/crates/axonml-autograd)
[![Docs.rs](https://docs.rs/axonml-autograd/badge.svg)](https://docs.rs/axonml-autograd)
[![Downloads](https://img.shields.io/crates/d/axonml-autograd.svg)](https://crates.io/crates/axonml-autograd)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Automatic differentiation engine for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-autograd` provides reverse-mode automatic differentiation (backpropagation) for the Axonml framework. It enables gradient computation through arbitrary computational graphs, which is essential for training neural networks. The implementation follows PyTorch's design philosophy with dynamic graph construction.

## Features

### Core Capabilities
- **Dynamic computational graphs** - Build graphs on-the-fly during forward pass
- **Reverse-mode autodiff** - Efficient gradient computation via backpropagation
- **Gradient accumulation** - Support for gradient accumulation across batches
- **Higher-order gradients** - Compute gradients of gradients (experimental)

### Variable System
- **Gradient tracking** - Automatic tracking of operations requiring gradients
- **Leaf detection** - Distinguish leaf variables from intermediate results
- **Gradient retention** - Option to retain gradients for non-leaf variables
- **In-place operation detection** - Safety checks for in-place modifications

### Context Managers
- **no_grad** - Disable gradient computation for inference
- **enable_grad** - Re-enable gradients within no_grad context
- **set_grad_enabled** - Programmatic gradient control

## Installation

```toml
[dependencies]
axonml-autograd = "0.1"
```

## Usage

### Basic Gradient Computation

```rust
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// Create a variable with gradient tracking
let x = Variable::new(
    Tensor::<f32>::from_vec(vec![2.0, 3.0], &[2]).unwrap(),
    true  // requires_grad = true
);

// Compute y = x^2
let y = x.pow(2.0);

// Compute z = sum(y) = x[0]^2 + x[1]^2
let z = y.sum();

// Backward pass computes gradients
z.backward();

// Access gradients: dz/dx = 2x = [4.0, 6.0]
let grad = x.grad().unwrap();
println!("Gradient: {:?}", grad);
```

### Multi-Variable Gradients

```rust
use axonml_autograd::Variable;
use axonml_tensor::{Tensor, randn};

let a = Variable::new(randn::<f32>(&[3, 4]), true);
let b = Variable::new(randn::<f32>(&[4, 5]), true);

// z = sum(a @ b)
let c = a.matmul(&b).unwrap();
let z = c.sum();

// Compute gradients for both a and b
z.backward();

let grad_a = a.grad().unwrap();  // Shape: [3, 4]
let grad_b = b.grad().unwrap();  // Shape: [4, 5]
```

### Disabling Gradients for Inference

```rust
use axonml_autograd::{Variable, no_grad};
use axonml_tensor::randn;

let x = Variable::new(randn::<f32>(&[10, 10]), true);

// Inside no_grad, operations don't track gradients
no_grad(|| {
    let y = x.relu();
    let z = y.sum();
    // z.backward() would fail here - no graph was built
    println!("Inference result: {}", z.data().sum());
});

// Outside no_grad, gradients are tracked again
let y = x.relu();
let z = y.sum();
z.backward();  // This works
```

### Gradient Accumulation

```rust
use axonml_autograd::Variable;
use axonml_tensor::randn;

let weights = Variable::new(randn::<f32>(&[100, 10]), true);

// Accumulate gradients over multiple mini-batches
for batch in mini_batches {
    let output = weights.matmul(&batch).unwrap();
    let loss = output.sum();
    loss.backward();  // Gradients accumulate
}

// Gradients now contain sum of all batch gradients
let accumulated_grad = weights.grad().unwrap();

// Zero gradients before next accumulation cycle
weights.zero_grad();
```

### Complex Computation Graphs

```rust
use axonml_autograd::Variable;
use axonml_tensor::randn;

let x = Variable::new(randn::<f32>(&[5, 5]), true);

// Complex expression with multiple paths
let a = x.relu();
let b = x.sigmoid();
let c = &a + &b;           // Both paths contribute
let d = c.pow(2.0);
let e = d.mean().unwrap();

e.backward();

// Gradient flows through both relu and sigmoid paths
let grad = x.grad().unwrap();
```

### Detaching from Graph

```rust
use axonml_autograd::Variable;
use axonml_tensor::randn;

let x = Variable::new(randn::<f32>(&[5, 5]), true);
let y = x.pow(2.0);

// Detach creates a new variable with no gradient history
let y_detached = y.detach();

// Operations on detached variable don't track gradients
let z = y_detached.sum();
// z.backward() would fail - y_detached has no graph
```

### Checking Gradient Requirements

```rust
use axonml_autograd::Variable;
use axonml_tensor::randn;

let x = Variable::new(randn::<f32>(&[5]), true);
let y = Variable::new(randn::<f32>(&[5]), false);

println!("x requires grad: {}", x.requires_grad());  // true
println!("y requires grad: {}", y.requires_grad());  // false

// Operations between grad and non-grad variables
let z = &x + &y;
println!("z requires grad: {}", z.requires_grad());  // true (inherits from x)
```

## Supported Operations

### Arithmetic
| Operation | Forward | Backward |
|-----------|---------|----------|
| Add | `a + b` | `da = 1, db = 1` |
| Sub | `a - b` | `da = 1, db = -1` |
| Mul | `a * b` | `da = b, db = a` |
| Div | `a / b` | `da = 1/b, db = -a/b²` |
| Pow | `a.pow(n)` | `da = n * a^(n-1)` |
| Neg | `-a` | `da = -1` |

### Matrix Operations
| Operation | Forward | Backward |
|-----------|---------|----------|
| MatMul | `a.matmul(b)` | `da = grad @ b.T, db = a.T @ grad` |
| Transpose | `a.t()` | `da = grad.t()` |

### Reductions
| Operation | Forward | Backward |
|-----------|---------|----------|
| Sum | `a.sum()` | `da = ones_like(a)` |
| Mean | `a.mean()` | `da = ones_like(a) / numel` |
| Sum Dim | `a.sum_dim(d)` | `da = broadcast(grad)` |

### Activations
| Operation | Forward | Backward |
|-----------|---------|----------|
| ReLU | `max(0, x)` | `da = (x > 0)` |
| Sigmoid | `1/(1+e^-x)` | `da = σ(x)(1-σ(x))` |
| Tanh | `tanh(x)` | `da = 1 - tanh²(x)` |
| GELU | `x*Φ(x)` | `da = Φ(x) + x*φ(x)` |
| Softmax | `e^x / Σe^x` | Jacobian-vector product |

### Shape Operations
| Operation | Forward | Backward |
|-----------|---------|----------|
| Reshape | `a.reshape(s)` | `da = grad.reshape(a.shape)` |
| Squeeze | `a.squeeze(d)` | `da = grad.unsqueeze(d)` |
| Unsqueeze | `a.unsqueeze(d)` | `da = grad.squeeze(d)` |

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Variable<T>` | Tensor wrapper with gradient tracking |
| `GradFn` | Gradient function for backward pass |
| `Graph` | Computational graph structure |

### Variable Methods

| Method | Description |
|--------|-------------|
| `new(tensor, requires_grad)` | Create new variable |
| `data()` | Access underlying tensor |
| `grad()` | Get computed gradient |
| `backward()` | Compute gradients |
| `zero_grad()` | Reset gradients to zero |
| `detach()` | Create detached copy |
| `requires_grad()` | Check if tracking gradients |
| `is_leaf()` | Check if leaf variable |

### Context Functions

| Function | Description |
|----------|-------------|
| `no_grad(f)` | Execute closure without gradient tracking |
| `enable_grad(f)` | Re-enable gradients in no_grad context |
| `set_grad_enabled(bool)` | Set gradient computation state |
| `is_grad_enabled()` | Check if gradients are enabled |

## Architecture

```
axonml-autograd
├── variable.rs    # Variable struct and operations
├── graph.rs       # Computational graph management
├── backward.rs    # Backward pass implementation
├── grad_fn.rs     # Gradient function traits
├── no_grad.rs     # Context managers
└── functions/
    ├── basic.rs       # Add, Mul, etc.
    ├── activation.rs  # ReLU, Sigmoid, etc.
    ├── loss.rs        # Loss function gradients
    └── linalg.rs      # MatMul, etc.
```

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

```toml
[dependencies]
axonml = "0.1"  # Includes axonml-autograd
```

## License

MIT OR Apache-2.0
