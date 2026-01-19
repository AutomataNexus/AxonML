# axonml-autograd

Automatic differentiation engine for the [Axonml](https://github.com/AutomataNexus/AxonML) ML framework.

## Overview

Reverse-mode automatic differentiation (backpropagation) with:

- **Variable** - Tensor wrapper with gradient tracking
- **Computational Graph** - Dynamic graph construction
- **Gradient Functions** - Backward pass for all operations
- **Context Managers** - `no_grad` for inference

## Usage

```rust
use axonml_autograd::Variable;
use axonml_tensor::Tensor;

let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(), true);
let y = x.pow(2.0).sum();
y.backward();
let grad = x.grad(); // dy/dx = 2x = [4.0, 6.0]
```

## Part of Axonml

This crate is part of the [Axonml](https://crates.io/crates/axonml) ML framework.

## License

MIT OR Apache-2.0
