# axonml-optim

Optimizers and learning rate schedulers for [Axonml](https://github.com/AutomataNexus/AxonML).

## Optimizers

- **SGD** - With momentum, Nesterov, weight decay
- **Adam** - Adaptive learning rates
- **AdamW** - Decoupled weight decay
- **RMSprop** - Moving average of squared gradients

## LR Schedulers

- StepLR, MultiStepLR, ExponentialLR
- CosineAnnealingLR, OneCycleLR, WarmupLR

## Usage

```rust
use axonml_optim::{Adam, Optimizer};

let mut optimizer = Adam::new(model.parameters(), 0.001);
optimizer.zero_grad();
loss.backward();
optimizer.step();
```

## License

MIT OR Apache-2.0
