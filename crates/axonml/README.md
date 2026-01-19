# Axonml

A complete, PyTorch-equivalent machine learning framework in pure Rust.

## Features

- **Tensors** - N-dimensional arrays with broadcasting
- **Autograd** - Automatic differentiation
- **Neural Networks** - Linear, Conv, RNN, LSTM, Attention
- **Optimizers** - SGD, Adam, AdamW, RMSprop
- **Data Loading** - Dataset, DataLoader, transforms
- **Vision** - ResNet, VGG, ViT architectures
- **LLM** - BERT, GPT-2 architectures
- **Serialization** - Save/load models, ONNX export
- **Quantization** - INT8/INT4 compression

## Quick Start

```rust
use axonml::prelude::*;

let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 10));

let mut optimizer = Adam::new(model.parameters(), 0.001);

for batch in dataloader.iter() {
    let output = model.forward(&batch.data);
    let loss = output.cross_entropy(&batch.targets);
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```

## Documentation

- [GitHub](https://github.com/AutomataNexus/AxonML)
- [API Docs](https://docs.rs/axonml)

## License

MIT OR Apache-2.0
