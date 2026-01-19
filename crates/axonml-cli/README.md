# axonml-cli

[![Crates.io](https://img.shields.io/crates/v/axonml-cli.svg)](https://crates.io/crates/axonml-cli)
[![Downloads](https://img.shields.io/crates/d/axonml-cli.svg)](https://crates.io/crates/axonml-cli)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Command-line interface for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-cli` provides a powerful command-line interface for working with Axonml models. Train, evaluate, convert, quantize, and benchmark models without writing code.

## Installation

```bash
# From crates.io
cargo install axonml-cli

# From source
git clone https://github.com/AutomataNexus/AxonML
cd AxonML
cargo install --path crates/axonml-cli
```

## Commands

### Model Operations

```bash
# Show model information
axonml info model.axonml

# Convert between formats
axonml convert model.onnx -o model.axonml
axonml convert model.axonml --format onnx -o model.onnx

# Run inference
axonml run model.axonml --input data.tensor --output result.tensor
```

### Quantization

```bash
# Quantize to INT8
axonml quant convert model.axonml --type q8_0 -o model_q8.axonml

# Quantize to INT4
axonml quant convert model.axonml --type q4_0 -o model_q4.axonml

# Show quantization info
axonml quant info model_q8.axonml

# Benchmark quantized model
axonml quant benchmark model_q8.axonml

# List supported formats
axonml quant list
```

### Training

```bash
# Train a model
axonml train config.toml

# Resume training from checkpoint
axonml train config.toml --resume checkpoint.pt

# Train with specific device
axonml train config.toml --device cuda:0
```

### Evaluation

```bash
# Evaluate model accuracy
axonml eval model.axonml --dataset test_data/

# Evaluate with metrics
axonml eval model.axonml --dataset test_data/ --metrics accuracy,f1,precision
```

### Benchmarking

```bash
# Benchmark inference speed
axonml benchmark model.axonml --batch-size 32 --warmup 10 --iterations 100

# Compare multiple models
axonml benchmark model.axonml model_q8.axonml model_q4.axonml

# Benchmark on specific device
axonml benchmark model.axonml --device cuda:0
```

### JIT Compilation

```bash
# Compile model with JIT
axonml jit compile model.axonml -o model.jit

# Export to WebAssembly
axonml jit compile model.axonml --target wasm -o model.wasm
```

### Profiling

```bash
# Profile model performance
axonml profile run model.axonml --input sample.tensor

# Profile with memory tracking
axonml profile run model.axonml --memory --input sample.tensor

# Generate HTML report
axonml profile run model.axonml --report profile.html
```

### LLM Commands

```bash
# Generate text
axonml llm generate --model gpt2 --prompt "Hello world" --max-tokens 100

# Interactive chat
axonml llm chat --model llama-2-7b-chat

# Convert to GGUF
axonml llm convert model.safetensors --format gguf -o model.gguf
```

### Dataset Operations

```bash
# Download standard dataset
axonml data download mnist --output ./data/

# Show dataset info
axonml data info ./data/mnist/

# Convert dataset format
axonml data convert images/ --format tensor --output dataset.tensor
```

## Configuration Files

### Training Config (config.toml)

```toml
[model]
type = "resnet18"
num_classes = 10

[data]
train_path = "./data/train"
val_path = "./data/val"
batch_size = 32
num_workers = 4

[training]
epochs = 100
learning_rate = 0.001
optimizer = "adam"
weight_decay = 0.0001

[training.scheduler]
type = "cosine"
warmup_epochs = 5

[checkpoint]
save_dir = "./checkpoints"
save_best = true
save_every = 10
```

## Output Formats

### Model Info Output

```
Model: model.axonml
Format: Axonml Native v1
Size: 44.7 MB
Parameters: 11.2M
Layers: 62

Input:
  - name: input
    shape: [batch, 3, 224, 224]
    dtype: f32

Output:
  - name: output
    shape: [batch, 1000]
    dtype: f32
```

### Benchmark Output

```
Benchmark Results
=================
Model: model.axonml
Device: CUDA:0
Batch Size: 32
Warmup: 10 iterations
Benchmark: 100 iterations

Latency:
  Mean: 12.34 ms
  Std: 0.56 ms
  Min: 11.23 ms
  Max: 14.67 ms
  P50: 12.21 ms
  P95: 13.45 ms
  P99: 14.12 ms

Throughput: 2594 samples/sec
Memory: 1.2 GB peak
```

## Global Options

```bash
# Set device
axonml --device cuda:0 <command>

# Set log level
axonml --log-level debug <command>

# Quiet mode
axonml --quiet <command>

# JSON output
axonml --json <command>
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AXONML_DEVICE` | Default device (cpu, cuda:0, etc.) |
| `AXONML_LOG_LEVEL` | Log level (error, warn, info, debug) |
| `AXONML_CACHE_DIR` | Model cache directory |
| `AXONML_DATA_DIR` | Default data directory |

## Part of Axonml

The CLI is included when you install the axonml package:

```bash
cargo install axonml
```

## License

MIT OR Apache-2.0
