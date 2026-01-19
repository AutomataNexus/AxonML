# axonml-quant

[![Crates.io](https://img.shields.io/crates/v/axonml-quant.svg)](https://crates.io/crates/axonml-quant)
[![Docs.rs](https://docs.rs/axonml-quant/badge.svg)](https://docs.rs/axonml-quant)
[![Downloads](https://img.shields.io/crates/d/axonml-quant.svg)](https://crates.io/crates/axonml-quant)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Model quantization for the [Axonml](https://github.com/AutomataNexus/AxonML) machine learning framework.

## Overview

`axonml-quant` provides model quantization for reduced memory footprint and faster inference. Supports INT8, INT4, and half-precision formats with minimal accuracy loss.

## Features

### Quantization Formats
- **Q8_0** - 8-bit integer (INT8)
- **Q4_0** - 4-bit integer, basic
- **Q4_1** - 4-bit with min/max scaling
- **Q5_0** - 5-bit integer
- **Q5_1** - 5-bit with min/max scaling
- **F16** - Half precision (16-bit float)

### Techniques
- **Post-training quantization** - No retraining required
- **Block-wise quantization** - Per-block scale factors
- **Calibration** - Data-driven range estimation

### Benefits
- **~8x compression** with Q4 formats
- **~4x compression** with Q8
- **Faster inference** on CPU
- **Reduced memory** for edge deployment

## Installation

```toml
[dependencies]
axonml-quant = "0.1"
```

## Usage

### Basic Quantization

```rust
use axonml_quant::{quantize, QuantFormat};

let model = load_model("model.axonml")?;

// Quantize to INT8
let quantized = quantize(&model, QuantFormat::Q8_0)?;
quantized.save("model_q8.axonml")?;

// Check compression
println!("Original: {} MB", model.size_mb());
println!("Quantized: {} MB", quantized.size_mb());
```

### INT4 Quantization

```rust
use axonml_quant::{quantize, QuantFormat};

// Q4_0: Simple 4-bit quantization
let q4_model = quantize(&model, QuantFormat::Q4_0)?;

// Q4_1: Better accuracy with min/max scaling
let q4_1_model = quantize(&model, QuantFormat::Q4_1)?;
```

### Calibration for Better Accuracy

```rust
use axonml_quant::{Quantizer, QuantFormat};

let quantizer = Quantizer::new(QuantFormat::Q8_0)
    .calibration_data(&calibration_dataset)  // Representative data
    .percentile(99.9);  // Clip outliers

let quantized = quantizer.quantize(&model)?;
```

### Half Precision (F16)

```rust
use axonml_quant::{quantize, QuantFormat};

// Convert to half precision
let fp16_model = quantize(&model, QuantFormat::F16)?;

// 2x compression, good accuracy
```

### Quantization Info

```rust
use axonml_quant::get_quant_info;

let info = get_quant_info(&quantized_model);
println!("Format: {:?}", info.format);
println!("Original size: {} bytes", info.original_size);
println!("Quantized size: {} bytes", info.quantized_size);
println!("Compression ratio: {:.2}x", info.compression_ratio);
```

### Inference with Quantized Model

```rust
use axonml_quant::load_quantized;

// Load and run quantized model
let model = load_quantized("model_q4.axonml")?;
let output = model.forward(&input);  // Automatic dequantization
```

## Format Comparison

| Format | Bits | Compression | Accuracy | Speed |
|--------|------|-------------|----------|-------|
| F32 | 32 | 1x (baseline) | Best | Baseline |
| F16 | 16 | 2x | Excellent | Faster |
| Q8_0 | 8 | 4x | Very Good | Faster |
| Q5_1 | 5 | ~6x | Good | Faster |
| Q4_1 | 4 | 8x | Good | Fastest |
| Q4_0 | 4 | 8x | Acceptable | Fastest |

## CLI Usage

```bash
# Quantize via CLI
axonml quant convert model.axonml --type q8_0 -o model_q8.axonml

# Show quantization info
axonml quant info model_q8.axonml

# Benchmark quantized model
axonml quant benchmark model_q8.axonml

# List supported formats
axonml quant list
```

## Part of Axonml

```toml
[dependencies]
axonml = "0.1"  # Includes quantization
```

## License

MIT OR Apache-2.0
