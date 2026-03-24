# axonml-quant

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-quant"><img src="https://img.shields.io/badge/crates.io-0.1.0-green.svg" alt="Version: 0.1.0"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-blueviolet.svg" alt="Part of AxonML"></a>
</p>

## Overview

`axonml-quant` provides model quantization support for reducing model size and improving inference performance. It supports multiple quantization formats including 8-bit, 4-bit, and half-precision floating point, with calibration methods for determining optimal quantization parameters.

## Features

- **Multiple Quantization Formats**: Supports Q8_0 (8-bit), Q4_0/Q4_1 (4-bit), Q5_0/Q5_1 (5-bit), F16 (half-precision), and F32 (full precision)
- **Block Quantization**: Per-block scale factors for improved accuracy with 32-element block size
- **Calibration Methods**: MinMax, Percentile, Entropy, and MeanStd calibration for optimal quantization parameters
- **Parallel Processing**: Uses Rayon for parallel quantization and dequantization operations
- **Compression Statistics**: Tracks compression ratios and quantization error metrics (RMSE, max error, mean error)
- **Model Quantization**: Batch quantization of named tensor collections for full model compression
- **Round-trip Support**: Full dequantization support to restore tensors to floating point

## Modules

| Module | Description |
|--------|-------------|
| `types` | Quantization type definitions, block structures (Q8Block, Q4Block, Q4_1Block), and QuantizedTensor |
| `quantize` | Functions for quantizing tensors to various formats with parallel processing |
| `dequantize` | Functions for converting quantized tensors back to floating point |
| `calibration` | Calibration data collection and methods for optimal quantization parameters |
| `error` | Error types and Result alias for quantization operations |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-quant = "0.1.0"
```

### Basic Quantization

```rust
use axonml_quant::{quantize_tensor, dequantize_tensor, QuantType};
use axonml_tensor::Tensor;

// Create a tensor
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;

// Quantize to 8-bit
let quantized = quantize_tensor(&tensor, QuantType::Q8_0)?;

// Check compression ratio
println!("Compression ratio: {:.2}x", quantized.compression_ratio());

// Dequantize back to f32
let restored = dequantize_tensor(&quantized)?;
```

### Model Quantization

```rust
use axonml_quant::{quantize_model, QuantType};

// Quantize multiple named tensors
let tensors = vec![
    ("weights", &weight_tensor),
    ("bias", &bias_tensor),
];
let quantized_model = quantize_model(&tensors, QuantType::Q4_0)?;
```

### Calibration

```rust
use axonml_quant::{calibrate, CalibrationMethod, CalibrationData};

// Calibrate using percentile method (99.9%)
let calib_data = calibrate(&sample_tensor, CalibrationMethod::Percentile(999))?;

// Get optimal scale for quantization
let scale = calib_data.symmetric_scale(QuantType::Q8_0);

// Or use asymmetric quantization
let (scale, zero_point) = calib_data.asymmetric_scale(QuantType::Q8_0);
```

### Quantization Error Analysis

```rust
use axonml_quant::{compute_quantization_stats, QuantType};

let stats = compute_quantization_stats(&original, &dequantized, QuantType::Q8_0);
println!("RMSE: {:.6}", stats.rmse);
println!("Max Error: {:.6}", stats.max_error);
println!("Mean Error: {:.6}", stats.mean_error);
println!("Compression: {:.2}x", stats.compression_ratio);
```

## Quantization Types

| Type | Bits | Block Size | Compression | Use Case |
|------|------|------------|-------------|----------|
| Q8_0 | 8 | 32 | 4x | High accuracy, moderate compression |
| Q4_0 | 4 | 32 | 8x | Good balance of size and accuracy |
| Q4_1 | 4 | 32 | ~6x | Better accuracy with min/max tracking |
| Q5_0 | 5 | 32 | ~6x | Middle ground between Q4 and Q8 |
| F16 | 16 | 1 | 2x | Minimal accuracy loss |
| F32 | 32 | 1 | 1x | No compression (reference) |

## Tests

Run the test suite:

```bash
cargo test -p axonml-quant
```

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
