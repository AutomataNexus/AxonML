# axonml-quant

<p align="center">
  <!-- Logo placeholder -->
  <img src="https://raw.githubusercontent.com/AutomataNexus/AxonML/main/AxonML-logo.png" alt="AxonML Logo" width="200" height="200" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-1.75+-orange.svg" alt="Rust: 1.75+"></a>
  <a href="https://crates.io/crates/axonml-quant"><img src="https://img.shields.io/badge/crates.io-0.6.5-green.svg" alt="Version: 0.6.5"></a>
  <a href="https://github.com/axonml/axonml"><img src="https://img.shields.io/badge/part%20of-AxonML-blueviolet.svg" alt="Part of AxonML"></a>
</p>

## Overview

`axonml-quant` provides model quantization for AxonML. It covers the GGUF-family
block formats (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, F16, F32) plus Microsoft's BitNet
b1.58 I2_S ternary format (GGUF dtype 36, 128-weight blocks, group-strided
2-bit layout). It exposes a `QuantizedLinear` drop-in layer, a `QuantizedModel`
wrapper with custom `AXQT` serialization, calibration methods (MinMax,
Percentile, Entropy/KL-divergence, MeanStd), and error analysis.

## Features

- **GGUF-family formats**: Q8_0 (8-bit), Q4_0 / Q4_1 (4-bit), Q5_0 / Q5_1 (5-bit), F16, F32 — all with 32-element block size except F16/F32
- **BitNet b1.58 I2_S**: 128-weight ternary blocks in Microsoft's group-strided 2-bit layout, verified against `microsoft/BitNet` reference (2026-04-14)
  - `matmul_i2s` — fused add-only f32-activation matmul (ternary: +act / −act / skip)
  - `matmul_i2s_i8` — int8-activation fused path with runtime AVX-VNNI dispatch + scalar fallback (+ `matmul_i2s_i8_avxvnni` feature-gated stub, currently delegates to scalar)
  - `quantize_row_to_int8` — per-row absmax int8 activation quantizer
- **Calibration methods**: `MinMax`, `Percentile(p×10)`, `Entropy` (TensorRT-style KL divergence), `MeanStd(k×10)`
- **Parallel processing**: Rayon-parallel block quantization/dequantization and per-column matmul
- **Error analysis**: `compute_quantization_stats` returns RMSE, max error, mean error, compression ratio
- **Inference layer**: `QuantizedLinear::forward_f32` supports batch matmul with Q8/Q4/Q4_1/Q5/Q5_1/F16/F32 weights; `forward_var` integrates with `axonml-autograd` (forward-only, no grad)
- **Model-level API**: `QuantizedModel::from_module` quantizes all `axonml_nn::Module` parameters; `load_into_module` dequantizes back for inference
- **Serialization**: custom `AXQT` binary format (magic `AXQT`, version 1) via `serialize_quantized` / `deserialize_quantized`

## Modules

| Module | Description |
|--------|-------------|
| `types` | `QuantType` enum, `Q8Block`, `Q4Block`, `Q4_1Block`, `Q5Block`, `Q5_1Block`, `QuantizedBlock`, `QuantizedTensor` |
| `quantize` | Tensor/model quantization (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, F16, F32) + `compute_quantization_error` / `compute_quantization_stats` |
| `dequantize` | Block and tensor reconstruction to f32 |
| `bitnet` | I2_S 1.58-bit ternary — `I2sBlock`, `dequantize_i2s(_block)`, `matmul_i2s`, `matmul_i2s_i8`, `quantize_row_to_int8`, `decode_trit`/`encode_trit`, `bytes_for_elements` |
| `calibration` | `CalibrationData` (Welford streaming mean/variance + histogram percentiles), `CalibrationMethod`, `calibrate`, `calibrate_batch` |
| `inference` | `QuantizedLinear`, `QuantizedModel`, `quantize_parameters`, `serialize_quantized` / `deserialize_quantized` |
| `error` | `QuantError` and `QuantResult` |

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
axonml-quant = "0.6.5"
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
use axonml_quant::{calibrate, QuantType};
use axonml_quant::calibration::CalibrationMethod;

// Calibrate using percentile method (99.9%) — argument is percentile * 10
let calib_data = calibrate(&sample_tensor, CalibrationMethod::Percentile(999))?;

// Get optimal scale for quantization
let scale = calib_data.symmetric_scale(QuantType::Q8_0);

// Or use asymmetric quantization
let (scale, zero_point) = calib_data.asymmetric_scale(QuantType::Q8_0);
```

### Quantization Error Analysis

```rust
use axonml_quant::quantize::{compute_quantization_stats};
use axonml_quant::QuantType;

let stats = compute_quantization_stats(&original, &dequantized, QuantType::Q8_0);
println!("RMSE: {:.6}", stats.rmse);
println!("Max Error: {:.6}", stats.max_error);
println!("Mean Error: {:.6}", stats.mean_error);
println!("Compression: {:.2}x", stats.compression_ratio);
```

### BitNet I2_S Ternary

```rust
use axonml_quant::bitnet::{matmul_i2s, matmul_i2s_i8, quantize_row_to_int8, I2S_BLOCK_SIZE};

// f32-activation path
let mut output = vec![0.0f32; m * n];
matmul_i2s(&activations, m, k, &weight_bytes, n, tensor_scale, &mut output);

// int8-activation path (AVX-VNNI-dispatched, scalar fallback)
let mut acts_i8 = vec![0i8; m * k];
let mut act_scales = vec![0.0f32; m];
for i in 0..m {
    act_scales[i] = quantize_row_to_int8(
        &activations[i * k..(i + 1) * k],
        &mut acts_i8[i * k..(i + 1) * k],
    );
}
matmul_i2s_i8(&acts_i8, &act_scales, m, k, &weight_bytes, n, tensor_scale, &mut output);
```

`k` must be a multiple of `I2S_BLOCK_SIZE` (128).

### Quantized Inference Layer

```rust
use axonml_quant::{QuantizedLinear, QuantType};

let qlinear = QuantizedLinear::from_linear_params(
    &weights,
    Some(&bias),
    /* in_features */ 512,
    /* out_features */ 128,
    QuantType::Q8_0,
);

let output = qlinear.forward_f32(&input_data, /* batch_size */ 1);
```

## Quantization Types

| Type | Bits | Block Size | Block Bytes | Compression | Use Case |
|------|------|------------|-------------|-------------|----------|
| Q8_0 | 8 | 32 | 34 | ~3.76x | High accuracy, moderate compression |
| Q4_0 | 4 | 32 | 18 | ~7.11x | Signed 4-bit, symmetric |
| Q4_1 | 4 | 32 | 20 | ~6.40x | Unsigned 4-bit, with per-block min |
| Q5_0 | 5 | 32 | 22 | ~5.82x | 5-bit signed, symmetric |
| Q5_1 | 5 | 32 | 24 | ~5.33x | 5-bit unsigned, with per-block min |
| I2_S (BitNet b1.58) | 2 | 128 | 32 | ~16x | Ternary {−1, 0, +1} with one tensor-wide scale |
| F16 | 16 | 1 | 2 | 2x | Minimal accuracy loss |
| F32 | 32 | 1 | 4 | 1x | No compression (reference) |

## Tests

```bash
cargo test -p axonml-quant
```

Test coverage includes: quant-type properties and string parsing, Q4 / Q5 / Q8
block pack-unpack roundtrips, Q-type quantization shape + compression checks,
calibration (MinMax, Percentile, symmetric/asymmetric scales, dynamic range),
AXQT serialize/deserialize roundtrip, `QuantizedLinear` forward parity vs f32
reference, and for BitNet: trit encode/decode, block roundtrip, group-strided
layout correctness, single/multi-block dequantization, fused vs reference
matmul agreement (f32 and int8), int8 activation roundtrip, and misaligned-`k`
rejection.

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.
