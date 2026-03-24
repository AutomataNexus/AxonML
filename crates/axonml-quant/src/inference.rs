//! Quantized Inference — fast inference with quantized weights
//!
//! # File
//! `crates/axonml-quant/src/inference.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 16, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use crate::dequantize::dequantize_tensor;
use crate::quantize::quantize_tensor;
use crate::types::{Q4Block, Q4_1Block, Q8Block, QuantType, QuantizedBlock, QuantizedTensor};
use axonml_tensor::Tensor;
use half::f16;
use rayon::prelude::*;

// =============================================================================
// Quantized Matmul Kernels
// =============================================================================

/// Compute dot product of f32 activation vector with Q8_0 quantized weight row.
/// This is the inner kernel — called per output element.
#[inline]
fn dot_q8_block(block: &Q8Block, activations: &[f32]) -> f32 {
    let scale = f32::from(block.scale);
    let mut sum = 0.0f32;
    for i in 0..block.data.len().min(activations.len()) {
        sum += (block.data[i] as f32) * activations[i];
    }
    sum * scale
}

/// Compute dot product of f32 activation vector with Q4_0 quantized weight row.
#[inline]
fn dot_q4_block(block: &Q4Block, activations: &[f32]) -> f32 {
    let scale = f32::from(block.scale);
    let unpacked = block.unpack();
    let mut sum = 0.0f32;
    for i in 0..unpacked.len().min(activations.len()) {
        sum += (unpacked[i] as f32) * activations[i];
    }
    sum * scale
}

/// Compute dot product of f32 activation vector with Q4_1 quantized weight row.
#[inline]
fn dot_q4_1_block(block: &Q4_1Block, activations: &[f32]) -> f32 {
    let scale = f32::from(block.scale);
    let min = f32::from(block.min);
    let unpacked = block.unpack();
    let mut sum = 0.0f32;
    for i in 0..unpacked.len().min(activations.len()) {
        sum += (unpacked[i] as f32 * scale + min) * activations[i];
    }
    sum
}

/// Compute dot product of f32 activation vector with F16 weight block.
#[inline]
fn dot_f16_block(data: &[f16], activations: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..data.len().min(activations.len()) {
        sum += f32::from(data[i]) * activations[i];
    }
    sum
}

/// Dot product of an activation vector against a single quantized block.
#[inline]
fn dot_block(block: &QuantizedBlock, activations: &[f32]) -> f32 {
    match block {
        QuantizedBlock::Q8(b) => dot_q8_block(b, activations),
        QuantizedBlock::Q4(b) => dot_q4_block(b, activations),
        QuantizedBlock::Q4_1(b) => dot_q4_1_block(b, activations),
        QuantizedBlock::F16(data) => dot_f16_block(data, activations),
        QuantizedBlock::F32(data) => {
            let mut sum = 0.0f32;
            for i in 0..data.len().min(activations.len()) {
                sum += data[i] * activations[i];
            }
            sum
        }
    }
}

// =============================================================================
// QuantizedLinear — drop-in replacement for Linear
// =============================================================================

/// A linear layer with quantized weights for fast inference.
///
/// Stores weights as `QuantizedTensor` (Q8/Q4/F16) and dequantizes on-the-fly
/// during the matrix multiply. Bias remains in f32.
///
/// # Usage
/// ```ignore
/// use axonml_quant::inference::QuantizedLinear;
/// use axonml_quant::QuantType;
///
/// let qlinear = QuantizedLinear::from_linear_params(&weights, Some(&bias), 512, 128, QuantType::Q8_0);
/// let output = qlinear.forward_f32(&input_data, 1);
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Quantized weight matrix (out_features x in_features, row-major).
    weight: QuantizedTensor,
    /// Bias vector (f32, not quantized).
    bias: Option<Vec<f32>>,
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Quantization type used.
    pub quant_type: QuantType,
    /// Number of blocks per output row (cached for fast access).
    blocks_per_row: usize,
}

impl QuantizedLinear {
    /// Create a QuantizedLinear from an axonml_nn::Linear layer.
    pub fn from_linear_params(
        weight_data: &[f32],
        bias_data: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        quant_type: QuantType,
    ) -> Self {
        // Weight shape is [out_features, in_features]
        let weight_tensor = Tensor::from_vec(weight_data.to_vec(), &[out_features, in_features])
            .expect("Failed to create weight tensor for quantization");

        let weight =
            quantize_tensor(&weight_tensor, quant_type).expect("Failed to quantize weight tensor");

        let block_size = quant_type.block_size();
        let blocks_per_row = (in_features + block_size - 1) / block_size;

        QuantizedLinear {
            weight,
            bias: bias_data.map(|b| b.to_vec()),
            in_features,
            out_features,
            quant_type,
            blocks_per_row,
        }
    }

    /// Forward pass: f32 input → f32 output.
    ///
    /// Input shape: `[batch, in_features]`
    /// Output shape: `[batch, out_features]`
    ///
    /// Performs quantized matrix multiplication: each output element is computed
    /// by iterating over the weight row's quantized blocks and accumulating
    /// dot products with the corresponding activation slices.
    pub fn forward_f32(&self, input: &[f32], batch_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * self.out_features];

        // For non-block types (F16, F32), extract flat weight data and do direct matmul
        if !self.quant_type.is_block_quantized() {
            let weight_flat = self.extract_flat_weights();
            output
                .par_chunks_mut(self.out_features)
                .enumerate()
                .for_each(|(b, out_row)| {
                    let input_row = &input[b * self.in_features..(b + 1) * self.in_features];
                    for o in 0..self.out_features {
                        let w_start = o * self.in_features;
                        let mut sum = 0.0f32;
                        for k in 0..self.in_features {
                            sum += weight_flat[w_start + k] * input_row[k];
                        }
                        if let Some(ref bias) = self.bias {
                            sum += bias[o];
                        }
                        out_row[o] = sum;
                    }
                });
            return output;
        }

        // Block-quantized path (Q8, Q4, Q4_1): iterate over blocks per weight row
        let block_size = self.quant_type.block_size();

        output
            .par_chunks_mut(self.out_features)
            .enumerate()
            .for_each(|(b, out_row)| {
                let input_row = &input[b * self.in_features..(b + 1) * self.in_features];

                for o in 0..self.out_features {
                    let row_block_start = o * self.blocks_per_row;
                    let mut sum = 0.0f32;

                    for blk_idx in 0..self.blocks_per_row {
                        let act_start = blk_idx * block_size;
                        let act_end = (act_start + block_size).min(self.in_features);
                        let act_slice = &input_row[act_start..act_end];

                        let block = &self.weight.blocks[row_block_start + blk_idx];
                        sum += dot_block(block, act_slice);
                    }

                    if let Some(ref bias) = self.bias {
                        sum += bias[o];
                    }

                    out_row[o] = sum;
                }
            });

        output
    }

    /// Extract flat f32 weights (for F16/F32 non-block types).
    fn extract_flat_weights(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.in_features * self.out_features);
        for block in &self.weight.blocks {
            match block {
                QuantizedBlock::F16(data) => {
                    flat.extend(data.iter().map(|v| f32::from(*v)));
                }
                QuantizedBlock::F32(data) => {
                    flat.extend_from_slice(data);
                }
                _ => {} // block-quantized handled separately
            }
        }
        flat
    }

    /// Forward pass with Variable input/output (for integration with autograd).
    ///
    /// Note: Quantized inference is forward-only (no gradient tracking).
    /// The output Variable has `requires_grad = false`.
    pub fn forward_var(&self, input: &axonml_autograd::Variable) -> axonml_autograd::Variable {
        let shape = input.shape();
        let batch = if shape.len() > 1 { shape[0] } else { 1 };
        let input_data = input.data().to_vec();

        let output_data = self.forward_f32(&input_data, batch);

        let output_tensor = Tensor::from_vec(output_data, &[batch, self.out_features])
            .expect("Failed to create output tensor");

        axonml_autograd::Variable::new(output_tensor, false)
    }

    /// Memory usage in bytes (weights only, excludes bias).
    pub fn weight_bytes(&self) -> usize {
        self.weight.size_bytes()
    }

    /// Compression ratio vs f32 weights.
    pub fn compression_ratio(&self) -> f32 {
        self.weight.compression_ratio()
    }

    /// Dequantize weights back to f32 (for debugging/validation).
    pub fn dequantize_weights(&self) -> Tensor<f32> {
        dequantize_tensor(&self.weight).expect("Failed to dequantize weights")
    }
}

// =============================================================================
// Model Quantization — convert a full model
// =============================================================================

/// Quantize all parameters of a model, returning a flat list of quantized tensors.
///
/// This extracts all parameters from a Module, quantizes each one, and returns
/// them in order. Use with `QuantizedModel` for full inference.
pub fn quantize_parameters(
    params: &[axonml_nn::Parameter],
    quant_type: QuantType,
) -> Vec<QuantizedTensor> {
    params
        .par_iter()
        .map(|param| {
            let tensor = param.data();
            quantize_tensor(&tensor, quant_type).expect("Failed to quantize parameter")
        })
        .collect()
}

// =============================================================================
// QuantizedModel — generic quantized inference wrapper
// =============================================================================

/// A fully quantized model for inference.
///
/// Wraps a collection of quantized parameters and provides fast forward pass
/// by dequantizing weights on-the-fly during computation.
///
/// # Usage
/// ```ignore
/// use axonml_quant::inference::QuantizedModel;
/// use axonml_quant::QuantType;
///
/// let qmodel = QuantizedModel::from_module(&model, QuantType::Q8_0);
/// println!("{}", qmodel.summary());
/// qmodel.load_into_module(&model); // dequant weights back for inference
/// ```
pub struct QuantizedModel {
    /// Quantized weight tensors (in parameter order).
    pub quantized_params: Vec<QuantizedTensor>,
    /// Quantization type used.
    pub quant_type: QuantType,
    /// Total original parameter count.
    pub total_params: usize,
    /// Total quantized size in bytes.
    pub total_bytes: usize,
    /// Original f32 size in bytes.
    pub original_bytes: usize,
}

impl QuantizedModel {
    /// Quantize a Module's parameters.
    pub fn from_module<M: axonml_nn::Module>(module: &M, quant_type: QuantType) -> Self {
        let params = module.parameters();
        let total_params: usize = params.iter().map(|p| p.numel()).sum();
        let original_bytes = total_params * 4;

        let quantized_params = quantize_parameters(&params, quant_type);

        let total_bytes: usize = quantized_params.iter().map(|q| q.size_bytes()).sum();

        QuantizedModel {
            quantized_params,
            quant_type,
            total_params,
            total_bytes,
            original_bytes,
        }
    }

    /// Load quantized weights back into a Module for inference.
    ///
    /// Dequantizes all parameters and updates the module's parameters in-place.
    /// This is the simplest integration path — the model runs at full f32 speed
    /// but loads from a compressed checkpoint.
    pub fn load_into_module<M: axonml_nn::Module>(&self, module: &M) {
        let params = module.parameters();
        for (param, qparam) in params.iter().zip(self.quantized_params.iter()) {
            let tensor = dequantize_tensor(qparam).expect("Failed to dequantize parameter");
            param.update_data(tensor);
        }
    }

    /// Compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        self.original_bytes as f32 / self.total_bytes as f32
    }

    /// Print a summary of the quantized model.
    pub fn summary(&self) -> String {
        format!(
            "QuantizedModel(type={}, params={}, f32={:.1}MB, quant={:.1}MB, ratio={:.1}x)",
            self.quant_type,
            self.total_params,
            self.original_bytes as f64 / 1024.0 / 1024.0,
            self.total_bytes as f64 / 1024.0 / 1024.0,
            self.compression_ratio(),
        )
    }
}

// =============================================================================
// Serialization — save/load quantized models
// =============================================================================

/// Serialize a QuantizedModel to bytes (for .axonml files).
pub fn serialize_quantized(model: &QuantizedModel) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic: "AXQT" (AxonML Quantized)
    buf.extend_from_slice(b"AXQT");
    // Version
    buf.push(1u8);
    // Quant type
    buf.push(match model.quant_type {
        QuantType::Q8_0 => 0,
        QuantType::Q4_0 => 1,
        QuantType::Q4_1 => 2,
        QuantType::Q5_0 => 3,
        QuantType::Q5_1 => 4,
        QuantType::F16 => 5,
        QuantType::F32 => 6,
    });
    // Number of tensors
    buf.extend_from_slice(&(model.quantized_params.len() as u32).to_le_bytes());
    // Total params
    buf.extend_from_slice(&(model.total_params as u64).to_le_bytes());

    // Each tensor: shape_len + shape + num_blocks + blocks
    for qt in &model.quantized_params {
        // Shape
        buf.extend_from_slice(&(qt.shape.len() as u32).to_le_bytes());
        for &dim in &qt.shape {
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        // Number of blocks
        buf.extend_from_slice(&(qt.blocks.len() as u32).to_le_bytes());
        // Block data
        for block in &qt.blocks {
            match block {
                QuantizedBlock::Q8(b) => {
                    buf.extend_from_slice(&b.to_bytes());
                }
                QuantizedBlock::Q4(b) => {
                    buf.extend_from_slice(&b.to_bytes());
                }
                QuantizedBlock::Q4_1(b) => {
                    buf.extend_from_slice(&b.to_bytes());
                }
                QuantizedBlock::F16(data) => {
                    for &v in data {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                QuantizedBlock::F32(data) => {
                    for &v in data {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }

    buf
}

/// Deserialize a QuantizedModel from bytes.
pub fn deserialize_quantized(data: &[u8]) -> Result<QuantizedModel, String> {
    if data.len() < 18 || &data[0..4] != b"AXQT" {
        return Err("Invalid quantized model file (bad magic)".to_string());
    }

    let version = data[4];
    if version != 1 {
        return Err(format!("Unsupported quantized model version: {version}"));
    }

    let quant_type = match data[5] {
        0 => QuantType::Q8_0,
        1 => QuantType::Q4_0,
        2 => QuantType::Q4_1,
        3 => QuantType::Q5_0,
        4 => QuantType::Q5_1,
        5 => QuantType::F16,
        6 => QuantType::F32,
        x => return Err(format!("Unknown quant type byte: {x}")),
    };

    let num_tensors = u32::from_le_bytes([data[6], data[7], data[8], data[9]]) as usize;
    let total_params = u64::from_le_bytes([
        data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17],
    ]) as usize;

    let mut offset = 18usize;
    let mut quantized_params = Vec::with_capacity(num_tensors);

    let block_bytes = quant_type.bytes_per_block();

    for _ in 0..num_tensors {
        if offset + 4 > data.len() {
            return Err("Truncated quantized model file".to_string());
        }

        // Shape
        let shape_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let dim = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            shape.push(dim);
            offset += 4;
        }

        // Number of blocks
        let num_blocks = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        // Read blocks
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            if offset + block_bytes > data.len() {
                return Err("Truncated block data".to_string());
            }

            let block = match quant_type {
                QuantType::Q8_0 => {
                    let b =
                        Q8Block::from_bytes(&data[offset..]).ok_or("Failed to parse Q8 block")?;
                    QuantizedBlock::Q8(b)
                }
                QuantType::Q4_0 => {
                    let b =
                        Q4Block::from_bytes(&data[offset..]).ok_or("Failed to parse Q4 block")?;
                    QuantizedBlock::Q4(b)
                }
                QuantType::Q4_1 => {
                    let scale = f16::from_le_bytes([data[offset], data[offset + 1]]);
                    let min = f16::from_le_bytes([data[offset + 2], data[offset + 3]]);
                    let mut block_data = [0u8; 16];
                    block_data.copy_from_slice(&data[offset + 4..offset + 20]);
                    QuantizedBlock::Q4_1(Q4_1Block::new(scale, min, block_data))
                }
                QuantType::F16 => {
                    let v = f16::from_le_bytes([data[offset], data[offset + 1]]);
                    QuantizedBlock::F16(vec![v])
                }
                QuantType::F32 => {
                    let v = f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    QuantizedBlock::F32(vec![v])
                }
                _ => return Err("Unsupported quant type for deserialization".to_string()),
            };

            blocks.push(block);
            offset += block_bytes;
        }

        quantized_params.push(QuantizedTensor::new(shape, quant_type, blocks));
    }

    let total_bytes: usize = quantized_params.iter().map(|q| q.size_bytes()).sum();
    let original_bytes = total_params * 4;

    Ok(QuantizedModel {
        quantized_params,
        quant_type,
        total_params,
        total_bytes,
        original_bytes,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_linear_q8_forward() {
        let in_f = 64;
        let out_f = 16;
        let weight: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32 * 0.01) - 5.0).collect();
        let bias: Vec<f32> = (0..out_f).map(|i| i as f32 * 0.1).collect();

        let ql =
            QuantizedLinear::from_linear_params(&weight, Some(&bias), in_f, out_f, QuantType::Q8_0);

        let input: Vec<f32> = (0..in_f).map(|i| i as f32 * 0.1).collect();
        let output = ql.forward_f32(&input, 1);

        assert_eq!(output.len(), out_f);
        // Output should not be all zeros
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 0.01, "Output should be non-zero, got sum={sum}");

        // Compare with f32 reference
        let ref_ql =
            QuantizedLinear::from_linear_params(&weight, Some(&bias), in_f, out_f, QuantType::F32);
        let ref_output = ref_ql.forward_f32(&input, 1);

        // Q8 should be very close to f32
        let max_err: f32 = output
            .iter()
            .zip(ref_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1.0, "Q8 error too large: {max_err}");
    }

    #[test]
    fn test_quantized_linear_q4_forward() {
        let in_f = 64;
        let out_f = 8;
        let weight: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32 * 0.02) - 5.0).collect();

        let ql = QuantizedLinear::from_linear_params(&weight, None, in_f, out_f, QuantType::Q4_0);

        let input: Vec<f32> = (0..in_f).map(|i| i as f32 * 0.1).collect();
        let output = ql.forward_f32(&input, 1);

        assert_eq!(output.len(), out_f);
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 0.01, "Output should be non-zero");
    }

    #[test]
    fn test_quantized_linear_batch() {
        let in_f = 32;
        let out_f = 8;
        let batch = 4;
        let weight: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32 * 0.01) - 1.0).collect();

        let ql = QuantizedLinear::from_linear_params(&weight, None, in_f, out_f, QuantType::Q8_0);

        let input: Vec<f32> = (0..batch * in_f).map(|i| i as f32 * 0.01).collect();
        let output = ql.forward_f32(&input, batch);

        assert_eq!(output.len(), batch * out_f);
    }

    #[test]
    fn test_quantized_linear_compression() {
        let in_f = 1024;
        let out_f = 512;
        let weight: Vec<f32> = vec![0.1; in_f * out_f];

        let ql_q8 =
            QuantizedLinear::from_linear_params(&weight, None, in_f, out_f, QuantType::Q8_0);
        let ql_q4 =
            QuantizedLinear::from_linear_params(&weight, None, in_f, out_f, QuantType::Q4_0);

        assert!(ql_q8.compression_ratio() > 3.5, "Q8 should compress ~4x");
        assert!(ql_q4.compression_ratio() > 6.0, "Q4 should compress ~7-8x");
    }

    #[test]
    fn test_quantized_model_roundtrip() {
        let in_f = 32;
        let out_f = 8;
        let weight: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let weight_tensor = Tensor::from_vec(weight.clone(), &[out_f, in_f]).unwrap();
        let qt = quantize_tensor(&weight_tensor, QuantType::Q8_0).unwrap();

        let model = QuantizedModel {
            quantized_params: vec![qt],
            quant_type: QuantType::Q8_0,
            total_params: in_f * out_f,
            total_bytes: 0,
            original_bytes: in_f * out_f * 4,
        };

        let serialized = serialize_quantized(&model);
        let deserialized = deserialize_quantized(&serialized).unwrap();

        assert_eq!(deserialized.quant_type, QuantType::Q8_0);
        assert_eq!(deserialized.quantized_params.len(), 1);
        assert_eq!(deserialized.quantized_params[0].shape, vec![out_f, in_f]);
    }

    #[test]
    fn test_quantized_linear_variable_forward() {
        let in_f = 32;
        let out_f = 8;
        let weight: Vec<f32> = (0..in_f * out_f).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let bias: Vec<f32> = vec![0.5; out_f];

        let ql =
            QuantizedLinear::from_linear_params(&weight, Some(&bias), in_f, out_f, QuantType::Q8_0);

        let input_tensor =
            Tensor::from_vec((0..2 * in_f).map(|i| i as f32 * 0.1).collect(), &[2, in_f]).unwrap();
        let input_var = axonml_autograd::Variable::new(input_tensor, false);

        let output = ql.forward_var(&input_var);

        assert_eq!(output.shape(), vec![2, out_f]);
    }
}
