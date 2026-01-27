//! Quantization Functions
//!
//! Functions for quantizing tensors to various formats.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_tensor::Tensor;
use half::f16;
use rayon::prelude::*;

use crate::error::QuantResult;
use crate::types::{Q4Block, Q4_1Block, Q8Block, QuantType, QuantizedBlock, QuantizedTensor};
use crate::DEFAULT_BLOCK_SIZE;

// =============================================================================
// Public API
// =============================================================================

/// Quantizes a tensor to the specified quantization type.
///
/// # Arguments
/// * `tensor` - The input tensor to quantize
/// * `quant_type` - The target quantization type
///
/// # Returns
/// A quantized tensor
///
/// # Example
/// ```ignore
/// use axonml_quant::{quantize_tensor, QuantType};
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
/// let quantized = quantize_tensor(&tensor, QuantType::Q8_0)?;
/// ```
pub fn quantize_tensor(
    tensor: &Tensor<f32>,
    quant_type: QuantType,
) -> QuantResult<QuantizedTensor> {
    let data = tensor.to_vec();
    let shape = tensor.shape().to_vec();

    match quant_type {
        QuantType::Q8_0 => quantize_q8_0(&data, shape),
        QuantType::Q4_0 => quantize_q4_0(&data, shape),
        QuantType::Q4_1 => quantize_q4_1(&data, shape),
        QuantType::Q5_0 | QuantType::Q5_1 => {
            // Fall back to Q4 for now
            quantize_q4_0(&data, shape)
        }
        QuantType::F16 => quantize_f16(&data, shape),
        QuantType::F32 => quantize_f32(&data, shape),
    }
}

/// Quantizes a model (collection of named tensors).
///
/// # Arguments
/// * `tensors` - Named tensors to quantize
/// * `quant_type` - The target quantization type
///
/// # Returns
/// A map of quantized tensors
pub fn quantize_model(
    tensors: &[(&str, &Tensor<f32>)],
    quant_type: QuantType,
) -> QuantResult<Vec<(String, QuantizedTensor)>> {
    tensors
        .par_iter()
        .map(|(name, tensor)| {
            let quantized = quantize_tensor(tensor, quant_type)?;
            Ok((name.to_string(), quantized))
        })
        .collect()
}

// =============================================================================
// Q8_0 Quantization
// =============================================================================

/// Quantizes data to Q8_0 format (8-bit with per-block scale).
fn quantize_q8_0(data: &[f32], shape: Vec<usize>) -> QuantResult<QuantizedTensor> {
    let block_size = DEFAULT_BLOCK_SIZE;
    let n_blocks = (data.len() + block_size - 1) / block_size;

    let blocks: Vec<QuantizedBlock> = (0..n_blocks)
        .into_par_iter()
        .map(|block_idx| {
            let start = block_idx * block_size;
            let end = (start + block_size).min(data.len());
            let block_data = &data[start..end];

            // Find max absolute value for scale
            let max_abs = block_data
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, |a, b| a.max(b));

            // Compute scale (avoid division by zero)
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

            // Quantize to int8
            let mut quantized = [0i8; 32];
            for (i, &val) in block_data.iter().enumerate() {
                let q = (val / scale).round().clamp(-127.0, 127.0) as i8;
                quantized[i] = q;
            }

            QuantizedBlock::Q8(Q8Block::new(f16::from_f32(scale), quantized))
        })
        .collect();

    Ok(QuantizedTensor::new(shape, QuantType::Q8_0, blocks))
}

// =============================================================================
// Q4_0 Quantization
// =============================================================================

/// Quantizes data to Q4_0 format (4-bit with per-block scale).
fn quantize_q4_0(data: &[f32], shape: Vec<usize>) -> QuantResult<QuantizedTensor> {
    let block_size = DEFAULT_BLOCK_SIZE;
    let n_blocks = (data.len() + block_size - 1) / block_size;

    let blocks: Vec<QuantizedBlock> = (0..n_blocks)
        .into_par_iter()
        .map(|block_idx| {
            let start = block_idx * block_size;
            let end = (start + block_size).min(data.len());
            let block_data = &data[start..end];

            // Find max absolute value for scale
            let max_abs = block_data
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, |a, b| a.max(b));

            // Compute scale (4-bit range is -8 to 7)
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };

            // Quantize to 4-bit (stored as i8 in range -8 to 7)
            let mut quantized = [0i8; 32];
            for (i, &val) in block_data.iter().enumerate() {
                let q = (val / scale).round().clamp(-8.0, 7.0) as i8;
                quantized[i] = q;
            }

            // Pack into bytes
            let packed = Q4Block::pack(&quantized);

            QuantizedBlock::Q4(Q4Block::new(f16::from_f32(scale), packed))
        })
        .collect();

    Ok(QuantizedTensor::new(shape, QuantType::Q4_0, blocks))
}

// =============================================================================
// Q4_1 Quantization
// =============================================================================

/// Quantizes data to Q4_1 format (4-bit with per-block scale and min).
fn quantize_q4_1(data: &[f32], shape: Vec<usize>) -> QuantResult<QuantizedTensor> {
    let block_size = DEFAULT_BLOCK_SIZE;
    let n_blocks = (data.len() + block_size - 1) / block_size;

    let blocks: Vec<QuantizedBlock> = (0..n_blocks)
        .into_par_iter()
        .map(|block_idx| {
            let start = block_idx * block_size;
            let end = (start + block_size).min(data.len());
            let block_data = &data[start..end];

            // Find min and max
            let min = block_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = block_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute scale (4-bit unsigned range is 0 to 15)
            let scale = if max > min { (max - min) / 15.0 } else { 1.0 };

            // Quantize to 4-bit unsigned
            let mut quantized = [0u8; 32];
            for (i, &val) in block_data.iter().enumerate() {
                let q = ((val - min) / scale).round().clamp(0.0, 15.0) as u8;
                quantized[i] = q;
            }

            // Pack into bytes
            let mut packed = [0u8; 16];
            for i in 0..16.min(block_data.len() / 2) {
                let low = quantized[i * 2] & 0x0F;
                let high = quantized.get(i * 2 + 1).copied().unwrap_or(0) & 0x0F;
                packed[i] = low | (high << 4);
            }

            QuantizedBlock::Q4_1(Q4_1Block::new(
                f16::from_f32(scale),
                f16::from_f32(min),
                packed,
            ))
        })
        .collect();

    Ok(QuantizedTensor::new(shape, QuantType::Q4_1, blocks))
}

// =============================================================================
// F16 Quantization
// =============================================================================

/// Quantizes data to F16 format (half precision).
fn quantize_f16(data: &[f32], shape: Vec<usize>) -> QuantResult<QuantizedTensor> {
    let f16_data: Vec<f16> = data.par_iter().map(|&x| f16::from_f32(x)).collect();

    let blocks = vec![QuantizedBlock::F16(f16_data)];

    Ok(QuantizedTensor::new(shape, QuantType::F16, blocks))
}

// =============================================================================
// F32 (No Quantization)
// =============================================================================

/// Stores data as F32 (no quantization, for comparison).
fn quantize_f32(data: &[f32], shape: Vec<usize>) -> QuantResult<QuantizedTensor> {
    let blocks = vec![QuantizedBlock::F32(data.to_vec())];
    Ok(QuantizedTensor::new(shape, QuantType::F32, blocks))
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Computes the quantization error (RMSE) between original and quantized.
pub fn compute_quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
    if original.len() != dequantized.len() || original.is_empty() {
        return f32::INFINITY;
    }

    let mse: f32 = original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / original.len() as f32;

    mse.sqrt()
}

/// Returns statistics about quantization error.
pub struct QuantizationStats {
    /// Root mean square error.
    pub rmse: f32,
    /// Maximum absolute error.
    pub max_error: f32,
    /// Mean absolute error.
    pub mean_error: f32,
    /// Compression ratio.
    pub compression_ratio: f32,
}

/// Computes detailed quantization statistics.
pub fn compute_quantization_stats(
    original: &[f32],
    dequantized: &[f32],
    quant_type: QuantType,
) -> QuantizationStats {
    let errors: Vec<f32> = original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    let mse: f32 = errors.iter().map(|e| e.powi(2)).sum::<f32>() / errors.len() as f32;
    let max_error = errors.iter().fold(0.0f32, |a, &b| a.max(b));
    let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;

    QuantizationStats {
        rmse: mse.sqrt(),
        max_error,
        mean_error,
        compression_ratio: quant_type.compression_ratio(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_q8_0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data.clone(), &[8]).unwrap();
        let quantized = quantize_tensor(&tensor, QuantType::Q8_0).unwrap();

        assert_eq!(quantized.quant_type, QuantType::Q8_0);
        assert_eq!(quantized.shape, vec![8]);
        assert_eq!(quantized.num_blocks(), 1);
    }

    #[test]
    fn test_quantize_q4_0() {
        let data: Vec<f32> = (0..64).map(|x| x as f32 / 10.0).collect();
        let tensor = Tensor::from_vec(data.clone(), &[64]).unwrap();
        let quantized = quantize_tensor(&tensor, QuantType::Q4_0).unwrap();

        assert_eq!(quantized.quant_type, QuantType::Q4_0);
        assert_eq!(quantized.num_blocks(), 2);
    }

    #[test]
    fn test_quantize_f16() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data.clone(), &[4]).unwrap();
        let quantized = quantize_tensor(&tensor, QuantType::F16).unwrap();

        assert_eq!(quantized.quant_type, QuantType::F16);
    }

    #[test]
    fn test_compression_ratio() {
        let data: Vec<f32> = (0..256).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, &[256]).unwrap();

        let q8 = quantize_tensor(&tensor, QuantType::Q8_0).unwrap();
        let q4 = quantize_tensor(&tensor, QuantType::Q4_0).unwrap();

        // Q8 should be about 4x compression, Q4 about 8x
        assert!(q8.compression_ratio() > 2.0);
        assert!(q4.compression_ratio() > q8.compression_ratio());
    }

    #[test]
    fn test_quantization_error() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let dequantized = vec![1.1, 2.0, 2.9, 4.1];

        let rmse = compute_quantization_error(&original, &dequantized);
        assert!(rmse > 0.0);
        assert!(rmse < 0.2);
    }
}
