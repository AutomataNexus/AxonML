//! Dequantization Functions
//!
//! Functions for converting quantized tensors back to floating point.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_tensor::Tensor;
use rayon::prelude::*;

use crate::error::{QuantError, QuantResult};
use crate::types::{QuantType, QuantizedTensor, QuantizedBlock, Q8Block, Q4Block, Q4_1Block};

// =============================================================================
// Public API
// =============================================================================

/// Dequantizes a quantized tensor back to f32.
///
/// # Arguments
/// * `quantized` - The quantized tensor to dequantize
///
/// # Returns
/// A tensor with f32 values
///
/// # Example
/// ```ignore
/// use axonml_quant::{quantize_tensor, dequantize_tensor, QuantType};
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
/// let quantized = quantize_tensor(&tensor, QuantType::Q8_0)?;
/// let dequantized = dequantize_tensor(&quantized)?;
/// ```
pub fn dequantize_tensor(quantized: &QuantizedTensor) -> QuantResult<Tensor<f32>> {
    let data = match quantized.quant_type {
        QuantType::Q8_0 => dequantize_q8_0(quantized),
        QuantType::Q4_0 => dequantize_q4_0(quantized),
        QuantType::Q4_1 => dequantize_q4_1(quantized),
        QuantType::Q5_0 | QuantType::Q5_1 => dequantize_q4_0(quantized), // Fallback
        QuantType::F16 => dequantize_f16(quantized),
        QuantType::F32 => dequantize_f32(quantized),
    }?;

    // Truncate to original size
    let expected_size = quantized.numel;
    let data = if data.len() > expected_size {
        data[..expected_size].to_vec()
    } else {
        data
    };

    Tensor::from_vec(data, &quantized.shape)
        .map_err(|e| QuantError::TensorConversion(format!("{:?}", e)))
}

/// Dequantizes a single block.
pub fn dequantize_block(block: &QuantizedBlock) -> Vec<f32> {
    match block {
        QuantizedBlock::Q8(b) => dequantize_q8_block(b),
        QuantizedBlock::Q4(b) => dequantize_q4_block(b),
        QuantizedBlock::Q4_1(b) => dequantize_q4_1_block(b),
        QuantizedBlock::F16(data) => data.iter().map(|x| x.to_f32()).collect(),
        QuantizedBlock::F32(data) => data.clone(),
    }
}

// =============================================================================
// Q8_0 Dequantization
// =============================================================================

/// Dequantizes Q8_0 data.
fn dequantize_q8_0(quantized: &QuantizedTensor) -> QuantResult<Vec<f32>> {
    let result: Vec<f32> = quantized
        .blocks
        .par_iter()
        .flat_map(|block| {
            if let QuantizedBlock::Q8(b) = block {
                dequantize_q8_block(b)
            } else {
                vec![0.0; 32]
            }
        })
        .collect();

    Ok(result)
}

/// Dequantizes a single Q8 block.
fn dequantize_q8_block(block: &Q8Block) -> Vec<f32> {
    let scale = block.scale.to_f32();
    block
        .data
        .iter()
        .map(|&q| q as f32 * scale)
        .collect()
}

// =============================================================================
// Q4_0 Dequantization
// =============================================================================

/// Dequantizes Q4_0 data.
fn dequantize_q4_0(quantized: &QuantizedTensor) -> QuantResult<Vec<f32>> {
    let result: Vec<f32> = quantized
        .blocks
        .par_iter()
        .flat_map(|block| {
            if let QuantizedBlock::Q4(b) = block {
                dequantize_q4_block(b)
            } else {
                vec![0.0; 32]
            }
        })
        .collect();

    Ok(result)
}

/// Dequantizes a single Q4 block.
fn dequantize_q4_block(block: &Q4Block) -> Vec<f32> {
    let scale = block.scale.to_f32();
    let unpacked = block.unpack();

    unpacked
        .iter()
        .map(|&q| q as f32 * scale)
        .collect()
}

// =============================================================================
// Q4_1 Dequantization
// =============================================================================

/// Dequantizes Q4_1 data.
fn dequantize_q4_1(quantized: &QuantizedTensor) -> QuantResult<Vec<f32>> {
    let result: Vec<f32> = quantized
        .blocks
        .par_iter()
        .flat_map(|block| {
            if let QuantizedBlock::Q4_1(b) = block {
                dequantize_q4_1_block(b)
            } else {
                vec![0.0; 32]
            }
        })
        .collect();

    Ok(result)
}

/// Dequantizes a single Q4_1 block.
fn dequantize_q4_1_block(block: &Q4_1Block) -> Vec<f32> {
    let scale = block.scale.to_f32();
    let min = block.min.to_f32();
    let unpacked = block.unpack();

    unpacked
        .iter()
        .map(|&q| q as f32 * scale + min)
        .collect()
}

// =============================================================================
// F16 Dequantization
// =============================================================================

/// Dequantizes F16 data.
fn dequantize_f16(quantized: &QuantizedTensor) -> QuantResult<Vec<f32>> {
    let result: Vec<f32> = quantized
        .blocks
        .iter()
        .flat_map(|block| {
            if let QuantizedBlock::F16(data) = block {
                data.iter().map(|x| x.to_f32()).collect()
            } else {
                vec![]
            }
        })
        .collect();

    Ok(result)
}

// =============================================================================
// F32 Dequantization (passthrough)
// =============================================================================

/// Dequantizes F32 data (passthrough).
fn dequantize_f32(quantized: &QuantizedTensor) -> QuantResult<Vec<f32>> {
    let result: Vec<f32> = quantized
        .blocks
        .iter()
        .flat_map(|block| {
            if let QuantizedBlock::F32(data) = block {
                data.clone()
            } else {
                vec![]
            }
        })
        .collect();

    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::quantize_tensor;

    #[test]
    fn test_roundtrip_q8_0() {
        let original: Vec<f32> = (0..64).map(|x| x as f32 / 10.0).collect();
        let tensor = Tensor::from_vec(original.clone(), &[64]).unwrap();

        let quantized = quantize_tensor(&tensor, QuantType::Q8_0).unwrap();
        let dequantized = dequantize_tensor(&quantized).unwrap();

        // Check shape preserved
        assert_eq!(dequantized.shape(), &[64]);

        // Check values are close (some error expected)
        let deq_data = dequantized.to_vec();
        for (orig, deq) in original.iter().zip(deq_data.iter()) {
            assert!((orig - deq).abs() < 0.1, "Q8 error too large: {} vs {}", orig, deq);
        }
    }

    #[test]
    fn test_roundtrip_q4_0() {
        let original: Vec<f32> = (0..64).map(|x| x as f32 / 10.0).collect();
        let tensor = Tensor::from_vec(original.clone(), &[64]).unwrap();

        let quantized = quantize_tensor(&tensor, QuantType::Q4_0).unwrap();
        let dequantized = dequantize_tensor(&quantized).unwrap();

        assert_eq!(dequantized.shape(), &[64]);

        // Q4 has more error but should still be reasonable
        let deq_data = dequantized.to_vec();
        let max_error: f32 = original
            .iter()
            .zip(deq_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(max_error < 2.0, "Q4 max error too large: {}", max_error);
    }

    #[test]
    fn test_roundtrip_f16() {
        let original = vec![1.0f32, 2.5, -3.0, 4.25];
        let tensor = Tensor::from_vec(original.clone(), &[4]).unwrap();

        let quantized = quantize_tensor(&tensor, QuantType::F16).unwrap();
        let dequantized = dequantize_tensor(&quantized).unwrap();

        let deq_data = dequantized.to_vec();
        for (orig, deq) in original.iter().zip(deq_data.iter()) {
            assert!((orig - deq).abs() < 0.01, "F16 error: {} vs {}", orig, deq);
        }
    }

    #[test]
    fn test_roundtrip_f32() {
        let original = vec![1.0f32, 2.5, -3.0, 4.25];
        let tensor = Tensor::from_vec(original.clone(), &[4]).unwrap();

        let quantized = quantize_tensor(&tensor, QuantType::F32).unwrap();
        let dequantized = dequantize_tensor(&quantized).unwrap();

        let deq_data = dequantized.to_vec();
        assert_eq!(original, deq_data);
    }
}
