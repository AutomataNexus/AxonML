//! Weight abstraction supporting both pre-dequantized f32 and lazy quantized storage.
//!
//! A `Weight` holds either:
//! - `F32(Tensor<f32>)` — fully dequantized, pre-transposed, optionally on GPU.
//!   Fast matmul, high memory.
//! - `Quantized { data, shape, dtype }` — raw GGUF bytes on CPU. Dequantized
//!   on every matmul call (to scratch memory, then dropped). Slower but
//!   keeps a 27B model to ~10GB instead of ~50GB.
//!
//! The transpose is applied implicitly: quantized data is stored as the
//! physical GGUF layout (rows=out, cols=in) and dequantized into a
//! [out, in] scratch buffer, which is then used as an f32 tensor that
//! gets transposed to [in, out] for the matmul (same convention as the
//! pre-dequantized path).

use axonml_core::Device;
use axonml_tensor::Tensor;
use rayon::prelude::*;

use super::gguf::{self, GgmlType};

/// A weight matrix: either pre-dequantized (fast, big) or lazily dequantized (slow, small).
pub enum Weight {
    /// Pre-dequantized f32 tensor, pre-transposed to `[in, out]` layout.
    F32(Tensor<f32>),

    /// Quantized bytes stored on CPU. Dequantized to scratch f32 per matmul.
    /// `dims` are GGUF's `[n_cols, n_rows]` (dims[0]=in, dims[1]=out).
    Quantized {
        data: Vec<u8>,
        dims: Vec<usize>,
        dtype: GgmlType,
    },
}

impl Weight {
    /// Construct from a dequantized f32 tensor (pre-transposed).
    pub fn from_f32(tensor: Tensor<f32>) -> Self {
        Weight::F32(tensor)
    }

    /// Construct by copying GGUF-quantized bytes.
    /// `dims[0]` = in_features, `dims[1]` = out_features.
    pub fn from_quantized(data: Vec<u8>, dims: Vec<usize>, dtype: GgmlType) -> Self {
        Weight::Quantized { data, dims, dtype }
    }

    /// Logical shape of the weight as `[in, out]` (post-transpose convention).
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Weight::F32(t) => t.shape().to_vec(),
            Weight::Quantized { dims, .. } => vec![dims[0], dims[1]],
        }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        match self {
            Weight::F32(t) => t.numel(),
            Weight::Quantized { dims, .. } => dims.iter().product(),
        }
    }

    /// Compressed bytes used in RAM.
    pub fn bytes(&self) -> usize {
        match self {
            Weight::F32(t) => t.numel() * 4,
            Weight::Quantized { data, .. } => data.len(),
        }
    }

    /// Move to device. Only affects F32 variant — quantized data stays on CPU
    /// (dequantization produces CPU scratch which is moved per-matmul).
    pub fn to_device(&mut self, device: Device) {
        if let Weight::F32(t) = self {
            if let Ok(moved) = t.to_device(device) {
                *t = moved;
            }
        }
    }

    /// Matmul: input `[m, in]` @ self `[in, out]` → output `[m, out]`.
    ///
    /// For F32 variant: direct tensor matmul (GPU path if tensor is on GPU).
    /// For Quantized variant: dequantize bytes into scratch [out, in] tensor,
    /// transpose to [in, out], and matmul.
    pub fn matmul(&self, input: &Tensor<f32>) -> Tensor<f32> {
        match self {
            Weight::F32(t) => input.matmul(t).expect("matmul failed"),
            Weight::Quantized { data, dims, dtype } => {
                // Dequantize to [out, in] (physical GGUF layout)
                let n_elem = dims[0] * dims[1];
                let mut buf = vec![0.0f32; n_elem];
                dequantize_into(&mut buf, data, n_elem, *dtype);

                // Build tensor in [out, in] then transpose to [in, out]
                let raw = Tensor::from_vec(buf, &[dims[1], dims[0]])
                    .expect("failed to build dequant tensor");
                let weight_t = raw.transpose(0, 1).expect("transpose failed");

                // Move to input's device (GPU) if input is on GPU
                let weight_t = if input.device() != weight_t.device() {
                    weight_t.to_device(input.device()).unwrap_or(weight_t)
                } else {
                    weight_t
                };

                input.matmul(&weight_t).expect("matmul failed")
            }
        }
    }
}

/// Dequantize `n_elements` values from `raw_data` using the given `dtype`.
/// Block-based dequantization (Q4_0, Q4_K, Q6_K, Q8_0) is parallelized via rayon.
fn dequantize_into(output: &mut [f32], raw_data: &[u8], n_elements: usize, dtype: GgmlType) {
    match dtype {
        GgmlType::F32 => {
            for (i, chunk) in raw_data.chunks_exact(4).enumerate().take(n_elements) {
                output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GgmlType::F16 => gguf::dequantize_f16(raw_data, output),
        GgmlType::BF16 => {
            for (i, chunk) in raw_data.chunks_exact(2).enumerate().take(n_elements) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                output[i] = f32::from_bits((bits as u32) << 16);
            }
        }
        GgmlType::Q8_0 => dequantize_blocks_par(output, raw_data, n_elements, 32, 34, gguf::dequantize_q8_0),
        GgmlType::Q4_0 => dequantize_blocks_par(output, raw_data, n_elements, 32, 18, gguf::dequantize_q4_0),
        GgmlType::Q4K => dequantize_blocks_par(output, raw_data, n_elements, 256, 144, gguf::dequantize_q4_k),
        GgmlType::Q6K => dequantize_blocks_par(output, raw_data, n_elements, 256, 210, gguf::dequantize_q6_k),
        other => {
            eprintln!("Unsupported quant type for lazy dequant: {:?}", other);
        }
    }
}

/// Dequantize a block-based quantization in parallel using rayon.
/// `block_size` = elements per block. `type_size` = bytes per block.
/// `dequant_block(&[u8] bytes, &mut [f32] out)` dequantizes one block.
fn dequantize_blocks_par(
    output: &mut [f32],
    raw_data: &[u8],
    n_elements: usize,
    block_size: usize,
    type_size: usize,
    dequant_block: fn(&[u8], &mut [f32]),
) {
    let n_blocks = n_elements / block_size;
    // Split output and input by block, process in parallel.
    output
        .par_chunks_mut(block_size)
        .take(n_blocks)
        .zip(raw_data.par_chunks(type_size).take(n_blocks))
        .for_each(|(out_block, in_block)| {
            if in_block.len() >= type_size {
                dequant_block(in_block, out_block);
            }
        });
}
