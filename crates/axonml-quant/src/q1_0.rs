//! PrismML `Q1_0` 1-bit Quantization — Dequant + Reference Matmul
//!
//! Implements Prism ML's `Q1_0` quant type (GGUF dtype 41), used by the
//! `prism-ml/Bonsai-*-gguf` family (Bonsai-8B is QAT'd from Qwen3-8B). Format
//! verified against the PrismML `llama.cpp` fork's `vec_dot_q1_0_q8_1` and
//! `quantize_row_q1_0_ref`.
//!
//! # Format
//!
//! - **Block size: 128 weights** (matches PrismML `QK1_0` and the BitNet
//!   I2_S choice in `crate::bitnet`).
//! - **Block stride: 18 bytes** — fp16 scale `d` (2 B) + 16 sign bytes
//!   (`qs`).
//! - **Encoding per bit:** linear, 1 bit per weight. Element `j` lives at
//!   `qs[j / 8]` bit `j % 8`. `bit = 1 → +d`, `bit = 0 → −d`. There is no
//!   zero state — pure binary `{−d, +d}`.
//! - **Per-block scale**, NOT tensor-wide: every 128 weights carry their own
//!   `d`. Effective bits per weight: `1 + 16/128 = 1.125`.
//!
//! # Why this is fast
//!
//! A Q1_0 × Q8_1 dot product expands each sign bit to ±1 (int8) and uses
//! CUDA `dp4a` (4× int8 MAC in one PTX instruction). The reference fork
//! does this in `vec_dot_q1_0_q8_1`. There is no branch and no zero check
//! (unlike I2_S ternary), so the kernel is structurally simpler than the
//! BitNet path while running at the same memory-bandwidth ceiling.
//!
//! # References
//! - Bonsai-8B model card: <https://huggingface.co/prism-ml/Bonsai-8B-gguf>
//! - Whitepaper: <https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf>
//! - Reference fork: <https://github.com/PrismML-Eng/llama.cpp> (branch `prism`)
//!
//! # File
//! `crates/axonml-quant/src/q1_0.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use half::f16;
use rayon::prelude::*;

// =============================================================================
// Constants
// =============================================================================

/// Weights per Q1_0 block (PrismML `QK1_0`).
pub const Q1_0_BLOCK_SIZE: usize = 128;

/// Bytes per Q1_0 block: fp16 scale (2) + 128 sign bits packed into 16 bytes.
pub const Q1_0_BYTES_PER_BLOCK: usize = 18;

/// Number of sign bytes per block (the `qs` array length).
pub const Q1_0_QS_BYTES: usize = 16;

// =============================================================================
// Block pack / unpack
// =============================================================================

/// A single Q1_0 block: fp16 scale + 128 packed sign bits.
///
/// `qs[j / 8]` bit `j % 8` is the sign of weight `j`: 1 → `+d`, 0 → `−d`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Q1_0Block {
    /// Per-block scale (fp16 on disk).
    pub d: f16,
    /// 128 sign bits packed little-endian within each byte.
    pub qs: [u8; Q1_0_QS_BYTES],
}

impl Q1_0Block {
    /// Pack 128 weights into a Q1_0 block. The scale is set to the block's
    /// mean absolute weight; signs are taken directly. Matches the
    /// reference `quantize_row_q1_0_ref`.
    pub fn pack(values: &[f32; Q1_0_BLOCK_SIZE]) -> Self {
        let mut sum_abs = 0.0_f32;
        for &v in values.iter() {
            sum_abs += v.abs();
        }
        let d = f16::from_f32(sum_abs / (Q1_0_BLOCK_SIZE as f32));

        let mut qs = [0u8; Q1_0_QS_BYTES];
        for (j, &v) in values.iter().enumerate() {
            if v >= 0.0 {
                qs[j / 8] |= 1u8 << (j % 8);
            }
        }
        Self { d, qs }
    }

    /// 18-byte raw view (fp16 scale little-endian + 16 sign bytes).
    pub fn to_bytes(&self) -> [u8; Q1_0_BYTES_PER_BLOCK] {
        let mut out = [0u8; Q1_0_BYTES_PER_BLOCK];
        let scale_bits = self.d.to_bits().to_le_bytes();
        out[0] = scale_bits[0];
        out[1] = scale_bits[1];
        out[2..].copy_from_slice(&self.qs);
        out
    }

    /// Parse a block from an 18-byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Q1_0_BYTES_PER_BLOCK {
            return None;
        }
        let d = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        let mut qs = [0u8; Q1_0_QS_BYTES];
        qs.copy_from_slice(&bytes[2..Q1_0_BYTES_PER_BLOCK]);
        Some(Self { d, qs })
    }

    /// Unpack the 128 weights back to f32.
    pub fn unpack(&self) -> [f32; Q1_0_BLOCK_SIZE] {
        let d = self.d.to_f32();
        let neg_d = -d;
        let mut out = [0.0_f32; Q1_0_BLOCK_SIZE];
        for (j, slot) in out.iter_mut().enumerate() {
            let bit = (self.qs[j / 8] >> (j % 8)) & 1;
            *slot = if bit == 1 { d } else { neg_d };
        }
        out
    }
}

// =============================================================================
// Dequantization
// =============================================================================

/// Dequantize a single Q1_0 block to 128 `f32` values.
///
/// # Panics
/// Debug-only: panics if `bytes.len() < 18` or `out.len() < 128`.
pub fn dequantize_q1_0_block(bytes: &[u8], out: &mut [f32]) {
    debug_assert!(bytes.len() >= Q1_0_BYTES_PER_BLOCK);
    debug_assert!(out.len() >= Q1_0_BLOCK_SIZE);
    let d = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
    let neg_d = -d;
    for (j, slot) in out.iter_mut().enumerate().take(Q1_0_BLOCK_SIZE) {
        let bit = (bytes[2 + j / 8] >> (j % 8)) & 1;
        *slot = if bit == 1 { d } else { neg_d };
    }
}

/// Dequantize a full Q1_0 weight buffer to f32.
///
/// `out.len()` must be a multiple of [`Q1_0_BLOCK_SIZE`]. Rayon-parallelized
/// over blocks.
pub fn dequantize_q1_0(bytes: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / Q1_0_BLOCK_SIZE;
    out.par_chunks_mut(Q1_0_BLOCK_SIZE)
        .take(n_blocks)
        .zip(bytes.par_chunks(Q1_0_BYTES_PER_BLOCK).take(n_blocks))
        .for_each(|(out_block, in_block)| {
            dequantize_q1_0_block(in_block, out_block);
        });
}

// =============================================================================
// Reference Matmul (CPU)
// =============================================================================

/// Reference matmul: `out = acts [m, k] @ weights^T [n, k]`.
///
/// `weight_bytes` is `n * (k / 128) * 18` bytes laid out row-major (one
/// row = one output feature). Scale is per-block, embedded in the weight
/// bytes (no tensor-wide scale at the tail, unlike I2_S).
///
/// Single-threaded reference; the GPU kernels handle the production path.
///
/// # Panics
/// Panics if `k` is not a multiple of [`Q1_0_BLOCK_SIZE`] or the input
/// shapes don't agree.
pub fn matmul_q1_0(
    acts: &[f32], // [m, k]
    weight_bytes: &[u8],
    out: &mut [f32], // [m, n]
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(
        k % Q1_0_BLOCK_SIZE == 0,
        "k must be a multiple of {Q1_0_BLOCK_SIZE}"
    );
    let n_blocks = k / Q1_0_BLOCK_SIZE;
    let row_bytes = n_blocks * Q1_0_BYTES_PER_BLOCK;
    assert_eq!(
        weight_bytes.len(),
        n * row_bytes,
        "weight_bytes shape mismatch"
    );
    assert_eq!(acts.len(), m * k, "acts shape mismatch");
    assert_eq!(out.len(), m * n, "out shape mismatch");

    let mut row_buf = vec![0.0_f32; k];
    for row in 0..n {
        let row_start = row * row_bytes;
        // Dequantize this output-feature row once; reuse across all m.
        for b in 0..n_blocks {
            let block = &weight_bytes
                [row_start + b * Q1_0_BYTES_PER_BLOCK..row_start + (b + 1) * Q1_0_BYTES_PER_BLOCK];
            dequantize_q1_0_block(
                block,
                &mut row_buf[b * Q1_0_BLOCK_SIZE..(b + 1) * Q1_0_BLOCK_SIZE],
            );
        }
        for mi in 0..m {
            let mut acc = 0.0_f32;
            let acts_row = &acts[mi * k..(mi + 1) * k];
            for j in 0..k {
                acc += acts_row[j] * row_buf[j];
            }
            out[mi * n + row] = acc;
        }
    }
}

// =============================================================================
// Size helper
// =============================================================================

/// Bytes needed to store `n_elements` Q1_0 weights.
///
/// `n_elements` must be a multiple of [`Q1_0_BLOCK_SIZE`].
pub fn bytes_for_elements(n_elements: usize) -> usize {
    assert!(n_elements % Q1_0_BLOCK_SIZE == 0);
    (n_elements / Q1_0_BLOCK_SIZE) * Q1_0_BYTES_PER_BLOCK
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_roundtrip() {
        let mut values = [0.0_f32; Q1_0_BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = if i % 3 == 0 { -1.5 } else { 0.7 };
        }
        let block = Q1_0Block::pack(&values);
        let bytes = block.to_bytes();
        let parsed = Q1_0Block::from_bytes(&bytes).expect("from_bytes");
        assert_eq!(parsed, block);

        let unpacked = parsed.unpack();
        // Q1_0 has no zero state — every weight should be ±d, sign-matching.
        let d = block.d.to_f32();
        for (orig, &got) in values.iter().zip(unpacked.iter()) {
            if *orig >= 0.0 {
                assert!((got - d).abs() < 1e-4, "expected +d={d} got {got}");
            } else {
                assert!((got + d).abs() < 1e-4, "expected -d={} got {got}", -d);
            }
        }
    }

    #[test]
    fn dequant_matches_block_unpack() {
        let mut values = [0.0_f32; Q1_0_BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as f32) - 64.0) * 0.1;
        }
        let block = Q1_0Block::pack(&values);
        let bytes = block.to_bytes();

        let mut out = [0.0_f32; Q1_0_BLOCK_SIZE];
        dequantize_q1_0_block(&bytes, &mut out);

        let expected = block.unpack();
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch {a} vs {b}");
        }
    }

    #[test]
    fn dequant_par_matches_serial() {
        // 4 blocks worth of weights.
        let n = 4 * Q1_0_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(4);
        let mut all_bytes = Vec::with_capacity(4 * Q1_0_BYTES_PER_BLOCK);
        for b in 0..4 {
            let mut values = [0.0_f32; Q1_0_BLOCK_SIZE];
            for (i, v) in values.iter_mut().enumerate() {
                *v = (((b * Q1_0_BLOCK_SIZE + i) as i32) % 7 - 3) as f32 * 0.5;
            }
            let blk = Q1_0Block::pack(&values);
            all_bytes.extend_from_slice(&blk.to_bytes());
            blocks.push(blk);
        }

        let mut par_out = vec![0.0_f32; n];
        dequantize_q1_0(&all_bytes, &mut par_out);

        let mut serial_out = vec![0.0_f32; n];
        for (i, blk) in blocks.iter().enumerate() {
            let unpacked = blk.unpack();
            serial_out[i * Q1_0_BLOCK_SIZE..(i + 1) * Q1_0_BLOCK_SIZE].copy_from_slice(&unpacked);
        }
        assert_eq!(par_out, serial_out);
    }

    #[test]
    fn matmul_matches_dequant_then_dot() {
        // Small reference: m=2, n=3, k=128 (one block per row).
        let m = 2;
        let n = 3;
        let k = Q1_0_BLOCK_SIZE;

        let mut acts = vec![0.0_f32; m * k];
        for (i, v) in acts.iter_mut().enumerate() {
            *v = (i as f32 * 0.013).sin();
        }

        let mut weight_bytes = Vec::with_capacity(n * Q1_0_BYTES_PER_BLOCK);
        let mut weight_f32 = vec![0.0_f32; n * k];
        for row in 0..n {
            let mut values = [0.0_f32; Q1_0_BLOCK_SIZE];
            for (i, v) in values.iter_mut().enumerate() {
                *v = (((row * 13 + i * 7) % 17) as f32 - 8.0) * 0.1;
            }
            let blk = Q1_0Block::pack(&values);
            weight_bytes.extend_from_slice(&blk.to_bytes());
            let unpacked = blk.unpack();
            weight_f32[row * k..(row + 1) * k].copy_from_slice(&unpacked);
        }

        let mut out = vec![0.0_f32; m * n];
        matmul_q1_0(&acts, &weight_bytes, &mut out, m, n, k);

        // Reference: dequant + naive matmul.
        let mut ref_out = vec![0.0_f32; m * n];
        for mi in 0..m {
            for row in 0..n {
                let mut acc = 0.0_f32;
                for j in 0..k {
                    acc += acts[mi * k + j] * weight_f32[row * k + j];
                }
                ref_out[mi * n + row] = acc;
            }
        }
        for (a, b) in out.iter().zip(ref_out.iter()) {
            assert!((a - b).abs() < 1e-3, "mismatch {a} vs {b}");
        }
    }

    #[test]
    fn bytes_for_elements_arithmetic() {
        assert_eq!(bytes_for_elements(128), 18);
        assert_eq!(bytes_for_elements(2 * 128), 36);
        assert_eq!(bytes_for_elements(8 * 128), 8 * 18);
    }

    #[test]
    #[should_panic]
    fn matmul_rejects_misaligned_k() {
        let acts = vec![0.0_f32; 100];
        let weight_bytes = vec![0u8; 100];
        let mut out = vec![0.0_f32; 1];
        matmul_q1_0(&acts, &weight_bytes, &mut out, 1, 1, 100); // k=100 not multiple of 128
    }
}
