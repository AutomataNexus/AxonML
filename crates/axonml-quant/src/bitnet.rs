//! BitNet b1.58 — ternary weight dequant + fused add-only matmul.
//!
//! Implements Microsoft's `I2_S` quant type (GGUF dtype 36), used by the
//! `bitnet.cpp` reference kernels and by every official BitNet b1.58 GGUF
//! release (including `microsoft/bitnet-b1.58-2B-4T-gguf`).
//!
//! # Format (verified against `microsoft/BitNet` reference, 2026-04-14)
//!
//! - **Block size: 128 weights** (`QK_I2_S = 128` on x86_64).
//! - **Block stride: 32 bytes** (128 × 2 bits packed).
//! - **Encoding per 2-bit code:** `0 → -1`, `1 → 0`, `2 → +1`, `3 → unused`.
//!   (From `quantize_i2_s` in `ggml-bitnet-mad.cpp`:
//!   `"q8 -> 0, 1, 2 | | | -1, 0, 1"`.)
//! - **Intra-block layout is NOT `4 consecutive weights / byte`** — it's a
//!   SIMD-friendly group-strided layout. Each 32-byte block stores 128
//!   weights as 4 groups of 32, multiplexed into the 2-bit positions of
//!   the 32 bytes:
//!
//!     - byte `k` bits **6..7** → weight `k`       (group 0, shift 6)
//!     - byte `k` bits **4..5** → weight `k + 32`  (group 1, shift 4)
//!     - byte `k` bits **2..3** → weight `k + 64`  (group 2, shift 2)
//!     - byte `k` bits **0..1** → weight `k + 96`  (group 3, shift 0)
//!
//!   (Encoder in the reference: `temp = q8 << (6 - 2*group_idx)`; decoder
//!   shifts the byte right by `6 - 2*group_idx` and ANDs with `0x03`.) The
//!   layout lets AVX2 load 32 bytes once, shift by 0/2/4/6 with a mask of
//!   `0x03`, and extract 128 trit codes in four 32-wide vector registers.
//!
//! - **One tensor-wide f32 scale** follows the packed data at offset
//!   `m * k / 4` bytes, padded to the next 32-byte boundary. The
//!   dequantized weight is `scale × trit[i]`. This module requires the
//!   caller to pass the scale separately — [`matmul_i2s`] and
//!   [`dequantize_i2s`] both accept `scale: f32`.
//!
//! # Why this is fast on CPU
//!
//! Matmul between an f32 activation matrix and ternary weights becomes
//! branchless accumulate-and-subtract: for each weight, either add the
//! activation (+1), subtract it (-1), or skip (0). The tensor-wide scale
//! applies once at the end of each output element. That's BitNet's
//! "add-only matmul" performance story. A SIMD fast path mirroring the
//! reference AVX2 kernel is a natural follow-up once the scalar path is
//! verified against Microsoft's released weights.
//!
//! # References
//! - Microsoft BitNet paper ("The Era of 1-bit LLMs"): <https://arxiv.org/abs/2402.17764>
//! - `microsoft/BitNet` on GitHub (bitnet.cpp reference kernels)

use rayon::prelude::*;

// =============================================================================
// Constants
// =============================================================================

/// Weights per I2_S block (Microsoft `QK_I2_S` on x86_64).
pub const I2S_BLOCK_SIZE: usize = 128;

/// Bytes per I2_S block (128 × 2 bits).
pub const I2S_BYTES_PER_BLOCK: usize = 32;

/// Group size within a block — 4 groups of 32 weights share the same 32
/// bytes but live at different bit-positions.
pub const I2S_GROUP_SIZE: usize = 32;

// =============================================================================
// Trit <-> 2-bit encoding (Microsoft bitnet.cpp convention)
// =============================================================================

/// Decode a 2-bit code to a trit in `{-1, 0, +1}`.
///
/// Encoding (Microsoft bitnet.cpp): `0 → -1`, `1 → 0`, `2 → +1`, `3 → 0`
/// (defensive; code 3 is unused by the reference encoder).
#[inline(always)]
pub const fn decode_trit(code: u8) -> i8 {
    match code & 0b11 {
        0 => -1,
        1 => 0,
        2 => 1,
        _ => 0,
    }
}

/// Encode a trit to a 2-bit code (inverse of [`decode_trit`]).
#[inline(always)]
const fn encode_trit(v: i8) -> u8 {
    if v > 0 {
        2
    } else if v < 0 {
        0
    } else {
        1
    }
}

// =============================================================================
// Block pack / unpack (primarily for tests; production loads raw bytes)
// =============================================================================

/// A single I2_S block: 128 ternary weights in the group-strided layout.
///
/// The tensor-wide scale is not stored on the block — see module docs.
#[derive(Debug, Clone)]
pub struct I2sBlock {
    /// 32 bytes holding 128 × 2-bit trits in group-strided form.
    pub data: [u8; I2S_BYTES_PER_BLOCK],
}

impl I2sBlock {
    /// Pack 128 trit values `{-1, 0, +1}` using the Microsoft layout.
    pub fn pack(values: &[i8; I2S_BLOCK_SIZE]) -> Self {
        let mut data = [0u8; I2S_BYTES_PER_BLOCK];
        // For each (group_idx, group_pos), OR the code at bits [6-2*g .. 7-2*g].
        for group_idx in 0..4 {
            let shift = 6 - 2 * group_idx;
            for group_pos in 0..I2S_GROUP_SIZE {
                let code = encode_trit(values[group_idx * I2S_GROUP_SIZE + group_pos]);
                data[group_pos] |= code << shift;
            }
        }
        Self { data }
    }

    /// 32-byte raw view.
    pub fn to_bytes(&self) -> [u8; I2S_BYTES_PER_BLOCK] {
        self.data
    }

    /// Parse a block from a 32-byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < I2S_BYTES_PER_BLOCK {
            return None;
        }
        let mut data = [0u8; I2S_BYTES_PER_BLOCK];
        data.copy_from_slice(&bytes[..I2S_BYTES_PER_BLOCK]);
        Some(Self { data })
    }

    /// Unpack all 128 trits in linear order `0..128`.
    pub fn unpack(&self) -> [i8; I2S_BLOCK_SIZE] {
        let mut out = [0i8; I2S_BLOCK_SIZE];
        for group_idx in 0..4 {
            let shift = 6 - 2 * group_idx;
            for group_pos in 0..I2S_GROUP_SIZE {
                let code = (self.data[group_pos] >> shift) & 0b11;
                out[group_idx * I2S_GROUP_SIZE + group_pos] = decode_trit(code);
            }
        }
        out
    }
}

// =============================================================================
// Dequantization
// =============================================================================

/// Dequantize a single I2_S block to 128 `f32` values.
///
/// Applies the tensor-wide `scale` so `out[i] = scale × trit[i]`.
///
/// # Panics
/// Debug-only: panics if `bytes.len() < 32` or `out.len() < 128`.
pub fn dequantize_i2s_block(bytes: &[u8], scale: f32, out: &mut [f32]) {
    debug_assert!(bytes.len() >= I2S_BYTES_PER_BLOCK);
    debug_assert!(out.len() >= I2S_BLOCK_SIZE);
    let b = &bytes[..I2S_BYTES_PER_BLOCK];
    for (group_pos, &byte) in b.iter().enumerate() {
        let c0 = decode_trit((byte >> 6) & 0b11) as f32; // group 0
        let c1 = decode_trit((byte >> 4) & 0b11) as f32; // group 1
        let c2 = decode_trit((byte >> 2) & 0b11) as f32; // group 2
        let c3 = decode_trit(byte & 0b11) as f32; // group 3
        out[group_pos] = c0 * scale;
        out[group_pos + I2S_GROUP_SIZE] = c1 * scale;
        out[group_pos + 2 * I2S_GROUP_SIZE] = c2 * scale;
        out[group_pos + 3 * I2S_GROUP_SIZE] = c3 * scale;
    }
}

/// Dequantize a full I2_S weight buffer to f32.
///
/// `out.len()` must be a multiple of [`I2S_BLOCK_SIZE`]. Rayon-parallelized
/// over blocks.
pub fn dequantize_i2s(bytes: &[u8], scale: f32, out: &mut [f32]) {
    let n_blocks = out.len() / I2S_BLOCK_SIZE;
    out.par_chunks_mut(I2S_BLOCK_SIZE)
        .take(n_blocks)
        .zip(bytes.par_chunks(I2S_BYTES_PER_BLOCK).take(n_blocks))
        .for_each(|(out_block, in_block)| {
            if in_block.len() >= I2S_BYTES_PER_BLOCK {
                dequantize_i2s_block(in_block, scale, out_block);
            }
        });
}

// =============================================================================
// Fused ternary matmul
// =============================================================================

/// Fused add-only ternary matmul: `output = scale × (activations @ weights^T)`.
///
/// # Shapes
/// - `activations`: `[m, k]` row-major f32
/// - `weight_bytes`: `[n, k]` — each output row is a contiguous run of
///   `k / 128` I2_S blocks (32 bytes each). `k` **must** be a multiple
///   of [`I2S_BLOCK_SIZE`] (128).
/// - `scale`: tensor-wide f32 scale read from the tail of the GGUF tensor
///   (see module docs)
/// - `output`: `[m, n]` row-major f32
///
/// # Panics
/// Panics if `k % 128 != 0`, or if shapes don't line up.
///
/// # Parallelism
/// Parallelizes over output columns `n`. Right grain for decode (`m == 1`)
/// — one independent dot product per column.
pub fn matmul_i2s(
    activations: &[f32],
    m: usize,
    k: usize,
    weight_bytes: &[u8],
    n: usize,
    scale: f32,
    output: &mut [f32],
) {
    assert!(
        k % I2S_BLOCK_SIZE == 0,
        "matmul_i2s: k ({k}) must be a multiple of {I2S_BLOCK_SIZE}",
    );
    assert_eq!(activations.len(), m * k, "activations shape mismatch");
    assert_eq!(output.len(), m * n, "output shape mismatch");
    let blocks_per_row = k / I2S_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * I2S_BYTES_PER_BLOCK;
    assert_eq!(
        weight_bytes.len(),
        n * bytes_per_row,
        "weight_bytes shape mismatch",
    );

    for i in 0..m {
        let act_row = &activations[i * k..(i + 1) * k];
        output[i * n..(i + 1) * n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, out_slot)| {
                let wrow = &weight_bytes[j * bytes_per_row..(j + 1) * bytes_per_row];
                *out_slot = dot_row_ternary(act_row, wrow, blocks_per_row) * scale;
            });
    }
}

/// Inner dot product for one activation row × one ternary weight row, in
/// the group-strided layout. Returns the UNSCALED sum — caller multiplies
/// by the tensor-wide scale.
#[inline(always)]
fn dot_row_ternary(act_row: &[f32], wrow: &[u8], blocks_per_row: usize) -> f32 {
    let mut acc = 0.0f32;
    for block_idx in 0..blocks_per_row {
        let block_off = block_idx * I2S_BYTES_PER_BLOCK;
        let block = &wrow[block_off..block_off + I2S_BYTES_PER_BLOCK];
        let k_base = block_idx * I2S_BLOCK_SIZE;
        // Autovec-friendly: pull the four group activations into locals,
        // decode once per byte, accumulate into four separate accumulators
        // (lets LLVM issue independent FMAs on wide targets).
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        for (group_pos, &byte) in block.iter().enumerate() {
            let t0 = decode_trit((byte >> 6) & 0b11) as f32;
            let t1 = decode_trit((byte >> 4) & 0b11) as f32;
            let t2 = decode_trit((byte >> 2) & 0b11) as f32;
            let t3 = decode_trit(byte & 0b11) as f32;
            let base = k_base + group_pos;
            a0 += act_row[base] * t0;
            a1 += act_row[base + I2S_GROUP_SIZE] * t1;
            a2 += act_row[base + 2 * I2S_GROUP_SIZE] * t2;
            a3 += act_row[base + 3 * I2S_GROUP_SIZE] * t3;
        }
        acc += a0 + a1 + a2 + a3;
    }
    acc
}

// =============================================================================
// Int8-activation fused ternary matmul (AVX-VNNI path)
// =============================================================================
//
// This is the "30-50% over llama.cpp" lever. The scalar `matmul_i2s` above
// materializes activations as f32 and walks through trit codes one byte at
// a time — memory-BW-bound and one instruction per trit on CPU.
//
// The fused int8 path does three things differently:
//
// 1. Activations are quantized to int8 with a per-row absmax scale before
//    the matmul. Bandwidth drops 4× (f32 → i8) on the activation side.
// 2. Trit codes stay in their 2-bit packed form and never decode to f32.
//    Bandwidth drops 16× on the weight side (f32 → 2 bits).
// 3. The dot product uses the VNNI `dpbusd` instruction (`_mm256_dpbusd_epi32`
//    on AVX-VNNI, `_mm512_dpbusd_epi32` on AVX-512 VNNI). dpbusd does 32
//    unsigned-byte × signed-byte multiplies and sums groups of 4 into int32
//    lanes, so one instruction handles 32 weight-activation pairs.
//
// Arithmetic trick: the trit codes `{0, 1, 2}` map to `{-1, 0, +1}` via
// `trit = code - 1`. dpbusd computes `sum(code × act)`, not `sum(trit × act)`.
// We recover the true dot product with a single per-row correction:
//
//     true_dot_j = code_dot_j - act_sum
//
// where `act_sum = sum_k(act_i[k])` (a single scalar per input row, computed
// once). The correction is two cheap int32 ops per output element.
//
// Microsoft's bitnet.cpp does this same accounting in its AVX2 kernels — we
// match their approach and then tune for Arrow Lake's AVX-VNNI-but-no-AVX-512
// ISA profile. Reference in `ggml-bitnet-mad.cpp::ggml_vec_dot_i2_i8_s_1x1`.

/// Quantize an f32 row to int8 with a per-row absmax scale such that
/// `f32[i] ≈ int8[i] * scale`. The scale is chosen so the largest-
/// magnitude element maps to ±127.
///
/// Returns the scale. Intended for **activation quantization** at the
/// beginning of an I2_S × int8 matmul — the matmul output multiplies by
/// this scale (along with the weight's tensor-wide scale) to recover f32
/// logits.
///
/// # Edge cases
/// - All-zero input → scale = 0, output is all zeros.
/// - Signals with a single enormous outlier — absmax is sensitive; callers
///   with outlier activations may want to clamp or use percentile-absmax
///   instead. Not a concern for typical post-norm transformer activations.
pub fn quantize_row_to_int8(input: &[f32], output: &mut [i8]) -> f32 {
    debug_assert_eq!(input.len(), output.len());
    let absmax = input.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if absmax == 0.0 {
        for o in output.iter_mut() {
            *o = 0;
        }
        return 0.0;
    }
    let scale = absmax / 127.0;
    let inv_scale = 1.0 / scale;
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        let q = (v * inv_scale).round();
        // Clamp to i8 range defensively — rounding can push ±127.5 → ±128.
        *o = q.clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Fused I2_S × int8 matmul: `out = weight_scale × (acts_int8 @ weights^T) × act_scales_per_row`.
///
/// # Shapes
/// - `acts_int8`: `[m, k]` row-major int8 (quantized via [`quantize_row_to_int8`])
/// - `act_scales`: `[m]` — per-row f32 scales from the int8 quantization
/// - `weight_bytes`: `[n, k]` as I2_S blocks (same layout as [`matmul_i2s`])
/// - `weight_scale`: tensor-wide f32 scale (from GGUF tail)
/// - `output`: `[m, n]` row-major f32
///
/// # Math
/// For each `(i, j)`:
///
/// ```text
/// out[i, j] = act_scales[i] * weight_scale * sum_k(trit[j, k] * act_i8[i, k])
///           = act_scales[i] * weight_scale * (sum_k(code[j, k] * act_i8[i, k]) - act_sum[i])
/// ```
///
/// where `code[j, k] ∈ {0, 1, 2}` is the raw 2-bit code and `act_sum[i] = sum_k(act_i8[i, k])`.
///
/// # Dispatch
/// At runtime we check for AVX-VNNI via `is_x86_feature_detected!("avxvnni")`
/// and take the SIMD path when available; otherwise fall back to a scalar
/// reference that matches the SIMD path bit-for-bit (lets tests run on any
/// host).
pub fn matmul_i2s_i8(
    acts_int8: &[i8],
    act_scales: &[f32],
    m: usize,
    k: usize,
    weight_bytes: &[u8],
    n: usize,
    weight_scale: f32,
    output: &mut [f32],
) {
    assert!(
        k % I2S_BLOCK_SIZE == 0,
        "matmul_i2s_i8: k ({k}) must be a multiple of {I2S_BLOCK_SIZE}",
    );
    assert_eq!(acts_int8.len(), m * k, "acts_int8 shape mismatch");
    assert_eq!(act_scales.len(), m, "act_scales length mismatch");
    assert_eq!(output.len(), m * n, "output shape mismatch");
    let blocks_per_row = k / I2S_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * I2S_BYTES_PER_BLOCK;
    assert_eq!(
        weight_bytes.len(),
        n * bytes_per_row,
        "weight_bytes shape mismatch",
    );

    // Runtime dispatch — AVX-VNNI variant fills in on a follow-up commit;
    // scalar path below is the correctness baseline.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avxvnni") && std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature-detected above.
            unsafe {
                matmul_i2s_i8_avxvnni(
                    acts_int8, act_scales, m, k, weight_bytes, n, weight_scale, output,
                );
            }
            return;
        }
    }

    matmul_i2s_i8_scalar(
        acts_int8, act_scales, m, k, weight_bytes, n, weight_scale, output,
    );
}

/// Scalar reference for [`matmul_i2s_i8`]. Used as a correctness baseline
/// for the AVX-VNNI fast path and as a fallback on non-x86_64 or
/// pre-AVX-VNNI CPUs.
fn matmul_i2s_i8_scalar(
    acts_int8: &[i8],
    act_scales: &[f32],
    m: usize,
    k: usize,
    weight_bytes: &[u8],
    n: usize,
    weight_scale: f32,
    output: &mut [f32],
) {
    let blocks_per_row = k / I2S_BLOCK_SIZE;
    let bytes_per_row = blocks_per_row * I2S_BYTES_PER_BLOCK;

    for i in 0..m {
        let act_row = &acts_int8[i * k..(i + 1) * k];
        let act_scale = act_scales[i];
        let act_sum: i32 = act_row.iter().map(|&x| x as i32).sum();
        let combined_scale = weight_scale * act_scale;

        output[i * n..(i + 1) * n]
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, out_slot)| {
                let wrow = &weight_bytes[j * bytes_per_row..(j + 1) * bytes_per_row];
                // sum_k(code[j,k] * act[k]) with code in {0,1,2}.
                let mut code_dot: i32 = 0;
                for block_idx in 0..blocks_per_row {
                    let block_off = block_idx * I2S_BYTES_PER_BLOCK;
                    let block = &wrow[block_off..block_off + I2S_BYTES_PER_BLOCK];
                    let k_base = block_idx * I2S_BLOCK_SIZE;
                    for (group_pos, &byte) in block.iter().enumerate() {
                        // Bit layout: bits 6-7 → pos k_base+group_pos
                        //             bits 4-5 → pos k_base+32+group_pos
                        //             bits 2-3 → pos k_base+64+group_pos
                        //             bits 0-1 → pos k_base+96+group_pos
                        let c0 = ((byte >> 6) & 0b11) as i32;
                        let c1 = ((byte >> 4) & 0b11) as i32;
                        let c2 = ((byte >> 2) & 0b11) as i32;
                        let c3 = (byte & 0b11) as i32;
                        let base = k_base + group_pos;
                        code_dot += c0 * act_row[base] as i32;
                        code_dot += c1 * act_row[base + I2S_GROUP_SIZE] as i32;
                        code_dot += c2 * act_row[base + 2 * I2S_GROUP_SIZE] as i32;
                        code_dot += c3 * act_row[base + 3 * I2S_GROUP_SIZE] as i32;
                    }
                }
                // trit = code - 1, so sum(trit*act) = sum(code*act) - sum(act).
                let trit_dot = code_dot - act_sum;
                *out_slot = (trit_dot as f32) * combined_scale;
            });
    }
}

/// AVX-VNNI fast path — **unimplemented**. Drop-in replacement for
/// [`matmul_i2s_i8_scalar`] once filled in.
///
/// # Planned inner loop (per block of 128 weights × 128 activations):
///
/// ```ignore
/// // Load 32 weight bytes (128 trits packed) and 128 int8 activations.
/// let bytes_v = _mm256_loadu_si256(block_ptr as *const __m256i);
/// let acts_g0 = _mm256_loadu_si256(act_ptr.add(k_base)         as *const __m256i); // pos k_base..+32
/// let acts_g1 = _mm256_loadu_si256(act_ptr.add(k_base + 32)    as *const __m256i);
/// let acts_g2 = _mm256_loadu_si256(act_ptr.add(k_base + 64)    as *const __m256i);
/// let acts_g3 = _mm256_loadu_si256(act_ptr.add(k_base + 96)    as *const __m256i);
///
/// // Extract 2-bit codes for each of 4 groups.
/// let mask = _mm256_set1_epi8(0x03);
/// let codes_g0 = _mm256_and_si256(_mm256_srli_epi16(bytes_v, 6), mask); // bits 6-7
/// let codes_g1 = _mm256_and_si256(_mm256_srli_epi16(bytes_v, 4), mask); // bits 4-5
/// let codes_g2 = _mm256_and_si256(_mm256_srli_epi16(bytes_v, 2), mask); // bits 2-3
/// let codes_g3 = _mm256_and_si256(bytes_v, mask);                      // bits 0-1
///
/// // 32 × (u8 × i8) → 8 × i32, accumulated.
/// acc = _mm256_dpbusd_epi32(acc, codes_g0, acts_g0);
/// acc = _mm256_dpbusd_epi32(acc, codes_g1, acts_g1);
/// acc = _mm256_dpbusd_epi32(acc, codes_g2, acts_g2);
/// acc = _mm256_dpbusd_epi32(acc, codes_g3, acts_g3);
/// ```
///
/// Four VNNI ops per 128-weight block. The outer loop iterates
/// `blocks_per_row` blocks per output column, then horizontally sums the
/// int32 lanes to a scalar, applies the `- act_sum` correction, and scales
/// by `combined_scale` to f32.
///
/// Rayon fan-out over output columns `n` (same as the scalar path). On
/// Arrow Lake (AVX-VNNI but no AVX-512), expect ~8-12× speedup over
/// scalar on the kernel alone; end-to-end wins compound because activation
/// bandwidth drops 4× and weight bandwidth stays 2-bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn matmul_i2s_i8_avxvnni(
    acts_int8: &[i8],
    act_scales: &[f32],
    m: usize,
    k: usize,
    weight_bytes: &[u8],
    n: usize,
    weight_scale: f32,
    output: &mut [f32],
) {
    // TODO: fill in. For now delegate to the scalar path so the public
    // API works on every machine — this is the scaffolding for the
    // follow-up perf commit.
    matmul_i2s_i8_scalar(
        acts_int8, act_scales, m, k, weight_bytes, n, weight_scale, output,
    );
}

// =============================================================================
// Size helpers
// =============================================================================

/// Bytes needed to store `n_elements` I2_S weights (excluding the 4-byte
/// tensor scale and any alignment padding — those live outside the packed
/// stream).
pub fn bytes_for_elements(n_elements: usize) -> usize {
    (n_elements / I2S_BLOCK_SIZE) * I2S_BYTES_PER_BLOCK
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trits(pattern: &[i8]) -> [i8; I2S_BLOCK_SIZE] {
        let mut out = [0i8; I2S_BLOCK_SIZE];
        for (i, v) in pattern.iter().cycle().take(I2S_BLOCK_SIZE).enumerate() {
            out[i] = *v;
        }
        out
    }

    #[test]
    fn trit_encode_decode_roundtrip() {
        assert_eq!(decode_trit(0b00), -1);
        assert_eq!(decode_trit(0b01), 0);
        assert_eq!(decode_trit(0b10), 1);
        assert_eq!(decode_trit(0b11), 0);
        assert_eq!(encode_trit(-1), 0);
        assert_eq!(encode_trit(0), 1);
        assert_eq!(encode_trit(1), 2);
        // Clamp.
        assert_eq!(encode_trit(42), 2);
        assert_eq!(encode_trit(-42), 0);
    }

    #[test]
    fn block_pack_unpack_roundtrip() {
        let values = make_trits(&[-1, 0, 1, 0, 1, -1, 0, 0]);
        let block = I2sBlock::pack(&values);
        let decoded = block.unpack();
        assert_eq!(&values[..], &decoded[..]);
    }

    #[test]
    fn block_bytes_roundtrip() {
        let values = make_trits(&[1, -1, 0]);
        let block = I2sBlock::pack(&values);
        let bytes = block.to_bytes();
        let parsed = I2sBlock::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.data, block.data);
        assert_eq!(parsed.unpack(), values);
    }

    #[test]
    fn dequantize_single_block() {
        // Alternating +1 / -1 with scale = 2.5 → +2.5 / -2.5.
        let values = make_trits(&[1, -1]);
        let block = I2sBlock::pack(&values);
        let bytes = block.to_bytes();
        let mut out = [0.0f32; I2S_BLOCK_SIZE];
        dequantize_i2s_block(&bytes, 2.5, &mut out);
        for (i, v) in out.iter().enumerate() {
            let expected = if i % 2 == 0 { 2.5 } else { -2.5 };
            assert!(
                (v - expected).abs() < 1e-6,
                "idx {i}: got {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn group_strided_layout_is_correct() {
        // Weight at logical position `group_idx * 32 + group_pos` must be
        // stored in byte `group_pos`'s bit-slice `(6 - 2*group_idx)..(8 - 2*group_idx)`.
        // Build a block where only position 0 is +1, rest are 0, and verify the
        // byte layout matches the Microsoft encoder formula directly.
        let mut values = [0i8; I2S_BLOCK_SIZE];
        values[0] = 1; // group 0, pos 0 → byte 0 bits 6..7 = code 2 = 0b10
        let block = I2sBlock::pack(&values);
        assert_eq!(
            block.data[0] & 0b1100_0000,
            0b1000_0000,
            "expected code=2 (+1) in byte 0 bits 6-7",
        );
        // All other bytes: zero bits 6-7 = code 0 = -1 oh wait — we want zero.
        // Since we packed 0s everywhere else, their code is 1 (0b01). Byte 0
        // bits 4-5 should encode group-1-pos-0 = 0 = code 1:
        assert_eq!(
            (block.data[0] >> 4) & 0b11,
            1,
            "expected code=1 (0) in byte 0 bits 4-5",
        );
    }

    #[test]
    fn dequantize_multi_block_tensor() {
        let n_blocks = 3;
        let n_elem = n_blocks * I2S_BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(n_blocks * I2S_BYTES_PER_BLOCK);
        let patterns: &[&[i8]] = &[&[1, 0, -1], &[-1, 1, 0], &[0, 0, 1, -1]];
        for b in 0..n_blocks {
            let block = I2sBlock::pack(&make_trits(patterns[b]));
            bytes.extend_from_slice(&block.to_bytes());
        }
        let mut out = vec![0.0f32; n_elem];
        dequantize_i2s(&bytes, 1.0, &mut out);

        // Spot-check first trit of each block.
        assert_eq!(out[0], 1.0);
        assert_eq!(out[I2S_BLOCK_SIZE], -1.0);
        assert_eq!(out[2 * I2S_BLOCK_SIZE], 0.0);
    }

    fn reference_matmul(
        activations: &[f32],
        m: usize,
        k: usize,
        weight_bytes: &[u8],
        n: usize,
        scale: f32,
        output: &mut [f32],
    ) {
        let mut w = vec![0.0f32; n * k];
        dequantize_i2s(weight_bytes, scale, &mut w);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += activations[i * k + kk] * w[j * k + kk];
                }
                output[i * n + j] = s;
            }
        }
    }

    #[test]
    fn matmul_matches_reference_small() {
        let m = 2;
        let k = I2S_BLOCK_SIZE;
        let n = 4;
        let scale = 0.125f32;

        let mut weight_bytes = Vec::new();
        let patterns: &[&[i8]] = &[&[1, 0, -1], &[-1, 1, 0], &[0, -1, 1], &[1, 1, -1, -1]];
        for j in 0..n {
            let vals = make_trits(patterns[j]);
            let block = I2sBlock::pack(&vals);
            weight_bytes.extend_from_slice(&block.to_bytes());
        }

        let mut activations = vec![0.0f32; m * k];
        for i in 0..m {
            for kk in 0..k {
                activations[i * k + kk] = (i as f32 + 1.0) * (kk as f32 / k as f32);
            }
        }

        let mut fused_out = vec![0.0f32; m * n];
        let mut ref_out = vec![0.0f32; m * n];
        matmul_i2s(&activations, m, k, &weight_bytes, n, scale, &mut fused_out);
        reference_matmul(&activations, m, k, &weight_bytes, n, scale, &mut ref_out);

        for (i, (f, r)) in fused_out.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-5,
                "mismatch at {i}: fused={f}, ref={r}",
            );
        }
    }

    #[test]
    fn matmul_matches_reference_multi_block() {
        let m = 3;
        let k = 3 * I2S_BLOCK_SIZE;
        let n = 5;
        let scale = 0.25f32;

        let mut weight_bytes = Vec::new();
        for j in 0..n {
            for b in 0..(k / I2S_BLOCK_SIZE) {
                let pattern = if (j + b) % 2 == 0 {
                    &[1, 0, -1, 1, -1][..]
                } else {
                    &[-1, -1, 1, 0, 1][..]
                };
                let block = I2sBlock::pack(&make_trits(pattern));
                weight_bytes.extend_from_slice(&block.to_bytes());
            }
        }

        let mut activations = vec![0.0f32; m * k];
        for i in 0..m {
            for kk in 0..k {
                activations[i * k + kk] = ((i + 1) as f32) * ((kk as f32).sin());
            }
        }

        let mut fused_out = vec![0.0f32; m * n];
        let mut ref_out = vec![0.0f32; m * n];
        matmul_i2s(&activations, m, k, &weight_bytes, n, scale, &mut fused_out);
        reference_matmul(&activations, m, k, &weight_bytes, n, scale, &mut ref_out);

        for (i, (f, r)) in fused_out.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-4,
                "mismatch at {i}: fused={f}, ref={r}",
            );
        }
    }

    #[test]
    fn bytes_for_elements_calculation() {
        assert_eq!(bytes_for_elements(128), 32);
        assert_eq!(bytes_for_elements(256), 64);
        assert_eq!(bytes_for_elements(1024), 256);
        assert_eq!(bytes_for_elements(0), 0);
    }

    #[test]
    fn int8_matmul_matches_f32_within_quant_error() {
        // Quantize activations + run both paths. They should agree within
        // the per-row absmax/127 quantization noise.
        let m = 2;
        let k = 2 * I2S_BLOCK_SIZE;
        let n = 6;
        let weight_scale = 0.1f32;

        let mut weight_bytes = Vec::new();
        for j in 0..n {
            for b in 0..(k / I2S_BLOCK_SIZE) {
                let pattern: &[i8] = if (j + b) % 2 == 0 {
                    &[1, 0, -1, 1]
                } else {
                    &[-1, 1, 0, -1]
                };
                let block = I2sBlock::pack(&make_trits(pattern));
                weight_bytes.extend_from_slice(&block.to_bytes());
            }
        }

        // f32 activations, deterministic-ish.
        let mut activations = vec![0.0f32; m * k];
        for i in 0..m {
            for kk in 0..k {
                activations[i * k + kk] = ((kk as f32) * 0.13 - 2.0).sin() * (1.0 + i as f32 * 0.1);
            }
        }

        let mut ref_out = vec![0.0f32; m * n];
        matmul_i2s(&activations, m, k, &weight_bytes, n, weight_scale, &mut ref_out);

        // Quantize activations per row.
        let mut acts_i8 = vec![0i8; m * k];
        let mut act_scales = vec![0.0f32; m];
        for i in 0..m {
            act_scales[i] = quantize_row_to_int8(
                &activations[i * k..(i + 1) * k],
                &mut acts_i8[i * k..(i + 1) * k],
            );
        }

        let mut i8_out = vec![0.0f32; m * n];
        matmul_i2s_i8(
            &acts_i8,
            &act_scales,
            m,
            k,
            &weight_bytes,
            n,
            weight_scale,
            &mut i8_out,
        );

        // Expected error: per-activation quantization error is ≤ scale/2,
        // and the dot product sums k=256 terms, so worst-case error scales
        // with sqrt(k) × max_weight_scale × act_scale/2. For our tiny test,
        // a relative tolerance of ~3% against the f32 reference is generous
        // enough to catch logic errors without flaking.
        for (i, (&r, &q)) in ref_out.iter().zip(i8_out.iter()).enumerate() {
            let abs_err = (r - q).abs();
            let rel_err = abs_err / r.abs().max(1e-6);
            assert!(
                rel_err < 0.05 || abs_err < 1e-3,
                "idx {i}: f32 ref = {r}, int8 quantized = {q}, rel_err = {rel_err}",
            );
        }
    }

    #[test]
    fn quantize_row_to_int8_roundtrip() {
        let input = [1.0f32, -2.0, 0.5, -0.5, 0.0, 2.0, -1.5];
        let mut output = [0i8; 7];
        let scale = quantize_row_to_int8(&input, &mut output);
        assert!(scale > 0.0);
        // Largest magnitude is 2.0; should map to ±127.
        let max_idx = input.iter().enumerate().max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()).unwrap().0;
        assert_eq!(output[max_idx].unsigned_abs(), 127);
        // Dequantized values should be close to the originals.
        for (i, &v) in input.iter().enumerate() {
            let recovered = output[i] as f32 * scale;
            assert!((recovered - v).abs() < scale, "idx {i}: {v} → {} (scale={scale})", recovered);
        }
    }

    #[test]
    fn quantize_row_to_int8_zero_input() {
        let input = [0.0f32; 8];
        let mut output = [0i8; 8];
        let scale = quantize_row_to_int8(&input, &mut output);
        assert_eq!(scale, 0.0);
        assert!(output.iter().all(|&x| x == 0));
    }

    #[test]
    #[should_panic(expected = "k")]
    fn matmul_rejects_misaligned_k() {
        let m = 1;
        let k = 100;
        let n = 1;
        let acts = vec![0.0; m * k];
        let weight_bytes = vec![0u8; I2S_BYTES_PER_BLOCK];
        let mut out = vec![0.0; m * n];
        matmul_i2s(&acts, m, k, &weight_bytes, n, 1.0, &mut out);
    }
}
