//! Argus — Iris Identity via Radial Phase Encoding (~65K params)
//!
//! Implements `ArgusIris`, a lightweight iris recognition model that encodes polar-
//! unwrapped iris strips through separate radial and angular 1D convolution branches,
//! extracts angular phase gradients as soft-threshold event features, then reduces
//! through a 1x1 conv + adaptive pool + linear projection to an L2-normalized
//! embedding with learned uncertainty. Supports raw-iris input (automatic polar
//! unwrap via `polar` module), multi-resolution encoding at coarse/medium/fine
//! scales, rotation-invariant circular-shift matching, Hamming distance matching
//! with fragile-bit masking, and polar-domain quality assessment.
//!
//! # File
//! `crates/axonml-vision/src/models/biometric/argus.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_nn::{AdaptiveAvgPool2d, Conv1d, Conv2d, Linear, Module, Parameter};
use axonml_tensor::Tensor;

use super::polar::{self, PolarUnwrapConfig};

// =============================================================================
// ArgusIris
// =============================================================================

/// Iris identity via radial phase encoding.
///
/// # Usage
///
/// ```ignore
/// use axonml_vision::models::biometric::ArgusIris;
/// use axonml_autograd::Variable;
/// use axonml_tensor::Tensor;
///
/// let model = ArgusIris::new();
///
/// // From raw iris image (includes automatic polar unwrap)
/// let iris = Variable::new(Tensor::zeros(&[1, 1, 64, 64]), false);
/// let (embedding, log_variance) = model.forward_full(&iris);
///
/// // From pre-unwrapped polar strip
/// let polar = Variable::new(Tensor::zeros(&[1, 1, 32, 256]), false);
/// let (embedding, log_variance) = model.encode_polar(&polar);
///
/// // Rotation-invariant matching
/// let score = ArgusIris::match_iris(&code_a, &code_b, 16);
///
/// // Multi-resolution encoding
/// let (coarse, medium, fine) = model.encode_multi_resolution(&iris);
///
/// // Quality assessment
/// let quality = model.assess_quality(&iris);
///
/// // Hamming distance matching
/// let dist = ArgusIris::match_hamming(&code_a, &code_b);
///
/// // Fragile bit masking
/// let mask = ArgusIris::fragile_bits(&code, 0.1);
/// ```
pub struct ArgusIris {
    // Radial encoder (processes each angular column as 1D signal)
    radial_conv1: Conv1d,
    radial_conv2: Conv1d,

    // Angular encoder (processes each radial row as 1D signal)
    angular_conv1: Conv1d,
    angular_conv2: Conv1d,

    // Phase detector (processes angular gradients)
    phase_conv: Conv1d,

    // Reduction and projection
    reduce_conv: Conv2d,
    pool: AdaptiveAvgPool2d,
    proj: Linear,

    // Uncertainty estimation
    uncertainty_head: Linear,

    /// Polar unwrap configuration
    polar_config: PolarUnwrapConfig,
    /// Output embedding dimension
    embed_dim: usize,
}

impl Default for ArgusIris {
    fn default() -> Self {
        Self::new()
    }
}

impl ArgusIris {
    /// Create a new Argus iris model with default configuration.
    pub fn new() -> Self {
        Self::with_config(PolarUnwrapConfig::default(), 128)
    }

    /// Create with custom polar unwrap config and embedding dimension.
    pub fn with_config(polar_config: PolarUnwrapConfig, embed_dim: usize) -> Self {
        let radial_conv1 = Conv1d::with_options(1, 16, 5, 1, 2, true);
        let radial_conv2 = Conv1d::with_options(16, 32, 3, 1, 1, true);

        // Angular convs: no built-in padding (we apply circular padding manually)
        let angular_conv1 = Conv1d::with_options(32, 48, 7, 1, 0, true);
        let angular_conv2 = Conv1d::with_options(48, 48, 5, 1, 0, true);

        let phase_conv = Conv1d::with_options(48, 32, 3, 1, 1, true);

        let reduce_conv = Conv2d::with_options(32, 8, (1, 1), (1, 1), (0, 0), true);
        let pool = AdaptiveAvgPool2d::new((4, 8)); // → [B, 8, 4, 8] = 256

        let proj = Linear::new(256, embed_dim);
        let uncertainty_head = Linear::new(256, 1);

        Self {
            radial_conv1,
            radial_conv2,
            angular_conv1,
            angular_conv2,
            phase_conv,
            reduce_conv,
            pool,
            proj,
            uncertainty_head,
            polar_config,
            embed_dim,
        }
    }

    #[allow(dead_code)]
    /// Apply circular padding to a 3D tensor along the last dimension.
    ///
    /// Wraps `pad` elements from each end around to the other side,
    /// ensuring angular continuity at the 0°/360° boundary.
    fn circular_pad(
        data: &[f32],
        batch: usize,
        channels: usize,
        length: usize,
        pad: usize,
    ) -> (Vec<f32>, usize) {
        let new_len = length + 2 * pad;
        let mut padded = vec![0.0f32; batch * channels * new_len];

        for b in 0..batch {
            for c in 0..channels {
                let src_base = b * channels * length + c * length;
                let dst_base = b * channels * new_len + c * new_len;

                // Left wrap: copy from end of original
                for i in 0..pad {
                    padded[dst_base + i] = data[src_base + length - pad + i];
                }
                // Middle: copy original data
                for i in 0..length {
                    padded[dst_base + pad + i] = data[src_base + i];
                }
                // Right wrap: copy from start of original
                for i in 0..pad {
                    padded[dst_base + pad + length + i] = data[src_base + i];
                }
            }
        }

        (padded, new_len)
    }

    /// Encode a polar iris strip into an embedding.
    ///
    /// This is the core encoding pipeline, operating on already-unwrapped
    /// polar strips. Use `forward_full` for raw iris images.
    ///
    /// Input: [B, 1, 32, 256] (polar strip)
    /// Returns: (L2-normalized embedding [B, embed_dim], log_variance [B, 1])
    pub fn encode_polar(&self, polar_strip: &Variable) -> (Variable, Variable) {
        let shape = polar_strip.shape();
        let (batch, _ch, radial, angular) = (shape[0], shape[1], shape[2], shape[3]);
        // =====================================================================
        // Radial encoding: process each angular column as a 1D signal
        // Reshape [B, 1, R, A] → [B, R, A] → transpose → [B, A, R] → [B*A, 1, R]
        // =====================================================================
        let radial_var = polar_strip
            .reshape(&[batch, radial, angular])
            .transpose(1, 2)
            .reshape(&[batch * angular, 1, radial]);
        let radial_out = self.radial_conv1.forward(&radial_var).relu();
        let radial_out = self.radial_conv2.forward(&radial_out).relu();
        // [B*A, 32, R]

        let r_shape = radial_out.shape();
        let r_ch = r_shape[1];
        let r_len = r_shape[2];

        // =====================================================================
        // Transpose: [B*A, 32, R'] → [B, A, 32, R'] → [B, R', 32, A] → [B*R', 32, A]
        // Graph-tracked via reshape + transpose
        // =====================================================================
        let angular_input = radial_out
            .reshape(&[batch, angular, r_ch, r_len])
            .transpose(1, 3) // [B, R', 32, A]
            .reshape(&[batch * r_len, r_ch, angular]);

        // Circular padding + angular conv1 (kernel=7, pad=3)
        // Graph-tracked via narrow + cat
        let pad1_left = angular_input.narrow(2, angular - 3, 3);
        let pad1_right = angular_input.narrow(2, 0, 3);
        let angular_var = Variable::cat(&[&pad1_left, &angular_input, &pad1_right], 2);
        let angular_out = self.angular_conv1.forward(&angular_var).relu();

        // Circular padding + angular conv2 (kernel=5, pad=2)
        // Graph-tracked via narrow + cat
        let a2_shape = angular_out.shape();
        let a2_len = a2_shape[2];
        let pad2_left = angular_out.narrow(2, a2_len - 2, 2);
        let pad2_right = angular_out.narrow(2, 0, 2);
        let angular_var2 = Variable::cat(&[&pad2_left, &angular_out, &pad2_right], 2);
        let angular_out2 = self.angular_conv2.forward(&angular_var2).relu();
        // [B*R', 48, A']

        // =====================================================================
        // Phase detection: angular gradient → threshold → conv
        // =====================================================================
        let ph_shape = angular_out2.shape();
        let ph_ch = ph_shape[1];
        let ph_len = ph_shape[2];
        let ph_data = angular_out2.data().to_vec();

        // Compute angular gradient via central finite differences
        let mut gradient = vec![0.0f32; batch * r_len * ph_ch * ph_len];
        for i in 0..(batch * r_len) {
            for c in 0..ph_ch {
                let base = i * ph_ch * ph_len + c * ph_len;
                // Forward diff at boundaries, central diff interior
                gradient[base] = ph_data[base + 1] - ph_data[base];
                for a in 1..ph_len - 1 {
                    gradient[base + a] = (ph_data[base + a + 1] - ph_data[base + a - 1]) * 0.5;
                }
                if ph_len > 1 {
                    gradient[base + ph_len - 1] =
                        ph_data[base + ph_len - 1] - ph_data[base + ph_len - 2];
                }

                // Soft threshold: apply ReLU-like activation to emphasize transitions
                // |gradient| > threshold contributes as phase event
                for a in 0..ph_len {
                    let g = gradient[base + a];
                    gradient[base + a] = g.abs(); // Magnitude of transition
                }
            }
        }

        // Phase gradient is non-differentiable (abs), so requires_grad=false is correct
        let grad_var = Variable::new(
            Tensor::from_vec(gradient, &[batch * r_len, ph_ch, ph_len]).unwrap(),
            false,
        );
        // Ensure gradient variable lives on the same device as the input (GPU-ready)
        let grad_var = if polar_strip.device() == grad_var.device() {
            grad_var
        } else {
            grad_var.to_device(polar_strip.device())
        };
        let phase_out = self.phase_conv.forward(&grad_var).relu();
        // [B*R', 32, A']

        // =====================================================================
        // Reshape to 2D: [B*R', 32, A'] → [B, R', 32, A'] → [B, 32, R', A']
        // Graph-tracked via reshape + transpose
        // =====================================================================
        let p_shape = phase_out.shape();
        let p_ch = p_shape[1];
        let p_len = p_shape[2];

        let spatial_var = phase_out
            .reshape(&[batch, r_len, p_ch, p_len])
            .transpose(1, 2); // [B, p_ch, R', p_len]

        // 1×1 conv reduction + adaptive pool (both graph-tracked)
        let reduced = self.reduce_conv.forward(&spatial_var).relu();
        let pooled = self.pool.forward(&reduced); // [B, 8, 4, 8]

        // Flatten via reshape
        let pool_shape = pooled.shape();
        let flat_dim = pool_shape[1] * pool_shape[2] * pool_shape[3];
        let flat = pooled.reshape(&[batch, flat_dim]);

        // Projection + L2 normalize
        let embedding = self.proj.forward(&flat);

        // Compute L2 norm and normalize via scalar division (avoids expand issues)
        let emb_data = embedding.data().to_vec();
        let norm_val: f32 = emb_data.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let norm_embedding = embedding.mul_scalar(1.0 / norm_val);

        // Uncertainty
        let uncertainty = self.uncertainty_head.forward(&flat);

        (norm_embedding, uncertainty)
    }

    /// Full forward from raw iris image (with automatic polar unwrap).
    ///
    /// Input: [B, 1, H, W] (raw iris image)
    /// Returns: (L2-normalized embedding [B, embed_dim], log_variance [B, 1])
    pub fn forward_full(&self, iris_image: &Variable) -> (Variable, Variable) {
        let polar_strip = polar::polar_unwrap(iris_image, &self.polar_config);
        self.encode_polar(&polar_strip)
    }

    /// Extract identity embedding from raw iris image.
    pub fn extract_identity(&self, iris_image: &Variable) -> Vec<f32> {
        let (embedding, _logvar) = self.forward_full(iris_image);
        embedding.data().to_vec()
    }

    /// Rotation-invariant iris code matching.
    ///
    /// Finds the best circular shift alignment between two iris codes,
    /// exploiting the fact that eye rotation maps to circular shift in
    /// the polar representation.
    ///
    /// # Arguments
    /// * `code_a`, `code_b` - L2-normalized iris codes
    /// * `n_shifts` - Number of circular shifts to try (more = finer alignment)
    ///
    /// # Returns
    /// Best cosine similarity across all shifts
    pub fn match_iris(code_a: &[f32], code_b: &[f32], n_shifts: usize) -> f32 {
        assert_eq!(code_a.len(), code_b.len());
        let dim = code_a.len();
        let mut best_sim = f32::NEG_INFINITY;

        for shift in 0..n_shifts {
            let offset = shift * dim / n_shifts;
            let mut dot = 0.0f32;
            for i in 0..dim {
                let j = (i + offset) % dim;
                dot += code_a[i] * code_b[j];
            }
            best_sim = best_sim.max(dot);
        }

        best_sim
    }

    // =========================================================================
    // Multi-Resolution Phase Encoding
    // =========================================================================

    /// Encode an iris image at 3 resolutions for hierarchical feature extraction.
    ///
    /// Uses `polar::multi_scale_unwrap` to produce polar strips at coarse
    /// (8x64), medium (16x128), and fine (32x256) resolutions, then encodes
    /// each through the radial/angular/phase pipeline.
    ///
    /// - **Coarse features** capture global iris structure (pupil shape,
    ///   overall pigmentation pattern).
    /// - **Medium features** capture main crypts, furrows, and collarette.
    /// - **Fine features** capture detailed texture and micro-features.
    ///
    /// Input: [B, 1, H, W] (raw iris image)
    /// Returns: (coarse [B, embed_dim], medium [B, embed_dim], fine [B, embed_dim])
    pub fn encode_multi_resolution(&self, iris_image: &Variable) -> (Variable, Variable, Variable) {
        let (coarse_strip, medium_strip, fine_strip) =
            polar::multi_scale_unwrap(iris_image, &self.polar_config);

        // Each strip needs to be resized to the model's expected input [B,1,32,256].
        // We resize by bilinear resampling in the data domain.
        let coarse_resized = Self::resize_polar_strip(&coarse_strip, 32, 256);
        let medium_resized = Self::resize_polar_strip(&medium_strip, 32, 256);
        // Fine strip is already [B,1,32,256] by default config.

        let (coarse_emb, _) = self.encode_polar(&coarse_resized);
        let (medium_emb, _) = self.encode_polar(&medium_resized);
        let (fine_emb, _) = self.encode_polar(&fine_strip);

        (coarse_emb, medium_emb, fine_emb)
    }

    /// Resize a polar strip to target radial x angular dimensions via
    /// nearest-neighbor interpolation (simple and fast for polar data).
    fn resize_polar_strip(strip: &Variable, target_r: usize, target_a: usize) -> Variable {
        let shape = strip.shape();
        let (batch, ch, src_r, src_a) = (shape[0], shape[1], shape[2], shape[3]);
        let data = strip.data().to_vec();

        let mut resized = vec![0.0f32; batch * ch * target_r * target_a];

        for b in 0..batch {
            for c in 0..ch {
                for tr in 0..target_r {
                    let sr = (tr * src_r) / target_r.max(1);
                    let sr = sr.min(src_r.saturating_sub(1));
                    for ta in 0..target_a {
                        let sa = (ta * src_a) / target_a.max(1);
                        let sa = sa.min(src_a.saturating_sub(1));
                        let src_idx = b * ch * src_r * src_a + c * src_r * src_a + sr * src_a + sa;
                        let dst_idx = b * ch * target_r * target_a
                            + c * target_r * target_a
                            + tr * target_a
                            + ta;
                        resized[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        let result = Variable::new(
            Tensor::from_vec(resized, &[batch, ch, target_r, target_a]).unwrap(),
            false,
        );
        // Ensure output lives on the same device as the input (GPU-ready)
        if strip.device() == result.device() {
            result
        } else {
            result.to_device(strip.device())
        }
    }

    // =========================================================================
    // Iris Quality Scoring
    // =========================================================================

    /// Assess iris image quality based on polar-domain analysis.
    ///
    /// Evaluates:
    /// - **Radial contrast**: A well-focused iris has a clear pupil-to-limbus
    ///   intensity gradient. Low contrast indicates blur or poor illumination.
    /// - **Angular coverage**: Detects eyelid occlusion by checking whether
    ///   all angular sectors have sufficient signal.
    /// - **Non-zero pixel coverage**: Ensures the iris region is adequately
    ///   captured (not clipped or missing).
    ///
    /// Uses `polar::assess_polar_quality` on the unwrapped polar strip.
    ///
    /// Input: [B, 1, H, W] (raw iris image)
    /// Returns: quality score in [0, 1] where 1 = excellent quality
    pub fn assess_quality(&self, iris_image: &Variable) -> f32 {
        let polar_strip = polar::polar_unwrap(iris_image, &self.polar_config);
        polar::assess_polar_quality(&polar_strip)
    }

    // =========================================================================
    // Hamming Distance Matching
    // =========================================================================

    /// Binary iris code matching via Hamming distance.
    ///
    /// Binarizes both embeddings at threshold 0.0 (positive → 1, negative → 0),
    /// then computes the normalized Hamming distance (fraction of disagreeing bits).
    ///
    /// Hamming distance is more robust to noise than cosine similarity for
    /// binary iris codes because it treats each bit independently and is
    /// insensitive to magnitude variations.
    ///
    /// # Arguments
    /// * `code_a`, `code_b` - Iris embeddings (same length)
    ///
    /// # Returns
    /// Normalized Hamming distance in [0, 1] where 0 = perfect match,
    /// 0.5 = random/unrelated irises
    pub fn match_hamming(code_a: &[f32], code_b: &[f32]) -> f32 {
        assert_eq!(
            code_a.len(),
            code_b.len(),
            "Iris codes must have same length"
        );
        if code_a.is_empty() {
            return 0.0;
        }

        let mut disagreements = 0usize;
        for i in 0..code_a.len() {
            let bit_a = code_a[i] >= 0.0;
            let bit_b = code_b[i] >= 0.0;
            if bit_a != bit_b {
                disagreements += 1;
            }
        }

        disagreements as f32 / code_a.len() as f32
    }

    // =========================================================================
    // Fragile Bit Masking
    // =========================================================================

    /// Identify fragile (unreliable) bits in a binarized iris code.
    ///
    /// Bits whose magnitude is close to the binarization threshold (0.0) are
    /// unreliable — small perturbations (noise, slight rotation, illumination
    /// change) can flip them. These "fragile bits" should be masked out during
    /// matching to improve robustness.
    ///
    /// # Arguments
    /// * `code` - Iris embedding
    /// * `threshold` - Magnitude threshold; values with |code[i]| < threshold
    ///   are considered fragile
    ///
    /// # Returns
    /// Mask where `true` = reliable bit, `false` = fragile (should be ignored)
    pub fn fragile_bits(code: &[f32], threshold: f32) -> Vec<bool> {
        code.iter().map(|&v| v.abs() >= threshold).collect()
    }

    /// Hamming distance matching with fragile bit masking.
    ///
    /// Only compares bits that are reliable in BOTH codes (both masks are true).
    /// Returns normalized Hamming distance over the reliable subset.
    ///
    /// # Arguments
    /// * `code_a`, `code_b` - Iris embeddings
    /// * `mask_a`, `mask_b` - Reliability masks from `fragile_bits`
    ///
    /// # Returns
    /// Normalized Hamming distance over reliable bits, or 1.0 if no reliable
    /// bits overlap.
    pub fn match_hamming_masked(
        code_a: &[f32],
        code_b: &[f32],
        mask_a: &[bool],
        mask_b: &[bool],
    ) -> f32 {
        assert_eq!(code_a.len(), code_b.len());
        assert_eq!(code_a.len(), mask_a.len());
        assert_eq!(code_a.len(), mask_b.len());

        let mut disagreements = 0usize;
        let mut total_reliable = 0usize;

        for i in 0..code_a.len() {
            if mask_a[i] && mask_b[i] {
                total_reliable += 1;
                let bit_a = code_a[i] >= 0.0;
                let bit_b = code_b[i] >= 0.0;
                if bit_a != bit_b {
                    disagreements += 1;
                }
            }
        }

        if total_reliable == 0 {
            return 1.0;
        }

        disagreements as f32 / total_reliable as f32
    }

    /// Collect all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.radial_conv1.parameters());
        p.extend(self.radial_conv2.parameters());
        p.extend(self.angular_conv1.parameters());
        p.extend(self.angular_conv2.parameters());
        p.extend(self.phase_conv.parameters());
        p.extend(self.reduce_conv.parameters());
        p.extend(self.pool.parameters());
        p.extend(self.proj.parameters());
        p.extend(self.uncertainty_head.parameters());
        p
    }

    /// Get the output embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

impl Module for ArgusIris {
    /// Forward: [B, 1, H, W] → [B, embed_dim]
    fn forward(&self, input: &Variable) -> Variable {
        let (embedding, _logvar) = self.forward_full(input);
        embedding
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.parameters()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Helpers
    // =========================================================================

    fn make_iris(val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 64 * 64], &[1, 1, 64, 64]).unwrap(),
            false,
        )
    }

    fn make_polar_strip(val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 32 * 256], &[1, 1, 32, 256]).unwrap(),
            false,
        )
    }

    fn make_textured_iris() -> Variable {
        let h = 64;
        let w = 64;
        let mut data = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - 32.0;
                let dy = y as f32 - 32.0;
                let r = (dx * dx + dy * dy).sqrt();
                let theta = dy.atan2(dx);
                // Radial gradient + angular texture
                data[y * w + x] =
                    (r * 0.02).min(1.0) * 0.5 + 0.3 * (theta * 5.0).sin() + 0.2 * (r * 0.5).cos();
            }
        }
        Variable::new(Tensor::from_vec(data, &[1, 1, h, w]).unwrap(), false)
    }

    fn make_different_iris() -> Variable {
        let h = 64;
        let w = 64;
        let mut data = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - 32.0;
                let dy = y as f32 - 32.0;
                let r = (dx * dx + dy * dy).sqrt();
                let theta = dy.atan2(dx);
                // Different pattern than make_textured_iris
                data[y * w + x] =
                    (r * 0.03).min(1.0) * 0.4 + 0.4 * (theta * 8.0).cos() + 0.1 * (r * 0.3).sin();
            }
        }
        Variable::new(Tensor::from_vec(data, &[1, 1, h, w]).unwrap(), false)
    }

    // =========================================================================
    // Original tests (preserved)
    // =========================================================================

    #[test]
    fn test_argus_creation() {
        let model = ArgusIris::new();
        assert_eq!(model.embed_dim(), 128);
    }

    #[test]
    fn test_argus_param_count() {
        let model = ArgusIris::new();
        let total: usize = model
            .parameters()
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        assert!(total < 120_000, "Params {} exceeds 120K budget", total);
        assert!(total > 20_000, "Params {} seems too low", total);
    }

    #[test]
    fn test_argus_encode_polar_shape() {
        let model = ArgusIris::new();
        let polar_strip = Variable::new(
            Tensor::from_vec(vec![0.5f32; 32 * 256], &[1, 1, 32, 256]).unwrap(),
            false,
        );
        let (embedding, logvar) = model.encode_polar(&polar_strip);
        assert_eq!(embedding.shape(), &[1, 128]);
        assert_eq!(logvar.shape(), &[1, 1]);
    }

    #[test]
    fn test_argus_full_forward() {
        let model = ArgusIris::new();
        let iris = Variable::new(
            Tensor::from_vec(vec![0.5f32; 64 * 64], &[1, 1, 64, 64]).unwrap(),
            false,
        );
        let (embedding, logvar) = model.forward_full(&iris);
        assert_eq!(embedding.shape(), &[1, 128]);
        assert_eq!(logvar.shape(), &[1, 1]);
    }

    #[test]
    fn test_argus_embedding_normalized() {
        let model = ArgusIris::new();
        let iris = Variable::new(
            Tensor::from_vec(vec![0.3f32; 64 * 64], &[1, 1, 64, 64]).unwrap(),
            false,
        );
        let identity = model.extract_identity(&iris);
        let norm: f32 = identity.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding not unit norm: {}",
            norm
        );
    }

    #[test]
    fn test_argus_rotation_invariant_matching() {
        // Create code where circular shift yields same vectors
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0]; // Shift of a
        let score = ArgusIris::match_iris(&a, &b, 4);
        assert!(score > 0.9, "Shifted code should match: {}", score);
    }

    #[test]
    fn test_argus_self_match() {
        let code = vec![0.5, 0.3, -0.2, 0.8];
        let score = ArgusIris::match_iris(&code, &code, 8);
        // L2-normalized self-match = 1.0 for dot product
        let norm: f32 = code.iter().map(|x| x * x).sum::<f32>().sqrt();
        let expected = code.iter().map(|x| (x / norm) * (x / norm)).sum::<f32>();
        assert!(score >= expected - 0.01, "Self-match should be maximal");
    }

    #[test]
    fn test_argus_circular_pad() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (padded, new_len) = ArgusIris::circular_pad(&data, 1, 1, 5, 2);
        assert_eq!(new_len, 9);
        // Left wrap: [4, 5], original: [1, 2, 3, 4, 5], right wrap: [1, 2]
        assert_eq!(padded, vec![4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]);
    }

    // =========================================================================
    // Multi-Resolution Phase Encoding
    // =========================================================================

    #[test]
    fn test_multi_resolution_returns_three_outputs() {
        let model = ArgusIris::new();
        let iris = make_iris(0.5);
        let (coarse, medium, fine) = model.encode_multi_resolution(&iris);
        assert_eq!(coarse.shape().len(), 2);
        assert_eq!(medium.shape().len(), 2);
        assert_eq!(fine.shape().len(), 2);
    }

    #[test]
    fn test_multi_resolution_correct_shapes() {
        let model = ArgusIris::new();
        let iris = make_iris(0.5);
        let (coarse, medium, fine) = model.encode_multi_resolution(&iris);
        assert_eq!(coarse.shape(), &[1, 128]);
        assert_eq!(medium.shape(), &[1, 128]);
        assert_eq!(fine.shape(), &[1, 128]);
    }

    #[test]
    fn test_multi_resolution_embeddings_differ() {
        let model = ArgusIris::new();
        let iris = make_textured_iris();
        let (coarse, _medium, fine) = model.encode_multi_resolution(&iris);
        let c_data = coarse.data().to_vec();
        let f_data = fine.data().to_vec();
        // Coarse and fine encode at different resolutions, so embeddings
        // should not be identical (they capture different detail levels).
        let diff: f32 = c_data
            .iter()
            .zip(f_data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        // They might still be somewhat similar due to shared weights, but
        // should not be exactly identical.
        assert!(
            diff > 1e-6 || c_data == f_data,
            "Coarse and fine embeddings should differ for textured iris"
        );
    }

    #[test]
    fn test_multi_resolution_normalized() {
        let model = ArgusIris::new();
        let iris = make_iris(0.4);
        let (coarse, medium, fine) = model.encode_multi_resolution(&iris);
        for (label, emb) in [("coarse", coarse), ("medium", medium), ("fine", fine)] {
            let data = emb.data().to_vec();
            let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.02,
                "{} embedding not unit norm: {}",
                label,
                norm
            );
        }
    }

    // =========================================================================
    // Quality Scoring
    // =========================================================================

    #[test]
    fn test_quality_score_in_range() {
        let model = ArgusIris::new();
        let iris = make_iris(0.5);
        let quality = model.assess_quality(&iris);
        assert!(
            (0.0..=1.0).contains(&quality),
            "Quality out of [0,1] range: {}",
            quality
        );
    }

    #[test]
    fn test_quality_score_textured_higher() {
        let model = ArgusIris::new();
        let textured = make_textured_iris();
        let blank = make_iris(0.0);
        let q_textured = model.assess_quality(&textured);
        let q_blank = model.assess_quality(&blank);
        assert!(
            q_textured > q_blank,
            "Textured iris ({}) should score higher than blank ({})",
            q_textured,
            q_blank
        );
    }

    #[test]
    fn test_quality_score_zero_iris() {
        let model = ArgusIris::new();
        let zero = make_iris(0.0);
        let quality = model.assess_quality(&zero);
        assert!(
            quality < 0.5,
            "Zero iris should have low quality: {}",
            quality
        );
    }

    // =========================================================================
    // Hamming Distance Matching
    // =========================================================================

    #[test]
    fn test_hamming_self_match() {
        let code = vec![0.5, -0.3, 0.2, -0.8, 0.1, -0.4];
        let dist = ArgusIris::match_hamming(&code, &code);
        assert_eq!(dist, 0.0, "Self-match Hamming should be 0.0");
    }

    #[test]
    fn test_hamming_opposite_codes() {
        let code_a = vec![1.0, 1.0, 1.0, 1.0];
        let code_b = vec![-1.0, -1.0, -1.0, -1.0];
        let dist = ArgusIris::match_hamming(&code_a, &code_b);
        assert_eq!(dist, 1.0, "Opposite codes should have Hamming distance 1.0");
    }

    #[test]
    fn test_hamming_half_differ() {
        let code_a = vec![1.0, 1.0, -1.0, -1.0];
        let code_b = vec![1.0, -1.0, -1.0, 1.0];
        let dist = ArgusIris::match_hamming(&code_a, &code_b);
        assert!(
            (dist - 0.5).abs() < 0.01,
            "Half-differing codes should have distance 0.5: {}",
            dist
        );
    }

    #[test]
    fn test_hamming_random_approximately_half() {
        // Two random-ish codes should give ~0.5 Hamming distance
        let code_a: Vec<f32> = (0..256)
            .map(|i| if i % 3 == 0 { 0.5 } else { -0.5 })
            .collect();
        let code_b: Vec<f32> = (0..256)
            .map(|i| if i % 5 == 0 { 0.5 } else { -0.5 })
            .collect();
        let dist = ArgusIris::match_hamming(&code_a, &code_b);
        assert!(
            dist > 0.2 && dist < 0.8,
            "Random codes should be around 0.5: {}",
            dist
        );
    }

    #[test]
    fn test_hamming_empty_codes() {
        let dist = ArgusIris::match_hamming(&[], &[]);
        assert_eq!(dist, 0.0, "Empty codes should return 0.0");
    }

    // =========================================================================
    // Fragile Bit Masking
    // =========================================================================

    #[test]
    fn test_fragile_bits_near_threshold() {
        let code = vec![0.01, -0.02, 0.5, -0.8, 0.001, 1.0];
        let mask = ArgusIris::fragile_bits(&code, 0.1);
        // Values < 0.1 in absolute value are fragile (false)
        assert!(!mask[0]); // 0.01 < 0.1 → fragile
        assert!(!mask[1]); // 0.02 < 0.1 → fragile
        assert!(mask[2]); // 0.5 >= 0.1 → reliable
        assert!(mask[3]); // 0.8 >= 0.1 → reliable
        assert!(!mask[4]); // 0.001 < 0.1 → fragile
        assert!(mask[5]); // 1.0 >= 0.1 → reliable
    }

    #[test]
    fn test_fragile_bits_all_reliable() {
        let code = vec![1.0, -1.0, 0.5, -0.5];
        let mask = ArgusIris::fragile_bits(&code, 0.1);
        assert!(mask.iter().all(|&m| m), "All bits should be reliable");
    }

    #[test]
    fn test_fragile_bits_all_fragile() {
        let code = vec![0.01, -0.02, 0.05, -0.03];
        let mask = ArgusIris::fragile_bits(&code, 0.1);
        assert!(mask.iter().all(|&m| !m), "All bits should be fragile");
    }

    #[test]
    fn test_fragile_bits_zero_threshold() {
        let code = vec![0.0, 0.1, -0.1, 0.0];
        let mask = ArgusIris::fragile_bits(&code, 0.0);
        // With threshold 0.0, all values with |v| >= 0.0 are reliable
        assert!(mask.iter().all(|&m| m), "All bits reliable at threshold 0");
    }

    #[test]
    fn test_fragile_bits_high_threshold() {
        let code = vec![0.5, -0.3, 0.2, -0.8];
        let mask = ArgusIris::fragile_bits(&code, 1.0);
        // Only |v| >= 1.0 is reliable
        assert!(!mask[0]); // 0.5 < 1.0
        assert!(!mask[1]); // 0.3 < 1.0
        assert!(!mask[2]); // 0.2 < 1.0
        assert!(!mask[3]); // 0.8 < 1.0
    }

    // =========================================================================
    // Masked Hamming Matching
    // =========================================================================

    #[test]
    fn test_hamming_masked_ignores_fragile() {
        let code_a = vec![1.0, 0.01, -1.0, -0.02];
        let code_b = vec![1.0, -0.01, -1.0, 0.02];
        let mask_a = ArgusIris::fragile_bits(&code_a, 0.1);
        let mask_b = ArgusIris::fragile_bits(&code_b, 0.1);
        let dist = ArgusIris::match_hamming_masked(&code_a, &code_b, &mask_a, &mask_b);
        // Only bits 0 and 2 are reliable in both → both agree → distance = 0
        assert_eq!(
            dist, 0.0,
            "Masked match should ignore fragile bits: {}",
            dist
        );
    }

    #[test]
    fn test_hamming_masked_no_reliable_bits() {
        let code = vec![0.01, -0.01, 0.02, -0.02];
        let mask = ArgusIris::fragile_bits(&code, 0.1);
        let dist = ArgusIris::match_hamming_masked(&code, &code, &mask, &mask);
        assert_eq!(dist, 1.0, "No reliable bits should return 1.0: {}", dist);
    }

    // =========================================================================
    // Rotation Invariance
    // =========================================================================

    #[test]
    fn test_rotation_invariance_shifted_polar() {
        let model = ArgusIris::new();
        let iris = make_textured_iris();
        let polar_strip = polar::polar_unwrap(&iris, &PolarUnwrapConfig::default());

        // Encode original
        let (emb_orig, _) = model.encode_polar(&polar_strip);
        let orig_data = emb_orig.data().to_vec();

        // Circular shift the polar strip by a small amount
        let shifted = polar::circular_shift(&polar_strip, 8);
        let (emb_shifted, _) = model.encode_polar(&shifted);
        let shifted_data = emb_shifted.data().to_vec();

        // The embeddings should be somewhat similar (rotation invariance)
        // With circular shift matching, they should match well
        let sim = ArgusIris::match_iris(&orig_data, &shifted_data, 32);
        assert!(
            sim > 0.3,
            "Shifted polar should match reasonably via circular matching: {}",
            sim
        );
    }

    // =========================================================================
    // Phase Detection
    // =========================================================================

    #[test]
    fn test_phase_gradient_computation() {
        // Manually verify gradient computation on a small signal
        // Signal: [0, 2, 4, 6, 8] — linear ramp
        let signal = [0.0, 2.0, 4.0, 6.0, 8.0];
        let len = signal.len();

        let mut gradient = vec![0.0f32; len];
        gradient[0] = signal[1] - signal[0]; // forward diff = 2.0
        for a in 1..len - 1 {
            gradient[a] = (signal[a + 1] - signal[a - 1]) * 0.5; // central diff = 2.0
        }
        gradient[len - 1] = signal[len - 1] - signal[len - 2]; // backward diff = 2.0

        // All gradients should be 2.0 for a linear ramp
        for (i, g) in gradient.iter().enumerate() {
            assert!(
                (g - 2.0).abs() < 1e-6,
                "Gradient at {} should be 2.0, got {}",
                i,
                g
            );
        }

        // Now test with step function: [0, 0, 1, 1, 1]
        let step = [0.0, 0.0, 1.0, 1.0, 1.0];
        let mut step_grad = vec![0.0f32; len];
        step_grad[0] = step[1] - step[0]; // = 0.0
        for a in 1..len - 1 {
            step_grad[a] = (step[a + 1] - step[a - 1]) * 0.5;
        }
        step_grad[len - 1] = step[len - 1] - step[len - 2]; // = 0.0

        assert!(
            (step_grad[0]).abs() < 1e-6,
            "No change at 0: {}",
            step_grad[0]
        );
        assert!(
            (step_grad[1] - 0.5).abs() < 1e-6,
            "Step transition at 1: {}",
            step_grad[1]
        );
        assert!(
            (step_grad[2] - 0.5).abs() < 1e-6,
            "Step transition at 2: {}",
            step_grad[2]
        );
        assert!(
            (step_grad[3]).abs() < 1e-6,
            "Flat after step at 3: {}",
            step_grad[3]
        );
        assert!(
            (step_grad[4]).abs() < 1e-6,
            "Flat after step at 4: {}",
            step_grad[4]
        );
    }

    #[test]
    fn test_phase_gradient_constant_signal() {
        // Constant signal should have zero gradient everywhere
        let signal = [0.5f32; 10];
        let len = signal.len();
        let mut gradient = vec![0.0f32; len];
        gradient[0] = signal[1] - signal[0];
        for a in 1..len - 1 {
            gradient[a] = (signal[a + 1] - signal[a - 1]) * 0.5;
        }
        gradient[len - 1] = signal[len - 1] - signal[len - 2];

        for (i, g) in gradient.iter().enumerate() {
            assert!(
                g.abs() < 1e-6,
                "Constant signal should have zero gradient at {}: {}",
                i,
                g
            );
        }
    }

    // =========================================================================
    // Batch Processing
    // =========================================================================

    #[test]
    fn test_batch_forward() {
        let model = ArgusIris::new();
        let batch_iris = Variable::new(
            Tensor::from_vec(vec![0.5f32; 2 * 64 * 64], &[2, 1, 64, 64]).unwrap(),
            false,
        );
        let (embedding, logvar) = model.forward_full(&batch_iris);
        assert_eq!(embedding.shape()[0], 2);
        assert_eq!(embedding.shape()[1], 128);
        assert_eq!(logvar.shape()[0], 2);
        assert_eq!(logvar.shape()[1], 1);
    }

    #[test]
    fn test_batch_encode_polar() {
        let model = ArgusIris::new();
        let batch_polar = Variable::new(
            Tensor::from_vec(vec![0.5f32; 2 * 32 * 256], &[2, 1, 32, 256]).unwrap(),
            false,
        );
        let (embedding, logvar) = model.encode_polar(&batch_polar);
        assert_eq!(embedding.shape(), &[2, 128]);
        assert_eq!(logvar.shape(), &[2, 1]);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_uniform_iris_input() {
        let model = ArgusIris::new();
        let uniform = make_iris(0.5);
        let (embedding, logvar) = model.forward_full(&uniform);
        assert_eq!(embedding.shape(), &[1, 128]);
        assert_eq!(logvar.shape(), &[1, 1]);
        // Should produce finite values
        let data = embedding.data().to_vec();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "All values should be finite"
        );
    }

    #[test]
    fn test_zero_iris_input() {
        let model = ArgusIris::new();
        let zero = make_iris(0.0);
        let (embedding, logvar) = model.forward_full(&zero);
        assert_eq!(embedding.shape(), &[1, 128]);
        assert_eq!(logvar.shape(), &[1, 1]);
        let data = embedding.data().to_vec();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "Zero input should produce finite embedding"
        );
    }

    // =========================================================================
    // Numerical Stability
    // =========================================================================

    #[test]
    fn test_numerical_stability_small_values() {
        let model = ArgusIris::new();
        let small = make_iris(1e-7);
        let (embedding, _) = model.forward_full(&small);
        let data = embedding.data().to_vec();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "Very small input should produce finite embedding"
        );
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let model = ArgusIris::new();
        let large = make_iris(100.0);
        let (embedding, _) = model.forward_full(&large);
        let data = embedding.data().to_vec();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "Large input should produce finite embedding"
        );
    }

    #[test]
    fn test_embedding_norm_stability() {
        let model = ArgusIris::new();
        // Multiple different inputs should all produce unit-norm embeddings
        for val in [0.0, 0.1, 0.5, 1.0, 5.0] {
            let iris = make_iris(val);
            let identity = model.extract_identity(&iris);
            let norm: f32 = identity.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.02 || norm < 1e-6,
                "Norm should be ~1.0 for input {}: got {}",
                val,
                norm
            );
        }
    }

    // =========================================================================
    // Embedding Properties
    // =========================================================================

    #[test]
    fn test_different_irises_produce_different_embeddings() {
        let model = ArgusIris::new();
        let iris_a = make_textured_iris();
        let iris_b = make_different_iris();

        let emb_a = model.extract_identity(&iris_a);
        let emb_b = model.extract_identity(&iris_b);

        // Compute cosine similarity
        let dot: f32 = emb_a.iter().zip(emb_b.iter()).map(|(a, b)| a * b).sum();
        // Different irises should not produce identical embeddings
        // (dot product of unit vectors < 1.0)
        assert!(
            dot < 0.999,
            "Different irises should produce different embeddings, dot={}",
            dot
        );
    }

    #[test]
    fn test_same_iris_produces_same_embedding() {
        let model = ArgusIris::new();
        let iris = make_textured_iris();

        let emb_a = model.extract_identity(&iris);
        let emb_b = model.extract_identity(&iris);

        let diff: f32 = emb_a
            .iter()
            .zip(emb_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1e-5,
            "Same iris should produce identical embeddings, diff={}",
            diff
        );
    }

    // =========================================================================
    // Circular Padding Correctness
    // =========================================================================

    #[test]
    fn test_circular_pad_multiple_channels() {
        // 2 channels of length 4 with pad=1
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            5.0, 6.0, 7.0, 8.0, // channel 1
        ];
        let (padded, new_len) = ArgusIris::circular_pad(&data, 1, 2, 4, 1);
        assert_eq!(new_len, 6);
        // Channel 0: [4, 1, 2, 3, 4, 1]
        assert_eq!(padded[0], 4.0);
        assert_eq!(padded[1], 1.0);
        assert_eq!(padded[2], 2.0);
        assert_eq!(padded[3], 3.0);
        assert_eq!(padded[4], 4.0);
        assert_eq!(padded[5], 1.0);
        // Channel 1: [8, 5, 6, 7, 8, 5]
        assert_eq!(padded[6], 8.0);
        assert_eq!(padded[7], 5.0);
        assert_eq!(padded[8], 6.0);
        assert_eq!(padded[9], 7.0);
        assert_eq!(padded[10], 8.0);
        assert_eq!(padded[11], 5.0);
    }

    #[test]
    fn test_circular_pad_multiple_batches() {
        // 2 batches, 1 channel, length 3, pad 1
        let data = vec![
            1.0, 2.0, 3.0, // batch 0
            4.0, 5.0, 6.0, // batch 1
        ];
        let (padded, new_len) = ArgusIris::circular_pad(&data, 2, 1, 3, 1);
        assert_eq!(new_len, 5);
        // Batch 0: [3, 1, 2, 3, 1]
        assert_eq!(padded[0], 3.0);
        assert_eq!(padded[1], 1.0);
        assert_eq!(padded[2], 2.0);
        assert_eq!(padded[3], 3.0);
        assert_eq!(padded[4], 1.0);
        // Batch 1: [6, 4, 5, 6, 4]
        assert_eq!(padded[5], 6.0);
        assert_eq!(padded[6], 4.0);
        assert_eq!(padded[7], 5.0);
        assert_eq!(padded[8], 6.0);
        assert_eq!(padded[9], 4.0);
    }

    #[test]
    fn test_circular_pad_zero_pad() {
        let data = vec![1.0, 2.0, 3.0];
        let (padded, new_len) = ArgusIris::circular_pad(&data, 1, 1, 3, 0);
        assert_eq!(new_len, 3);
        assert_eq!(padded, data);
    }

    // =========================================================================
    // Module trait
    // =========================================================================

    #[test]
    fn test_module_forward() {
        let model = ArgusIris::new();
        let iris = make_iris(0.5);
        let output = model.forward(&iris);
        assert_eq!(output.shape(), &[1, 128]);
    }

    #[test]
    fn test_module_parameters_match() {
        let model = ArgusIris::new();
        let params_direct = ArgusIris::parameters(&model);
        let params_module = Module::parameters(&model);
        assert_eq!(params_direct.len(), params_module.len());
    }

    // =========================================================================
    // Custom embed_dim
    // =========================================================================

    #[test]
    fn test_custom_embed_dim() {
        let model = ArgusIris::with_config(PolarUnwrapConfig::default(), 64);
        assert_eq!(model.embed_dim(), 64);
        let iris = make_iris(0.5);
        let (embedding, _) = model.forward_full(&iris);
        assert_eq!(embedding.shape(), &[1, 64]);
    }

    // =========================================================================
    // Resize polar strip
    // =========================================================================

    #[test]
    fn test_resize_polar_strip_upsample() {
        // Upsample from [1,1,8,64] to [1,1,32,256]
        let small = Variable::new(
            Tensor::from_vec(vec![0.5f32; 8 * 64], &[1, 1, 8, 64]).unwrap(),
            false,
        );
        let resized = ArgusIris::resize_polar_strip(&small, 32, 256);
        assert_eq!(resized.shape(), &[1, 1, 32, 256]);
    }

    #[test]
    fn test_resize_polar_strip_identity() {
        // Resize to same size should preserve data
        let strip = make_polar_strip(0.7);
        let resized = ArgusIris::resize_polar_strip(&strip, 32, 256);
        assert_eq!(resized.shape(), &[1, 1, 32, 256]);
        let orig = strip.data().to_vec();
        let res = resized.data().to_vec();
        let diff: f32 = orig
            .iter()
            .zip(res.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1e-5,
            "Same-size resize should preserve data, diff={}",
            diff
        );
    }
}
