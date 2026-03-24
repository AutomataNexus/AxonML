//! Ariadne — Fingerprint via Ridge Event Fields (~65K params)
//!
//! # File
//! `crates/axonml-vision/src/models/biometric/ariadne.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_nn::{AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, Module, Parameter};
use axonml_tensor::Tensor;
use std::f32::consts::PI;

// =============================================================================
// Singularity Types
// =============================================================================

/// Kind of fingerprint singularity point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingularityKind {
    /// Core: where ridge lines converge (positive Poincare index +1/2)
    Core,
    /// Delta: where three ridge directions meet (negative Poincare index -1/2)
    Delta,
}

/// A detected singularity in the fingerprint orientation field.
#[derive(Debug, Clone)]
pub struct Singularity {
    /// Type of singularity (Core or Delta).
    pub kind: SingularityKind,
    /// X coordinate in the orientation cell grid.
    pub x: f32,
    /// Y coordinate in the orientation cell grid.
    pub y: f32,
    /// Strength of the singularity (absolute Poincare index deviation).
    pub strength: f32,
}

// =============================================================================
// Gabor Initialization
// =============================================================================

/// Generate a Gabor wavelet kernel for fingerprint ridge detection.
///
/// The kernel combines a Gaussian envelope with a sinusoidal carrier,
/// creating an orientation-selective filter. During training, the filter
/// weights are fine-tuned while maintaining the general Gabor structure
/// as initialization.
///
/// # Arguments
/// * `ksize` - Kernel spatial size (e.g. 7 for 7x7)
/// * `theta` - Orientation angle in radians [0, pi)
/// * `sigma` - Gaussian envelope standard deviation
/// * `lambda` - Sinusoidal wavelength (ridge spacing)
/// * `psi` - Phase offset
fn gabor_kernel(ksize: usize, theta: f32, sigma: f32, lambda: f32, psi: f32) -> Vec<f32> {
    let half = ksize as f32 / 2.0;
    let mut kernel = vec![0.0f32; ksize * ksize];

    for y in 0..ksize {
        for x in 0..ksize {
            let xf = x as f32 - half + 0.5;
            let yf = y as f32 - half + 0.5;

            let x_rot = xf * theta.cos() + yf * theta.sin();
            let y_rot = -xf * theta.sin() + yf * theta.cos();

            let gaussian = (-0.5 * (x_rot * x_rot + y_rot * y_rot) / (sigma * sigma)).exp();
            let sinusoid = (2.0 * PI * x_rot / lambda + psi).cos();

            kernel[y * ksize + x] = gaussian * sinusoid;
        }
    }

    // L1 normalize to prevent magnitude explosion
    let sum: f32 = kernel.iter().map(|v| v.abs()).sum();
    if sum > 1e-8 {
        for v in &mut kernel {
            *v /= sum;
        }
    }

    kernel
}

// =============================================================================
// DWSepBlock — Depthwise Separable + Residual
// =============================================================================

/// Depthwise separable conv block for fingerprint field encoding.
struct DWSepBlock {
    dw_conv: Conv2d,
    dw_bn: BatchNorm2d,
    pw_conv: Conv2d,
    pw_bn: BatchNorm2d,
    project: Option<(Conv2d, BatchNorm2d)>,
}

impl DWSepBlock {
    fn new(in_ch: usize, out_ch: usize, stride: usize) -> Self {
        let dw_conv =
            Conv2d::with_groups(in_ch, in_ch, (3, 3), (stride, stride), (1, 1), true, in_ch);
        let dw_bn = BatchNorm2d::new(in_ch);
        let pw_conv = Conv2d::with_options(in_ch, out_ch, (1, 1), (1, 1), (0, 0), true);
        let pw_bn = BatchNorm2d::new(out_ch);

        let project = if in_ch != out_ch || stride != 1 {
            Some((
                Conv2d::with_options(in_ch, out_ch, (1, 1), (stride, stride), (0, 0), true),
                BatchNorm2d::new(out_ch),
            ))
        } else {
            None
        };

        Self {
            dw_conv,
            dw_bn,
            pw_conv,
            pw_bn,
            project,
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let identity = if let Some((ref c, ref b)) = self.project {
            b.forward(&c.forward(x))
        } else {
            x.clone()
        };
        let out = self.dw_bn.forward(&self.dw_conv.forward(x)).relu();
        let out = self.pw_bn.forward(&self.pw_conv.forward(&out));
        out.add_var(&identity).relu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        p.extend(self.dw_conv.parameters());
        p.extend(self.dw_bn.parameters());
        p.extend(self.pw_conv.parameters());
        p.extend(self.pw_bn.parameters());
        if let Some((ref c, ref b)) = self.project {
            p.extend(c.parameters());
            p.extend(b.parameters());
        }
        p
    }
}

// =============================================================================
// AriadneFingerprint
// =============================================================================

/// Fingerprint identity via ridge event fields.
///
/// # Usage
///
/// ```ignore
/// use axonml_vision::models::biometric::AriadneFingerprint;
/// use axonml_autograd::Variable;
/// use axonml_tensor::Tensor;
///
/// let model = AriadneFingerprint::new();
///
/// // Fingerprint image [B, 1, 128, 128]
/// let fp = Variable::new(Tensor::zeros(&[1, 1, 128, 128]), false);
/// let (embedding, log_variance) = model.forward_full(&fp);
/// // embedding: [1, 128], L2-normalized
/// ```
pub struct AriadneFingerprint {
    /// 8 Gabor-initialized orientation filters (learnable)
    gabor_filters: Vec<Conv2d>,

    // Field encoder (3 depthwise separable blocks)
    field_block1: DWSepBlock,
    field_block2: DWSepBlock,
    field_block3: DWSepBlock,

    // Spatial hash + pooling
    spatial_conv: Conv2d,
    spatial_bn: BatchNorm2d,
    pool: AdaptiveAvgPool2d,

    // Projection head
    proj1: Linear,
    proj2: Linear,

    // Uncertainty estimation
    uncertainty_head: Linear,

    /// Number of Gabor orientations
    n_orientations: usize,
    /// Output embedding dimension
    embed_dim: usize,
}

impl AriadneFingerprint {
    /// Create a new Ariadne fingerprint model with defaults.
    ///
    /// Default: 8 orientations, 128-dim embedding
    pub fn new() -> Self {
        Self::with_config(8, 128)
    }

    /// Create with custom configuration.
    ///
    /// # Arguments
    /// * `n_orientations` - Number of Gabor filter orientations (typically 8)
    /// * `embed_dim` - Dimension of the output embedding
    pub fn with_config(n_orientations: usize, embed_dim: usize) -> Self {
        // Create Gabor filters at evenly-spaced orientations
        let mut gabor_filters = Vec::new();
        for i in 0..n_orientations {
            let theta = (i as f32) * PI / (n_orientations as f32);
            let mut conv = Conv2d::with_options(1, 1, (7, 7), (1, 1), (3, 3), false);

            // Initialize weights as Gabor kernel (will be fine-tuned during training)
            let kernel = gabor_kernel(7, theta, 2.0, 4.0, 0.0);
            let kernel_tensor = Tensor::from_vec(kernel, &[1, 1, 7, 7]).unwrap();
            conv.weight = Parameter::named(format!("gabor_{}", i), kernel_tensor, true);

            gabor_filters.push(conv);
        }

        let field_block1 = DWSepBlock::new(2, 16, 2); // [2, 128, 128] -> [16, 64, 64]
        let field_block2 = DWSepBlock::new(16, 32, 2); // -> [32, 32, 32]
        let field_block3 = DWSepBlock::new(32, 64, 1); // -> [64, 32, 32]

        let spatial_conv = Conv2d::with_options(64, 16, (1, 1), (1, 1), (0, 0), true);
        let spatial_bn = BatchNorm2d::new(16);
        let pool = AdaptiveAvgPool2d::new((4, 4)); // [16, 32, 32] -> [16, 4, 4] = 256

        let proj1 = Linear::new(256, 64);
        let proj2 = Linear::new(64, embed_dim);
        let uncertainty_head = Linear::new(256, 1);

        Self {
            gabor_filters,
            field_block1,
            field_block2,
            field_block3,
            spatial_conv,
            spatial_bn,
            pool,
            proj1,
            proj2,
            uncertainty_head,
            n_orientations,
            embed_dim,
        }
    }

    /// Extract ridge event field from fingerprint image.
    ///
    /// Applies 8 Gabor filters and computes a 2-channel event field:
    /// - Channel 0: dominant ridge orientation (encoded as atan2(sin,cos)/pi -> [-1,1])
    /// - Channel 1: response magnitude (max across orientations)
    ///
    /// Input: [B, 1, 128, 128] -> Output: [B, 2, 128, 128]
    fn extract_ridge_events(&self, fingerprint: &Variable) -> Variable {
        let shape = fingerprint.shape();
        let (batch, _ch, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // Apply each Gabor filter (graph-tracked through Conv2d)
        let mut responses = Vec::new();
        for filter in &self.gabor_filters {
            let resp = filter.forward(fingerprint); // [B, 1, H, W]
            responses.push(resp.data().to_vec());
        }

        // Compute dominant orientation and magnitude via soft-argmax
        let spatial = h * w;
        let mut ridge_events = vec![0.0f32; batch * 2 * spatial];

        for b in 0..batch {
            for s in 0..spatial {
                let mut max_response = 0.0f32;
                let mut weighted_sin = 0.0f32;
                let mut weighted_cos = 0.0f32;

                for (i, resp) in responses.iter().enumerate() {
                    let val = resp[b * spatial + s].abs();
                    let theta = (i as f32) * PI / (self.n_orientations as f32);

                    // Temperature-scaled soft-max weighting for smooth orientation
                    let w = (val * 5.0).exp();
                    weighted_sin += w * theta.sin();
                    weighted_cos += w * theta.cos();

                    if val > max_response {
                        max_response = val;
                    }
                }

                // Dominant orientation encoded as normalized angle
                let dominant_angle = weighted_sin.atan2(weighted_cos);
                ridge_events[b * 2 * spatial + s] = dominant_angle / PI;
                // Response magnitude
                ridge_events[b * 2 * spatial + spatial + s] = max_response;
            }
        }

        let result = Variable::new(
            Tensor::from_vec(ridge_events, &[batch, 2, h, w]).unwrap(),
            fingerprint.requires_grad(),
        );
        // Ensure output lives on the same device as the input (GPU-ready)
        if fingerprint.device() != result.device() {
            result.to_device(fingerprint.device())
        } else {
            result
        }
    }

    /// Compute Gabor response magnitudes for each orientation.
    ///
    /// Returns raw response data as Vec<Vec<f32>> (one per orientation)
    /// and the spatial dimensions (batch, h, w).
    fn gabor_responses(&self, fingerprint: &Variable) -> (Vec<Vec<f32>>, usize, usize, usize) {
        let shape = fingerprint.shape();
        let (batch, h, w) = (shape[0], shape[2], shape[3]);

        let mut responses = Vec::new();
        for filter in &self.gabor_filters {
            let resp = filter.forward(fingerprint);
            responses.push(resp.data().to_vec());
        }

        (responses, batch, h, w)
    }

    /// Compute the per-pixel orientation field from Gabor responses.
    ///
    /// Returns a vector of (angle_in_radians, magnitude) per spatial location,
    /// laid out as [batch * h * w].
    fn orientation_field(
        responses: &[Vec<f32>],
        n_orientations: usize,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Vec<(f32, f32)> {
        let spatial = h * w;
        let mut field = Vec::with_capacity(batch * spatial);

        for b in 0..batch {
            for s in 0..spatial {
                let mut weighted_sin = 0.0f32;
                let mut weighted_cos = 0.0f32;
                let mut max_mag = 0.0f32;

                for (i, resp) in responses.iter().enumerate() {
                    let val = resp[b * spatial + s].abs();
                    let theta = (i as f32) * PI / (n_orientations as f32);

                    let w_val = (val * 5.0).exp();
                    weighted_sin += w_val * theta.sin();
                    weighted_cos += w_val * theta.cos();

                    if val > max_mag {
                        max_mag = val;
                    }
                }

                let angle = weighted_sin.atan2(weighted_cos);
                field.push((angle, max_mag));
            }
        }

        field
    }

    /// Compute a ridge density quality map over 8x8 cells.
    ///
    /// For each cell, aggregates the Gabor response magnitudes across all
    /// orientations. High-quality fingerprint regions exhibit consistent,
    /// moderate density. Very high or very low density indicates poor
    /// quality (over-inked, dry, or background).
    ///
    /// Input: [B, 1, H, W]
    /// Output: [B, 1, cells_h, cells_w] where cells_h = H/8, cells_w = W/8
    pub fn ridge_density_map(&self, fingerprint: &Variable) -> Variable {
        let (responses, batch, h, w) = self.gabor_responses(fingerprint);
        let spatial = h * w;
        let cell_size = 8;
        let cells_h = h / cell_size;
        let cells_w = w / cell_size;

        let mut density = vec![0.0f32; batch * cells_h * cells_w];

        for b in 0..batch {
            for cy in 0..cells_h {
                for cx in 0..cells_w {
                    let mut cell_sum = 0.0f32;
                    let mut count = 0usize;

                    for dy in 0..cell_size {
                        for dx in 0..cell_size {
                            let py = cy * cell_size + dy;
                            let px = cx * cell_size + dx;
                            if py < h && px < w {
                                let idx = b * spatial + py * w + px;
                                // Sum absolute responses across all orientations
                                for resp in &responses {
                                    cell_sum += resp[idx].abs();
                                }
                                count += 1;
                            }
                        }
                    }

                    let n_orient = self.n_orientations as f32;
                    let avg = if count > 0 {
                        cell_sum / (count as f32 * n_orient)
                    } else {
                        0.0
                    };
                    density[b * cells_h * cells_w + cy * cells_w + cx] = avg;
                }
            }
        }

        let result = Variable::new(
            Tensor::from_vec(density, &[batch, 1, cells_h, cells_w]).unwrap(),
            false,
        );
        // Ensure output lives on the same device as the input (GPU-ready)
        if fingerprint.device() != result.device() {
            result.to_device(fingerprint.device())
        } else {
            result
        }
    }

    /// Detect core and delta singularities using the Poincare index method.
    ///
    /// Computes the orientation field from Gabor responses, then evaluates the
    /// Poincare index around each interior pixel. A full +pi winding indicates
    /// a core (ridge convergence), while -pi indicates a delta (tri-radii).
    ///
    /// Only the first sample in the batch is analyzed (batch index 0).
    ///
    /// Returns a list of `Singularity` structs with grid coordinates and strength.
    pub fn detect_singularities(&self, fingerprint: &Variable) -> Vec<Singularity> {
        let (responses, batch, h, w) = self.gabor_responses(fingerprint);
        if batch == 0 || h < 3 || w < 3 {
            return Vec::new();
        }

        let field = Self::orientation_field(&responses, self.n_orientations, 1, h, w);

        // Poincare index: sum of angle differences around a closed path
        // around each pixel. Core: index ~ +pi, Delta: index ~ -pi.
        let mut singularities = Vec::new();

        // Closed path neighbors (clockwise): right, down-right, down, down-left, left, up-left, up, up-right
        let neighbors: [(isize, isize); 8] = [
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ];

        // Threshold for singularity detection (ideally +/- pi)
        let core_threshold = PI * 0.6;
        let delta_threshold = PI * 0.6;

        for y in 1..(h as isize - 1) {
            for x in 1..(w as isize - 1) {
                let mut poincare_sum = 0.0f32;

                for k in 0..8 {
                    let (dy1, dx1) = neighbors[k];
                    let (dy2, dx2) = neighbors[(k + 1) % 8];

                    let idx1 = (y + dy1) as usize * w + (x + dx1) as usize;
                    let idx2 = (y + dy2) as usize * w + (x + dx2) as usize;

                    let angle1 = field[idx1].0;
                    let angle2 = field[idx2].0;

                    // Angle difference wrapped to [-pi, pi]
                    let mut diff = angle2 - angle1;
                    while diff > PI {
                        diff -= 2.0 * PI;
                    }
                    while diff < -PI {
                        diff += 2.0 * PI;
                    }

                    poincare_sum += diff;
                }

                let idx_center = y as usize * w + x as usize;
                let magnitude = field[idx_center].1;

                // Only consider points with sufficient ridge magnitude
                if magnitude < 1e-6 {
                    continue;
                }

                if poincare_sum > core_threshold {
                    singularities.push(Singularity {
                        kind: SingularityKind::Core,
                        x: x as f32,
                        y: y as f32,
                        strength: poincare_sum.abs() / PI,
                    });
                } else if poincare_sum < -delta_threshold {
                    singularities.push(Singularity {
                        kind: SingularityKind::Delta,
                        x: x as f32,
                        y: y as f32,
                        strength: poincare_sum.abs() / PI,
                    });
                }
            }
        }

        singularities
    }

    /// Match a partial fingerprint embedding against a full enrollment.
    ///
    /// Both `full` and `partial` are spatial hash embeddings (flat vectors).
    /// The method divides the full embedding into sub-regions and finds the
    /// best-matching sub-region for the partial embedding. This enables
    /// matching latent or partial prints against enrolled full prints.
    ///
    /// # Arguments
    /// * `full` - Full enrollment embedding (e.g. 128-dim)
    /// * `partial` - Partial print embedding (same dimensionality)
    /// * `overlap_threshold` - Minimum sub-region similarity to consider a match
    ///
    /// # Returns
    /// Best local similarity score in [0, 1].
    pub fn match_partial(full: &[f32], partial: &[f32], overlap_threshold: f32) -> f32 {
        if full.is_empty() || partial.is_empty() || full.len() != partial.len() {
            return 0.0;
        }

        let dim = full.len();

        // Divide embedding into sub-regions (4 equal chunks for spatial hash matching)
        let n_regions = 4.min(dim);
        let region_size = dim / n_regions;
        if region_size == 0 {
            return 0.0;
        }

        let mut best_score = 0.0f32;

        // For each sub-region of the partial print, find best match in full print
        for p_start in (0..dim).step_by(region_size) {
            let p_end = (p_start + region_size).min(dim);
            let partial_region = &partial[p_start..p_end];

            // Compare against every sub-region in the full embedding
            for f_start in (0..dim).step_by(region_size) {
                let f_end = (f_start + region_size).min(dim);
                let full_region = &full[f_start..f_end];

                if partial_region.len() != full_region.len() {
                    continue;
                }

                // Cosine similarity of this sub-region pair
                let mut dot = 0.0f32;
                let mut norm_p = 0.0f32;
                let mut norm_f = 0.0f32;

                for i in 0..partial_region.len() {
                    dot += partial_region[i] * full_region[i];
                    norm_p += partial_region[i] * partial_region[i];
                    norm_f += full_region[i] * full_region[i];
                }

                let denom = (norm_p.sqrt() * norm_f.sqrt()).max(1e-8);
                let sim = dot / denom;

                if sim > best_score {
                    best_score = sim;
                }
            }
        }

        // Also compute global similarity
        let mut dot = 0.0f32;
        let mut norm_p = 0.0f32;
        let mut norm_f = 0.0f32;
        for i in 0..dim {
            dot += partial[i] * full[i];
            norm_p += partial[i] * partial[i];
            norm_f += full[i] * full[i];
        }
        let global_sim = dot / (norm_p.sqrt() * norm_f.sqrt()).max(1e-8);

        // Combine local best with global, weighted toward local for partial prints
        let combined = 0.6 * best_score + 0.4 * global_sim;

        if combined >= overlap_threshold {
            combined
        } else {
            combined // Return score regardless; threshold is informational
        }
    }

    /// Measure the orientation consistency of the ridge flow field.
    ///
    /// Computes how smoothly the orientation changes across neighboring pixels.
    /// A high-quality fingerprint has smooth, slowly-varying orientation.
    /// Damaged, dry, or noisy prints have rapid, inconsistent orientation changes.
    ///
    /// Returns a scalar in [0, 1] where 1.0 = perfectly consistent,
    /// 0.0 = completely random orientations.
    pub fn orientation_consistency(&self, fingerprint: &Variable) -> f32 {
        let (responses, _batch, h, w) = self.gabor_responses(fingerprint);
        if h < 2 || w < 2 {
            return 0.0;
        }

        // Compute orientation field for first batch element only
        let field = Self::orientation_field(&responses, self.n_orientations, 1, h, w);

        // Measure consistency: average cosine similarity of orientation
        // between neighboring pixels (horizontal + vertical)
        let mut consistency_sum = 0.0f64;
        let mut count = 0u64;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let angle0 = field[idx].0;
                let mag0 = field[idx].1;

                // Skip low-magnitude (background) regions
                if mag0 < 1e-6 {
                    continue;
                }

                // Horizontal neighbor
                if x + 1 < w {
                    let idx_r = y * w + (x + 1);
                    let angle_r = field[idx_r].0;
                    let mag_r = field[idx_r].1;

                    if mag_r > 1e-6 {
                        // Orientation consistency = cos(2 * (angle_diff))
                        // Factor of 2 because ridge orientation has pi ambiguity
                        let diff = angle0 - angle_r;
                        let cos_val = (2.0 * diff).cos();
                        consistency_sum += cos_val as f64;
                        count += 1;
                    }
                }

                // Vertical neighbor
                if y + 1 < h {
                    let idx_d = (y + 1) * w + x;
                    let angle_d = field[idx_d].0;
                    let mag_d = field[idx_d].1;

                    if mag_d > 1e-6 {
                        let diff = angle0 - angle_d;
                        let cos_val = (2.0 * diff).cos();
                        consistency_sum += cos_val as f64;
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            return 0.0;
        }

        // Map from [-1, 1] average cosine to [0, 1] score
        let avg_cos = (consistency_sum / count as f64) as f32;
        ((avg_cos + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Full forward pass with embedding and uncertainty output.
    ///
    /// Input: [B, 1, 128, 128]
    /// Returns: (L2-normalized embedding [B, embed_dim], log_variance [B, 1])
    pub fn forward_full(&self, fingerprint: &Variable) -> (Variable, Variable) {
        let ridge_events = self.extract_ridge_events(fingerprint);

        // Field encoder
        let x = self.field_block1.forward(&ridge_events);
        let x = self.field_block2.forward(&x);
        let x = self.field_block3.forward(&x);

        // Spatial hash + adaptive pooling (graph-tracked via AdaptiveAvgPool2d)
        let x = self
            .spatial_bn
            .forward(&self.spatial_conv.forward(&x))
            .relu();
        let x = self.pool.forward(&x); // [B, 16, 4, 4]

        // Flatten using Variable::reshape
        let x_shape = x.shape();
        let batch = x_shape[0];
        let flat_dim = x_shape[1] * x_shape[2] * x_shape[3];
        let flat = x.reshape(&[batch, flat_dim]);

        // Projection
        let proj = self.proj1.forward(&flat).relu();
        let embedding = self.proj2.forward(&proj);

        // L2 normalize via scalar division (graph-tracked on embedding)
        let emb_data = embedding.data().to_vec();
        let norm_val: f32 = emb_data.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let norm_embedding = embedding.mul_scalar(1.0 / norm_val);

        // Uncertainty
        let uncertainty = self.uncertainty_head.forward(&flat);

        (norm_embedding, uncertainty)
    }

    /// Extract fingerprint identity embedding.
    ///
    /// Returns L2-normalized embedding as Vec<f32>.
    pub fn extract_identity(&self, fingerprint: &Variable) -> Vec<f32> {
        let (embedding, _logvar) = self.forward_full(fingerprint);
        embedding.data().to_vec()
    }

    /// Collect all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for filter in &self.gabor_filters {
            p.extend(filter.parameters());
        }
        p.extend(self.field_block1.parameters());
        p.extend(self.field_block2.parameters());
        p.extend(self.field_block3.parameters());
        p.extend(self.spatial_conv.parameters());
        p.extend(self.spatial_bn.parameters());
        p.extend(self.pool.parameters());
        p.extend(self.proj1.parameters());
        p.extend(self.proj2.parameters());
        p.extend(self.uncertainty_head.parameters());
        p
    }

    /// Get the output embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get the number of Gabor orientations.
    pub fn n_orientations(&self) -> usize {
        self.n_orientations
    }
}

impl Module for AriadneFingerprint {
    /// Forward: [B, 1, 128, 128] -> [B, embed_dim]
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
    // Existing tests — preserved
    // =========================================================================

    #[test]
    fn test_gabor_kernel_properties() {
        let kernel = gabor_kernel(7, 0.0, 2.0, 4.0, 0.0);
        assert_eq!(kernel.len(), 49);
        // Should be L1-normalized
        let l1: f32 = kernel.iter().map(|v| v.abs()).sum();
        assert!((l1 - 1.0).abs() < 0.01, "Not L1-normalized: {}", l1);
        // Should have both positive and negative values (sinusoidal)
        assert!(kernel.iter().any(|&v| v > 0.01), "No positive values");
        assert!(kernel.iter().any(|&v| v < -0.01), "No negative values");
    }

    #[test]
    fn test_gabor_orientations_differ() {
        let k0 = gabor_kernel(7, 0.0, 2.0, 4.0, 0.0);
        let k90 = gabor_kernel(7, PI / 2.0, 2.0, 4.0, 0.0);
        // Different orientations should produce different kernels
        let diff: f32 = k0.iter().zip(k90.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.1,
            "Orientations 0 and 90 degrees should differ: {}",
            diff
        );
    }

    #[test]
    fn test_ariadne_creation() {
        let model = AriadneFingerprint::new();
        assert_eq!(model.embed_dim(), 128);
        assert_eq!(model.n_orientations(), 8);
    }

    #[test]
    fn test_ariadne_param_count() {
        let model = AriadneFingerprint::new();
        let total: usize = model
            .parameters()
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum();
        assert!(total < 100_000, "Params {} exceeds 100K budget", total);
        assert!(total > 20_000, "Params {} seems too low", total);
    }

    #[test]
    fn test_ariadne_forward_shape() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), &[1, 128]);
    }

    #[test]
    fn test_ariadne_full_forward() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let (embedding, logvar) = model.forward_full(&input);
        assert_eq!(embedding.shape(), &[1, 128]);
        assert_eq!(logvar.shape(), &[1, 1]);
    }

    #[test]
    fn test_ariadne_embedding_normalized() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.3f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let identity = model.extract_identity(&input);
        let norm: f32 = identity.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding not unit norm: {}",
            norm
        );
    }

    #[test]
    fn test_ariadne_ridge_events_shape() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let events = model.extract_ridge_events(&input);
        assert_eq!(events.shape(), &[1, 2, 128, 128]);

        // Channel 0 (orientation) should be in [-1, 1]
        let data = events.data().to_vec();
        let spatial = 128 * 128;
        for i in 0..spatial {
            assert!(
                data[i] >= -1.0 && data[i] <= 1.0,
                "Orientation {} out of [-1,1]: {}",
                i,
                data[i]
            );
        }
    }

    // =========================================================================
    // Ridge density map tests
    // =========================================================================

    #[test]
    fn test_ridge_density_map_shape() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let density = model.ridge_density_map(&input);
        // 128 / 8 = 16 cells each direction
        assert_eq!(density.shape(), &[1, 1, 16, 16]);
    }

    #[test]
    fn test_ridge_density_map_non_negative() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.2f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let density = model.ridge_density_map(&input);
        let data = density.data().to_vec();
        for (i, &val) in data.iter().enumerate() {
            assert!(val >= 0.0, "Density cell {} is negative: {}", i, val);
        }
    }

    #[test]
    fn test_ridge_density_map_zero_input() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(Tensor::zeros(&[1, 1, 128, 128]), false);
        let density = model.ridge_density_map(&input);
        let data = density.data().to_vec();
        // Zero input should produce zero or near-zero density
        for &val in &data {
            assert!(val >= 0.0, "Density should be non-negative");
        }
    }

    // =========================================================================
    // Singularity detection tests
    // =========================================================================

    #[test]
    fn test_singularity_detection_returns_valid() {
        let model = AriadneFingerprint::new();
        // Create a non-trivial pattern: concentric ridge-like structure
        let mut data = vec![0.0f32; 128 * 128];
        for y in 0..128 {
            for x in 0..128 {
                let cx = x as f32 - 64.0;
                let cy = y as f32 - 64.0;
                // Whorl pattern: radial oscillation
                let r = (cx * cx + cy * cy).sqrt();
                data[y * 128 + x] = (r * 0.3).sin() * 0.5;
            }
        }
        let input = Variable::new(Tensor::from_vec(data, &[1, 1, 128, 128]).unwrap(), false);
        let singularities = model.detect_singularities(&input);
        // Validate all returned singularities have valid fields
        for s in &singularities {
            assert!(s.x >= 0.0 && s.x < 128.0, "x out of range: {}", s.x);
            assert!(s.y >= 0.0 && s.y < 128.0, "y out of range: {}", s.y);
            assert!(
                s.strength > 0.0,
                "strength should be positive: {}",
                s.strength
            );
            assert!(
                s.kind == SingularityKind::Core || s.kind == SingularityKind::Delta,
                "Invalid singularity kind"
            );
        }
    }

    #[test]
    fn test_singularity_detection_zero_input() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(Tensor::zeros(&[1, 1, 128, 128]), false);
        let singularities = model.detect_singularities(&input);
        // Zero input has no magnitude, so no singularities should be detected
        assert!(
            singularities.is_empty(),
            "Zero input should yield no singularities"
        );
    }

    #[test]
    fn test_singularity_detection_small_image() {
        let model = AriadneFingerprint::new();
        // Very small image (below kernel size, but padded conv handles it)
        let data = vec![0.5f32; 1 * 1 * 8 * 8];
        let input = Variable::new(Tensor::from_vec(data, &[1, 1, 8, 8]).unwrap(), false);
        let singularities = model.detect_singularities(&input);
        // Should not panic; may return empty or a few detections
        for s in &singularities {
            assert!(s.strength > 0.0);
        }
    }

    #[test]
    fn test_singularity_kind_equality() {
        assert_eq!(SingularityKind::Core, SingularityKind::Core);
        assert_eq!(SingularityKind::Delta, SingularityKind::Delta);
        assert_ne!(SingularityKind::Core, SingularityKind::Delta);
    }

    // =========================================================================
    // Partial matching tests
    // =========================================================================

    #[test]
    fn test_partial_match_identical() {
        let emb = vec![0.5f32; 128];
        let score = AriadneFingerprint::match_partial(&emb, &emb, 0.0);
        assert!(
            score > 0.95,
            "Identical embeddings should match highly: {}",
            score
        );
    }

    #[test]
    fn test_partial_match_random_low() {
        // Two very different embeddings
        let full: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let partial: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1 + 50.0).cos()).collect();
        let score = AriadneFingerprint::match_partial(&full, &partial, 0.0);
        // Dissimilar embeddings should score lower than identical
        let self_score = AriadneFingerprint::match_partial(&full, &full, 0.0);
        assert!(
            score < self_score,
            "Random should score lower than self: {} vs {}",
            score,
            self_score
        );
    }

    #[test]
    fn test_partial_match_empty() {
        assert_eq!(AriadneFingerprint::match_partial(&[], &[], 0.0), 0.0);
        assert_eq!(AriadneFingerprint::match_partial(&[1.0], &[], 0.0), 0.0);
    }

    #[test]
    fn test_partial_match_mismatched_length() {
        let a = vec![1.0f32; 64];
        let b = vec![1.0f32; 128];
        let score = AriadneFingerprint::match_partial(&a, &b, 0.0);
        assert_eq!(score, 0.0, "Mismatched lengths should return 0.0");
    }

    // =========================================================================
    // Orientation consistency tests
    // =========================================================================

    #[test]
    fn test_orientation_consistency_uniform() {
        let model = AriadneFingerprint::new();
        // Uniform constant input -> all Gabor responses same -> consistent orientation
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let score = model.orientation_consistency(&input);
        assert!(
            score > 0.5,
            "Uniform input should have high consistency: {}",
            score
        );
    }

    #[test]
    fn test_orientation_consistency_random_lower() {
        let model = AriadneFingerprint::new();
        // Random noise input should have lower consistency than uniform
        let mut data = vec![0.0f32; 128 * 128];
        let mut seed = 42u64;
        for val in data.iter_mut() {
            // Simple LCG pseudo-random
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        }
        let random_input = Variable::new(Tensor::from_vec(data, &[1, 1, 128, 128]).unwrap(), false);
        let uniform_input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );

        let random_score = model.orientation_consistency(&random_input);
        let uniform_score = model.orientation_consistency(&uniform_input);

        assert!(
            random_score < uniform_score,
            "Random should be less consistent than uniform: {} vs {}",
            random_score,
            uniform_score
        );
    }

    #[test]
    fn test_orientation_consistency_zero_input() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(Tensor::zeros(&[1, 1, 128, 128]), false);
        let score = model.orientation_consistency(&input);
        // Zero magnitude means no foreground; score should be 0 or very low
        assert!(
            score >= 0.0 && score <= 1.0,
            "Score out of [0,1]: {}",
            score
        );
    }

    #[test]
    fn test_orientation_consistency_in_range() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.3f32; 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let score = model.orientation_consistency(&input);
        assert!(score >= 0.0 && score <= 1.0, "Score {} not in [0,1]", score);
    }

    // =========================================================================
    // Gabor kernel extended tests
    // =========================================================================

    #[test]
    fn test_gabor_kernel_symmetry_horizontal() {
        // theta=0 should produce vertical symmetry (symmetric about horizontal center line)
        let kernel = gabor_kernel(7, 0.0, 2.0, 4.0, 0.0);
        // For theta=0, the Gaussian is rotationally symmetric and the sinusoid
        // depends only on x. So rows should be symmetric top-to-bottom.
        for y in 0..3 {
            for x in 0..7 {
                let top = kernel[y * 7 + x];
                let bot = kernel[(6 - y) * 7 + x];
                assert!(
                    (top - bot).abs() < 1e-5,
                    "Vertical symmetry broken at ({}, {}): {} vs {}",
                    x,
                    y,
                    top,
                    bot
                );
            }
        }
    }

    #[test]
    fn test_gabor_kernel_different_sigma() {
        let k_narrow = gabor_kernel(7, 0.0, 1.0, 4.0, 0.0);
        let k_wide = gabor_kernel(7, 0.0, 3.0, 4.0, 0.0);
        // Different sigma should produce different kernels
        let diff: f32 = k_narrow
            .iter()
            .zip(k_wide.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.05,
            "Different sigma should produce different kernels: {}",
            diff
        );
    }

    #[test]
    fn test_gabor_kernel_different_lambda() {
        let k1 = gabor_kernel(7, 0.0, 2.0, 3.0, 0.0);
        let k2 = gabor_kernel(7, 0.0, 2.0, 6.0, 0.0);
        let diff: f32 = k1.iter().zip(k2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.05,
            "Different lambda should produce different kernels: {}",
            diff
        );
    }

    #[test]
    fn test_gabor_kernel_l1_normalized_various_orientations() {
        for i in 0..8 {
            let theta = (i as f32) * PI / 8.0;
            let kernel = gabor_kernel(7, theta, 2.0, 4.0, 0.0);
            let l1: f32 = kernel.iter().map(|v| v.abs()).sum();
            assert!(
                (l1 - 1.0).abs() < 0.01,
                "Orientation {} not L1-normalized: {}",
                i,
                l1
            );
        }
    }

    // =========================================================================
    // Batch processing tests
    // =========================================================================

    #[test]
    fn test_batch_forward_shape() {
        let model = AriadneFingerprint::new();
        let batch_size = 3;
        let input = Variable::new(
            Tensor::from_vec(
                vec![0.4f32; batch_size * 1 * 128 * 128],
                &[batch_size, 1, 128, 128],
            )
            .unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), &[batch_size, 128]);
    }

    #[test]
    fn test_batch_ridge_density_map() {
        let model = AriadneFingerprint::new();
        let batch_size = 2;
        let input = Variable::new(
            Tensor::from_vec(
                vec![0.3f32; batch_size * 1 * 128 * 128],
                &[batch_size, 1, 128, 128],
            )
            .unwrap(),
            false,
        );
        let density = model.ridge_density_map(&input);
        assert_eq!(density.shape(), &[batch_size, 1, 16, 16]);
    }

    // =========================================================================
    // Numerical stability tests
    // =========================================================================

    #[test]
    fn test_forward_large_values_no_nan() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![100.0f32; 1 * 1 * 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let (embedding, logvar) = model.forward_full(&input);
        let emb_data = embedding.data().to_vec();
        let lv_data = logvar.data().to_vec();
        for &v in &emb_data {
            assert!(!v.is_nan(), "NaN in embedding with large input");
            assert!(!v.is_infinite(), "Inf in embedding with large input");
        }
        for &v in &lv_data {
            assert!(!v.is_nan(), "NaN in log_variance with large input");
        }
    }

    #[test]
    fn test_orientation_consistency_large_values() {
        let model = AriadneFingerprint::new();
        let input = Variable::new(
            Tensor::from_vec(vec![50.0f32; 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let score = model.orientation_consistency(&input);
        assert!(
            !score.is_nan(),
            "NaN orientation consistency with large values"
        );
        assert!(score >= 0.0 && score <= 1.0, "Score {} out of [0,1]", score);
    }

    // =========================================================================
    // Embedding space properties
    // =========================================================================

    #[test]
    fn test_different_inputs_different_embeddings() {
        let model = AriadneFingerprint::new();

        let input_a = Variable::new(
            Tensor::from_vec(vec![0.2f32; 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let input_b = Variable::new(
            Tensor::from_vec(vec![0.8f32; 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );

        let emb_a = model.extract_identity(&input_a);
        let emb_b = model.extract_identity(&input_b);

        // Different inputs should produce different embeddings
        let diff: f32 = emb_a
            .iter()
            .zip(emb_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "Different inputs should produce different embeddings, diff: {}",
            diff
        );
    }

    #[test]
    fn test_same_input_same_embedding() {
        let model = AriadneFingerprint::new();

        let data = vec![0.4f32; 128 * 128];
        let input1 = Variable::new(
            Tensor::from_vec(data.clone(), &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let input2 = Variable::new(Tensor::from_vec(data, &[1, 1, 128, 128]).unwrap(), false);

        let emb1 = model.extract_identity(&input1);
        let emb2 = model.extract_identity(&input2);

        let diff: f32 = emb1
            .iter()
            .zip(emb2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 1e-4,
            "Same input should produce same embedding, diff: {}",
            diff
        );
    }

    #[test]
    fn test_embedding_dimensionality() {
        let model = AriadneFingerprint::with_config(8, 64);
        assert_eq!(model.embed_dim(), 64);

        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 128 * 128], &[1, 1, 128, 128]).unwrap(),
            false,
        );
        let emb = model.extract_identity(&input);
        assert_eq!(emb.len(), 64);
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_very_small_print() {
        let model = AriadneFingerprint::new();
        // Minimum viable size (Conv2d with 7x7 kernel needs >= 7x7 with padding 3)
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 1 * 1 * 8 * 8], &[1, 1, 8, 8]).unwrap(),
            false,
        );
        // Should not panic
        let events = model.extract_ridge_events(&input);
        assert_eq!(events.shape()[0], 1);
        assert_eq!(events.shape()[1], 2);
    }

    #[test]
    fn test_ridge_density_small_image() {
        let model = AriadneFingerprint::new();
        // 16x16 image: cells = 16/8 = 2x2
        let input = Variable::new(
            Tensor::from_vec(vec![0.5f32; 16 * 16], &[1, 1, 16, 16]).unwrap(),
            false,
        );
        let density = model.ridge_density_map(&input);
        assert_eq!(density.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_partial_match_single_element() {
        let a = vec![1.0f32];
        let score = AriadneFingerprint::match_partial(&a, &a, 0.0);
        assert!(
            score > 0.9,
            "Single-element identical match should be high: {}",
            score
        );
    }
}
