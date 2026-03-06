//! Polar Unwrap Utilities for Iris Processing
//!
//! Geometric transformations for converting iris images from Cartesian
//! to polar coordinates. Includes quality-weighted unwrap, multi-scale
//! unwrap, angular histograms, and radial profile extraction.
//!
//! @version 0.2.0

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// =============================================================================
// Polar Unwrap Configuration
// =============================================================================

/// Configuration for iris polar unwrap.
#[derive(Debug, Clone)]
pub struct PolarUnwrapConfig {
    /// Number of radial bins (height of output strip).
    pub radial_bins: usize,
    /// Number of angular bins (width of output strip).
    pub angular_bins: usize,
    /// Center X of pupil (normalized 0-1).
    pub center_x: f32,
    /// Center Y of pupil (normalized 0-1).
    pub center_y: f32,
    /// Inner radius (pupil boundary, normalized).
    pub inner_radius: f32,
    /// Outer radius (iris boundary, normalized).
    pub outer_radius: f32,
}

impl Default for PolarUnwrapConfig {
    fn default() -> Self {
        Self {
            radial_bins: 32,
            angular_bins: 256,
            center_x: 0.5,
            center_y: 0.5,
            inner_radius: 0.15,
            outer_radius: 0.45,
        }
    }
}

impl PolarUnwrapConfig {
    /// Create config for a detected pupil/iris boundary.
    pub fn from_detection(
        center_x: f32,
        center_y: f32,
        pupil_radius: f32,
        iris_radius: f32,
    ) -> Self {
        Self {
            center_x,
            center_y,
            inner_radius: pupil_radius,
            outer_radius: iris_radius,
            ..Default::default()
        }
    }

    /// High-resolution unwrap for detailed analysis.
    pub fn high_res() -> Self {
        Self {
            radial_bins: 64,
            angular_bins: 512,
            ..Default::default()
        }
    }

    /// Low-resolution unwrap for fast edge processing.
    pub fn low_res() -> Self {
        Self {
            radial_bins: 16,
            angular_bins: 128,
            ..Default::default()
        }
    }
}

// =============================================================================
// Core Polar Unwrap
// =============================================================================

/// Unwrap an iris image from Cartesian to polar coordinates.
///
/// Input: grayscale iris image [B, 1, H, W]
/// Output: polar strip [B, 1, radial_bins, angular_bins]
///
/// The unwrap samples from the annulus between inner_radius and outer_radius,
/// producing a rectangular strip where:
/// - Rows = radial position (pupil -> iris edge)
/// - Columns = angular position (0 -> 2pi)
pub fn polar_unwrap(image: &Variable, config: &PolarUnwrapConfig) -> Variable {
    let shape = image.shape();
    let (batch, _ch, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let data = image.data().to_vec();

    let rb = config.radial_bins;
    let ab = config.angular_bins;
    let mut output = vec![0.0f32; batch * rb * ab];

    let h_f = h as f32;
    let w_f = w as f32;
    let cx = config.center_x * w_f;
    let cy = config.center_y * h_f;
    let r_inner = config.inner_radius * w_f.min(h_f);
    let r_outer = config.outer_radius * w_f.min(h_f);

    for b in 0..batch {
        for ri in 0..rb {
            let r = r_inner + (ri as f32 / (rb - 1).max(1) as f32) * (r_outer - r_inner);

            for ai in 0..ab {
                let theta = 2.0 * std::f32::consts::PI * (ai as f32) / (ab as f32);
                let sx = cx + r * theta.cos();
                let sy = cy + r * theta.sin();
                let val = bilinear_sample(&data, b, h, w, sx, sy);
                output[b * rb * ab + ri * ab + ai] = val;
            }
        }
    }

    Variable::new(
        Tensor::from_vec(output, &[batch, 1, rb, ab]).unwrap(),
        false,
    )
}

/// Bilinear interpolation sampling from a single-channel image.
fn bilinear_sample(data: &[f32], batch_idx: usize, h: usize, w: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let base = batch_idx * h * w;

    let get = |yi: isize, xi: isize| -> f32 {
        if yi >= 0 && yi < h as isize && xi >= 0 && xi < w as isize {
            data[base + yi as usize * w + xi as usize]
        } else {
            0.0
        }
    };

    let v00 = get(y0, x0);
    let v01 = get(y0, x1);
    let v10 = get(y1, x0);
    let v11 = get(y1, x1);

    v00 * (1.0 - fx) * (1.0 - fy) + v01 * fx * (1.0 - fy) + v10 * (1.0 - fx) * fy + v11 * fx * fy
}

// =============================================================================
// Circular Shift
// =============================================================================

/// Circular shift a polar strip along the angular axis.
/// Used for rotation-invariant matching: rotated iris = circularly shifted code.
///
/// Input: [B, C, R, A]
/// Output: [B, C, R, A] shifted by `shift` positions along the last dim.
pub fn circular_shift(strip: &Variable, shift: isize) -> Variable {
    let shape = strip.shape();
    let (batch, ch, r, a) = (shape[0], shape[1], shape[2], shape[3]);
    let data = strip.data().to_vec();

    let mut shifted = vec![0.0f32; data.len()];
    for b in 0..batch {
        for c in 0..ch {
            for ri in 0..r {
                for ai in 0..a {
                    let src_ai = ((ai as isize - shift).rem_euclid(a as isize)) as usize;
                    let dst = b * ch * r * a + c * r * a + ri * a + ai;
                    let src = b * ch * r * a + c * r * a + ri * a + src_ai;
                    shifted[dst] = data[src];
                }
            }
        }
    }

    Variable::new(
        Tensor::from_vec(shifted, &[batch, ch, r, a]).unwrap(),
        false,
    )
}

// =============================================================================
// Normalized Polar Unwrap
// =============================================================================

/// Polar unwrap with per-ring intensity normalization.
///
/// Each radial ring is independently normalized to zero mean and unit variance.
/// This compensates for radial intensity gradients common in iris images
/// (lighter near pupil, darker toward limbus).
///
/// Input: grayscale iris image [B, 1, H, W]
/// Output: normalized polar strip [B, 1, radial_bins, angular_bins]
pub fn normalized_polar_unwrap(image: &Variable, config: &PolarUnwrapConfig) -> Variable {
    let raw = polar_unwrap(image, config);
    let shape = raw.shape();
    let (batch, _ch, rb, ab) = (shape[0], shape[1], shape[2], shape[3]);
    let mut data = raw.data().to_vec();

    for b in 0..batch {
        for ri in 0..rb {
            let base = b * rb * ab + ri * ab;
            let ring = &data[base..base + ab];

            let mean: f32 = ring.iter().sum::<f32>() / ab as f32;
            let var: f32 = ring.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / ab as f32;
            let std = (var + 1e-8).sqrt();

            for ai in 0..ab {
                data[base + ai] = (data[base + ai] - mean) / std;
            }
        }
    }

    Variable::new(
        Tensor::from_vec(data, &[batch, 1, rb, ab]).unwrap(),
        false,
    )
}

// =============================================================================
// Angular Histogram
// =============================================================================

/// Compute angular intensity histogram from a polar strip.
///
/// Summarizes the angular distribution of iris texture at each radial level.
/// Useful for rotation estimation and quality assessment.
///
/// Input: polar strip [B, 1, R, A]
/// Output: histogram [B, n_bins] (aggregated across radial levels)
pub fn angular_histogram(strip: &Variable, n_bins: usize) -> Vec<f32> {
    let shape = strip.shape();
    let (batch, _ch, rb, ab) = (shape[0], shape[1], shape[2], shape[3]);
    let data = strip.data().to_vec();

    let mut hist = vec![0.0f32; batch * n_bins];
    let bin_width = ab / n_bins;

    for b in 0..batch {
        for ri in 0..rb {
            for ai in 0..ab {
                let bin = (ai / bin_width.max(1)).min(n_bins - 1);
                let val = data[b * rb * ab + ri * ab + ai].abs();
                hist[b * n_bins + bin] += val;
            }
        }
    }

    // Normalize per-batch
    for b in 0..batch {
        let total: f32 = hist[b * n_bins..(b + 1) * n_bins].iter().sum();
        if total > 1e-8 {
            for i in 0..n_bins {
                hist[b * n_bins + i] /= total;
            }
        }
    }

    hist
}

// =============================================================================
// Radial Profile
// =============================================================================

/// Extract radial intensity profile from a polar strip.
///
/// Averages intensity along the angular axis at each radial position.
/// The radial profile captures the pupil-to-limbus gradient and can be
/// used for quality assessment (well-focused iris has clear radial gradient).
///
/// Input: polar strip [B, 1, R, A]
/// Output: radial profile [B, R]
pub fn radial_profile(strip: &Variable) -> Vec<f32> {
    let shape = strip.shape();
    let (batch, _ch, rb, ab) = (shape[0], shape[1], shape[2], shape[3]);
    let data = strip.data().to_vec();

    let mut profile = vec![0.0f32; batch * rb];

    for b in 0..batch {
        for ri in 0..rb {
            let base = b * rb * ab + ri * ab;
            let mean: f32 = data[base..base + ab].iter().sum::<f32>() / ab as f32;
            profile[b * rb + ri] = mean;
        }
    }

    profile
}

/// Compute radial contrast (std of radial profile).
/// Higher contrast = better-focused iris with clear structure.
pub fn radial_contrast(strip: &Variable) -> f32 {
    let profile = radial_profile(strip);
    if profile.is_empty() {
        return 0.0;
    }
    let mean: f32 = profile.iter().sum::<f32>() / profile.len() as f32;
    let var: f32 = profile.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / profile.len() as f32;
    var.sqrt()
}

// =============================================================================
// Quality Assessment
// =============================================================================

/// Assess quality of a polar iris strip.
///
/// Returns a quality score [0, 1] based on:
/// - Radial contrast (higher = better focus)
/// - Angular uniformity (detect eyelid occlusion)
/// - Overall intensity range
pub fn assess_polar_quality(strip: &Variable) -> f32 {
    let shape = strip.shape();
    let (batch, _ch, rb, ab) = (shape[0], shape[1], shape[2], shape[3]);
    let data = strip.data().to_vec();

    let mut quality = 0.0f32;

    for b in 0..batch {
        // 1. Radial contrast score (0-1)
        let base = b * rb * ab;
        let slice = &data[base..base + rb * ab];
        let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-8);
        let contrast_score = (range / 2.0).min(1.0);

        // 2. Angular coverage score: detect if large angular sectors are missing (occlusion)
        let mut sector_means = vec![0.0f32; 8]; // 8 sectors
        let sector_size = ab / 8;
        for si in 0..8 {
            let mut sum = 0.0f32;
            for ri in 0..rb {
                for ai in 0..sector_size {
                    let idx = base + ri * ab + si * sector_size + ai;
                    sum += data[idx].abs();
                }
            }
            sector_means[si] = sum / (rb * sector_size) as f32;
        }
        let sector_mean: f32 = sector_means.iter().sum::<f32>() / 8.0;
        let min_sector = sector_means.iter().cloned().fold(f32::INFINITY, f32::min);
        // If any sector is much lower than mean, probably occluded
        let coverage_score = if sector_mean > 1e-8 {
            (min_sector / sector_mean).min(1.0)
        } else {
            0.0
        };

        // 3. Non-zero coverage
        let nonzero = slice.iter().filter(|&&v| v.abs() > 1e-6).count();
        let coverage = nonzero as f32 / slice.len() as f32;

        quality += (contrast_score * 0.4 + coverage_score * 0.3 + coverage * 0.3) / batch as f32;
    }

    quality
}

// =============================================================================
// Multi-Scale Unwrap
// =============================================================================

/// Multi-scale polar unwrap for hierarchical iris encoding.
///
/// Returns polar strips at 3 resolutions:
/// - Coarse: [B, 1, 8, 64] — captures global iris structure
/// - Medium: [B, 1, 16, 128] — captures main crypts and furrows
/// - Fine: [B, 1, 32, 256] — captures detailed texture
pub fn multi_scale_unwrap(image: &Variable, config: &PolarUnwrapConfig) -> (Variable, Variable, Variable) {
    let coarse_config = PolarUnwrapConfig {
        radial_bins: 8,
        angular_bins: 64,
        ..config.clone()
    };
    let medium_config = PolarUnwrapConfig {
        radial_bins: 16,
        angular_bins: 128,
        ..config.clone()
    };

    let coarse = polar_unwrap(image, &coarse_config);
    let medium = polar_unwrap(image, &medium_config);
    let fine = polar_unwrap(image, config);

    (coarse, medium, fine)
}

// =============================================================================
// Rotation Estimation
// =============================================================================

/// Estimate the angular rotation between two polar strips.
///
/// Uses cross-correlation along the angular axis to find the best alignment.
/// Returns the estimated rotation in angular bin units.
pub fn estimate_rotation(strip_a: &Variable, strip_b: &Variable, max_shift: usize) -> (isize, f32) {
    let shape_a = strip_a.shape();
    let shape_b = strip_b.shape();
    assert_eq!(shape_a, shape_b, "Strips must have same shape");

    let (_batch, _ch, rb, ab) = (shape_a[0], shape_a[1], shape_a[2], shape_a[3]);
    let data_a = strip_a.data().to_vec();
    let data_b = strip_b.data().to_vec();

    let mut best_shift: isize = 0;
    let mut best_corr = f32::NEG_INFINITY;

    for s in -(max_shift as isize)..=(max_shift as isize) {
        let mut corr = 0.0f32;
        for ri in 0..rb {
            for ai in 0..ab {
                let ai_shifted = ((ai as isize + s).rem_euclid(ab as isize)) as usize;
                corr += data_a[ri * ab + ai] * data_b[ri * ab + ai_shifted];
            }
        }
        if corr > best_corr {
            best_corr = corr;
            best_shift = s;
        }
    }

    (best_shift, best_corr)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_iris(val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 1 * 1 * 64 * 64], &[1, 1, 64, 64]).unwrap(),
            false,
        )
    }

    fn make_strip(val: f32) -> Variable {
        Variable::new(
            Tensor::from_vec(vec![val; 1 * 1 * 32 * 256], &[1, 1, 32, 256]).unwrap(),
            false,
        )
    }

    #[test]
    fn test_polar_unwrap_shape() {
        let config = PolarUnwrapConfig::default();
        let polar = polar_unwrap(&make_iris(0.5), &config);
        assert_eq!(polar.shape(), &[1, 1, 32, 256]);
    }

    #[test]
    fn test_polar_unwrap_high_res() {
        let config = PolarUnwrapConfig::high_res();
        let polar = polar_unwrap(&make_iris(0.5), &config);
        assert_eq!(polar.shape(), &[1, 1, 64, 512]);
    }

    #[test]
    fn test_polar_unwrap_low_res() {
        let config = PolarUnwrapConfig::low_res();
        let polar = polar_unwrap(&make_iris(0.5), &config);
        assert_eq!(polar.shape(), &[1, 1, 16, 128]);
    }

    #[test]
    fn test_polar_unwrap_center_pixel() {
        let mut data = vec![0.0f32; 64 * 64];
        data[32 * 64 + 32] = 1.0;
        let image = Variable::new(Tensor::from_vec(data, &[1, 1, 64, 64]).unwrap(), false);
        let config = PolarUnwrapConfig::default();
        let polar = polar_unwrap(&image, &config);
        assert_eq!(polar.shape(), &[1, 1, 32, 256]);
    }

    #[test]
    fn test_polar_unwrap_custom_detection() {
        let config = PolarUnwrapConfig::from_detection(0.4, 0.6, 0.1, 0.35);
        assert_eq!(config.center_x, 0.4);
        assert_eq!(config.center_y, 0.6);
        let polar = polar_unwrap(&make_iris(0.5), &config);
        assert_eq!(polar.shape(), &[1, 1, 32, 256]);
    }

    #[test]
    fn test_polar_unwrap_uniform_input() {
        let polar = polar_unwrap(&make_iris(0.7), &PolarUnwrapConfig::default());
        let data = polar.data().to_vec();
        // Uniform input should produce approximately uniform output
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 0.7).abs() < 0.2, "Mean should be near 0.7: {}", mean);
    }

    #[test]
    fn test_circular_shift_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let strip = Variable::new(Tensor::from_vec(data.clone(), &[1, 1, 1, 4]).unwrap(), false);
        let shifted = circular_shift(&strip, 0);
        assert_eq!(shifted.data().to_vec(), data);
    }

    #[test]
    fn test_circular_shift_by_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let strip = Variable::new(Tensor::from_vec(data, &[1, 1, 1, 4]).unwrap(), false);
        let shifted = circular_shift(&strip, 1);
        assert_eq!(shifted.data().to_vec(), vec![4.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_circular_shift_full_cycle() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let strip = Variable::new(Tensor::from_vec(data.clone(), &[1, 1, 1, 4]).unwrap(), false);
        let shifted = circular_shift(&strip, 4);
        assert_eq!(shifted.data().to_vec(), data);
    }

    #[test]
    fn test_circular_shift_negative() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let strip = Variable::new(Tensor::from_vec(data, &[1, 1, 1, 4]).unwrap(), false);
        let shifted = circular_shift(&strip, -1);
        assert_eq!(shifted.data().to_vec(), vec![2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn test_normalized_polar_unwrap_shape() {
        let config = PolarUnwrapConfig::default();
        let polar = normalized_polar_unwrap(&make_iris(0.5), &config);
        assert_eq!(polar.shape(), &[1, 1, 32, 256]);
    }

    #[test]
    fn test_normalized_polar_unwrap_zero_mean_rings() {
        let config = PolarUnwrapConfig::default();
        let polar = normalized_polar_unwrap(&make_iris(0.5), &config);
        let data = polar.data().to_vec();
        let ab = 256;
        // Each radial ring should have approximately zero mean
        for ri in 0..32 {
            let ring = &data[ri * ab..(ri + 1) * ab];
            let mean: f32 = ring.iter().sum::<f32>() / ab as f32;
            assert!(mean.abs() < 0.1, "Ring {} mean should be ~0: {}", ri, mean);
        }
    }

    #[test]
    fn test_angular_histogram_shape() {
        let strip = make_strip(0.5);
        let hist = angular_histogram(&strip, 8);
        assert_eq!(hist.len(), 8);
    }

    #[test]
    fn test_angular_histogram_sums_to_one() {
        let strip = make_strip(0.5);
        let hist = angular_histogram(&strip, 16);
        let total: f32 = hist.iter().sum();
        assert!((total - 1.0).abs() < 0.01, "Histogram should sum to 1: {}", total);
    }

    #[test]
    fn test_angular_histogram_uniform_input() {
        let strip = make_strip(0.5);
        let hist = angular_histogram(&strip, 4);
        // Uniform input should give uniform histogram
        for &h in &hist {
            assert!((h - 0.25).abs() < 0.05, "Should be ~0.25: {}", h);
        }
    }

    #[test]
    fn test_radial_profile_shape() {
        let strip = make_strip(0.5);
        let profile = radial_profile(&strip);
        assert_eq!(profile.len(), 32);
    }

    #[test]
    fn test_radial_profile_uniform() {
        let strip = make_strip(0.5);
        let profile = radial_profile(&strip);
        for &v in &profile {
            assert!((v - 0.5).abs() < 0.01, "Should be ~0.5: {}", v);
        }
    }

    #[test]
    fn test_radial_contrast_uniform() {
        let strip = make_strip(0.5);
        let contrast = radial_contrast(&strip);
        assert!(contrast < 0.01, "Uniform strip should have low contrast: {}", contrast);
    }

    #[test]
    fn test_radial_contrast_gradient() {
        // Create strip with radial gradient
        let rb = 32;
        let ab = 256;
        let mut data = vec![0.0f32; rb * ab];
        for ri in 0..rb {
            for ai in 0..ab {
                data[ri * ab + ai] = ri as f32 / rb as f32;
            }
        }
        let strip = Variable::new(Tensor::from_vec(data, &[1, 1, rb, ab]).unwrap(), false);
        let contrast = radial_contrast(&strip);
        assert!(contrast > 0.1, "Gradient should have high contrast: {}", contrast);
    }

    #[test]
    fn test_assess_polar_quality_good() {
        // Create strip with reasonable variation
        let rb = 32;
        let ab = 256;
        let mut data = vec![0.0f32; rb * ab];
        for ri in 0..rb {
            for ai in 0..ab {
                data[ri * ab + ai] = 0.3 + 0.4 * ((ai as f32 * 0.1).sin() * (ri as f32 * 0.2).cos());
            }
        }
        let strip = Variable::new(Tensor::from_vec(data, &[1, 1, rb, ab]).unwrap(), false);
        let quality = assess_polar_quality(&strip);
        assert!(quality > 0.0 && quality <= 1.0, "Quality out of range: {}", quality);
    }

    #[test]
    fn test_assess_polar_quality_blank() {
        let strip = make_strip(0.0);
        let quality = assess_polar_quality(&strip);
        assert!(quality < 0.5, "Blank strip should have low quality: {}", quality);
    }

    #[test]
    fn test_multi_scale_unwrap_shapes() {
        let config = PolarUnwrapConfig::default();
        let (coarse, medium, fine) = multi_scale_unwrap(&make_iris(0.5), &config);
        assert_eq!(coarse.shape(), &[1, 1, 8, 64]);
        assert_eq!(medium.shape(), &[1, 1, 16, 128]);
        assert_eq!(fine.shape(), &[1, 1, 32, 256]);
    }

    #[test]
    fn test_estimate_rotation_zero_shift() {
        // Use a non-uniform strip so cross-correlation is meaningful
        let ab = 256;
        let rb = 32;
        let mut data = vec![0.0f32; rb * ab];
        for ri in 0..rb {
            for ai in 0..ab {
                data[ri * ab + ai] = (ai as f32 * 0.1).sin();
            }
        }
        let strip = Variable::new(Tensor::from_vec(data, &[1, 1, rb, ab]).unwrap(), false);
        let (shift, _corr) = estimate_rotation(&strip, &strip, 16);
        assert_eq!(shift, 0, "Same strip should have zero rotation");
    }

    #[test]
    fn test_estimate_rotation_known_shift() {
        // Create a non-uniform strip
        let ab = 256;
        let rb = 32;
        let mut data = vec![0.0f32; rb * ab];
        for ri in 0..rb {
            for ai in 0..ab {
                data[ri * ab + ai] = (ai as f32 * 0.05).sin();
            }
        }
        let strip_a = Variable::new(Tensor::from_vec(data, &[1, 1, rb, ab]).unwrap(), false);
        let strip_b = circular_shift(&strip_a, 5);
        let (shift, _corr) = estimate_rotation(&strip_a, &strip_b, 16);
        assert!((shift - 5).abs() <= 1, "Should detect shift of ~5, got {}", shift);
    }

    #[test]
    fn test_bilinear_sample_center() {
        let data = vec![0.0, 0.0, 0.0, 1.0]; // 2x2 image, bottom-right = 1
        let val = bilinear_sample(&data, 0, 2, 2, 0.5, 0.5);
        assert!((val - 0.25).abs() < 0.01, "Center of 2x2 should be 0.25: {}", val);
    }

    #[test]
    fn test_bilinear_sample_out_of_bounds() {
        let data = vec![1.0; 4]; // 2x2
        let val = bilinear_sample(&data, 0, 2, 2, -1.0, -1.0);
        assert!(val.is_finite(), "Out-of-bounds should produce finite value");
    }
}
