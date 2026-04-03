//! Transforms - Data Augmentation and Preprocessing
//!
//! # File
//! `crates/axonml-data/src/transforms.rs`
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

use axonml_tensor::Tensor;
use rand::Rng;

// =============================================================================
// Transform Trait
// =============================================================================

/// Trait for data transformations.
pub trait Transform: Send + Sync {
    /// Applies the transform to a tensor.
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32>;
}

// =============================================================================
// Compose
// =============================================================================

/// Composes multiple transforms into a single transform.
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    /// Creates a new Compose from a vector of transforms.
    #[must_use]
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    /// Creates an empty Compose.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Adds a transform to the composition.
    pub fn add<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }
}

impl Transform for Compose {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut result = input.clone();
        for transform in &self.transforms {
            result = transform.apply(&result);
        }
        result
    }
}

// =============================================================================
// ToTensor
// =============================================================================

/// Converts input to a tensor (identity for already-tensor inputs).
pub struct ToTensor;

impl ToTensor {
    /// Creates a new `ToTensor` transform.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for ToTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for ToTensor {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        input.clone()
    }
}

// =============================================================================
// Normalize
// =============================================================================

/// Normalizes a tensor with mean and standard deviation.
///
/// Supports both scalar normalization (applied uniformly) and per-channel
/// normalization (PyTorch-style `transforms.Normalize(mean=[...], std=[...])`).
/// For per-channel mode, the tensor is expected to have shape `[C, H, W]` or
/// `[N, C, H, W]`.
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    /// Creates a new scalar Normalize transform (applied uniformly to all elements).
    #[must_use]
    pub fn new(mean: f32, std: f32) -> Self {
        Self {
            mean: vec![mean],
            std: vec![std],
        }
    }

    /// Creates a per-channel Normalize transform (PyTorch-compatible).
    ///
    /// For a `[C, H, W]` tensor, `mean` and `std` must have length `C`.
    /// Each channel is normalized independently: `output[c] = (input[c] - mean[c]) / std[c]`.
    #[must_use]
    pub fn per_channel(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(mean.len(), std.len(), "mean and std must have same length");
        Self { mean, std }
    }

    /// Creates a Normalize for standard normal distribution (mean=0, std=1).
    #[must_use]
    pub fn standard() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Creates a Normalize for [0,1] to [-1,1] conversion (mean=0.5, std=0.5).
    #[must_use]
    pub fn zero_centered() -> Self {
        Self::new(0.5, 0.5)
    }

    /// Creates a Normalize for ImageNet (3-channel RGB).
    #[must_use]
    pub fn imagenet() -> Self {
        Self::per_channel(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
    }
}

impl Transform for Normalize {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();
        let mut data = input.to_vec();

        if self.mean.len() == 1 {
            // Scalar normalization — apply uniformly
            let m = self.mean[0];
            let s = self.std[0];
            for x in &mut data {
                *x = (*x - m) / s;
            }
        } else {
            // Per-channel normalization
            let num_channels = self.mean.len();

            if shape.len() == 3 && shape[0] == num_channels {
                // [C, H, W]
                let spatial = shape[1] * shape[2];
                for c in 0..num_channels {
                    let offset = c * spatial;
                    let m = self.mean[c];
                    let s = self.std[c];
                    for i in 0..spatial {
                        data[offset + i] = (data[offset + i] - m) / s;
                    }
                }
            } else if shape.len() == 4 && shape[1] == num_channels {
                // [N, C, H, W]
                let spatial = shape[2] * shape[3];
                let sample_size = num_channels * spatial;
                for n in 0..shape[0] {
                    for c in 0..num_channels {
                        let offset = n * sample_size + c * spatial;
                        let m = self.mean[c];
                        let s = self.std[c];
                        for i in 0..spatial {
                            data[offset + i] = (data[offset + i] - m) / s;
                        }
                    }
                }
            } else {
                // Fallback: apply first channel's mean/std uniformly
                let m = self.mean[0];
                let s = self.std[0];
                for x in &mut data {
                    *x = (*x - m) / s;
                }
            }
        }

        Tensor::from_vec(data, shape).unwrap()
    }
}

// =============================================================================
// RandomNoise
// =============================================================================

/// Adds random Gaussian noise to the input.
pub struct RandomNoise {
    std: f32,
}

impl RandomNoise {
    /// Creates a new `RandomNoise` transform.
    #[must_use]
    pub fn new(std: f32) -> Self {
        Self { std }
    }
}

impl Transform for RandomNoise {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        if self.std == 0.0 {
            return input.clone();
        }

        let mut rng = rand::thread_rng();
        let data = input.to_vec();
        let noisy: Vec<f32> = data
            .iter()
            .map(|&x| {
                // Box-Muller transform for Gaussian noise
                let u1: f32 = rng.r#gen();
                let u2: f32 = rng.r#gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                x + z * self.std
            })
            .collect();
        Tensor::from_vec(noisy, input.shape()).unwrap()
    }
}

// =============================================================================
// RandomCrop
// =============================================================================

/// Randomly crops a portion of the input.
pub struct RandomCrop {
    size: Vec<usize>,
}

impl RandomCrop {
    /// Creates a new `RandomCrop` with target size.
    #[must_use]
    pub fn new(size: Vec<usize>) -> Self {
        Self { size }
    }

    /// Creates a `RandomCrop` for 2D images.
    #[must_use]
    pub fn new_2d(height: usize, width: usize) -> Self {
        Self::new(vec![height, width])
    }
}

impl Transform for RandomCrop {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();

        // Determine spatial dimensions (last N dimensions where N = size.len())
        if shape.len() < self.size.len() {
            return input.clone();
        }

        let spatial_start = shape.len() - self.size.len();
        let mut rng = rand::thread_rng();

        // Calculate random offsets for each spatial dimension
        let mut offsets = Vec::with_capacity(self.size.len());
        for (i, &target_dim) in self.size.iter().enumerate() {
            let input_dim = shape[spatial_start + i];
            if input_dim <= target_dim {
                offsets.push(0);
            } else {
                offsets.push(rng.gen_range(0..=input_dim - target_dim));
            }
        }

        // Calculate actual crop sizes (clamped to input dimensions)
        let crop_sizes: Vec<usize> = self
            .size
            .iter()
            .enumerate()
            .map(|(i, &s)| s.min(shape[spatial_start + i]))
            .collect();

        let data = input.to_vec();

        // Handle 1D case
        if shape.len() == 1 && self.size.len() == 1 {
            let start = offsets[0];
            let end = start + crop_sizes[0];
            let cropped = data[start..end].to_vec();
            let len = cropped.len();
            return Tensor::from_vec(cropped, &[len]).unwrap();
        }

        // Handle 2D case (H x W)
        if shape.len() == 2 && self.size.len() == 2 {
            let (_h, w) = (shape[0], shape[1]);
            let (crop_h, crop_w) = (crop_sizes[0], crop_sizes[1]);
            let (off_h, off_w) = (offsets[0], offsets[1]);

            let mut cropped = Vec::with_capacity(crop_h * crop_w);
            for row in off_h..off_h + crop_h {
                for col in off_w..off_w + crop_w {
                    cropped.push(data[row * w + col]);
                }
            }
            return Tensor::from_vec(cropped, &[crop_h, crop_w]).unwrap();
        }

        // Handle 3D case (C x H x W) - common for images
        if shape.len() == 3 && self.size.len() == 2 {
            let (c, h, w) = (shape[0], shape[1], shape[2]);
            let (crop_h, crop_w) = (crop_sizes[0], crop_sizes[1]);
            let (off_h, off_w) = (offsets[0], offsets[1]);

            let mut cropped = Vec::with_capacity(c * crop_h * crop_w);
            for channel in 0..c {
                for row in off_h..off_h + crop_h {
                    for col in off_w..off_w + crop_w {
                        cropped.push(data[channel * h * w + row * w + col]);
                    }
                }
            }
            return Tensor::from_vec(cropped, &[c, crop_h, crop_w]).unwrap();
        }

        // Handle 4D case (N x C x H x W) - batched images
        if shape.len() == 4 && self.size.len() == 2 {
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let (crop_h, crop_w) = (crop_sizes[0], crop_sizes[1]);
            let (off_h, off_w) = (offsets[0], offsets[1]);

            let mut cropped = Vec::with_capacity(n * c * crop_h * crop_w);
            for batch in 0..n {
                for channel in 0..c {
                    for row in off_h..off_h + crop_h {
                        for col in off_w..off_w + crop_w {
                            let idx = batch * c * h * w + channel * h * w + row * w + col;
                            cropped.push(data[idx]);
                        }
                    }
                }
            }
            return Tensor::from_vec(cropped, &[n, c, crop_h, crop_w]).unwrap();
        }

        // Fallback for unsupported dimensions - shouldn't reach here in practice
        input.clone()
    }
}

// =============================================================================
// RandomFlip
// =============================================================================

/// Randomly flips the input along a specified dimension.
pub struct RandomFlip {
    dim: usize,
    probability: f32,
}

impl RandomFlip {
    /// Creates a new `RandomFlip`.
    #[must_use]
    pub fn new(dim: usize, probability: f32) -> Self {
        Self {
            dim,
            probability: probability.clamp(0.0, 1.0),
        }
    }

    /// Creates a horizontal flip (dim=1 for `HxW` images).
    #[must_use]
    pub fn horizontal() -> Self {
        Self::new(1, 0.5)
    }

    /// Creates a vertical flip (dim=0 for `HxW` images).
    #[must_use]
    pub fn vertical() -> Self {
        Self::new(0, 0.5)
    }
}

impl Transform for RandomFlip {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut rng = rand::thread_rng();
        if rng.r#gen::<f32>() > self.probability {
            return input.clone();
        }

        let shape = input.shape();
        if self.dim >= shape.len() {
            return input.clone();
        }

        let data = input.to_vec();
        let ndim = shape.len();

        // Generic N-dimensional flip along self.dim:
        // Compute strides, then for each element map the flipped index.
        let total = data.len();
        let mut flipped = vec![0.0f32; total];

        // Compute strides (row-major)
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let dim = self.dim;
        let dim_size = shape[dim];
        let dim_stride = strides[dim];

        for i in 0..total {
            // Extract the coordinate along the flip dimension
            let coord_in_dim = (i / dim_stride) % dim_size;
            let flipped_coord = dim_size - 1 - coord_in_dim;
            // Compute the source index with the flipped coordinate
            let diff = flipped_coord as isize - coord_in_dim as isize;
            let src = (i as isize + diff * dim_stride as isize) as usize;
            flipped[i] = data[src];
        }

        Tensor::from_vec(flipped, shape).unwrap()
    }
}

// =============================================================================
// Scale
// =============================================================================

/// Scales tensor values by a constant factor.
pub struct Scale {
    factor: f32,
}

impl Scale {
    /// Creates a new Scale transform.
    #[must_use]
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }
}

impl Transform for Scale {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        input.mul_scalar(self.factor)
    }
}

// =============================================================================
// Clamp
// =============================================================================

/// Clamps tensor values to a specified range.
pub struct Clamp {
    min: f32,
    max: f32,
}

impl Clamp {
    /// Creates a new Clamp transform.
    #[must_use]
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Creates a Clamp for [0, 1] range.
    #[must_use]
    pub fn zero_one() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Creates a Clamp for [-1, 1] range.
    #[must_use]
    pub fn symmetric() -> Self {
        Self::new(-1.0, 1.0)
    }
}

impl Transform for Clamp {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        let clamped: Vec<f32> = data.iter().map(|&x| x.clamp(self.min, self.max)).collect();
        Tensor::from_vec(clamped, input.shape()).unwrap()
    }
}

// =============================================================================
// Flatten
// =============================================================================

/// Flattens the tensor to 1D.
pub struct Flatten;

impl Flatten {
    /// Creates a new Flatten transform.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Flatten {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        Tensor::from_vec(data.clone(), &[data.len()]).unwrap()
    }
}

// =============================================================================
// Reshape
// =============================================================================

/// Reshapes the tensor to a specified shape.
pub struct Reshape {
    shape: Vec<usize>,
}

impl Reshape {
    /// Creates a new Reshape transform.
    #[must_use]
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl Transform for Reshape {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        let expected_size: usize = self.shape.iter().product();

        if data.len() != expected_size {
            // Size mismatch, return original
            return input.clone();
        }

        Tensor::from_vec(data, &self.shape).unwrap()
    }
}

// =============================================================================
// Dropout Transform
// =============================================================================

/// Applies dropout by randomly zeroing elements during training.
///
/// Respects train/eval mode: dropout is only applied when `training` is true.
/// Use `set_training(false)` to disable dropout during evaluation.
pub struct DropoutTransform {
    probability: f32,
    training: std::sync::atomic::AtomicBool,
}

impl DropoutTransform {
    /// Creates a new `DropoutTransform` in training mode.
    #[must_use]
    pub fn new(probability: f32) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            training: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Sets whether this transform is in training mode.
    pub fn set_training(&self, training: bool) {
        self.training
            .store(training, std::sync::atomic::Ordering::Relaxed);
    }

    /// Returns whether this transform is in training mode.
    pub fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Transform for DropoutTransform {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // No dropout in eval mode or with p=0
        if !self.is_training() || self.probability == 0.0 {
            return input.clone();
        }

        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.probability);
        let data = input.to_vec();

        let dropped: Vec<f32> = data
            .iter()
            .map(|&x| {
                if rng.r#gen::<f32>() < self.probability {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect();

        Tensor::from_vec(dropped, input.shape()).unwrap()
    }
}

// =============================================================================
// Lambda Transform
// =============================================================================

/// Applies a custom function as a transform.
pub struct Lambda<F>
where
    F: Fn(&Tensor<f32>) -> Tensor<f32> + Send + Sync,
{
    func: F,
}

impl<F> Lambda<F>
where
    F: Fn(&Tensor<f32>) -> Tensor<f32> + Send + Sync,
{
    /// Creates a new Lambda transform.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> Transform for Lambda<F>
where
    F: Fn(&Tensor<f32>) -> Tensor<f32> + Send + Sync,
{
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        (self.func)(input)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let normalize = Normalize::new(2.5, 0.5);

        let output = normalize.apply(&input);
        let expected = [-3.0, -1.0, 1.0, 3.0];

        let result = output.to_vec();
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_per_channel() {
        // 2 channels, 2x2 spatial => [2, 2, 2]
        let input =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[2, 2, 2]).unwrap();
        let normalize = Normalize::per_channel(vec![0.0, 10.0], vec![1.0, 10.0]);

        let output = normalize.apply(&input);
        let result = output.to_vec();
        // Channel 0: (x - 0) / 1 = x
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[3] - 4.0).abs() < 1e-6);
        // Channel 1: (x - 10) / 10
        assert!((result[4] - 0.0).abs() < 1e-6); // (10-10)/10
        assert!((result[5] - 1.0).abs() < 1e-6); // (20-10)/10
    }

    #[test]
    fn test_scale() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let scale = Scale::new(2.0);

        let output = scale.apply(&input);
        assert_eq!(output.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_clamp() {
        let input = Tensor::from_vec(vec![-1.0, 0.5, 2.0], &[3]).unwrap();
        let clamp = Clamp::zero_one();

        let output = clamp.apply(&input);
        assert_eq!(output.to_vec(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_flatten() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let flatten = Flatten::new();

        let output = flatten.apply(&input);
        assert_eq!(output.shape(), &[4]);
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reshape() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let reshape = Reshape::new(vec![2, 3]);

        let output = reshape.apply(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_compose() {
        let normalize = Normalize::new(0.0, 1.0);
        let scale = Scale::new(2.0);

        let compose = Compose::new(vec![Box::new(normalize), Box::new(scale)]);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = compose.apply(&input);

        // normalize(x) = x, then scale by 2
        assert_eq!(output.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_compose_builder() {
        let compose = Compose::empty()
            .add(Normalize::new(0.0, 1.0))
            .add(Scale::new(2.0));

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = compose.apply(&input);

        assert_eq!(output.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_random_noise() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let noise = RandomNoise::new(0.0);

        // With std=0, output should equal input
        let output = noise.apply(&input);
        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_random_flip_1d() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let flip = RandomFlip::new(0, 1.0); // Always flip

        let output = flip.apply(&input);
        assert_eq!(output.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_random_flip_2d_horizontal() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let flip = RandomFlip::new(1, 1.0); // Always flip horizontal

        let output = flip.apply(&input);
        // [[1, 2], [3, 4]] -> [[2, 1], [4, 3]]
        assert_eq!(output.to_vec(), vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_random_flip_2d_vertical() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let flip = RandomFlip::new(0, 1.0); // Always flip vertical

        let output = flip.apply(&input);
        // [[1, 2], [3, 4]] -> [[3, 4], [1, 2]]
        assert_eq!(output.to_vec(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_random_flip_3d() {
        // C=1, H=2, W=2 — flip along dim=2 (horizontal in spatial)
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        let flip = RandomFlip::new(2, 1.0); // Always flip along W

        let output = flip.apply(&input);
        // [[1,2],[3,4]] → [[2,1],[4,3]]
        assert_eq!(output.to_vec(), vec![2.0, 1.0, 4.0, 3.0]);
        assert_eq!(output.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_random_flip_4d() {
        // N=1, C=1, H=2, W=2 — flip along dim=2 (vertical flip of H)
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let flip = RandomFlip::new(2, 1.0); // Flip along H

        let output = flip.apply(&input);
        // Rows flipped: [[1,2],[3,4]] → [[3,4],[1,2]]
        assert_eq!(output.to_vec(), vec![3.0, 4.0, 1.0, 2.0]);
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_dropout_eval_mode() {
        let input = Tensor::from_vec(vec![1.0; 100], &[100]).unwrap();
        let dropout = DropoutTransform::new(0.5);

        // In training mode, should drop elements
        let output_train = dropout.apply(&input);
        let zeros_train = output_train.to_vec().iter().filter(|&&x| x == 0.0).count();
        assert!(zeros_train > 0, "Training mode should drop elements");

        // Switch to eval mode — should be identity
        dropout.set_training(false);
        let output_eval = dropout.apply(&input);
        assert_eq!(output_eval.to_vec(), vec![1.0; 100]);
    }

    #[test]
    fn test_dropout_transform() {
        let input = Tensor::from_vec(vec![1.0; 1000], &[1000]).unwrap();
        let dropout = DropoutTransform::new(0.5);

        let output = dropout.apply(&input);
        let output_vec = output.to_vec();

        // About half should be zero
        let zeros = output_vec.iter().filter(|&&x| x == 0.0).count();
        assert!(
            zeros > 300 && zeros < 700,
            "Expected ~500 zeros, got {zeros}"
        );

        // Non-zeros should be scaled by 2 (1/(1-0.5))
        let nonzeros: Vec<f32> = output_vec.iter().filter(|&&x| x != 0.0).copied().collect();
        for x in nonzeros {
            assert!((x - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lambda() {
        let lambda = Lambda::new(|t: &Tensor<f32>| t.mul_scalar(3.0));

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let output = lambda.apply(&input);

        assert_eq!(output.to_vec(), vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_to_tensor() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let to_tensor = ToTensor::new();

        let output = to_tensor.apply(&input);
        assert_eq!(output.to_vec(), input.to_vec());
    }

    #[test]
    fn test_normalize_variants() {
        let standard = Normalize::standard();
        assert_eq!(standard.mean, vec![0.0]);
        assert_eq!(standard.std, vec![1.0]);

        let zero_centered = Normalize::zero_centered();
        assert_eq!(zero_centered.mean, vec![0.5]);
        assert_eq!(zero_centered.std, vec![0.5]);
    }

    #[test]
    fn test_random_crop_1d() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let crop = RandomCrop::new(vec![3]);

        let output = crop.apply(&input);
        assert_eq!(output.shape(), &[3]);
    }

    #[test]
    fn test_random_crop_2d() {
        // 4x4 image
        let input = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[4, 4]).unwrap();
        let crop = RandomCrop::new_2d(2, 2);

        let output = crop.apply(&input);
        assert_eq!(output.shape(), &[2, 2]);
        // Verify values are contiguous from the original
        let vals = output.to_vec();
        assert_eq!(vals.len(), 4);
    }

    #[test]
    fn test_random_crop_3d() {
        // 2 channels x 4x4 image
        let input = Tensor::from_vec((1..=32).map(|x| x as f32).collect(), &[2, 4, 4]).unwrap();
        let crop = RandomCrop::new_2d(2, 2);

        let output = crop.apply(&input);
        assert_eq!(output.shape(), &[2, 2, 2]); // 2 channels, 2x2 spatial
    }
}
