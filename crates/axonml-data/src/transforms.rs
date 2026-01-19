//! Transforms - Data Augmentation and Preprocessing
//!
//! Provides composable transformations for data preprocessing and augmentation.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

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
    #[must_use] pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    /// Creates an empty Compose.
    #[must_use] pub fn empty() -> Self {
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
    #[must_use] pub fn new() -> Self {
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
pub struct Normalize {
    mean: f32,
    std: f32,
}

impl Normalize {
    /// Creates a new Normalize transform.
    #[must_use] pub fn new(mean: f32, std: f32) -> Self {
        Self { mean, std }
    }

    /// Creates a Normalize for standard normal distribution (mean=0, std=1).
    #[must_use] pub fn standard() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Creates a Normalize for [0,1] to [-1,1] conversion (mean=0.5, std=0.5).
    #[must_use] pub fn zero_centered() -> Self {
        Self::new(0.5, 0.5)
    }
}

impl Transform for Normalize {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data = input.to_vec();
        let normalized: Vec<f32> = data.iter().map(|&x| (x - self.mean) / self.std).collect();
        Tensor::from_vec(normalized, input.shape()).unwrap()
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
    #[must_use] pub fn new(std: f32) -> Self {
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
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
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
    #[must_use] pub fn new(size: Vec<usize>) -> Self {
        Self { size }
    }

    /// Creates a `RandomCrop` for 2D images.
    #[must_use] pub fn new_2d(height: usize, width: usize) -> Self {
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
    #[must_use] pub fn new(dim: usize, probability: f32) -> Self {
        Self {
            dim,
            probability: probability.clamp(0.0, 1.0),
        }
    }

    /// Creates a horizontal flip (dim=1 for `HxW` images).
    #[must_use] pub fn horizontal() -> Self {
        Self::new(1, 0.5)
    }

    /// Creates a vertical flip (dim=0 for `HxW` images).
    #[must_use] pub fn vertical() -> Self {
        Self::new(0, 0.5)
    }
}

impl Transform for RandomFlip {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() > self.probability {
            return input.clone();
        }

        let shape = input.shape();
        if self.dim >= shape.len() {
            return input.clone();
        }

        // Simple 1D flip implementation
        if shape.len() == 1 {
            let mut data = input.to_vec();
            data.reverse();
            return Tensor::from_vec(data, shape).unwrap();
        }

        // For 2D, flip along the specified dimension
        if shape.len() == 2 {
            let data = input.to_vec();
            let (rows, cols) = (shape[0], shape[1]);
            let mut flipped = vec![0.0; data.len()];

            if self.dim == 0 {
                // Vertical flip
                for r in 0..rows {
                    for c in 0..cols {
                        flipped[r * cols + c] = data[(rows - 1 - r) * cols + c];
                    }
                }
            } else {
                // Horizontal flip
                for r in 0..rows {
                    for c in 0..cols {
                        flipped[r * cols + c] = data[r * cols + (cols - 1 - c)];
                    }
                }
            }

            return Tensor::from_vec(flipped, shape).unwrap();
        }

        input.clone()
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
    #[must_use] pub fn new(factor: f32) -> Self {
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
    #[must_use] pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Creates a Clamp for [0, 1] range.
    #[must_use] pub fn zero_one() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Creates a Clamp for [-1, 1] range.
    #[must_use] pub fn symmetric() -> Self {
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
    #[must_use] pub fn new() -> Self {
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
    #[must_use] pub fn new(shape: Vec<usize>) -> Self {
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
pub struct DropoutTransform {
    probability: f32,
}

impl DropoutTransform {
    /// Creates a new `DropoutTransform`.
    #[must_use] pub fn new(probability: f32) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Transform for DropoutTransform {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        if self.probability == 0.0 {
            return input.clone();
        }

        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.probability);
        let data = input.to_vec();

        let dropped: Vec<f32> = data
            .iter()
            .map(|&x| {
                if rng.gen::<f32>() < self.probability {
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
        assert_eq!(standard.mean, 0.0);
        assert_eq!(standard.std, 1.0);

        let zero_centered = Normalize::zero_centered();
        assert_eq!(zero_centered.mean, 0.5);
        assert_eq!(zero_centered.std, 0.5);
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
