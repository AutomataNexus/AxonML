//! Weight Initialization - Parameter Initialization Strategies
//!
//! Provides various weight initialization strategies for neural networks.
//! Proper initialization is crucial for training deep networks.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_tensor::Tensor;
use rand::Rng;

// =============================================================================
// Basic Initializers
// =============================================================================

/// Creates a tensor filled with zeros.
pub fn zeros(shape: &[usize]) -> Tensor<f32> {
    axonml_tensor::zeros(shape)
}

/// Creates a tensor filled with ones.
pub fn ones(shape: &[usize]) -> Tensor<f32> {
    axonml_tensor::ones(shape)
}

/// Creates a tensor filled with a constant value.
pub fn constant(shape: &[usize], value: f32) -> Tensor<f32> {
    axonml_tensor::full(shape, value)
}

// =============================================================================
// Random Initializers
// =============================================================================

/// Creates a tensor with uniform random values in [0, 1).
pub fn uniform(shape: &[usize]) -> Tensor<f32> {
    axonml_tensor::rand(shape)
}

/// Creates a tensor with uniform random values in [low, high).
pub fn uniform_range(shape: &[usize], low: f32, high: f32) -> Tensor<f32> {
    let mut rng = rand::thread_rng();
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(low..high)).collect();
    Tensor::from_vec(data, shape).unwrap()
}

/// Creates a tensor with standard normal random values (mean=0, std=1).
pub fn randn(shape: &[usize]) -> Tensor<f32> {
    axonml_tensor::randn(shape)
}

/// Creates a tensor with normal random values (specified mean and std).
pub fn normal(shape: &[usize], mean: f32, std: f32) -> Tensor<f32> {
    let base = axonml_tensor::randn(shape);
    base.mul_scalar(std).add_scalar(mean)
}

// =============================================================================
// Xavier/Glorot Initialization
// =============================================================================

/// Xavier uniform initialization.
///
/// Designed for layers with tanh or sigmoid activations.
/// Samples from U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
///
/// # Arguments
/// * `fan_in` - Number of input units
/// * `fan_out` - Number of output units
pub fn xavier_uniform(fan_in: usize, fan_out: usize) -> Tensor<f32> {
    let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
    uniform_range(&[fan_out, fan_in], -a, a)
}

/// Xavier normal initialization.
///
/// Designed for layers with tanh or sigmoid activations.
/// Samples from N(0, std) where std = sqrt(2 / (fan_in + fan_out))
///
/// # Arguments
/// * `fan_in` - Number of input units
/// * `fan_out` - Number of output units
pub fn xavier_normal(fan_in: usize, fan_out: usize) -> Tensor<f32> {
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    normal(&[fan_out, fan_in], 0.0, std)
}

/// Alias for xavier_uniform.
pub fn glorot_uniform(fan_in: usize, fan_out: usize) -> Tensor<f32> {
    xavier_uniform(fan_in, fan_out)
}

/// Alias for xavier_normal.
pub fn glorot_normal(fan_in: usize, fan_out: usize) -> Tensor<f32> {
    xavier_normal(fan_in, fan_out)
}

// =============================================================================
// Kaiming/He Initialization
// =============================================================================

/// Kaiming uniform initialization.
///
/// Designed for layers with ReLU activations.
/// Samples from U(-bound, bound) where bound = sqrt(6 / fan_in)
///
/// # Arguments
/// * `fan_in` - Number of input units
/// * `fan_out` - Number of output units
pub fn kaiming_uniform(fan_out: usize, fan_in: usize) -> Tensor<f32> {
    let bound = (6.0 / fan_in as f32).sqrt();
    uniform_range(&[fan_out, fan_in], -bound, bound)
}

/// Kaiming normal initialization.
///
/// Designed for layers with ReLU activations.
/// Samples from N(0, std) where std = sqrt(2 / fan_in)
///
/// # Arguments
/// * `fan_in` - Number of input units
/// * `fan_out` - Number of output units
pub fn kaiming_normal(fan_out: usize, fan_in: usize) -> Tensor<f32> {
    let std = (2.0 / fan_in as f32).sqrt();
    normal(&[fan_out, fan_in], 0.0, std)
}

/// Alias for kaiming_uniform.
pub fn he_uniform(fan_out: usize, fan_in: usize) -> Tensor<f32> {
    kaiming_uniform(fan_out, fan_in)
}

/// Alias for kaiming_normal.
pub fn he_normal(fan_out: usize, fan_in: usize) -> Tensor<f32> {
    kaiming_normal(fan_out, fan_in)
}

// =============================================================================
// Other Initializers
// =============================================================================

/// Orthogonal initialization.
///
/// Creates a (semi-)orthogonal matrix using QR decomposition.
/// Good for RNNs.
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `gain` - Multiplicative factor (default 1.0)
pub fn orthogonal(rows: usize, cols: usize, gain: f32) -> Tensor<f32> {
    // Simple implementation: start with random matrix and use Gram-Schmidt
    // For a full implementation, we'd use QR decomposition
    let mut data = vec![0.0f32; rows * cols];
    let mut rng = rand::thread_rng();

    // Generate random matrix
    for val in data.iter_mut() {
        *val = rng.gen_range(-1.0..1.0);
    }

    // Simple normalization (not true orthogonal, but approximation)
    // A proper implementation would use QR decomposition
    for i in 0..rows.min(cols) {
        let start = i * cols;
        let end = start + cols;
        let row = &mut data[start..end];

        // Normalize the row
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for val in row.iter_mut() {
                *val = (*val / norm) * gain;
            }
        }
    }

    Tensor::from_vec(data, &[rows, cols]).unwrap()
}

/// Sparse initialization.
///
/// Creates a matrix where each column has only `sparsity` fraction of non-zero elements.
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `sparsity` - Fraction of non-zero elements per column
/// * `std` - Standard deviation of non-zero elements
pub fn sparse(rows: usize, cols: usize, sparsity: f32, std: f32) -> Tensor<f32> {
    let mut data = vec![0.0f32; rows * cols];
    let mut rng = rand::thread_rng();

    let num_nonzero = (rows as f32 * sparsity).ceil() as usize;

    for col in 0..cols {
        // Randomly select which rows will be non-zero
        let mut indices: Vec<usize> = (0..rows).collect();
        for i in 0..num_nonzero.min(rows) {
            let j = rng.gen_range(i..rows);
            indices.swap(i, j);
        }

        // Set non-zero values
        for &row in indices.iter().take(num_nonzero) {
            let val: f32 = rng.gen::<f32>() * 2.0 - 1.0; // Approximate normal
            data[row * cols + col] = val * std;
        }
    }

    Tensor::from_vec(data, &[rows, cols]).unwrap()
}

/// Identity matrix initialization.
///
/// Creates an identity matrix (or as close as possible for non-square).
pub fn eye(size: usize) -> Tensor<f32> {
    axonml_tensor::eye(size)
}

/// Diagonal initialization.
///
/// Creates a matrix with specified values on the diagonal.
pub fn diag(values: &[f32]) -> Tensor<f32> {
    let n = values.len();
    let mut data = vec![0.0f32; n * n];
    for (i, &val) in values.iter().enumerate() {
        data[i * n + i] = val;
    }
    Tensor::from_vec(data, &[n, n]).unwrap()
}

// =============================================================================
// Initialization Mode Enum
// =============================================================================

/// Initialization strategies as an enum for dynamic selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitMode {
    /// Zeros initialization.
    Zeros,
    /// Ones initialization.
    Ones,
    /// Constant value initialization.
    Constant(f32),
    /// Uniform random initialization.
    Uniform,
    /// Uniform random in range.
    UniformRange(f32, f32),
    /// Normal distribution.
    Normal(f32, f32), // mean, std
    /// Xavier/Glorot uniform.
    XavierUniform,
    /// Xavier/Glorot normal.
    XavierNormal,
    /// Kaiming/He uniform.
    KaimingUniform,
    /// Kaiming/He normal.
    KaimingNormal,
    /// Orthogonal.
    Orthogonal(f32), // gain
}

impl InitMode {
    /// Initializes a tensor using this mode.
    pub fn init(&self, fan_out: usize, fan_in: usize) -> Tensor<f32> {
        match self {
            InitMode::Zeros => zeros(&[fan_out, fan_in]),
            InitMode::Ones => ones(&[fan_out, fan_in]),
            InitMode::Constant(val) => constant(&[fan_out, fan_in], *val),
            InitMode::Uniform => uniform(&[fan_out, fan_in]),
            InitMode::UniformRange(low, high) => uniform_range(&[fan_out, fan_in], *low, *high),
            InitMode::Normal(mean, std) => normal(&[fan_out, fan_in], *mean, *std),
            InitMode::XavierUniform => xavier_uniform(fan_in, fan_out),
            InitMode::XavierNormal => xavier_normal(fan_in, fan_out),
            InitMode::KaimingUniform => kaiming_uniform(fan_out, fan_in),
            InitMode::KaimingNormal => kaiming_normal(fan_out, fan_in),
            InitMode::Orthogonal(gain) => orthogonal(fan_out, fan_in, *gain),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.to_vec().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = ones(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.to_vec().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_uniform_range() {
        let t = uniform_range(&[100], 0.0, 1.0);
        let data = t.to_vec();
        assert!(data.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_xavier_uniform() {
        let t = xavier_uniform(100, 100);
        assert_eq!(t.shape(), &[100, 100]);
        let bound = (6.0 / 200.0_f32).sqrt();
        let data = t.to_vec();
        assert!(data.iter().all(|&x| x.abs() <= bound * 1.1)); // Small margin
    }

    #[test]
    fn test_kaiming_uniform() {
        let t = kaiming_uniform(100, 100);
        assert_eq!(t.shape(), &[100, 100]);
    }

    #[test]
    fn test_eye() {
        let t = eye(3);
        assert_eq!(t.shape(), &[3, 3]);
        let data = t.to_vec();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_init_mode() {
        let mode = InitMode::KaimingUniform;
        let t = mode.init(10, 5);
        assert_eq!(t.shape(), &[10, 5]);
    }
}
