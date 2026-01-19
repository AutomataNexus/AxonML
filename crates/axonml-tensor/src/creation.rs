//! Tensor Creation Functions
//!
//! Provides convenient functions for creating tensors with various initializations
//! including zeros, ones, random values, ranges, and more.
//!
//! # Key Features
//! - Factory functions for common tensor initializations
//! - Random tensor generation with various distributions
//! - Range and linspace functions
//!

//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rand_distr::{Normal, StandardNormal, Uniform};

use axonml_core::dtype::{Float, Numeric, Scalar};

use crate::tensor::Tensor;

// =============================================================================
// Zero and One Initialization
// =============================================================================

/// Creates a tensor filled with zeros.
///
/// # Arguments
/// * `shape` - Shape of the tensor
///
/// # Example
/// ```rust,ignore
/// use axonml_tensor::zeros;
/// let t = zeros::<f32>(&[2, 3]);
/// ```
#[must_use] pub fn zeros<T: Scalar>(shape: &[usize]) -> Tensor<T> {
    let numel: usize = shape.iter().product();
    let data = vec![T::zeroed(); numel];
    Tensor::from_vec(data, shape).unwrap()
}

/// Creates a tensor filled with ones.
///
/// # Arguments
/// * `shape` - Shape of the tensor
#[must_use] pub fn ones<T: Numeric>(shape: &[usize]) -> Tensor<T> {
    full(shape, T::one())
}

/// Creates a tensor filled with a specific value.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `value` - Fill value
pub fn full<T: Scalar>(shape: &[usize], value: T) -> Tensor<T> {
    let numel: usize = shape.iter().product();
    let data = vec![value; numel];
    Tensor::from_vec(data, shape).unwrap()
}

/// Creates a tensor with the same shape as another, filled with zeros.
#[must_use] pub fn zeros_like<T: Scalar>(other: &Tensor<T>) -> Tensor<T> {
    zeros(other.shape())
}

/// Creates a tensor with the same shape as another, filled with ones.
#[must_use] pub fn ones_like<T: Numeric>(other: &Tensor<T>) -> Tensor<T> {
    ones(other.shape())
}

/// Creates a tensor with the same shape as another, filled with a value.
pub fn full_like<T: Scalar>(other: &Tensor<T>, value: T) -> Tensor<T> {
    full(other.shape(), value)
}

// =============================================================================
// Identity and Diagonal
// =============================================================================

/// Creates a 2D identity matrix.
///
/// # Arguments
/// * `n` - Size of the matrix (n x n)
#[must_use] pub fn eye<T: Numeric>(n: usize) -> Tensor<T> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }
    Tensor::from_vec(data, &[n, n]).unwrap()
}

/// Creates a 2D tensor with the given diagonal values.
///
/// # Arguments
/// * `diag` - Values for the diagonal
pub fn diag<T: Numeric>(diag: &[T]) -> Tensor<T> {
    let n = diag.len();
    let mut data = vec![T::zero(); n * n];
    for (i, &val) in diag.iter().enumerate() {
        data[i * n + i] = val;
    }
    Tensor::from_vec(data, &[n, n]).unwrap()
}

// =============================================================================
// Random Initialization
// =============================================================================

/// Creates a tensor with uniformly distributed random values in [0, 1).
///
/// # Arguments
/// * `shape` - Shape of the tensor
#[must_use] pub fn rand<T: Float>(shape: &[usize]) -> Tensor<T>
where
    Standard: Distribution<T>,
{
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data: Vec<T> = (0..numel).map(|_| rng.gen()).collect();
    Tensor::from_vec(data, shape).unwrap()
}

/// Creates a tensor with normally distributed random values (mean=0, std=1).
///
/// # Arguments
/// * `shape` - Shape of the tensor
#[must_use] pub fn randn<T: Float>(shape: &[usize]) -> Tensor<T>
where
    StandardNormal: Distribution<T>,
{
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let normal = StandardNormal;
    let data: Vec<T> = (0..numel).map(|_| normal.sample(&mut rng)).collect();
    Tensor::from_vec(data, shape).unwrap()
}

/// Creates a tensor with uniformly distributed random values in [low, high).
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (exclusive)
pub fn uniform<T: Float>(shape: &[usize], low: T, high: T) -> Tensor<T>
where
    T: rand::distributions::uniform::SampleUniform,
{
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    let data: Vec<T> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    Tensor::from_vec(data, shape).unwrap()
}

/// Creates a tensor with normally distributed random values.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `mean` - Mean of the distribution
/// * `std` - Standard deviation of the distribution
pub fn normal<T: Float>(shape: &[usize], mean: T, std: T) -> Tensor<T>
where
    T: rand::distributions::uniform::SampleUniform,
    StandardNormal: Distribution<T>,
{
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Normal::new(mean, std).unwrap();
    let data: Vec<T> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    Tensor::from_vec(data, shape).unwrap()
}

/// Creates a tensor with random integers in [low, high).
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (exclusive)
#[must_use] pub fn randint<T: Numeric>(shape: &[usize], low: i64, high: i64) -> Tensor<T>
where
    T: num_traits::NumCast,
{
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    let data: Vec<T> = (0..numel)
        .map(|_| T::from(dist.sample(&mut rng)).unwrap())
        .collect();
    Tensor::from_vec(data, shape).unwrap()
}

// =============================================================================
// Range Functions
// =============================================================================

/// Creates a 1D tensor with values from start to end (exclusive) with step.
///
/// # Arguments
/// * `start` - Start value
/// * `end` - End value (exclusive)
/// * `step` - Step size
pub fn arange<T: Numeric>(start: T, end: T, step: T) -> Tensor<T>
where
    T: num_traits::NumCast + PartialOrd,
{
    let mut data = Vec::new();
    let mut current = start;

    if step > T::zero() {
        while current < end {
            data.push(current);
            current = current + step;
        }
    } else if step < T::zero() {
        while current > end {
            data.push(current);
            current = current + step;
        }
    }

    let len = data.len();
    Tensor::from_vec(data, &[len]).unwrap()
}

/// Creates a 1D tensor with `num` evenly spaced values from start to end.
///
/// # Arguments
/// * `start` - Start value
/// * `end` - End value (inclusive)
/// * `num` - Number of values
pub fn linspace<T: Float>(start: T, end: T, num: usize) -> Tensor<T> {
    if num == 0 {
        return Tensor::from_vec(vec![], &[0]).unwrap();
    }

    if num == 1 {
        return Tensor::from_vec(vec![start], &[1]).unwrap();
    }

    let step = (end - start) / T::from(num - 1).unwrap();
    let data: Vec<T> = (0..num)
        .map(|i| start + step * T::from(i).unwrap())
        .collect();

    Tensor::from_vec(data, &[num]).unwrap()
}

/// Creates a 1D tensor with `num` logarithmically spaced values.
///
/// # Arguments
/// * `start` - Start exponent (base^start)
/// * `end` - End exponent (base^end)
/// * `num` - Number of values
/// * `base` - Base of the logarithm
pub fn logspace<T: Float>(start: T, end: T, num: usize, base: T) -> Tensor<T> {
    if num == 0 {
        return Tensor::from_vec(vec![], &[0]).unwrap();
    }

    let lin = linspace(start, end, num);
    let data: Vec<T> = lin.to_vec().iter().map(|&x| base.pow_value(x)).collect();

    Tensor::from_vec(data, &[num]).unwrap()
}

// =============================================================================
// Empty Tensor
// =============================================================================

/// Creates an uninitialized tensor (values are undefined).
///
/// # Safety
/// The tensor contents are uninitialized. Reading before writing is undefined.
///
/// # Arguments
/// * `shape` - Shape of the tensor
#[must_use] pub fn empty<T: Scalar>(shape: &[usize]) -> Tensor<T> {
    zeros(shape)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = zeros::<f32>(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        for val in t.to_vec() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_ones() {
        let t = ones::<f32>(&[2, 3]);
        for val in t.to_vec() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_full() {
        let t = full::<f32>(&[2, 3], 42.0);
        for val in t.to_vec() {
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    fn test_eye() {
        let t = eye::<f32>(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[1, 1]).unwrap(), 1.0);
        assert_eq!(t.get(&[2, 2]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 1]).unwrap(), 0.0);
    }

    #[test]
    fn test_rand() {
        let t = rand::<f32>(&[100]);
        for val in t.to_vec() {
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_arange() {
        let t = arange::<f32>(0.0, 5.0, 1.0);
        assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let t = arange::<f32>(0.0, 1.0, 0.2);
        assert_eq!(t.numel(), 5);
    }

    #[test]
    fn test_linspace() {
        let t = linspace::<f32>(0.0, 1.0, 5);
        let data = t.to_vec();
        assert_eq!(data.len(), 5);
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zeros_like() {
        let a = ones::<f32>(&[2, 3]);
        let b = zeros_like(&a);
        assert_eq!(b.shape(), &[2, 3]);
        for val in b.to_vec() {
            assert_eq!(val, 0.0);
        }
    }
}
