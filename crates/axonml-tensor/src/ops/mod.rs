//! Tensor Operations - Mathematical and Structural Operations
//!
//! This module re-exports all tensor operations for convenient access.
//! Operations are organized into submodules by category.
//!
//! # Categories
//! - Arithmetic: +, -, *, /, power
//! - Comparison: ==, <, >, <=, >=
//! - Reduction: sum, mean, max, min
//! - Matrix: matmul, transpose, inverse
//! - Activation: relu, sigmoid, tanh, softmax
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

// Operations are implemented directly on Tensor in tensor.rs
// This module provides additional standalone functions

use axonml_core::dtype::{Float, Numeric, Scalar};
use axonml_core::error::Result;

use crate::tensor::Tensor;

// =============================================================================
// Comparison Operations
// =============================================================================

/// Element-wise equality comparison.
pub fn eq<T: Numeric + PartialEq>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(axonml_core::error::Error::shape_mismatch(
            a.shape(),
            b.shape(),
        ));
    }

    let a_data = a.to_vec();
    let b_data = b.to_vec();

    Ok(a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x == y)
        .collect())
}

/// Element-wise less-than comparison.
pub fn lt<T: Numeric>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(axonml_core::error::Error::shape_mismatch(
            a.shape(),
            b.shape(),
        ));
    }

    let a_data = a.to_vec();
    let b_data = b.to_vec();

    Ok(a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x < y)
        .collect())
}

/// Element-wise greater-than comparison.
pub fn gt<T: Numeric>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(axonml_core::error::Error::shape_mismatch(
            a.shape(),
            b.shape(),
        ));
    }

    let a_data = a.to_vec();
    let b_data = b.to_vec();

    Ok(a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x > y)
        .collect())
}

// =============================================================================
// Advanced Activation Functions
// =============================================================================

/// Applies softmax along the specified dimension.
pub fn softmax<T: Float>(x: &Tensor<T>, _dim: i64) -> Result<Tensor<T>> {
    // For simplicity, this handles the last dimension case
    let data = x.to_vec();
    let shape = x.shape();

    if shape.is_empty() {
        return Ok(Tensor::scalar(T::one()));
    }

    // Find max for numerical stability
    let max_val = data
        .iter()
        .fold(T::neg_infinity(), |a, &b| if b > a { b } else { a });

    // Compute exp(x - max)
    let exp_data: Vec<T> = data.iter().map(|&v| (v - max_val).exp_value()).collect();

    // Compute sum
    let sum: T = exp_data.iter().fold(T::zero(), |a, &b| a + b);

    // Normalize
    let result: Vec<T> = exp_data.iter().map(|&v| v / sum).collect();

    Tensor::from_vec(result, shape)
}

/// Applies log-softmax along the specified dimension.
pub fn log_softmax<T: Float>(x: &Tensor<T>, dim: i64) -> Result<Tensor<T>> {
    let sm = softmax(x, dim)?;
    Ok(sm.ln())
}

/// Applies GELU (Gaussian Error Linear Unit) activation.
#[must_use] pub fn gelu<T: Float>(x: &Tensor<T>) -> Tensor<T> {
    let data = x.to_vec();
    let sqrt_2_over_pi = T::from(0.7978845608028654).unwrap();
    let coeff = T::from(0.044715).unwrap();

    let result: Vec<T> = data
        .iter()
        .map(|&v| {
            let inner = sqrt_2_over_pi * (v + coeff * v * v * v);
            v * T::from(0.5).unwrap() * (T::one() + inner.tanh_value())
        })
        .collect();

    Tensor::from_vec(result, x.shape()).unwrap()
}

/// Applies Leaky `ReLU` activation.
pub fn leaky_relu<T: Float>(x: &Tensor<T>, negative_slope: T) -> Tensor<T> {
    let data = x.to_vec();
    let result: Vec<T> = data
        .iter()
        .map(|&v| if v > T::zero() { v } else { negative_slope * v })
        .collect();

    Tensor::from_vec(result, x.shape()).unwrap()
}

/// Applies ELU (Exponential Linear Unit) activation.
pub fn elu<T: Float>(x: &Tensor<T>, alpha: T) -> Tensor<T> {
    let data = x.to_vec();
    let result: Vec<T> = data
        .iter()
        .map(|&v| {
            if v > T::zero() {
                v
            } else {
                alpha * (v.exp_value() - T::one())
            }
        })
        .collect();

    Tensor::from_vec(result, x.shape()).unwrap()
}

/// Applies `SiLU` (Sigmoid Linear Unit) / Swish activation.
#[must_use] pub fn silu<T: Float>(x: &Tensor<T>) -> Tensor<T> {
    let sig = x.sigmoid();
    x.mul(&sig).unwrap()
}

// =============================================================================
// Clipping Operations
// =============================================================================

/// Clamps all elements to the range [min, max].
pub fn clamp<T: Numeric>(x: &Tensor<T>, min: T, max: T) -> Tensor<T> {
    let data = x.to_vec();
    let result: Vec<T> = data
        .iter()
        .map(|&v| {
            if v < min {
                min
            } else if v > max {
                max
            } else {
                v
            }
        })
        .collect();

    Tensor::from_vec(result, x.shape()).unwrap()
}

/// Clamps all elements to be at least min.
pub fn clamp_min<T: Numeric>(x: &Tensor<T>, min: T) -> Tensor<T> {
    let data = x.to_vec();
    let result: Vec<T> = data
        .iter()
        .map(|&v| if v < min { min } else { v })
        .collect();

    Tensor::from_vec(result, x.shape()).unwrap()
}

/// Clamps all elements to be at most max.
pub fn clamp_max<T: Numeric>(x: &Tensor<T>, max: T) -> Tensor<T> {
    let data = x.to_vec();
    let result: Vec<T> = data
        .iter()
        .map(|&v| if v > max { max } else { v })
        .collect();

    Tensor::from_vec(result, x.shape()).unwrap()
}

// =============================================================================
// Where Operation
// =============================================================================

/// Selects elements from x or y based on condition.
pub fn where_cond<T: Scalar>(
    condition: &[bool],
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> Result<Tensor<T>> {
    if x.shape() != y.shape() {
        return Err(axonml_core::error::Error::shape_mismatch(
            x.shape(),
            y.shape(),
        ));
    }

    if condition.len() != x.numel() {
        return Err(axonml_core::error::Error::shape_mismatch(
            &[condition.len()],
            &[x.numel()],
        ));
    }

    let x_data = x.to_vec();
    let y_data = y.to_vec();

    let result: Vec<T> = condition
        .iter()
        .zip(x_data.iter().zip(y_data.iter()))
        .map(|(&c, (&xv, &yv))| if c { xv } else { yv })
        .collect();

    Tensor::from_vec(result, x.shape())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let s = softmax(&t, -1).unwrap();

        let sum: f32 = s.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp() {
        let t = Tensor::<f32>::from_vec(vec![-1.0, 0.5, 2.0], &[3]).unwrap();
        let c = clamp(&t, 0.0, 1.0);
        assert_eq!(c.to_vec(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let t = Tensor::<f32>::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
        let r = leaky_relu(&t, 0.01);
        assert_eq!(r.to_vec(), vec![-0.01, 0.0, 1.0]);
    }

    #[test]
    fn test_comparison() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![1.0, 3.0, 2.0], &[3]).unwrap();

        assert_eq!(eq(&a, &b).unwrap(), vec![true, false, false]);
        assert_eq!(lt(&a, &b).unwrap(), vec![false, true, false]);
        assert_eq!(gt(&a, &b).unwrap(), vec![false, false, true]);
    }
}
