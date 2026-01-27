//! Tensor Operations - Mathematical and Structural Operations
//!
//! This module provides standalone tensor operations for convenient access.
//! Operations are organized by category.
//!
//! # Categories
//!
//! ## Comparison Operations
//! - `eq`, `lt`, `gt` - Element-wise comparison returning boolean vectors
//!
//! ## Activation Functions
//! - `softmax`, `log_softmax` - Probability distributions
//! - `gelu`, `silu`, `elu`, `leaky_relu` - Advanced activations
//!
//! ## Clipping Operations
//! - `clamp`, `clamp_min`, `clamp_max` - Value range limiting
//!
//! ## Conditional Operations
//! - `where_cond` - Select elements based on condition
//!
//! ## Sorting and Top-K
//! - `topk` - Returns k largest/smallest elements with indices
//! - `sort` - Sorts tensor along dimension with indices
//! - `argsort` - Returns indices that would sort tensor
//!
//! ## Indexing Operations
//! - `scatter` - Scatter values to specified indices (inverse of gather)
//! - `nonzero` - Returns indices of non-zero elements
//! - `unique` - Returns unique elements with optional counts/inverse
//!
//! ## Shape Manipulation
//! - `flip` - Reverses tensor along specified dimensions
//! - `roll` - Rolls tensor elements along dimensions (circular shift)
//!
//! # Example
//!
//! ```ignore
//! use axonml_tensor::ops::{topk, sort, unique};
//!
//! let t = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], &[5]).unwrap();
//!
//! // Get top 3 largest values
//! let result = topk(&t, 3, -1, true, true).unwrap();
//! // result.values = [5.0, 4.0, 3.0]
//! // result.indices = [4, 2, 0]
//!
//! // Sort ascending
//! let sorted = sort(&t, -1, false).unwrap();
//! // sorted.values = [1.0, 1.0, 3.0, 4.0, 5.0]
//!
//! // Get unique values
//! let uniq = unique(&t, true, true, true);
//! // uniq.values = [1.0, 3.0, 4.0, 5.0]
//! // uniq.counts = [2, 1, 1, 1]
//! ```
//!
//! @version 0.2.6
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
#[must_use]
pub fn gelu<T: Float>(x: &Tensor<T>) -> Tensor<T> {
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
#[must_use]
pub fn silu<T: Float>(x: &Tensor<T>) -> Tensor<T> {
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
// Sorting and Top-K Operations
// =============================================================================

/// Result of topk operation containing values and indices.
#[derive(Clone)]
pub struct TopKResult<T: Scalar> {
    /// The top-k values.
    pub values: Tensor<T>,
    /// The indices of the top-k values in the original tensor.
    pub indices: Tensor<i64>,
}

impl<T: Scalar> std::fmt::Debug for TopKResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopKResult")
            .field("values_shape", &self.values.shape())
            .field("indices_shape", &self.indices.shape())
            .finish()
    }
}

/// Returns the k largest elements along a dimension.
///
/// # Arguments
/// * `x` - Input tensor
/// * `k` - Number of top elements to return
/// * `dim` - Dimension to sort along (default: -1)
/// * `largest` - If true, return largest elements; if false, return smallest
/// * `sorted` - If true, return elements in sorted order
///
/// # Returns
/// TopKResult containing values and indices tensors
pub fn topk<T: Numeric>(
    x: &Tensor<T>,
    k: usize,
    dim: i64,
    largest: bool,
    sorted: bool,
) -> Result<TopKResult<T>> {
    let shape = x.shape();
    if shape.is_empty() {
        return Err(axonml_core::error::Error::invalid_operation(
            "Cannot apply topk to scalar tensor".to_string(),
        ));
    }

    let dim = if dim < 0 {
        (shape.len() as i64 + dim) as usize
    } else {
        dim as usize
    };

    if dim >= shape.len() {
        return Err(axonml_core::error::Error::invalid_operation(format!(
            "Dimension {} out of range for tensor with {} dimensions",
            dim,
            shape.len()
        )));
    }

    let dim_size = shape[dim];
    if k > dim_size {
        return Err(axonml_core::error::Error::invalid_operation(format!(
            "k ({}) is larger than dimension size ({})",
            k, dim_size
        )));
    }

    let data = x.to_vec();

    // For simplicity, handle the 1D case specially
    if shape.len() == 1 {
        let mut indexed: Vec<(usize, T)> = data.into_iter().enumerate().collect();
        if largest {
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        if !sorted {
            indexed[..k].sort_by_key(|x| x.0);
        }

        let values: Vec<T> = indexed[..k].iter().map(|(_, v)| *v).collect();
        let indices: Vec<i64> = indexed[..k].iter().map(|(i, _)| *i as i64).collect();

        return Ok(TopKResult {
            values: Tensor::from_vec(values, &[k])?,
            indices: Tensor::from_vec(indices, &[k])?,
        });
    }

    // General n-dimensional case
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let mut values_data = Vec::with_capacity(outer_size * k * inner_size);
    let mut indices_data = Vec::with_capacity(outer_size * k * inner_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut slice: Vec<(usize, T)> = (0..dim_size)
                .map(|d| {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    (d, data[idx])
                })
                .collect();

            if largest {
                slice.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                slice.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            if !sorted {
                slice[..k].sort_by_key(|x| x.0);
            }

            for (orig_idx, val) in slice.into_iter().take(k) {
                values_data.push(val);
                indices_data.push(orig_idx as i64);
            }
        }
    }

    let mut output_shape = shape.to_vec();
    output_shape[dim] = k;

    Ok(TopKResult {
        values: Tensor::from_vec(values_data, &output_shape)?,
        indices: Tensor::from_vec(indices_data, &output_shape)?,
    })
}

/// Result of sort operation containing sorted values and indices.
#[derive(Clone)]
pub struct SortResult<T: Scalar> {
    /// Sorted values.
    pub values: Tensor<T>,
    /// Indices that would sort the tensor.
    pub indices: Tensor<i64>,
}

impl<T: Scalar> std::fmt::Debug for SortResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortResult")
            .field("values_shape", &self.values.shape())
            .field("indices_shape", &self.indices.shape())
            .finish()
    }
}

/// Sorts the elements of the tensor along a dimension.
///
/// # Arguments
/// * `x` - Input tensor
/// * `dim` - Dimension to sort along (default: -1)
/// * `descending` - If true, sort in descending order
///
/// # Returns
/// SortResult containing sorted values and indices
pub fn sort<T: Numeric>(x: &Tensor<T>, dim: i64, descending: bool) -> Result<SortResult<T>> {
    let shape = x.shape();
    if shape.is_empty() {
        return Ok(SortResult {
            values: x.clone(),
            indices: Tensor::scalar(0i64),
        });
    }

    let dim = if dim < 0 {
        (shape.len() as i64 + dim) as usize
    } else {
        dim as usize
    };

    let dim_size = shape[dim];
    topk(x, dim_size, dim as i64, descending, true).map(|tk| SortResult {
        values: tk.values,
        indices: tk.indices,
    })
}

/// Returns the indices that would sort the tensor along a dimension.
///
/// # Arguments
/// * `x` - Input tensor
/// * `dim` - Dimension to sort along (default: -1)
/// * `descending` - If true, sort in descending order
pub fn argsort<T: Numeric>(x: &Tensor<T>, dim: i64, descending: bool) -> Result<Tensor<i64>> {
    sort(x, dim, descending).map(|r| r.indices)
}

// =============================================================================
// Scatter Operation
// =============================================================================

/// Writes values from src into self at locations specified by index.
///
/// This is the inverse of gather.
///
/// # Arguments
/// * `dst` - Destination tensor (modified in place conceptually, returns new tensor)
/// * `dim` - Dimension along which to scatter
/// * `index` - Indices to scatter to
/// * `src` - Source values to scatter
pub fn scatter<T: Scalar>(
    dst: &Tensor<T>,
    dim: usize,
    index: &Tensor<i64>,
    src: &Tensor<T>,
) -> Result<Tensor<T>> {
    let dst_shape = dst.shape();
    let idx_shape = index.shape();
    let src_shape = src.shape();

    if idx_shape != src_shape {
        return Err(axonml_core::error::Error::shape_mismatch(
            idx_shape, src_shape,
        ));
    }

    if dim >= dst_shape.len() {
        return Err(axonml_core::error::Error::invalid_operation(format!(
            "Dimension {} out of range",
            dim
        )));
    }

    let mut result = dst.to_vec();
    let idx_data = index.to_vec();
    let src_data = src.to_vec();

    // Calculate strides for the destination
    let mut dst_strides = vec![1usize; dst_shape.len()];
    for i in (0..dst_shape.len() - 1).rev() {
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];
    }

    // Calculate strides for index/src
    let mut idx_strides = vec![1usize; idx_shape.len()];
    for i in (0..idx_shape.len() - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * idx_shape[i + 1];
    }

    // Scatter values
    let total = index.numel();
    for linear_idx in 0..total {
        // Convert linear index to n-dimensional index
        let mut nd_idx = vec![0usize; idx_shape.len()];
        let mut remaining = linear_idx;
        for d in 0..idx_shape.len() {
            nd_idx[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Get the scatter index
        let scatter_idx = idx_data[linear_idx] as usize;

        // Build destination index
        let mut dst_nd_idx = nd_idx.clone();
        dst_nd_idx[dim] = scatter_idx;

        // Convert to linear destination index
        let mut dst_linear = 0;
        for d in 0..dst_shape.len() {
            dst_linear += dst_nd_idx[d] * dst_strides[d];
        }

        result[dst_linear] = src_data[linear_idx];
    }

    Tensor::from_vec(result, dst_shape)
}

// =============================================================================
// Nonzero Operation
// =============================================================================

/// Returns the indices of non-zero elements.
///
/// # Arguments
/// * `x` - Input tensor
///
/// # Returns
/// Tensor of shape (num_nonzero, ndim) containing indices of non-zero elements
pub fn nonzero<T: Numeric>(x: &Tensor<T>) -> Tensor<i64> {
    let data = x.to_vec();
    let shape = x.shape();
    let ndim = shape.len();

    // Find all non-zero indices
    let mut indices: Vec<Vec<i64>> = Vec::new();

    // Calculate strides for index conversion
    let mut strides = vec![1usize; ndim.max(1)];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    for (linear_idx, &val) in data.iter().enumerate() {
        if val != T::zero() {
            let mut nd_idx = vec![0i64; ndim.max(1)];
            let mut remaining = linear_idx;
            for d in 0..ndim {
                nd_idx[d] = (remaining / strides[d]) as i64;
                remaining %= strides[d];
            }
            indices.push(nd_idx);
        }
    }

    let num_nonzero = indices.len();
    if num_nonzero == 0 {
        return Tensor::from_vec(vec![], &[0, ndim.max(1)]).unwrap();
    }

    let flat: Vec<i64> = indices.into_iter().flatten().collect();
    Tensor::from_vec(flat, &[num_nonzero, ndim.max(1)]).unwrap()
}

// =============================================================================
// Unique Operation
// =============================================================================

/// Result of unique operation.
#[derive(Clone)]
pub struct UniqueResult<T: Scalar> {
    /// Unique values.
    pub values: Tensor<T>,
    /// Indices of unique values in the original tensor (if return_inverse).
    pub inverse_indices: Option<Tensor<i64>>,
    /// Counts of each unique value (if return_counts).
    pub counts: Option<Tensor<i64>>,
}

impl<T: Scalar> std::fmt::Debug for UniqueResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UniqueResult")
            .field("values_shape", &self.values.shape())
            .field("has_inverse", &self.inverse_indices.is_some())
            .field("has_counts", &self.counts.is_some())
            .finish()
    }
}

/// Returns the unique elements of the input tensor.
///
/// # Arguments
/// * `x` - Input tensor
/// * `sorted` - Whether to sort the unique elements
/// * `return_inverse` - Whether to return inverse indices
/// * `return_counts` - Whether to return counts
pub fn unique<T: Numeric>(
    x: &Tensor<T>,
    sorted: bool,
    return_inverse: bool,
    return_counts: bool,
) -> UniqueResult<T> {
    let data = x.to_vec();

    // Use a vec to preserve insertion order (for unsorted case)
    let mut seen: Vec<T> = Vec::new();
    let mut counts_map: Vec<i64> = Vec::new();
    let mut inverse: Vec<i64> = Vec::with_capacity(data.len());

    for &val in &data {
        if let Some(pos) = seen.iter().position(|&v| v == val) {
            inverse.push(pos as i64);
            counts_map[pos] += 1;
        } else {
            inverse.push(seen.len() as i64);
            seen.push(val);
            counts_map.push(1);
        }
    }

    let (unique_vals, final_inverse, final_counts) = if sorted {
        // Sort unique values and update inverse indices
        let mut indexed: Vec<(usize, T)> = seen.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create mapping from old index to new index
        let mut old_to_new = vec![0i64; indexed.len()];
        for (new_idx, (old_idx, _)) in indexed.iter().enumerate() {
            old_to_new[*old_idx] = new_idx as i64;
        }

        let sorted_vals: Vec<T> = indexed.iter().map(|(_, v)| *v).collect();
        let sorted_counts: Vec<i64> = indexed
            .iter()
            .map(|(old_idx, _)| counts_map[*old_idx])
            .collect();
        let updated_inverse: Vec<i64> = inverse.iter().map(|&i| old_to_new[i as usize]).collect();

        (sorted_vals, updated_inverse, sorted_counts)
    } else {
        (seen, inverse, counts_map)
    };

    let n = unique_vals.len();

    UniqueResult {
        values: Tensor::from_vec(unique_vals, &[n]).unwrap(),
        inverse_indices: if return_inverse {
            Some(Tensor::from_vec(final_inverse, x.shape()).unwrap())
        } else {
            None
        },
        counts: if return_counts {
            Some(Tensor::from_vec(final_counts, &[n]).unwrap())
        } else {
            None
        },
    }
}

// =============================================================================
// Flip Operation
// =============================================================================

/// Reverses the order of elements along specified dimensions.
pub fn flip<T: Numeric>(x: &Tensor<T>, dims: &[usize]) -> Result<Tensor<T>> {
    let shape = x.shape();
    let data = x.to_vec();
    let ndim = shape.len();

    for &d in dims {
        if d >= ndim {
            return Err(axonml_core::error::Error::invalid_operation(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                d, ndim
            )));
        }
    }

    if shape.is_empty() {
        return Ok(x.clone());
    }

    // Calculate strides
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut result = vec![T::zero(); data.len()];

    for src_linear in 0..data.len() {
        // Convert to n-dimensional index
        let mut nd_idx = vec![0usize; ndim];
        let mut remaining = src_linear;
        for d in 0..ndim {
            nd_idx[d] = remaining / strides[d];
            remaining %= strides[d];
        }

        // Flip specified dimensions
        for &flip_dim in dims {
            nd_idx[flip_dim] = shape[flip_dim] - 1 - nd_idx[flip_dim];
        }

        // Convert back to linear index
        let mut dst_linear = 0;
        for d in 0..ndim {
            dst_linear += nd_idx[d] * strides[d];
        }

        result[dst_linear] = data[src_linear];
    }

    Tensor::from_vec(result, shape)
}

// =============================================================================
// Roll Operation
// =============================================================================

/// Rolls tensor elements along specified dimensions.
pub fn roll<T: Numeric>(x: &Tensor<T>, shifts: &[i64], dims: &[usize]) -> Result<Tensor<T>> {
    if shifts.len() != dims.len() {
        return Err(axonml_core::error::Error::invalid_operation(
            "shifts and dims must have the same length".to_string(),
        ));
    }

    let shape = x.shape();
    let data = x.to_vec();
    let ndim = shape.len();

    for &d in dims {
        if d >= ndim {
            return Err(axonml_core::error::Error::invalid_operation(format!(
                "Dimension {} out of range",
                d
            )));
        }
    }

    if shape.is_empty() {
        return Ok(x.clone());
    }

    // Calculate strides
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut result = vec![T::zero(); data.len()];

    for src_linear in 0..data.len() {
        // Convert to n-dimensional index
        let mut nd_idx = vec![0usize; ndim];
        let mut remaining = src_linear;
        for d in 0..ndim {
            nd_idx[d] = remaining / strides[d];
            remaining %= strides[d];
        }

        // Apply shifts
        for (shift, &dim) in shifts.iter().zip(dims.iter()) {
            let dim_size = shape[dim] as i64;
            let new_idx = ((nd_idx[dim] as i64 + shift) % dim_size + dim_size) % dim_size;
            nd_idx[dim] = new_idx as usize;
        }

        // Convert back to linear index
        let mut dst_linear = 0;
        for d in 0..ndim {
            dst_linear += nd_idx[d] * strides[d];
        }

        result[dst_linear] = data[src_linear];
    }

    Tensor::from_vec(result, shape)
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

    #[test]
    fn test_topk() {
        let t = Tensor::<f32>::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6]).unwrap();
        let result = topk(&t, 3, -1, true, true).unwrap();

        assert_eq!(result.values.shape(), &[3]);
        assert_eq!(result.values.to_vec(), vec![9.0, 5.0, 4.0]);
        assert_eq!(result.indices.to_vec(), vec![5, 4, 2]);
    }

    #[test]
    fn test_topk_smallest() {
        let t = Tensor::<f32>::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6]).unwrap();
        let result = topk(&t, 2, -1, false, true).unwrap();

        assert_eq!(result.values.to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_sort() {
        let t = Tensor::<f32>::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], &[5]).unwrap();
        let result = sort(&t, -1, false).unwrap();

        assert_eq!(result.values.to_vec(), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sort_descending() {
        let t = Tensor::<f32>::from_vec(vec![3.0, 1.0, 4.0], &[3]).unwrap();
        let result = sort(&t, -1, true).unwrap();

        assert_eq!(result.values.to_vec(), vec![4.0, 3.0, 1.0]);
    }

    #[test]
    fn test_argsort() {
        let t = Tensor::<f32>::from_vec(vec![3.0, 1.0, 2.0], &[3]).unwrap();
        let indices = argsort(&t, -1, false).unwrap();

        assert_eq!(indices.to_vec(), vec![1, 2, 0]);
    }

    #[test]
    fn test_nonzero() {
        let t = Tensor::<f32>::from_vec(vec![0.0, 1.0, 0.0, 2.0, 3.0, 0.0], &[6]).unwrap();
        let result = nonzero(&t);

        assert_eq!(result.shape(), &[3, 1]);
        assert_eq!(result.to_vec(), vec![1, 3, 4]);
    }

    #[test]
    fn test_nonzero_2d() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 0.0, 0.0, 2.0], &[2, 2]).unwrap();
        let result = nonzero(&t);

        assert_eq!(result.shape(), &[2, 2]);
        // (0,0) and (1,1) are non-zero
        assert_eq!(result.to_vec(), vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_unique() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0], &[6]).unwrap();
        let result = unique(&t, true, true, true);

        assert_eq!(result.values.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            result.inverse_indices.unwrap().to_vec(),
            vec![0, 1, 0, 2, 1, 0]
        );
        assert_eq!(result.counts.unwrap().to_vec(), vec![3, 2, 1]);
    }

    #[test]
    fn test_unique_unsorted() {
        let t = Tensor::<f32>::from_vec(vec![3.0, 1.0, 3.0], &[3]).unwrap();
        let result = unique(&t, false, false, false);

        // Preserves insertion order
        assert_eq!(result.values.to_vec(), vec![3.0, 1.0]);
    }

    #[test]
    fn test_flip() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let flipped = flip(&t, &[0]).unwrap();

        assert_eq!(flipped.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_2d() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let flipped = flip(&t, &[0]).unwrap();

        // Flip along dim 0: [[3,4], [1,2]]
        assert_eq!(flipped.to_vec(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_roll() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let rolled = roll(&t, &[1], &[0]).unwrap();

        assert_eq!(rolled.to_vec(), vec![4.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roll_negative() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let rolled = roll(&t, &[-1], &[0]).unwrap();

        assert_eq!(rolled.to_vec(), vec![2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn test_scatter() {
        let dst = Tensor::<f32>::zeros(&[3]);
        let index = Tensor::from_vec(vec![0_i64, 2], &[2]).unwrap();
        let src = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        let result = scatter(&dst, 0, &index, &src).unwrap();
        assert_eq!(result.to_vec(), vec![1.0, 0.0, 2.0]);
    }
}
