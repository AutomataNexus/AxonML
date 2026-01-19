//! Shape and Strides - Tensor Dimension Management
//!
//! Provides types and functions for managing tensor shapes, strides, and
//! broadcasting rules. Shapes define the dimensions of a tensor, while
//! strides define how to traverse the underlying storage.
//!
//! # Key Features
//! - Efficient shape representation with small-vector optimization
//! - Stride computation for contiguous and transposed layouts
//! - Broadcasting support following `NumPy` rules
//! - Shape validation and manipulation
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use smallvec::SmallVec;

use axonml_core::error::{Error, Result};

// =============================================================================
// Type Aliases
// =============================================================================

/// Shape type - dimensions of a tensor.
/// Uses `SmallVec` for stack allocation of small shapes (up to 6 dimensions).
pub type Shape = SmallVec<[usize; 6]>;

/// Strides type - step sizes for each dimension.
pub type Strides = SmallVec<[isize; 6]>;

// =============================================================================
// Shape Utilities
// =============================================================================

/// Computes the total number of elements from a shape.
///
/// # Arguments
/// * `shape` - The tensor shape
///
/// # Returns
/// Total number of elements (product of dimensions).
#[must_use]
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Computes row-major (C-order) strides for a shape.
///
/// # Arguments
/// * `shape` - The tensor shape
///
/// # Returns
/// Strides for contiguous row-major layout.
#[must_use]
pub fn contiguous_strides(shape: &[usize]) -> Strides {
    if shape.is_empty() {
        return Strides::new();
    }

    let mut strides = Strides::with_capacity(shape.len());
    let mut stride = 1isize;

    // Compute strides from right to left
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim as isize;
    }

    strides.reverse();
    strides
}

/// Checks if strides represent a contiguous memory layout.
///
/// # Arguments
/// * `shape` - The tensor shape
/// * `strides` - The tensor strides
///
/// # Returns
/// True if the tensor is contiguous in memory.
#[must_use]
pub fn is_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    let expected = contiguous_strides(shape);
    strides == expected.as_slice()
}

/// Computes the linear index from multi-dimensional indices.
///
/// # Arguments
/// * `indices` - Multi-dimensional indices
/// * `strides` - Tensor strides
///
/// # Returns
/// Linear offset into storage.
#[must_use]
pub fn linear_index(indices: &[usize], strides: &[isize]) -> usize {
    debug_assert_eq!(indices.len(), strides.len());

    let mut offset = 0isize;
    for (&idx, &stride) in indices.iter().zip(strides.iter()) {
        offset += idx as isize * stride;
    }
    offset as usize
}

/// Converts a linear index to multi-dimensional indices.
///
/// # Arguments
/// * `linear` - Linear index
/// * `shape` - Tensor shape
///
/// # Returns
/// Multi-dimensional indices.
#[must_use]
pub fn unravel_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];

    for (i, &dim) in shape.iter().enumerate().rev() {
        indices[i] = linear % dim;
        linear /= dim;
    }

    indices
}

// =============================================================================
// Broadcasting
// =============================================================================

/// Computes the broadcast shape of two shapes.
///
/// Broadcasting follows `NumPy` rules:
/// 1. Shapes are aligned from the right
/// 2. Dimensions are compatible if equal or one of them is 1
/// 3. Missing dimensions are treated as 1
///
/// # Arguments
/// * `shape1` - First shape
/// * `shape2` - Second shape
///
/// # Returns
/// Broadcast shape, or error if shapes are incompatible.
pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Shape> {
    let max_ndim = shape1.len().max(shape2.len());
    let mut result = Shape::with_capacity(max_ndim);

    // Iterate from right to left
    for i in 0..max_ndim {
        let d1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };

        let d2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        if d1 == d2 {
            result.push(d1);
        } else if d1 == 1 {
            result.push(d2);
        } else if d2 == 1 {
            result.push(d1);
        } else {
            return Err(Error::BroadcastError {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
            });
        }
    }

    result.reverse();
    Ok(result)
}

/// Computes broadcast strides for a shape to match a target shape.
///
/// # Arguments
/// * `shape` - Original shape
/// * `strides` - Original strides
/// * `target_shape` - Target broadcast shape
///
/// # Returns
/// New strides for broadcasting (0 stride for broadcast dimensions).
#[must_use] pub fn broadcast_strides(shape: &[usize], strides: &[isize], target_shape: &[usize]) -> Strides {
    let mut result = Strides::with_capacity(target_shape.len());
    let shape_offset = target_shape.len() - shape.len();

    for (i, &target_dim) in target_shape.iter().enumerate() {
        if i < shape_offset {
            // Dimension doesn't exist in original - broadcast
            result.push(0);
        } else {
            let orig_idx = i - shape_offset;
            let orig_dim = shape[orig_idx];

            if orig_dim == target_dim {
                result.push(strides[orig_idx]);
            } else if orig_dim == 1 {
                // Broadcast dimension
                result.push(0);
            } else {
                // Should not happen if broadcast_shape was computed correctly
                result.push(strides[orig_idx]);
            }
        }
    }

    result
}

/// Checks if two shapes are broadcastable.
#[must_use]
pub fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
    broadcast_shape(shape1, shape2).is_ok()
}

// =============================================================================
// Shape Manipulation
// =============================================================================

/// Reshapes a tensor shape, validating that total elements match.
///
/// Supports -1 in one dimension to infer the size.
///
/// # Arguments
/// * `old_shape` - Current shape
/// * `new_shape` - Target shape (can contain -1)
///
/// # Returns
/// Resolved shape, or error if incompatible.
pub fn reshape(old_shape: &[usize], new_shape: &[isize]) -> Result<Shape> {
    let old_numel = numel(old_shape);
    let mut result = Shape::with_capacity(new_shape.len());
    let mut infer_idx = None;
    let mut known_numel = 1usize;

    for (i, &dim) in new_shape.iter().enumerate() {
        if dim == -1 {
            if infer_idx.is_some() {
                return Err(Error::invalid_operation("Can only have one -1 in reshape"));
            }
            infer_idx = Some(i);
            result.push(0); // Placeholder
        } else if dim < 0 {
            return Err(Error::invalid_operation("Invalid dimension in reshape"));
        } else {
            let d = dim as usize;
            known_numel *= d;
            result.push(d);
        }
    }

    if let Some(idx) = infer_idx {
        if old_numel % known_numel != 0 {
            return Err(Error::invalid_operation(
                "Cannot infer dimension: not evenly divisible",
            ));
        }
        result[idx] = old_numel / known_numel;
    } else if known_numel != old_numel {
        return Err(Error::shape_mismatch(old_shape, &result));
    }

    Ok(result)
}

/// Computes the shape after squeezing (removing dimensions of size 1).
///
/// # Arguments
/// * `shape` - Input shape
/// * `dim` - Optional dimension to squeeze (None = all)
///
/// # Returns
/// Squeezed shape.
#[must_use]
pub fn squeeze(shape: &[usize], dim: Option<usize>) -> Shape {
    match dim {
        Some(d) => {
            let mut result = Shape::from_slice(shape);
            if d < shape.len() && shape[d] == 1 {
                result.remove(d);
            }
            result
        }
        None => shape.iter().copied().filter(|&d| d != 1).collect(),
    }
}

/// Computes the shape after unsqueezing (adding a dimension of size 1).
///
/// # Arguments
/// * `shape` - Input shape
/// * `dim` - Dimension at which to insert
///
/// # Returns
/// Unsqueezed shape, or error if dim is invalid.
pub fn unsqueeze(shape: &[usize], dim: usize) -> Result<Shape> {
    if dim > shape.len() {
        return Err(Error::InvalidDimension {
            index: dim as i64,
            ndim: shape.len(),
        });
    }

    let mut result = Shape::with_capacity(shape.len() + 1);
    result.extend_from_slice(&shape[..dim]);
    result.push(1);
    result.extend_from_slice(&shape[dim..]);
    Ok(result)
}

/// Computes the shape after transposing dimensions.
///
/// # Arguments
/// * `shape` - Input shape
/// * `dim0` - First dimension
/// * `dim1` - Second dimension
///
/// # Returns
/// Transposed shape and strides modifier.
pub fn transpose_shape(shape: &[usize], dim0: usize, dim1: usize) -> Result<Shape> {
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(Error::InvalidDimension {
            index: dim0.max(dim1) as i64,
            ndim: shape.len(),
        });
    }

    let mut result = Shape::from_slice(shape);
    result.swap(dim0, dim1);
    Ok(result)
}

/// Swaps two stride values.
#[must_use] pub fn transpose_strides(strides: &[isize], dim0: usize, dim1: usize) -> Strides {
    let mut result = Strides::from_slice(strides);
    result.swap(dim0, dim1);
    result
}

// =============================================================================
// Validation
// =============================================================================

/// Normalizes a dimension index, supporting negative indexing.
///
/// # Arguments
/// * `dim` - Dimension index (can be negative)
/// * `ndim` - Number of dimensions
///
/// # Returns
/// Normalized positive index, or error if out of bounds.
pub fn normalize_dim(dim: i64, ndim: usize) -> Result<usize> {
    let ndim_i64 = ndim as i64;

    let normalized = if dim < 0 { dim + ndim_i64 } else { dim };

    if normalized < 0 || normalized >= ndim_i64 {
        return Err(Error::InvalidDimension { index: dim, ndim });
    }

    Ok(normalized as usize)
}

/// Validates that indices are within bounds for a shape.
pub fn validate_indices(indices: &[usize], shape: &[usize]) -> Result<()> {
    if indices.len() != shape.len() {
        return Err(Error::invalid_operation(format!(
            "Expected {} indices, got {}",
            shape.len(),
            indices.len()
        )));
    }

    for (&idx, &dim) in indices.iter().zip(shape.iter()) {
        if idx >= dim {
            return Err(Error::IndexOutOfBounds {
                index: idx,
                size: dim,
            });
        }
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numel() {
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[5]), 5);
    }

    #[test]
    fn test_contiguous_strides() {
        let shape = [2, 3, 4];
        let strides = contiguous_strides(&shape);
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn test_is_contiguous() {
        let shape = [2, 3];
        let strides = contiguous_strides(&shape);
        assert!(is_contiguous(&shape, &strides));

        let non_contig_strides: Strides = smallvec::smallvec![1, 2];
        assert!(!is_contiguous(&shape, &non_contig_strides));
    }

    #[test]
    fn test_broadcast_shape() {
        // Same shapes
        assert_eq!(
            broadcast_shape(&[2, 3], &[2, 3]).unwrap().as_slice(),
            &[2, 3]
        );

        // Broadcasting
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap().as_slice(), &[2, 3]);

        assert_eq!(
            broadcast_shape(&[2, 1], &[1, 3]).unwrap().as_slice(),
            &[2, 3]
        );

        assert_eq!(
            broadcast_shape(&[5, 1, 3], &[2, 3]).unwrap().as_slice(),
            &[5, 2, 3]
        );

        // Incompatible
        assert!(broadcast_shape(&[2, 3], &[2, 4]).is_err());
    }

    #[test]
    fn test_reshape() {
        let old_shape = [2, 3, 4];

        // Simple reshape
        let new = reshape(&old_shape, &[6, 4]).unwrap();
        assert_eq!(new.as_slice(), &[6, 4]);

        // With -1 inference
        let new = reshape(&old_shape, &[-1, 4]).unwrap();
        assert_eq!(new.as_slice(), &[6, 4]);

        // Invalid
        assert!(reshape(&old_shape, &[5, 5]).is_err());
    }

    #[test]
    fn test_squeeze() {
        let shape = [1, 2, 1, 3, 1];

        // Squeeze all
        let squeezed = squeeze(&shape, None);
        assert_eq!(squeezed.as_slice(), &[2, 3]);

        // Squeeze specific dimension
        let squeezed = squeeze(&shape, Some(0));
        assert_eq!(squeezed.as_slice(), &[2, 1, 3, 1]);
    }

    #[test]
    fn test_unsqueeze() {
        let shape = [2, 3];

        let unsqueezed = unsqueeze(&shape, 0).unwrap();
        assert_eq!(unsqueezed.as_slice(), &[1, 2, 3]);

        let unsqueezed = unsqueeze(&shape, 1).unwrap();
        assert_eq!(unsqueezed.as_slice(), &[2, 1, 3]);

        let unsqueezed = unsqueeze(&shape, 2).unwrap();
        assert_eq!(unsqueezed.as_slice(), &[2, 3, 1]);
    }

    #[test]
    fn test_normalize_dim() {
        assert_eq!(normalize_dim(0, 3).unwrap(), 0);
        assert_eq!(normalize_dim(-1, 3).unwrap(), 2);
        assert_eq!(normalize_dim(-3, 3).unwrap(), 0);

        assert!(normalize_dim(3, 3).is_err());
        assert!(normalize_dim(-4, 3).is_err());
    }

    #[test]
    fn test_linear_index() {
        // 2x3 matrix, row-major
        let strides: Strides = smallvec::smallvec![3, 1];

        assert_eq!(linear_index(&[0, 0], &strides), 0);
        assert_eq!(linear_index(&[0, 1], &strides), 1);
        assert_eq!(linear_index(&[1, 0], &strides), 3);
        assert_eq!(linear_index(&[1, 2], &strides), 5);
    }

    #[test]
    fn test_unravel_index() {
        let shape = [2, 3, 4];

        assert_eq!(unravel_index(0, &shape), vec![0, 0, 0]);
        assert_eq!(unravel_index(1, &shape), vec![0, 0, 1]);
        assert_eq!(unravel_index(4, &shape), vec![0, 1, 0]);
        assert_eq!(unravel_index(12, &shape), vec![1, 0, 0]);
    }
}
