//! Views, slicing, and advanced indexing operations on `Tensor<T>`.
//!
//! 486 lines. `narrow` (sub-range along a dim), `select` (index one dim
//! away), `index_select` (gather along a dim by index tensor), `chunk`
//! (split into N equal parts), `split` (split by explicit size), `gather`
//! (index into data along a dim), `scatter` (write values at indexed
//! positions), `nonzero` (coordinates of non-zero elements), `unique`
//! (sorted deduplicated values). All return new `Tensor<T>` instances;
//! narrow/select use strided zero-copy views when possible.
//!
//! # File
//! `crates/axonml-tensor/src/view.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::alloc_prelude::*;

use axonml_core::dtype::{Numeric, Scalar};
use axonml_core::error::{Error, Result};

use crate::shape::{Shape, numel};
use crate::tensor::Tensor;

// =============================================================================
// Slice Specification
// =============================================================================

/// Specifies how to slice along a single dimension.
#[derive(Debug, Clone, Copy)]
pub enum SliceSpec {
    /// Select a single index, reducing dimensionality.
    Index(isize),
    /// Select a range [start, stop) with optional step.
    Range {
        /// Start index (inclusive), None = beginning
        start: Option<isize>,
        /// Stop index (exclusive), None = end
        stop: Option<isize>,
        /// Step size, default 1
        step: isize,
    },
    /// Keep all elements in this dimension.
    All,
    /// Add a new dimension of size 1.
    NewAxis,
}

impl SliceSpec {
    /// Creates a range slice from start to stop.
    #[must_use]
    pub fn range(start: isize, stop: isize) -> Self {
        Self::Range {
            start: Some(start),
            stop: Some(stop),
            step: 1,
        }
    }

    /// Creates a range slice with step.
    #[must_use]
    pub fn range_step(start: isize, stop: isize, step: isize) -> Self {
        Self::Range {
            start: Some(start),
            stop: Some(stop),
            step,
        }
    }

    /// Creates a slice from start to end.
    #[must_use]
    pub fn from(start: isize) -> Self {
        Self::Range {
            start: Some(start),
            stop: None,
            step: 1,
        }
    }

    /// Creates a slice from beginning to stop.
    #[must_use]
    pub fn to(stop: isize) -> Self {
        Self::Range {
            start: None,
            stop: Some(stop),
            step: 1,
        }
    }
}

// =============================================================================
// Slicing Implementation
// =============================================================================

impl<T: Scalar> Tensor<T> {
    /// Returns a slice of the tensor along the first dimension.
    ///
    /// # Arguments
    /// * `start` - Start index (inclusive)
    /// * `end` - End index (exclusive)
    pub fn slice_dim0(&self, start: usize, end: usize) -> Result<Self> {
        if self.ndim() == 0 {
            return Err(Error::invalid_operation("Cannot slice a scalar"));
        }

        let dim_size = self.shape[0];
        if start > end || end > dim_size {
            return Err(Error::IndexOutOfBounds {
                index: end,
                size: dim_size,
            });
        }

        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;

        let new_offset = self.offset + start * self.strides[0] as usize;

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }

    /// Returns a view selecting a single index along a dimension.
    ///
    /// This reduces the dimensionality by 1.
    ///
    /// # Arguments
    /// * `dim` - Dimension to select from
    /// * `index` - Index to select
    pub fn select(&self, dim: usize, index: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(Error::InvalidDimension {
                index: dim as i64,
                ndim: self.ndim(),
            });
        }

        if index >= self.shape[dim] {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.shape[dim],
            });
        }

        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);

        let mut new_strides = self.strides.clone();
        new_strides.remove(dim);

        let new_offset = self.offset + index * self.strides[dim] as usize;

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
        })
    }

    /// Returns a narrow view along a dimension.
    ///
    /// # Arguments
    /// * `dim` - Dimension to narrow
    /// * `start` - Start index
    /// * `length` - Length of the narrow view
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(Error::InvalidDimension {
                index: dim as i64,
                ndim: self.ndim(),
            });
        }

        if start + length > self.shape[dim] {
            return Err(Error::IndexOutOfBounds {
                index: start + length,
                size: self.shape[dim],
            });
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = length;

        let new_offset = self.offset + start * self.strides[dim] as usize;

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }

    /// Splits the tensor into chunks along a dimension.
    ///
    /// # Arguments
    /// * `chunks` - Number of chunks
    /// * `dim` - Dimension to split along
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        if dim >= self.ndim() {
            return Err(Error::InvalidDimension {
                index: dim as i64,
                ndim: self.ndim(),
            });
        }

        let dim_size = self.shape[dim];
        let chunk_size = dim_size.div_ceil(chunks);
        let mut result = Vec::with_capacity(chunks);

        let mut start = 0;
        while start < dim_size {
            let length = (chunk_size).min(dim_size - start);
            result.push(self.narrow(dim, start, length)?);
            start += length;
        }

        Ok(result)
    }

    /// Splits the tensor into parts of specified sizes along a dimension.
    ///
    /// # Arguments
    /// * `sizes` - Size of each part
    /// * `dim` - Dimension to split along
    pub fn split(&self, sizes: &[usize], dim: usize) -> Result<Vec<Self>> {
        if dim >= self.ndim() {
            return Err(Error::InvalidDimension {
                index: dim as i64,
                ndim: self.ndim(),
            });
        }

        let total: usize = sizes.iter().sum();
        if total != self.shape[dim] {
            return Err(Error::invalid_operation(format!(
                "Split sizes {} don't sum to dimension size {}",
                total, self.shape[dim]
            )));
        }

        let mut result = Vec::with_capacity(sizes.len());
        let mut start = 0;

        for &size in sizes {
            result.push(self.narrow(dim, start, size)?);
            start += size;
        }

        Ok(result)
    }
}

// =============================================================================
// Indexing Implementation
// =============================================================================

impl<T: Numeric> Tensor<T> {
    /// Gathers values along a dimension according to indices.
    ///
    /// # Arguments
    /// * `dim` - Dimension to gather along
    /// * `indices` - Indices tensor
    pub fn gather(&self, dim: usize, indices: &Tensor<i64>) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(Error::InvalidDimension {
                index: dim as i64,
                ndim: self.ndim(),
            });
        }

        let output_shape = indices.shape();
        let mut output_data = vec![T::zero(); numel(output_shape)];

        let indices_data = indices.to_vec();
        let self_data = self.to_vec();

        for (out_idx, &index) in indices_data.iter().enumerate() {
            let index = index as usize;
            if index >= self.shape[dim] {
                return Err(Error::IndexOutOfBounds {
                    index,
                    size: self.shape[dim],
                });
            }
            output_data[out_idx] = self_data[index];
        }

        Tensor::from_vec(output_data, output_shape)
    }

    /// Returns elements selected by a boolean mask.
    ///
    /// # Arguments
    /// * `mask` - Boolean mask tensor
    pub fn masked_select(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.numel() {
            return Err(Error::shape_mismatch(&[mask.len()], &[self.numel()]));
        }

        let data = self.to_vec();
        let selected: Vec<T> = data
            .into_iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m)
            .map(|(v, _)| v)
            .collect();

        let len = selected.len();
        Tensor::from_vec(selected, &[len])
    }

    /// Sets elements according to a boolean mask.
    ///
    /// # Arguments
    /// * `mask` - Boolean mask tensor
    /// * `value` - Value to set where mask is true
    pub fn masked_fill_(&self, mask: &[bool], value: T) -> Result<()> {
        if mask.len() != self.numel() {
            return Err(Error::shape_mismatch(&[mask.len()], &[self.numel()]));
        }

        if !self.is_contiguous() {
            return Err(Error::NotContiguous);
        }

        {
            let mut guard = self.storage.as_slice_mut();
            for (idx, &m) in mask.iter().enumerate() {
                if m {
                    guard[self.offset + idx] = value;
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// Concatenation and Stacking
// =============================================================================

/// Concatenates tensors along an existing dimension.
///
/// # Arguments
/// * `tensors` - Slice of tensors to concatenate
/// * `dim` - Dimension along which to concatenate
pub fn cat<T: Scalar>(tensors: &[Tensor<T>], dim: usize) -> Result<Tensor<T>> {
    if tensors.is_empty() {
        return Err(Error::invalid_operation("Cannot concatenate empty list"));
    }

    let first = &tensors[0];
    let ndim = first.ndim();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            index: dim as i64,
            ndim,
        });
    }

    for t in tensors.iter().skip(1) {
        if t.ndim() != ndim {
            return Err(Error::invalid_operation(
                "All tensors must have same number of dimensions",
            ));
        }
        for (d, (&s1, &s2)) in first.shape().iter().zip(t.shape().iter()).enumerate() {
            if d != dim && s1 != s2 {
                return Err(Error::shape_mismatch(first.shape(), t.shape()));
            }
        }
    }

    let mut output_shape = Shape::from_slice(first.shape());
    output_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();

    let total_numel = numel(&output_shape);
    let mut output_data = vec![T::zeroed(); total_numel];

    let outer: usize = first.shape()[..dim].iter().product();
    let inner: usize = first.shape()[dim + 1..].iter().product();
    let out_dim: usize = output_shape[dim];
    let mut dim_off = 0usize;
    for t in tensors {
        let data = t.to_vec();
        let ds = t.shape()[dim];
        for o in 0..outer {
            let src = &data[o * ds * inner..(o + 1) * ds * inner];
            let dst_start = (o * out_dim + dim_off) * inner;
            for (j, val) in src.iter().enumerate() {
                output_data[dst_start + j] = *val;
            }
        }
        dim_off += ds;
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Stacks tensors along a new dimension.
///
/// # Arguments
/// * `tensors` - Slice of tensors to stack
/// * `dim` - Dimension at which to insert the new axis
pub fn stack<T: Scalar>(tensors: &[Tensor<T>], dim: usize) -> Result<Tensor<T>> {
    if tensors.is_empty() {
        return Err(Error::invalid_operation("Cannot stack empty list"));
    }

    let unsqueezed: Result<Vec<Tensor<T>>> =
        tensors.iter().map(|t| t.unsqueeze(dim as i64)).collect();

    cat(&unsqueezed?, dim)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_dim0() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let s = t.slice_dim0(1, 3).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.get(&[0, 0]).unwrap(), 3.0);
    }

    #[test]
    fn test_select() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let s = t.select(0, 1).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_narrow() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        let n = t.narrow(0, 1, 3).unwrap();
        assert_eq!(n.shape(), &[3]);
        assert_eq!(n.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_chunk() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();

        let chunks = t.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].to_vec(), vec![1.0, 2.0]);
        assert_eq!(chunks[1].to_vec(), vec![3.0, 4.0]);
        assert_eq!(chunks[2].to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_cat() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap();

        let c = cat(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[4]);
        assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cat_inner_axis() {
        let a =
            Tensor::<f32>::from_vec((0..12).map(|x| x as f32).collect(), &[1, 3, 2, 2]).unwrap();
        let b =
            Tensor::<f32>::from_vec((100..106).map(|x| x as f32).collect(), &[1, 3, 1, 2]).unwrap();
        let c = cat(&[a, b], 2).unwrap();
        assert_eq!(c.shape(), &[1, 3, 3, 2]);
        assert_eq!(
            c.to_vec(),
            vec![
                0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 4.0, 5.0, 6.0, 7.0, 102.0, 103.0, 8.0, 9.0, 10.0,
                11.0, 104.0, 105.0,
            ]
        );
    }

    #[test]
    fn test_stack() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap();

        let c = stack(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }
}
