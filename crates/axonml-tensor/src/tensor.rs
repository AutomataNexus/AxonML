//! Tensor - Core N-Dimensional Array Type
//!
//! The `Tensor` struct is the fundamental data structure in Axonml. It represents
//! an N-dimensional array of numeric values with support for automatic broadcasting,
//! device placement, and efficient memory sharing through views.
//!
//! # Key Features
//! - Generic over element type (f32, f64, i32, etc.)
//! - Efficient views with shared storage
//! - Device-agnostic operations
//! - Broadcasting support
//! - Lazy operations where possible
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use axonml_core::backends::CpuBackend;
use axonml_core::dtype::{Float, Numeric, Scalar};
use axonml_core::error::{Error, Result};
use axonml_core::storage::Storage;
use axonml_core::Device;
use num_traits::NumCast;

use crate::shape::{
    broadcast_shape, broadcast_strides, contiguous_strides, is_contiguous, linear_index,
    normalize_dim, numel, reshape, squeeze, transpose_shape, transpose_strides, unsqueeze, Shape,
    Strides,
};

// =============================================================================
// Tensor Struct
// =============================================================================

/// An N-dimensional array of numeric values.
///
/// Tensors are the core data structure for all computations in Axonml.
/// They support arbitrary dimensions, automatic broadcasting, and efficient
/// memory sharing between views.
#[derive(Clone)]
pub struct Tensor<T: Scalar> {
    /// Underlying data storage (reference-counted).
    pub(crate) storage: Storage<T>,
    /// Shape of the tensor (dimensions).
    pub(crate) shape: Shape,
    /// Strides for each dimension.
    pub(crate) strides: Strides,
    /// Offset into storage (for views).
    pub(crate) offset: usize,
}

impl<T: Scalar> Tensor<T> {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Creates a new tensor from storage with the given shape.
    ///
    /// # Arguments
    /// * `storage` - The underlying data storage
    /// * `shape` - Shape of the tensor
    ///
    /// # Returns
    /// New tensor, or error if shape doesn't match storage size.
    pub fn from_storage(storage: Storage<T>, shape: &[usize]) -> Result<Self> {
        let total = numel(shape);
        if total != storage.len() {
            return Err(Error::shape_mismatch(&[storage.len()], shape));
        }

        let shape = Shape::from_slice(shape);
        let strides = contiguous_strides(&shape);

        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Creates a new tensor from a vector with the given shape.
    ///
    /// # Arguments
    /// * `data` - Vector of data
    /// * `shape` - Shape of the tensor
    ///
    /// # Returns
    /// New tensor, or error if shape doesn't match data length.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let storage = Storage::from_vec(data, Device::Cpu);
        Self::from_storage(storage, shape)
    }

    /// Creates a new tensor from a slice with the given shape.
    ///
    /// # Arguments
    /// * `data` - Slice of data to copy
    /// * `shape` - Shape of the tensor
    ///
    /// # Returns
    /// New tensor, or error if shape doesn't match data length.
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Self> {
        let storage = Storage::from_slice(data, Device::Cpu);
        Self::from_storage(storage, shape)
    }

    /// Creates a scalar tensor (0-dimensional).
    ///
    /// # Arguments
    /// * `value` - The scalar value
    ///
    /// # Returns
    /// New 0-dimensional tensor.
    pub fn scalar(value: T) -> Self {
        Self {
            storage: Storage::from_vec(vec![value], Device::Cpu),
            shape: Shape::new(),
            strides: Strides::new(),
            offset: 0,
        }
    }

    /// Creates a tensor filled with zeros.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        crate::creation::zeros(shape)
    }

    /// Creates a tensor filled with ones.
    #[must_use]
    pub fn ones(shape: &[usize]) -> Self
    where
        T: Numeric,
    {
        crate::creation::ones(shape)
    }

    /// Creates a tensor filled with a constant value.
    #[must_use]
    pub fn full(shape: &[usize], value: T) -> Self {
        crate::creation::full(shape, value)
    }

    /// Creates a tensor with random values from standard normal distribution.
    #[must_use]
    pub fn randn(shape: &[usize]) -> Self
    where
        T: Float,
        rand_distr::StandardNormal: rand::distributions::Distribution<T>,
    {
        crate::creation::randn(shape)
    }

    /// Creates a tensor with random values from uniform distribution [0, 1).
    #[must_use]
    pub fn rand(shape: &[usize]) -> Self
    where
        T: Float,
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        crate::creation::rand(shape)
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /// Returns the shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides of the tensor.
    #[must_use]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    /// Returns true if the tensor is empty (has zero elements).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Returns the size of a specific dimension.
    ///
    /// # Arguments
    /// * `dim` - Dimension index (supports negative indexing)
    pub fn size(&self, dim: i64) -> Result<usize> {
        let idx = normalize_dim(dim, self.ndim())?;
        Ok(self.shape[idx])
    }

    /// Returns the device this tensor is on.
    #[must_use]
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Returns true if the tensor is contiguous in memory.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.shape, &self.strides)
    }

    /// Returns true if this tensor is a scalar (0-dimensional).
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    // =========================================================================
    // Data Access
    // =========================================================================

    /// Returns the element at the given indices.
    ///
    /// # Arguments
    /// * `indices` - Multi-dimensional indices
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        if indices.len() != self.ndim() {
            return Err(Error::invalid_operation(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        for (&idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim {
                return Err(Error::IndexOutOfBounds {
                    index: idx,
                    size: dim,
                });
            }
        }

        let offset = self.offset + linear_index(indices, &self.strides);
        Ok(self.storage.as_slice()[offset])
    }

    /// Sets the element at the given indices.
    ///
    /// # Arguments
    /// * `indices` - Multi-dimensional indices
    /// * `value` - Value to set
    pub fn set(&self, indices: &[usize], value: T) -> Result<()> {
        if indices.len() != self.ndim() {
            return Err(Error::invalid_operation(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        for (&idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim {
                return Err(Error::IndexOutOfBounds {
                    index: idx,
                    size: dim,
                });
            }
        }

        let offset = self.offset + linear_index(indices, &self.strides);
        self.storage.as_slice_mut()[offset] = value;
        Ok(())
    }

    /// Returns the scalar value for a 0-dimensional tensor.
    pub fn item(&self) -> Result<T> {
        if self.numel() != 1 {
            return Err(Error::invalid_operation(
                "item() only works on single-element tensors",
            ));
        }

        if self.is_scalar() {
            Ok(self.storage.as_slice()[self.offset])
        } else {
            // Single element but not scalar shape
            let indices = vec![0; self.ndim()];
            self.get(&indices)
        }
    }

    /// Returns the data as a contiguous vector.
    ///
    /// If the tensor is already contiguous, this returns a reference.
    /// Otherwise, it copies the data into a new contiguous vector.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        if self.is_contiguous() {
            let storage = self.storage.as_slice();
            storage[self.offset..self.offset + self.numel()].to_vec()
        } else {
            let mut result = Vec::with_capacity(self.numel());
            self.copy_data_to(&mut result);
            result
        }
    }

    /// Copies data to a slice, handling non-contiguous layouts.
    fn copy_data_to(&self, dst: &mut Vec<T>) {
        dst.clear();
        let storage = self.storage.as_slice();

        // Iterate through all indices
        let total = self.numel();
        for i in 0..total {
            let indices = crate::shape::unravel_index(i, &self.shape);
            let offset = self.offset + linear_index(&indices, &self.strides);
            dst.push(storage[offset]);
        }
    }

    // =========================================================================
    // Shape Operations
    // =========================================================================

    /// Returns a new tensor with the specified shape.
    ///
    /// The total number of elements must remain the same.
    /// Supports -1 in one dimension to infer the size.
    ///
    /// # Arguments
    /// * `new_shape` - Target shape
    pub fn reshape(&self, new_shape: &[isize]) -> Result<Self> {
        let shape = reshape(&self.shape, new_shape)?;

        if self.is_contiguous() {
            // Can just change shape without copying
            Ok(Self {
                storage: self.storage.clone(),
                strides: contiguous_strides(&shape),
                shape,
                offset: self.offset,
            })
        } else {
            // Need to make contiguous first
            let contig = self.contiguous();
            Ok(Self {
                storage: contig.storage,
                strides: contiguous_strides(&shape),
                shape,
                offset: 0,
            })
        }
    }

    /// Returns a new tensor with a flattened shape.
    #[must_use]
    pub fn flatten(&self) -> Self {
        self.reshape(&[-1]).expect("Flatten should never fail")
    }

    /// Returns a new tensor with dimensions of size 1 removed.
    ///
    /// # Arguments
    /// * `dim` - Optional specific dimension to squeeze
    pub fn squeeze(&self, dim: Option<i64>) -> Result<Self> {
        let dim = match dim {
            Some(d) => Some(normalize_dim(d, self.ndim())?),
            None => None,
        };

        let new_shape = squeeze(&self.shape, dim);
        let new_strides: Strides = match dim {
            Some(d) => {
                let mut s = self.strides.clone();
                if d < self.shape.len() && self.shape[d] == 1 {
                    s.remove(d);
                }
                s
            }
            None => self
                .shape
                .iter()
                .zip(self.strides.iter())
                .filter(|(&dim, _)| dim != 1)
                .map(|(_, &stride)| stride)
                .collect(),
        };

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Returns a new tensor with a dimension of size 1 inserted.
    ///
    /// # Arguments
    /// * `dim` - Position to insert the new dimension
    pub fn unsqueeze(&self, dim: i64) -> Result<Self> {
        let normalized = if dim < 0 {
            (dim + self.ndim() as i64 + 1) as usize
        } else {
            dim as usize
        };

        let new_shape = unsqueeze(&self.shape, normalized)?;
        let mut new_strides = Strides::with_capacity(new_shape.len());

        for (i, _) in new_shape.iter().enumerate() {
            if i < normalized {
                new_strides.push(self.strides.get(i).copied().unwrap_or(1));
            } else if i == normalized {
                // Stride for new dimension (doesn't matter since size is 1)
                new_strides.push(1);
            } else {
                new_strides.push(self.strides[i - 1]);
            }
        }

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Transposes two dimensions.
    ///
    /// # Arguments
    /// * `dim0` - First dimension
    /// * `dim1` - Second dimension
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<Self> {
        let d0 = normalize_dim(dim0, self.ndim())?;
        let d1 = normalize_dim(dim1, self.ndim())?;

        let new_shape = transpose_shape(&self.shape, d0, d1)?;
        let new_strides = transpose_strides(&self.strides, d0, d1);

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Returns the transpose of a 2D tensor.
    pub fn t(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(Error::invalid_operation("t() only works on 2D tensors"));
        }
        self.transpose(0, 1)
    }

    /// Returns a permuted tensor with dimensions reordered.
    ///
    /// # Arguments
    /// * `dims` - New order of dimensions
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.ndim() {
            return Err(Error::invalid_operation(format!(
                "Expected {} dimensions, got {}",
                self.ndim(),
                dims.len()
            )));
        }

        // Check that dims is a permutation
        let mut seen = vec![false; self.ndim()];
        for &d in dims {
            if d >= self.ndim() {
                return Err(Error::InvalidDimension {
                    index: d as i64,
                    ndim: self.ndim(),
                });
            }
            if seen[d] {
                return Err(Error::invalid_operation("Duplicate dimension in permute"));
            }
            seen[d] = true;
        }

        let new_shape: Shape = dims.iter().map(|&d| self.shape[d]).collect();
        let new_strides: Strides = dims.iter().map(|&d| self.strides[d]).collect();

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Returns a contiguous copy of the tensor.
    #[must_use]
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() && self.offset == 0 {
            return self.clone();
        }

        let data = self.to_vec();
        Self::from_vec(data, &self.shape).expect("Contiguous should never fail")
    }

    // =========================================================================
    // Device Operations
    // =========================================================================

    /// Transfers the tensor to a different device.
    ///
    /// # Arguments
    /// * `device` - Target device
    pub fn to_device(&self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }

        let contig = self.contiguous();
        let new_storage = contig.storage.to_device(device)?;

        Ok(Self {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
        })
    }

    /// Transfers to CPU.
    pub fn cpu(&self) -> Result<Self> {
        self.to_device(Device::Cpu)
    }

    // =========================================================================
    // Deep Copy
    // =========================================================================

    /// Creates a deep copy of this tensor with its own storage.
    #[must_use]
    pub fn clone_deep(&self) -> Self {
        let data = self.to_vec();
        Self::from_vec(data, &self.shape).expect("Deep clone should never fail")
    }
}

// =============================================================================
// Numeric Operations
// =============================================================================

impl<T: Numeric> Tensor<T> {
    /// Fills the tensor with a value.
    pub fn fill_(&self, value: T) {
        let mut data = self.storage.as_slice_mut();
        CpuBackend::fill(&mut data, value);
    }

    /// Fills the tensor with zeros.
    pub fn zero_(&self) {
        self.fill_(T::zero());
    }

    // =========================================================================
    // Reduction Operations
    // =========================================================================

    /// Returns the sum of all elements.
    #[must_use]
    pub fn sum(&self) -> Self {
        let data = self.to_vec();
        let result = CpuBackend::sum(&data);
        Self::scalar(result)
    }

    /// Returns the product of all elements.
    #[must_use]
    pub fn prod(&self) -> Self {
        let data = self.to_vec();
        let result = CpuBackend::prod(&data);
        Self::scalar(result)
    }

    /// Returns the maximum element.
    pub fn max(&self) -> Result<Self> {
        if self.is_empty() {
            return Err(Error::EmptyTensor);
        }
        let data = self.to_vec();
        let result = CpuBackend::max(&data).unwrap();
        Ok(Self::scalar(result))
    }

    /// Returns the minimum element.
    pub fn min(&self) -> Result<Self> {
        if self.is_empty() {
            return Err(Error::EmptyTensor);
        }
        let data = self.to_vec();
        let result = CpuBackend::min(&data).unwrap();
        Ok(Self::scalar(result))
    }

    /// Returns the index of the maximum element.
    pub fn argmax(&self) -> Result<usize> {
        if self.is_empty() {
            return Err(Error::EmptyTensor);
        }
        let data = self.to_vec();
        Ok(CpuBackend::argmax(&data).unwrap())
    }

    /// Returns the index of the minimum element.
    pub fn argmin(&self) -> Result<usize> {
        if self.is_empty() {
            return Err(Error::EmptyTensor);
        }
        let data = self.to_vec();
        Ok(CpuBackend::argmin(&data).unwrap())
    }
}

// =============================================================================
// Float Operations
// =============================================================================

impl<T: Float> Tensor<T> {
    /// Returns the mean of all elements.
    pub fn mean(&self) -> Result<Self> {
        if self.is_empty() {
            return Err(Error::EmptyTensor);
        }
        let data = self.to_vec();
        let result = CpuBackend::mean(&data).unwrap();
        Ok(Self::scalar(result))
    }

    // =========================================================================
    // Activation Functions
    // =========================================================================

    /// Applies `ReLU` activation: max(0, x).
    #[must_use]
    pub fn relu(&self) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::relu(&mut result, &data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies sigmoid activation: 1 / (1 + exp(-x)).
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::sigmoid(&mut result, &data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies tanh activation.
    #[must_use]
    pub fn tanh(&self) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::tanh(&mut result, &data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies exponential function.
    #[must_use]
    pub fn exp(&self) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::exp(&mut result, &data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies natural logarithm.
    #[must_use]
    pub fn ln(&self) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::ln(&mut result, &data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies square root.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::sqrt(&mut result, &data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Computes element-wise power.
    #[must_use]
    pub fn pow(&self, exp: T) -> Self {
        let data = self.to_vec();
        let result: Vec<T> = data.iter().map(|&x| x.pow_value(exp)).collect();
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// GELU activation function (Gaussian Error Linear Unit).
    #[must_use]
    pub fn gelu(&self) -> Self {
        crate::ops::gelu(self)
    }

    /// SiLU/Swish activation function.
    #[must_use]
    pub fn silu(&self) -> Self {
        crate::ops::silu(self)
    }

    /// Softmax along specified dimension.
    #[must_use]
    pub fn softmax(&self, dim: i32) -> Self {
        crate::ops::softmax(self, dim as i64).unwrap_or_else(|_| self.clone())
    }

    /// Log softmax along specified dimension.
    #[must_use]
    pub fn log_softmax(&self, dim: i32) -> Self {
        let softmax_result = self.softmax(dim);
        softmax_result.ln()
    }

    /// Mean along a dimension.
    #[must_use]
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Self {
        let ndim = self.ndim();
        let dim = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return self.clone();
        }

        let dim_size = self.shape[dim];
        let data = self.to_vec();
        let mut new_shape = self.shape.clone();

        if keepdim {
            new_shape[dim] = 1;
        } else {
            new_shape.remove(dim);
        }

        if new_shape.is_empty() {
            new_shape = smallvec::smallvec![1];
        }

        let new_numel: usize = new_shape.iter().product();
        let mut result = vec![T::zero(); new_numel];

        // Compute strides for iteration
        let outer_size: usize = self.shape[..dim].iter().product();
        let inner_size: usize = self.shape[dim + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = T::zero();
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    sum = sum + data[idx];
                }
                let mean = sum / NumCast::from(dim_size).unwrap();
                let result_idx = outer * inner_size + inner;
                result[result_idx] = mean;
            }
        }

        Self::from_vec(result, &new_shape).unwrap()
    }

    /// Variance along a dimension.
    #[must_use]
    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Self {
        let mean = self.mean_dim(dim, true);
        let diff = self.sub(&mean).unwrap_or_else(|_| self.clone());
        let squared = diff.mul(&diff).unwrap_or_else(|_| self.clone());
        squared.mean_dim(dim, keepdim)
    }

    /// Broadcasts tensor to a new shape.
    #[must_use]
    pub fn broadcast_to(&self, shape: &[usize]) -> Self {
        if self.shape.as_slice() == shape {
            return self.clone();
        }

        let result_shape = broadcast_shape(&self.shape, shape).unwrap_or_else(|_| shape.into());
        let self_strides = broadcast_strides(&self.shape, &self.strides, &result_shape);

        let total = numel(&result_shape);
        let mut result_data = vec![T::zero(); total];
        let self_data = self.storage.as_slice();

        for i in 0..total {
            let indices = crate::shape::unravel_index(i, &result_shape);
            let self_idx = self.offset + linear_index(&indices, &self_strides);
            result_data[i] = self_data[self_idx];
        }

        Self::from_vec(result_data, &result_shape).unwrap()
    }

    /// Slices the tensor using ranges for each dimension.
    #[must_use]
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Self {
        let mut new_shape = Vec::with_capacity(self.ndim());
        for (i, range) in ranges.iter().enumerate() {
            if i < self.ndim() {
                new_shape.push(range.end - range.start);
            }
        }
        // Keep remaining dimensions unchanged
        for i in ranges.len()..self.ndim() {
            new_shape.push(self.shape[i]);
        }

        let new_numel: usize = new_shape.iter().product();
        let mut result_data = vec![T::zero(); new_numel];
        let self_data = self.to_vec();

        // Copy data with proper indexing
        let mut result_idx = 0;
        Self::slice_recursive(
            &self_data,
            &self.shape,
            ranges,
            0,
            0,
            &mut result_data,
            &mut result_idx,
        );

        Self::from_vec(result_data, &new_shape).unwrap()
    }

    fn slice_recursive(
        data: &[T],
        shape: &[usize],
        ranges: &[std::ops::Range<usize>],
        dim: usize,
        offset: usize,
        result: &mut [T],
        result_idx: &mut usize,
    ) {
        if dim == shape.len() {
            result[*result_idx] = data[offset];
            *result_idx += 1;
            return;
        }

        let stride: usize = shape[dim + 1..].iter().product();
        let (start, end) = if dim < ranges.len() {
            (ranges[dim].start, ranges[dim].end)
        } else {
            (0, shape[dim])
        };

        for i in start..end {
            Self::slice_recursive(
                data,
                shape,
                ranges,
                dim + 1,
                offset + i * stride,
                result,
                result_idx,
            );
        }
    }
}

// =============================================================================
// Arithmetic Operator Implementations
// =============================================================================

impl<T: Numeric> Tensor<T> {
    /// Element-wise addition with broadcasting.
    pub fn add(&self, other: &Self) -> Result<Self> {
        let result_shape = broadcast_shape(&self.shape, &other.shape)?;
        let self_strides = broadcast_strides(&self.shape, &self.strides, &result_shape);
        let other_strides = broadcast_strides(&other.shape, &other.strides, &result_shape);

        let total = numel(&result_shape);
        let mut result_data = vec![T::zero(); total];

        let self_data = self.storage.as_slice();
        let other_data = other.storage.as_slice();

        for i in 0..total {
            let indices = crate::shape::unravel_index(i, &result_shape);
            let self_idx = self.offset + linear_index(&indices, &self_strides);
            let other_idx = other.offset + linear_index(&indices, &other_strides);
            result_data[i] = self_data[self_idx] + other_data[other_idx];
        }

        Self::from_vec(result_data, &result_shape)
    }

    /// Element-wise subtraction with broadcasting.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let result_shape = broadcast_shape(&self.shape, &other.shape)?;
        let self_strides = broadcast_strides(&self.shape, &self.strides, &result_shape);
        let other_strides = broadcast_strides(&other.shape, &other.strides, &result_shape);

        let total = numel(&result_shape);
        let mut result_data = vec![T::zero(); total];

        let self_data = self.storage.as_slice();
        let other_data = other.storage.as_slice();

        for i in 0..total {
            let indices = crate::shape::unravel_index(i, &result_shape);
            let self_idx = self.offset + linear_index(&indices, &self_strides);
            let other_idx = other.offset + linear_index(&indices, &other_strides);
            result_data[i] = self_data[self_idx] - other_data[other_idx];
        }

        Self::from_vec(result_data, &result_shape)
    }

    /// Element-wise multiplication with broadcasting.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let result_shape = broadcast_shape(&self.shape, &other.shape)?;
        let self_strides = broadcast_strides(&self.shape, &self.strides, &result_shape);
        let other_strides = broadcast_strides(&other.shape, &other.strides, &result_shape);

        let total = numel(&result_shape);
        let mut result_data = vec![T::zero(); total];

        let self_data = self.storage.as_slice();
        let other_data = other.storage.as_slice();

        for i in 0..total {
            let indices = crate::shape::unravel_index(i, &result_shape);
            let self_idx = self.offset + linear_index(&indices, &self_strides);
            let other_idx = other.offset + linear_index(&indices, &other_strides);
            result_data[i] = self_data[self_idx] * other_data[other_idx];
        }

        Self::from_vec(result_data, &result_shape)
    }

    /// Element-wise division with broadcasting.
    pub fn div(&self, other: &Self) -> Result<Self> {
        let result_shape = broadcast_shape(&self.shape, &other.shape)?;
        let self_strides = broadcast_strides(&self.shape, &self.strides, &result_shape);
        let other_strides = broadcast_strides(&other.shape, &other.strides, &result_shape);

        let total = numel(&result_shape);
        let mut result_data = vec![T::zero(); total];

        let self_data = self.storage.as_slice();
        let other_data = other.storage.as_slice();

        for i in 0..total {
            let indices = crate::shape::unravel_index(i, &result_shape);
            let self_idx = self.offset + linear_index(&indices, &self_strides);
            let other_idx = other.offset + linear_index(&indices, &other_strides);
            result_data[i] = self_data[self_idx] / other_data[other_idx];
        }

        Self::from_vec(result_data, &result_shape)
    }

    /// Scalar addition.
    #[must_use]
    pub fn add_scalar(&self, scalar: T) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::add_scalar(&mut result, &data, scalar);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn mul_scalar(&self, scalar: T) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::mul_scalar(&mut result, &data, scalar);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Element-wise negation.
    #[must_use]
    pub fn neg(&self) -> Self {
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::neg(&mut result, &data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Matrix multiplication with batching support.
    ///
    /// Supports:
    /// - 2D @ 2D: [m, k] @ [k, n] -> [m, n]
    /// - 3D @ 3D: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    /// - 4D @ 4D: [b1, b2, m, k] @ [b1, b2, k, n] -> [b1, b2, m, n]
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(Error::invalid_operation(
                "matmul requires at least 2D tensors",
            ));
        }

        let m = self.shape[self.ndim() - 2];
        let k1 = self.shape[self.ndim() - 1];
        let k2 = other.shape[other.ndim() - 2];
        let n = other.shape[other.ndim() - 1];

        if k1 != k2 {
            return Err(Error::invalid_operation(format!(
                "matmul inner dimensions must match: {k1} vs {k2}"
            )));
        }

        // For 2D matrices, do simple matmul
        if self.ndim() == 2 && other.ndim() == 2 {
            let a_data = self.contiguous().to_vec();
            let b_data = other.contiguous().to_vec();
            let mut c_data = vec![T::zero(); m * n];
            CpuBackend::matmul(&mut c_data, &a_data, &b_data, m, n, k1);
            return Self::from_vec(c_data, &[m, n]);
        }

        // For batched matmul, compute batch size
        let batch_dims_self: Vec<usize> = self.shape[..self.ndim() - 2].to_vec();
        let batch_dims_other: Vec<usize> = other.shape[..other.ndim() - 2].to_vec();

        if batch_dims_self != batch_dims_other {
            return Err(Error::invalid_operation(format!(
                "matmul batch dimensions must match: {:?} vs {:?}",
                batch_dims_self, batch_dims_other
            )));
        }

        let batch_size: usize = batch_dims_self.iter().product();
        let a_stride = m * k1;
        let b_stride = k1 * n;
        let c_stride = m * n;

        let a_data = self.contiguous().to_vec();
        let b_data = other.contiguous().to_vec();
        let mut c_data = vec![T::zero(); batch_size * m * n];

        // Loop over batches and compute matmul for each
        for batch in 0..batch_size {
            let a_slice = &a_data[batch * a_stride..(batch + 1) * a_stride];
            let b_slice = &b_data[batch * b_stride..(batch + 1) * b_stride];
            let c_slice = &mut c_data[batch * c_stride..(batch + 1) * c_stride];
            CpuBackend::matmul(c_slice, a_slice, b_slice, m, n, k1);
        }

        // Build output shape: batch_dims + [m, n]
        let mut output_shape = batch_dims_self;
        output_shape.push(m);
        output_shape.push(n);

        Self::from_vec(c_data, &output_shape)
    }

    /// Dot product for 1D tensors.
    pub fn dot(&self, other: &Self) -> Result<Self> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(Error::invalid_operation("dot requires 1D tensors"));
        }

        if self.shape[0] != other.shape[0] {
            return Err(Error::shape_mismatch(&self.shape, &other.shape));
        }

        let a_data = self.to_vec();
        let b_data = other.to_vec();
        let result = CpuBackend::dot(&a_data, &b_data);

        Ok(Self::scalar(result))
    }
}

// =============================================================================
// Operator Trait Implementations
// =============================================================================

impl<T: Numeric> Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: Self) -> Self::Output {
        self.add(other).expect("Addition failed")
    }
}

impl<T: Numeric> Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(other).expect("Subtraction failed")
    }
}

impl<T: Numeric> Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(other).expect("Multiplication failed")
    }
}

impl<T: Numeric> Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, other: Self) -> Self::Output {
        self.div(other).expect("Division failed")
    }
}

impl<T: Numeric> Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self.neg()
    }
}

// Scalar operations
impl<T: Numeric> Add<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, scalar: T) -> Self::Output {
        self.add_scalar(scalar)
    }
}

impl<T: Numeric> Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, scalar: T) -> Self::Output {
        self.mul_scalar(scalar)
    }
}

// =============================================================================
// Display Implementation
// =============================================================================

impl<T: Scalar + fmt::Display> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, device={}",
            self.shape(),
            self.device()
        )?;
        if self.numel() <= 10 {
            write!(f, ", data={:?}", self.to_vec())?;
        }
        write!(f, ")")
    }
}

impl<T: Scalar + fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            write!(f, "{}", self.item().unwrap())
        } else if self.ndim() == 1 {
            write!(f, "[")?;
            let data = self.to_vec();
            for (i, val) in data.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{val}")?;
            }
            write!(f, "]")
        } else {
            write!(f, "Tensor(shape={:?})", self.shape())
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
    fn test_from_vec() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_get_set() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(t.get(&[1, 0]).unwrap(), 3.0);
        assert_eq!(t.get(&[1, 1]).unwrap(), 4.0);

        t.set(&[0, 0], 99.0).unwrap();
        assert_eq!(t.get(&[0, 0]).unwrap(), 99.0);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);

        let r = t.reshape(&[-1]).unwrap();
        assert_eq!(r.shape(), &[6]);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let r = t.t().unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(r.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(r.get(&[1, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let c = &a + &b;
        assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);

        let d = &a * &b;
        assert_eq!(d.to_vec(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_broadcasting() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![10.0], &[1]).unwrap();

        let c = &a + &b;
        assert_eq!(c.to_vec(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_sum() {
        let t = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let s = t.sum();
        assert_eq!(s.item().unwrap(), 10.0);
    }

    #[test]
    fn test_matmul() {
        // 2x2 @ 2x2
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_relu() {
        let t = Tensor::<f32>::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let r = t.relu();
        assert_eq!(r.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_scalar() {
        let s = Tensor::<f32>::scalar(42.0);
        assert!(s.is_scalar());
        assert_eq!(s.numel(), 1);
        assert_eq!(s.item().unwrap(), 42.0);
    }
}
