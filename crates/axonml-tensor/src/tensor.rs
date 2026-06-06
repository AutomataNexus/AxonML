//! Core `Tensor<T>` struct — 2170 lines, 76 public methods.
//!
//! Constructors (from_vec, from_slice, scalar, zeros, ones, randn, rand, full),
//! properties (shape, numel, ndim, device, is_contiguous, strides, offset),
//! shape ops (reshape, transpose, t, squeeze, unsqueeze, expand, permute,
//! contiguous, narrow, select, index_select, cat, chunk, split, flip, roll),
//! arithmetic (add, sub, mul, div, neg, abs, pow, add_scalar, mul_scalar,
//! where_cond, clamp, clamp_min), reductions (sum, mean, max, min, prod,
//! argmax, argmin, sum_dim, mean_dim, var_dim), activations (relu, sigmoid,
//! tanh, softmax, log_softmax, gelu, silu, elu, leaky_relu), matmul (CPU
//! via CpuBackend::matmul with GEMV fast path, GPU via cuBLAS + quantized
//! Q4_K/Q6_K dispatch), indexing (gather, scatter, nonzero, unique, sort,
//! argsort, topk), comparison (eq, gt, lt), cast (to, to_device), data
//! access (to_vec, get, item), map/zip_map/zip_map3, and Display.
//!
//! # File
//! `crates/axonml-tensor/src/tensor.rs`
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

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use axonml_core::Device;
use axonml_core::backends::CpuBackend;
#[cfg(feature = "cuda")]
use axonml_core::backends::CudaBackend;
use axonml_core::dtype::{Float, Numeric, Scalar};
use axonml_core::error::{Error, Result};
use axonml_core::storage::Storage;
use num_traits::NumCast;

// =============================================================================
// CUDA Acceleration
// =============================================================================

#[cfg(feature = "cuda")]
mod cuda_accel {
    use super::*;
    use axonml_core::backends::cuda::get_cuda_backend;

    /// Get the global CUDA backend (delegates to core singleton).
    pub fn get_cuda() -> Option<&'static CudaBackend> {
        get_cuda_backend()
    }

    /// GPU-accelerated matmul: copies data to GPU, runs cuBLAS GEMM, copies back.
    /// Returns None if GPU is unavailable or an error occurs.
    pub fn cuda_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Option<Vec<f32>> {
        let cuda = get_cuda()?;

        let a_gpu = cuda.htod_copy(a).ok()?;
        let b_gpu = cuda.htod_copy(b).ok()?;
        let mut c_gpu = cuda.alloc::<f32>(m * n).ok()?;

        // cuBLAS GEMM: C(m,n) = A(m,k) @ B(k,n) in row-major
        // In column-major terms: C^T(n,m) = B^T(n,k) @ A^T(k,m)
        cuda.gemm_f32(
            false, false, n, m, k, 1.0, &b_gpu, n, &a_gpu, k, 0.0, &mut c_gpu, n,
        )
        .ok()?;

        cuda.dtoh_copy(&c_gpu).ok()
    }
}

use crate::shape::{
    Shape, Strides, broadcast_shape, broadcast_strides, contiguous_strides, is_contiguous,
    linear_index, normalize_dim, numel, reshape, squeeze, transpose_shape, transpose_strides,
    unsqueeze,
};

// =============================================================================
// GPU Dispatch Helpers
// =============================================================================
//
// These enable calling Tensor<f32> GPU methods from generic Tensor<T> code
// when T is verified to be f32 via TypeId check at runtime.

#[cfg(feature = "cuda")]
unsafe fn gpu_ref<T: Scalar>(t: &Tensor<T>) -> &Tensor<f32> {
    assert!(
        is_f32::<T>(),
        "gpu_ref: only Tensor<f32> can be used for GPU operations, got {:?}",
        T::DTYPE
    );
    // SAFETY: T is f32 (asserted above), Tensor<f32> and Tensor<T> have identical layout
    unsafe { &*(t as *const Tensor<T> as *const Tensor<f32>) }
}

#[cfg(feature = "cuda")]
unsafe fn gpu_ref_mut<T: Scalar>(t: &mut Tensor<T>) -> &mut Tensor<f32> {
    assert!(
        is_f32::<T>(),
        "gpu_ref_mut: only Tensor<f32> can be used for GPU operations, got {:?}",
        T::DTYPE
    );
    unsafe { &mut *(t as *mut Tensor<T> as *mut Tensor<f32>) }
}

#[cfg(feature = "cuda")]
unsafe fn gpu_into<T: Scalar>(t: Tensor<f32>) -> Tensor<T> {
    assert!(
        is_f32::<T>(),
        "gpu_into: only Tensor<f32> can be produced from GPU operations, got {:?}",
        T::DTYPE
    );
    // SAFETY: T is f32 (asserted above), ownership transfer via ptr::read + forget
    unsafe {
        let out = std::ptr::read(&t as *const Tensor<f32> as *const Tensor<T>);
        std::mem::forget(t);
        out
    }
}

#[cfg(feature = "cuda")]
fn is_f32<T: 'static>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
}

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

        // Use to_vec() which handles both CPU and GPU tensors safely
        let data = self.to_vec();
        if data.is_empty() {
            Err(Error::invalid_operation("item() on empty tensor"))
        } else {
            Ok(data[0])
        }
    }

    /// Returns the data as a contiguous vector.
    ///
    /// If the tensor is already contiguous, this returns a reference.
    /// Otherwise, it copies the data into a new contiguous vector.
    /// For GPU tensors (f32 only), performs a D2H copy.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        // GPU path: GPU storage is always f32
        #[cfg(feature = "cuda")]
        if self.storage.is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            let f32_vec = self_f32.to_vec_gpu();
            unsafe {
                let mut v = std::mem::ManuallyDrop::new(f32_vec);
                return Vec::from_raw_parts(v.as_mut_ptr() as *mut T, v.len(), v.capacity());
            }
        }

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
                .filter(|(dim, _)| **dim != 1)
                .map(|(_, stride)| *stride)
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

        #[cfg(feature = "cuda")]
        if self.storage.is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            let result = self_f32.contiguous_gpu();
            return unsafe { gpu_into(result) };
        }

        let data = self.to_vec();
        Self::from_vec(data, &self.shape).expect("Contiguous should never fail")
    }

    // =========================================================================
    // Functional Map Operations (zero-copy for CPU tensors)
    // =========================================================================

    /// Apply a function element-wise, producing a new tensor with the same shape.
    ///
    /// Avoids the to_vec() → map → from_vec() pattern by operating directly
    /// on contiguous storage.
    #[must_use]
    pub fn map<F: Fn(T) -> T>(&self, f: F) -> Self {
        let storage = self.storage.as_slice();
        let fast = self.is_contiguous() && self.offset == 0;
        let slice: &[T] = if fast { &storage[..self.numel()] } else { &[] };
        let owned: Option<Vec<T>> = if fast { None } else { Some(self.to_vec()) };
        let data: &[T] = owned.as_deref().unwrap_or(slice);
        let result: Vec<T> = data.iter().copied().map(f).collect();
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Apply a binary function element-wise with another tensor of the same shape.
    ///
    /// This is the primary zero-allocation pattern for backward functions:
    /// instead of `a.to_vec()` + `b.to_vec()` + zip + `from_vec()`,
    /// use `a.zip_map(&b, |x, y| ...)` which does a single allocation.
    #[must_use]
    pub fn zip_map<F: Fn(T, T) -> T>(&self, other: &Self, f: F) -> Self {
        let sa = self.storage.as_slice();
        let fa = self.is_contiguous() && self.offset == 0;
        let sa_slice: &[T] = if fa { &sa[..self.numel()] } else { &[] };
        let oa: Option<Vec<T>> = if fa { None } else { Some(self.to_vec()) };
        let a: &[T] = oa.as_deref().unwrap_or(sa_slice);

        let sb = other.storage.as_slice();
        let fb = other.is_contiguous() && other.offset == 0;
        let sb_slice: &[T] = if fb { &sb[..other.numel()] } else { &[] };
        let ob: Option<Vec<T>> = if fb { None } else { Some(other.to_vec()) };
        let b: &[T] = ob.as_deref().unwrap_or(sb_slice);

        debug_assert_eq!(
            a.len(),
            b.len(),
            "zip_map requires same number of elements: {} vs {}",
            a.len(),
            b.len()
        );
        let result: Vec<T> = a.iter().copied().zip(b.iter().copied()).map(|(x, y)| f(x, y)).collect();
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Apply a ternary function element-wise with two other tensors.
    #[must_use]
    pub fn zip_map3<F: Fn(T, T, T) -> T>(&self, b: &Self, c: &Self, f: F) -> Self {
        let sa = self.storage.as_slice();
        let fa = self.is_contiguous() && self.offset == 0;
        let sa_slice: &[T] = if fa { &sa[..self.numel()] } else { &[] };
        let oa: Option<Vec<T>> = if fa { None } else { Some(self.to_vec()) };
        let a_data: &[T] = oa.as_deref().unwrap_or(sa_slice);

        let sb = b.storage.as_slice();
        let fb = b.is_contiguous() && b.offset == 0;
        let sb_slice: &[T] = if fb { &sb[..b.numel()] } else { &[] };
        let ob: Option<Vec<T>> = if fb { None } else { Some(b.to_vec()) };
        let b_data: &[T] = ob.as_deref().unwrap_or(sb_slice);

        let sc = c.storage.as_slice();
        let fc = c.is_contiguous() && c.offset == 0;
        let sc_slice: &[T] = if fc { &sc[..c.numel()] } else { &[] };
        let oc: Option<Vec<T>> = if fc { None } else { Some(c.to_vec()) };
        let c_data: &[T] = oc.as_deref().unwrap_or(sc_slice);

        debug_assert_eq!(a_data.len(), b_data.len());
        debug_assert_eq!(a_data.len(), c_data.len());
        let result: Vec<T> = a_data
            .iter().copied()
            .zip(b_data.iter().copied())
            .zip(c_data.iter().copied())
            .map(|((a, b), c)| f(a, b, c))
            .collect();
        Self::from_vec(result, &self.shape).unwrap()
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

        #[cfg(feature = "cuda")]
        if self.storage.is_gpu() || device.is_gpu() {
            if !is_f32::<T>() {
                return Err(Error::invalid_operation(format!(
                    "Tensor<{}>.to_device(GPU) is not supported (GPU tensors are f32-only). \
                     Keep token IDs / integer tensors on CPU; Embedding::lookup and the autograd \
                     cross-entropy path handle the CPU-index → GPU-weight crossing internally.",
                    std::any::type_name::<T>()
                )));
            }
            let self_f32 = unsafe { gpu_ref(self) };
            let result = self_f32.to_device_f32(device)?;
            return Ok(unsafe { gpu_into(result) });
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
        let cpu = Self::from_vec(data, &self.shape).expect("Deep clone should never fail");
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            return cpu.to_device(self.device()).unwrap();
        }
        cpu
    }
}

// =============================================================================
// Numeric Operations
// =============================================================================

impl<T: Numeric> Tensor<T> {
    /// Fills the tensor with a value.
    ///
    /// # Panics
    /// Panics on GPU tensors. Use `Tensor::from_vec(vec![value; n], shape)`
    /// followed by `.to_device()` instead.
    pub fn fill_(&self, value: T) {
        assert!(
            self.storage.is_cpu(),
            "fill_() not supported on GPU tensors — create a new tensor and transfer instead"
        );
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

    /// Returns the sum of all elements as a scalar tensor.
    ///
    /// On GPU, uses native CUDA reduction kernels (no CPU round-trip).
    #[must_use]
    pub fn sum(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            let mut t = self_f32.clone();
            while t.ndim() > 1 {
                t = t.sum_dim_cuda(0);
            }
            if t.numel() > 1 {
                t = t.sum_dim_cuda(0);
            }
            return unsafe { gpu_into(t) };
        }

        let data = self.to_vec();
        let result = CpuBackend::sum(&data);
        Self::scalar(result)
    }

    /// Returns the product of all elements.
    ///
    /// GPU: D2H round-trip (no CUDA prod reduction kernel yet).
    #[must_use]
    pub fn prod(&self) -> Self {
        let data = self.to_vec();
        let result = CpuBackend::prod(&data);
        let s = Self::scalar(result);
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            return s
                .to_device(self.device())
                .expect("prod: device transfer failed");
        }
        s
    }

    /// Returns the maximum element.
    ///
    /// GPU: D2H round-trip (no CUDA max reduction kernel yet).
    pub fn max(&self) -> Result<Self> {
        if self.is_empty() {
            return Err(Error::EmptyTensor);
        }
        let data = self.to_vec();
        let result = CpuBackend::max(&data).expect("max on non-empty tensor");
        let s = Self::scalar(result);
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            return Ok(s
                .to_device(self.device())
                .expect("max: device transfer failed"));
        }
        Ok(s)
    }

    /// Returns the minimum element.
    ///
    /// GPU: D2H round-trip (no CUDA min reduction kernel yet).
    pub fn min(&self) -> Result<Self> {
        if self.is_empty() {
            return Err(Error::EmptyTensor);
        }
        let data = self.to_vec();
        let result = CpuBackend::min(&data).expect("min on non-empty tensor");
        let s = Self::scalar(result);
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            return Ok(s
                .to_device(self.device())
                .expect("min: device transfer failed"));
        }
        Ok(s)
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

    /// Concatenates tensors along a dimension.
    ///
    /// All tensors must have the same shape except along the cat dimension.
    pub fn cat(tensors: &[&Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(Error::invalid_operation("cat requires at least one tensor"));
        }
        let ndim = tensors[0].ndim();
        if dim >= ndim {
            return Err(Error::invalid_operation("cat dimension out of range"));
        }

        for t in &tensors[1..] {
            if t.ndim() != ndim {
                return Err(Error::invalid_operation(
                    "cat: all tensors must have same ndim",
                ));
            }
            for d in 0..ndim {
                if d != dim && t.shape[d] != tensors[0].shape[d] {
                    return Err(Error::invalid_operation(
                        "cat: shapes must match on non-cat dims",
                    ));
                }
            }
        }

        let total_dim_size: usize = tensors.iter().map(|t| t.shape[dim]).sum();
        let mut out_shape: Vec<usize> = tensors[0].shape.to_vec();
        out_shape[dim] = total_dim_size;

        let outer_size: usize = out_shape[..dim].iter().product();
        let inner_size: usize = out_shape[dim + 1..].iter().product();
        let total_numel: usize = out_shape.iter().product();
        let mut result = vec![T::zero(); total_numel];

        let mut dim_offset = 0;
        for t in tensors {
            let t_data = t.contiguous().to_vec();
            let t_dim_size = t.shape[dim];
            for outer in 0..outer_size {
                for d in 0..t_dim_size {
                    let src_base = outer * t_dim_size * inner_size + d * inner_size;
                    let dst_base =
                        outer * total_dim_size * inner_size + (dim_offset + d) * inner_size;
                    result[dst_base..dst_base + inner_size]
                        .copy_from_slice(&t_data[src_base..src_base + inner_size]);
                }
            }
            dim_offset += t_dim_size;
        }

        let out = Self::from_vec(result, &out_shape)?;
        #[cfg(feature = "cuda")]
        if tensors[0].device().is_gpu() {
            return Ok(out.to_device(tensors[0].device()).unwrap());
        }
        Ok(out)
    }
}

// =============================================================================
// Float Operations
// =============================================================================

impl<T: Float> Tensor<T> {
    /// Returns the mean of all elements.
    /// Returns the mean of all elements.
    ///
    /// On GPU, uses native CUDA sum reduction then divides by numel.
    pub fn mean(&self) -> Result<Self> {
        if self.is_empty() {
            return Err(Error::EmptyTensor);
        }
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            let s = self.sum(); // uses CUDA sum_dim chain
            let n = self.numel() as f32;
            // mul_scalar stays on GPU
            return Ok(s.mul_scalar(T::from(1.0 / n as f64).unwrap_or(T::zero())));
        }

        let data = self.to_vec();
        let result = CpuBackend::mean(&data).expect("mean on non-empty tensor");
        Ok(Self::scalar(result))
    }

    // =========================================================================
    // Activation Functions
    // =========================================================================

    /// Applies `ReLU` activation: max(0, x).
    #[must_use]
    pub fn relu(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).relu_cuda()) };
        }
        // Fast path for CPU contiguous (common in inference): avoid to_vec copy, feed storage slice directly to parallel CpuBackend.
        let storage = self.storage.as_slice();
        let fast = self.is_contiguous() && self.offset == 0;
        let slice: &[T] = if fast { &storage[..self.numel()] } else { &[] };
        let owned: Option<Vec<T>> = if fast { None } else { Some(self.to_vec()) };
        let data: &[T] = owned.as_deref().unwrap_or(slice);
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::relu(&mut result, data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies sigmoid activation: 1 / (1 + exp(-x)).
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).sigmoid_cuda()) };
        }
        let storage = self.storage.as_slice();
        let fast = self.is_contiguous() && self.offset == 0;
        let slice: &[T] = if fast { &storage[..self.numel()] } else { &[] };
        let owned: Option<Vec<T>> = if fast { None } else { Some(self.to_vec()) };
        let data: &[T] = owned.as_deref().unwrap_or(slice);
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::sigmoid(&mut result, data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies tanh activation.
    #[must_use]
    pub fn tanh(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).tanh_cuda()) };
        }
        let storage = self.storage.as_slice();
        let fast = self.is_contiguous() && self.offset == 0;
        let slice: &[T] = if fast { &storage[..self.numel()] } else { &[] };
        let owned: Option<Vec<T>> = if fast { None } else { Some(self.to_vec()) };
        let data: &[T] = owned.as_deref().unwrap_or(slice);
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::tanh(&mut result, data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies exponential function.
    #[must_use]
    pub fn exp(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).exp_cuda()) };
        }
        let storage = self.storage.as_slice();
        let fast = self.is_contiguous() && self.offset == 0;
        let slice: &[T] = if fast { &storage[..self.numel()] } else { &[] };
        let owned: Option<Vec<T>> = if fast { None } else { Some(self.to_vec()) };
        let data: &[T] = owned.as_deref().unwrap_or(slice);
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::exp(&mut result, data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies natural logarithm.
    #[must_use]
    pub fn ln(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).ln_cuda()) };
        }
        let storage = self.storage.as_slice();
        let fast = self.is_contiguous() && self.offset == 0;
        let slice: &[T] = if fast { &storage[..self.numel()] } else { &[] };
        let owned: Option<Vec<T>> = if fast { None } else { Some(self.to_vec()) };
        let data: &[T] = owned.as_deref().unwrap_or(slice);
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::ln(&mut result, data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Applies square root.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).sqrt_cuda()) };
        }
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::sqrt(&mut result, &data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Computes element-wise power.
    #[must_use]
    pub fn pow(&self, exp: T) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let exp_f32: f32 = unsafe { *(&exp as *const T as *const f32) };
            return unsafe { gpu_into(gpu_ref(self).pow_cuda(exp_f32)) };
        }
        let data = self.to_vec();
        let result: Vec<T> = data.iter().map(|&x| x.pow_value(exp)).collect();
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// GELU activation function (Gaussian Error Linear Unit).
    #[must_use]
    pub fn gelu(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).gelu_cuda()) };
        }
        crate::ops::gelu(self)
    }

    /// SiLU/Swish activation function.
    #[must_use]
    pub fn silu(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).silu_cuda()) };
        }
        crate::ops::silu(self)
    }

    /// Fused SiLU backward: `grad_input = grad_output * σ(x) * (1 + x*(1-σ(x)))`
    /// on GPU in a single kernel launch. `self` is the saved forward input `x`.
    /// For CPU tensors callers fall back to the per-element f32 path in
    /// `Tensor<f32>::silu_backward_cpu`.
    #[must_use]
    pub fn silu_backward(&self, grad_output: &Self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let go = unsafe { gpu_ref(grad_output) };
            return unsafe { gpu_into(gpu_ref(self).silu_backward_cuda(go)) };
        }
        // CPU fallback: only defined for f32 (matches original SiluBackward).
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>(),
            "silu_backward CPU path requires f32",
        );
        let x_f32 = unsafe { &*std::ptr::from_ref::<Self>(self).cast::<Tensor<f32>>() };
        let g_f32 = unsafe { &*std::ptr::from_ref::<Self>(grad_output).cast::<Tensor<f32>>() };
        let result_f32 = x_f32.zip_map(g_f32, |x, g| {
            let sig = 1.0f32 / (1.0f32 + (-x).exp());
            g * (sig + x * sig * (1.0f32 - sig))
        });
        unsafe { std::ptr::read(std::ptr::from_ref::<Tensor<f32>>(&result_f32).cast::<Self>()) }
    }

    /// RMSNorm with a per-element weight scale: `out = x * w / sqrt(mean(x²) + eps)`.
    ///
    /// Decode-step kernel — one CTA, suitable for single-token activations of
    /// any hidden size up to ~16K. CPU fallback uses a serial reduction.
    #[must_use]
    /// Single-token LayerNorm (mean-subtracting, affine). Falcon arch.
    /// `out[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]`.
    pub fn layer_norm_tokenwise(&self, gamma: &Self, beta: &Self, eps: f32) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).layer_norm_tokenwise_cuda(
                    gpu_ref(gamma),
                    gpu_ref(beta),
                    eps,
                ))
            };
        }
        // CPU fallback.
        let x = self.to_vec();
        let g = gamma.to_vec();
        let b = beta.to_vec();
        let n = x.len();
        let n_f = n as f32;
        let mean: f32 = x.iter().map(|v| v.to_f32().unwrap_or(0.0)).sum::<f32>() / n_f;
        let var: f32 = x
            .iter()
            .map(|v| {
                let d = v.to_f32().unwrap_or(0.0) - mean;
                d * d
            })
            .sum::<f32>()
            / n_f;
        let inv = (var + eps).sqrt().recip();
        let mut out: Vec<T> = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x[i].to_f32().unwrap_or(0.0);
            let gi = g[i].to_f32().unwrap_or(0.0);
            let bi = b[i].to_f32().unwrap_or(0.0);
            let v = (xi - mean) * inv * gi + bi;
            out.push(num_traits::cast(v).unwrap_or_else(T::zero));
        }
        Self::from_vec(out, &self.shape).expect("layer_norm_tokenwise: build output")
    }

    /// Element-wise tanh-approximation GELU. Falcon MLP activation.
    pub fn gelu_tanh(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).gelu_tanh_cuda()) };
        }
        let x = self.to_vec();
        const K: f32 = 0.797_884_6;
        let mut out: Vec<T> = Vec::with_capacity(x.len());
        for xi in &x {
            let v = xi.to_f32().unwrap_or(0.0);
            let y = 0.5 * v * (1.0 + (K * (v + 0.044715 * v * v * v)).tanh());
            out.push(num_traits::cast(y).unwrap_or_else(T::zero));
        }
        Self::from_vec(out, &self.shape).expect("gelu_tanh: build output")
    }

    /// In-place scaled accumulate: `self += other * scalar`. One kernel
    /// launch on GPU (instead of `mul_scalar` + `add` = two launches).
    /// MoE expert-accumulate hot path.
    pub fn scaled_add_inplace_(&mut self, other: &Self, scalar: f32) {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            unsafe {
                gpu_ref_mut(self).scaled_add_inplace_cuda_(gpu_ref(other), scalar);
            }
            return;
        }
        let o = other.to_vec();
        let x = self.to_vec();
        let mut out: Vec<T> = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let v = x[i].to_f32().unwrap_or(0.0) + o[i].to_f32().unwrap_or(0.0) * scalar;
            out.push(num_traits::cast(v).unwrap_or_else(T::zero));
        }
        *self = Self::from_vec(out, &self.shape).expect("scaled_add_inplace_: build output");
    }

    /// In-place parallel-residual add: `self += attn + ffn`. Falcon arch.
    pub fn parallel_residual_add_(&mut self, attn: &Self, ffn: &Self) {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            unsafe {
                gpu_ref_mut(self).parallel_residual_add_cuda_(gpu_ref(attn), gpu_ref(ffn));
            }
            return;
        }
        let a = attn.to_vec();
        let f = ffn.to_vec();
        let x = self.to_vec();
        let mut out: Vec<T> = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let v = x[i].to_f32().unwrap_or(0.0)
                + a[i].to_f32().unwrap_or(0.0)
                + f[i].to_f32().unwrap_or(0.0);
            out.push(num_traits::cast(v).unwrap_or_else(T::zero));
        }
        *self = Self::from_vec(out, &self.shape).expect("parallel_residual_add_: build output");
    }

    /// RMSNorm with a per-element weight scale. GPU-accelerated when enabled;
    /// CPU fallback is correct but decode-only paths should stay on GPU.
    pub fn rms_norm(&self, weight: &Self, eps: f32) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).rms_norm_cuda(gpu_ref(weight), eps)) };
        }
        // CPU fallback — parallelized for serious pure-CPU / Hailo-host performance.
        // (decode on big GPU should still prefer the CUDA path)
        let x = self.to_vec();
        let w = weight.to_vec();
        assert_eq!(x.len(), w.len(), "rms_norm: weight length must match input");
        let n = x.len();

        // Parallel sum of squares (f64 for numerical stability on large hidden dims)
        let sum_sq = if n >= 4096 {
            use rayon::prelude::*;
            x.par_iter()
                .map(|v| {
                    let f: f64 = v.to_f32().unwrap_or(0.0).into();
                    f * f
                })
                .reduce(|| 0.0, |a, b| a + b)
        } else {
            let mut s = 0.0f64;
            for v in &x {
                let f: f64 = v.to_f32().unwrap_or(0.0).into();
                s += f * f;
            }
            s
        };

        let scale = ((sum_sq / n as f64) + eps as f64).sqrt().recip() as f32;

        let mut out: Vec<T> = vec![T::zero(); n];
        if n >= 4096 {
            use rayon::prelude::*;
            out.par_iter_mut()
                .zip(x.par_iter().zip(w.par_iter()))
                .for_each(|(o, (xi, wi))| {
                    let v = xi.to_f32().unwrap_or(0.0) * scale * wi.to_f32().unwrap_or(0.0);
                    *o = num_traits::cast(v).unwrap_or_else(T::zero);
                });
        } else {
            for i in 0..n {
                let v = x[i].to_f32().unwrap_or(0.0) * scale * w[i].to_f32().unwrap_or(0.0);
                out[i] = num_traits::cast(v).unwrap_or_else(T::zero);
            }
        }
        Self::from_vec(out, &self.shape).expect("rms_norm: build output tensor")
    }

    /// Qwen3 QK-norm: per-head RMS_norm over the last `head_dim` axis.
    /// `self` is `[n_heads * head_dim]` (all heads flattened); `weight`
    /// is `[head_dim]` broadcast across every head. Returns a new tensor
    /// with the per-head norm applied; original is unchanged.
    #[must_use]
    pub fn rms_norm_heads(&self, weight: &Self, n_heads: usize, head_dim: usize, eps: f32) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).rms_norm_heads_cuda(gpu_ref(weight), n_heads, head_dim, eps))
            };
        }
        // CPU fallback — per-head rms_norm. Parallel over heads (independent).
        let x = self.to_vec();
        let w = weight.to_vec();
        assert_eq!(x.len(), n_heads * head_dim);
        assert_eq!(w.len(), head_dim);
        let mut out: Vec<T> = vec![T::zero(); x.len()];
        if n_heads > 1 {
            use rayon::prelude::*;
            let out_ptr = out.as_mut_ptr() as usize;
            (0..n_heads).into_par_iter().for_each(|h| {
                let out_ptr = out_ptr as *mut T;
                let base = h * head_dim;
                let mut sum_sq = 0.0f64;
                for i in 0..head_dim {
                    let f: f64 = x[base + i].to_f32().unwrap_or(0.0).into();
                    sum_sq += f * f;
                }
                let scale = ((sum_sq / head_dim as f64) + eps as f64).sqrt().recip() as f32;
                for i in 0..head_dim {
                    let v = x[base + i].to_f32().unwrap_or(0.0) * scale * w[i].to_f32().unwrap_or(0.0);
                    unsafe {
                        *out_ptr.add(base + i) = num_traits::cast(v).unwrap_or_else(T::zero);
                    }
                }
            });
        } else {
            for h in 0..n_heads {
                let base = h * head_dim;
                let mut sum_sq = 0.0f64;
                for i in 0..head_dim {
                    let f: f64 = x[base + i].to_f32().unwrap_or(0.0).into();
                    sum_sq += f * f;
                }
                let scale = ((sum_sq / head_dim as f64) + eps as f64).sqrt().recip() as f32;
                for i in 0..head_dim {
                    let v = x[base + i].to_f32().unwrap_or(0.0) * scale * w[i].to_f32().unwrap_or(0.0);
                    out[base + i] = num_traits::cast(v).unwrap_or_else(T::zero);
                }
            }
        }
        Self::from_vec(out, &self.shape).expect("rms_norm_heads: build output tensor")
    }

    /// Rotary position embedding in the LLaMA / Qwen / Mistral split-halves
    /// layout. Returns a new tensor with the rotation applied; original is
    /// unchanged. Input is `[n_heads * head_dim]` (single-token, all heads
    /// flattened).
    #[must_use]
    pub fn apply_rope_split_halves(
        &self,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos: usize,
    ) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).rope_split_halves_cuda(n_heads, head_dim, theta, pos))
            };
        }
        // CPU fallback - delegates to CpuBackend (will be parallelized for large cases;
        // currently efficient sequential over heads). Important for pure CPU use and
        // as reference forward when AxonML models target Hailo via NexusFoundry.
        let mut x = self.to_vec();
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SAFETY: checked + identical repr for f32 slice.
            let x_f32: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f32, x.len())
            };
            CpuBackend::apply_rope_split_halves_f32(x_f32, n_heads, head_dim, theta, pos);
        } else {
            let half = head_dim / 2;
            for h in 0..n_heads {
                for d in 0..half {
                    let base = h * head_dim + d;
                    let exponent = -(2.0f32 * d as f32) / head_dim as f32;
                    let angle = pos as f32 * theta.powf(exponent);
                    let (s, c) = angle.sin_cos();
                    let a = x[base].to_f32().unwrap_or(0.0);
                    let b = x[base + half].to_f32().unwrap_or(0.0);
                    x[base] = num_traits::cast(c * a - s * b).unwrap_or_else(T::zero);
                    x[base + half] = num_traits::cast(s * a + c * b).unwrap_or_else(T::zero);
                }
            }
        }
        Self::from_vec(x, &self.shape).expect("apply_rope: build output tensor")
    }

    /// Fused residual-add + batched RMSNorm: returns `(RMSNorm(self + b), self + b)`.
    /// The raw sum is saved for the backward pass so it doesn't need to rerun
    /// the add. `self` and `b` are `[m, n]`; `weight` is `[n]`.
    ///
    /// Replaces the per-layer `residual.add(x).rms_norm(weight)` pair with one
    /// kernel — eliminates a broadcast_add + alloc + RMSNorm kernel launch per
    /// residual path (2 × per Qwen3 layer).
    #[must_use]
    pub fn add_rmsnorm_batched(
        &self,
        b: &Self,
        weight: &Self,
        m: usize,
        n: usize,
        eps: f32,
    ) -> (Self, Self) {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let (out, sum) = unsafe {
                gpu_ref(self).add_rmsnorm_batched_cuda(gpu_ref(b), gpu_ref(weight), m, n, eps)
            };
            return (unsafe { gpu_into(out) }, unsafe { gpu_into(sum) });
        }
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>(),
            "add_rmsnorm_batched CPU path requires f32",
        );
        let av = self.to_vec();
        let bv = b.to_vec();
        let w = weight.to_vec();
        assert_eq!(av.len(), m * n);
        assert_eq!(bv.len(), m * n);
        assert_eq!(w.len(), n);
        let mut sum_out: Vec<T> = Vec::with_capacity(m * n);
        let mut out: Vec<T> = Vec::with_capacity(m * n);
        for t in 0..m {
            let base = t * n;
            let mut sum_sq = 0.0f64;
            for i in 0..n {
                let ai = av[base + i].to_f32().unwrap_or(0.0);
                let bi = bv[base + i].to_f32().unwrap_or(0.0);
                let s = ai + bi;
                sum_out.push(num_traits::cast(s).unwrap_or_else(T::zero));
                sum_sq += (s as f64) * (s as f64);
            }
            let scale = ((sum_sq / n as f64) + eps as f64).sqrt().recip() as f32;
            for i in 0..n {
                let s = sum_out[base + i].to_f32().unwrap_or(0.0);
                let wi = w[i].to_f32().unwrap_or(0.0);
                out.push(num_traits::cast(s * scale * wi).unwrap_or_else(T::zero));
            }
        }
        let out_t = Self::from_vec(out, &[m, n]).expect("add_rmsnorm_batched: build output");
        let sum_t = Self::from_vec(sum_out, &[m, n]).expect("add_rmsnorm_batched: build sum");
        (out_t, sum_t)
    }

    /// Fused causal-scaled softmax. `self` is the raw attention scores
    /// `[..., Tq, Tk]`; applies `softmax(scale * scores + causal_mask)`
    /// over the last dim. `offset` is the KV-cache position offset (0
    /// during training). Masked positions (j > offset + i) are exactly 0.
    ///
    /// Replaces the `mul_scalar(scale) + add(mask) + softmax(-1)` chain —
    /// 3 kernels + a CPU mask alloc per call collapse to 1 kernel launch.
    #[must_use]
    pub fn softmax_causal_scaled(&self, tq: usize, tk: usize, offset: usize, scale: f32) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).softmax_causal_scaled_cuda(tq, tk, offset, scale))
            };
        }
        // CPU fallback: same math, row by row.
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>(),
            "softmax_causal_scaled CPU path requires f32",
        );
        let total = self.numel();
        assert!(total % tk == 0 && (total / tk) % tq == 0);
        let num_rows = total / tk;
        let src = self.to_vec();
        let mut out: Vec<T> = Vec::with_capacity(total);
        for r in 0..num_rows {
            let q_pos = r % tq;
            let max_k = offset + q_pos;
            let base = r * tk;
            let mut row_max = f32::NEG_INFINITY;
            for j in 0..tk {
                let v = if j > max_k {
                    f32::NEG_INFINITY
                } else {
                    src[base + j].to_f32().unwrap_or(0.0) * scale
                };
                if v > row_max {
                    row_max = v;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..tk {
                if j > max_k {
                    continue;
                }
                let v = src[base + j].to_f32().unwrap_or(0.0) * scale;
                sum += (v - row_max).exp();
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for j in 0..tk {
                let v = if j > max_k {
                    0.0
                } else {
                    let s = src[base + j].to_f32().unwrap_or(0.0) * scale;
                    (s - row_max).exp() * inv
                };
                out.push(num_traits::cast(v).unwrap_or_else(T::zero));
            }
        }
        Self::from_vec(out, self.shape()).expect("softmax_causal_scaled: build output")
    }

    /// Fused causal-scaled softmax backward wrt raw scores. `self` is the
    /// saved forward output `p` (masked positions are 0); `grad_output`
    /// is `dL/dp`. Returns `grad_scores = scale * p * (grad_out - Σ(p·grad_out))`
    /// per row; masked positions naturally zero because `p = 0`.
    #[must_use]
    pub fn softmax_causal_scaled_bwd(&self, grad_output: &Self, tk: usize, scale: f32) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).softmax_causal_scaled_bwd_cuda(
                    gpu_ref(grad_output),
                    tk,
                    scale,
                ))
            };
        }
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>(),
            "softmax_causal_scaled_bwd CPU path requires f32",
        );
        let total = self.numel();
        assert_eq!(total, grad_output.numel());
        assert!(total % tk == 0);
        let num_rows = total / tk;
        let p = self.to_vec();
        let g = grad_output.to_vec();
        let mut out: Vec<T> = Vec::with_capacity(total);
        for r in 0..num_rows {
            let base = r * tk;
            let mut dot = 0.0f32;
            for j in 0..tk {
                let pj = p[base + j].to_f32().unwrap_or(0.0);
                let gj = g[base + j].to_f32().unwrap_or(0.0);
                dot += pj * gj;
            }
            for j in 0..tk {
                let pj = p[base + j].to_f32().unwrap_or(0.0);
                let gj = g[base + j].to_f32().unwrap_or(0.0);
                let v = scale * pj * (gj - dot);
                out.push(num_traits::cast(v).unwrap_or_else(T::zero));
            }
        }
        Self::from_vec(out, self.shape()).expect("softmax_causal_scaled_bwd: build output")
    }

    /// Batched RMSNorm backward (grad_input only). `self` is the saved
    /// forward input `[m, n]`, `weight` is `[n]`, `grad_output` is `[m, n]`.
    /// Returns `[m, n]` grad_input matching the CPU-only reference math in
    /// `axonml-llm::RMSNormBackward` (weight-gradient path not required
    /// because the existing autograd path doesn't route grads to the weight).
    #[must_use]
    pub fn rms_norm_bwd_batched(
        &self,
        weight: &Self,
        grad_output: &Self,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).rms_norm_bwd_batched_cuda(
                    gpu_ref(weight),
                    gpu_ref(grad_output),
                    m,
                    n,
                    eps,
                ))
            };
        }
        // CPU fallback: same math as axonml-llm's RMSNormBackward::apply.
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>(),
            "rms_norm_bwd_batched CPU path requires f32",
        );
        let x = self.to_vec();
        let w = weight.to_vec();
        let g = grad_output.to_vec();
        assert_eq!(x.len(), m * n);
        assert_eq!(w.len(), n);
        assert_eq!(g.len(), m * n);
        let mut out: Vec<T> = Vec::with_capacity(m * n);
        let d = n as f32;
        for t in 0..m {
            let base = t * n;
            let mut sum_sq = 0.0f64;
            let mut dot = 0.0f64;
            for i in 0..n {
                let xi = x[base + i].to_f32().unwrap_or(0.0);
                let wi = w[i].to_f32().unwrap_or(0.0);
                let gi = g[base + i].to_f32().unwrap_or(0.0);
                sum_sq += (xi as f64) * (xi as f64);
                dot += (xi as f64) * (wi as f64) * (gi as f64);
            }
            let rms_inv = ((sum_sq / n as f64) + eps as f64).sqrt().recip() as f32;
            let rms3_inv = rms_inv * rms_inv * rms_inv;
            let dot_scaled = (dot as f32) * rms3_inv / d;
            for i in 0..n {
                let xi = x[base + i].to_f32().unwrap_or(0.0);
                let wi = w[i].to_f32().unwrap_or(0.0);
                let gi = g[base + i].to_f32().unwrap_or(0.0);
                let term1 = wi * gi * rms_inv;
                let term2 = xi * dot_scaled;
                out.push(num_traits::cast(term1 - term2).unwrap_or_else(T::zero));
            }
        }
        Self::from_vec(out, &[m, n]).expect("rms_norm_bwd_batched: build output")
    }

    /// Batched RMSNorm over `m` tokens. `self` is `[m, n]`; `weight` is `[n]`.
    #[must_use]
    pub fn rms_norm_batched(&self, weight: &Self, m: usize, n: usize, eps: f32) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).rms_norm_batched_cuda(gpu_ref(weight), m, n, eps))
            };
        }
        // CPU fallback: independent rms_norm over each of m rows.
        let x = self.to_vec();
        let w = weight.to_vec();
        assert_eq!(x.len(), m * n, "rms_norm_batched: expected m*n");
        assert_eq!(w.len(), n, "rms_norm_batched: weight len mismatch");
        let mut out: Vec<T> = Vec::with_capacity(m * n);
        for t in 0..m {
            let base = t * n;
            let mut sum_sq = 0.0f64;
            for i in 0..n {
                let f: f64 = x[base + i].to_f32().unwrap_or(0.0).into();
                sum_sq += f * f;
            }
            let scale = ((sum_sq / n as f64) + eps as f64).sqrt().recip() as f32;
            for i in 0..n {
                let v = x[base + i].to_f32().unwrap_or(0.0) * scale * w[i].to_f32().unwrap_or(0.0);
                out.push(num_traits::cast(v).unwrap_or_else(T::zero));
            }
        }
        Self::from_vec(out, &[m, n]).expect("rms_norm_batched: build output")
    }

    /// Batched Qwen3 QK-norm over `m` tokens. `self` is `[m, n_heads * head_dim]`;
    /// `weight` is `[head_dim]` broadcast across every (token, head).
    #[must_use]
    pub fn rms_norm_heads_batched(
        &self,
        weight: &Self,
        m: usize,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).rms_norm_heads_batched_cuda(
                    gpu_ref(weight),
                    m,
                    n_heads,
                    head_dim,
                    eps,
                ))
            };
        }
        // CPU fallback: m × rms_norm_heads.
        let x = self.to_vec();
        let w = weight.to_vec();
        let total = m * n_heads * head_dim;
        assert_eq!(x.len(), total);
        assert_eq!(w.len(), head_dim);
        let mut out: Vec<T> = Vec::with_capacity(total);
        for t in 0..m {
            for h in 0..n_heads {
                let base = t * n_heads * head_dim + h * head_dim;
                let mut sum_sq = 0.0f64;
                for i in 0..head_dim {
                    let f: f64 = x[base + i].to_f32().unwrap_or(0.0).into();
                    sum_sq += f * f;
                }
                let scale = ((sum_sq / head_dim as f64) + eps as f64).sqrt().recip() as f32;
                for i in 0..head_dim {
                    let v =
                        x[base + i].to_f32().unwrap_or(0.0) * scale * w[i].to_f32().unwrap_or(0.0);
                    out.push(num_traits::cast(v).unwrap_or_else(T::zero));
                }
            }
        }
        Self::from_vec(out, &self.shape).expect("rms_norm_heads_batched: build output")
    }

    /// Batched split-halves RoPE over `m` tokens at positions
    /// `[pos_start, pos_start + m)`. `self` is `[m, n_heads * head_dim]`.
    #[must_use]
    pub fn apply_rope_split_halves_batched(
        &self,
        m: usize,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(
                    gpu_ref(self).apply_rope_split_halves_batched_cuda(
                        m, n_heads, head_dim, theta, pos_start,
                    ),
                )
            };
        }
        // CPU fallback - parallel over m tokens using rayon (flat per-row work).
        // Serious optimization for CPU prefill and Hailo ref paths.
        let mut x = self.to_vec();
        let half = head_dim / 2;
        let row_stride = n_heads * head_dim;
        if x.len() >= 4096 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            use rayon::prelude::*;
            // Sequential for now (complex strided); outer batch parallel possible in caller for large m.
            // Reductions/rms/swiglu have full parallel.
            let x_f32: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f32, x.len())
            };
            x_f32.par_chunks_mut(row_stride).enumerate().for_each(|(t, chunk)| {
                let pos = pos_start + t;
                for h in 0..n_heads {
                    for d in 0..half {
                        let base = h * head_dim + d;
                        let exponent = -(2.0f32 * d as f32) / head_dim as f32;
                        let angle = pos as f32 * theta.powf(exponent);
                        let (s, c) = angle.sin_cos();
                        let a = chunk[base];
                        let b_val = chunk[base + half];
                        chunk[base] = c * a - s * b_val;
                        chunk[base + half] = s * a + c * b_val;
                    }
                }
            });
        } else {
            for t in 0..m {
                let pos = pos_start + t;
                for h in 0..n_heads {
                    for d in 0..half {
                        let base = t * row_stride + h * head_dim + d;
                        let exponent = -(2.0f32 * d as f32) / head_dim as f32;
                        let angle = pos as f32 * theta.powf(exponent);
                        let (s, c) = angle.sin_cos();
                        let a = x[base].to_f32().unwrap_or(0.0);
                        let b = x[base + half].to_f32().unwrap_or(0.0);
                        x[base] = num_traits::cast(c * a - s * b).unwrap_or_else(T::zero);
                        x[base + half] = num_traits::cast(s * a + c * b).unwrap_or_else(T::zero);
                    }
                }
            }
        }
        Self::from_vec(x, &self.shape).expect("rope_batched: build output")
    }

    /// Head-major split-halves RoPE backward (inverse rotation) for
    /// `[bs, n_heads, seq, head_dim]`. `self` is `grad_output`.
    #[must_use]
    pub fn rope_split_halves_bhsd_bwd(
        &self,
        bs: usize,
        n_heads: usize,
        seq: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(
                    gpu_ref(self).rope_split_halves_bhsd_bwd_cuda(
                        bs, n_heads, seq, head_dim, theta, pos_start,
                    ),
                )
            };
        }
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>(),
            "rope_split_halves_bhsd_bwd CPU path requires f32",
        );
        let g = self.to_vec();
        // CPU fallback - parallel via rayon over bs/heads/seq.
        let mut out: Vec<T> = g.clone();
        let half = head_dim / 2;
        if out.len() >= 4096 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            use rayon::prelude::*;
            // Parallel over tokens using par_chunks_mut.
            let out_f32: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out.len())
            };
            let _tokens = bs * n_heads * seq;
            out_f32.par_chunks_mut(head_dim).enumerate().for_each(|(tok, chunk)| {
                let _b = tok / (n_heads * seq);
                let _h = (tok / seq) % n_heads;
                let t = tok % seq;
                let pos = pos_start + t;
                for d in 0..half {
                    let exponent = -(2.0f32 * d as f32) / head_dim as f32;
                    let angle = pos as f32 * theta.powf(exponent);
                    let (s, c) = angle.sin_cos();
                    let dy1 = chunk[d];
                    let dy2 = chunk[d + half];
                    chunk[d] = c * dy1 + s * dy2;
                    chunk[d + half] = -s * dy1 + c * dy2;
                }
            });
        } else {
            for b in 0..bs {
                for h in 0..n_heads {
                    for t in 0..seq {
                        let pos = pos_start + t;
                        let base = ((b * n_heads + h) * seq + t) * head_dim;
                        for d in 0..half {
                            let exponent = -(2.0f32 * d as f32) / head_dim as f32;
                            let angle = pos as f32 * theta.powf(exponent);
                            let (s, c) = angle.sin_cos();
                            let dy1 = g[base + d].to_f32().unwrap_or(0.0);
                            let dy2 = g[base + d + half].to_f32().unwrap_or(0.0);
                            out[base + d] = num_traits::cast(c * dy1 + s * dy2).unwrap_or_else(T::zero);
                            out[base + d + half] =
                                num_traits::cast(-s * dy1 + c * dy2).unwrap_or_else(T::zero);
                        }
                    }
                }
            }
        }
        Self::from_vec(out, &self.shape).expect("rope_bhsd_bwd: build output")
    }

    /// GQA `repeat_kv`: duplicate each KV head `n_rep` times consecutively.
    /// `self` shape is `[bs, kv_heads, seq, head_dim]`, output is
    /// `[bs, kv_heads * n_rep, seq, head_dim]`.
    #[must_use]
    pub fn repeat_kv(
        &self,
        bs: usize,
        kv_heads: usize,
        n_rep: usize,
        seq: usize,
        head_dim: usize,
    ) -> Self {
        if n_rep == 1 {
            return self.clone();
        }
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).repeat_kv_cuda(bs, kv_heads, n_rep, seq, head_dim))
            };
        }
        // CPU fallback.
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>(),
            "repeat_kv CPU path requires f32",
        );
        let src = self.to_vec();
        let mut out: Vec<T> = Vec::with_capacity(bs * kv_heads * n_rep * seq * head_dim);
        for b in 0..bs {
            for h in 0..kv_heads {
                for _ in 0..n_rep {
                    for t in 0..seq {
                        let base = ((b * kv_heads + h) * seq + t) * head_dim;
                        for d in 0..head_dim {
                            out.push(src[base + d]);
                        }
                    }
                }
            }
        }
        let shape = [bs, kv_heads * n_rep, seq, head_dim];
        Self::from_vec(out, &shape).expect("repeat_kv: build output")
    }

    /// Head-major split-halves RoPE for Qwen3 / LLaMA training forward.
    /// `self` is `[bs, n_heads, seq, head_dim]` contiguous. Rotates each
    /// (b, h, t) token at position `pos_start + t`.
    #[must_use]
    pub fn apply_rope_split_halves_bhsd(
        &self,
        bs: usize,
        n_heads: usize,
        seq: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe {
                gpu_into(gpu_ref(self).apply_rope_split_halves_bhsd_cuda(
                    bs, n_heads, seq, head_dim, theta, pos_start,
                ))
            };
        }
        // CPU fallback - parallel over bs*heads*seq using rayon.
        // Win for CPU and Hailo ref.
        let mut x = self.to_vec();
        let half = head_dim / 2;
        if x.len() >= 4096 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            use rayon::prelude::*;
            // Parallel over tokens (b*h*t) using par_chunks_mut on head_dim chunks.
            let x_f32: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f32, x.len())
            };
            let _tokens = bs * n_heads * seq;
            x_f32.par_chunks_mut(head_dim).enumerate().for_each(|(tok, chunk)| {
                let _b = tok / (n_heads * seq);
                let _h = (tok / seq) % n_heads;
                let t = tok % seq;
                let pos = pos_start + t;
                for d in 0..half {
                    let exponent = -(2.0f32 * d as f32) / head_dim as f32;
                    let angle = pos as f32 * theta.powf(exponent);
                    let (s, c) = angle.sin_cos();
                    let a = chunk[d];
                    let bv = chunk[d + half];
                    chunk[d] = c * a - s * bv;
                    chunk[d + half] = s * a + c * bv;
                }
            });
        } else {
            let half = head_dim / 2;
            for b in 0..bs {
                for h in 0..n_heads {
                    for t in 0..seq {
                        let pos = pos_start + t;
                        let base = ((b * n_heads + h) * seq + t) * head_dim;
                        for d in 0..half {
                            let exponent = -(2.0f32 * d as f32) / head_dim as f32;
                            let angle = pos as f32 * theta.powf(exponent);
                            let (s, c) = angle.sin_cos();
                            let a = x[base + d].to_f32().unwrap_or(0.0);
                            let bv = x[base + d + half].to_f32().unwrap_or(0.0);
                            x[base + d] = num_traits::cast(c * a - s * bv).unwrap_or_else(T::zero);
                            x[base + d + half] =
                                num_traits::cast(s * a + c * bv).unwrap_or_else(T::zero);
                        }
                    }
                }
            }
        }
        Self::from_vec(x, &self.shape).expect("rope_bhsd: build output")
    }

    /// Broadcast per-column bias add across `m` rows: `out[t, c] = self[t, c] + bias[c]`.
    #[must_use]
    pub fn add_bias_batched(&self, bias: &Self, m: usize, n: usize) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).add_bias_batched_cuda(gpu_ref(bias), m, n)) };
        }
        // CPU fallback.
        let x = self.to_vec();
        let b = bias.to_vec();
        assert_eq!(x.len(), m * n);
        assert_eq!(b.len(), n);
        let mut out: Vec<T> = Vec::with_capacity(m * n);
        for t in 0..m {
            for c in 0..n {
                let v = x[t * n + c].to_f32().unwrap_or(0.0) + b[c].to_f32().unwrap_or(0.0);
                out.push(num_traits::cast(v).unwrap_or_else(T::zero));
            }
        }
        Self::from_vec(out, &[m, n]).expect("add_bias_batched: build output")
    }

    /// Fused SwiGLU backward. `self` is the saved forward gate, `up` is the
    /// saved forward up, `grad_output` is `dL/dy`. Returns `(grad_gate, grad_up)`.
    /// Replaces the `SiluBackward + MulBackward` kernel pair on the MLP path
    /// with a single kernel producing both gradients.
    #[must_use]
    pub fn swiglu_bwd(&self, up: &Self, grad_output: &Self) -> (Self, Self) {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let (gg, gu) =
                unsafe { gpu_ref(self).swiglu_bwd_cuda(gpu_ref(up), gpu_ref(grad_output)) };
            return (unsafe { gpu_into(gg) }, unsafe { gpu_into(gu) });
        }
        assert!(
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>(),
            "swiglu_bwd CPU path requires f32",
        );
        let g = self.to_vec();
        let u = up.to_vec();
        let go = grad_output.to_vec();
        let n = g.len();
        assert_eq!(u.len(), n);
        assert_eq!(go.len(), n);
        let mut grad_gate: Vec<T> = vec![T::zero(); n];
        let mut grad_up: Vec<T> = vec![T::zero(); n];

        if n >= 4096 {
            use rayon::prelude::*;
            let gg_ptr = grad_gate.as_mut_ptr() as usize;
            let gu_ptr = grad_up.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                let gg_ptr = gg_ptr as *mut T;
                let gu_ptr = gu_ptr as *mut T;
                let gi = g[i].to_f32().unwrap_or(0.0);
                let ui = u[i].to_f32().unwrap_or(0.0);
                let goi = go[i].to_f32().unwrap_or(0.0);
                let sig = 1.0f32 / (1.0f32 + (-gi).exp());
                let silu_g = gi * sig;
                let silu_deriv = sig * (1.0f32 + gi * (1.0f32 - sig));
                unsafe {
                    *gg_ptr.add(i) = num_traits::cast(goi * ui * silu_deriv).unwrap_or_else(T::zero);
                    *gu_ptr.add(i) = num_traits::cast(goi * silu_g).unwrap_or_else(T::zero);
                }
            });
        } else {
            for i in 0..n {
                let gi = g[i].to_f32().unwrap_or(0.0);
                let ui = u[i].to_f32().unwrap_or(0.0);
                let goi = go[i].to_f32().unwrap_or(0.0);
                let sig = 1.0f32 / (1.0f32 + (-gi).exp());
                let silu_g = gi * sig;
                let silu_deriv = sig * (1.0f32 + gi * (1.0f32 - sig));
                grad_gate[i] = num_traits::cast(goi * ui * silu_deriv).unwrap_or_else(T::zero);
                grad_up[i] = num_traits::cast(goi * silu_g).unwrap_or_else(T::zero);
            }
        }

        let gg_t = Self::from_vec(grad_gate, &self.shape).expect("swiglu_bwd: grad_gate");
        let gu_t = Self::from_vec(grad_up, &self.shape).expect("swiglu_bwd: grad_up");
        (gg_t, gu_t)
    }

    /// Fused SwiGLU: `out = SiLU(self) * up`. `self` is the gate.
    #[must_use]
    pub fn swiglu(&self, up: &Self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).swiglu_cuda(gpu_ref(up))) };
        }
        // CPU fallback: silu(gate) * up. Parallelized for serious CPU performance.
        let g = self.to_vec();
        let u = up.to_vec();
        let mut out: Vec<T> = vec![T::zero(); g.len()];

        if g.len() >= 4096 {
            use rayon::prelude::*;
            out.par_iter_mut()
                .zip(g.par_iter().zip(u.par_iter()))
                .for_each(|(o, (gi, ui))| {
                    let g32 = gi.to_f32().unwrap_or(0.0);
                    let silu = g32 / (1.0 + (-g32).exp());
                    *o = num_traits::cast(silu * ui.to_f32().unwrap_or(0.0)).unwrap_or_else(T::zero);
                });
        } else {
            for i in 0..g.len() {
                let gi = g[i].to_f32().unwrap_or(0.0);
                let silu = gi / (1.0 + (-gi).exp());
                out[i] = num_traits::cast(silu * u[i].to_f32().unwrap_or(0.0)).unwrap_or_else(T::zero);
            }
        }
        Self::from_vec(out, &self.shape).expect("swiglu: build output tensor")
    }

    /// Read-only access to the underlying GPU storage as a `CudaSlice<f32>`.
    /// Panics if the tensor is on CPU. Used by downstream crates (e.g.
    /// nexus-serve) that need to pass the GPU buffer directly into a kernel
    /// without going through `.to_vec()` + re-upload.
    ///
    /// Only valid for `Tensor<f32>` — the underlying storage is always f32
    /// on GPU regardless of the Tensor's generic type. The guard holds a
    /// read lock on the storage for its lifetime.
    #[cfg(feature = "cuda")]
    pub fn as_cuda_slice_read(&self) -> axonml_core::storage::CudaSliceReadGuard<'_> {
        assert!(
            self.device().is_gpu(),
            "as_cuda_slice_read: tensor must be on GPU"
        );
        assert!(is_f32::<T>(), "as_cuda_slice_read: GPU storage is f32-only");
        let self_f32 = unsafe { gpu_ref(self) };
        self_f32.storage.as_cuda_slice()
    }

    /// Write-guarded access to the underlying GPU storage as a mutable
    /// `CudaSlice<f32>`. Same contract as `as_cuda_slice_read` but takes the
    /// storage's write lock — used for in-place kernels and for the pre-
    /// bound workspace-tensor path under CUDA graph capture.
    ///
    /// Panics if the tensor is on CPU or is not `Tensor<f32>`.
    #[cfg(feature = "cuda")]
    pub fn as_cuda_slice_write(&self) -> axonml_core::storage::CudaSliceWriteGuard<'_> {
        assert!(
            self.device().is_gpu(),
            "as_cuda_slice_write: tensor must be on GPU"
        );
        assert!(
            is_f32::<T>(),
            "as_cuda_slice_write: GPU storage is f32-only"
        );
        let self_f32 = unsafe { gpu_ref(self) };
        self_f32.storage.as_cuda_slice_mut()
    }

    /// BitNet b1.58 fused gate: `out = ReLU(self)² * up`. `self` is the gate.
    #[must_use]
    pub fn relu2_gate(&self, up: &Self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            return unsafe { gpu_into(gpu_ref(self).relu2_gate_cuda(gpu_ref(up))) };
        }
        // CPU fallback.
        let g = self.to_vec();
        let u = up.to_vec();
        let mut out: Vec<T> = Vec::with_capacity(g.len());
        for i in 0..g.len() {
            let gi = g[i].to_f32().unwrap_or(0.0).max(0.0);
            out.push(
                num_traits::cast(gi * gi * u[i].to_f32().unwrap_or(0.0)).unwrap_or_else(T::zero),
            );
        }
        Self::from_vec(out, &self.shape).expect("relu2_gate: build output tensor")
    }

    /// Softmax along specified dimension.
    #[must_use]
    pub fn softmax(&self, dim: i32) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            return unsafe { gpu_into(self_f32.softmax_cuda(dim).expect("CUDA softmax failed")) };
        }
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

        // GPU fast path: sum_dim then divide by dim_size (all on GPU)
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            let summed = if keepdim {
                self_f32.sum_dim_keepdim_cuda(dim)
            } else {
                self_f32.sum_dim_cuda(dim)
            };
            let dim_size = self.shape[dim];
            let result = summed.mul_scalar_cuda(1.0 / dim_size as f32);
            return unsafe { gpu_into(result) };
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

    /// Sum along a dimension.
    #[must_use]
    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Self {
        let ndim = self.ndim();
        let dim = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return self.clone();
        }

        // GPU fast path: use CUDA sum_dim kernel (no CPU copies)
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            let result = if keepdim {
                self_f32.sum_dim_keepdim_cuda(dim)
            } else {
                self_f32.sum_dim_cuda(dim)
            };
            return unsafe { gpu_into(result) };
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

        let outer_size: usize = self.shape[..dim].iter().product();
        let inner_size: usize = self.shape[dim + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = T::zero();
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    sum = sum + data[idx];
                }
                let result_idx = outer * inner_size + inner;
                result[result_idx] = sum;
            }
        }

        Self::from_vec(result, &new_shape).unwrap()
    }

    /// Variance along a dimension.
    #[must_use]
    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Self {
        // variance = E[x²] - E[x]²  (saves one full-size intermediate allocation)
        let mean = self.mean_dim(dim, true);
        let sq = self.mul(self).unwrap_or_else(|_| self.clone());
        let mean_sq = sq.mean_dim(dim, keepdim);
        let mean_keepdim = if keepdim {
            mean.clone()
        } else {
            self.mean_dim(dim, keepdim)
        };
        let mean_squared = mean_keepdim
            .mul(&mean_keepdim)
            .unwrap_or_else(|_| mean_keepdim.clone());
        mean_sq
            .sub(&mean_squared)
            .unwrap_or_else(|_| mean_sq.clone())
    }

    /// Broadcasts tensor to a new shape.
    #[must_use]
    pub fn broadcast_to(&self, shape: &[usize]) -> Self {
        if self.shape.as_slice() == shape {
            return self.clone();
        }

        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            return unsafe {
                gpu_into(
                    self_f32
                        .broadcast_to_cuda(shape)
                        .expect("CUDA broadcast_to failed"),
                )
            };
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

        let out = Self::from_vec(result_data, &new_shape).unwrap();
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            return out.to_device(self.device()).unwrap();
        }
        out
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
        #[cfg(feature = "cuda")]
        {
            let self_gpu = self.device().is_gpu();
            let other_gpu = other.device().is_gpu();
            if self_gpu || other_gpu {
                assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
                if self_gpu && other_gpu {
                    let (s, o) = unsafe { (gpu_ref(self), gpu_ref(other)) };
                    if self.shape == other.shape {
                        return Ok(unsafe { gpu_into(s.add_cuda(o)?) });
                    } else {
                        return Ok(unsafe { gpu_into(s.broadcast_add_cuda(o)?) });
                    }
                }
                // Mixed device — move to GPU, then operate
                let target_device = if self_gpu {
                    self.device()
                } else {
                    other.device()
                };
                let a_gpu = if self_gpu {
                    self.clone()
                } else {
                    self.to_device(target_device)?
                };
                let b_gpu = if other_gpu {
                    other.clone()
                } else {
                    other.to_device(target_device)?
                };
                return a_gpu.add(&b_gpu);
            }
        }
        // Fast path: same shape, both contiguous — no index arithmetic needed
        if self.shape == other.shape && self.is_contiguous() && other.is_contiguous() {
            let a = self.storage.as_slice();
            let b = other.storage.as_slice();
            let ao = self.offset;
            let bo = other.offset;
            let n = numel(&self.shape);
            let mut result_data = vec![T::zero(); n];
            for i in 0..n {
                result_data[i] = a[ao + i] + b[bo + i];
            }
            return Self::from_vec(result_data, &self.shape);
        }

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
        #[cfg(feature = "cuda")]
        {
            let self_gpu = self.device().is_gpu();
            let other_gpu = other.device().is_gpu();
            if self_gpu || other_gpu {
                assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
                if self_gpu && other_gpu {
                    let (s, o) = unsafe { (gpu_ref(self), gpu_ref(other)) };
                    if self.shape == other.shape {
                        return Ok(unsafe { gpu_into(s.sub_cuda(o)?) });
                    } else {
                        return Ok(unsafe { gpu_into(s.broadcast_sub_cuda(o)?) });
                    }
                }
                let target = if self_gpu {
                    self.device()
                } else {
                    other.device()
                };
                let a_gpu = if self_gpu {
                    self.clone()
                } else {
                    self.to_device(target)?
                };
                let b_gpu = if other_gpu {
                    other.clone()
                } else {
                    other.to_device(target)?
                };
                return a_gpu.sub(&b_gpu);
            }
        }
        // Fast path: same shape, contiguous
        if self.shape == other.shape && self.is_contiguous() && other.is_contiguous() {
            let a = self.storage.as_slice();
            let b = other.storage.as_slice();
            let (ao, bo) = (self.offset, other.offset);
            let n = numel(&self.shape);
            let mut r = vec![T::zero(); n];
            for i in 0..n {
                r[i] = a[ao + i] - b[bo + i];
            }
            return Self::from_vec(r, &self.shape);
        }

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
        #[cfg(feature = "cuda")]
        {
            let self_gpu = self.device().is_gpu();
            let other_gpu = other.device().is_gpu();
            if self_gpu || other_gpu {
                assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
                if self_gpu && other_gpu {
                    let (s, o) = unsafe { (gpu_ref(self), gpu_ref(other)) };
                    if self.shape == other.shape {
                        return Ok(unsafe { gpu_into(s.mul_cuda(o)?) });
                    } else {
                        return Ok(unsafe { gpu_into(s.broadcast_mul_cuda(o)?) });
                    }
                }
                let target = if self_gpu {
                    self.device()
                } else {
                    other.device()
                };
                let a_gpu = if self_gpu {
                    self.clone()
                } else {
                    self.to_device(target)?
                };
                let b_gpu = if other_gpu {
                    other.clone()
                } else {
                    other.to_device(target)?
                };
                return a_gpu.mul(&b_gpu);
            }
        }
        // Fast path: same shape, contiguous
        if self.shape == other.shape && self.is_contiguous() && other.is_contiguous() {
            let a = self.storage.as_slice();
            let b = other.storage.as_slice();
            let (ao, bo) = (self.offset, other.offset);
            let n = numel(&self.shape);
            let mut r = vec![T::zero(); n];
            for i in 0..n {
                r[i] = a[ao + i] * b[bo + i];
            }
            return Self::from_vec(r, &self.shape);
        }

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
        #[cfg(feature = "cuda")]
        {
            let self_gpu = self.device().is_gpu();
            let other_gpu = other.device().is_gpu();
            if self_gpu || other_gpu {
                assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
                if self_gpu && other_gpu {
                    let (s, o) = unsafe { (gpu_ref(self), gpu_ref(other)) };
                    if self.shape == other.shape {
                        return Ok(unsafe { gpu_into(s.div_cuda(o)?) });
                    } else {
                        return Ok(unsafe { gpu_into(s.broadcast_div_cuda(o)?) });
                    }
                }
                let target = if self_gpu {
                    self.device()
                } else {
                    other.device()
                };
                let a_gpu = if self_gpu {
                    self.clone()
                } else {
                    self.to_device(target)?
                };
                let b_gpu = if other_gpu {
                    other.clone()
                } else {
                    other.to_device(target)?
                };
                return a_gpu.div(&b_gpu);
            }
        }
        // Fast path: same shape, contiguous
        if self.shape == other.shape && self.is_contiguous() && other.is_contiguous() {
            let a = self.storage.as_slice();
            let b = other.storage.as_slice();
            let (ao, bo) = (self.offset, other.offset);
            let n = numel(&self.shape);
            let mut r = vec![T::zero(); n];
            for i in 0..n {
                r[i] = a[ao + i] / b[bo + i];
            }
            return Self::from_vec(r, &self.shape);
        }

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
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            let scalar_f32: f32 = unsafe { *(&scalar as *const T as *const f32) };
            return unsafe { gpu_into(self_f32.add_scalar_cuda(scalar_f32)) };
        }
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::add_scalar(&mut result, &data, scalar);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn mul_scalar(&self, scalar: T) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            let scalar_f32: f32 = unsafe { *(&scalar as *const T as *const f32) };
            return unsafe { gpu_into(self_f32.mul_scalar_cuda(scalar_f32)) };
        }
        let data = self.to_vec();
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::mul_scalar(&mut result, &data, scalar);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Element-wise negation.
    #[must_use]
    pub fn neg(&self) -> Self {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let self_f32 = unsafe { gpu_ref(self) };
            return unsafe { gpu_into(self_f32.neg_cuda()) };
        }
        let storage = self.storage.as_slice();
        let fast = self.is_contiguous() && self.offset == 0;
        let slice: &[T] = if fast { &storage[..self.numel()] } else { &[] };
        let owned: Option<Vec<T>> = if fast { None } else { Some(self.to_vec()) };
        let data: &[T] = owned.as_deref().unwrap_or(slice);
        let mut result = vec![T::zero(); data.len()];
        CpuBackend::neg(&mut result, data);
        Self::from_vec(result, &self.shape).unwrap()
    }

    /// Matrix multiplication with batching support.
    ///
    /// Supports:
    /// - 2D @ 2D: [m, k] @ [k, n] -> [m, n]
    /// - 3D @ 3D: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    /// - 4D @ 4D: [b1, b2, m, k] @ [b1, b2, k, n] -> [b1, b2, m, n]
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.device().is_gpu() {
            assert!(is_f32::<T>(), "GPU tensors are only supported for f32");
            let (s, o) = unsafe { (gpu_ref(self), gpu_ref(other)) };
            return Ok(unsafe { gpu_into(s.matmul_cuda(o)?) });
        }
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

        // For 2D matrices, do simple matmul.
        //
        // Fast path: when both tensors are already contiguous with offset 0
        // (the common case for pre-loaded weights and intermediate activations),
        // read the storage slices directly and skip the `.to_vec()` allocation-
        // and-copy. This is critical for LLM inference where the same weight
        // matrix is multiplied by a different input for every decoded token:
        // without this path we pay a full weight-matrix memcpy per matmul
        // (~10 MB × 7 matmuls × 42 layers = ~3 GB of memcpy per decode token
        // on an 8B model).
        if self.ndim() == 2 && other.ndim() == 2 {
            let self_fast = self.is_contiguous() && self.offset == 0;
            let other_fast = other.is_contiguous() && other.offset == 0;

            // We still need owned buffers for CPU matmul (matrixmultiply + our
            // gemv reads from slices, but the cuda fallback owns Vec<f32>).
            // Whenever possible, avoid materializing the slow side — at worst
            // one allocation, not two.
            let self_storage = self.storage.as_slice();
            let other_storage = other.storage.as_slice();

            let a_slice: &[T] = if self_fast {
                &self_storage[..m * k1]
            } else {
                // Fall back to materializing — keep the Vec alive for the call.
                // Hoisted into an Option so the borrow lives long enough.
                &[]
            };
            let b_slice: &[T] = if other_fast {
                &other_storage[..k1 * n]
            } else {
                &[]
            };

            // Materialization fallbacks (only when we couldn't use a direct slice).
            let a_owned: Option<Vec<T>> = if self_fast {
                None
            } else {
                Some(self.contiguous().to_vec())
            };
            let b_owned: Option<Vec<T>> = if other_fast {
                None
            } else {
                Some(other.contiguous().to_vec())
            };
            let a: &[T] = a_owned.as_deref().unwrap_or(a_slice);
            let b: &[T] = b_owned.as_deref().unwrap_or(b_slice);

            // GPU-accelerated matmul for CPU tensors: only for very large matrices
            // where transfer overhead is negligible relative to compute.
            // For GPU-resident tensors, the dispatch at the top of matmul() handles it.
            #[cfg(feature = "cuda")]
            {
                let flops = m * n * k1;
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                    && flops >= 4_000_000
                {
                    debug_assert!(std::mem::size_of::<T>() == std::mem::size_of::<f32>());
                    // SAFETY: T is f32 (checked by TypeId above), same size and layout
                    let a_f32: &[f32] = unsafe { std::mem::transmute(a) };
                    let b_f32: &[f32] = unsafe { std::mem::transmute(b) };
                    if let Some(c_f32) = cuda_accel::cuda_matmul(a_f32, b_f32, m, n, k1) {
                        // SAFETY: T is f32, Vec<f32> → Vec<T> is a no-op transmute
                        let c_t: Vec<T> = unsafe {
                            let mut v = std::mem::ManuallyDrop::new(c_f32);
                            Vec::from_raw_parts(v.as_mut_ptr() as *mut T, v.len(), v.capacity())
                        };
                        return Self::from_vec(c_t, &[m, n]);
                    }
                }
            }

            // CpuBackend::matmul overwrites every element via sgemm/gemv
            // with beta=0, so the zero-init memset here is never observed.
            // Previously used `Vec::with_capacity + set_len` to avoid the
            // memset entirely, but clippy::uninit_vec flags that pattern —
            // and for any matmul big enough to matter, the memset cost is
            // dominated by the O(m*n*k) FMA work that follows. For tiny
            // matmuls the memset is ~μs.
            let mut c_data: Vec<T> = vec![T::zero(); m * n];
            CpuBackend::matmul(&mut c_data, a, b, m, n, k1);
            return Self::from_vec(c_data, &[m, n]);
        }

        // For batched matmul, compute batch size
        let batch_dims_self: Vec<usize> = self.shape[..self.ndim() - 2].to_vec();
        let batch_dims_other: Vec<usize> = other.shape[..other.ndim() - 2].to_vec();

        // Broadcast batch dimensions (PyTorch parity)
        let broadcast_batch = if batch_dims_self == batch_dims_other {
            None
        } else {
            // Pad to same length
            let max_len = batch_dims_self.len().max(batch_dims_other.len());
            let pad_a = vec![1usize; max_len - batch_dims_self.len()];
            let pad_b = vec![1usize; max_len - batch_dims_other.len()];
            let a_dims: Vec<usize> = pad_a
                .iter()
                .chain(batch_dims_self.iter())
                .copied()
                .collect();
            let b_dims: Vec<usize> = pad_b
                .iter()
                .chain(batch_dims_other.iter())
                .copied()
                .collect();

            let mut out_dims = Vec::with_capacity(max_len);
            for i in 0..max_len {
                if a_dims[i] == b_dims[i] {
                    out_dims.push(a_dims[i]);
                } else if a_dims[i] == 1 {
                    out_dims.push(b_dims[i]);
                } else if b_dims[i] == 1 {
                    out_dims.push(a_dims[i]);
                } else {
                    return Err(Error::invalid_operation(format!(
                        "matmul batch dimensions not broadcastable: {:?} vs {:?}",
                        batch_dims_self, batch_dims_other
                    )));
                }
            }
            Some((a_dims, b_dims, out_dims))
        };

        let (batch_size, a_batch_idx, b_batch_idx) =
            if let Some((a_dims, b_dims, out_dims)) = &broadcast_batch {
                let bs: usize = out_dims.iter().product();
                // Build index mapping: for each output batch, which a and b batch to use
                let mut a_idx = Vec::with_capacity(bs);
                let mut b_idx = Vec::with_capacity(bs);
                for flat in 0..bs {
                    let mut remaining = flat;
                    let mut ai = 0usize;
                    let mut bi = 0usize;
                    let mut a_stride_acc = 1usize;
                    let mut b_stride_acc = 1usize;
                    for d in (0..out_dims.len()).rev() {
                        let out_d = out_dims[d];
                        let idx = remaining % out_d;
                        remaining /= out_d;
                        let a_d = a_dims[d];
                        let b_d = b_dims[d];
                        ai += (idx % a_d) * a_stride_acc;
                        bi += (idx % b_d) * b_stride_acc;
                        a_stride_acc *= a_d;
                        b_stride_acc *= b_d;
                    }
                    a_idx.push(ai);
                    b_idx.push(bi);
                }
                (bs, a_idx, b_idx)
            } else {
                let bs: usize = batch_dims_self.iter().product();
                let idx: Vec<usize> = (0..bs).collect();
                (bs, idx.clone(), idx)
            };

        let a_stride = m * k1;
        let b_stride = k1 * n;
        let c_stride = m * n;

        let a_data = self.contiguous().to_vec();
        let b_data = other.contiguous().to_vec();
        let mut c_data = vec![T::zero(); batch_size * m * n];

        // Try GPU acceleration for f32 batched matmul (only for large enough matrices)
        #[cfg(feature = "cuda")]
        {
            let flops = m * n * k1;
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && flops >= 4_000_000 {
                let a_f32: &[f32] = unsafe { std::mem::transmute(a_data.as_slice()) };
                let b_f32: &[f32] = unsafe { std::mem::transmute(b_data.as_slice()) };
                let mut gpu_ok = true;
                for batch in 0..batch_size {
                    let ai = a_batch_idx[batch];
                    let bi = b_batch_idx[batch];
                    let a_slice = &a_f32[ai * a_stride..(ai + 1) * a_stride];
                    let b_slice = &b_f32[bi * b_stride..(bi + 1) * b_stride];
                    if let Some(c_batch) = cuda_accel::cuda_matmul(a_slice, b_slice, m, n, k1) {
                        c_data[batch * c_stride..(batch + 1) * c_stride]
                            .copy_from_slice(unsafe { std::mem::transmute(c_batch.as_slice()) });
                    } else {
                        gpu_ok = false;
                        break;
                    }
                }
                if gpu_ok {
                    let mut output_shape = batch_dims_self;
                    output_shape.push(m);
                    output_shape.push(n);
                    return Self::from_vec(c_data, &output_shape);
                }
                // Fall through to CPU if GPU failed
                c_data = vec![T::zero(); batch_size * m * n];
            }
        }

        // CPU fallback: parallel over batches when there is real work
        // (common for attention heads, batched prefill, training micro-batches).
        use rayon::prelude::*;
        if batch_size > 1 && (batch_size * m * n) >= 4096 {
            // SAFETY: each batch writes to a disjoint [batch*c_stride .. (batch+1)*c_stride)
            // region of c_data; a/b slices are read-only and already materialized.
            // Capture idx vecs via usize-ptr so the closure is Send+Sync (consistent
            // with other parallel paths in tensor + autograd).
            let a_idx_ptr = a_batch_idx.as_ptr() as usize;
            let b_idx_ptr = b_batch_idx.as_ptr() as usize;
            let c_ptr = c_data.as_mut_ptr() as usize;
            let a_data_ptr = a_data.as_ptr() as usize;
            let b_data_ptr = b_data.as_ptr() as usize;
            (0..batch_size).into_par_iter().for_each(|batch| {
                let ai = unsafe { *(a_idx_ptr as *const usize).add(batch) };
                let bi = unsafe { *(b_idx_ptr as *const usize).add(batch) };
                let a_slice = unsafe {
                    let base = (a_data_ptr as *const T).add(ai * a_stride);
                    std::slice::from_raw_parts(base, a_stride)
                };
                let b_slice = unsafe {
                    let base = (b_data_ptr as *const T).add(bi * b_stride);
                    std::slice::from_raw_parts(base, b_stride)
                };
                let c_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        (c_ptr as *mut T).add(batch * c_stride),
                        c_stride,
                    )
                };
                CpuBackend::matmul(c_slice, a_slice, b_slice, m, n, k1);
            });
        } else {
            for batch in 0..batch_size {
                let ai = a_batch_idx[batch];
                let bi = b_batch_idx[batch];
                let a_slice = &a_data[ai * a_stride..(ai + 1) * a_stride];
                let b_slice = &b_data[bi * b_stride..(bi + 1) * b_stride];
                let c_slice = &mut c_data[batch * c_stride..(batch + 1) * c_stride];
                CpuBackend::matmul(c_slice, a_slice, b_slice, m, n, k1);
            }
        }

        // Build output shape: broadcast batch dims + [m, n]
        let mut output_shape = if let Some((_, _, ref out_dims)) = broadcast_batch {
            out_dims.clone()
        } else {
            batch_dims_self
        };
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
// f32 ↔ f16 Casting for AMP (Automatic Mixed Precision)
// =============================================================================

impl Tensor<f32> {
    /// Cast this f32 tensor to f16 values stored as f32.
    ///
    /// Each value is rounded to f16 precision. This simulates half-precision
    /// computation while keeping the tensor type as f32, which is how AMP
    /// works — the autograd graph stays f32 but computation uses f16 precision.
    ///
    /// On GPU, this uses a CUDA kernel for fast conversion.
    /// On CPU, this uses the `half` crate.
    #[must_use]
    pub fn to_f16_precision(&self) -> Self {
        let data = self.to_vec();
        let f16_data: Vec<f32> = data
            .iter()
            .map(|&v| {
                let h = half::f16::from_f32(v);
                h.to_f32()
            })
            .collect();
        Self::from_vec(f16_data, self.shape()).unwrap()
    }

    /// Cast f16-precision values back to full f32 precision.
    ///
    /// This is a no-op since the data is already stored as f32.
    /// Included for API symmetry with `to_f16_precision()`.
    #[must_use]
    pub fn to_f32_precision(&self) -> Self {
        self.clone()
    }

    /// Returns true if applying f16 precision would change any values.
    /// Useful for debugging AMP-related numerical issues.
    #[must_use]
    pub fn has_f16_rounding_error(&self) -> bool {
        let data = self.to_vec();
        data.iter().any(|&v| {
            let h = half::f16::from_f32(v);
            (h.to_f32() - v).abs() > f32::EPSILON
        })
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
        let r = t.reshape(&[3, 2]).expect("reshape failed");
        assert_eq!(r.shape(), &[3, 2]);

        let r = t.reshape(&[-1]).expect("reshape failed");
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

    // Regression test for L15 footgun hardening (deficiency #5):
    // Tensor<u32>.to_device(GPU) must return a clear Error (was assert panic).
    #[test]
    #[cfg(feature = "cuda")]
    fn test_u32_to_gpu_is_error_not_panic() {
        let ids: Tensor<u32> = Tensor::from_vec(vec![1u32, 2, 3], &[3]).unwrap();
        let dev = Device::Cuda(0);
        let res = ids.to_device(dev);
        assert!(res.is_err());
        let msg = res.unwrap_err().to_string();
        assert!(
            msg.contains("GPU tensors are f32-only") || msg.contains("not supported"),
            "expected helpful error, got: {}",
            msg
        );
    }
}
