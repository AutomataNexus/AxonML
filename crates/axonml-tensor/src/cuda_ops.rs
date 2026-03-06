//! GPU Tensor Operations
//!
//! Implements tensor operations on GPU-resident f32 data using CUDA kernels.
//! Each method operates directly on CudaSlice data without CPU copies.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

#[cfg(feature = "cuda")]
use axonml_core::backends::cuda::get_cuda_backend;
#[cfg(feature = "cuda")]
use axonml_core::backends::cuda_pool::pool_alloc;
#[cfg(feature = "cuda")]
use axonml_core::error::Result;
#[cfg(feature = "cuda")]
use axonml_core::storage::Storage;
#[cfg(feature = "cuda")]
use axonml_core::Device;

#[cfg(feature = "cuda")]
use crate::shape::{contiguous_strides, Shape};
#[cfg(feature = "cuda")]
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
impl Tensor<f32> {
    // =========================================================================
    // Element-wise Binary Operations (GPU)
    // =========================================================================

    /// GPU element-wise addition. Both tensors must be contiguous, same shape, same device.
    pub(crate) fn add_cuda(&self, other: &Self) -> Result<Self> {
        let a_data = self.contiguous_gpu();
        let b_data = other.contiguous_gpu();
        let len = a_data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let a_guard = a_data.storage.as_cuda_slice();
        let b_guard = b_data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.add_f32(&mut out, a_guard.slice(), b_guard.slice(), len)
            .expect("CUDA add_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        })
    }

    /// GPU element-wise subtraction.
    pub(crate) fn sub_cuda(&self, other: &Self) -> Result<Self> {
        let a_data = self.contiguous_gpu();
        let b_data = other.contiguous_gpu();
        let len = a_data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let a_guard = a_data.storage.as_cuda_slice();
        let b_guard = b_data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.sub_f32(&mut out, a_guard.slice(), b_guard.slice(), len)
            .expect("CUDA sub_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        })
    }

    /// GPU element-wise multiplication.
    pub(crate) fn mul_cuda(&self, other: &Self) -> Result<Self> {
        let a_data = self.contiguous_gpu();
        let b_data = other.contiguous_gpu();
        let len = a_data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let a_guard = a_data.storage.as_cuda_slice();
        let b_guard = b_data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.mul_f32(&mut out, a_guard.slice(), b_guard.slice(), len)
            .expect("CUDA mul_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        })
    }

    /// GPU element-wise division.
    pub(crate) fn div_cuda(&self, other: &Self) -> Result<Self> {
        let a_data = self.contiguous_gpu();
        let b_data = other.contiguous_gpu();
        let len = a_data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let a_guard = a_data.storage.as_cuda_slice();
        let b_guard = b_data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.div_f32(&mut out, a_guard.slice(), b_guard.slice(), len)
            .expect("CUDA div_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        })
    }

    // =========================================================================
    // Broadcast Binary Operations (GPU)
    // =========================================================================

    /// GPU broadcast addition. Handles different shapes via modular indexing.
    /// Both tensors must be on GPU and contiguous. The smaller tensor is broadcast.
    ///
    /// Supports: [M,N] + [N], [B,M,N] + [N], [B,M,N] + [M,N], [M,N] + [M,1], etc.
    /// Requirement: larger_numel % smaller_numel == 0 (standard broadcasting).
    pub(crate) fn broadcast_add_cuda(&self, other: &Self) -> Result<Self> {
        let a = self.contiguous_gpu();
        let b = other.contiguous_gpu();
        let a_n = a.numel();
        let b_n = b.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let result_shape = crate::shape::broadcast_shape(&self.shape, &other.shape)?;
        let out_n = crate::shape::numel(&result_shape);
        let mut out = pool_alloc(out_n).expect("GPU pool alloc failed");

        let a_guard = a.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();

        if a_n >= b_n {
            // b is smaller, broadcast b: out[i] = a[i] + b[i % b_n]
            if a_n == out_n {
                cuda.broadcast_add_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_add_f32 failed");
            } else {
                // Both need broadcasting — materialize a first
                let a_bcast = a.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let a2_guard = a_bcast.storage.as_cuda_slice();
                cuda.broadcast_add_f32(&mut out, a2_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_add_f32 failed");
            }
        } else {
            // a is smaller, broadcast a: out[i] = a[i % a_n] + b[i]
            if b_n == out_n {
                cuda.broadcast_add_rev_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, a_n)
                    .expect("CUDA broadcast_add_rev_f32 failed");
            } else {
                let b_bcast = b.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let b2_guard = b_bcast.storage.as_cuda_slice();
                cuda.broadcast_add_rev_f32(&mut out, a_guard.slice(), b2_guard.slice(), out_n, a_n)
                    .expect("CUDA broadcast_add_rev_f32 failed");
            }
        }

        let storage = Storage::from_cuda_slice(out, out_n, self.device());
        Ok(Self {
            storage,
            shape: result_shape,
            strides: contiguous_strides(&crate::shape::broadcast_shape(&self.shape, &other.shape)?),
            offset: 0,
        })
    }

    /// GPU broadcast subtraction.
    pub(crate) fn broadcast_sub_cuda(&self, other: &Self) -> Result<Self> {
        let a = self.contiguous_gpu();
        let b = other.contiguous_gpu();
        let a_n = a.numel();
        let b_n = b.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let result_shape = crate::shape::broadcast_shape(&self.shape, &other.shape)?;
        let out_n = crate::shape::numel(&result_shape);
        let mut out = pool_alloc(out_n).expect("GPU pool alloc failed");

        let a_guard = a.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();

        if a_n >= b_n {
            if a_n == out_n {
                cuda.broadcast_sub_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_sub_f32 failed");
            } else {
                let a_bcast = a.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let a2_guard = a_bcast.storage.as_cuda_slice();
                cuda.broadcast_sub_f32(&mut out, a2_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_sub_f32 failed");
            }
        } else {
            if b_n == out_n {
                cuda.broadcast_sub_rev_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, a_n)
                    .expect("CUDA broadcast_sub_rev_f32 failed");
            } else {
                let b_bcast = b.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let b2_guard = b_bcast.storage.as_cuda_slice();
                cuda.broadcast_sub_rev_f32(&mut out, a_guard.slice(), b2_guard.slice(), out_n, a_n)
                    .expect("CUDA broadcast_sub_rev_f32 failed");
            }
        }

        let storage = Storage::from_cuda_slice(out, out_n, self.device());
        Ok(Self {
            storage,
            shape: result_shape,
            strides: contiguous_strides(&crate::shape::broadcast_shape(&self.shape, &other.shape)?),
            offset: 0,
        })
    }

    /// GPU broadcast multiplication.
    pub(crate) fn broadcast_mul_cuda(&self, other: &Self) -> Result<Self> {
        let a = self.contiguous_gpu();
        let b = other.contiguous_gpu();
        let a_n = a.numel();
        let b_n = b.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let result_shape = crate::shape::broadcast_shape(&self.shape, &other.shape)?;
        let out_n = crate::shape::numel(&result_shape);
        let mut out = pool_alloc(out_n).expect("GPU pool alloc failed");

        let a_guard = a.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();

        if a_n >= b_n {
            if a_n == out_n {
                cuda.broadcast_mul_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_mul_f32 failed");
            } else {
                let a_bcast = a.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let a2_guard = a_bcast.storage.as_cuda_slice();
                cuda.broadcast_mul_f32(&mut out, a2_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_mul_f32 failed");
            }
        } else {
            if b_n == out_n {
                cuda.broadcast_mul_rev_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, a_n)
                    .expect("CUDA broadcast_mul_rev_f32 failed");
            } else {
                let b_bcast = b.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let b2_guard = b_bcast.storage.as_cuda_slice();
                cuda.broadcast_mul_rev_f32(&mut out, a_guard.slice(), b2_guard.slice(), out_n, a_n)
                    .expect("CUDA broadcast_mul_rev_f32 failed");
            }
        }

        let storage = Storage::from_cuda_slice(out, out_n, self.device());
        Ok(Self {
            storage,
            shape: result_shape,
            strides: contiguous_strides(&crate::shape::broadcast_shape(&self.shape, &other.shape)?),
            offset: 0,
        })
    }

    /// GPU broadcast division.
    pub(crate) fn broadcast_div_cuda(&self, other: &Self) -> Result<Self> {
        let a = self.contiguous_gpu();
        let b = other.contiguous_gpu();
        let a_n = a.numel();
        let b_n = b.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let result_shape = crate::shape::broadcast_shape(&self.shape, &other.shape)?;
        let out_n = crate::shape::numel(&result_shape);
        let mut out = pool_alloc(out_n).expect("GPU pool alloc failed");

        let a_guard = a.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();

        if a_n >= b_n {
            if a_n == out_n {
                cuda.broadcast_div_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_div_f32 failed");
            } else {
                let a_bcast = a.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let a2_guard = a_bcast.storage.as_cuda_slice();
                cuda.broadcast_div_f32(&mut out, a2_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_div_f32 failed");
            }
        } else {
            if b_n == out_n {
                cuda.broadcast_div_rev_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, a_n)
                    .expect("CUDA broadcast_div_rev_f32 failed");
            } else {
                let b_bcast = b.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let b2_guard = b_bcast.storage.as_cuda_slice();
                cuda.broadcast_div_rev_f32(&mut out, a_guard.slice(), b2_guard.slice(), out_n, a_n)
                    .expect("CUDA broadcast_div_rev_f32 failed");
            }
        }

        let storage = Storage::from_cuda_slice(out, out_n, self.device());
        Ok(Self {
            storage,
            shape: result_shape,
            strides: contiguous_strides(&crate::shape::broadcast_shape(&self.shape, &other.shape)?),
            offset: 0,
        })
    }

    // =========================================================================
    // Element-wise Unary Operations (GPU)
    // =========================================================================

    /// GPU negation.
    pub(crate) fn neg_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.neg_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA neg_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU ReLU activation.
    pub(crate) fn relu_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.relu_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA relu_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU sigmoid activation.
    pub(crate) fn sigmoid_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.sigmoid_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA sigmoid_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU tanh activation.
    pub(crate) fn tanh_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.tanh_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA tanh_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU exp.
    pub(crate) fn exp_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.exp_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA exp_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU natural log.
    pub(crate) fn ln_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.log_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA log_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU sqrt.
    pub(crate) fn sqrt_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.sqrt_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA sqrt_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU power with scalar exponent.
    pub(crate) fn pow_cuda(&self, exp: f32) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.pow_scalar_f32(&mut out, src_guard.slice(), exp, len)
            .expect("CUDA pow_scalar_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU GELU activation.
    pub(crate) fn gelu_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.gelu_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA gelu_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU SiLU activation.
    pub(crate) fn silu_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.silu_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA silu_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU scalar multiplication — fully on-device, no CPU round-trip.
    pub(crate) fn mul_scalar_cuda(&self, scalar: f32) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        // Copy src → out on device, then scale out in-place
        cuda.broadcast_copy_f32(&mut out, src_guard.slice(), len, len)
            .expect("CUDA broadcast_copy_f32 failed");
        cuda.scale_f32(&mut out, scalar, len)
            .expect("CUDA scale_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU scalar addition — fully on-device.
    pub(crate) fn add_scalar_cuda(&self, scalar: f32) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.add_scalar_f32(&mut out, src_guard.slice(), scalar, len)
            .expect("CUDA add_scalar_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU softmax along last dimension — fully on-device.
    pub(crate) fn softmax_cuda(&self, dim: i32) -> Result<Self> {
        let data = self.contiguous_gpu();
        let ndim = data.shape.len();
        let total = data.numel();

        // Normalize dim
        let d = if dim < 0 { ndim as i32 + dim } else { dim } as usize;

        // For softmax along last dim (most common), use the row kernel directly
        if d == ndim - 1 {
            let row_size = data.shape[ndim - 1];
            let num_rows = total / row_size;
            let cuda = get_cuda_backend().expect("CUDA backend not available");

            let src_guard = data.storage.as_cuda_slice();
            let mut out = pool_alloc(total).expect("GPU pool alloc failed");

            // Copy data to output (softmax kernel is in-place)
            cuda.broadcast_copy_f32(&mut out, src_guard.slice(), total, total)
                .expect("CUDA broadcast_copy_f32 failed");

            cuda.softmax_row_f32(&mut out, num_rows, row_size)
                .expect("CUDA softmax_row_f32 failed");

            let storage = Storage::from_cuda_slice(out, total, self.device());
            Ok(Self {
                storage,
                shape: data.shape.clone(),
                strides: contiguous_strides(&data.shape),
                offset: 0,
            })
        } else {
            // For non-last dim softmax: transpose so target dim is last,
            // apply softmax, transpose back
            let mut perm: Vec<usize> = (0..ndim).collect();
            perm.swap(d, ndim - 1);
            let transposed = data.permute(&perm)?;
            let t_contig = transposed.contiguous_gpu();
            let t_result = t_contig.softmax_cuda(ndim as i32 - 1)?;
            // Inverse permutation is the same swap
            Ok(t_result.permute(&perm)?.contiguous_gpu())
        }
    }

    /// GPU broadcast_to — fully on-device using broadcast_copy kernel.
    pub(crate) fn broadcast_to_cuda(&self, target_shape: &[usize]) -> Result<Self> {
        let data = self.contiguous_gpu();
        let src_len = data.numel();
        let out_len = crate::shape::numel(target_shape);
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        // Simple case: target is an exact multiple of source (trailing dims match)
        // This covers [N] → [M,N], [N] → [B,M,N], [M,N] → [B,M,N], etc.
        if out_len % src_len == 0 {
            let src_guard = data.storage.as_cuda_slice();
            let mut out = pool_alloc(out_len).expect("GPU pool alloc failed");

            cuda.broadcast_copy_f32(&mut out, src_guard.slice(), out_len, src_len)
                .expect("CUDA broadcast_copy_f32 failed");

            let storage = Storage::from_cuda_slice(out, out_len, self.device());
            return Ok(Self {
                storage,
                shape: crate::shape::Shape::from_slice(target_shape),
                strides: contiguous_strides(&crate::shape::Shape::from_slice(target_shape)),
                offset: 0,
            });
        }

        // General case (e.g., [M,1] → [M,N]): compute gather indices on CPU,
        // upload to GPU, then gather on device
        let result_shape: crate::shape::Shape = target_shape.into();
        let src_strides = crate::shape::broadcast_strides(&data.shape, &data.strides, &result_shape);

        let indices: Vec<u32> = (0..out_len)
            .map(|i| {
                let coords = crate::shape::unravel_index(i, &result_shape);
                let src_idx = data.offset + crate::shape::linear_index(&coords, &src_strides);
                src_idx as u32
            })
            .collect();

        let idx_gpu = cuda.htod_copy(&indices).expect("htod indices failed");
        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(out_len).expect("GPU pool alloc failed");

        cuda.gather_contiguous_f32(&mut out, src_guard.slice(), &idx_gpu, out_len)
            .expect("CUDA gather_contiguous_f32 failed");

        let storage = Storage::from_cuda_slice(out, out_len, self.device());
        Ok(Self {
            storage,
            shape: result_shape,
            strides: contiguous_strides(&crate::shape::Shape::from_slice(target_shape)),
            offset: 0,
        })
    }

    // =========================================================================
    // Matrix Multiplication (GPU) — the critical speedup
    // =========================================================================

    /// GPU matrix multiplication using cuBLAS GEMM — no CPU copies.
    pub(crate) fn matmul_cuda(&self, other: &Self) -> Result<Self> {
        let a = self.contiguous_gpu();
        let b = other.contiguous_gpu();

        let m = a.shape[a.shape.len() - 2];
        let k = a.shape[a.shape.len() - 1];
        let n = b.shape[b.shape.len() - 1];

        let cuda = get_cuda_backend().expect("CUDA backend not available");

        if a.shape.len() == 2 && b.shape.len() == 2 {
            // 2D matmul: C(m,n) = A(m,k) @ B(k,n)
            // cuBLAS is column-major, so: C^T(n,m) = B^T(n,k) @ A^T(k,m)
            let a_guard = a.storage.as_cuda_slice();
            let b_guard = b.storage.as_cuda_slice();
            let mut c_gpu = pool_alloc(m * n).expect("GPU pool alloc failed");

            cuda.gemm_f32(
                false, false,
                n, m, k,
                1.0,
                b_guard.slice(), n,
                a_guard.slice(), k,
                0.0,
                &mut c_gpu, n,
            ).expect("cuBLAS gemm failed");

            let storage = Storage::from_cuda_slice(c_gpu, m * n, self.device());
            return Ok(Self {
                storage,
                shape: Shape::from_slice(&[m, n]),
                strides: contiguous_strides(&Shape::from_slice(&[m, n])),
                offset: 0,
            });
        }

        // Batched matmul: fully on-device using cublasSgemmStridedBatched
        let batch_dims: Vec<usize> = a.shape[..a.shape.len() - 2].to_vec();
        let batch_size: usize = batch_dims.iter().product();
        let a_stride = (m * k) as i64;
        let b_stride = (k * n) as i64;
        let c_stride = (m * n) as i64;
        let total = batch_size * m * n;

        let a_guard = a.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();
        let mut c_gpu = pool_alloc(total).expect("GPU pool alloc failed");

        // cuBLAS is column-major, so we compute B^T @ A^T = (A @ B)^T
        // with the row-major data already laid out as transposed column-major
        cuda.gemm_strided_batched_f32(
            false, false,
            n, m, k,
            1.0,
            b_guard.slice(), n, b_stride,
            a_guard.slice(), k, a_stride,
            0.0,
            &mut c_gpu, n, c_stride,
            batch_size,
        ).expect("cuBLAS strided batched gemm failed");

        let mut output_shape = batch_dims;
        output_shape.push(m);
        output_shape.push(n);

        let storage = Storage::from_cuda_slice(c_gpu, total, self.device());
        Ok(Self {
            storage,
            shape: Shape::from_slice(&output_shape),
            strides: contiguous_strides(&Shape::from_slice(&output_shape)),
            offset: 0,
        })
    }

    // =========================================================================
    // GPU Data Access Helpers
    // =========================================================================

    /// Returns data as Vec<f32>, handling GPU D2H copy.
    pub(crate) fn to_vec_gpu(&self) -> Vec<f32> {
        self.storage.to_vec_f32()
    }

    /// Returns a contiguous GPU tensor — fully on-device using gather kernel.
    pub(crate) fn contiguous_gpu(&self) -> Self {
        if self.is_contiguous() && self.offset == 0 {
            return self.clone();
        }
        let total = self.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let ndim = self.shape.len();
        let offset = self.offset;

        // Compute gather indices on CPU using direct stride arithmetic
        // (no per-element Vec allocation — fixed-size stack array instead)
        let shape = self.shape.as_slice();
        let strides = self.strides.as_slice();

        // Precompute contiguous strides for output indexing
        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * shape[i + 1];
        }

        let mut indices = vec![0u32; total];

        // Use iterative coordinate tracking instead of unravel_index per element
        let mut coords = vec![0usize; ndim];
        for i in 0..total {
            // Compute source linear index from coords and strides
            let mut src_idx = offset as isize;
            for d in 0..ndim {
                src_idx += coords[d] as isize * strides[d];
            }
            indices[i] = src_idx as u32;

            // Increment coordinates (like an odometer, rightmost digit first)
            for d in (0..ndim).rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        let idx_gpu = cuda.htod_copy(&indices).expect("htod indices failed");
        let src_guard = self.storage.as_cuda_slice();
        let mut out = pool_alloc(total).expect("GPU pool alloc failed");

        cuda.gather_contiguous_f32(&mut out, src_guard.slice(), &idx_gpu, total)
            .expect("CUDA gather_contiguous_f32 failed");

        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Transfers tensor to device, using the f32-specific path.
    pub fn to_device_f32(&self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }

        let contig = if self.storage.is_gpu() {
            self.contiguous_gpu()
        } else {
            self.contiguous()
        };

        let new_storage = contig.storage.to_device_f32(device)?;

        Ok(Self {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        })
    }

    /// GPU LayerNorm: per-row normalization with affine transform.
    ///
    /// Input, gamma, beta must all be on GPU. Runs entirely on device.
    /// Returns output tensor with same shape as input.
    pub fn layer_norm_cuda(
        &self,
        gamma: &Self,
        beta: &Self,
        norm_size: usize,
        eps: f32,
    ) -> Result<Self> {
        let input_data = self.contiguous_gpu();
        let total_len = input_data.numel();
        let num_rows = total_len / norm_size;
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let input_guard = input_data.storage.as_cuda_slice();
        let gamma_guard = gamma.storage.as_cuda_slice();
        let beta_guard = beta.storage.as_cuda_slice();
        let mut out = pool_alloc(total_len).expect("GPU pool alloc failed for LayerNorm");

        cuda.layer_norm_f32(
            &mut out,
            input_guard.slice(),
            gamma_guard.slice(),
            beta_guard.slice(),
            norm_size,
            eps,
            num_rows,
        )
        .expect("CUDA layer_norm_f32 failed");

        let storage = Storage::from_cuda_slice(out, total_len, self.device());
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        })
    }

    /// GPU embedding gather: gathers rows from weight using flat gather indices.
    ///
    /// `gather_indices` is a flat u32 array of length `output_size` where each element
    /// is the index into the flat weight array to read from.
    /// Weight must be on GPU. Output is a new GPU tensor with the given shape.
    pub fn embedding_gather_cuda(
        &self,
        gather_indices: &[u32],
        output_shape: &[usize],
    ) -> Self {
        let output_size = output_shape.iter().product::<usize>();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let idx_gpu = cuda.htod_copy(gather_indices).expect("htod gather indices failed");
        let weight_guard = self.storage.as_cuda_slice();
        let mut out = pool_alloc(output_size).expect("GPU pool alloc failed");

        cuda.gather_contiguous_f32(&mut out, weight_guard.slice(), &idx_gpu, output_size)
            .expect("CUDA gather_contiguous_f32 failed");

        let storage = Storage::from_cuda_slice(out, output_size, self.device());
        Self {
            storage,
            shape: crate::shape::Shape::from_slice(output_shape),
            strides: contiguous_strides(&crate::shape::Shape::from_slice(output_shape)),
            offset: 0,
        }
    }

    // =========================================================================
    // Backward Activation Kernels (GPU)
    // =========================================================================

    /// GPU sum along a dimension. Fully on-device, no CPU copies.
    pub(crate) fn sum_dim_cuda(&self, dim: usize) -> Self {
        let data = self.contiguous_gpu();
        let ndim = data.shape.len();

        let outer_size: usize = data.shape[..dim].iter().product();
        let dim_size = data.shape[dim];
        let inner_size: usize = data.shape[dim + 1..].iter().product();
        let out_len = outer_size * inner_size;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(out_len).expect("GPU pool alloc failed");

        cuda.sum_dim_f32(&mut out, src_guard.slice(), outer_size, dim_size, inner_size)
            .expect("CUDA sum_dim_f32 failed");

        // Build output shape (dim removed)
        let mut out_shape: Vec<usize> = Vec::with_capacity(ndim - 1);
        for (i, &s) in data.shape.iter().enumerate() {
            if i != dim {
                out_shape.push(s);
            }
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let shape = Shape::from_slice(&out_shape);
        let storage = Storage::from_cuda_slice(out, out_len, self.device());
        Self {
            storage,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        }
    }

    /// GPU sum along a dimension with keepdim=true. Fully on-device.
    pub(crate) fn sum_dim_keepdim_cuda(&self, dim: usize) -> Self {
        let data = self.contiguous_gpu();

        let outer_size: usize = data.shape[..dim].iter().product();
        let dim_size = data.shape[dim];
        let inner_size: usize = data.shape[dim + 1..].iter().product();
        let out_len = outer_size * inner_size;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(out_len).expect("GPU pool alloc failed");

        cuda.sum_dim_f32(&mut out, src_guard.slice(), outer_size, dim_size, inner_size)
            .expect("CUDA sum_dim_f32 failed");

        // Build output shape with dim=1 at the reduced position
        let mut out_shape: Vec<usize> = data.shape.to_vec();
        out_shape[dim] = 1;
        let shape = Shape::from_slice(&out_shape);
        let storage = Storage::from_cuda_slice(out, out_len, self.device());
        Self {
            storage,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        }
    }

    /// GPU ReLU backward: grad_output * (input > 0).
    pub fn relu_backward_cuda(&self, input: &Self) -> Self {
        let grad = self.contiguous_gpu();
        let inp = input.contiguous_gpu();
        let len = grad.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let grad_guard = grad.storage.as_cuda_slice();
        let inp_guard = inp.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.relu_backward_f32(&mut out, grad_guard.slice(), inp_guard.slice(), len)
            .expect("CUDA relu_backward_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU Sigmoid backward: grad_output * output * (1 - output).
    pub fn sigmoid_backward_cuda(&self, output: &Self) -> Self {
        let grad = self.contiguous_gpu();
        let out_data = output.contiguous_gpu();
        let len = grad.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let grad_guard = grad.storage.as_cuda_slice();
        let out_guard = out_data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.sigmoid_backward_f32(&mut out, grad_guard.slice(), out_guard.slice(), len)
            .expect("CUDA sigmoid_backward_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU Softmax backward: result[i] = softmax[i] * (grad[i] - dot(softmax, grad)) per row.
    /// Both self (grad_output) and softmax_output must be GPU-resident.
    pub fn softmax_backward_cuda(&self, softmax_output: &Self) -> Self {
        let grad = self.contiguous_gpu();
        let sout = softmax_output.contiguous_gpu();
        let total = grad.numel();
        let ndim = grad.shape.len();
        let row_size = grad.shape[ndim - 1];
        let num_rows = total / row_size;
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let grad_guard = grad.storage.as_cuda_slice();
        let sout_guard = sout.storage.as_cuda_slice();
        let mut out = pool_alloc(total).expect("GPU pool alloc failed");

        cuda.softmax_backward_row_f32(
            &mut out,
            sout_guard.slice(),
            grad_guard.slice(),
            num_rows,
            row_size,
        )
        .expect("CUDA softmax_backward_row_f32 failed");

        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU LayerNorm backward: compute d_input.
    /// self = grad_output, input = forward input, gamma = weight.
    pub fn layer_norm_backward_dinput_cuda(
        &self,
        input: &Self,
        gamma: &Self,
        norm_size: usize,
        eps: f32,
    ) -> Self {
        let grad = self.contiguous_gpu();
        let inp = input.contiguous_gpu();
        let total = grad.numel();
        let num_rows = total / norm_size;
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let grad_guard = grad.storage.as_cuda_slice();
        let inp_guard = inp.storage.as_cuda_slice();
        let gamma_guard = gamma.storage.as_cuda_slice();
        let mut out = pool_alloc(total).expect("GPU pool alloc failed");

        cuda.layer_norm_backward_dinput_f32(
            &mut out,
            grad_guard.slice(),
            inp_guard.slice(),
            gamma_guard.slice(),
            norm_size,
            eps,
            num_rows,
        )
        .expect("CUDA layer_norm_backward_dinput_f32 failed");

        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU LayerNorm backward: compute d_weight and d_bias.
    /// self = grad_output, input = forward input.
    /// Returns (d_weight, d_bias) both of shape [norm_size].
    pub fn layer_norm_backward_dweight_dbias_cuda(
        &self,
        input: &Self,
        norm_size: usize,
        eps: f32,
    ) -> (Self, Self) {
        let grad = self.contiguous_gpu();
        let inp = input.contiguous_gpu();
        let total = grad.numel();
        let num_rows = total / norm_size;
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let grad_guard = grad.storage.as_cuda_slice();
        let inp_guard = inp.storage.as_cuda_slice();
        let mut d_weight = pool_alloc(norm_size).expect("GPU pool alloc failed");
        let mut d_bias = pool_alloc(norm_size).expect("GPU pool alloc failed");

        cuda.layer_norm_backward_dweight_dbias_f32(
            &mut d_weight,
            &mut d_bias,
            grad_guard.slice(),
            inp_guard.slice(),
            norm_size,
            eps,
            num_rows,
        )
        .expect("CUDA layer_norm_backward_dweight_dbias_f32 failed");

        let w_shape = Shape::from_slice(&[norm_size]);
        let dw = Self {
            storage: Storage::from_cuda_slice(d_weight, norm_size, self.device()),
            shape: w_shape.clone(),
            strides: contiguous_strides(&w_shape),
            offset: 0,
        };
        let db = Self {
            storage: Storage::from_cuda_slice(d_bias, norm_size, self.device()),
            shape: w_shape.clone(),
            strides: contiguous_strides(&w_shape),
            offset: 0,
        };
        (dw, db)
    }

    /// GPU Tanh backward: grad_output * (1 - output^2).
    pub fn tanh_backward_cuda(&self, output: &Self) -> Self {
        let grad = self.contiguous_gpu();
        let out_data = output.contiguous_gpu();
        let len = grad.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let grad_guard = grad.storage.as_cuda_slice();
        let out_guard = out_data.storage.as_cuda_slice();
        let mut out = pool_alloc(len).expect("GPU pool alloc failed");

        cuda.tanh_backward_f32(&mut out, grad_guard.slice(), out_guard.slice(), len)
            .expect("CUDA tanh_backward_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU CrossEntropy forward: fused softmax + NLL loss.
    /// self = logits [N, C], targets = class indices as f32 [N].
    /// Returns (losses [N], softmax_probs [N, C]).
    pub fn cross_entropy_fwd_cuda(&self, targets: &Self) -> (Self, Self) {
        let logits = self.contiguous_gpu();
        let tgt = targets.contiguous_gpu();
        let batch_size = logits.shape[0];
        let num_classes = logits.shape[1];

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let logits_guard = logits.storage.as_cuda_slice();
        let tgt_guard = tgt.storage.as_cuda_slice();

        let mut losses_gpu = pool_alloc(batch_size).expect("GPU pool alloc");
        let mut softmax_gpu = pool_alloc(batch_size * num_classes).expect("GPU pool alloc");

        cuda.cross_entropy_fwd_f32(
            logits_guard.slice(),
            tgt_guard.slice(),
            &mut losses_gpu,
            &mut softmax_gpu,
            batch_size,
            num_classes,
        )
        .expect("CUDA cross_entropy_fwd_f32 failed");

        let loss_shape = Shape::from_slice(&[batch_size]);
        let losses = Self {
            storage: Storage::from_cuda_slice(losses_gpu, batch_size, self.device()),
            shape: loss_shape.clone(),
            strides: contiguous_strides(&loss_shape),
            offset: 0,
        };

        let sm_shape = Shape::from_slice(&[batch_size, num_classes]);
        let softmax = Self {
            storage: Storage::from_cuda_slice(softmax_gpu, batch_size * num_classes, self.device()),
            shape: sm_shape.clone(),
            strides: contiguous_strides(&sm_shape),
            offset: 0,
        };

        (losses, softmax)
    }

    /// GPU CrossEntropy backward: grad = (softmax - one_hot(target)) * grad_output.
    /// self = softmax_probs [N, C], targets = class indices as f32 [N],
    /// grad_output = upstream gradient [N].
    /// Returns grad_input [N, C].
    pub fn cross_entropy_bwd_cuda(&self, targets: &Self, grad_output: &Self) -> Self {
        let softmax = self.contiguous_gpu();
        let tgt = targets.contiguous_gpu();
        let grad_out = grad_output.contiguous_gpu();
        let batch_size = softmax.shape[0];
        let num_classes = softmax.shape[1];
        let total = batch_size * num_classes;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let sm_guard = softmax.storage.as_cuda_slice();
        let tgt_guard = tgt.storage.as_cuda_slice();
        let grad_guard = grad_out.storage.as_cuda_slice();
        let mut grad_input = pool_alloc(total).expect("GPU pool alloc");

        cuda.cross_entropy_bwd_f32(
            sm_guard.slice(),
            tgt_guard.slice(),
            grad_guard.slice(),
            &mut grad_input,
            batch_size,
            num_classes,
        )
        .expect("CUDA cross_entropy_bwd_f32 failed");

        let out_shape = Shape::from_slice(&[batch_size, num_classes]);
        Self {
            storage: Storage::from_cuda_slice(grad_input, total, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        }
    }

    /// GPU implementation of NarrowBackward: scatters `self` (the gradient of
    /// a narrow/slice) into a zero tensor of `input_shape` at the correct offset
    /// along `dim` starting at `start`. All operations stay on GPU.
    pub fn narrow_backward_cuda(
        &self,
        input_shape: &[usize],
        dim: usize,
        start: usize,
    ) -> Self {
        let numel: usize = input_shape.iter().product();
        let cuda = get_cuda_backend().expect("CUDA backend");

        // Allocate zero-initialized output on GPU
        let mut dst = pool_alloc(numel).expect("GPU pool alloc for narrow_backward");
        cuda.memset_zeros_f32(&mut dst)
            .expect("CUDA memset_zeros failed");

        // Ensure gradient is contiguous
        let grad_contig = self.contiguous_gpu();
        let src_guard = grad_contig.storage.as_cuda_slice();

        // Compute strided copy parameters
        let inner_size: usize = input_shape[dim + 1..].iter().product::<usize>().max(1);
        let offset_elements = start * inner_size;
        let outer_size: usize = input_shape[..dim].iter().product::<usize>().max(1);
        let dim_full = input_shape[dim];
        let dim_narrow = self.shape()[dim];
        let block_src = dim_narrow * inner_size;
        let block_dst = dim_full * inner_size;

        if outer_size == 1 {
            // Single contiguous block at offset
            cuda.memcpy_dtod_f32(
                &mut dst,
                offset_elements,
                src_guard.slice(),
                0,
                grad_contig.shape.iter().product::<usize>(),
            )
            .expect("CUDA memcpy_dtod failed");
        } else {
            // Strided copy: for each outer block
            for o in 0..outer_size {
                let src_off = o * block_src;
                let dst_off = o * block_dst + offset_elements;
                cuda.memcpy_dtod_f32(&mut dst, dst_off, src_guard.slice(), src_off, block_src)
                    .expect("CUDA memcpy_dtod failed");
            }
        }

        let out_shape = Shape::from_slice(input_shape);
        Self {
            storage: Storage::from_cuda_slice(dst, numel, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        }
    }
}
