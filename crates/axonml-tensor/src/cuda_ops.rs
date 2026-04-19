//! CUDA GPU operations on `Tensor<f32>` — 3215 lines, 34 public methods.
//!
//! Feature-gated (`cuda`). Methods on `Tensor<f32>` that dispatch to the
//! `CudaBackend` singleton: `to_device` / `contiguous_gpu` / `to_vec` (for
//! GPU tensors), elementwise (add/sub/mul/div/scalar/neg/abs/pow), activations
//! (relu/sigmoid/tanh/gelu/silu/elu/leaky_relu/softmax/log_softmax),
//! reductions (sum/mean/max/min), matmul (cuBLAS GEMM), layernorm, RMSNorm,
//! transpose, embedding_gather, dropout, and quantized matmul dispatch
//! (`q4k_gemv_cuda`, `q4k_gemm_cuda`, `q6k_gemv_cuda`, `q6k_gemm_cuda`
//! for in-shader Q4_K/Q6_K dequant). Also `pool_alloc` + `get_cuda_backend`
//! helper re-exports for other crates.
//!
//! # File
//! `crates/axonml-tensor/src/cuda_ops.rs`
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

#[cfg(feature = "cuda")]
use axonml_core::Device;
#[cfg(feature = "cuda")]
use axonml_core::backends::cuda::get_cuda_backend;
#[cfg(feature = "cuda")]
use axonml_core::backends::cuda_pool::{pool_alloc, pool_alloc_uninit};
#[cfg(feature = "cuda")]
use axonml_core::error::Result;
#[cfg(feature = "cuda")]
use axonml_core::storage::Storage;

#[cfg(feature = "cuda")]
use crate::shape::{Shape, contiguous_strides};
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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(out_n).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(out_n).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(out_n).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(out_n).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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

    /// Fused SiLU backward on GPU. Computes `grad_input = grad_output *
    /// σ(x) * (1 + x*(1 - σ(x)))` in a single kernel launch. Replaces the
    /// SiluBackward::apply chain of 7 tensor ops + ones-H2D.
    pub(crate) fn silu_backward_cuda(&self, grad_output: &Self) -> Self {
        assert!(self.device().is_gpu(), "silu_backward_cuda: self on GPU");
        assert_eq!(
            self.shape(),
            grad_output.shape(),
            "silu_backward_cuda: shape mismatch"
        );
        let data = self.contiguous_gpu();
        let g = grad_output.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let g_guard = g.storage.as_cuda_slice();
        // pool_alloc_uninit: kernel writes every element exactly once.
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

        cuda.silu_backward_f32(&mut out, src_guard.slice(), g_guard.slice(), len)
            .expect("CUDA silu_backward_f32 failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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
            // broadcast_copy overwrites every byte before softmax reads it.
            let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

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
            // broadcast_copy overwrites every output byte — uninit safe.
            let mut out = pool_alloc_uninit(out_len).expect("GPU pool alloc failed");

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
        let src_strides =
            crate::shape::broadcast_strides(&data.shape, &data.strides, &result_shape);

        let indices: Vec<u32> = (0..out_len)
            .map(|i| {
                let coords = crate::shape::unravel_index(i, &result_shape);
                let src_idx = data.offset + crate::shape::linear_index(&coords, &src_strides);
                src_idx as u32
            })
            .collect();

        let idx_gpu = cuda.htod_copy(&indices).expect("htod indices failed");
        let src_guard = data.storage.as_cuda_slice();
        // gather_contiguous writes every output element — uninit safe.
        let mut out = pool_alloc_uninit(out_len).expect("GPU pool alloc failed");

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
    // Quantized matrix multiplication (GPU, dequant-in-shader)
    // =========================================================================

    /// Q4_K GEMM: `self` is `[m, in]` on GPU, `w` is a device-side `[out, in]`
    /// weight matrix in raw Q4_K bytes. Returns `[m, out]` on GPU.
    pub fn q4k_gemm_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu(), "q4k_gemm_cuda: self must be on GPU");
        assert_eq!(
            in_dim % 256,
            0,
            "q4k_gemm_cuda: in_dim must be a multiple of 256"
        );

        let a_data = self.contiguous_gpu();
        // self shape can be [m, in] or flat [m*in]; normalize.
        let numel = a_data.numel();
        assert!(
            numel % in_dim == 0,
            "q4k_gemm_cuda: numel ({}) not divisible by in_dim ({})",
            numel,
            in_dim
        );
        let m = numel / in_dim;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        // pool_alloc_uninit: q4k_gemm_f32 writes every element of `out`
        // (one thread per output element, total threads = m * out_dim).
        let mut out = pool_alloc_uninit(m * out_dim).expect("GPU pool alloc failed");

        // Order-matched GEMM — bit-identical to per-row q4k_gemv_f32.
        cuda.q4k_gemm_matched_f32(w, a_guard.slice(), &mut out, m, out_dim, in_dim)
            .expect("CUDA q4k_gemm_matched_f32 failed");

        let shape = Shape::from_slice(&[m, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, m * out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Q4_K GEMV: `self` is a `[1, in]` row vector on GPU, `w` is a device-side
    /// `[out, in]` weight matrix stored as raw Q4_K super-block bytes. Returns
    /// a `[1, out]` row vector on GPU.
    ///
    /// Calls the `q4k_gemv_f32` kernel — see `axonml-core/.../q4k_matmul.cu`.
    ///
    /// Requirements:
    ///   - `self.device()` is GPU, `self.numel() == in_dim`
    ///   - `in_dim % 256 == 0`
    ///   - `w.len() == out_dim * (in_dim / 256) * 144`
    pub fn q4k_gemv_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu(), "q4k_gemv_cuda: self must be on GPU");
        assert_eq!(
            self.numel(),
            in_dim,
            "q4k_gemv_cuda: self.numel() ({}) != in_dim ({})",
            self.numel(),
            in_dim
        );
        assert_eq!(
            in_dim % 256,
            0,
            "q4k_gemv_cuda: in_dim must be a multiple of 256"
        );

        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        // pool_alloc_uninit is safe here: q4k_gemv_f32 writes every element
        // of `out` in [0, out_dim) via the `c[j] = sum` store path (one
        // warp per output row, all rows covered by the grid).
        let mut out = pool_alloc_uninit(out_dim).expect("GPU pool alloc failed");

        cuda.q4k_gemv_f32(w, a_guard.slice(), &mut out, out_dim, in_dim)
            .expect("CUDA q4k_gemv_f32 failed");

        let shape = Shape::from_slice(&[1, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Q6_K GEMM: `self` is `[m, in]` on GPU, `w` is a device-side `[out, in]`
    /// weight matrix in raw Q6_K bytes. Returns `[m, out]` on GPU.
    /// Q5_0 GEMV — `self` is `[1, in]` f32 on GPU, `w` is device-side
    /// Q5_0 raw bytes `[out, in]`. Returns `[1, out]`.
    pub fn q5_0_gemv_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "q5_0_gemv_cuda: self must be on GPU"
        );
        assert_eq!(self.numel(), in_dim);
        assert_eq!(in_dim % 32, 0);
        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(out_dim).expect("GPU pool alloc failed");
        cuda.q5_0_gemv_f32(w, a_guard.slice(), &mut out, out_dim, in_dim)
            .expect("CUDA q5_0_gemv_f32 failed");
        let shape = Shape::from_slice(&[1, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn q5_0_gemm_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu());
        assert_eq!(in_dim % 32, 0);
        let a_data = self.contiguous_gpu();
        let numel = a_data.numel();
        assert!(numel % in_dim == 0);
        let m = numel / in_dim;
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(m * out_dim).expect("GPU pool alloc failed");
        cuda.q5_0_gemm_f32(w, a_guard.slice(), &mut out, m, out_dim, in_dim)
            .expect("CUDA q5_0_gemm_f32 failed");
        let shape = Shape::from_slice(&[m, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, m * out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn q5_1_gemv_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu());
        assert_eq!(self.numel(), in_dim);
        assert_eq!(in_dim % 32, 0);
        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(out_dim).expect("GPU pool alloc failed");
        cuda.q5_1_gemv_f32(w, a_guard.slice(), &mut out, out_dim, in_dim)
            .expect("CUDA q5_1_gemv_f32 failed");
        let shape = Shape::from_slice(&[1, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn q5_1_gemm_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu());
        assert_eq!(in_dim % 32, 0);
        let a_data = self.contiguous_gpu();
        let numel = a_data.numel();
        assert!(numel % in_dim == 0);
        let m = numel / in_dim;
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(m * out_dim).expect("GPU pool alloc failed");
        cuda.q5_1_gemm_f32(w, a_guard.slice(), &mut out, m, out_dim, in_dim)
            .expect("CUDA q5_1_gemm_f32 failed");
        let shape = Shape::from_slice(&[m, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, m * out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// BitNet I2_S GEMV — `self` is `[1, k]` f32 on GPU, `w` is device-side
    /// packed ternary bytes `[n, k/128 * 32]` (scale NOT included). Returns
    /// `[1, n]`. `scale` is the tensor-wide f32 scale read once at load.
    pub fn i2s_gemv_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        scale: f32,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu(), "i2s_gemv_cuda: self must be on GPU");
        assert_eq!(self.numel(), k);
        assert_eq!(k % 128, 0);
        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(n).expect("GPU pool alloc failed");
        cuda.i2s_gemv_f32(w, a_guard.slice(), &mut out, scale, n, k)
            .expect("CUDA i2s_gemv_f32 failed");
        let shape = Shape::from_slice(&[1, n]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, n, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn i2s_gemm_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        scale: f32,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu());
        assert_eq!(k % 128, 0);
        let a_data = self.contiguous_gpu();
        let numel = a_data.numel();
        assert!(numel % k == 0);
        let m = numel / k;
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(m * n).expect("GPU pool alloc failed");
        cuda.i2s_gemm_f32(w, a_guard.slice(), &mut out, scale, m, n, k)
            .expect("CUDA i2s_gemm_f32 failed");
        let shape = Shape::from_slice(&[m, n]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, m * n, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Q8_0 GEMV — `self` is `[1, in]` f32 on GPU, `w` is device-side
    /// Q8_0 raw bytes `[out, in]`. Returns `[1, out]`.
    pub fn q8_0_gemv_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "q8_0_gemv_cuda: self must be on GPU"
        );
        assert_eq!(self.numel(), in_dim);
        assert_eq!(in_dim % 32, 0);
        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(out_dim).expect("GPU pool alloc failed");
        cuda.q8_0_gemv_f32(w, a_guard.slice(), &mut out, out_dim, in_dim)
            .expect("CUDA q8_0_gemv_f32 failed");
        let shape = Shape::from_slice(&[1, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn q8_0_gemm_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu());
        assert_eq!(in_dim % 32, 0);
        let a_data = self.contiguous_gpu();
        let numel = a_data.numel();
        assert!(numel % in_dim == 0);
        let m = numel / in_dim;
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(m * out_dim).expect("GPU pool alloc failed");
        cuda.q8_0_gemm_f32(w, a_guard.slice(), &mut out, m, out_dim, in_dim)
            .expect("CUDA q8_0_gemm_f32 failed");
        let shape = Shape::from_slice(&[m, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, m * out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Q5_K GEMM: `self` is `[m, in]` f32 on GPU, `w` is a device-side
    /// `[out, in]` weight matrix in raw Q5_K super-block bytes (176 bytes
    /// per 256-element block). Returns `[m, out]`.
    pub fn q5k_gemm_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu(), "q5k_gemm_cuda: self must be on GPU");
        assert_eq!(
            in_dim % 256,
            0,
            "q5k_gemm_cuda: in_dim must be a multiple of 256"
        );

        let a_data = self.contiguous_gpu();
        let numel = a_data.numel();
        assert!(
            numel % in_dim == 0,
            "q5k_gemm_cuda: numel ({}) not divisible by in_dim ({})",
            numel,
            in_dim
        );
        let m = numel / in_dim;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(m * out_dim).expect("GPU pool alloc failed");

        // Order-matched GEMM — bit-identical to per-row q5k_gemv_f32. The
        // older naive q5k_gemm_f32 (one-thread-per-output) produces ~7e-6
        // max-abs-diff per call, which is fine for Qwen3/DeepSeek but
        // compounds over 32 layers beyond Phi-3's K/V tolerance. The
        // matched kernel uses the same warp-cooperative reduction and
        // split-at-half 2-warp layout as the GEMV, launched as a 2D grid
        // with blockIdx.y selecting the batch row.
        cuda.q5k_gemm_matched_f32(w, a_guard.slice(), &mut out, m, out_dim, in_dim)
            .expect("CUDA q5k_gemm_matched_f32 failed");

        let shape = Shape::from_slice(&[m, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, m * out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Q5_K GEMV: `self` is `[1, in]` row vector on GPU, `w` is device-side
    /// `[out, in]` Q5_K raw bytes. Returns `[1, out]`.
    pub fn q5k_gemv_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu(), "q5k_gemv_cuda: self must be on GPU");
        assert_eq!(
            self.numel(),
            in_dim,
            "q5k_gemv_cuda: self.numel() ({}) != in_dim ({})",
            self.numel(),
            in_dim
        );
        assert_eq!(
            in_dim % 256,
            0,
            "q5k_gemv_cuda: in_dim must be a multiple of 256"
        );

        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(out_dim).expect("GPU pool alloc failed");

        cuda.q5k_gemv_f32(w, a_guard.slice(), &mut out, out_dim, in_dim)
            .expect("CUDA q5k_gemv_f32 failed");

        let shape = Shape::from_slice(&[1, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn q6k_gemm_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu(), "q6k_gemm_cuda: self must be on GPU");
        assert_eq!(
            in_dim % 256,
            0,
            "q6k_gemm_cuda: in_dim must be a multiple of 256"
        );

        let a_data = self.contiguous_gpu();
        let numel = a_data.numel();
        assert!(
            numel % in_dim == 0,
            "q6k_gemm_cuda: numel ({}) not divisible by in_dim ({})",
            numel,
            in_dim
        );
        let m = numel / in_dim;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        // pool_alloc_uninit: q6k_gemm_f32 writes every element (same
        // one-thread-per-output contract as q4k_gemm_f32).
        let mut out = pool_alloc_uninit(m * out_dim).expect("GPU pool alloc failed");

        // Order-matched GEMM — bit-identical to per-row q6k_gemv_f32.
        cuda.q6k_gemm_matched_f32(w, a_guard.slice(), &mut out, m, out_dim, in_dim)
            .expect("CUDA q6k_gemm_matched_f32 failed");

        let shape = Shape::from_slice(&[m, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, m * out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Q6_K GEMV: `self` is `[1, in]` row vector on GPU, `w` is a device-side
    /// `[out, in]` weight matrix in raw Q6_K super-block bytes. Returns `[1, out]`.
    pub fn q6k_gemv_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self> {
        assert!(self.device().is_gpu(), "q6k_gemv_cuda: self must be on GPU");
        assert_eq!(
            self.numel(),
            in_dim,
            "q6k_gemv_cuda: self.numel() ({}) != in_dim ({})",
            self.numel(),
            in_dim
        );
        assert_eq!(
            in_dim % 256,
            0,
            "q6k_gemv_cuda: in_dim must be a multiple of 256"
        );

        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        // pool_alloc_uninit: q6k_gemv_f32 writes every output element via
        // the one-warp-per-row store path (same structure as q4k_gemv_f32).
        let mut out = pool_alloc_uninit(out_dim).expect("GPU pool alloc failed");

        cuda.q6k_gemv_f32(w, a_guard.slice(), &mut out, out_dim, in_dim)
            .expect("CUDA q6k_gemv_f32 failed");

        let shape = Shape::from_slice(&[1, out_dim]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, out_dim, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    // =========================================================================
    // Matrix Multiplication (GPU) — the critical speedup
    // =========================================================================

    /// GPU matrix multiplication using cuBLAS GEMM — no CPU copies.
    pub(crate) fn matmul_cuda(&self, other: &Self) -> Result<Self> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        // Detect if a tensor is a simple transpose of the last two dims.
        // If so, we can pass the transpose flag to cuBLAS and avoid
        // the expensive CPU gather-index computation in contiguous_gpu().
        fn is_last2_transposed(t: &Tensor<f32>) -> bool {
            let nd = t.ndim();
            if nd < 2 {
                return false;
            }
            let strides = t.strides.as_slice();
            // A row-major [.., M, K] has strides [.., K, 1].
            // Transposed [.., K, M] (view) has strides [.., 1, K] with shape [.., M, K].
            // Detect: stride[-1] > stride[-2], meaning columns are strided, rows are packed.
            strides[nd - 1] > strides[nd - 2]
        }

        // Check if batch dims (all except last 2) are contiguous.
        // Returns true if only the last 2 dims are transposed.
        fn batch_contiguous(t: &Tensor<f32>) -> bool {
            let nd = t.ndim();
            if nd <= 2 {
                return true;
            }
            let strides = t.strides.as_slice();
            let shape = t.shape.as_slice();
            // Check batch strides are standard row-major
            // For [B1, B2, .., M, K] with potential last-2 transpose:
            // batch stride[i] should = product of shape[i+1..] considering the last-2 block
            let mat_size = shape[nd - 2] * shape[nd - 1];
            let mut expected = mat_size as isize;
            for i in (0..nd - 2).rev() {
                if strides[i] != expected {
                    return false;
                }
                expected *= shape[i] as isize;
            }
            true
        }

        let a_transposed = is_last2_transposed(self) && batch_contiguous(self) && self.offset == 0;
        let b_transposed =
            is_last2_transposed(other) && batch_contiguous(other) && other.offset == 0;

        // For transposed tensors, the "logical" shape has the last two dims swapped
        // relative to the memory layout. We pass the original (pre-transpose) dims
        // to cuBLAS with the transpose flag.
        let a = if a_transposed {
            self.clone()
        } else {
            self.contiguous_gpu()
        };
        let b = if b_transposed {
            other.clone()
        } else {
            other.contiguous_gpu()
        };

        // Logical matmul dimensions from the SHAPES (not memory layout)
        let m = a.shape[a.shape.len() - 2];
        let k = a.shape[a.shape.len() - 1];
        let n = b.shape[b.shape.len() - 1];

        // Guard: cuBLAS requires all dimensions > 0
        if m == 0 || k == 0 || n == 0 {
            let out_shape: Vec<usize> = if a.shape.len() == 2 {
                vec![m, n]
            } else {
                let mut s: Vec<usize> = a.shape[..a.shape.len() - 2].to_vec();
                s.push(m);
                s.push(n);
                s
            };
            let total: usize = out_shape.iter().product();
            return Ok(Self::from_vec(vec![0.0f32; total], &out_shape)?);
        }

        if a.shape.len() == 2 && b.shape.len() == 2 {
            // 2D matmul: C(m,n) = A(m,k) @ B(k,n)
            // cuBLAS column-major: C^T(n,m) = B_cm @ A_cm
            // Row-major A(m,k) is column-major A^T(k,m).
            // If A is transposed in row-major, its memory is A_orig(k,m) which in
            // column-major is A_orig^T(m,k) — so we pass trans=true to undo it.
            let a_guard = a.storage.as_cuda_slice();
            let b_guard = b.storage.as_cuda_slice();
            // cuBLAS GEMM writes every output element with beta=0 — uninit safe.
            let mut c_gpu =
                pool_alloc_uninit(m * n).map_err(|e| crate::Error::InvalidOperation {
                    message: format!("GPU OOM in 2D matmul ({}x{}x{}): {}", m, k, n, e),
                })?;

            // cuBLAS sees column-major data:
            // Row-major A(m,k) → col-major view as (k,m) = A^T
            // If a_transposed: memory is (k,m) row-major → col-major (m,k) → needs op_T to get (k,m)
            let (lda, op_a) = if a_transposed { (m, true) } else { (k, false) };
            let (ldb, op_b) = if b_transposed { (k, true) } else { (n, false) };

            // cuBLAS: C^T(n,m) = B_col(op_b) @ A_col(op_a)
            // Validate lda/ldb/ldc — cuBLAS requires lda >= max(1, rows_of_op(A))
            let lda_min = if op_a { m } else { k };
            let ldb_min = if op_b { k } else { n };
            assert!(
                lda >= lda_min.max(1),
                "cuBLAS lda={} < min={} (m={}, k={}, op_a={})",
                lda,
                lda_min,
                m,
                k,
                op_a
            );
            assert!(
                ldb >= ldb_min.max(1),
                "cuBLAS ldb={} < min={} (k={}, n={}, op_b={})",
                ldb,
                ldb_min,
                k,
                n,
                op_b
            );
            assert!(n >= 1, "cuBLAS ldc=n={} must be >= 1", n);

            cuda.gemm_f32(
                op_b,
                op_a,
                n,
                m,
                k,
                1.0,
                b_guard.slice(),
                ldb,
                a_guard.slice(),
                lda,
                0.0,
                &mut c_gpu,
                n,
            )
            .expect("cuBLAS gemm failed");

            let storage = Storage::from_cuda_slice(c_gpu, m * n, self.device());
            return Ok(Self {
                storage,
                shape: Shape::from_slice(&[m, n]),
                strides: contiguous_strides(&Shape::from_slice(&[m, n])),
                offset: 0,
            });
        }

        // Batched matmul: cublasSgemmStridedBatched
        let batch_dims: Vec<usize> = a.shape[..a.shape.len() - 2].to_vec();
        let batch_size: usize = batch_dims.iter().product();

        // Guard: cuBLAS requires all dimensions > 0
        if batch_size == 0 || m == 0 || k == 0 || n == 0 {
            let mut out_shape = batch_dims.clone();
            out_shape.push(m);
            out_shape.push(n);
            let total: usize = out_shape.iter().product();
            return Ok(Self::from_vec(vec![0.0f32; total.max(1)], &out_shape)?);
        }

        let total = batch_size * m * n;

        let a_guard = a.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();

        // Row-major batched matmul: C[b](m,n) = A[b](m,k) @ B[b](k,n)
        //
        // cuBLAS is column-major. Row-major data viewed as col-major is transposed.
        // We use the identity: C_row = A_row @ B_row  ↔  C_col^T = B_col^T @ A_col^T
        //
        // cuBLAS call: C_cublas(cublas_m, cublas_n) = op(A_cublas) @ op(B_cublas)
        //   cublas_m = n (our n), cublas_n = m (our m)
        //   A_cublas = our B data, B_cublas = our A data
        //
        // For non-transposed row-major matrices (the common case):
        //   our B(k,n) in row-major = (n,k) in col-major → transa='T', lda=n
        //   our A(m,k) in row-major = (k,m) in col-major → transb='T', ldb=k
        //   C stored col-major (n,m) → ldc=n
        //
        // For "transposed" matrices (memory layout has last 2 dims swapped):
        //   our B "transposed": memory is (n,k) row-major = (k,n) col-major → transa='N', lda=k
        //   our A "transposed": memory is (k,m) row-major = (m,k) col-major → transb='N', ldb=m

        // Row-major B(k,n) viewed as col-major = (n,k). We need cublas op(A) = (n,k).
        //   transa='N': A_cublas is (cublas_m=n, k) col-major, lda=n. Matches (n,k). ✓
        //   If b_transposed: memory is (n,k) row-major = (k,n) col-major. Need (n,k) → transa='T', lda=k.
        let (cublas_transa, cublas_lda) = if b_transposed {
            (true, k) // memory (n,k) row → (k,n) col → transpose to (n,k), lda=k
        } else {
            (false, n) // memory (k,n) row → (n,k) col → no transpose needed, lda=n
        };
        // Row-major A(m,k) viewed as col-major = (k,m). We need cublas op(B) = (k,m).
        //   transb='N': B_cublas is (k, cublas_n=m) col-major, ldb=k. Matches (k,m). ✓
        //   If a_transposed: memory is (k,m) row-major = (m,k) col-major. Need (k,m) → transb='T', ldb=m.
        let (cublas_transb, cublas_ldb) = if a_transposed {
            (true, m) // memory (k,m) row → (m,k) col → transpose to (k,m), ldb=m
        } else {
            (false, k) // memory (m,k) row → (k,m) col → no transpose needed, ldb=k
        };
        let cublas_ldc = n;
        // Strided batched GEMM: one cuBLAS call on the whole batch — no CPU
        // round-trip, no per-batch alloc. All operands already live in a
        // single contiguous GPU buffer with fixed stride between batches.
        //
        // History: an earlier implementation here hit SgemmStridedBatched
        // driver issues on some GPUs and fell back to a CPU-assembled loop
        // (D2H both inputs → per-batch H2D → GEMM → per-batch D2H → reassemble
        // on CPU → H2D result). That was ~313 ms/call on Qwen3-0.6B training
        // and dominated backward (79% of the pass). Reverted to on-device
        // strided batched; the driver issue no longer reproduces on the
        // current cudarc + 580-series CUDA stack.
        let stride_a_elems = (m * k) as i64;
        let stride_b_elems = (k * n) as i64;
        let stride_c_elems = (m * n) as i64;

        let a_guard = a.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();
        let mut c_gpu = pool_alloc_uninit(total).map_err(|e| crate::Error::InvalidOperation {
            message: format!(
                "GPU pool alloc in batched matmul ({}x{}x{}x{}): {}",
                batch_size, m, k, n, e
            ),
        })?;

        cuda.gemm_strided_batched_f32(
            cublas_transa,
            cublas_transb,
            n,
            m,
            k,
            1.0,
            b_guard.slice(),
            cublas_lda,
            stride_b_elems,
            a_guard.slice(),
            cublas_ldb,
            stride_a_elems,
            0.0,
            &mut c_gpu,
            cublas_ldc,
            stride_c_elems,
            batch_size,
        )
        .map_err(|e| crate::Error::InvalidOperation {
            message: format!(
                "cuBLAS strided batched gemm failed (batch={}, m={}, n={}, k={}): {:?}",
                batch_size, m, n, k, e,
            ),
        })?;

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

    /// Returns a contiguous GPU tensor — fully on-device using strided gather kernel.
    /// Computes gather indices directly on GPU, avoiding CPU index computation.
    pub(crate) fn contiguous_gpu(&self) -> Self {
        if self.is_contiguous() && self.offset == 0 {
            return self.clone();
        }
        let total = self.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let ndim = self.shape.len();
        let offset = self.offset;
        let shape = self.shape.as_slice();
        let strides = self.strides.as_slice();

        // Upload shape and strides to GPU (small arrays, cheap transfer)
        let shape_u32: Vec<u32> = shape.iter().map(|&s| s as u32).collect();
        let strides_i64: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
        let shape_gpu = cuda.htod_copy(&shape_u32).expect("htod shape failed");
        let strides_gpu = cuda.htod_copy(&strides_i64).expect("htod strides failed");

        let src_guard = self.storage.as_cuda_slice();
        // strided_gather_f32 writes every output position — uninit safe.
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.strided_gather_f32(
            src_guard.slice(),
            &mut out,
            &strides_gpu,
            &shape_gpu,
            ndim,
            offset,
            total,
        )
        .expect("CUDA strided_gather_f32 failed");

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
    pub fn embedding_gather_cuda(&self, gather_indices: &[u32], output_shape: &[usize]) -> Self {
        let output_size = output_shape.iter().product::<usize>();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let idx_gpu = cuda
            .htod_copy(gather_indices)
            .expect("htod gather indices failed");
        let weight_guard = self.storage.as_cuda_slice();
        // gather writes every output element — uninit safe.
        let mut out = pool_alloc_uninit(output_size).expect("GPU pool alloc failed");

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

    /// Embedding backward: scatter-add grad_output into weight gradient on GPU.
    /// grad_output shape: [num_indices, emb_dim] (contiguous on GPU)
    /// indices: token indices as u32 (small, uploaded from CPU)
    /// Returns: Tensor of shape [num_embeddings, emb_dim] with accumulated gradients.
    pub fn embedding_scatter_add_cuda(
        &self,
        indices: &[u32],
        num_embeddings: usize,
        emb_dim: usize,
    ) -> Self {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let num_indices = indices.len();
        let total_n = num_indices * emb_dim;

        // Upload indices to GPU (small: batch_size * seq_len u32 values)
        let idx_gpu = cuda.htod_copy(indices).expect("htod indices failed");

        // Allocate zeroed output on GPU: [num_embeddings, emb_dim]
        let out_size = num_embeddings * emb_dim;
        let mut out = pool_alloc(out_size).expect("GPU pool alloc failed");
        cuda.memset_zeros_f32(&mut out)
            .expect("memset zeros failed");

        // Ensure grad_output is contiguous on GPU
        let grad = self.contiguous_gpu();
        let grad_guard = grad.storage.as_cuda_slice();

        cuda.embedding_scatter_add_f32(grad_guard.slice(), &idx_gpu, &mut out, total_n, emb_dim)
            .expect("CUDA embedding_scatter_add_f32 failed");

        let shape = crate::shape::Shape::from_slice(&[num_embeddings, emb_dim]);
        let storage = Storage::from_cuda_slice(out, out_size, self.device());
        Self {
            storage,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        }
    }

    /// Fused Adam optimizer step: updates param, exp_avg, exp_avg_sq in-place on GPU.
    /// Single kernel launch per parameter — eliminates 8+ separate tensor ops.
    ///
    /// `self` is the parameter tensor (modified in-place).
    /// `grad` is the gradient tensor.
    /// `exp_avg` and `exp_avg_sq` are the optimizer state tensors (modified in-place).
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_inplace(
        &self,
        grad: &Self,
        exp_avg: &Self,
        exp_avg_sq: &Self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
    ) {
        let n = self.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        // Get mutable access to param, exp_avg, exp_avg_sq
        let mut param_guard = self.storage.as_cuda_slice_mut();
        let grad_guard = grad.storage.as_cuda_slice();
        let mut avg_guard = exp_avg.storage.as_cuda_slice_mut();
        let mut sq_guard = exp_avg_sq.storage.as_cuda_slice_mut();

        cuda.adam_step_f32(
            param_guard.slice_mut(),
            grad_guard.slice(),
            avg_guard.slice_mut(),
            sq_guard.slice_mut(),
            n,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        )
        .expect("CUDA adam_step_f32 failed");
    }

    /// Compute total gradient norm and clip in-place on GPU.
    /// Single GPU→CPU copy of 1 float for the norm, then scale kernels if needed.
    /// Returns the total L2 norm before clipping.
    pub fn clip_grad_norm_cuda(grads: &[Self], max_norm: f32) -> f32 {
        if grads.is_empty() {
            return 0.0;
        }
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        // Single accumulator for ALL params
        let mut acc = pool_alloc(1).expect("GPU pool alloc failed");
        cuda.memset_zeros_f32(&mut acc).expect("memset failed");

        // Launch norm kernel for each grad — all atomically add to same accumulator
        // No GPU→CPU copies in this loop
        for grad in grads {
            let data = grad.contiguous_gpu();
            let n = data.numel();
            let guard = data.storage.as_cuda_slice();
            cuda.grad_norm_sq_f32(guard.slice(), &mut acc, n)
                .expect("CUDA grad_norm_sq_f32 failed");
        }

        // ONE GPU→CPU copy: 1 float
        let result = cuda.dtoh_copy(&acc).expect("dtoh failed");
        let total_norm = result[0].sqrt();

        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            for grad in grads {
                let n = grad.numel();
                let mut guard = grad.storage.as_cuda_slice_mut();
                cuda.grad_scale_f32(guard.slice_mut(), n, scale)
                    .expect("CUDA grad_scale_f32 failed");
            }
        }

        total_norm
    }

    /// Scale all elements in-place: self[i] *= scale. No CPU copies.
    pub fn grad_scale_inplace(&self, scale: f32) {
        let n = self.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let mut guard = self.storage.as_cuda_slice_mut();
        cuda.grad_scale_f32(guard.slice_mut(), n, scale)
            .expect("CUDA grad_scale_f32 failed");
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

        cuda.sum_dim_f32(
            &mut out,
            src_guard.slice(),
            outer_size,
            dim_size,
            inner_size,
        )
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

        cuda.sum_dim_f32(
            &mut out,
            src_guard.slice(),
            outer_size,
            dim_size,
            inner_size,
        )
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
    pub fn narrow_backward_cuda(&self, input_shape: &[usize], dim: usize, start: usize) -> Self {
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

    // =========================================================================
    // Attention Mask Expansion (GPU)
    // =========================================================================

    /// Expand attention mask on GPU: converts 0→-1e9 (masked) and broadcasts to
    /// [batch, heads, tgt_len, src_len]. Supports causal [T,S] and padding [B,S] masks.
    ///
    /// Returns `None` if GPU expansion fails or mask shape is unsupported.
    pub fn mask_expand_cuda(
        &self,
        output_shape: &[usize],
        batch_size: usize,
        num_heads: usize,
        tgt_len: usize,
        src_len: usize,
    ) -> Option<Self> {
        let cuda = get_cuda_backend()?;
        let data = self.contiguous_gpu();
        let mask_shape = &data.shape;
        let total: usize = output_shape.iter().product();

        let mask_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(total).ok()?;

        let result = if mask_shape.len() == 2
            && mask_shape[0] == tgt_len
            && mask_shape[1] == src_len
        {
            cuda.mask_expand_causal_f32(mask_guard.slice(), &mut out, total, tgt_len, src_len)
        } else if mask_shape.len() == 2 && mask_shape[0] == batch_size && mask_shape[1] == src_len {
            cuda.mask_expand_padding_f32(
                mask_guard.slice(),
                &mut out,
                total,
                num_heads,
                tgt_len,
                src_len,
            )
        } else {
            return None;
        };

        result.ok()?;

        let out_shape = Shape::from_slice(output_shape);
        Some(Self {
            storage: Storage::from_cuda_slice(out, total, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        })
    }

    // =========================================================================
    // Fused LSTM Gate Computation (GPU)
    // =========================================================================

    /// Fused LSTM gate kernel: takes pre-summed gates [batch, 4*hidden] and
    /// previous cell state [batch, hidden], applies sigmoid/tanh activations
    /// and computes new (h, c) in a single kernel launch.
    ///
    /// Returns (h_new, c_new) tensors on GPU.
    pub fn lstm_gates_fused(&self, c_prev: &Self, hidden_size: usize) -> Option<(Self, Self)> {
        let batch_size = self.shape()[0];
        let total = batch_size * hidden_size;
        let cuda = get_cuda_backend()?;

        let gates_contig = self.contiguous_gpu();
        let c_contig = c_prev.contiguous_gpu();
        let gates_guard = gates_contig.storage.as_cuda_slice();
        let c_guard = c_contig.storage.as_cuda_slice();

        let mut h_out = pool_alloc(total).ok()?;
        let mut c_out = pool_alloc(total).ok()?;

        cuda.lstm_gates_f32(
            gates_guard.slice(),
            c_guard.slice(),
            &mut h_out,
            &mut c_out,
            hidden_size,
            total,
        )
        .ok()?;

        let h_storage = Storage::from_cuda_slice(h_out, total, self.device());
        let c_storage = Storage::from_cuda_slice(c_out, total, self.device());

        let sh = Shape::from_slice(&[batch_size, hidden_size]);
        let h_tensor = Self {
            storage: h_storage,
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        };
        let c_tensor = Self {
            storage: c_storage,
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        };

        Some((h_tensor, c_tensor))
    }

    /// Fused GRU gate kernel: takes ih gates [batch, 3*hidden], hh gates [batch, 3*hidden],
    /// and previous hidden [batch, hidden], computes new h in a single kernel.
    pub fn gru_gates_fused(
        &self,
        gates_hh: &Self,
        h_prev: &Self,
        hidden_size: usize,
    ) -> Option<Self> {
        let batch_size = self.shape()[0];
        let total = batch_size * hidden_size;
        let cuda = get_cuda_backend()?;

        let ih_contig = self.contiguous_gpu();
        let hh_contig = gates_hh.contiguous_gpu();
        let h_contig = h_prev.contiguous_gpu();
        let ih_guard = ih_contig.storage.as_cuda_slice();
        let hh_guard = hh_contig.storage.as_cuda_slice();
        let h_guard = h_contig.storage.as_cuda_slice();

        let mut h_out = pool_alloc(total).ok()?;

        cuda.gru_gates_f32(
            ih_guard.slice(),
            hh_guard.slice(),
            h_guard.slice(),
            &mut h_out,
            hidden_size,
            total,
        )
        .ok()?;

        let h_storage = Storage::from_cuda_slice(h_out, total, self.device());

        let sh = Shape::from_slice(&[batch_size, hidden_size]);
        Some(Self {
            storage: h_storage,
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        })
    }

    // =========================================================================
    // Fused LSTM Gate Backward (GPU)
    // =========================================================================

    /// Fused LSTM gate backward on GPU.
    ///
    /// Given saved forward state and incoming gradients, computes gate gradients
    /// [batch, 4*hidden] and cell gradient to previous timestep [batch, hidden].
    ///
    /// - `self`: gates [batch, 4*hidden] pre-activation from forward
    /// - `c_prev`: [batch, hidden]
    /// - `c_new`: [batch, hidden]
    /// - `grad_h`: [batch, hidden]
    /// - `grad_c_next`: [batch, hidden]
    ///
    /// Returns (grad_gates [batch, 4*hidden], grad_c_prev [batch, hidden]).
    pub fn lstm_gates_backward_fused(
        &self,
        c_prev: &Self,
        c_new: &Self,
        grad_h: &Self,
        grad_c_next: &Self,
        hidden_size: usize,
    ) -> Option<(Self, Self)> {
        let batch_size = grad_h.shape()[0];
        let total = batch_size * hidden_size;
        let cuda = get_cuda_backend()?;

        let gates_contig = self.contiguous_gpu();
        let c_prev_contig = c_prev.contiguous_gpu();
        let c_new_contig = c_new.contiguous_gpu();
        let grad_h_contig = grad_h.contiguous_gpu();
        let grad_c_contig = grad_c_next.contiguous_gpu();

        let gates_guard = gates_contig.storage.as_cuda_slice();
        let c_prev_guard = c_prev_contig.storage.as_cuda_slice();
        let c_new_guard = c_new_contig.storage.as_cuda_slice();
        let grad_h_guard = grad_h_contig.storage.as_cuda_slice();
        let grad_c_guard = grad_c_contig.storage.as_cuda_slice();

        let mut grad_gates_out = pool_alloc(batch_size * 4 * hidden_size).ok()?;
        let mut grad_c_prev_out = pool_alloc(total).ok()?;

        cuda.lstm_gates_backward_f32(
            gates_guard.slice(),
            c_prev_guard.slice(),
            c_new_guard.slice(),
            grad_h_guard.slice(),
            grad_c_guard.slice(),
            &mut grad_gates_out,
            &mut grad_c_prev_out,
            hidden_size,
            total,
        )
        .ok()?;

        let grad_gates_storage =
            Storage::from_cuda_slice(grad_gates_out, batch_size * 4 * hidden_size, self.device());
        let grad_c_prev_storage = Storage::from_cuda_slice(grad_c_prev_out, total, self.device());

        let sh_gates = Shape::from_slice(&[batch_size, 4 * hidden_size]);
        let sh_hidden = Shape::from_slice(&[batch_size, hidden_size]);
        let grad_gates_tensor = Self {
            storage: grad_gates_storage,
            shape: sh_gates.clone(),
            strides: contiguous_strides(&sh_gates),
            offset: 0,
        };
        let grad_c_prev_tensor = Self {
            storage: grad_c_prev_storage,
            shape: sh_hidden.clone(),
            strides: contiguous_strides(&sh_hidden),
            offset: 0,
        };

        Some((grad_gates_tensor, grad_c_prev_tensor))
    }

    // =========================================================================
    // Fused GRU Gate Backward (GPU)
    // =========================================================================

    /// Fused GRU gate backward on GPU.
    ///
    /// Given saved forward state and incoming gradient, computes ih/hh gate
    /// gradients and hidden state gradient to previous timestep.
    ///
    /// - `self`: gates_ih [batch, 3*hidden] pre-activation from forward
    /// - `gates_hh`: [batch, 3*hidden] pre-activation from forward
    /// - `h_prev`: [batch, hidden]
    /// - `grad_h_new`: [batch, hidden]
    ///
    /// Returns (grad_gates_ih [batch, 3*hidden], grad_gates_hh [batch, 3*hidden], grad_h_prev [batch, hidden]).
    pub fn gru_gates_backward_fused(
        &self,
        gates_hh: &Self,
        h_prev: &Self,
        grad_h_new: &Self,
        hidden_size: usize,
    ) -> Option<(Self, Self, Self)> {
        let batch_size = grad_h_new.shape()[0];
        let total = batch_size * hidden_size;
        let cuda = get_cuda_backend()?;

        let ih_contig = self.contiguous_gpu();
        let hh_contig = gates_hh.contiguous_gpu();
        let h_contig = h_prev.contiguous_gpu();
        let grad_contig = grad_h_new.contiguous_gpu();

        let ih_guard = ih_contig.storage.as_cuda_slice();
        let hh_guard = hh_contig.storage.as_cuda_slice();
        let h_guard = h_contig.storage.as_cuda_slice();
        let grad_guard = grad_contig.storage.as_cuda_slice();

        let mut grad_ih_out = pool_alloc(batch_size * 3 * hidden_size).ok()?;
        let mut grad_hh_out = pool_alloc(batch_size * 3 * hidden_size).ok()?;
        let mut grad_h_prev_out = pool_alloc(total).ok()?;

        cuda.gru_gates_backward_f32(
            ih_guard.slice(),
            hh_guard.slice(),
            h_guard.slice(),
            grad_guard.slice(),
            &mut grad_ih_out,
            &mut grad_hh_out,
            &mut grad_h_prev_out,
            hidden_size,
            total,
        )
        .ok()?;

        let grad_ih_storage =
            Storage::from_cuda_slice(grad_ih_out, batch_size * 3 * hidden_size, self.device());
        let grad_hh_storage =
            Storage::from_cuda_slice(grad_hh_out, batch_size * 3 * hidden_size, self.device());
        let grad_h_prev_storage = Storage::from_cuda_slice(grad_h_prev_out, total, self.device());

        let sh_3h = Shape::from_slice(&[batch_size, 3 * hidden_size]);
        let sh_h = Shape::from_slice(&[batch_size, hidden_size]);
        let grad_ih_tensor = Self {
            storage: grad_ih_storage,
            shape: sh_3h.clone(),
            strides: contiguous_strides(&sh_3h),
            offset: 0,
        };
        let grad_hh_tensor = Self {
            storage: grad_hh_storage,
            shape: sh_3h.clone(),
            strides: contiguous_strides(&sh_3h),
            offset: 0,
        };
        let grad_h_prev_tensor = Self {
            storage: grad_h_prev_storage,
            shape: sh_h.clone(),
            strides: contiguous_strides(&sh_h),
            offset: 0,
        };

        Some((grad_ih_tensor, grad_hh_tensor, grad_h_prev_tensor))
    }

    // =========================================================================
    // Fused BatchNorm Forward (GPU)
    // =========================================================================

    /// BatchNorm forward on GPU: 2-pass (stats + normalize).
    ///
    /// - `self`: input [N, C, spatial...]
    /// - `gamma`: [C] scale
    /// - `beta`: [C] bias
    /// - Returns (output, mean, var) all on GPU
    pub fn batchnorm_fused(
        &self,
        gamma: &Self,
        beta: &Self,
        eps: f32,
        channels: usize,
        spatial: usize,
    ) -> Option<(Self, Vec<f32>, Vec<f32>)> {
        let cuda = get_cuda_backend()?;
        let total = self.numel();
        let n = total / (channels * spatial);

        let input_contig = self.contiguous_gpu();
        let gamma_contig = gamma.contiguous_gpu();
        let beta_contig = beta.contiguous_gpu();

        let input_guard = input_contig.storage.as_cuda_slice();
        let gamma_guard = gamma_contig.storage.as_cuda_slice();
        let beta_guard = beta_contig.storage.as_cuda_slice();

        // Pass 1: compute sum and sum_sq per channel
        let zeros_c = vec![0.0f32; channels];
        let mut sum_gpu = cuda.htod_copy(&zeros_c).ok()?;
        let mut sum_sq_gpu = cuda.htod_copy(&zeros_c).ok()?;

        cuda.batchnorm_stats_f32(
            input_guard.slice(),
            &mut sum_gpu,
            &mut sum_sq_gpu,
            n,
            channels,
            spatial,
        )
        .ok()?;

        // Copy stats back to CPU for running mean/var update + backward storage
        let sum_cpu = cuda.dtoh_copy::<f32>(&sum_gpu).ok()?;
        let sum_sq_cpu = cuda.dtoh_copy::<f32>(&sum_sq_gpu).ok()?;

        let n_per_ch = (n * spatial) as f32;
        let mut mean_cpu = vec![0.0f32; channels];
        let mut var_cpu = vec![0.0f32; channels];
        for c in 0..channels {
            mean_cpu[c] = sum_cpu[c] / n_per_ch;
            var_cpu[c] = sum_sq_cpu[c] / n_per_ch - mean_cpu[c] * mean_cpu[c];
        }

        // Upload mean/var to GPU for pass 2
        let mean_gpu = cuda.htod_copy(&mean_cpu).ok()?;
        let var_gpu = cuda.htod_copy(&var_cpu).ok()?;

        // Pass 2: normalize + affine
        let mut out_gpu = pool_alloc(total).ok()?;

        cuda.batchnorm_norm_f32(
            input_guard.slice(),
            &mean_gpu,
            &var_gpu,
            gamma_guard.slice(),
            beta_guard.slice(),
            &mut out_gpu,
            eps,
            channels,
            spatial,
            total,
        )
        .ok()?;

        let out_storage = Storage::from_cuda_slice(out_gpu, total, self.device());
        let out_tensor = Self {
            storage: out_storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        };

        Some((out_tensor, mean_cpu, var_cpu))
    }

    // =========================================================================
    // GPU-Resident Conv2d (im2col + cuBLAS GEMM)
    // =========================================================================

    /// GPU-resident Conv2d forward: im2col + GEMM + bias add, all on GPU.
    ///
    /// `self` is the input tensor `[N, C_in, H, W]` on GPU.
    /// `weight` is `[C_out, C_in, kH, kW]` on GPU.
    /// `bias` is optional `[C_out]` on GPU.
    ///
    /// Returns output `[N, C_out, H_out, W_out]` on GPU.
    /// Groups=1 only. Returns `None` if any GPU operation fails.
    pub fn conv2d_cuda(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Option<Self> {
        // All tensors must be GPU-resident
        if !self.device().is_gpu() || !weight.device().is_gpu() {
            return None;
        }
        if let Some(b) = bias {
            if !b.device().is_gpu() {
                return None;
            }
        }
        let cuda = get_cuda_backend()?;

        let batch_size = self.shape[0];
        let in_channels = self.shape[1];
        let in_height = self.shape[2];
        let in_width = self.shape[3];
        let out_channels = weight.shape[0];
        let kernel_h = weight.shape[2];
        let kernel_w = weight.shape[3];
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        let out_h = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
        let col_h = in_channels * kernel_h * kernel_w;
        let col_w = out_h * out_w;
        let col_n = col_h * col_w;
        let spatial = out_h * out_w;
        let out_per_batch = out_channels * spatial;
        let in_per_batch = in_channels * in_height * in_width;

        // Ensure input and weight are contiguous on GPU
        let input_data = self.contiguous_gpu();
        let weight_data = weight.contiguous_gpu();

        let input_guard = input_data.storage.as_cuda_slice();
        let weight_guard = weight_data.storage.as_cuda_slice();

        // Upload im2col parameters (small, cheap)
        let im2col_params: [u32; 10] = [
            in_height as u32,
            in_width as u32,
            kernel_h as u32,
            kernel_w as u32,
            pad_h as u32,
            pad_w as u32,
            stride_h as u32,
            stride_w as u32,
            out_h as u32,
            out_w as u32,
        ];
        let params_gpu = cuda.htod_copy(&im2col_params[..]).ok()?;

        // Keep bias tensor alive, then borrow its GPU slice
        let bias_data = bias.map(|b| b.contiguous_gpu());
        let bias_guard = bias_data.as_ref().map(|b| b.storage.as_cuda_slice());

        // Pool-allocate col buffer (reused across batches)
        let mut col_gpu = pool_alloc(col_n).ok()?;

        // Per-batch input buffer for im2col (d2d copy into here)
        let mut input_batch_gpu = pool_alloc(in_per_batch).ok()?;

        // Per-batch output buffer for GEMM
        let mut batch_out_gpu = pool_alloc(out_per_batch).ok()?;

        // Allocate output buffer for ALL batches on GPU
        let total_out = batch_size * out_per_batch;
        let mut out_gpu = pool_alloc(total_out).ok()?;

        for b in 0..batch_size {
            // d2d copy: input[b] from full buffer → per-batch buffer (GPU→GPU, fast)
            cuda.memcpy_dtod_f32(
                &mut input_batch_gpu,
                0,
                input_guard.slice(),
                b * in_per_batch,
                in_per_batch,
            )
            .ok()?;

            // GPU im2col: input_batch [C_in, H, W] → col [col_h, col_w]
            cuda.im2col_f32(&input_batch_gpu, &mut col_gpu, &params_gpu, col_n)
                .ok()?;

            // GPU GEMM: batch_out = weight @ col
            // weight: [out_channels, col_h] row-major
            // col: [col_h, col_w] row-major
            // result: [out_channels, col_w] row-major
            //
            // cuBLAS column-major: C^T = B^T @ A^T
            // m=col_w, n=out_channels, k=col_h
            cuda.gemm_f32(
                false,
                false,
                col_w,
                out_channels,
                col_h,
                1.0,
                &col_gpu,
                col_w,
                weight_guard.slice(),
                col_h,
                0.0,
                &mut batch_out_gpu,
                col_w,
            )
            .ok()?;

            // GPU bias add (in-place on batch_out_gpu)
            if let Some(ref bg) = bias_guard {
                cuda.bias_add_channels_f32(&mut batch_out_gpu, bg.slice(), spatial, out_per_batch)
                    .ok()?;
            }

            // d2d copy: batch output → final output buffer at right offset
            cuda.memcpy_dtod_f32(
                &mut out_gpu,
                b * out_per_batch,
                &batch_out_gpu,
                0,
                out_per_batch,
            )
            .ok()?;
        }

        let out_shape = Shape::from_slice(&[batch_size, out_channels, out_h, out_w]);
        Some(Self {
            storage: Storage::from_cuda_slice(out_gpu, total_out, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        })
    }

    /// GPU-resident Conv2d forward using cuDNN.
    ///
    /// `self` is `[N, C_in, H, W]` on GPU.
    /// `weight` is `[C_out, C_in/groups, kH, kW]` on GPU.
    /// `bias` is optional `[C_out]` on GPU.
    /// `groups` is the number of convolution groups.
    ///
    /// Returns `None` if cuDNN is not available or any operation fails.
    /// Caller should fall back to im2col+GEMM.
    #[cfg(feature = "cudnn")]
    pub fn conv2d_cudnn(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Option<Self> {
        if !self.device().is_gpu() || !weight.device().is_gpu() {
            return None;
        }
        if let Some(b) = bias {
            if !b.device().is_gpu() {
                return None;
            }
        }

        let cuda = get_cuda_backend()?;
        let cudnn_handle = cuda.cudnn()?;

        let batch_size = self.shape[0];
        let in_channels = self.shape[1];
        let in_height = self.shape[2];
        let in_width = self.shape[3];
        let out_channels = weight.shape[0];
        let kernel_h = weight.shape[2];
        let kernel_w = weight.shape[3];
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        let out_h = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

        let input_contig = self.contiguous_gpu();
        let weight_contig = weight.contiguous_gpu();
        let input_guard = input_contig.storage.as_cuda_slice();
        let weight_guard = weight_contig.storage.as_cuda_slice();

        let bias_contig = bias.map(|b| b.contiguous_gpu());
        let bias_guard = bias_contig.as_ref().map(|b| b.storage.as_cuda_slice());

        let output_slice = axonml_core::backends::cudnn_ops::cudnn_conv2d_forward(
            cudnn_handle,
            cuda.stream(),
            cuda,
            input_guard.slice(),
            weight_guard.slice(),
            bias_guard.as_ref().map(|g| g.slice()),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        )?;

        let total_out = batch_size * out_channels * out_h * out_w;
        let out_shape = Shape::from_slice(&[batch_size, out_channels, out_h, out_w]);
        Some(Self {
            storage: Storage::from_cuda_slice(output_slice, total_out, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        })
    }

    /// GPU-resident grouped Conv2d forward (depthwise separable, etc.).
    ///
    /// Runs each group as a separate im2col + GEMM on GPU.
    /// `self` is `[N, C_in, H, W]` on GPU.
    /// `weight` is `[C_out, C_in/groups, kH, kW]` on GPU.
    /// `bias` is optional `[C_out]` on GPU.
    pub fn conv2d_grouped_cuda(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Option<Self> {
        // All tensors must be GPU-resident
        if !self.device().is_gpu() || !weight.device().is_gpu() {
            return None;
        }
        if let Some(b) = bias {
            if !b.device().is_gpu() {
                return None;
            }
        }
        let cuda = get_cuda_backend()?;

        let batch_size = self.shape[0];
        let in_channels = self.shape[1];
        let in_height = self.shape[2];
        let in_width = self.shape[3];
        let out_channels = weight.shape[0];
        let kernel_h = weight.shape[2];
        let kernel_w = weight.shape[3];
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        let in_channels_per_group = in_channels / groups;
        let out_channels_per_group = out_channels / groups;

        let out_h = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
        let col_h = in_channels_per_group * kernel_h * kernel_w;
        let col_w = out_h * out_w;
        let col_n = col_h * col_w;
        let spatial = out_h * out_w;
        let in_spatial = in_height * in_width;
        let out_per_batch = out_channels * spatial;

        let input_data = self.contiguous_gpu();
        let weight_data = weight.contiguous_gpu();

        let input_guard = input_data.storage.as_cuda_slice();
        let weight_guard = weight_data.storage.as_cuda_slice();

        // im2col params for per-group input (in_channels_per_group channels)
        let params_arr: [u32; 10] = [
            in_height as u32,
            in_width as u32,
            kernel_h as u32,
            kernel_w as u32,
            pad_h as u32,
            pad_w as u32,
            stride_h as u32,
            stride_w as u32,
            out_h as u32,
            out_w as u32,
        ];
        let params_gpu = cuda.htod_copy(&params_arr[..]).ok()?;

        let bias_data = bias.map(|b| b.contiguous_gpu());
        let bias_guard = bias_data.as_ref().map(|b| b.storage.as_cuda_slice());

        let mut col_gpu = pool_alloc(col_n).ok()?;
        let mut input_group_gpu = pool_alloc(in_channels_per_group * in_spatial).ok()?;
        let mut group_out_gpu = pool_alloc(out_channels_per_group * spatial).ok()?;

        let total_out = batch_size * out_per_batch;
        let mut out_gpu = pool_alloc(total_out).ok()?;

        for b in 0..batch_size {
            for g in 0..groups {
                let ic_start = g * in_channels_per_group;
                let oc_start = g * out_channels_per_group;

                // d2d copy: input channels for this group
                let in_group_size = in_channels_per_group * in_spatial;
                let in_offset = b * in_channels * in_spatial + ic_start * in_spatial;
                cuda.memcpy_dtod_f32(
                    &mut input_group_gpu,
                    0,
                    input_guard.slice(),
                    in_offset,
                    in_group_size,
                )
                .ok()?;

                // im2col on group input
                cuda.im2col_f32(&input_group_gpu, &mut col_gpu, &params_gpu, col_n)
                    .ok()?;

                // Weight for this group: offset into weight buffer
                let w_offset = oc_start * in_channels_per_group * kernel_h * kernel_w;
                let w_size = out_channels_per_group * col_h;

                // Copy group weight to contiguous buffer for GEMM
                let mut weight_group_gpu = pool_alloc(w_size).ok()?;
                cuda.memcpy_dtod_f32(
                    &mut weight_group_gpu,
                    0,
                    weight_guard.slice(),
                    w_offset,
                    w_size,
                )
                .ok()?;

                // GEMM: group_out = weight_group @ col
                cuda.gemm_f32(
                    false,
                    false,
                    col_w,
                    out_channels_per_group,
                    col_h,
                    1.0,
                    &col_gpu,
                    col_w,
                    &weight_group_gpu,
                    col_h,
                    0.0,
                    &mut group_out_gpu,
                    col_w,
                )
                .ok()?;

                // Bias add for this group's channels
                if let Some(ref bg) = bias_guard {
                    // Copy group bias
                    let mut bias_group = pool_alloc(out_channels_per_group).ok()?;
                    cuda.memcpy_dtod_f32(
                        &mut bias_group,
                        0,
                        bg.slice(),
                        oc_start,
                        out_channels_per_group,
                    )
                    .ok()?;
                    cuda.bias_add_channels_f32(
                        &mut group_out_gpu,
                        &bias_group,
                        spatial,
                        out_channels_per_group * spatial,
                    )
                    .ok()?;
                }

                // Copy group output into final buffer
                let out_offset = b * out_per_batch + oc_start * spatial;
                cuda.memcpy_dtod_f32(
                    &mut out_gpu,
                    out_offset,
                    &group_out_gpu,
                    0,
                    out_channels_per_group * spatial,
                )
                .ok()?;
            }
        }

        let out_shape = Shape::from_slice(&[batch_size, out_channels, out_h, out_w]);
        Some(Self {
            storage: Storage::from_cuda_slice(out_gpu, total_out, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        })
    }

    /// GPU-resident Conv2d backward: computes grad_input, grad_weight, and optionally grad_bias.
    ///
    /// `self` is `grad_output` `[N, C_out, H_out, W_out]` on GPU.
    /// `saved_input` is `[N, C_in, H_in, W_in]` on GPU.
    /// `saved_weight` is `[C_out, C_in, kH, kW]` on GPU.
    ///
    /// Returns `(grad_input, grad_weight, Option<grad_bias>)`, all GPU-resident.
    /// Groups=1 only. Returns `None` if any GPU operation fails.
    pub fn conv2d_backward_cuda(
        &self,
        saved_input: &Self,
        saved_weight: &Self,
        input_shape: &[usize],
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        has_bias: bool,
    ) -> Option<(Self, Self, Option<Self>)> {
        // All tensors must be GPU-resident
        if !self.device().is_gpu()
            || !saved_input.device().is_gpu()
            || !saved_weight.device().is_gpu()
        {
            return None;
        }
        let cuda = get_cuda_backend()?;

        let batch_size = input_shape[0];
        let in_h = input_shape[2];
        let in_w = input_shape[3];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let out_h = self.shape[2];
        let out_w = self.shape[3];
        let col_h = in_channels * kh * kw;
        let col_w = out_h * out_w;
        let col_n = col_h * col_w;
        let spatial = out_h * out_w;
        let in_per_batch = in_channels * in_h * in_w;
        let out_per_batch = out_channels * spatial;

        let grad_out_data = self.contiguous_gpu();
        let input_data = saved_input.contiguous_gpu();
        let weight_data = saved_weight.contiguous_gpu();

        let grad_out_guard = grad_out_data.storage.as_cuda_slice();
        let input_guard = input_data.storage.as_cuda_slice();
        let weight_guard = weight_data.storage.as_cuda_slice();

        // im2col/col2im params
        let params_arr: [u32; 10] = [
            in_h as u32,
            in_w as u32,
            kh as u32,
            kw as u32,
            ph as u32,
            pw as u32,
            sh as u32,
            sw as u32,
            out_h as u32,
            out_w as u32,
        ];
        let params_gpu = cuda.htod_copy(&params_arr[..]).ok()?;

        // Buffers
        let mut col_gpu = pool_alloc(col_n).ok()?;
        let mut grad_out_batch = pool_alloc(out_per_batch).ok()?;
        let mut input_batch = pool_alloc(in_per_batch).ok()?;

        // Accumulate grad_weight across batches on GPU
        let weight_n = out_channels * col_h;
        let mut grad_weight_gpu = pool_alloc(weight_n).ok()?;
        // Zero-init grad_weight
        let zeros_w = vec![0.0f32; weight_n];
        let zeros_gpu = cuda.htod_copy(&zeros_w).ok()?;
        cuda.memcpy_dtod_f32(&mut grad_weight_gpu, 0, &zeros_gpu, 0, weight_n)
            .ok()?;

        // grad_input buffer
        let total_input = batch_size * in_per_batch;
        let mut grad_input_gpu = pool_alloc(total_input).ok()?;

        // Zero buffer for per-batch grad_input init
        let mut zero_batch = pool_alloc(in_per_batch).ok()?;
        {
            let zeros_in = vec![0.0f32; in_per_batch];
            let z = cuda.htod_copy(&zeros_in).ok()?;
            cuda.memcpy_dtod_f32(&mut zero_batch, 0, &z, 0, in_per_batch)
                .ok()?;
        }

        // grad_bias accumulator (computed on GPU if needed)
        let mut grad_bias_gpu = if has_bias {
            let gb = pool_alloc(out_channels).ok()?;
            Some(gb)
        } else {
            None
        };
        // Zero-init grad_bias
        if let Some(ref mut gb) = grad_bias_gpu {
            let zeros_b = vec![0.0f32; out_channels];
            let zb = cuda.htod_copy(&zeros_b).ok()?;
            cuda.memcpy_dtod_f32(gb, 0, &zb, 0, out_channels).ok()?;
        }

        for b in 0..batch_size {
            // Copy grad_output for this batch
            cuda.memcpy_dtod_f32(
                &mut grad_out_batch,
                0,
                grad_out_guard.slice(),
                b * out_per_batch,
                out_per_batch,
            )
            .ok()?;

            // Copy input for this batch
            cuda.memcpy_dtod_f32(
                &mut input_batch,
                0,
                input_guard.slice(),
                b * in_per_batch,
                in_per_batch,
            )
            .ok()?;

            // === grad_input: col = weight^T @ grad_out, then col2im ===
            // weight: [out_channels, col_h] row-major
            // grad_out: [out_channels, spatial] row-major
            // col = weight^T @ grad_out → [col_h, spatial]
            //
            // cuBLAS: C^T(spatial, col_h) = grad_out^T(spatial, oc) @ weight(oc, col_h)
            // m=spatial, n=col_h, k=out_channels, transA=false, transB=false
            // But we need weight^T @ grad_out in row-major.
            // Row-major weight^T(col_h, oc) → col-major (oc, col_h)
            // We want: col(col_h, spatial) = weight^T(col_h, oc) @ grad_out(oc, spatial)
            // col-major: col^T(spatial, col_h) = grad_out^T(spatial, oc) @ weight(oc, col_h)
            // m=spatial, n=col_h, k=oc
            // col = weight^T @ grad_out → [col_h, spatial]
            // Row-major: weight(oc, col_h), grad_out(oc, spatial)
            // cuBLAS col-major: C^T(spatial, col_h) = grad_out^T(spatial, oc) @ weight(oc, col_h)
            // m=spatial, n=col_h, k=oc, lda=spatial, ldb=out_channels (NOT col_h!), ldc=spatial
            cuda.gemm_f32(
                false,
                false,
                spatial,
                col_h,
                out_channels,
                1.0,
                &grad_out_batch,
                spatial,
                weight_guard.slice(),
                out_channels, // ldb = out_channels (leading dim of weight in col-major view)
                0.0,
                &mut col_gpu,
                spatial,
            )
            .ok()?;

            // Zero the per-batch grad_input region
            let gi_offset = b * in_per_batch;
            cuda.memcpy_dtod_f32(&mut grad_input_gpu, gi_offset, &zero_batch, 0, in_per_batch)
                .ok()?;

            // col2im: col [col_h, spatial] → grad_input[b] [C_in, H, W]
            // We need to write to grad_input_gpu at offset gi_offset.
            // col2im kernel writes to output starting at base pointer.
            // Use a temporary buffer, then d2d copy back.
            let mut gi_batch = pool_alloc(in_per_batch).ok()?;
            cuda.memcpy_dtod_f32(&mut gi_batch, 0, &zero_batch, 0, in_per_batch)
                .ok()?;

            cuda.col2im_f32(&col_gpu, &mut gi_batch, &params_gpu, col_n)
                .ok()?;

            cuda.memcpy_dtod_f32(&mut grad_input_gpu, gi_offset, &gi_batch, 0, in_per_batch)
                .ok()?;

            // === grad_weight: grad_weight += grad_out @ col^T ===
            // im2col input for this batch
            cuda.im2col_f32(&input_batch, &mut col_gpu, &params_gpu, col_n)
                .ok()?;

            // grad_out: [oc, spatial] row-major
            // col: [col_h, spatial] row-major
            // grad_weight += grad_out @ col^T → [oc, col_h]
            //
            // cuBLAS: gw^T(col_h, oc) = col(col_h, spatial) @ grad_out^T(spatial, oc)
            // But in row-major → col-major mapping:
            // gw^T(col_h, oc) = col_cm @ grad_out_cm
            // m=col_h, n=oc, k=spatial, beta=1.0 to accumulate
            cuda.gemm_f32(
                true,
                false,
                col_h,
                out_channels,
                spatial,
                1.0,
                &col_gpu,
                spatial,
                &grad_out_batch,
                spatial,
                1.0,
                &mut grad_weight_gpu,
                col_h,
            )
            .ok()?;

            // === grad_bias: bias_grad += sum over spatial of grad_out ===
            if let Some(ref mut gb) = grad_bias_gpu {
                // Sum each channel's spatial values using GEMM:
                // grad_out [oc, spatial] @ ones[spatial, 1] → [oc, 1]
                // This is oc dot products.
                // But we don't have a ones vector... use a simple CPU fallback for bias.
                // Bias grad is tiny (just out_channels values), not worth a custom kernel.
                let go_cpu = cuda.dtoh_copy(&grad_out_batch).ok()?;
                let mut bias_acc = cuda.dtoh_copy(gb).ok()?;
                for oc in 0..out_channels {
                    let mut sum = 0.0f32;
                    for s in 0..spatial {
                        sum += go_cpu[oc * spatial + s];
                    }
                    bias_acc[oc] += sum;
                }
                let ba_gpu = cuda.htod_copy(&bias_acc).ok()?;
                cuda.memcpy_dtod_f32(gb, 0, &ba_gpu, 0, out_channels).ok()?;
            }
        }

        // Build output tensors
        let gi_shape = Shape::from_slice(input_shape);
        let grad_input_t = Self {
            storage: Storage::from_cuda_slice(grad_input_gpu, total_input, self.device()),
            shape: gi_shape.clone(),
            strides: contiguous_strides(&gi_shape),
            offset: 0,
        };

        let gw_shape = Shape::from_slice(&[out_channels, in_channels, kh, kw]);
        let grad_weight_t = Self {
            storage: Storage::from_cuda_slice(grad_weight_gpu, weight_n, self.device()),
            shape: gw_shape.clone(),
            strides: contiguous_strides(&gw_shape),
            offset: 0,
        };

        let grad_bias_t = grad_bias_gpu.map(|gb| {
            let gb_shape = Shape::from_slice(&[out_channels]);
            Self {
                storage: Storage::from_cuda_slice(gb, out_channels, self.device()),
                shape: gb_shape.clone(),
                strides: contiguous_strides(&gb_shape),
                offset: 0,
            }
        });

        Some((grad_input_t, grad_weight_t, grad_bias_t))
    }

    // =========================================================================
    // Pooling Operations (GPU)
    // =========================================================================

    /// GPU MaxPool2d forward. Input must be [N, C, H, W] on GPU.
    /// Returns (output_tensor, indices_vec) where indices are flat i32 offsets.
    /// Output tensor stays on GPU.
    pub fn maxpool2d_cuda(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Option<(Self, Vec<i32>)> {
        if !self.device().is_gpu() {
            return None;
        }
        let cuda = get_cuda_backend()?;

        let batch = self.shape[0];
        let channels = self.shape[1];
        let in_h = self.shape[2];
        let in_w = self.shape[3];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;

        let out_h = (in_h + 2 * ph - kh) / sh + 1;
        let out_w = (in_w + 2 * pw - kw) / sw + 1;
        let total = batch * channels * out_h * out_w;

        // Ensure contiguous on GPU
        let input_data = self.contiguous_gpu();
        let input_guard = input_data.storage.as_cuda_slice();

        // Upload params
        let params: [u32; 8] = [
            in_h as u32,
            in_w as u32,
            kh as u32,
            kw as u32,
            sh as u32,
            sw as u32,
            ph as u32,
            pw as u32,
        ];
        let params_gpu = cuda.htod_copy(&params[..]).ok()?;

        // Allocate output + indices on GPU
        let mut output_gpu = pool_alloc(total).ok()?;
        let mut indices_gpu = cuda.alloc::<i32>(total).ok()?;

        cuda.maxpool2d_fwd_f32(
            input_guard.slice(),
            &mut output_gpu,
            &mut indices_gpu,
            &params_gpu,
            channels,
            out_h,
            out_w,
            total,
        )
        .ok()?;

        // Download indices to CPU (needed for backward bookkeeping)
        let indices = cuda.dtoh_copy(&indices_gpu).ok()?;

        let out_shape = Shape::from_slice(&[batch, channels, out_h, out_w]);
        let output = Self {
            storage: Storage::from_cuda_slice(output_gpu, total, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        };

        Some((output, indices))
    }

    /// GPU AvgPool2d forward. Input must be [N, C, H, W] on GPU.
    /// Output tensor stays on GPU.
    pub fn avgpool2d_cuda(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Option<Self> {
        if !self.device().is_gpu() {
            return None;
        }
        let cuda = get_cuda_backend()?;

        let batch = self.shape[0];
        let channels = self.shape[1];
        let in_h = self.shape[2];
        let in_w = self.shape[3];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;

        let out_h = (in_h + 2 * ph - kh) / sh + 1;
        let out_w = (in_w + 2 * pw - kw) / sw + 1;
        let total = batch * channels * out_h * out_w;

        let input_data = self.contiguous_gpu();
        let input_guard = input_data.storage.as_cuda_slice();

        let params: [u32; 9] = [
            in_h as u32,
            in_w as u32,
            kh as u32,
            kw as u32,
            sh as u32,
            sw as u32,
            ph as u32,
            pw as u32,
            count_include_pad as u32,
        ];
        let params_gpu = cuda.htod_copy(&params[..]).ok()?;

        let mut output_gpu = pool_alloc(total).ok()?;

        cuda.avgpool2d_fwd_f32(
            input_guard.slice(),
            &mut output_gpu,
            &params_gpu,
            channels,
            out_h,
            out_w,
            total,
        )
        .ok()?;

        let out_shape = Shape::from_slice(&[batch, channels, out_h, out_w]);
        Some(Self {
            storage: Storage::from_cuda_slice(output_gpu, total, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        })
    }

    // =========================================================================
    // Fused Scaled Dot-Product Attention
    // =========================================================================

    /// Fused attention on GPU: computes softmax(Q @ K^T * scale) @ V
    /// without materializing the full N*N attention matrix in global memory.
    ///
    /// - `self` (Q): [B, H, Tq, D]
    /// - `k`: [B, H, Tk, D]
    /// - `v`: [B, H, Tk, D]
    /// - Returns output [B, H, Tq, D] on GPU
    ///
    /// `is_causal`: if true, applies causal mask (positions j > i are masked out).
    ///
    /// For very long sequences (>2048), consider the CPU tiled Flash Attention
    /// in axonml-llm which uses online softmax with O(N) memory.
    pub fn fused_attention_cuda(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        is_causal: bool,
    ) -> Option<Self> {
        let cuda = get_cuda_backend()?;

        let q_shape = self.shape();
        assert!(q_shape.len() == 4, "Q must be [B, H, Tq, D]");
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let tgt_len = q_shape[2];
        let head_dim = q_shape[3];
        let src_len = k.shape()[2];

        let total_out = batch_size * num_heads * tgt_len * head_dim;

        let q_contig = self.contiguous_gpu();
        let k_contig = k.contiguous_gpu();
        let v_contig = v.contiguous_gpu();

        let q_guard = q_contig.storage.as_cuda_slice();
        let k_guard = k_contig.storage.as_cuda_slice();
        let v_guard = v_contig.storage.as_cuda_slice();

        let mut out_gpu = pool_alloc(total_out).ok()?;

        cuda.fused_attention_fwd_f32(
            q_guard.slice(),
            k_guard.slice(),
            v_guard.slice(),
            &mut out_gpu,
            scale,
            batch_size,
            num_heads,
            tgt_len,
            src_len,
            head_dim,
            is_causal,
        )
        .ok()?;

        let out_shape = Shape::from_slice(&[batch_size, num_heads, tgt_len, head_dim]);
        Some(Self {
            storage: Storage::from_cuda_slice(out_gpu, total_out, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        })
    }

    /// Fused attention backward on GPU: computes grad_Q, grad_K, grad_V by
    /// recomputing attention weights from Q, K, O without storing the N*N matrix.
    ///
    /// - `self` (Q): [B, H, Tq, D]
    /// - `k`: [B, H, Tk, D]
    /// - `v`: [B, H, Tk, D]
    /// - `output`: [B, H, Tq, D]  (forward output)
    /// - `grad_output`: [B, H, Tq, D]
    /// - Returns (grad_Q, grad_K, grad_V) on GPU, or None if kernel unavailable
    pub fn fused_attention_bwd_cuda(
        &self,
        k: &Self,
        v: &Self,
        output: &Self,
        grad_output: &Self,
        scale: f32,
        is_causal: bool,
    ) -> Option<(Self, Self, Self)> {
        let cuda = get_cuda_backend()?;

        let q_shape = self.shape();
        assert!(q_shape.len() == 4, "Q must be [B, H, Tq, D]");
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let tgt_len = q_shape[2];
        let head_dim = q_shape[3];
        let src_len = k.shape()[2];

        let total_q = batch_size * num_heads * tgt_len * head_dim;
        let total_kv = batch_size * num_heads * src_len * head_dim;

        let q_contig = self.contiguous_gpu();
        let k_contig = k.contiguous_gpu();
        let v_contig = v.contiguous_gpu();
        let o_contig = output.contiguous_gpu();
        let go_contig = grad_output.contiguous_gpu();

        let q_guard = q_contig.storage.as_cuda_slice();
        let k_guard = k_contig.storage.as_cuda_slice();
        let v_guard = v_contig.storage.as_cuda_slice();
        let o_guard = o_contig.storage.as_cuda_slice();
        let go_guard = go_contig.storage.as_cuda_slice();

        // Zero-initialized output buffers. pool_alloc zeros on-GPU via
        // cuMemsetD8Async (no CPU alloc / no PCIe H2D). The kernel accumulates
        // into these buffers, so zero-init is required.
        let mut gq_gpu = pool_alloc(total_q).ok()?;
        let mut gk_gpu = pool_alloc(total_kv).ok()?;
        let mut gv_gpu = pool_alloc(total_kv).ok()?;

        cuda.fused_attention_bwd_f32(
            q_guard.slice(),
            k_guard.slice(),
            v_guard.slice(),
            o_guard.slice(),
            go_guard.slice(),
            &mut gq_gpu,
            &mut gk_gpu,
            &mut gv_gpu,
            scale,
            batch_size,
            num_heads,
            tgt_len,
            src_len,
            head_dim,
            is_causal,
        )
        .ok()?;

        let q_out_shape = Shape::from_slice(&[batch_size, num_heads, tgt_len, head_dim]);
        let kv_out_shape = Shape::from_slice(&[batch_size, num_heads, src_len, head_dim]);

        let grad_q = Self {
            storage: Storage::from_cuda_slice(gq_gpu, total_q, self.device()),
            shape: q_out_shape.clone(),
            strides: contiguous_strides(&q_out_shape),
            offset: 0,
        };
        let grad_k = Self {
            storage: Storage::from_cuda_slice(gk_gpu, total_kv, self.device()),
            shape: kv_out_shape.clone(),
            strides: contiguous_strides(&kv_out_shape),
            offset: 0,
        };
        let grad_v = Self {
            storage: Storage::from_cuda_slice(gv_gpu, total_kv, self.device()),
            shape: kv_out_shape.clone(),
            strides: contiguous_strides(&kv_out_shape),
            offset: 0,
        };

        Some((grad_q, grad_k, grad_v))
    }

    // =========================================================================
    // Transformer Per-Layer Ops (GPU)
    //
    // Decode-step kernels added in axonml-core/cuda_kernels/transformer_ops.cu.
    // These let nexus-serve keep activations on the device through the whole
    // layer instead of round-tripping CPU↔GPU after every matmul.
    // =========================================================================

    /// GPU RMSNorm with a per-element weight scale.
    /// Input shape `[n]` (single token); weight shape `[n]`.
    /// Qwen3 QK-norm: per-head RMS_norm over the last `head_dim` axis.
    /// `self` is `[n_heads * head_dim]`; `weight` is `[head_dim]`
    /// broadcast across all heads. Returns a new tensor with the norm
    /// applied. Kernel operates in place on a fresh copy of self.
    pub(crate) fn rms_norm_heads_cuda(
        &self,
        weight: &Self,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Self {
        let data = self.contiguous_gpu();
        let w = weight.contiguous_gpu();
        debug_assert_eq!(
            data.numel(),
            n_heads * head_dim,
            "rms_norm_heads: tensor must be [n_heads * head_dim]"
        );
        debug_assert_eq!(
            w.numel(),
            head_dim,
            "rms_norm_heads: weight must be [head_dim]"
        );
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let w_guard = w.storage.as_cuda_slice();
        // pool_alloc_uninit: rms_norm_heads_f32 writes every output element
        // via the per-lane normalize+multiply store loop.
        let mut out = pool_alloc_uninit(data.numel()).expect("GPU pool alloc failed");

        // Kernel reads from `src`, writes to `out` — no broadcast_copy prep
        // needed. The sum-of-squares reduction completes before any write
        // to `out`, so src/out aliasing would be safe if we wanted it.
        cuda.rms_norm_heads_f32(
            &mut out,
            src_guard.slice(),
            w_guard.slice(),
            n_heads,
            head_dim,
            eps,
        )
        .expect("CUDA rms_norm_heads_f32 failed");

        let storage = Storage::from_cuda_slice(out, data.numel(), self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    pub(crate) fn rms_norm_cuda(&self, weight: &Self, eps: f32) -> Self {
        let data = self.contiguous_gpu();
        let w = weight.contiguous_gpu();
        let len = data.numel();
        debug_assert_eq!(len, w.numel(), "rms_norm: weight length must match input");
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let w_guard = w.storage.as_cuda_slice();
        // pool_alloc_uninit: rms_norm_f32 writes every out[i] = scale*x[i]*w[i].
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

        cuda.rms_norm_f32(&mut out, src_guard.slice(), w_guard.slice(), len, eps)
            .expect("CUDA rms_norm_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Single-token LayerNorm on GPU:
    /// `out[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]`.
    /// Used by legacy Falcon's decode path.
    pub(crate) fn layer_norm_tokenwise_cuda(&self, gamma: &Self, beta: &Self, eps: f32) -> Self {
        let data = self.contiguous_gpu();
        let g = gamma.contiguous_gpu();
        let b = beta.contiguous_gpu();
        let len = data.numel();
        debug_assert_eq!(len, g.numel(), "layer_norm: gamma length must match input");
        debug_assert_eq!(len, b.numel(), "layer_norm: beta length must match input");
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let g_guard = g.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

        cuda.layer_norm_tokenwise_f32(
            &mut out,
            src_guard.slice(),
            g_guard.slice(),
            b_guard.slice(),
            len,
            eps,
        )
        .expect("CUDA layer_norm_tokenwise_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Element-wise tanh-approximation GELU on GPU. Returns a new tensor.
    pub(crate) fn gelu_tanh_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

        cuda.gelu_tanh_f32(&mut out, src_guard.slice(), len)
            .expect("CUDA gelu_tanh_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// In-place `self += other * scalar`. Both tensors on the same GPU,
    /// same numel. Fuses a `mul_scalar(...)` + `add(...)` kernel pair
    /// into one launch — MoE expert-accumulate hot path.
    pub(crate) fn scaled_add_inplace_cuda_(&mut self, other: &Self, scalar: f32) {
        debug_assert_eq!(
            self.numel(),
            other.numel(),
            "scaled_add_inplace: numel mismatch"
        );
        let o = other.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        if !self.is_contiguous() {
            *self = self.contiguous();
        }
        let o_guard = o.storage.as_cuda_slice();
        let mut self_guard = self.storage.as_cuda_slice_mut();
        cuda.scaled_add_inplace_f32(
            self_guard.slice_mut(),
            o_guard.slice(),
            self.numel(),
            scalar,
        )
        .expect("CUDA scaled_add_inplace_f32 failed");
    }

    /// Parallel-residual in-place update for Falcon: `self += attn + ffn`.
    /// Fuses two element-wise adds into one kernel launch. All three
    /// tensors must be on the same GPU device and have the same numel.
    pub(crate) fn parallel_residual_add_cuda_(&mut self, attn: &Self, ffn: &Self) {
        debug_assert_eq!(
            self.numel(),
            attn.numel(),
            "parallel_residual_add: attn numel mismatch"
        );
        debug_assert_eq!(
            self.numel(),
            ffn.numel(),
            "parallel_residual_add: ffn numel mismatch"
        );
        let a = attn.contiguous_gpu();
        let f = ffn.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        // `self` must be contiguous on GPU for the in-place write to be
        // correct. Materialize a contiguous copy if needed (rare — decode
        // tensors are built contiguous by earlier kernels).
        if !self.is_contiguous() {
            *self = self.contiguous();
        }

        let a_guard = a.storage.as_cuda_slice();
        let f_guard = f.storage.as_cuda_slice();
        let mut self_guard = self.storage.as_cuda_slice_mut();
        cuda.parallel_residual_add_f32(
            self_guard.slice_mut(),
            a_guard.slice(),
            f_guard.slice(),
            self.numel(),
        )
        .expect("CUDA parallel_residual_add_f32 failed");
    }

    /// GPU RoPE in the LLaMA / Qwen / Mistral split-halves layout. Returns
    /// a new tensor with the rotation applied; original is unchanged.
    /// Input shape `[n_heads * head_dim]` (single token, all heads flattened).
    pub(crate) fn rope_split_halves_cuda(
        &self,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos: usize,
    ) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        debug_assert_eq!(
            len,
            n_heads * head_dim,
            "rope: tensor length must equal n_heads * head_dim"
        );
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        // pool_alloc_uninit: rope kernel writes every output element (both
        // halves of every pair) via the per-thread store.
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

        // Kernel reads from `src`, writes to `out` — no broadcast_copy prep
        // needed. Each thread reads src[base] + src[base+half] BEFORE any
        // write, so src/out aliasing is safe (fresh buffer is for new-tensor
        // semantics, not correctness).
        cuda.rope_split_halves_f32(&mut out, src_guard.slice(), n_heads, head_dim, theta, pos)
            .expect("CUDA rope_split_halves_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU fused SwiGLU: `out[i] = SiLU(self[i]) * up[i]`. `self` is the gate.
    pub(crate) fn swiglu_cuda(&self, up: &Self) -> Self {
        let g = self.contiguous_gpu();
        let u = up.contiguous_gpu();
        let len = g.numel();
        debug_assert_eq!(len, u.numel(), "swiglu: gate and up must be same length");
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let g_guard = g.storage.as_cuda_slice();
        let u_guard = u.storage.as_cuda_slice();
        // pool_alloc_uninit: swiglu_f32 writes every out[i] = silu(g[i])*u[i].
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

        cuda.swiglu_f32(&mut out, g_guard.slice(), u_guard.slice(), len)
            .expect("CUDA swiglu_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GPU BitNet b1.58 fused gate: `out[i] = ReLU(self[i])² * up[i]`.
    /// `self` is the gate.
    pub(crate) fn relu2_gate_cuda(&self, up: &Self) -> Self {
        let g = self.contiguous_gpu();
        let u = up.contiguous_gpu();
        let len = g.numel();
        debug_assert_eq!(
            len,
            u.numel(),
            "relu2_gate: gate and up must be same length"
        );
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let g_guard = g.storage.as_cuda_slice();
        let u_guard = u.storage.as_cuda_slice();
        // pool_alloc_uninit: relu2_gate_f32 writes every out[i] = relu(g)²*u.
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

        cuda.relu2_gate_f32(&mut out, g_guard.slice(), u_guard.slice(), len)
            .expect("CUDA relu2_gate_f32 failed");

        let storage = Storage::from_cuda_slice(out, len, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Fused causal-scaled softmax. `self` is the raw attention scores with
    /// shape `[B, H, Tq, Tk]` or any leading-dims + `[Tq, Tk]` layout; the
    /// kernel treats it as `[B*H*Tq, Tk]` flattened. Returns a tensor of the
    /// same shape, with masked positions exactly 0.
    pub(crate) fn softmax_causal_scaled_cuda(
        &self,
        tq: usize,
        tk: usize,
        offset: usize,
        scale: f32,
    ) -> Self {
        let data = self.contiguous_gpu();
        let total = data.numel();
        debug_assert!(
            total % tk == 0,
            "softmax_causal_scaled: tk must divide numel"
        );
        let num_rows = total / tk;
        debug_assert!(
            num_rows % tq == 0,
            "softmax_causal_scaled: tq must divide num_rows"
        );
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        // Kernel writes every output position (mask-off → 0, in-bounds → softmax).
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.softmax_causal_scaled_f32(
            &mut out,
            src_guard.slice(),
            num_rows,
            tq,
            tk,
            offset,
            scale,
        )
        .expect("CUDA softmax_causal_scaled_f32 failed");

        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Fused causal-scaled softmax backward wrt raw scores. `self` is the
    /// saved forward output `p` (masked positions are 0); `grad_output`
    /// matches its shape. Returns `grad_scores` = `scale * p * (grad_out - Σ(p·grad_out))`.
    pub(crate) fn softmax_causal_scaled_bwd_cuda(
        &self,
        grad_output: &Self,
        tk: usize,
        scale: f32,
    ) -> Self {
        let p = self.contiguous_gpu();
        let g = grad_output.contiguous_gpu();
        let total = p.numel();
        debug_assert_eq!(total, g.numel());
        debug_assert!(total % tk == 0);
        let num_rows = total / tk;
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let p_guard = p.storage.as_cuda_slice();
        let g_guard = g.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.softmax_causal_scaled_bwd_f32(
            &mut out,
            p_guard.slice(),
            g_guard.slice(),
            num_rows,
            tk,
            scale,
        )
        .expect("CUDA softmax_causal_scaled_bwd_f32 failed");

        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Batched RMSNorm backward — computes grad_input only (the current
    /// autograd path treats RMSNorm weight as a frozen parameter, matching
    /// the CPU-only RMSNormBackward in axonml-llm). `self` = saved_input
    /// `[m, n]`, `weight` `[n]`, `grad_output` `[m, n]` — all on GPU.
    pub(crate) fn rms_norm_bwd_batched_cuda(
        &self,
        weight: &Self,
        grad_output: &Self,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Self {
        let x = self.contiguous_gpu();
        let w = weight.contiguous_gpu();
        let g = grad_output.contiguous_gpu();
        debug_assert_eq!(x.numel(), m * n);
        debug_assert_eq!(w.numel(), n);
        debug_assert_eq!(g.numel(), m * n);
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let x_guard = x.storage.as_cuda_slice();
        let w_guard = w.storage.as_cuda_slice();
        let g_guard = g.storage.as_cuda_slice();
        // Kernel writes every grad_input element — uninit safe.
        let mut out = pool_alloc_uninit(m * n).expect("GPU pool alloc failed");

        cuda.rms_norm_bwd_batched_f32(
            &mut out,
            x_guard.slice(),
            w_guard.slice(),
            g_guard.slice(),
            m,
            n,
            eps,
        )
        .expect("CUDA rms_norm_bwd_batched_f32 failed");

        let storage = Storage::from_cuda_slice(out, m * n, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Batched RMSNorm over `m` tokens. `self` must be `[m, n]` contiguous
    /// on GPU; `weight` is `[n]` on GPU. Returns `[m, n]`.
    pub(crate) fn rms_norm_batched_cuda(
        &self,
        weight: &Self,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Self {
        let data = self.contiguous_gpu();
        let w = weight.contiguous_gpu();
        debug_assert_eq!(
            data.numel(),
            m * n,
            "rms_norm_batched: expected m*n elements"
        );
        debug_assert_eq!(w.numel(), n, "rms_norm_batched: weight must be [n]");
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let w_guard = w.storage.as_cuda_slice();
        // pool_alloc_uninit: kernel writes every out[i].
        let mut out = pool_alloc_uninit(m * n).expect("GPU pool alloc failed");

        cuda.rms_norm_batched_f32(&mut out, src_guard.slice(), w_guard.slice(), m, n, eps)
            .expect("CUDA rms_norm_batched_f32 failed");

        let storage = Storage::from_cuda_slice(out, m * n, self.device());
        Self {
            storage,
            shape: vec![m, n].into(),
            strides: contiguous_strides(&[m, n]),
            offset: 0,
        }
    }

    /// Batched per-head RMSNorm (Qwen3 QK-norm) over `m` tokens. `self`
    /// must be `[m, n_heads * head_dim]` contiguous GPU; `weight` is
    /// `[head_dim]`. Returns a new tensor with the norm applied.
    pub(crate) fn rms_norm_heads_batched_cuda(
        &self,
        weight: &Self,
        m: usize,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Self {
        let data = self.contiguous_gpu();
        let w = weight.contiguous_gpu();
        let total = m * n_heads * head_dim;
        debug_assert_eq!(
            data.numel(),
            total,
            "rms_norm_heads_batched: shape mismatch"
        );
        debug_assert_eq!(
            w.numel(),
            head_dim,
            "rms_norm_heads_batched: weight must be [head_dim]"
        );
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let w_guard = w.storage.as_cuda_slice();
        // pool_alloc_uninit: rms_norm_heads_batched_f32 writes every output
        // element via the per-lane normalize+multiply store loop.
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.rms_norm_heads_batched_f32(
            &mut out,
            src_guard.slice(),
            w_guard.slice(),
            m,
            n_heads,
            head_dim,
            eps,
        )
        .expect("CUDA rms_norm_heads_batched_f32 failed");

        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Batched split-halves RoPE. `self` must be `[m, n_heads * head_dim]`
    /// contiguous GPU. Rotates token `t` at position `(pos_start + t)`.
    pub(crate) fn apply_rope_split_halves_batched_cuda(
        &self,
        m: usize,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Self {
        let data = self.contiguous_gpu();
        let total = m * n_heads * head_dim;
        debug_assert_eq!(data.numel(), total, "rope_batched: shape mismatch");
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        // pool_alloc_uninit: rope kernel writes every output pair via the
        // per-thread store (both halves covered).
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.rope_split_halves_batched_f32(
            &mut out,
            src_guard.slice(),
            m,
            n_heads,
            head_dim,
            theta,
            pos_start,
        )
        .expect("CUDA rope_split_halves_batched_f32 failed");

        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Broadcast per-column bias add for a `[m, n]` tensor. Consumes a
    /// fresh copy — callers that already own a unique buffer should use
    /// the in-place backend call directly.
    pub(crate) fn add_bias_batched_cuda(&self, bias: &Self, m: usize, n: usize) -> Self {
        let data = self.contiguous_gpu();
        let b = bias.contiguous_gpu();
        debug_assert_eq!(data.numel(), m * n, "add_bias_batched: shape mismatch");
        debug_assert_eq!(b.numel(), n, "add_bias_batched: bias must be [n]");
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let b_guard = b.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(m * n).expect("GPU pool alloc failed");

        // Copy src -> out, then add bias in place (kernel reads+writes).
        cuda.broadcast_copy_f32(&mut out, src_guard.slice(), m * n, m * n)
            .expect("CUDA broadcast_copy_f32 failed");
        cuda.add_bias_batched_f32(&mut out, b_guard.slice(), m, n)
            .expect("CUDA add_bias_batched_f32 failed");

        let storage = Storage::from_cuda_slice(out, m * n, self.device());
        Self {
            storage,
            shape: vec![m, n].into(),
            strides: contiguous_strides(&[m, n]),
            offset: 0,
        }
    }
}
