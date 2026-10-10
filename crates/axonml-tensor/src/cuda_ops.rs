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

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::alloc_prelude::*;

#[cfg(feature = "cuda")]
use axonml_core::Device;

#[cfg(feature = "cuda")]
use axonml_core::backends::cuda::get_cuda_backend;

#[cfg(feature = "cuda")]
use axonml_core::backends::cuda_pool::{pool_alloc, pool_alloc_uninit, pool_free};

#[cfg(feature = "cuda")]
use axonml_core::error::Result;

#[cfg(feature = "cuda")]
use axonml_core::storage::Storage;

#[cfg(feature = "cuda")]
use crate::shape::{Shape, contiguous_strides};

#[cfg(feature = "cuda")]
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
/// `AXONML_NO_DIRECT_DEPTHWISE=1` forces true-depthwise convs back through the grouped im2col+GEMM
/// path. Escape hatch and A/B handle for the direct kernels; unset in normal use.
fn no_direct_depthwise() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AXONML_NO_DIRECT_DEPTHWISE").is_ok())
}

/// An instantiated decode graph that a sequence state may own and carry across
/// worker threads. cudarc's [`CudaGraph`](cudarc::driver::CudaGraph) holds raw
/// driver handles and is not `Send`; the old `u64` handles were, implicitly.
///
/// SAFETY of the `Send` impl: a graph is only ever launched through
/// [`Tensor::graph_launch`] on the backend's single decode stream, from whichever
/// thread currently holds the sequence — never from two threads at once — and the
/// driver handles carry no thread affinity. Dropping it destroys exec + graph.
pub struct ReplayGraph(pub cudarc::driver::CudaGraph);
// SAFETY: see the type-level note — exclusive ownership, one stream, no affinity.
unsafe impl Send for ReplayGraph {}

impl Tensor<f32> {
    /// Per-output-channel LSQ fake-quant forward on GPU (QAT): returns a tensor of
    /// the same shape whose values are `clamp(round(x/s_c), qn, qp) * s_c`, channel
    /// `c = (i/stride) % ch`. `scale` is a `[ch]` GPU tensor. No CPU round-trip.
    #[allow(clippy::too_many_arguments)]
    pub fn fake_quant_pc_fwd_cuda(
        &self,
        scale: &Self,
        qn: f32,
        qp: f32,
        ch: usize,
        stride: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "fake_quant_pc_fwd_cuda: self must be on GPU"
        );
        let data = self.contiguous_gpu();
        let n = data.numel();
        let sc = scale.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let in_guard = data.storage.as_cuda_slice();
        let sc_guard = sc.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(n).expect("GPU pool alloc failed");
        cuda.fake_quant_pc_fwd_f32(
            in_guard.slice(),
            sc_guard.slice(),
            &mut out,
            qn,
            qp,
            ch as u32,
            stride as u32,
            n,
        )
        .map_err(|e| axonml_core::error::Error::InvalidOperation {
            message: format!("fake_quant_pc_fwd_f32 failed: {e}"),
        })?;
        let shape = data.shape.clone();
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, n, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Per-output-channel LSQ fake-quant backward on GPU (QAT): returns
    /// `(grad_x, gs_contrib)` — the STE input gradient and the per-element
    /// step-size gradient contribution. Folded from the vendored AxonML core.
    #[allow(clippy::too_many_arguments)]
    pub fn fake_quant_pc_bwd_cuda(
        &self,
        scale: &Self,
        grad_output: &Self,
        qn: f32,
        qp: f32,
        ch: usize,
        stride: usize,
    ) -> Result<(Self, Self)> {
        assert!(
            self.device().is_gpu(),
            "fake_quant_pc_bwd_cuda: self must be on GPU"
        );
        let dev = self.device();
        let data = self.contiguous_gpu();
        let n = data.numel();
        let scale_g = if scale.device().is_gpu() {
            scale.clone()
        } else {
            scale.to_device(dev).unwrap_or_else(|_| scale.clone())
        };
        let go_in = if grad_output.device().is_gpu() {
            grad_output.clone()
        } else {
            grad_output
                .to_device(dev)
                .unwrap_or_else(|_| grad_output.clone())
        };
        let sc = scale_g.contiguous_gpu();
        let go = go_in.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let in_guard = data.storage.as_cuda_slice();
        let sc_guard = sc.storage.as_cuda_slice();
        let go_guard = go.storage.as_cuda_slice();
        let mut gx = pool_alloc_uninit(n).expect("GPU pool alloc failed");
        let mut gs = pool_alloc_uninit(n).expect("GPU pool alloc failed");
        cuda.fake_quant_pc_bwd_f32(
            in_guard.slice(),
            sc_guard.slice(),
            go_guard.slice(),
            &mut gx,
            &mut gs,
            qn,
            qp,
            ch as u32,
            stride as u32,
            n,
        )
        .map_err(|e| axonml_core::error::Error::InvalidOperation {
            message: format!("fake_quant_pc_bwd_f32 failed: {e}"),
        })?;
        let shape = data.shape.clone();
        let strides = contiguous_strides(&shape);
        let grad_x = Self {
            storage: Storage::from_cuda_slice(gx, n, self.device()),
            shape: shape.clone(),
            strides: strides.clone(),
            offset: 0,
        };
        let gs_contrib = Self {
            storage: Storage::from_cuda_slice(gs, n, self.device()),
            shape,
            strides,
            offset: 0,
        };
        Ok((grad_x, gs_contrib))
    }

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
            if a_n == out_n {
                cuda.broadcast_add_f32(&mut out, a_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_add_f32 failed");
            } else {
                let a_bcast = a.broadcast_to(result_shape.as_slice()).contiguous_gpu();
                let a2_guard = a_bcast.storage.as_cuda_slice();
                cuda.broadcast_add_f32(&mut out, a2_guard.slice(), b_guard.slice(), out_n, b_n)
                    .expect("CUDA broadcast_add_f32 failed");
            }
        } else {
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

        let d = if dim < 0 { ndim as i32 + dim } else { dim } as usize;

        if d == ndim - 1 {
            let row_size = data.shape[ndim - 1];
            let num_rows = total / row_size;
            let cuda = get_cuda_backend().expect("CUDA backend not available");

            let src_guard = data.storage.as_cuda_slice();
            let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

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
            let mut perm: Vec<usize> = (0..ndim).collect();
            perm.swap(d, ndim - 1);
            let transposed = data.permute(&perm)?;
            let t_contig = transposed.contiguous_gpu();
            let t_result = t_contig.softmax_cuda(ndim as i32 - 1)?;
            Ok(t_result.permute(&perm)?.contiguous_gpu())
        }
    }

    /// GPU broadcast_to — fully on-device using broadcast_copy kernel.
    pub(crate) fn broadcast_to_cuda(&self, target_shape: &[usize]) -> Result<Self> {
        let data = self.contiguous_gpu();
        let src_len = data.numel();
        let out_len = crate::shape::numel(target_shape);
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        if out_len % src_len == 0 {
            let src_guard = data.storage.as_cuda_slice();
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
        let mut out = pool_alloc_uninit(m * out_dim).expect("GPU pool alloc failed");

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

    /// Q5_0 dequant-in-shader GEMM on the GPU.
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

    /// Q5_1 dequant-in-shader GEMV on the GPU (per-token decode path).
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

    /// Q5_1 dequant-in-shader GEMM on the GPU (batched-prefill path).
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

    /// I2_S ternary (BitNet b1.58) dequant-in-shader GEMM on the GPU.
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

    /// PrismML Q1_0 (1-bit) GEMV — `self` is `[1, k]` f32 on GPU, `w` is
    /// device-side packed Q1_0 bytes `[n, k/128 * 18]` (per-block fp16
    /// scale embedded). Returns `[1, n]`.
    pub fn q1_0_gemv_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "q1_0_gemv_cuda: self must be on GPU"
        );
        assert_eq!(self.numel(), k);
        assert_eq!(k % 128, 0);
        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(n).expect("GPU pool alloc failed");
        cuda.q1_0_gemv_f32(w, a_guard.slice(), &mut out, n, k)
            .expect("CUDA q1_0_gemv_f32 failed");
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

    /// PrismML Q1_0 DP4A path — int8 activation quant + dp4a GEMV.
    ///
    /// Same shape contract as `q1_0_gemv_cuda`: `self` is `[1, k]` f32 on
    /// GPU. Internally allocates int8 acts + fp16 scale buffers via the
    /// stream allocator (per-matmul scratch — short-lived, freed at fn
    /// scope end), fires `q1_0_quantize_acts_q8`, then `q1_0_gemv_dp4a_f32`.
    /// Returns `[1, n]`.
    pub fn q1_0_gemv_dp4a_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "q1_0_gemv_dp4a_cuda: self must be on GPU"
        );
        assert_eq!(self.numel(), k);
        assert_eq!(k % 128, 0);
        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();

        let mut a_q: cudarc::driver::CudaSlice<u8> =
            unsafe { cuda.stream().alloc::<u8>(k) }.expect("stream alloc u8 (int8 acts) failed");
        let mut a_d: cudarc::driver::CudaSlice<u16> = unsafe { cuda.stream().alloc::<u16>(k / 32) }
            .expect("stream alloc u16 (fp16-as-bits) failed");

        cuda.q1_0_quantize_acts_q8(a_guard.slice(), &mut a_q, &mut a_d, k)
            .expect("CUDA q1_0_quantize_acts_q8 failed");

        let mut out = pool_alloc_uninit(n).expect("GPU pool alloc failed");
        cuda.q1_0_gemv_dp4a_f32(w, &a_q, &a_d, &mut out, n, k)
            .expect("CUDA q1_0_gemv_dp4a_f32 failed");
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

    /// Q1_0 fused single-launch DP4A GEMV.
    ///
    /// Same shape contract as `q1_0_gemv_cuda`. Activation quant
    /// happens inside the kernel, in shared memory — no scratch
    /// allocation. Single launch.
    pub fn q1_0_gemv_fused_dp4a_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "q1_0_gemv_fused_dp4a_cuda: self must be on GPU"
        );
        assert_eq!(self.numel(), k);
        assert_eq!(k % 128, 0);
        let a_data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(n).expect("GPU pool alloc failed");
        cuda.q1_0_gemv_fused_dp4a_f32(w, a_guard.slice(), &mut out, n, k)
            .expect("CUDA q1_0_gemv_fused_dp4a_f32 failed");
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

    /// PrismML Q1_0 dequant-in-shader GEMM on the GPU.
    pub fn q1_0_gemm_cuda(
        &self,
        w: &cudarc::driver::CudaSlice<u8>,
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
        cuda.q1_0_gemm_f32(w, a_guard.slice(), &mut out, m, n, k)
            .expect("CUDA q1_0_gemm_f32 failed");
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

    /// Raw-i8 ternary matmul — `self` is `[m, k]` f32 on GPU, `w` is a
    /// host-side `&[i8]` of length `n * k` (axonml-nn TernaryLinear's
    /// shadow-quantize output). Picks GEMV (m=1) or GEMM (m>1) based on
    /// the leading axis of `self`. Returns `[m, n]` on GPU.
    ///
    /// Internally uploads `w` to a stream-allocated `CudaSlice<u8>`
    /// (i8 reinterpret); freed at fn-scope end. Per call, but each
    /// upload is k×n bytes which is cheap vs the matmul cost on
    /// realistic shapes.
    /// Quantize an f32 shadow weight `self` to ternary `{-1, 0, +1}` i8
    /// on GPU using absmean. `self` must be GPU-resident. Returns the
    /// new GPU-resident i8 buffer (as `CudaSlice<u8>` reinterpret) plus
    /// the f32 scale. Pass to `ternary_matmul_cuda_buf` /
    /// `ternary_grad_input_cuda_buf` without a host upload.
    pub fn quantize_to_ternary_cuda(&self) -> Result<(cudarc::driver::CudaSlice<u8>, f32)> {
        assert!(
            self.device().is_gpu(),
            "quantize_to_ternary_cuda: self must be on GPU"
        );
        let n = self.numel();
        let data = self.contiguous_gpu();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let guard = data.storage.as_cuda_slice();
        let mut out_i8: cudarc::driver::CudaSlice<u8> = unsafe { cuda.stream().alloc::<u8>(n) }
            .map_err(|e| axonml_core::error::Error::InvalidOperation {
                message: format!("stream alloc u8 (ternary quant) failed: {e}"),
            })?;
        let scale = cuda
            .ternary_quantize_weights(guard.slice(), &mut out_i8, n)
            .map_err(|e| axonml_core::error::Error::InvalidOperation {
                message: format!("ternary_quantize_weights failed: {e}"),
            })?;
        Ok((out_i8, scale))
    }

    /// `ternary_matmul_cuda` variant that takes a pre-uploaded GPU
    /// ternary buffer. Skips per-call htod_copy. Same shape contract.
    pub fn ternary_matmul_cuda_buf(
        &self,
        w_gpu: &cudarc::driver::CudaSlice<u8>,
        scale: f32,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "ternary_matmul_cuda_buf: self must be on GPU"
        );
        let a_data = self.contiguous_gpu();
        let numel = a_data.numel();
        assert!(numel % k == 0);
        let m = numel / k;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(m * n).expect("GPU pool alloc failed");
        if m == 1 {
            cuda.ternary_gemv_f32(w_gpu, a_guard.slice(), &mut out, scale, n, k)
                .expect("CUDA ternary_gemv_f32 failed");
        } else {
            cuda.ternary_gemm_f32(w_gpu, a_guard.slice(), &mut out, scale, m, n, k)
                .expect("CUDA ternary_gemm_f32 failed");
        }

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

    /// `ternary_grad_input_cuda` variant that takes a pre-uploaded GPU
    /// ternary buffer. Skips per-call htod_copy.
    pub fn ternary_grad_input_cuda_buf(
        &self,
        w_gpu: &cudarc::driver::CudaSlice<u8>,
        scale: f32,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "ternary_grad_input_cuda_buf: self must be on GPU"
        );
        let g_data = self.contiguous_gpu();
        let numel = g_data.numel();
        assert!(numel % out_features == 0);
        let batch_size = numel / out_features;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let g_guard = g_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(batch_size * in_features).expect("GPU pool alloc failed");
        cuda.ternary_grad_input_f32(
            w_gpu,
            g_guard.slice(),
            &mut out,
            scale,
            batch_size,
            in_features,
            out_features,
        )
        .expect("CUDA ternary_grad_input_f32 failed");

        let shape = Shape::from_slice(&[batch_size, in_features]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, batch_size * in_features, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Raw-i8 ternary matmul on GPU — host weights uploaded per call.
    ///
    /// `self` is `[m, k]` f32 GPU activation; `w_i8` is host-side
    /// `&[i8]` of length `n * k` (axonml-nn TernaryLinear's
    /// shadow-quantize output). `scale` is the tensor-wide absmean.
    /// Picks GEMV (m=1) or GEMM (m>1) based on `self`'s leading axis.
    /// Returns `[m, n]` on GPU.
    ///
    /// Per-call htod_copy of n*k bytes — cheap vs the matmul cost at
    /// realistic shapes; for the keep-buffer-on-GPU path use
    /// `ternary_matmul_cuda_buf` paired with `quantize_to_ternary_cuda`.
    pub fn ternary_matmul_cuda(&self, w_i8: &[i8], scale: f32, n: usize, k: usize) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "ternary_matmul_cuda: self must be on GPU"
        );
        assert_eq!(w_i8.len(), n * k, "ternary weight shape mismatch");
        let a_data = self.contiguous_gpu();
        let numel = a_data.numel();
        assert!(numel % k == 0);
        let m = numel / k;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let a_guard = a_data.storage.as_cuda_slice();

        let w_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(w_i8.as_ptr() as *const u8, w_i8.len()) };
        let w_gpu = cuda
            .htod_copy(w_bytes)
            .expect("htod_copy ternary weights failed");

        let mut out = pool_alloc_uninit(m * n).expect("GPU pool alloc failed");
        if m == 1 {
            cuda.ternary_gemv_f32(&w_gpu, a_guard.slice(), &mut out, scale, n, k)
                .expect("CUDA ternary_gemv_f32 failed");
        } else {
            cuda.ternary_gemm_f32(&w_gpu, a_guard.slice(), &mut out, scale, m, n, k)
                .expect("CUDA ternary_gemm_f32 failed");
        }

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

    /// Backward grad_input for raw-i8 TernaryLinear:
    /// `grad_in = scale * ternary^T @ grad_output`. `self` is the
    /// `[batch, out]` grad_output on GPU; weights uploaded internally.
    /// Returns `[batch, in]` on GPU.
    pub fn ternary_grad_input_cuda(
        &self,
        w_i8: &[i8],
        scale: f32,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "ternary_grad_input_cuda: self must be on GPU"
        );
        assert_eq!(w_i8.len(), out_features * in_features);
        let g_data = self.contiguous_gpu();
        let numel = g_data.numel();
        assert!(numel % out_features == 0);
        let batch_size = numel / out_features;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let g_guard = g_data.storage.as_cuda_slice();

        let w_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(w_i8.as_ptr() as *const u8, w_i8.len()) };
        let w_gpu = cuda
            .htod_copy(w_bytes)
            .expect("htod_copy ternary weights failed");

        let mut out = pool_alloc_uninit(batch_size * in_features).expect("GPU pool alloc failed");
        cuda.ternary_grad_input_f32(
            &w_gpu,
            g_guard.slice(),
            &mut out,
            scale,
            batch_size,
            in_features,
            out_features,
        )
        .expect("CUDA ternary_grad_input_f32 failed");

        let shape = Shape::from_slice(&[batch_size, in_features]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, batch_size * in_features, self.device());
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
        })
    }

    /// Backward grad_bias for TernaryLinear: sum across the batch axis.
    /// `self` is `[batch, out]` grad_output on GPU. Returns `[out]`.
    pub fn ternary_grad_bias_cuda(&self, out_features: usize) -> Result<Self> {
        assert!(
            self.device().is_gpu(),
            "ternary_grad_bias_cuda: self must be on GPU"
        );
        let g_data = self.contiguous_gpu();
        let numel = g_data.numel();
        assert!(numel % out_features == 0);
        let batch_size = numel / out_features;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let g_guard = g_data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(out_features).expect("GPU pool alloc failed");
        cuda.ternary_grad_bias_f32(g_guard.slice(), &mut out, batch_size, out_features)
            .expect("CUDA ternary_grad_bias_f32 failed");

        let shape = Shape::from_slice(&[out_features]);
        let strides = contiguous_strides(&shape);
        let storage = Storage::from_cuda_slice(out, out_features, self.device());
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

    /// Q8_0 dequant-in-shader GEMM on the GPU.
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

    /// Q6_K dequant-in-shader GEMM on the GPU.
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
        let mut out = pool_alloc_uninit(m * out_dim).expect("GPU pool alloc failed");

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

    /// GPU matrix multiplication using cuBLAS GEMM — no CPU copies.
    pub(crate) fn matmul_cuda(&self, other: &Self) -> Result<Self> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        fn is_last2_transposed(t: &Tensor<f32>) -> bool {
            let nd = t.ndim();
            if nd < 2 {
                return false;
            }
            let strides = t.strides.as_slice();
            strides[nd - 1] > strides[nd - 2]
        }

        fn batch_contiguous(t: &Tensor<f32>) -> bool {
            let nd = t.ndim();
            if nd <= 2 {
                return true;
            }
            let strides = t.strides.as_slice();
            let shape = t.shape.as_slice();
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

        let m = a.shape[a.shape.len() - 2];
        let k = a.shape[a.shape.len() - 1];
        let n = b.shape[b.shape.len() - 1];

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
            let a_guard = a.storage.as_cuda_slice();
            let b_guard = b.storage.as_cuda_slice();
            let mut c_gpu =
                pool_alloc_uninit(m * n).map_err(|e| crate::Error::InvalidOperation {
                    message: format!("GPU OOM in 2D matmul ({}x{}x{}): {}", m, k, n, e),
                })?;

            let (lda, op_a) = if a_transposed { (m, true) } else { (k, false) };
            let (ldb, op_b) = if b_transposed { (k, true) } else { (n, false) };

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

        let batch_dims: Vec<usize> = a.shape[..a.shape.len() - 2].to_vec();
        let batch_size: usize = batch_dims.iter().product();

        if batch_size == 0 || m == 0 || k == 0 || n == 0 {
            let mut out_shape = batch_dims.clone();
            out_shape.push(m);
            out_shape.push(n);
            let total: usize = out_shape.iter().product();
            return Ok(Self::from_vec(vec![0.0f32; total.max(1)], &out_shape)?);
        }

        let total = batch_size * m * n;

        let _a_guard = a.storage.as_cuda_slice();
        let _b_guard = b.storage.as_cuda_slice();

        let (cublas_transa, cublas_lda) = if b_transposed { (true, k) } else { (false, n) };
        let (cublas_transb, cublas_ldb) = if a_transposed { (true, m) } else { (false, k) };
        let cublas_ldc = n;
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

    /// Returns data as Vec<f32>, handling GPU D2H copy.
    pub(crate) fn to_vec_gpu(&self) -> Vec<f32> {
        self.storage.to_vec_f32()
    }

    /// Stream-ordered async D2H of `self`'s first `n` elements into a PINNED
    /// `dst_tensor` at `dst_offset` — no host sync (queued on the compute stream,
    /// ordered after prior async dtod work). Call `device_sync()` once after all
    /// chunks. Used by an offload backward's grad tile assembly.
    pub fn dtoh_prefix_into_stream(&self, n: usize, dst_tensor: &Self, dst_offset: usize) {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let src_guard = self.storage.as_cuda_slice();
        dst_tensor.storage.with_pinned_mut(|pin, base| {
            cuda.dtoh_into_n_stream(src_guard.slice(), n, pin, base + dst_offset)
                .expect("dtoh_prefix_into_stream pinned");
        });
    }

    /// Returns a contiguous GPU tensor — fully on-device using strided gather kernel.
    /// Computes gather indices directly on GPU, avoiding CPU index computation.
    pub fn matmul_into_at(
        &self,
        other: &Self,
        out: &Self,
        row_offset: usize,
        beta: f32,
    ) -> Result<Self> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        assert!(
            self.ndim() == 2 && other.ndim() == 2,
            "matmul_into_at: 2D only"
        );
        fn cublas_view(t: &Tensor<f32>) -> Option<(bool, usize, usize)> {
            let s = t.strides.as_slice();
            let sh = t.shape.as_slice();
            if s[1] == 1 && s[0] >= sh[1] as isize {
                Some((false, s[0] as usize, t.offset))
            } else if s[0] == 1 && s[1] >= sh[0] as isize {
                Some((true, s[1] as usize, t.offset))
            } else {
                None
            }
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        let a_contig = if cublas_view(self).is_some() {
            None
        } else {
            Some(self.contiguous_gpu())
        };
        let a_ref = a_contig.as_ref().unwrap_or(self);
        let (op_a, lda, a_off) = cublas_view(a_ref).expect("matmul_into_at: A not cuBLAS-usable");
        let b_contig = if cublas_view(other).is_some() {
            None
        } else {
            Some(other.contiguous_gpu())
        };
        let b_ref = b_contig.as_ref().unwrap_or(other);
        let (op_b, ldb, b_off) = cublas_view(b_ref).expect("matmul_into_at: B not cuBLAS-usable");
        {
            let a_guard = a_ref.storage.as_cuda_slice();
            let b_guard = b_ref.storage.as_cuda_slice();
            let mut out_guard = out.storage.as_cuda_slice_mut();
            cuda.gemm_f32_at(
                op_b,
                op_a,
                n,
                m,
                k,
                1.0,
                b_guard.slice(),
                b_off,
                ldb,
                a_guard.slice(),
                a_off,
                lda,
                beta,
                out_guard.slice_mut(),
                row_offset * n,
                n,
            )
            .expect("gemm_f32_at failed");
        }
        out.narrow(0, row_offset, m)
    }

    /// GPU argmax along a dimension. Fully on-device; returns f32 indices
    /// (exact for indices < 2^24). Ties resolve to the lowest index.
    pub(crate) fn argmax_dim_cuda(&self, dim: usize, keepdim: bool) -> Self {
        let data = self.contiguous_gpu();
        let ndim = data.shape.len();

        let outer_size: usize = data.shape[..dim].iter().product();
        let dim_size = data.shape[dim];
        let inner_size: usize = data.shape[dim + 1..].iter().product();
        let out_len = outer_size * inner_size;

        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc(out_len).expect("GPU pool alloc failed");

        cuda.argmax_dim_f32(
            &mut out,
            src_guard.slice(),
            outer_size,
            dim_size,
            inner_size,
        )
        .expect("CUDA argmax_dim_f32 failed");

        let shape = if keepdim {
            let mut s = data.shape.to_vec();
            s[dim] = 1;
            Shape::from_slice(&s)
        } else {
            let mut s: Vec<usize> = Vec::with_capacity(ndim.saturating_sub(1));
            for (i, &d) in data.shape.iter().enumerate() {
                if i != dim {
                    s.push(d);
                }
            }
            if s.is_empty() {
                s.push(1);
            }
            Shape::from_slice(&s)
        };
        let storage = Storage::from_cuda_slice(out, out_len, self.device());
        Self {
            storage,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        }
    }

    /// Builds a GPU tensor from host `data` by staging through a reused
    /// thread-local pinned buffer, so the H2D DMAs at full PCIe bandwidth (the
    /// pageable `to_device` path is ~1/9th the speed under WSL). Streams
    /// offloaded weight tiles onto the GPU cheaply.
    pub fn from_host_pinned(data: &[f32], shape: &[usize]) -> Self {
        use std::cell::RefCell;
        thread_local! {
            static STAGE: RefCell<Option<axonml_core::backends::cuda::PinnedBuffer>> =
                const { RefCell::new(None) };
        }
        let n = data.len();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        STAGE.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.as_ref().map_or(true, |p| p.len() < n) {
                *opt = Some(
                    axonml_core::backends::cuda::PinnedBuffer::alloc(n).expect("pinned alloc"),
                );
            }
            let pinned = opt.as_mut().unwrap();
            pinned.as_slice_mut()[..n].copy_from_slice(data);
            let slice = cuda
                .htod_copy(&pinned.as_slice()[..n])
                .expect("pinned htod");
            cuda.sync();
            let storage = Storage::from_cuda_slice_unmanaged(slice, n, Device::Cuda(0));
            Self::from_storage(storage, shape).expect("tensor from pinned storage")
        })
    }

    /// Uninitialized GPU tensor of `shape` (pool alloc, no H2D). Only safe when
    /// the caller writes every element before reading — used to assemble a
    /// weight-sized grad on-device (each row-tile `dtod_write_at`'s its slice)
    /// so the whole grad ships to host in ONE big contiguous pinned D2H instead
    /// of hundreds of small ones.
    pub fn empty_cuda(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        let out = pool_alloc_uninit(len).expect("GPU pool alloc failed");
        let storage = Storage::from_cuda_slice(out, len, Device::Cuda(0));
        Self::from_storage(storage, shape).expect("empty_cuda from_storage")
    }

    /// Record a timing CUDA event on the stream; returns a `u64` handle. Bracket
    /// GPU work with two of these + `event_elapsed_ms` for real per-op GPU time.
    /// Record a timing event on the backend stream (see `CudaBackend::event_record`).
    pub fn event_record() -> cudarc::driver::CudaEvent {
        get_cuda_backend()
            .expect("CUDA backend not available")
            .event_record()
            .expect("event_record")
    }

    /// Whether the backend stream is currently inside a CUDA-graph capture.
    pub fn graph_is_capturing() -> bool {
        get_cuda_backend()
            .expect("CUDA backend not available")
            .stream_is_capturing()
    }

    /// Begin capturing this stream's launch sequence into a CUDA graph (fixed-set
    /// decode only — capturing a training step is FORBIDDEN on WSL). Thread-local
    /// capture mode, as the pre-audit backend used.
    pub fn graph_begin_capture() {
        use cudarc::driver::sys::CUstreamCaptureMode;
        get_cuda_backend()
            .expect("CUDA backend not available")
            .stream()
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .expect("cuStreamBeginCapture failed");
    }

    /// End capture and instantiate. `None` when the capture was invalidated by an
    /// illegal mid-capture op or recorded nothing. Auto-free-on-launch, because a
    /// capture skips the pool cache and records real alloc nodes.
    pub fn graph_end_capture() -> Option<ReplayGraph> {
        use cudarc::driver::sys::CUgraphInstantiate_flags;
        get_cuda_backend()
            .expect("CUDA backend not available")
            .stream()
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .ok()
            .flatten()
            .map(ReplayGraph)
    }

    /// Replay an instantiated graph — one driver call for the whole recorded sequence.
    pub fn graph_launch(graph: &ReplayGraph) {
        graph.0.launch().expect("cuGraphLaunch failed");
    }

    /// Node-type histogram of a captured graph (diagnostic):
    /// `(total, kernel, memcpy, memset, host, mem_alloc, mem_free, other)`.
    pub fn graph_node_stats(
        graph: &ReplayGraph,
    ) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        use cudarc::driver::sys;
        // SAFETY: `graph` owns a live CUgraph for the whole call; the driver only
        // reads it, and `nodes` is sized from the count the driver reported.
        unsafe {
            let g = graph.0.cu_graph();
            let mut n: usize = 0;
            if sys::cuGraphGetNodes(g, std::ptr::null_mut(), &mut n) != sys::CUresult::CUDA_SUCCESS
            {
                return (0, 0, 0, 0, 0, 0, 0, 0);
            }
            let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
            let mut n2 = n;
            if sys::cuGraphGetNodes(g, nodes.as_mut_ptr(), &mut n2) != sys::CUresult::CUDA_SUCCESS {
                return (n, 0, 0, 0, 0, 0, 0, 0);
            }
            let (mut kern, mut cpy, mut set, mut host, mut al, mut fr, mut oth) =
                (0, 0, 0, 0, 0, 0, 0);
            for &node in &nodes {
                let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
                if sys::cuGraphNodeGetType(node, &mut ty) != sys::CUresult::CUDA_SUCCESS {
                    oth += 1;
                    continue;
                }
                match ty {
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => kern += 1,
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => cpy += 1,
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => set += 1,
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST => host += 1,
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => al += 1,
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => fr += 1,
                    _ => oth += 1,
                }
            }
            (n, kern, cpy, set, host, al, fr, oth)
        }
    }

    /// MEMCPY node breakdown by direction: `(htod, dtod, dtoh, other)`.
    pub fn graph_memcpy_kinds(graph: &ReplayGraph) -> (usize, usize, usize, usize) {
        use cudarc::driver::sys;
        // SAFETY: as in `graph_node_stats`; the params struct is a plain C POD the
        // driver fills in full on success.
        unsafe {
            let g = graph.0.cu_graph();
            let mut n: usize = 0;
            if sys::cuGraphGetNodes(g, std::ptr::null_mut(), &mut n) != sys::CUresult::CUDA_SUCCESS
            {
                return (0, 0, 0, 0);
            }
            let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
            let mut n2 = n;
            if sys::cuGraphGetNodes(g, nodes.as_mut_ptr(), &mut n2) != sys::CUresult::CUDA_SUCCESS {
                return (0, 0, 0, 0);
            }
            let (mut htod, mut dtod, mut dtoh, mut other) = (0, 0, 0, 0);
            for &node in &nodes {
                let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
                if sys::cuGraphNodeGetType(node, &mut ty) != sys::CUresult::CUDA_SUCCESS {
                    continue;
                }
                if ty != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY {
                    continue;
                }
                let mut p: sys::CUDA_MEMCPY3D_st = std::mem::MaybeUninit::zeroed().assume_init();
                if sys::cuGraphMemcpyNodeGetParams(node, &mut p) != sys::CUresult::CUDA_SUCCESS {
                    other += 1;
                    continue;
                }
                let host_t = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
                let dev_t = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
                if p.srcMemoryType == host_t && p.dstMemoryType == dev_t {
                    htod += 1;
                } else if p.srcMemoryType == dev_t && p.dstMemoryType == dev_t {
                    dtod += 1;
                } else if p.srcMemoryType == dev_t && p.dstMemoryType == host_t {
                    dtoh += 1;
                } else {
                    other += 1;
                }
            }
            (htod, dtod, dtoh, other)
        }
    }

    /// Blocks until every operation queued on the backend stream has completed.
    pub fn device_sync() {
        get_cuda_backend()
            .expect("CUDA backend not available")
            .sync();
    }

    /// In-place host→device overwrite of `self`'s contiguous data (then syncs).
    pub fn copy_from_host_inplace(&self, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.numel(),
            "copy_from_host_inplace: length mismatch"
        );
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let mut guard = self.storage.as_cuda_slice_mut();
        cuda.htod_into(data, guard.slice_mut())
            .expect("copy_from_host_inplace htod");
        cuda.sync();
    }

    /// Creates an uninitialized CPU tensor backed by page-locked (pinned) host
    /// memory. Device→host copies DMA straight into it at full bandwidth and the
    /// CPU optimizer reads it in place — no pageable staging copy. Caller must
    /// fill every element before reading.
    pub fn pinned_uninit(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        let storage = Storage::<f32>::pinned_uninit(len);
        Self::from_storage(storage, shape).expect("pinned tensor from storage")
    }

    /// Elapsed GPU milliseconds between two recorded events.
    pub fn event_elapsed_ms(
        start: &cudarc::driver::CudaEvent,
        stop: &cudarc::driver::CudaEvent,
    ) -> f32 {
        get_cuda_backend()
            .expect("CUDA backend not available")
            .event_elapsed_ms(start, stop)
    }

    /// Returns a contiguous, zero-offset copy of this GPU tensor, or `self` cloned when it already is one.
    pub fn contiguous_gpu(&self) -> Self {
        if self.is_contiguous() && self.offset == 0 {
            return self.clone();
        }
        let total = self.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let ndim = self.shape.len();
        let offset = self.offset;
        let shape = self.shape.as_slice();
        let strides = self.strides.as_slice();

        let shape_u32: Vec<u32> = shape.iter().map(|&s| s as u32).collect();
        let strides_i64: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
        let shape_guard = cuda.upload_shape_scratch(&shape_u32);
        let strides_guard = cuda.upload_strides_scratch(&strides_i64);

        let src_guard = self.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.strided_gather_f32(
            src_guard.slice(),
            &mut out,
            &strides_guard,
            &shape_guard,
            ndim,
            offset,
            total,
        )
        .expect("CUDA strided_gather_f32 failed");
        drop(shape_guard);
        drop(strides_guard);

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

        let mut idx_gpu =
            axonml_core::backends::cuda_pool::pool_alloc_uninit_u32(gather_indices.len())
                .expect("pool_alloc_uninit_u32 for embedding gather indices");
        cuda.htod_into(gather_indices, &mut idx_gpu)
            .expect("htod_into gather indices failed");

        let weight_guard = self.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(output_size).expect("GPU pool alloc failed");

        cuda.gather_contiguous_f32(&mut out, weight_guard.slice(), &idx_gpu, output_size)
            .expect("CUDA gather_contiguous_f32 failed");

        axonml_core::backends::cuda_pool::pool_free_u32(idx_gpu);

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

        let idx_gpu = cuda.htod_copy(indices).expect("htod indices failed");

        let out_size = num_embeddings * emb_dim;
        let mut out = pool_alloc(out_size).expect("GPU pool alloc failed");
        cuda.memset_zeros_f32(&mut out)
            .expect("memset zeros failed");

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

        let mut acc = pool_alloc(1).expect("GPU pool alloc failed");
        cuda.memset_zeros_f32(&mut acc).expect("memset failed");

        for grad in grads {
            let data = grad.contiguous_gpu();
            let n = data.numel();
            let guard = data.storage.as_cuda_slice();
            cuda.grad_norm_sq_f32(guard.slice(), &mut acc, n)
                .expect("CUDA grad_norm_sq_f32 failed");
        }

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
    /// GPU pooling-backward scatter: `self` is grad_output (GPU), `indices` are the saved per-output
    /// input positions (host `usize`). Returns grad_input of `in_numel`, zeros except scattered adds.
    /// Replaces a full grad_out D2H + host loop + grad_in H2D.
    pub fn maxpool_scatter_cuda(&self, indices: &[usize], in_numel: usize) -> Option<Self> {
        if !self.device().is_gpu() {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let go = self.contiguous_gpu();
        let go_guard = go.storage.as_cuda_slice();
        let idx_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let idx_gpu = cuda.htod_copy(&idx_u32).ok()?;
        let mut grad_in = pool_alloc(in_numel).ok()?;
        cuda.memset_zeros_f32(&mut grad_in).ok()?;
        cuda.scatter_add_u32_f32(
            go_guard.slice(),
            &idx_gpu,
            &mut grad_in,
            in_numel,
            idx_u32.len(),
        )
        .ok()?;
        let sh = Shape::from_slice(&[in_numel]);
        Some(Self {
            storage: Storage::from_cuda_slice(grad_in, in_numel, self.device()),
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        })
    }

    /// Fuse a two-input elementwise chain (`self`=a, `other`=b) into one JIT'd launch. Chain may use
    /// AddTensor/MulTensor/SubTensor to reference b. Same-shape only. Returns None off-GPU.
    pub fn fuse_binary_chain(
        &self,
        other: &Self,
        chain: &[crate::fused_chain::ChainOp],
    ) -> Option<Self> {
        if chain.is_empty() || !self.device().is_gpu() || !other.device().is_gpu() {
            return None;
        }
        if self.shape.as_slice() != other.shape.as_slice() {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let (key, expr) = crate::fused_chain::chain_codegen_binary(chain);
        let a = self.contiguous_gpu();
        let b = other.contiguous_gpu();
        let n: usize = a.shape.iter().product();
        let ag = a.storage.as_cuda_slice();
        let bg = b.storage.as_cuda_slice();
        let mut out = pool_alloc(n).ok()?;
        cuda.fused_chain_binary_f32(&key, &expr, ag.slice(), bg.slice(), &mut out, n)
            .ok()?;
        let sh = Shape::from_slice(a.shape.as_slice());
        Some(Self {
            storage: Storage::from_cuda_slice(out, n, self.device()),
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        })
    }

    /// Fuse a unary elementwise chain into a single JIT'd kernel launch (one global read + write
    /// for the whole chain instead of one launch per op). Returns None off-GPU / empty chain.
    pub fn fuse_unary_chain(&self, chain: &[crate::fused_chain::ChainOp]) -> Option<Self> {
        if chain.is_empty() || !self.device().is_gpu() {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let (key, expr) = crate::fused_chain::chain_codegen(chain);
        let x = self.contiguous_gpu();
        let n: usize = x.shape.iter().product();
        let xg = x.storage.as_cuda_slice();
        let mut out = pool_alloc(n).ok()?;
        cuda.fused_chain_unary_f32(&key, &expr, xg.slice(), &mut out, n)
            .ok()?;
        let sh = Shape::from_slice(x.shape.as_slice());
        Some(Self {
            storage: Storage::from_cuda_slice(out, n, self.device()),
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        })
    }

    /// Fused mul backward: `self` is grad_output; returns (grad_lhs, grad_rhs) in one launch.
    /// Same-shape only (broadcast handled by the caller falling back).
    pub fn mul_backward_cuda(&self, lhs: &Self, rhs: &Self) -> Option<(Self, Self)> {
        if !self.device().is_gpu() || !lhs.device().is_gpu() || !rhs.device().is_gpu() {
            return None;
        }
        if self.shape.as_slice() != lhs.shape.as_slice()
            || self.shape.as_slice() != rhs.shape.as_slice()
        {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let go = self.contiguous_gpu();
        let l = lhs.contiguous_gpu();
        let r = rhs.contiguous_gpu();
        let n = go.shape.iter().product::<usize>();
        let gog = go.storage.as_cuda_slice();
        let lg = l.storage.as_cuda_slice();
        let rg = r.storage.as_cuda_slice();
        let mut gl = pool_alloc(n).ok()?;
        let mut gr = pool_alloc(n).ok()?;
        cuda.mul_backward_f32(gog.slice(), lg.slice(), rg.slice(), &mut gl, &mut gr, n)
            .ok()?;
        let sh = Shape::from_slice(go.shape.as_slice());
        let mk = |buf| Self {
            storage: Storage::from_cuda_slice(buf, n, self.device()),
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        };
        Some((mk(gl), mk(gr)))
    }

    /// ConvTranspose2d backward on GPU. `self` is grad_output; returns (d_input, d_weight, d_bias?).
    #[allow(clippy::too_many_arguments)]
    pub fn convtranspose2d_backward_cuda(
        &self,
        input: &Self,
        weight: &Self,
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        has_bias: bool,
    ) -> Option<(Self, Self, Option<Self>)> {
        if !self.device().is_gpu() || !input.device().is_gpu() {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let go = self.contiguous_gpu();
        let inp = input.contiguous_gpu();
        let w = weight.contiguous_gpu();
        let gos = go.shape.as_slice();
        let (batch, out_h, out_w) = (gos[0], gos[2], gos[3]);
        let ins = inp.shape.as_slice();
        let (in_h, in_w) = (ins[2], ins[3]);
        let (kh, kw) = kernel;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let p = cuda
            .htod_copy(&[
                batch as u32,
                in_ch as u32,
                out_ch as u32,
                in_h as u32,
                in_w as u32,
                out_h as u32,
                out_w as u32,
                kh as u32,
                kw as u32,
                sh as u32,
                sw as u32,
                ph as u32,
                pw as u32,
            ])
            .ok()?;
        let gog = go.storage.as_cuda_slice();
        let ig = inp.storage.as_cuda_slice();
        let wg = w.storage.as_cuda_slice();
        let n_in = batch * in_ch * in_h * in_w;
        let n_w = in_ch * out_ch * kh * kw;
        let mut d_input = pool_alloc(n_in).ok()?;
        let mut d_weight = pool_alloc(n_w).ok()?;
        cuda.convtranspose2d_bwd_input_f32(gog.slice(), wg.slice(), &mut d_input, &p, n_in)
            .ok()?;
        cuda.convtranspose2d_bwd_weight_f32(ig.slice(), gog.slice(), &mut d_weight, &p, n_w)
            .ok()?;
        let mkt = |buf, numel, sh: &[usize]| {
            let s = Shape::from_slice(sh);
            Self {
                storage: Storage::from_cuda_slice(buf, numel, self.device()),
                shape: s.clone(),
                strides: contiguous_strides(&s),
                offset: 0,
            }
        };
        let d_bias = if has_bias {
            let mut gb = pool_alloc(out_ch).ok()?;
            cuda.memset_zeros_f32(&mut gb).ok()?;
            cuda.sum_bias_f32(gog.slice(), &mut gb, out_h * out_w, out_ch, batch)
                .ok()?;
            Some(mkt(gb, out_ch, &[out_ch]))
        } else {
            None
        };
        Some((
            mkt(d_input, n_in, &[batch, in_ch, in_h, in_w]),
            mkt(d_weight, n_w, &[in_ch, out_ch, kh, kw]),
            d_bias,
        ))
    }

    /// GroupNorm backward on GPU. `self` is grad_output; returns (d_input, d_weight, d_bias).
    pub fn groupnorm_backward_cuda(
        &self,
        input: &Self,
        weight: &Self,
        num_groups: usize,
        eps: f32,
    ) -> Option<(Self, Self, Self)> {
        if !self.device().is_gpu() || !input.device().is_gpu() {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let shape = input.shape.as_slice().to_vec();
        let (batch, channels) = (shape[0], shape[1]);
        let spatial: usize = shape[2..].iter().product();
        let n = batch * channels * spatial;
        let num_bg = batch * num_groups;
        let inp = input.contiguous_gpu();
        let go = self.contiguous_gpu();
        let w = weight.contiguous_gpu();
        let ig = inp.storage.as_cuda_slice();
        let gg = go.storage.as_cuda_slice();
        let wg = w.storage.as_cuda_slice();
        let params = cuda
            .htod_copy(&[
                batch as u32,
                channels as u32,
                spatial as u32,
                num_groups as u32,
            ])
            .ok()?;
        let mut stats = pool_alloc(num_bg * 4).ok()?;
        cuda.groupnorm_bwd_stats_f32(
            ig.slice(),
            gg.slice(),
            wg.slice(),
            &params,
            (batch, channels, spatial, num_groups),
            eps,
            &mut stats,
            num_bg,
        )
        .ok()?;
        let mut d_input = pool_alloc(n).ok()?;
        let mut d_weight = pool_alloc(channels).ok()?;
        let mut d_bias = pool_alloc(channels).ok()?;
        cuda.memset_zeros_f32(&mut d_weight).ok()?;
        cuda.memset_zeros_f32(&mut d_bias).ok()?;
        cuda.groupnorm_bwd_apply_f32(
            ig.slice(),
            gg.slice(),
            wg.slice(),
            &stats,
            &params,
            &mut d_input,
            &mut d_weight,
            &mut d_bias,
            n,
        )
        .ok()?;
        let mk = |buf, numel, sh: &[usize]| {
            let s = Shape::from_slice(sh);
            Self {
                storage: Storage::from_cuda_slice(buf, numel, self.device()),
                shape: s.clone(),
                strides: contiguous_strides(&s),
                offset: 0,
            }
        };
        Some((
            mk(d_input, n, &shape),
            mk(d_weight, channels, &[channels]),
            mk(d_bias, channels, &[channels]),
        ))
    }

    /// AdaptiveAvgPool2d backward on GPU: `self` is grad_output; returns grad_input `[b,c,in_h,in_w]`.
    pub fn adaptive_avgpool2d_bwd_cuda(
        &self,
        input_shape: &[usize],
        out_h: usize,
        out_w: usize,
    ) -> Option<Self> {
        if !self.device().is_gpu() {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let (b, c, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let n = b * c * in_h * in_w;
        let go = self.contiguous_gpu();
        let go_guard = go.storage.as_cuda_slice();
        let params = cuda
            .htod_copy(&[
                b as u32,
                c as u32,
                in_h as u32,
                in_w as u32,
                out_h as u32,
                out_w as u32,
            ])
            .ok()?;
        let mut grad_in = pool_alloc(n).ok()?;
        cuda.adaptive_avgpool2d_bwd_f32(go_guard.slice(), &mut grad_in, &params, n)
            .ok()?;
        let sh = Shape::from_slice(input_shape);
        Some(Self {
            storage: Storage::from_cuda_slice(grad_in, n, self.device()),
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        })
    }

    /// Scatters this narrowed gradient back into a zero tensor of `input_shape` at `start` along `dim`, on device.
    pub fn narrow_backward_cuda(&self, input_shape: &[usize], dim: usize, start: usize) -> Self {
        let numel: usize = input_shape.iter().product();
        let cuda = get_cuda_backend().expect("CUDA backend");

        let mut dst = pool_alloc(numel).expect("GPU pool alloc for narrow_backward");
        cuda.memset_zeros_f32(&mut dst)
            .expect("CUDA memset_zeros failed");

        let grad_contig = self.contiguous_gpu();
        let src_guard = grad_contig.storage.as_cuda_slice();

        let inner_size: usize = input_shape[dim + 1..].iter().product::<usize>().max(1);
        let offset_elements = start * inner_size;
        let outer_size: usize = input_shape[..dim].iter().product::<usize>().max(1);
        let dim_full = input_shape[dim];
        let dim_narrow = self.shape()[dim];
        let block_src = dim_narrow * inner_size;
        let block_dst = dim_full * inner_size;

        if outer_size == 1 {
            cuda.memcpy_dtod_f32(
                &mut dst,
                offset_elements,
                src_guard.slice(),
                0,
                grad_contig.shape.iter().product::<usize>(),
            )
            .expect("CUDA memcpy_dtod failed");
        } else {
            cuda.strided_block_copy_f32(
                src_guard.slice(),
                &mut dst,
                block_src,
                block_dst,
                offset_elements,
                outer_size * block_src,
            )
            .expect("CUDA strided_block_copy failed");
        }

        let out_shape = Shape::from_slice(input_shape);
        Self {
            storage: Storage::from_cuda_slice(dst, numel, self.device()),
            shape: out_shape.clone(),
            strides: contiguous_strides(&out_shape),
            offset: 0,
        }
    }

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

        let sum_cpu = cuda.dtoh_copy::<f32>(&sum_gpu).ok()?;
        let sum_sq_cpu = cuda.dtoh_copy::<f32>(&sum_sq_gpu).ok()?;

        let n_per_ch = (n * spatial) as f32;
        let mut mean_cpu = vec![0.0f32; channels];
        let mut var_cpu = vec![0.0f32; channels];
        for c in 0..channels {
            mean_cpu[c] = sum_cpu[c] / n_per_ch;
            var_cpu[c] = sum_sq_cpu[c] / n_per_ch - mean_cpu[c] * mean_cpu[c];
        }

        let mean_gpu = cuda.htod_copy(&mean_cpu).ok()?;
        let var_gpu = cuda.htod_copy(&var_cpu).ok()?;

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

    fn interp_index_map(
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        out_h: usize,
        out_w: usize,
    ) -> Vec<u32> {
        let scale_h = h as f32 / out_h as f32;
        let scale_w = w as f32 / out_w as f32;
        let mut idx = Vec::with_capacity(n * c * out_h * out_w);
        for b in 0..n {
            for ch in 0..c {
                let base = b * c * h * w + ch * h * w;
                for oh in 0..out_h {
                    let ih = (((oh as f32 + 0.5) * scale_h) as usize).min(h - 1);
                    for ow in 0..out_w {
                        let iw = (((ow as f32 + 0.5) * scale_w) as usize).min(w - 1);
                        idx.push((base + ih * w + iw) as u32);
                    }
                }
            }
        }
        idx
    }

    /// Nearest-neighbour resize of a `[N, C, H, W]` GPU tensor to `out_h` x `out_w`; `None` when not applicable.
    pub fn interpolate_nearest_cuda(&self, out_h: usize, out_w: usize) -> Option<Self> {
        if !self.device().is_gpu() || self.shape.len() != 4 {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        if h == 0 || w == 0 {
            return None;
        }
        let out_len = n * c * out_h * out_w;
        let src = self.contiguous_gpu();
        let idx = Self::interp_index_map(n, c, h, w, out_h, out_w);
        let idx_gpu = cuda.htod_copy(&idx).ok()?;
        let guard = src.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(out_len).ok()?;
        cuda.gather_contiguous_f32(&mut out, guard.slice(), &idx_gpu, out_len)
            .ok()?;
        drop(guard);
        let storage = Storage::from_cuda_slice(out, out_len, self.device());
        let shape: Shape = vec![n, c, out_h, out_w].into();
        Some(Self {
            strides: contiguous_strides(&shape),
            shape,
            storage,
            offset: 0,
        })
    }

    /// Backward of [`Self::interpolate_nearest_cuda`]: accumulates this gradient into `in_shape`; `None` when not applicable.
    pub fn interpolate_nearest_backward_cuda(&self, in_shape: &[usize]) -> Option<Self> {
        if !self.device().is_gpu() || self.shape.len() != 4 || in_shape.len() != 4 {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (out_h, out_w) = (self.shape[2], self.shape[3]);
        if h == 0 || w == 0 {
            return None;
        }
        let out_len = n * c * out_h * out_w;
        let in_numel = n * c * h * w;
        let go = self.contiguous_gpu();
        let idx = Self::interp_index_map(n, c, h, w, out_h, out_w);
        let idx_gpu = cuda.htod_copy(&idx).ok()?;
        let guard = go.storage.as_cuda_slice();
        let mut gi = pool_alloc(in_numel).ok()?;
        cuda.scatter_add_u32_f32(guard.slice(), &idx_gpu, &mut gi, in_numel, out_len)
            .ok()?;
        drop(guard);
        let storage = Storage::from_cuda_slice(gi, in_numel, self.device());
        let shape: Shape = in_shape.to_vec().into();
        Some(Self {
            strides: contiguous_strides(&shape),
            shape,
            storage,
            offset: 0,
        })
    }

    /// Batched 2-D convolution on device (im2col + strided-batched GEMM), with optional bias.
    pub fn conv2d_cuda(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
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

        // Ensure input and weight are contiguous on GPU
        let input_data = self.contiguous_gpu();
        let weight_data = weight.contiguous_gpu();

        let weight_guard = weight_data.storage.as_cuda_slice();

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
        let mut bparams = [0u32; 11];
        bparams[..10].copy_from_slice(&im2col_params);
        bparams[10] = in_channels as u32;
        let bparams_gpu = cuda.htod_copy(&bparams[..]).ok()?;

        let bias_data = bias.map(|b| b.contiguous_gpu());
        let bias_guard = bias_data.as_ref().map(|b| b.storage.as_cuda_slice());

        let total_out = batch_size * out_per_batch;
        let mut out_gpu = pool_alloc(total_out).ok()?;

        // ── whole-batch im2col + ONE strided-batched GEMM ──
        let input_guard = input_data.storage.as_cuda_slice();
        let mut col_gpu = pool_alloc(batch_size * col_n).ok()?;
        cuda.im2col_batched_f32(
            input_guard.slice(),
            &mut col_gpu,
            &bparams_gpu,
            batch_size * col_n,
        )
        .ok()?;

        cuda.gemm_strided_batched_f32(
            false,
            false,
            col_w,
            out_channels,
            col_h,
            1.0,
            &col_gpu,
            col_w,
            col_n as i64,
            weight_guard.slice(),
            col_h,
            0,
            0.0,
            &mut out_gpu,
            col_w,
            out_per_batch as i64,
            batch_size,
        )
        .ok()?;

        if let Some(ref bg) = bias_guard {
            cuda.bias_add_channels_batched_f32(
                &mut out_gpu,
                bg.slice(),
                spatial,
                out_channels,
                total_out,
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

    #[cfg(feature = "cudnn")]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_backward_cudnn(
        &self,
        saved_input: &Self,
        saved_weight: &Self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        has_bias: bool,
    ) -> Option<(Self, Self, Option<Self>)> {
        if !self.device().is_gpu()
            || !saved_input.device().is_gpu()
            || !saved_weight.device().is_gpu()
        {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let cudnn = cuda.cudnn()?;
        let n = saved_input.shape[0];
        let in_h = saved_input.shape[2];
        let in_w = saved_input.shape[3];
        let (kh, kw) = kernel_size;
        let out_h = self.shape[2];
        let out_w = self.shape[3];
        let groups = (in_channels / saved_weight.shape[1].max(1)).max(1);

        let go = self.contiguous_gpu();
        let inp = saved_input.contiguous_gpu();
        let w = saved_weight.contiguous_gpu();
        let go_g = go.storage.as_cuda_slice();
        let inp_g = inp.storage.as_cuda_slice();
        let w_g = w.storage.as_cuda_slice();

        let gi = axonml_core::backends::cudnn_ops::cudnn_conv2d_backward_data(
            cudnn,
            cuda.stream(),
            go_g.slice(),
            w_g.slice(),
            n,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kh,
            kw,
            out_h,
            out_w,
            stride,
            padding,
            groups,
        )?;
        let gw = axonml_core::backends::cudnn_ops::cudnn_conv2d_backward_filter(
            cudnn,
            cuda.stream(),
            go_g.slice(),
            inp_g.slice(),
            n,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kh,
            kw,
            out_h,
            out_w,
            stride,
            padding,
            groups,
        )?;

        let gi_shape = Shape::from_slice(&[n, in_channels, in_h, in_w]);
        let grad_input = Self {
            storage: Storage::from_cuda_slice(gi, n * in_channels * in_h * in_w, self.device()),
            shape: gi_shape.clone(),
            strides: contiguous_strides(&gi_shape),
            offset: 0,
        };
        let gw_numel = out_channels * (in_channels / groups) * kh * kw;
        let gw_shape = Shape::from_slice(&[out_channels, in_channels / groups, kh, kw]);
        let grad_weight = Self {
            storage: Storage::from_cuda_slice(gw, gw_numel, self.device()),
            shape: gw_shape.clone(),
            strides: contiguous_strides(&gw_shape),
            offset: 0,
        };

        let grad_bias = if has_bias {
            let god = go.to_vec();
            let hw = out_h * out_w;
            let mut gb = vec![0f32; out_channels];
            for ni in 0..n {
                for c in 0..out_channels {
                    let base = (ni * out_channels + c) * hw;
                    let mut acc = 0f32;
                    for k in 0..hw {
                        acc += god[base + k];
                    }
                    gb[c] += acc;
                }
            }
            Tensor::from_vec(gb, &[out_channels])
                .ok()
                .and_then(|t| t.to_device(self.device()).ok())
        } else {
            None
        };

        Some((grad_input, grad_weight, grad_bias))
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
        let out_per_batch = out_channels * spatial;

        let input_data = self.contiguous_gpu();
        let weight_data = weight.contiguous_gpu();

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

        let bias_data = bias.map(|b| b.contiguous_gpu());

        // ── direct depthwise (groups == C_in == C_out) ──
        if in_channels_per_group == 1 && out_channels_per_group == 1 && !no_direct_depthwise() {
            let dparams: [u32; 12] = [
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
                in_channels as u32,
                batch_size as u32,
            ];
            let dparams_gpu = cuda.htod_copy(&dparams[..]).ok()?;
            let input_guard = input_data.storage.as_cuda_slice();
            let weight_guard = weight_data.storage.as_cuda_slice();
            let total_out = batch_size * out_per_batch;
            let mut out_gpu = pool_alloc_uninit(total_out).ok()?;
            cuda.depthwise_fwd_f32(
                input_guard.slice(),
                weight_guard.slice(),
                &mut out_gpu,
                &dparams_gpu,
                total_out,
            )
            .ok()?;
            if let Some(bd) = bias_data.as_ref() {
                let bias_guard = bd.storage.as_cuda_slice();
                cuda.bias_add_channels_batched_f32(
                    &mut out_gpu,
                    bias_guard.slice(),
                    spatial,
                    out_channels,
                    total_out,
                )
                .ok()?;
            }
            let out_shape = Shape::from_slice(&[batch_size, out_channels, out_h, out_w]);
            return Some(Self {
                storage: Storage::from_cuda_slice(out_gpu, total_out, self.device()),
                strides: contiguous_strides(&out_shape),
                shape: out_shape,
                offset: 0,
            });
        }

        let mut gparams = [0u32; 13];
        gparams[..10].copy_from_slice(&params_arr);
        gparams[10] = in_channels_per_group as u32;
        gparams[11] = in_channels as u32;
        gparams[12] = batch_size as u32;
        let gparams_gpu = cuda.htod_copy(&gparams[..]).ok()?;

        let total_out = batch_size * out_per_batch;
        let mut out_gpu = pool_alloc(total_out).ok()?;

        // ── one im2col for the whole batch AND all groups, then one strided-batched GEMM per group ──
        let input_guard = input_data.storage.as_cuda_slice();
        let weight_guard = weight_data.storage.as_cuda_slice();
        let mut col_gpu = pool_alloc(groups * batch_size * col_n).ok()?;
        cuda.im2col_group_batched_f32(
            input_guard.slice(),
            &mut col_gpu,
            &gparams_gpu,
            groups * batch_size * col_n,
        )
        .ok()?;

        let w_per_group = out_channels_per_group * col_h;
        for g in 0..groups {
            cuda.gemm_strided_batched_f32_at(
                false,
                false,
                col_w,
                out_channels_per_group,
                col_h,
                1.0,
                &col_gpu,
                g * batch_size * col_n,
                col_w,
                col_n as i64,
                weight_guard.slice(),
                g * w_per_group,
                col_h,
                0,
                0.0,
                &mut out_gpu,
                g * out_channels_per_group * spatial,
                col_w,
                out_per_batch as i64,
                batch_size,
            )
            .ok()?;
        }

        if let Some(bd) = bias_data.as_ref() {
            let bias_guard = bd.storage.as_cuda_slice();
            cuda.bias_add_channels_batched_f32(
                &mut out_gpu,
                bias_guard.slice(),
                spatial,
                out_channels,
                total_out,
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
        if !self.device().is_gpu()
            || !saved_input.device().is_gpu()
            || !saved_weight.device().is_gpu()
        {
            return None;
        }
        let cuda = get_cuda_backend()?;

        #[cfg(feature = "cudnn")]
        {
            if let Some(r) = self.conv2d_backward_cudnn(
                saved_input,
                saved_weight,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                has_bias,
            ) {
                return Some(r);
            }
        }
        // Grouped/depthwise (groups>1) is handled by a dedicated branch further down. It used to
        // bail here to the CPU fallback: the dense path derives col_h from the FULL in_channels and
        // has no notion of groups, so running it for groups>1 is simply the wrong computation.

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
        let weight_guard = weight_data.storage.as_cuda_slice();
        let input_guard = input_data.storage.as_cuda_slice();

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

        // ── grouped / depthwise backward ──
        let groups = (in_channels / saved_weight.shape[1].max(1)).max(1);
        if groups > 1 {
            let icg = in_channels / groups;
            let ocg = out_channels / groups;

            // ── direct depthwise (icg == ocg == 1) — see conv2d_grouped_cuda for why GEMM loses ──
            if icg == 1 && ocg == 1 && !no_direct_depthwise() {
                let dparams: [u32; 12] = [
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
                    in_channels as u32,
                    batch_size as u32,
                ];
                let dparams_gpu = cuda.htod_copy(&dparams[..]).ok()?;
                let total_input = batch_size * in_per_batch;
                let w_numel = in_channels * kh * kw;

                let mut grad_input_gpu = pool_alloc_uninit(total_input).ok()?;
                cuda.depthwise_grad_input_f32(
                    grad_out_guard.slice(),
                    weight_guard.slice(),
                    &mut grad_input_gpu,
                    &dparams_gpu,
                    total_input,
                )
                .ok()?;

                let mut grad_weight_gpu = pool_alloc(w_numel).ok()?;
                cuda.memset_zeros_f32(&mut grad_weight_gpu).ok()?;
                cuda.depthwise_grad_weight_f32(
                    grad_out_guard.slice(),
                    input_guard.slice(),
                    &mut grad_weight_gpu,
                    &dparams_gpu,
                    w_numel,
                )
                .ok()?;

                let grad_bias_t = if has_bias {
                    let mut gb = pool_alloc(out_channels).ok()?;
                    cuda.memset_zeros_f32(&mut gb).ok()?;
                    cuda.sum_bias_f32(
                        grad_out_guard.slice(),
                        &mut gb,
                        spatial,
                        out_channels,
                        batch_size,
                    )
                    .ok()?;
                    let sh_b = Shape::from_slice(&[out_channels]);
                    Some(Self {
                        storage: Storage::from_cuda_slice(gb, out_channels, self.device()),
                        strides: contiguous_strides(&sh_b),
                        shape: sh_b,
                        offset: 0,
                    })
                } else {
                    None
                };

                let gi_shape = Shape::from_slice(&[batch_size, in_channels, in_h, in_w]);
                let gw_shape = Shape::from_slice(&[out_channels, 1, kh, kw]);
                return Some((
                    Self {
                        storage: Storage::from_cuda_slice(
                            grad_input_gpu,
                            total_input,
                            self.device(),
                        ),
                        strides: contiguous_strides(&gi_shape),
                        shape: gi_shape,
                        offset: 0,
                    },
                    Self {
                        storage: Storage::from_cuda_slice(grad_weight_gpu, w_numel, self.device()),
                        strides: contiguous_strides(&gw_shape),
                        shape: gw_shape,
                        offset: 0,
                    },
                    grad_bias_t,
                ));
            }

            let col_h_g = icg * kh * kw;
            let col_n_g = col_h_g * spatial;
            let w_per_group = ocg * col_h_g;
            let weight_n_g = out_channels * col_h_g;

            let mut gparams = [0u32; 13];
            gparams[..10].copy_from_slice(&params_arr);
            gparams[10] = icg as u32;
            gparams[11] = in_channels as u32;
            gparams[12] = batch_size as u32;
            let gparams_gpu = cuda.htod_copy(&gparams[..]).ok()?;

            let mut grad_weight_gpu = pool_alloc(weight_n_g).ok()?;
            cuda.memset_zeros_f32(&mut grad_weight_gpu).ok()?;
            let total_input = batch_size * in_per_batch;
            let mut grad_input_gpu = pool_alloc(total_input).ok()?;
            let mut col_gpu = pool_alloc(groups * batch_size * col_n_g).ok()?;

            for g in 0..groups {
                cuda.gemm_strided_batched_f32_at(
                    false,
                    true,
                    spatial,
                    col_h_g,
                    ocg,
                    1.0,
                    grad_out_guard.slice(),
                    g * ocg * spatial,
                    spatial,
                    out_per_batch as i64,
                    weight_guard.slice(),
                    g * w_per_group,
                    col_h_g,
                    0,
                    0.0,
                    &mut col_gpu,
                    g * batch_size * col_n_g,
                    spatial,
                    col_n_g as i64,
                    batch_size,
                )
                .ok()?;
            }
            cuda.memset_zeros_f32(&mut grad_input_gpu).ok()?;
            cuda.col2im_group_batched_f32(
                &col_gpu,
                &mut grad_input_gpu,
                &gparams_gpu,
                groups * batch_size * col_n_g,
            )
            .ok()?;

            cuda.im2col_group_batched_f32(
                input_guard.slice(),
                &mut col_gpu,
                &gparams_gpu,
                groups * batch_size * col_n_g,
            )
            .ok()?;
            let mut gw_partial = pool_alloc(batch_size * w_per_group).ok()?;
            for g in 0..groups {
                cuda.gemm_strided_batched_f32_at(
                    true,
                    false,
                    col_h_g,
                    ocg,
                    spatial,
                    1.0,
                    &col_gpu,
                    g * batch_size * col_n_g,
                    spatial,
                    col_n_g as i64,
                    grad_out_guard.slice(),
                    g * ocg * spatial,
                    spatial,
                    out_per_batch as i64,
                    0.0,
                    &mut gw_partial,
                    0,
                    col_h_g,
                    w_per_group as i64,
                    batch_size,
                )
                .ok()?;
                cuda.sum_batch_at_f32(
                    &gw_partial,
                    &mut grad_weight_gpu,
                    g * w_per_group,
                    w_per_group,
                    batch_size,
                )
                .ok()?;
            }

            let grad_bias_t = if has_bias {
                let mut gb = pool_alloc(out_channels).ok()?;
                cuda.memset_zeros_f32(&mut gb).ok()?;
                cuda.sum_bias_f32(
                    grad_out_guard.slice(),
                    &mut gb,
                    spatial,
                    out_channels,
                    batch_size,
                )
                .ok()?;
                let sh = Shape::from_slice(&[out_channels]);
                Some(Self {
                    storage: Storage::from_cuda_slice(gb, out_channels, self.device()),
                    strides: contiguous_strides(&sh),
                    shape: sh,
                    offset: 0,
                })
            } else {
                None
            };

            let gi_shape = Shape::from_slice(&[batch_size, in_channels, in_h, in_w]);
            let gw_shape = Shape::from_slice(&[out_channels, icg, kh, kw]);
            return Some((
                Self {
                    storage: Storage::from_cuda_slice(grad_input_gpu, total_input, self.device()),
                    strides: contiguous_strides(&gi_shape),
                    shape: gi_shape,
                    offset: 0,
                },
                Self {
                    storage: Storage::from_cuda_slice(grad_weight_gpu, weight_n_g, self.device()),
                    strides: contiguous_strides(&gw_shape),
                    shape: gw_shape,
                    offset: 0,
                },
                grad_bias_t,
            ));
        }

        let mut bparams = [0u32; 11];
        bparams[..10].copy_from_slice(&params_arr);
        bparams[10] = in_channels as u32;
        let bparams_gpu = cuda.htod_copy(&bparams[..]).ok()?;

        let weight_n = out_channels * col_h;
        let mut grad_weight_gpu = pool_alloc(weight_n).ok()?;
        cuda.memset_zeros_f32(&mut grad_weight_gpu).ok()?;

        let total_input = batch_size * in_per_batch;
        let mut grad_input_gpu = pool_alloc(total_input).ok()?;

        let mut grad_bias_gpu = if has_bias {
            let gb = pool_alloc(out_channels).ok()?;
            Some(gb)
        } else {
            None
        };
        if let Some(ref mut gb) = grad_bias_gpu {
            cuda.memset_zeros_f32(gb).ok()?;
        }

        // ── whole-batch backward ──
        const MAX_PARTIAL_ELEMS: usize = 128 * 1024 * 1024;
        let batched_gw = batch_size.saturating_mul(weight_n) <= MAX_PARTIAL_ELEMS;

        let mut col_gpu = pool_alloc(batch_size * col_n).ok()?;

        cuda.gemm_strided_batched_f32(
            false,
            true,
            spatial,
            col_h,
            out_channels,
            1.0,
            grad_out_guard.slice(),
            spatial,
            out_per_batch as i64,
            weight_guard.slice(),
            col_h,
            0,
            0.0,
            &mut col_gpu,
            spatial,
            col_n as i64,
            batch_size,
        )
        .ok()?;

        cuda.memset_zeros_f32(&mut grad_input_gpu).ok()?;
        cuda.col2im_batched_f32(
            &col_gpu,
            &mut grad_input_gpu,
            &bparams_gpu,
            batch_size * col_n,
        )
        .ok()?;

        cuda.im2col_batched_f32(
            input_guard.slice(),
            &mut col_gpu,
            &bparams_gpu,
            batch_size * col_n,
        )
        .ok()?;

        if batched_gw {
            let mut gw_partial = pool_alloc(batch_size * weight_n).ok()?;
            cuda.gemm_strided_batched_f32(
                true,
                false,
                col_h,
                out_channels,
                spatial,
                1.0,
                &col_gpu,
                spatial,
                col_n as i64,
                grad_out_guard.slice(),
                spatial,
                out_per_batch as i64,
                0.0,
                &mut gw_partial,
                col_h,
                weight_n as i64,
                batch_size,
            )
            .ok()?;
            cuda.sum_batch_f32(&gw_partial, &mut grad_weight_gpu, weight_n, batch_size)
                .ok()?;
        } else {
            for b in 0..batch_size {
                cuda.gemm_f32_at(
                    true,
                    false,
                    col_h,
                    out_channels,
                    spatial,
                    1.0,
                    &col_gpu,
                    b * col_n,
                    spatial,
                    grad_out_guard.slice(),
                    b * out_per_batch,
                    spatial,
                    1.0,
                    &mut grad_weight_gpu,
                    0,
                    col_h,
                )
                .ok()?;
            }
        }

        // === grad_bias: sum grad_out over batch+spatial per channel, ON STREAM. ===
        if let Some(ref mut gb) = grad_bias_gpu {
            cuda.sum_bias_f32(
                grad_out_guard.slice(),
                gb,
                spatial,
                out_channels,
                batch_size,
            )
            .ok()?;
        }

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

    /// GPU-resident BatchNorm2d backward. `self` is `grad_output` `[N,C,H,W]`.
    /// `mean`/`var`/`gamma` are the saved per-channel batch stats (`[C]`, host).
    /// Returns `(grad_input, grad_weight, grad_bias)`, all GPU-resident. Replaces
    /// the full-tensor `to_vec` CPU path in `BatchNorm2dBackward`.
    pub fn batchnorm2d_backward_cuda(
        &self,
        saved_input: &Self,
        mean: &[f32],
        var: &[f32],
        gamma: &[f32],
        eps: f32,
    ) -> Option<(Self, Self, Self)> {
        if !self.device().is_gpu() || !saved_input.device().is_gpu() {
            return None;
        }
        if self.shape.len() != 4 {
            return None;
        }
        let cuda = get_cuda_backend()?;
        let n = self.shape[0];
        let c = self.shape[1];
        let spatial = self.shape[2] * self.shape[3];
        let total = n * c * spatial;

        let grad_data = self.contiguous_gpu();
        let input_data = saved_input.contiguous_gpu();
        let grad_guard = grad_data.storage.as_cuda_slice();
        let input_guard = input_data.storage.as_cuda_slice();

        let mean_gpu = cuda.htod_copy(mean).ok()?;
        let var_gpu = cuda.htod_copy(var).ok()?;
        let gamma_gpu = cuda.htod_copy(gamma).ok()?;

        let mut sum_grad = pool_alloc(c).ok()?;
        let mut sum_grad_xhat = pool_alloc(c).ok()?;
        cuda.memset_zeros_f32(&mut sum_grad).ok()?;
        cuda.memset_zeros_f32(&mut sum_grad_xhat).ok()?;

        cuda.batchnorm_bwd_reduce_f32(
            grad_guard.slice(),
            input_guard.slice(),
            &mean_gpu,
            &var_gpu,
            &mut sum_grad,
            &mut sum_grad_xhat,
            eps,
            n,
            c,
            spatial,
        )
        .ok()?;

        let mut grad_input = pool_alloc(total).ok()?;
        cuda.batchnorm_bwd_input_f32(
            grad_guard.slice(),
            input_guard.slice(),
            &mean_gpu,
            &var_gpu,
            &gamma_gpu,
            &sum_grad,
            &sum_grad_xhat,
            &mut grad_input,
            eps,
            n,
            c,
            spatial,
        )
        .ok()?;

        let gi_shape =
            Shape::from_slice(&[self.shape[0], self.shape[1], self.shape[2], self.shape[3]]);
        let grad_input_t = Self {
            storage: Storage::from_cuda_slice(grad_input, total, self.device()),
            shape: gi_shape.clone(),
            strides: contiguous_strides(&gi_shape),
            offset: 0,
        };
        let c_shape = Shape::from_slice(&[c]);
        let grad_weight_t = Self {
            storage: Storage::from_cuda_slice(sum_grad_xhat, c, self.device()),
            shape: c_shape.clone(),
            strides: contiguous_strides(&c_shape),
            offset: 0,
        };
        let grad_bias_t = Self {
            storage: Storage::from_cuda_slice(sum_grad, c, self.device()),
            shape: c_shape.clone(),
            strides: contiguous_strides(&c_shape),
            offset: 0,
        };
        Some((grad_input_t, grad_weight_t, grad_bias_t))
    }

    /// GPU-resident concatenation along `dim`. Within one outer-row an input's
    /// cat-dim slice is contiguous in both source and destination, so it needs
    /// only one d2d copy per (input, outer-row) — no host round-trip. Replaces
    /// the full-tensor `to_vec` CPU path in `Tensor::cat`.
    pub fn cat_cuda(tensors: &[&Self], dim: usize, out_shape: &[usize]) -> Option<Self> {
        let cuda = get_cuda_backend()?;
        let total_dim_size = out_shape[dim];
        let outer_size: usize = out_shape[..dim].iter().product();
        let inner_size: usize = out_shape[dim + 1..].iter().product();
        let total_numel: usize = out_shape.iter().product();
        let mut out = pool_alloc(total_numel).ok()?;

        let mut dim_offset = 0usize;
        for t in tensors {
            if !t.device().is_gpu() {
                return None;
            }
            let tc = t.contiguous_gpu();
            let src_guard = tc.storage.as_cuda_slice();
            let t_dim_size = t.shape[dim];
            let block = t_dim_size * inner_size;
            cuda.strided_block_copy_f32(
                src_guard.slice(),
                &mut out,
                block,
                total_dim_size * inner_size,
                dim_offset * inner_size,
                outer_size * block,
            )
            .ok()?;
            dim_offset += t_dim_size;
        }

        let sh = Shape::from_slice(out_shape);
        Some(Self {
            storage: Storage::from_cuda_slice(out, total_numel, tensors[0].device()),
            shape: sh.clone(),
            strides: contiguous_strides(&sh),
            offset: 0,
        })
    }

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

        let input_data = self.contiguous_gpu();
        let input_guard = input_data.storage.as_cuda_slice();

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
        let mut out = pool_alloc_uninit(data.numel()).expect("GPU pool alloc failed");

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
        let mut out = pool_alloc_uninit(len).expect("GPU pool alloc failed");

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

    /// GPU SwiGLU backward. `self` is the saved forward gate; `up` is the
    /// saved forward up; `grad_output` is `dL/dy`. Returns `(grad_gate, grad_up)`.
    pub(crate) fn swiglu_bwd_cuda(&self, up: &Self, grad_output: &Self) -> (Self, Self) {
        let g = self.contiguous_gpu();
        let u = up.contiguous_gpu();
        let go = grad_output.contiguous_gpu();
        let len = g.numel();
        debug_assert_eq!(len, u.numel());
        debug_assert_eq!(len, go.numel());
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let g_guard = g.storage.as_cuda_slice();
        let u_guard = u.storage.as_cuda_slice();
        let go_guard = go.storage.as_cuda_slice();

        let mut grad_gate = pool_alloc_uninit(len).expect("GPU pool alloc failed");
        let mut grad_up = pool_alloc_uninit(len).expect("GPU pool alloc failed");

        cuda.swiglu_bwd_f32(
            &mut grad_gate,
            &mut grad_up,
            g_guard.slice(),
            u_guard.slice(),
            go_guard.slice(),
            len,
        )
        .expect("CUDA swiglu_bwd_f32 failed");

        let gg_tensor = Self {
            storage: Storage::from_cuda_slice(grad_gate, len, self.device()),
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        };
        let gu_tensor = Self {
            storage: Storage::from_cuda_slice(grad_up, len, self.device()),
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        };
        (gg_tensor, gu_tensor)
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

    /// Fused residual-add + batched RMSNorm forward.
    /// `self` and `b` are shape `[m, n]`; `weight` is `[n]`. Returns
    /// `(normalized_output, sum = self + b)` — the sum is saved so the
    /// backward can reconstruct rms without re-running the add.
    pub(crate) fn add_rmsnorm_batched_cuda(
        &self,
        b: &Self,
        weight: &Self,
        m: usize,
        n: usize,
        eps: f32,
    ) -> (Self, Self) {
        let a = self.contiguous_gpu();
        let bb = b.contiguous_gpu();
        let w = weight.contiguous_gpu();
        debug_assert_eq!(a.numel(), m * n);
        debug_assert_eq!(bb.numel(), m * n);
        debug_assert_eq!(w.numel(), n);
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let a_guard = a.storage.as_cuda_slice();
        let b_guard = bb.storage.as_cuda_slice();
        let w_guard = w.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(m * n).expect("GPU pool alloc failed");
        let mut sum_out = pool_alloc_uninit(m * n).expect("GPU pool alloc failed");

        cuda.add_rmsnorm_batched_f32(
            &mut out,
            &mut sum_out,
            a_guard.slice(),
            b_guard.slice(),
            w_guard.slice(),
            m,
            n,
            eps,
        )
        .expect("CUDA add_rmsnorm_batched_f32 failed");

        let shape = self.shape.clone();
        let out_tensor = Self {
            storage: Storage::from_cuda_slice(out, m * n, self.device()),
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        };
        let sum_tensor = Self {
            storage: Storage::from_cuda_slice(sum_out, m * n, self.device()),
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
            offset: 0,
        };
        (out_tensor, sum_tensor)
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

    /// Trainable-scale RMSNorm gradient: `grad_w[j] = Σ_i grad_out[i, j] · x[i, j] / rms_i`
    /// as `[splits, n]` column partials (reduced by the caller). `self` = saved input `[m, n]`.
    pub(crate) fn rms_norm_bwd_weight_partial_cuda(
        &self,
        grad_output: &Self,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Self {
        let x = self.contiguous_gpu();
        let g = grad_output.contiguous_gpu();
        debug_assert_eq!(x.numel(), m * n);
        debug_assert_eq!(g.numel(), m * n);
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let x_guard = x.storage.as_cuda_slice();
        let g_guard = g.storage.as_cuda_slice();
        let splits = m.clamp(1, 64);
        let rows_per_split = m.div_ceil(splits);
        let splits = m.div_ceil(rows_per_split);
        let mut inv_rms = pool_alloc_uninit(m).expect("GPU pool alloc failed");
        cuda.rms_inv_rows_f32(&mut inv_rms, x_guard.slice(), m, n, eps)
            .expect("CUDA rms_inv_rows_f32 failed");
        let mut partial = pool_alloc_uninit(splits * n).expect("GPU pool alloc failed");
        cuda.rms_norm_bwd_weight_partial_f32(
            &mut partial,
            x_guard.slice(),
            g_guard.slice(),
            &inv_rms,
            m,
            n,
            rows_per_split,
            splits,
        )
        .expect("CUDA rms_norm_bwd_weight_partial_f32 failed");
        pool_free(inv_rms);
        let shape = Shape::from_slice(&[splits, n]);
        let storage = Storage::from_cuda_slice(partial, splits * n, self.device());
        Self {
            storage,
            shape: shape.clone(),
            strides: contiguous_strides(&shape),
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

    /// Head-major split-halves RoPE backward. `self` is `grad_output`
    /// shape `[bs, n_heads, seq, head_dim]`. Returns `grad_input`.
    pub(crate) fn rope_split_halves_bhsd_bwd_cuda(
        &self,
        bs: usize,
        n_heads: usize,
        seq: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Self {
        let g = self.contiguous_gpu();
        let total = bs * n_heads * seq * head_dim;
        debug_assert_eq!(g.numel(), total);
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let g_guard = g.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.rope_split_halves_bhsd_bwd_f32(
            &mut out,
            g_guard.slice(),
            bs,
            n_heads,
            seq,
            head_dim,
            theta,
            pos_start,
        )
        .expect("CUDA rope_split_halves_bhsd_bwd_f32 failed");

        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// GQA repeat_kv on GPU: expand `[bs, kv_heads, seq, head_dim]` to
    /// `[bs, kv_heads * n_rep, seq, head_dim]`. Single kernel, no host traffic.
    pub(crate) fn repeat_kv_cuda(
        &self,
        bs: usize,
        kv_heads: usize,
        n_rep: usize,
        seq: usize,
        head_dim: usize,
    ) -> Self {
        let data = self.contiguous_gpu();
        let total = bs * kv_heads * n_rep * seq * head_dim;
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.repeat_kv_f32(
            &mut out,
            src_guard.slice(),
            bs,
            kv_heads,
            n_rep,
            seq,
            head_dim,
        )
        .expect("CUDA repeat_kv_f32 failed");

        let shape = vec![bs, kv_heads * n_rep, seq, head_dim];
        let storage = Storage::from_cuda_slice(out, total, self.device());
        Self {
            storage,
            shape: shape.clone().into(),
            strides: contiguous_strides(&shape),
            offset: 0,
        }
    }

    /// Head-major split-halves RoPE. `self` shape `[bs, n_heads, seq, head_dim]`
    /// contiguous on GPU. Rotates each (b, h, t) token at position
    /// `pos_start + t`. Matches the shape Qwen3/LLaMA training produces after
    /// reshape+transpose — avoids the transpose-to-token-major round-trip
    /// the batched kernel would otherwise need.
    pub(crate) fn apply_rope_split_halves_bhsd_cuda(
        &self,
        bs: usize,
        n_heads: usize,
        seq: usize,
        head_dim: usize,
        theta: f32,
        pos_start: usize,
    ) -> Self {
        let data = self.contiguous_gpu();
        let total = bs * n_heads * seq * head_dim;
        debug_assert_eq!(data.numel(), total, "rope_bhsd: shape mismatch");
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = pool_alloc_uninit(total).expect("GPU pool alloc failed");

        cuda.rope_split_halves_bhsd_f32(
            &mut out,
            src_guard.slice(),
            bs,
            n_heads,
            seq,
            head_dim,
            theta,
            pos_start,
        )
        .expect("CUDA rope_split_halves_bhsd_f32 failed");

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

// ── multi-tensor plans: one launch over many tensors (clip, ternary STE) ──
/// Device-side pointer/length/block tables for a fixed set of same-device f32 tensors.
pub struct MtPlan {
    /// Device pointer of each tensor.
    pub ptrs: cudarc::driver::CudaSlice<u64>,
    /// Element count of each tensor.
    pub lens: cudarc::driver::CudaSlice<u32>,
    /// Tensor index for each launch block.
    pub block_tensor: cudarc::driver::CudaSlice<u32>,
    /// Element offset within its tensor for each launch block.
    pub block_offset: cudarc::driver::CudaSlice<u32>,
    /// Total launch blocks across all tensors.
    pub n_blocks: usize,
    /// Number of tensors in the plan.
    pub n_tensors: usize,
    /// Device every tensor in the plan lives on.
    pub device: Device,
    /// Handles kept alive for the plan's lifetime: the pointer table must never outlive its buffers.
    pub held: Vec<Tensor<f32>>,
}

impl MtPlan {
    /// Build the tables; each block covers 2*BLOCK_SIZE elements of one tensor.
    pub fn new(tensors: &[&Tensor<f32>]) -> MtPlan {
        use cudarc::driver::DevicePtr;
        let cuda = get_cuda_backend().expect("CUDA backend");
        let stream = cuda.stream();
        let per = 2 * axonml_core::backends::cuda_kernels::BLOCK_SIZE as usize;
        let (mut ptrs, mut lens, mut bt, mut bo) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut held: Vec<Tensor<f32>> = Vec::with_capacity(tensors.len());
        for (ti, t) in tensors.iter().enumerate() {
            assert!(
                t.is_contiguous(),
                "MtPlan: tensor {ti} must be contiguous (make it so at the source; a copy here would not be written back)"
            );
            let t = (*t).clone();
            let base = {
                let g = t.storage.as_cuda_slice();
                let (p, _guard) = g.slice().device_ptr(stream);
                p as u64
            };
            ptrs.push(base + (t.offset * 4) as u64);
            let n = t.numel();
            lens.push(n as u32);
            let mut off = 0usize;
            while off < n {
                bt.push(ti as u32);
                bo.push(off as u32);
                off += per;
            }
            held.push(t);
        }
        let device = tensors
            .first()
            .map(|t| t.device())
            .unwrap_or(Device::Cuda(0));
        MtPlan {
            ptrs: cuda.upload_u64(&ptrs).expect("mt ptrs"),
            lens: cuda.upload_u32(&lens).expect("mt lens"),
            block_tensor: cuda.upload_u32(&bt).expect("mt bt"),
            block_offset: cuda.upload_u32(&bo).expect("mt bo"),
            n_blocks: bt.len(),
            n_tensors: tensors.len(),
            device,
            held,
        }
    }

    /// Sum of squares over every tensor, as a device [1] tensor (no host sync).
    pub fn sumsq(&self) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend");
        let mut out = pool_alloc(1).expect("mt out");
        cuda.memset_zeros_f32(&mut out).expect("memset");
        cuda.mt_reduce_f32(
            "mt_sumsq_f32",
            &self.ptrs,
            &self.lens,
            &self.block_tensor,
            &self.block_offset,
            &mut out,
            self.n_blocks,
        )
        .expect("mt_sumsq");
        Tensor::from_storage(Storage::from_cuda_slice(out, 1, self.device), &[1])
            .expect("mt tensor")
    }

    /// Per-tensor sum of |x|, as a device [n_tensors] tensor.
    pub fn abssums(&self) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend");
        let mut out = pool_alloc(self.n_tensors).expect("mt out");
        cuda.memset_zeros_f32(&mut out).expect("memset");
        cuda.mt_reduce_f32(
            "mt_abssum_f32",
            &self.ptrs,
            &self.lens,
            &self.block_tensor,
            &self.block_offset,
            &mut out,
            self.n_blocks,
        )
        .expect("mt_abssum");
        Tensor::from_storage(
            Storage::from_cuda_slice(out, self.n_tensors, self.device),
            &[self.n_tensors],
        )
        .expect("mt tensor")
    }

    /// Ternary STE forward of every source tensor into the matching destination (same shapes).
    pub fn ternarize_into(&self, dst: &MtPlan, abssums: &Tensor<f32>) {
        let cuda = get_cuda_backend().expect("CUDA backend");
        let g = abssums.storage.as_cuda_slice();
        cuda.mt_ternarize_f32(
            &self.ptrs,
            &dst.ptrs,
            &self.lens,
            &self.block_tensor,
            &self.block_offset,
            g.slice(),
            self.n_blocks,
        )
        .expect("mt_ternarize");
    }

    /// Scale every tensor in place by a device scalar ([1] tensor).
    pub fn scale_by(&self, factor: &Tensor<f32>) {
        let cuda = get_cuda_backend().expect("CUDA backend");
        let g = factor.storage.as_cuda_slice();
        cuda.mt_scale_f32(
            &self.ptrs,
            &self.lens,
            &self.block_tensor,
            &self.block_offset,
            g.slice(),
            self.n_blocks,
        )
        .expect("mt_scale");
    }
}
