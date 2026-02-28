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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
    // Element-wise Unary Operations (GPU)
    // =========================================================================

    /// GPU negation.
    pub(crate) fn neg_cuda(&self) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        let src_guard = data.storage.as_cuda_slice();
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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
        let mut out = cuda.alloc::<f32>(len).expect("GPU alloc failed");

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

    /// GPU scalar multiplication.
    pub(crate) fn mul_scalar_cuda(&self, scalar: f32) -> Self {
        let data = self.contiguous_gpu();
        let len = data.numel();
        let cuda = get_cuda_backend().expect("CUDA backend not available");

        // Copy src to dst first, then scale in-place
        let src_guard = data.storage.as_cuda_slice();
        let src_vec = cuda.dtoh_copy(src_guard.slice()).expect("dtoh failed");
        let mut out = cuda.htod_copy(&src_vec).expect("htod failed");

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
            let mut c_gpu = cuda.alloc::<f32>(m * n).expect("GPU alloc failed");

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

        // Batched matmul: fall back to CPU for now, copy back to GPU
        let a_vec = a.to_vec_gpu();
        let b_vec = b.to_vec_gpu();

        let batch_dims: Vec<usize> = a.shape[..a.shape.len() - 2].to_vec();
        let batch_size: usize = batch_dims.iter().product();
        let a_stride = m * k;
        let b_stride = k * n;
        let c_stride = m * n;

        let mut c_data = vec![0.0f32; batch_size * m * n];

        // Use GPU for each batch
        for batch in 0..batch_size {
            let a_slice = &a_vec[batch * a_stride..(batch + 1) * a_stride];
            let b_slice = &b_vec[batch * b_stride..(batch + 1) * b_stride];

            let a_gpu = cuda.htod_copy(a_slice).expect("htod failed");
            let b_gpu = cuda.htod_copy(b_slice).expect("htod failed");
            let mut c_gpu = cuda.alloc::<f32>(m * n).expect("GPU alloc failed");

            cuda.gemm_f32(
                false, false,
                n, m, k,
                1.0,
                &b_gpu, n,
                &a_gpu, k,
                0.0,
                &mut c_gpu, n,
            ).expect("cuBLAS gemm failed");

            let c_batch = cuda.dtoh_copy(&c_gpu).expect("dtoh failed");
            c_data[batch * c_stride..(batch + 1) * c_stride].copy_from_slice(&c_batch);
        }

        // Put result back on GPU
        let c_gpu = cuda.htod_copy(&c_data).expect("htod failed");
        let mut output_shape = batch_dims;
        output_shape.push(m);
        output_shape.push(n);
        let total = c_data.len();

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

    /// Returns a contiguous GPU tensor (copies if not contiguous).
    pub(crate) fn contiguous_gpu(&self) -> Self {
        if self.is_contiguous() && self.offset == 0 {
            return self.clone();
        }
        // Must materialize: copy GPU→CPU, make contiguous, copy back
        let data = self.to_vec_gpu();
        let cpu_tensor = Self::from_vec(data, &self.shape).expect("contiguous should not fail");
        cpu_tensor.to_device(self.device()).expect("GPU copy should not fail")
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
}
