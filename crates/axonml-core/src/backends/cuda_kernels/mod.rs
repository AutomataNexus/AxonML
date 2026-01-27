//! CUDA Kernel Registry
//!
//! Manages loading and caching of CUDA kernels for element-wise operations.
//! Kernels are compiled from PTX at runtime using cudarc.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use super::cuda::CudaError;

/// Block size for kernel launches (256 threads per block is typical optimal)
pub const BLOCK_SIZE: u32 = 256;

/// Embedded PTX for element-wise operations
/// These are compiled from .cu files using: nvcc -ptx -arch=sm_50 --use_fast_math
#[cfg(feature = "cuda")]
pub const ELEMENTWISE_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// add_f32 kernel
.visible .entry add_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__exit;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    add.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__exit:
    ret;
}

// scale_f32 kernel
.visible .entry scale_f32(
    .param .u64 data,
    .param .f32 alpha,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [data];
    ld.param.f32 %f1, [alpha];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__exit2;

    cvt.u64.u32 %rd2, %r2;
    shl.b64 %rd3, %rd2, 2;
    add.s64 %rd4, %rd1, %rd3;

    ld.global.f32 %f2, [%rd4];
    mul.f32 %f2, %f2, %f1;
    st.global.f32 [%rd4], %f2;

$L__exit2:
    ret;
}

// mul_f32 kernel
.visible .entry mul_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 out,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__exit3;

    cvt.u64.u32 %rd4, %r2;
    shl.b64 %rd5, %rd4, 2;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    add.s64 %rd8, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    mul.f32 %f1, %f1, %f2;
    st.global.f32 [%rd8], %f1;

$L__exit3:
    ret;
}
"#;

/// Embedded PTX for activation functions
#[cfg(feature = "cuda")]
pub const ACTIVATIONS_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// relu_f32 kernel
.visible .entry relu_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__relu_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    max.f32 %f1, %f1, 0f00000000;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f1;

$L__relu_exit:
    ret;
}

// sigmoid_f32 kernel (using ex2.approx for fast exp)
.visible .entry sigmoid_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<5>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__sig_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    neg.f32 %f1, %f1;
    mul.f32 %f1, %f1, 0f3FB8AA3B;  // 1/ln(2)
    ex2.approx.f32 %f2, %f1;
    add.f32 %f3, %f2, 0f3F800000;  // 1.0
    rcp.approx.f32 %f4, %f3;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f4;

$L__sig_exit:
    ret;
}

// tanh_f32 kernel
.visible .entry tanh_f32(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<8>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra $L__tanh_exit;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;

    ld.global.f32 %f1, [%rd5];
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    mul.f32 %f2, %f1, 0f40000000;  // 2.0
    mul.f32 %f2, %f2, 0f3FB8AA3B;  // 1/ln(2)
    ex2.approx.f32 %f3, %f2;
    add.f32 %f4, %f3, 0fBF800000;  // -1.0
    add.f32 %f5, %f3, 0f3F800000;  // 1.0
    div.approx.f32 %f6, %f4, %f5;

    add.s64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f6;

$L__tanh_exit:
    ret;
}
"#;

/// CUDA Kernel registry for managing loaded kernels
#[cfg(feature = "cuda")]
pub struct CudaKernels {
    device: Arc<CudaDevice>,
    functions: HashMap<String, CudaFunction>,
}

#[cfg(feature = "cuda")]
impl CudaKernels {
    /// Load kernels from embedded PTX
    pub fn load(device: Arc<CudaDevice>) -> Result<Self, CudaError> {
        let mut kernels = Self {
            device,
            functions: HashMap::new(),
        };

        // Load element-wise kernels
        kernels.load_module(
            "elementwise",
            ELEMENTWISE_PTX,
            &["add_f32", "mul_f32", "scale_f32"],
        )?;

        // Load activation kernels
        kernels.load_module(
            "activations",
            ACTIVATIONS_PTX,
            &["relu_f32", "sigmoid_f32", "tanh_f32"],
        )?;

        Ok(kernels)
    }

    fn load_module(
        &mut self,
        name: &'static str,
        ptx: &'static str,
        functions: &'static [&'static str],
    ) -> Result<(), CudaError> {
        self.device
            .load_ptx(ptx.into(), name, functions)
            .map_err(|e| CudaError::ModuleLoadFailed(e.to_string()))?;

        for func_name in functions {
            let func = self
                .device
                .get_func(name, func_name)
                .ok_or_else(|| CudaError::KernelNotFound(func_name.to_string()))?;
            self.functions.insert(func_name.to_string(), func);
        }

        Ok(())
    }

    /// Get a kernel function by name
    pub fn get(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }

    /// Check if a kernel is available
    pub fn has(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

/// Compute optimal launch configuration for a given number of elements
#[cfg(feature = "cuda")]
pub fn launch_config(n: usize) -> LaunchConfig {
    let num_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config() {
        let cfg = launch_config(1000);
        assert_eq!(cfg.block_dim, (256, 1, 1));
        assert_eq!(cfg.grid_dim, (4, 1, 1)); // ceil(1000/256) = 4
    }

    #[test]
    fn test_launch_config_large() {
        let cfg = launch_config(1_000_000);
        assert_eq!(cfg.grid_dim, (3907, 1, 1)); // ceil(1000000/256) = 3907
    }
}
