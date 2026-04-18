//! weight — Matrix-Weight Abstraction + CUDA Dispatch
//!
//! The [`Weight`] enum wraps every shape a single matmul-weight can take in
//! nexus-serve:
//! - [`Weight::F32`]: fully dequantized, pre-transposed `Tensor<f32>` — fast
//!   matmul, high memory. Can live on CPU or GPU via `to_device`.
//! - [`Weight::Quantized`]: raw GGUF-packed bytes on CPU (Q4_0, Q4_K, Q6_K,
//!   Q8_0, F16, BF16, I2S/BitNet). Stores `dims[0]=in, dims[1]=out` + the
//!   `GgmlType`. On CUDA builds carries an `OnceLock<CudaSlice<u8>>`
//!   `gpu_cache` that lazily holds the one-time H2D upload of `data` —
//!   reused on every GEMV/GEMM dispatch, eliminates the per-matmul ~15-80
//!   MB H2D copy that dominated decode bandwidth (see WORK_STATE W17).
//! - [`Weight::QuantizedGpu`]: (reserved) GPU-resident quantized bytes. Not
//!   used in session 2 — construction panics to catch accidents.
//!
//! Matmul dispatch ([`Weight::matmul`]):
//! 1. `F32`: direct `Tensor::matmul`.
//! 2. `Quantized` + Q4_K + GPU input → cache-upload + `q4k_gemv_cuda` /
//!    `q4k_gemm_cuda` (dequant-in-shader).
//! 3. `Quantized` + Q6_K + GPU input → cache-upload + `q6k_gemv_cuda` /
//!    `q6k_gemm_cuda`.
//! 4. `Quantized` + I2S (BitNet) → fused CPU [`i2s_matmul`] (ternary
//!    add-only, never materializes f32 weights).
//! 5. Fallback: [`Weight::cpu_dequant_matmul`] — dequantize into `[out, in]`
//!    then `CpuBackend::matmul_f32_bt` (parallel GEMV at m=1, zero-copy
//!    stride-transposed sgemm at m>1). Avoids the old `transpose(0,1).
//!    contiguous()` memcpy that pegged one core.
//!
//! Support helpers:
//! - [`dequantize_into`]: dispatches per-dtype to the block kernels in
//!   `super::gguf` (`dequantize_q4_k`, `dequantize_q6_k`, etc.) via
//!   [`dequantize_blocks_par`] — a BATCH=64 rayon parallel block driver that
//!   amortizes task-scheduling overhead across all cores.
//! - [`i2s_matmul`]: pulls the trailing 4-byte f32 tensor-wide scale that
//!   `MappedGguf::load_tensor_raw` appends, then calls
//!   `axonml_quant::bitnet::matmul_i2s`.
//!
//! Instrumentation (env-gated, zero cost when off):
//! - `I2S_TRACE=1` → prints avg total / kernel / overhead every 100 I2S
//!   calls (via `I2S_CALLS` / `I2S_KERNEL_NS` / `I2S_TOTAL_NS`).
//! - `MM_TRACE=1` → same for general CPU dequant+matmul (via `MM_CALLS` /
//!   `MM_DEQUANT_NS` / `MM_KERNEL_NS` / `MM_TOTAL_NS`).
//! - W17 one-shot dispatch markers (`Q4K_GPU_FIRED`, `CPU_DEQUANT_FIRED`,
//!   `Q4K_GPU_SKIPPED_CPU_INPUT`, `Q6K_GPU_FIRED`, `Q6K_GPU_SKIPPED_CPU_INPUT`)
//!   print the first time each branch fires during a request.
//!
//! The transpose is applied implicitly: quantized data is stored as the
//! physical GGUF layout (rows=out, cols=in) and dequantized into a
//! [out, in] scratch buffer, which is then used as an f32 tensor that
//! gets transposed to [in, out] for the matmul (same convention as the
//! pre-dequantized path).
//!
//! # File
//! `nexus-serve/src/model/weight.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use axonml_core::Device;
use axonml_quant::bitnet;
use axonml_tensor::Tensor;
use rayon::prelude::*;

use super::gguf::{self, GgmlType};

// =============================================================================
// Instrumentation
// =============================================================================

// Instrumentation for BitNet ternary matmul — prints stats every N calls so
// we can see per-call wall time and identify whether the kernel or the
// surrounding Tensor↔Vec conversion is dominating. Set env `I2S_TRACE=1`
// to enable. Zero cost otherwise.
static I2S_CALLS: AtomicU64 = AtomicU64::new(0);
static I2S_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static I2S_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

fn i2s_trace_enabled() -> bool {
    std::env::var("I2S_TRACE").map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}

// Same idea for the general CPU dequant+matmul path (Q4_K / Q6_K / Q8_0 /
// F16 etc). Set `MM_TRACE=1` to see where time is going — dequant,
// kernel, or tensor/alloc overhead.
static MM_CALLS: AtomicU64 = AtomicU64::new(0);
static MM_DEQUANT_NS: AtomicU64 = AtomicU64::new(0);
static MM_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static MM_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

fn matmul_trace_enabled() -> bool {
    std::env::var("MM_TRACE").map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}

// One-shot dispatch diagnostics for W17 step 1: which Q4_K path actually
// fires during Oracle prefill? Each of these prints the first time its
// branch is taken, so a single Oracle request makes it obvious whether
// we're on the GPU kernel, falling back to CPU dequant, or missing the
// GPU branch entirely due to an input-device mismatch. Zero cost after
// the first fire.
#[cfg(feature = "cuda")]
static Q4K_GPU_FIRED: AtomicBool = AtomicBool::new(false);
static CPU_DEQUANT_FIRED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda")]
static Q4K_GPU_SKIPPED_CPU_INPUT: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda")]
static Q6K_GPU_FIRED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda")]
static Q6K_GPU_SKIPPED_CPU_INPUT: AtomicBool = AtomicBool::new(false);

// =============================================================================
// Weight Enum
// =============================================================================

/// A weight matrix: either pre-dequantized (fast, big) or lazily dequantized (slow, small).
pub enum Weight {
    /// Pre-dequantized f32 tensor, pre-transposed to `[in, out]` layout.
    F32(Tensor<f32>),

    /// Quantized bytes stored on CPU. Dequantized to scratch f32 per matmul.
    /// `dims` are GGUF's `[n_cols, n_rows]` (dims[0]=in, dims[1]=out).
    ///
    /// W17 step 2: when cuda is enabled, `gpu_cache` lazily holds a single
    /// GPU upload of `data` populated on the first successful Q4_K/Q6_K
    /// GPU-kernel matmul. Every subsequent matmul reuses it — eliminates
    /// the ~15-80 MB per-matmul `htod_copy` that dominated decode H2D
    /// bandwidth (see WORK_STATE W17). The CPU `data` copy stays
    /// authoritative (Quantized is still the CPU variant).
    Quantized {
        data: Vec<u8>,
        dims: Vec<usize>,
        dtype: GgmlType,
        #[cfg(feature = "cuda")]
        gpu_cache: std::sync::OnceLock<cudarc::driver::CudaSlice<u8>>,
    },

    /// Quantized bytes stored on GPU. Dispatched through a custom CUDA kernel
    /// that dequants each super-block in-shader during matmul — saves the
    /// dequant-to-scratch + H2D cost of the CPU `Quantized` variant.
    ///
    /// Session 2 scope (2026-04-13): Q4_K GEMV only. Other dtype / GEMM shapes
    /// fall back to the CPU `Quantized` path via `Weight::matmul`.
    #[cfg(feature = "cuda")]
    QuantizedGpu {
        data: cudarc::driver::CudaSlice<u8>,
        dims: Vec<usize>,
        dtype: GgmlType,
    },
}

impl Weight {
    /// Construct from a dequantized f32 tensor (pre-transposed).
    pub fn from_f32(tensor: Tensor<f32>) -> Self {
        Weight::F32(tensor)
    }

    /// Construct by copying GGUF-quantized bytes.
    /// `dims[0]` = in_features, `dims[1]` = out_features.
    pub fn from_quantized(data: Vec<u8>, dims: Vec<usize>, dtype: GgmlType) -> Self {
        Weight::Quantized {
            data,
            dims,
            dtype,
            #[cfg(feature = "cuda")]
            gpu_cache: std::sync::OnceLock::new(),
        }
    }

    /// Logical shape of the weight as `[in, out]` (post-transpose convention).
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Weight::F32(t) => t.shape().to_vec(),
            Weight::Quantized { dims, .. } => dims.clone(),
            #[cfg(feature = "cuda")]
            Weight::QuantizedGpu { dims, .. } => dims.clone(),
        }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        match self {
            Weight::F32(t) => t.numel(),
            Weight::Quantized { dims, .. } => dims.iter().product(),
            #[cfg(feature = "cuda")]
            Weight::QuantizedGpu { dims, .. } => dims.iter().product(),
        }
    }

    /// Compressed bytes used in RAM / VRAM.
    pub fn bytes(&self) -> usize {
        match self {
            Weight::F32(t) => t.numel() * 4,
            Weight::Quantized { data, .. } => data.len(),
            #[cfg(feature = "cuda")]
            Weight::QuantizedGpu { data, .. } => data.len(),
        }
    }

    /// Move to device. Only affects F32 variant — quantized data stays on CPU
    /// (dequantization produces CPU scratch which is moved per-matmul).
    ///
    /// Session 2 note: the Q4_K GPU kernel exists and is dispatched by
    /// `Weight::matmul`, but we intentionally do NOT auto-upload quantized
    /// bytes at load time — prefill (`m > 1`) still uses the CPU dequant
    /// path, and having a GPU-only variant would force expensive D2H
    /// round-trips on every prefill matmul. Instead, `Weight::matmul` does
    /// an ephemeral H2D upload on GEMV (`m == 1`) and runs the kernel. A
    /// caching layer is a session-3 optimization.
    pub fn to_device(&mut self, device: Device) {
        if let Weight::F32(t) = self {
            if let Ok(moved) = t.to_device(device) {
                *t = moved;
            }
        }
    }

    /// Matmul: input `[m, in]` @ self `[in, out]` → output `[m, out]`.
    ///
    /// Dispatch:
    ///   - `F32`: direct tensor matmul (GPU if tensor is on GPU, CPU otherwise).
    ///   - `Quantized` + Q4_K + GEMV (m=1) + GPU input → ephemeral upload to
    ///      GPU, run `q4k_gemv_f32` dequant-in-shader kernel. Session-2 fast
    ///      path for decode.
    ///   - `Quantized` otherwise: dequantize bytes into CPU scratch `[out, in]`,
    ///      transpose to `[in, out]`, matmul.
    ///   - `QuantizedGpu`: (not used by default in session 2 — reserved for
    ///      session-3 cached-upload optimization). Falls through to panic to
    ///      catch any accidental construction during testing.
    pub fn matmul(&self, input: &Tensor<f32>) -> Tensor<f32> {
        match self {
            Weight::F32(t) => input.matmul(t).expect("matmul failed"),
            #[cfg(feature = "cuda")]
            Weight::QuantizedGpu { .. } => {
                panic!(
                    "QuantizedGpu matmul hit — session 2 keeps weights CPU-resident \
                     and uploads per-GEMV; QuantizedGpu variant is reserved for a \
                     future cached-upload session."
                );
            }
            Weight::Quantized {
                data,
                dims,
                dtype,
                #[cfg(feature = "cuda")]
                gpu_cache,
            } => {
                // Q4_K / Q6_K GPU fast path: when input is on GPU and the
                // weight dtype has a dequant-in-shader kernel, dispatch to
                // the kernel. W17 step 2: the weight bytes are uploaded to
                // GPU ONCE via `OnceLock::get_or_init` and reused on every
                // subsequent matmul. Before this, each matmul re-uploaded
                // its weight slice (~15-80 MB) on the critical decode
                // path, which was the dominant H2D cost and dropped decode
                // tok/s by ~30-40% on DeepSeek-7B. The OnceLock is inside
                // the `Quantized` variant so cache lifetime == weight
                // lifetime; no manual invalidation needed.
                #[cfg(feature = "cuda")]
                {
                    if *dtype == GgmlType::Q4K
                        && dims.len() == 2
                        && dims[0] % 256 == 0
                        && input.device().is_gpu()
                    {
                        if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                            let w_gpu = gpu_cache.get_or_init(|| {
                                cuda.htod_copy(data.as_slice())
                                    .expect("Q4_K gpu_cache htod_copy failed")
                            });
                            let m_shape = input.shape().first().copied().unwrap_or(1);
                            if !Q4K_GPU_FIRED.swap(true, Ordering::Relaxed) {
                                eprintln!(
                                    "[W17-dispatch] Q4_K GPU kernel FIRED (first time, cached) \
                                     m={m_shape} in={} out={} input_dev={:?}",
                                    dims[0], dims[1], input.device(),
                                );
                            }
                            // GEMV (m=1, decode) and GEMM (m>1, prefill)
                            // both go to the GPU kernel; only the launch
                            // geometry differs.
                            let result = if m_shape == 1 {
                                input.q4k_gemv_cuda(w_gpu, dims[1], dims[0])
                            } else {
                                input.q4k_gemm_cuda(w_gpu, dims[1], dims[0])
                            };
                            return result.expect("q4k_{gemv,gemm}_cuda failed");
                        }
                    } else if *dtype == GgmlType::Q4K
                        && dims.len() == 2
                        && dims[0] % 256 == 0
                        && !input.device().is_gpu()
                        && !Q4K_GPU_SKIPPED_CPU_INPUT.swap(true, Ordering::Relaxed)
                    {
                        eprintln!(
                            "[W17-dispatch] Q4_K GPU kernel SKIPPED — input on CPU \
                             (first time). in={} out={} input_dev={:?}",
                            dims[0], dims[1], input.device(),
                        );
                    }

                    // Q6_K GEMV/GEMM GPU fast path (W17 step 5 + step 2).
                    if *dtype == GgmlType::Q6K
                        && dims.len() == 2
                        && dims[0] % 256 == 0
                        && input.device().is_gpu()
                    {
                        if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                            let w_gpu = gpu_cache.get_or_init(|| {
                                cuda.htod_copy(data.as_slice())
                                    .expect("Q6_K gpu_cache htod_copy failed")
                            });
                            let m_shape = input.shape().first().copied().unwrap_or(1);
                            if !Q6K_GPU_FIRED.swap(true, Ordering::Relaxed) {
                                eprintln!(
                                    "[W17-dispatch] Q6_K GPU kernel FIRED (first time, cached) \
                                     m={m_shape} in={} out={} input_dev={:?}",
                                    dims[0], dims[1], input.device(),
                                );
                            }
                            let result = if m_shape == 1 {
                                input.q6k_gemv_cuda(w_gpu, dims[1], dims[0])
                            } else {
                                input.q6k_gemm_cuda(w_gpu, dims[1], dims[0])
                            };
                            return result.expect("q6k_{gemv,gemm}_cuda failed");
                        }
                    } else if *dtype == GgmlType::Q6K
                        && dims.len() == 2
                        && dims[0] % 256 == 0
                        && !input.device().is_gpu()
                        && !Q6K_GPU_SKIPPED_CPU_INPUT.swap(true, Ordering::Relaxed)
                    {
                        eprintln!(
                            "[W17-dispatch] Q6_K GPU kernel SKIPPED — input on CPU \
                             (first time). in={} out={} input_dev={:?}",
                            dims[0], dims[1], input.device(),
                        );
                    }

                    // Q5_K GEMV/GEMM GPU fast path — Phi-3's attn_qkv
                    // lives here (Mistral Q5_K_M bodies too). Before this
                    // path landed, Q5_K weights were eagerly dequanted to
                    // F32 at load time and hit the plain F32 matmul on
                    // every decode — Phi-3-mini decode was 1.66 tok/s
                    // because of this one tensor type.
                    if *dtype == GgmlType::Q5K
                        && dims.len() == 2
                        && dims[0] % 256 == 0
                        && input.device().is_gpu()
                    {
                        if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                            let w_gpu = gpu_cache.get_or_init(|| {
                                cuda.htod_copy(data.as_slice())
                                    .expect("Q5_K gpu_cache htod_copy failed")
                            });
                            let _ = cuda;
                            let m_shape = input.shape().first().copied().unwrap_or(1);
                            let result = if m_shape == 1 {
                                input.q5k_gemv_cuda(w_gpu, dims[1], dims[0])
                            } else {
                                input.q5k_gemm_cuda(w_gpu, dims[1], dims[0])
                            };
                            return result.expect("q5k_{gemv,gemm}_cuda failed");
                        }
                    }

                    // Q5_0 / Q5_1 GPU fast path — legacy Falcon (Falcon-
                    // 7B/40B) bodies. Before this landed, both types
                    // fell through to `cpu_dequant_matmul` which re-
                    // dequants the whole weight to f32 on CPU every
                    // call — Falcon-7B decode was 0.4 tok/s because
                    // of this.
                    if (*dtype == GgmlType::Q5_0 || *dtype == GgmlType::Q5_1)
                        && dims.len() == 2
                        && dims[0] % 32 == 0
                        && input.device().is_gpu()
                    {
                        if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                            let w_gpu = gpu_cache.get_or_init(|| {
                                cuda.htod_copy(data.as_slice())
                                    .expect("Q5_0/Q5_1 gpu_cache htod_copy failed")
                            });
                            let _ = cuda;
                            let m_shape = input.shape().first().copied().unwrap_or(1);
                            let result = match (*dtype, m_shape) {
                                (GgmlType::Q5_0, 1) => input.q5_0_gemv_cuda(w_gpu, dims[1], dims[0]),
                                (GgmlType::Q5_0, _) => input.q5_0_gemm_cuda(w_gpu, dims[1], dims[0]),
                                (GgmlType::Q5_1, 1) => input.q5_1_gemv_cuda(w_gpu, dims[1], dims[0]),
                                (GgmlType::Q5_1, _) => input.q5_1_gemm_cuda(w_gpu, dims[1], dims[0]),
                                _ => unreachable!(),
                            };
                            return result.expect("q5_{0,1}_{gemv,gemm}_cuda failed");
                        }
                    }

                    // Q8_0 GPU fast path — Falcon-7B LM head. Before this
                    // landed, the 4544 × 65024 LM head re-dequanted on CPU
                    // every decode token, dominating Falcon's 10 tok/s
                    // budget.
                    if *dtype == GgmlType::Q8_0
                        && dims.len() == 2
                        && dims[0] % 32 == 0
                        && input.device().is_gpu()
                    {
                        if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                            let w_gpu = gpu_cache.get_or_init(|| {
                                cuda.htod_copy(data.as_slice())
                                    .expect("Q8_0 gpu_cache htod_copy failed")
                            });
                            let _ = cuda;
                            let m_shape = input.shape().first().copied().unwrap_or(1);
                            let result = if m_shape == 1 {
                                input.q8_0_gemv_cuda(w_gpu, dims[1], dims[0])
                            } else {
                                input.q8_0_gemm_cuda(w_gpu, dims[1], dims[0])
                            };
                            return result.expect("q8_0_{gemv,gemm}_cuda failed");
                        }
                    }
                }
                // BitNet I2_S fast path: the whole point of 1.58-bit is an
                // add-only matmul that never materializes f32 weights. Skip
                // the CPU dequant+GEMM path entirely.
                //
                // This path is CPU-only regardless of input device; BitNet's
                // performance story is CPU-native. If inputs are on GPU, pull
                // them back for this layer. (A GPU ternary kernel is a
                // future optimization — would need a CUDA port of
                // axonml_quant::bitnet::matmul_i2s.)
                if *dtype == GgmlType::I2S {
                    return i2s_matmul(data, dims, input);
                }
                // Fallback: classic CPU dequant-into-scratch path.
                // (Pulled from the original implementation below.)
                Self::cpu_dequant_matmul(data, dims, *dtype, input)
            }
        }
    }

    /// CPU dequant + matmul path.
    ///
    /// Shapes (GGUF convention):
    /// - `dims[0] = in_features`, `dims[1] = out_features`
    /// - Raw bytes decode to `out × in` row-major (this IS the physical GGUF
    ///   layout of a weight tensor).
    /// - Input is `[m, in]`; output is `[m, out]`.
    ///
    /// The old implementation wrapped the dequant buffer in a `Tensor` shaped
    /// `[out, in]` and called `Tensor::transpose(0, 1).contiguous()` to flip
    /// it to `[in, out]` before `Tensor::matmul`. That transpose is a
    /// single-threaded `O(out*in)` memcpy — measured at >30 s/token on 14B
    /// decode, pegging one core and starving the 23 rayon workers.
    ///
    /// New path: call `CpuBackend::matmul_f32_bt` directly with the
    /// dequantized `[out, in]` buffer. That computes `C = A @ B^T` where
    /// `B` is `[n, k] = [out, in]` — exactly what we want — via a parallel
    /// GEMV for `m=1` (decode) or a zero-copy stride-transposed sgemm for
    /// `m>1` (prefill). No transpose, no copy, scales to all cores.
    fn cpu_dequant_matmul(
        data: &[u8],
        dims: &[usize],
        dtype: GgmlType,
        input: &Tensor<f32>,
    ) -> Tensor<f32> {
        if !CPU_DEQUANT_FIRED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[W17-dispatch] cpu_dequant_matmul FIRED (first time) \
                 dtype={:?} in={} out={} input_dev={:?}",
                dtype, dims[0], dims[1], input.device(),
            );
        }
        let trace = matmul_trace_enabled();
        let t0 = if trace { Some(std::time::Instant::now()) } else { None };

        let in_features = dims[0];
        let out_features = dims[1];
        let n_elem = in_features * out_features;

        // Dequantize to [out, in] (physical GGUF layout).
        let mut weight_buf = vec![0.0f32; n_elem];
        dequantize_into(&mut weight_buf, data, n_elem, dtype);
        let t_dequant = t0.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);

        // Ensure input is on CPU in a contiguous row-major layout.
        let input_cpu = if input.device().is_gpu() {
            input
                .to_device(Device::Cpu)
                .expect("cpu_dequant_matmul: failed to move input to CPU")
        } else {
            input.clone()
        };
        let input_shape = input_cpu.shape().to_vec();
        let m = input_shape.iter().take(input_shape.len() - 1).product::<usize>();
        let k = *input_shape.last().unwrap_or(&0);
        debug_assert_eq!(k, in_features);

        let acts: Vec<f32> = input_cpu.to_vec();
        let mut out_buf: Vec<f32> = Vec::with_capacity(m * out_features);
        // SAFETY: matmul_f32_bt writes every element before any read.
        unsafe { out_buf.set_len(m * out_features); }
        let t_before_mm = t0.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);

        axonml_core::backends::cpu::CpuBackend::matmul_f32_bt(
            &mut out_buf,
            &acts,
            &weight_buf,
            m,
            out_features,
            in_features,
        );
        let t_after_mm = t0.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);

        // Build the output tensor with the final shape (input's leading dims
        // preserved, last dim becomes out_features).
        let mut out_shape = input_shape;
        *out_shape.last_mut().unwrap() = out_features;
        let out_cpu = Tensor::from_vec(out_buf, &out_shape)
            .expect("cpu_dequant_matmul: failed to build output tensor");

        if trace {
            let total = t0.unwrap().elapsed().as_nanos() as u64;
            MM_CALLS.fetch_add(1, Ordering::Relaxed);
            MM_DEQUANT_NS.fetch_add(t_dequant, Ordering::Relaxed);
            MM_KERNEL_NS.fetch_add(t_after_mm.saturating_sub(t_before_mm), Ordering::Relaxed);
            MM_TOTAL_NS.fetch_add(total, Ordering::Relaxed);
            let calls = MM_CALLS.load(Ordering::Relaxed);
            if calls % 200 == 0 {
                let deq = MM_DEQUANT_NS.load(Ordering::Relaxed) as f64 / calls as f64 / 1e6;
                let kern = MM_KERNEL_NS.load(Ordering::Relaxed) as f64 / calls as f64 / 1e6;
                let tot = MM_TOTAL_NS.load(Ordering::Relaxed) as f64 / calls as f64 / 1e6;
                eprintln!(
                    "[MM] calls={calls} avg_total={tot:.2}ms dequant={deq:.2}ms kernel={kern:.2}ms other={:.2}ms  last m={m} k={in_features} n={out_features} dtype={:?}",
                    tot - deq - kern, dtype,
                );
            }
        }

        if input.device().is_gpu() {
            out_cpu.to_device(input.device()).unwrap_or(out_cpu)
        } else {
            out_cpu
        }
    }
}

// =============================================================================
// Dequantization Helpers
// =============================================================================

/// Dequantize `n_elements` values from `raw_data` using the given `dtype`.
/// Block-based dequantization (Q4_0, Q4_K, Q6_K, Q8_0, I2S) is parallelized via rayon.
fn dequantize_into(output: &mut [f32], raw_data: &[u8], n_elements: usize, dtype: GgmlType) {
    match dtype {
        GgmlType::F32 => {
            for (i, chunk) in raw_data.chunks_exact(4).enumerate().take(n_elements) {
                output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        GgmlType::F16 => gguf::dequantize_f16(raw_data, output),
        GgmlType::BF16 => {
            for (i, chunk) in raw_data.chunks_exact(2).enumerate().take(n_elements) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                output[i] = f32::from_bits((bits as u32) << 16);
            }
        }
        GgmlType::Q8_0 => dequantize_blocks_par(output, raw_data, n_elements, 32, 34, gguf::dequantize_q8_0),
        GgmlType::Q4_0 => dequantize_blocks_par(output, raw_data, n_elements, 32, 18, gguf::dequantize_q4_0),
        GgmlType::Q5_0 => dequantize_blocks_par(output, raw_data, n_elements, 32, 22, gguf::dequantize_q5_0),
        GgmlType::Q5_1 => dequantize_blocks_par(output, raw_data, n_elements, 32, 24, gguf::dequantize_q5_1),
        GgmlType::Q4K => dequantize_blocks_par(output, raw_data, n_elements, 256, 144, gguf::dequantize_q4_k),
        GgmlType::Q5K => dequantize_blocks_par(output, raw_data, n_elements, 256, 176, gguf::dequantize_q5_k),
        GgmlType::Q6K => dequantize_blocks_par(output, raw_data, n_elements, 256, 210, gguf::dequantize_q6_k),
        GgmlType::I2S => {
            // Unreachable in practice: `Weight::matmul` intercepts I2S and
            // routes to the fused `i2s_matmul` path (which handles the
            // trailing tensor-wide scale). If we ever land here, the
            // Weight dispatch is buggy — produce zeros rather than panic.
            eprintln!("WARN: I2S hit generic dequantize path (Weight dispatch bug)");
            for v in output.iter_mut() { *v = 0.0; }
        }
        other => {
            eprintln!("Unsupported quant type for lazy dequant: {:?}", other);
        }
    }
}

// =============================================================================
// BitNet I2_S Matmul
// =============================================================================

/// BitNet I2_S matmul: `input [m, in]` @ ternary-weights `[in, out]` → `[m, out]`.
///
/// The GGUF weight layout is `[out, in]` (rows = output features). The fused
/// `axonml_quant::bitnet::matmul_i2s` kernel computes `acts @ W^T` with
/// `W` shape `[n, k]`, which matches GGUF's physical storage directly — no
/// transpose needed.
///
/// CPU-only: `input` is moved to CPU if it isn't already. The output returns
/// on the same device as the input (moved back at the end) so callers don't
/// have to know about the CPU detour.
fn i2s_matmul(data: &[u8], dims: &[usize], input: &Tensor<f32>) -> Tensor<f32> {
    let trace = i2s_trace_enabled();
    let t_total = if trace { Some(std::time::Instant::now()) } else { None };

    let in_features = dims[0];
    let out_features = dims[1];
    assert!(
        in_features % bitnet::I2S_BLOCK_SIZE == 0,
        "I2S weight in_features ({in_features}) must be a multiple of {}",
        bitnet::I2S_BLOCK_SIZE,
    );

    // Tensor-wide scale lives in the final 4 bytes of `data` — see
    // `MappedGguf::load_tensor_raw` for where we appended it.
    assert!(
        data.len() >= 4,
        "I2S weight buffer missing trailing f32 scale",
    );
    let scale_off = data.len() - 4;
    let scale = f32::from_le_bytes([
        data[scale_off], data[scale_off + 1], data[scale_off + 2], data[scale_off + 3],
    ]);
    let packed = &data[..scale_off];

    let orig_device = input.device();
    let input_cpu = if orig_device.is_gpu() {
        input
            .to_device(Device::Cpu)
            .expect("I2S matmul: failed to move input to CPU")
    } else {
        input.clone()
    };

    let input_shape = input_cpu.shape().to_vec();
    let m = input_shape.iter().take(input_shape.len() - 1).product::<usize>();
    let k = *input_shape.last().unwrap_or(&0);
    assert_eq!(k, in_features, "I2S matmul: input last-dim mismatch");

    let acts: Vec<f32> = input_cpu.to_vec();
    let mut out_buf = vec![0.0f32; m * out_features];

    let t_kernel = if trace { Some(std::time::Instant::now()) } else { None };
    bitnet::matmul_i2s(&acts, m, k, packed, out_features, scale, &mut out_buf);
    let kernel_ns = t_kernel.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);

    let mut out_shape: Vec<usize> = input_shape;
    *out_shape.last_mut().unwrap() = out_features;
    let out_cpu = Tensor::from_vec(out_buf, &out_shape)
        .expect("I2S matmul: failed to build output tensor");

    let result = if orig_device.is_gpu() {
        out_cpu.to_device(orig_device).unwrap_or(out_cpu)
    } else {
        out_cpu
    };

    if let Some(t0) = t_total {
        let total_ns = t0.elapsed().as_nanos() as u64;
        I2S_KERNEL_NS.fetch_add(kernel_ns, Ordering::Relaxed);
        I2S_TOTAL_NS.fetch_add(total_ns, Ordering::Relaxed);
        let calls = I2S_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if calls % 100 == 0 {
            let k_ms = I2S_KERNEL_NS.load(Ordering::Relaxed) as f64 / calls as f64 / 1e6;
            let t_ms = I2S_TOTAL_NS.load(Ordering::Relaxed) as f64 / calls as f64 / 1e6;
            eprintln!(
                "[I2S] calls={calls} avg_total={t_ms:.2}ms avg_kernel={k_ms:.2}ms (conv+alloc overhead={:.2}ms) last_shape m={m} k={k} n={out_features}",
                t_ms - k_ms,
            );
        }
    }

    result
}

// =============================================================================
// Parallel Block Dequantization
// =============================================================================

/// Dequantize a block-based quantization in parallel using rayon.
/// `block_size` = elements per block. `type_size` = bytes per block.
/// `dequant_block(&[u8] bytes, &mut [f32] out)` dequantizes one block.
///
/// Performance note: per-block tasks are ~1 μs each — way below rayon's
/// task-overhead floor (~2-5 μs for schedule + work-steal). Naively fanning
/// one rayon task per 256-element block means overhead dominates and only
/// 2-3 workers effectively engage. Observed 14B Q4_K inference with that
/// layout: 9.3 ms per dequant for 26M elements ≈ 29 % of peak single-core
/// memory bandwidth, i.e. ~3 effective cores saturating. Batching ~BATCH
/// blocks per rayon task gives each worker ~64-128 μs of continuous work,
/// well above the scheduling floor, and rayon's work-stealing spreads the
/// batches across all 24 cores.
fn dequantize_blocks_par(
    output: &mut [f32],
    raw_data: &[u8],
    n_elements: usize,
    block_size: usize,
    type_size: usize,
    dequant_block: fn(&[u8], &mut [f32]),
) {
    const BATCH: usize = 64; // blocks per rayon task
    let n_blocks = n_elements / block_size;
    let chunk_elems = block_size * BATCH;
    let chunk_bytes = type_size * BATCH;

    output
        .par_chunks_mut(chunk_elems)
        .zip(raw_data.par_chunks(chunk_bytes))
        .for_each(|(out_chunk, in_chunk)| {
            // How many complete blocks are actually in this chunk (handles
            // the last partial chunk cleanly).
            let out_blocks = out_chunk.len() / block_size;
            let in_blocks = in_chunk.len() / type_size;
            let mut blocks = out_blocks.min(in_blocks);
            // Respect the global n_blocks cap (never dequant past the tensor).
            // Compute this chunk's starting block index from the chunk's size.
            // Since rayon `par_chunks` guarantees chunks in-order with fixed
            // stride, this is just chunk_idx * BATCH — but we don't get an
            // index here. Instead, clamp based on output length match: if we
            // are the last chunk, `out_blocks` is already the tail count.
            let _ = n_blocks;
            // Work through this chunk's blocks serially — they share L1/L2
            // cache state, so one worker handling BATCH of them has great
            // locality. Rayon spreads CHUNKS across workers.
            for b in 0..blocks.max(0) {
                let in_off = b * type_size;
                let out_off = b * block_size;
                let in_block = &in_chunk[in_off..in_off + type_size];
                let out_block = &mut out_chunk[out_off..out_off + block_size];
                dequant_block(in_block, out_block);
            }
            // Suppress unused warning in release.
            blocks += 0;
        });
}

// =============================================================================
// Fused Q4_K GPU Matmuls for the Decode Path
//
// Collapses the Q/K/V (three) and gate/up (two) separate GEMV launches per
// layer into single fused launches. Each reduces the kernel-launch overhead
// that dominates at single-token decode, where the actual arithmetic per
// output column is tiny compared to the launch + sync cost.
//
// Per token: saves (3 - 1) + (2 - 1) = 3 launches per layer × num_layers.
// On Qwen-7B (28 layers) that's 84 fewer launches per decoded token.
// =============================================================================

/// One-shot diagnostic — prints the first time each fused dispatch succeeds.
#[cfg(feature = "cuda")]
static FUSED_QKV_FIRED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(feature = "cuda")]
static FUSED_GATE_UP_FIRED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fused Q4_K GEMV for Q/K/V projections. Returns `Some((q, k, v))` on the
/// GPU fast path; returns `None` if any of the three weights isn't a
/// GPU-compatible Q4_K `Quantized` variant, the input isn't on GPU, or the
/// shapes don't fit the `in_dim % 256 == 0` kernel requirement — in which
/// case the caller must fall back to three separate `Weight::matmul` calls.
#[cfg(feature = "cuda")]
pub fn fused_qkv_q4k_matmul_gpu(
    q_weight: &Weight,
    k_weight: &Weight,
    v_weight: &Weight,
    input: &Tensor<f32>,
) -> Option<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
    use axonml_core::backends::cuda::get_cuda_backend;
    use axonml_core::storage::Storage;

    if !input.device().is_gpu() {
        return None;
    }
    let cuda = get_cuda_backend()?;

    // Unpack and validate all three weights.
    let (q_data, q_dims) = q4k_quantized_bytes(q_weight)?;
    let (k_data, k_dims) = q4k_quantized_bytes(k_weight)?;
    let (v_data, v_dims) = q4k_quantized_bytes(v_weight)?;

    // All three must share `in_dim`; outputs may differ (GQA makes K/V smaller).
    let in_dim = q_dims[0];
    if k_dims[0] != in_dim || v_dims[0] != in_dim || in_dim % 256 != 0 {
        return None;
    }
    let q_out = q_dims[1];
    let k_out = k_dims[1];
    let v_out = v_dims[1];

    let (q_gpu, k_gpu, v_gpu) = match (q_weight, k_weight, v_weight) {
        (
            Weight::Quantized { gpu_cache: qc, .. },
            Weight::Quantized { gpu_cache: kc, .. },
            Weight::Quantized { gpu_cache: vc, .. },
        ) => {
            let qg = qc.get_or_init(|| {
                cuda.htod_copy(q_data.as_slice())
                    .expect("fused QKV: q gpu_cache htod_copy failed")
            });
            let kg = kc.get_or_init(|| {
                cuda.htod_copy(k_data.as_slice())
                    .expect("fused QKV: k gpu_cache htod_copy failed")
            });
            let vg = vc.get_or_init(|| {
                cuda.htod_copy(v_data.as_slice())
                    .expect("fused QKV: v gpu_cache htod_copy failed")
            });
            (qg, kg, vg)
        }
        _ => return None,
    };

    // Read the input's GPU slice, allocate three output buffers, launch the
    // fused kernel, wrap each output as a Tensor on the same GPU device.
    use axonml_core::backends::cuda_pool::pool_alloc_uninit;
    let in_guard = input.as_cuda_slice_read();
    // Allocate through the pool so the bucket-rounded backing slice matches
    // what later pool_alloc callers expect when they reuse these buckets.
    // (Raw alloc_uninit produces an exact-size slice that, once pool_free'd,
    // poisons the next_power_of_2 bucket with an undersized block, causing
    // a length-mismatch panic in Storage::to_vec_f32 on subsequent reuse.)
    let mut q_out_buf = pool_alloc_uninit(q_out).ok()?;
    let mut k_out_buf = pool_alloc_uninit(k_out).ok()?;
    let mut v_out_buf = pool_alloc_uninit(v_out).ok()?;

    cuda.q4k_gemv_fused_qkv_f32(
        q_gpu,
        k_gpu,
        v_gpu,
        in_guard.slice(),
        &mut q_out_buf,
        &mut k_out_buf,
        &mut v_out_buf,
        q_out,
        k_out,
        v_out,
        in_dim,
    )
    .ok()?;
    drop(in_guard);

    if !FUSED_QKV_FIRED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[W18-dispatch] fused Q4_K QKV kernel FIRED (first time) \
             in={in_dim} q_out={q_out} k_out={k_out} v_out={v_out}"
        );
    }

    let dev = input.device();
    let q_t = Tensor::from_storage(Storage::from_cuda_slice(q_out_buf, q_out, dev.clone()), &[1, q_out]).ok()?;
    let k_t = Tensor::from_storage(Storage::from_cuda_slice(k_out_buf, k_out, dev.clone()), &[1, k_out]).ok()?;
    let v_t = Tensor::from_storage(Storage::from_cuda_slice(v_out_buf, v_out, dev), &[1, v_out]).ok()?;
    Some((q_t, k_t, v_t))
}

static FUSED_QKV_BIAS_FIRED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fused Q4_K QKV matmul WITH bias-add absorbed into the output write.
///
/// Variant of [`fused_qkv_q4k_matmul_gpu`] that additionally applies Q,
/// K, V biases inline. Requires all three biases to be present as GPU
/// tensors (Qwen2 / DeepSeek always have them). Saves three separate
/// elementwise-add kernel launches per layer.
///
/// Returns `None` and lets the caller fall through to the no-bias
/// variant + separate `q.add(b)` etc. if any precondition isn't met.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_bias_q4k_matmul_gpu(
    q_weight: &Weight,
    k_weight: &Weight,
    v_weight: &Weight,
    input: &Tensor<f32>,
    q_bias: &Tensor<f32>,
    k_bias: &Tensor<f32>,
    v_bias: &Tensor<f32>,
) -> Option<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
    use axonml_core::backends::cuda::get_cuda_backend;
    use axonml_core::backends::cuda_pool::pool_alloc_uninit;
    use axonml_core::storage::Storage;

    if !input.device().is_gpu()
        || !q_bias.device().is_gpu()
        || !k_bias.device().is_gpu()
        || !v_bias.device().is_gpu()
    {
        return None;
    }
    let cuda = get_cuda_backend()?;

    let (q_data, q_dims) = q4k_quantized_bytes(q_weight)?;
    let (k_data, k_dims) = q4k_quantized_bytes(k_weight)?;
    let (v_data, v_dims) = q4k_quantized_bytes(v_weight)?;

    let in_dim = q_dims[0];
    if k_dims[0] != in_dim || v_dims[0] != in_dim || in_dim % 256 != 0 {
        return None;
    }
    let q_out = q_dims[1];
    let k_out = k_dims[1];
    let v_out = v_dims[1];
    if q_bias.numel() != q_out || k_bias.numel() != k_out || v_bias.numel() != v_out {
        return None;
    }

    let (q_gpu, k_gpu, v_gpu) = match (q_weight, k_weight, v_weight) {
        (
            Weight::Quantized { gpu_cache: qc, .. },
            Weight::Quantized { gpu_cache: kc, .. },
            Weight::Quantized { gpu_cache: vc, .. },
        ) => {
            let qg = qc.get_or_init(|| {
                cuda.htod_copy(q_data.as_slice())
                    .expect("fused QKV+bias: q gpu_cache htod_copy failed")
            });
            let kg = kc.get_or_init(|| {
                cuda.htod_copy(k_data.as_slice())
                    .expect("fused QKV+bias: k gpu_cache htod_copy failed")
            });
            let vg = vc.get_or_init(|| {
                cuda.htod_copy(v_data.as_slice())
                    .expect("fused QKV+bias: v gpu_cache htod_copy failed")
            });
            (qg, kg, vg)
        }
        _ => return None,
    };

    let in_guard = input.as_cuda_slice_read();
    let q_bias_guard = q_bias.as_cuda_slice_read();
    let k_bias_guard = k_bias.as_cuda_slice_read();
    let v_bias_guard = v_bias.as_cuda_slice_read();

    let mut q_out_buf = pool_alloc_uninit(q_out).ok()?;
    let mut k_out_buf = pool_alloc_uninit(k_out).ok()?;
    let mut v_out_buf = pool_alloc_uninit(v_out).ok()?;

    cuda.q4k_gemv_fused_qkv_bias_f32(
        q_gpu, k_gpu, v_gpu,
        in_guard.slice(),
        q_bias_guard.slice(),
        k_bias_guard.slice(),
        v_bias_guard.slice(),
        &mut q_out_buf, &mut k_out_buf, &mut v_out_buf,
        q_out, k_out, v_out, in_dim,
    )
    .ok()?;
    drop(in_guard);
    drop(q_bias_guard);
    drop(k_bias_guard);
    drop(v_bias_guard);

    if !FUSED_QKV_BIAS_FIRED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[perf] fused Q4_K QKV+bias kernel FIRED (first time) \
             in={in_dim} q_out={q_out} k_out={k_out} v_out={v_out}"
        );
    }

    let dev = input.device();
    let q_t = Tensor::from_storage(Storage::from_cuda_slice(q_out_buf, q_out, dev.clone()), &[1, q_out]).ok()?;
    let k_t = Tensor::from_storage(Storage::from_cuda_slice(k_out_buf, k_out, dev.clone()), &[1, k_out]).ok()?;
    let v_t = Tensor::from_storage(Storage::from_cuda_slice(v_out_buf, v_out, dev), &[1, v_out]).ok()?;
    Some((q_t, k_t, v_t))
}

/// Fused Q4_K GEMV for gate+up projections (SwiGLU / ReLU² FFN). Returns
/// `Some((gate, up))` on the GPU fast path; returns `None` if either weight
/// isn't a GPU-compatible Q4_K variant — caller falls back to two matmuls.
#[cfg(feature = "cuda")]
pub fn fused_gate_up_q4k_matmul_gpu(
    gate_weight: &Weight,
    up_weight: &Weight,
    input: &Tensor<f32>,
) -> Option<(Tensor<f32>, Tensor<f32>)> {
    use axonml_core::backends::cuda::get_cuda_backend;
    use axonml_core::storage::Storage;

    if !input.device().is_gpu() {
        return None;
    }
    let cuda = get_cuda_backend()?;

    let (gate_data, gate_dims) = q4k_quantized_bytes(gate_weight)?;
    let (up_data, up_dims) = q4k_quantized_bytes(up_weight)?;

    let in_dim = gate_dims[0];
    if up_dims[0] != in_dim
        || gate_dims[1] != up_dims[1]
        || in_dim % 256 != 0
    {
        return None;
    }
    let inter = gate_dims[1];

    let (gate_gpu, up_gpu) = match (gate_weight, up_weight) {
        (
            Weight::Quantized { gpu_cache: gc, .. },
            Weight::Quantized { gpu_cache: uc, .. },
        ) => {
            let gg = gc.get_or_init(|| {
                cuda.htod_copy(gate_data.as_slice())
                    .expect("fused gate/up: gate gpu_cache htod_copy failed")
            });
            let ug = uc.get_or_init(|| {
                cuda.htod_copy(up_data.as_slice())
                    .expect("fused gate/up: up gpu_cache htod_copy failed")
            });
            (gg, ug)
        }
        _ => return None,
    };

    use axonml_core::backends::cuda_pool::pool_alloc_uninit;
    let in_guard = input.as_cuda_slice_read();
    // Pool-allocated so the bucket-rounded slice matches later pool_alloc
    // reuse — see note in fused_qkv_q4k_matmul_gpu.
    let mut gate_out_buf = pool_alloc_uninit(inter).ok()?;
    let mut up_out_buf = pool_alloc_uninit(inter).ok()?;

    cuda.q4k_gemv_fused_gate_up_f32(
        gate_gpu,
        up_gpu,
        in_guard.slice(),
        &mut gate_out_buf,
        &mut up_out_buf,
        inter,
        in_dim,
    )
    .ok()?;
    drop(in_guard);

    if !FUSED_GATE_UP_FIRED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[W18-dispatch] fused Q4_K gate/up kernel FIRED (first time) \
             in={in_dim} inter={inter}"
        );
    }

    let dev = input.device();
    let gate_t = Tensor::from_storage(Storage::from_cuda_slice(gate_out_buf, inter, dev.clone()), &[1, inter]).ok()?;
    let up_t = Tensor::from_storage(Storage::from_cuda_slice(up_out_buf, inter, dev), &[1, inter]).ok()?;
    Some((gate_t, up_t))
}

/// Fused Q4_K matmul + residual add: returns `x_in + matmul(input, weight)`
/// as a new tensor. One kernel launch instead of the matmul/add pair, no
/// intermediate projection buffer in HBM. Used by the decode layer for
/// both O-proj (post-attention residual) and down-proj (post-FFN residual).
///
/// Returns `None` if any precondition is unmet (weight not Q4_K, dims
/// mismatch, CPU input, no CUDA) — caller falls back to the standard
/// matmul + Tensor::add path.
#[cfg(feature = "cuda")]
pub fn residual_q4k_matmul_gpu(
    weight: &Weight,
    input: &Tensor<f32>,
    x_in: &Tensor<f32>,
) -> Option<Tensor<f32>> {
    use axonml_core::backends::cuda::get_cuda_backend;
    use axonml_core::storage::Storage;

    if !input.device().is_gpu() || !x_in.device().is_gpu() {
        return None;
    }
    let cuda = get_cuda_backend()?;

    let (w_data, w_dims) = q4k_quantized_bytes(weight)?;
    let in_dim = w_dims[0];
    let out_dim = w_dims[1];
    if in_dim % 256 != 0 {
        return None;
    }
    if x_in.numel() != out_dim {
        return None;
    }

    let w_gpu = match weight {
        Weight::Quantized { gpu_cache, .. } => gpu_cache.get_or_init(|| {
            cuda.htod_copy(w_data.as_slice())
                .expect("residual q4k: weight gpu_cache htod_copy failed")
        }),
        _ => return None,
    };

    use axonml_core::backends::cuda_pool::pool_alloc_uninit;
    let in_guard = input.as_cuda_slice_read();
    let x_guard = x_in.as_cuda_slice_read();
    let mut out_buf = pool_alloc_uninit(out_dim).ok()?;

    cuda.q4k_gemv_residual_f32(
        w_gpu,
        in_guard.slice(),
        x_guard.slice(),
        &mut out_buf,
        out_dim,
        in_dim,
    )
    .ok()?;
    drop(in_guard);
    drop(x_guard);

    static FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !FIRED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[perf] fused Q4_K matmul+residual kernel FIRED (first time) \
             in={in_dim} out={out_dim}"
        );
    }

    let dev = input.device();
    Tensor::from_storage(
        Storage::from_cuda_slice(out_buf, out_dim, dev),
        &[1, out_dim],
    )
    .ok()
}

/// Fused Q4_K gate/up matmul + SwiGLU: returns `silu(gate · input) * (up · input)`
/// as a [1, inter] tensor. One launch, no gate_c/up_c intermediates.
///
/// Returns `None` on any shape/dtype mismatch or CPU input — caller falls
/// back to the separate gate/up matmul + swiglu kernel path.
#[cfg(feature = "cuda")]
pub fn fused_gate_up_swiglu_q4k_matmul_gpu(
    gate_weight: &Weight,
    up_weight: &Weight,
    input: &Tensor<f32>,
) -> Option<Tensor<f32>> {
    use axonml_core::backends::cuda::get_cuda_backend;
    use axonml_core::storage::Storage;

    if !input.device().is_gpu() {
        return None;
    }
    let cuda = get_cuda_backend()?;

    let (gate_data, gate_dims) = q4k_quantized_bytes(gate_weight)?;
    let (up_data, up_dims) = q4k_quantized_bytes(up_weight)?;

    let in_dim = gate_dims[0];
    if up_dims[0] != in_dim || gate_dims[1] != up_dims[1] || in_dim % 256 != 0 {
        return None;
    }
    let inter = gate_dims[1];

    let (gate_gpu, up_gpu) = match (gate_weight, up_weight) {
        (
            Weight::Quantized { gpu_cache: gc, .. },
            Weight::Quantized { gpu_cache: uc, .. },
        ) => {
            let gg = gc.get_or_init(|| {
                cuda.htod_copy(gate_data.as_slice())
                    .expect("fused gate/up+swiglu: gate gpu_cache htod_copy failed")
            });
            let ug = uc.get_or_init(|| {
                cuda.htod_copy(up_data.as_slice())
                    .expect("fused gate/up+swiglu: up gpu_cache htod_copy failed")
            });
            (gg, ug)
        }
        _ => return None,
    };

    use axonml_core::backends::cuda_pool::pool_alloc_uninit;
    let in_guard = input.as_cuda_slice_read();
    let mut ffn_buf = pool_alloc_uninit(inter).ok()?;

    cuda.q4k_gemv_fused_gate_up_swiglu_f32(
        gate_gpu,
        up_gpu,
        in_guard.slice(),
        &mut ffn_buf,
        inter,
        in_dim,
    )
    .ok()?;
    drop(in_guard);

    static FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !FIRED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[perf] fused Q4_K gate/up+SwiGLU kernel FIRED (first time) \
             in={in_dim} inter={inter}"
        );
    }

    let dev = input.device();
    Tensor::from_storage(
        Storage::from_cuda_slice(ffn_buf, inter, dev),
        &[1, inter],
    )
    .ok()
}

/// Helper — extract the `(data bytes, dims)` from a `Weight::Quantized` if
/// it's Q4_K. Returns `None` for every other variant / dtype.
#[cfg(feature = "cuda")]
fn q4k_quantized_bytes(w: &Weight) -> Option<(&Vec<u8>, &Vec<usize>)> {
    match w {
        Weight::Quantized { data, dims, dtype, .. } if *dtype == GgmlType::Q4K && dims.len() == 2 => {
            Some((data, dims))
        }
        _ => None,
    }
}

fn q5k_quantized_bytes(w: &Weight) -> Option<(&Vec<u8>, &Vec<usize>)> {
    match w {
        Weight::Quantized { data, dims, dtype, .. } if *dtype == GgmlType::Q5K && dims.len() == 2 => {
            Some((data, dims))
        }
        _ => None,
    }
}

static FUSED_QKV_Q5K_FIRED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fused Q5_K QKV matmul — one kernel launch for Q/K/V projections.
///
/// Phi-3's `attn_qkv` is Q5_K with split `[q_out, k_out, v_out]`.
/// The load path slices it into three separate Q5_K `Weight`s; this
/// function re-fuses the matmul at decode time so those three
/// projections run as one CUDA kernel instead of three.
///
/// Returns None if any of the three weights isn't 2D Q5_K, shapes
/// don't share `in_dim`, input isn't on GPU, or CUDA isn't available —
/// in which case the caller should fall back to three separate
/// `Weight::matmul` calls.
pub fn fused_qkv_q5k_matmul_gpu(
    q_weight: &Weight,
    k_weight: &Weight,
    v_weight: &Weight,
    input: &Tensor<f32>,
) -> Option<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
    use axonml_core::backends::cuda::get_cuda_backend;
    use axonml_core::storage::Storage;

    if !input.device().is_gpu() {
        return None;
    }
    let cuda = get_cuda_backend()?;

    let (q_data, q_dims) = q5k_quantized_bytes(q_weight)?;
    let (k_data, k_dims) = q5k_quantized_bytes(k_weight)?;
    let (v_data, v_dims) = q5k_quantized_bytes(v_weight)?;

    let in_dim = q_dims[0];
    if k_dims[0] != in_dim || v_dims[0] != in_dim || in_dim % 256 != 0 {
        return None;
    }
    let q_out = q_dims[1];
    let k_out = k_dims[1];
    let v_out = v_dims[1];

    let (q_gpu, k_gpu, v_gpu) = match (q_weight, k_weight, v_weight) {
        (
            Weight::Quantized { gpu_cache: qc, .. },
            Weight::Quantized { gpu_cache: kc, .. },
            Weight::Quantized { gpu_cache: vc, .. },
        ) => {
            let qg = qc.get_or_init(|| {
                cuda.htod_copy(q_data.as_slice())
                    .expect("fused Q5K QKV: q gpu_cache htod_copy failed")
            });
            let kg = kc.get_or_init(|| {
                cuda.htod_copy(k_data.as_slice())
                    .expect("fused Q5K QKV: k gpu_cache htod_copy failed")
            });
            let vg = vc.get_or_init(|| {
                cuda.htod_copy(v_data.as_slice())
                    .expect("fused Q5K QKV: v gpu_cache htod_copy failed")
            });
            (qg, kg, vg)
        }
        _ => return None,
    };

    use axonml_core::backends::cuda_pool::pool_alloc_uninit;
    let in_guard = input.as_cuda_slice_read();
    let mut q_out_buf = pool_alloc_uninit(q_out).ok()?;
    let mut k_out_buf = pool_alloc_uninit(k_out).ok()?;
    let mut v_out_buf = pool_alloc_uninit(v_out).ok()?;

    cuda.q5k_gemv_fused_qkv_f32(
        q_gpu, k_gpu, v_gpu,
        in_guard.slice(),
        &mut q_out_buf, &mut k_out_buf, &mut v_out_buf,
        q_out, k_out, v_out, in_dim,
    )
    .ok()?;
    drop(in_guard);

    if !FUSED_QKV_Q5K_FIRED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[perf] fused Q5_K QKV kernel FIRED (first time) \
             in={in_dim} q_out={q_out} k_out={k_out} v_out={v_out}"
        );
    }

    let dev = input.device();
    let q_t = Tensor::from_storage(
        Storage::from_cuda_slice(q_out_buf, q_out, dev.clone()), &[1, q_out]).ok()?;
    let k_t = Tensor::from_storage(
        Storage::from_cuda_slice(k_out_buf, k_out, dev.clone()), &[1, k_out]).ok()?;
    let v_t = Tensor::from_storage(
        Storage::from_cuda_slice(v_out_buf, v_out, dev), &[1, v_out]).ok()?;
    Some((q_t, k_t, v_t))
}

fn q5_1_quantized_bytes(w: &Weight) -> Option<(&Vec<u8>, &Vec<usize>)> {
    match w {
        Weight::Quantized { data, dims, dtype, .. }
            if *dtype == GgmlType::Q5_1 && dims.len() == 2 =>
        {
            Some((data, dims))
        }
        _ => None,
    }
}

#[cfg(feature = "cuda")]
static FUSED_QKV_Q5_1_FIRED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fused Q5_1 QKV matmul — one kernel launch for Q/K/V projections.
///
/// Target: Falcon-7B's attn_qkv (Q5_1, MQA). The load path splits the
/// original fused GGUF tensor into three Weights; this function re-fuses
/// the matmul at decode time. Key win: Falcon's K and V each have only
/// 64 output rows, too small to fill the GPU on their own — the fused
/// grid combines them with Q's 4544 rows in one launch.
#[cfg(feature = "cuda")]
pub fn fused_qkv_q5_1_matmul_gpu(
    q_weight: &Weight,
    k_weight: &Weight,
    v_weight: &Weight,
    input: &Tensor<f32>,
) -> Option<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
    use axonml_core::backends::cuda::get_cuda_backend;
    use axonml_core::storage::Storage;

    if !input.device().is_gpu() {
        return None;
    }
    let cuda = get_cuda_backend()?;

    let (q_data, q_dims) = q5_1_quantized_bytes(q_weight)?;
    let (k_data, k_dims) = q5_1_quantized_bytes(k_weight)?;
    let (v_data, v_dims) = q5_1_quantized_bytes(v_weight)?;

    let in_dim = q_dims[0];
    if k_dims[0] != in_dim || v_dims[0] != in_dim || in_dim % 32 != 0 {
        return None;
    }
    let q_out = q_dims[1];
    let k_out = k_dims[1];
    let v_out = v_dims[1];

    let (q_gpu, k_gpu, v_gpu) = match (q_weight, k_weight, v_weight) {
        (
            Weight::Quantized { gpu_cache: qc, .. },
            Weight::Quantized { gpu_cache: kc, .. },
            Weight::Quantized { gpu_cache: vc, .. },
        ) => {
            let qg = qc.get_or_init(|| {
                cuda.htod_copy(q_data.as_slice())
                    .expect("fused Q5_1 QKV: q gpu_cache htod_copy failed")
            });
            let kg = kc.get_or_init(|| {
                cuda.htod_copy(k_data.as_slice())
                    .expect("fused Q5_1 QKV: k gpu_cache htod_copy failed")
            });
            let vg = vc.get_or_init(|| {
                cuda.htod_copy(v_data.as_slice())
                    .expect("fused Q5_1 QKV: v gpu_cache htod_copy failed")
            });
            (qg, kg, vg)
        }
        _ => return None,
    };

    use axonml_core::backends::cuda_pool::pool_alloc_uninit;
    let in_guard = input.as_cuda_slice_read();
    let mut q_out_buf = pool_alloc_uninit(q_out).ok()?;
    let mut k_out_buf = pool_alloc_uninit(k_out).ok()?;
    let mut v_out_buf = pool_alloc_uninit(v_out).ok()?;

    cuda.q5_1_gemv_fused_qkv_f32(
        q_gpu, k_gpu, v_gpu,
        in_guard.slice(),
        &mut q_out_buf, &mut k_out_buf, &mut v_out_buf,
        q_out, k_out, v_out, in_dim,
    )
    .ok()?;
    drop(in_guard);

    if !FUSED_QKV_Q5_1_FIRED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[perf] fused Q5_1 QKV kernel FIRED (first time) \
             in={in_dim} q_out={q_out} k_out={k_out} v_out={v_out}"
        );
    }

    let dev = input.device();
    let q_t = Tensor::from_storage(
        Storage::from_cuda_slice(q_out_buf, q_out, dev.clone()), &[1, q_out]).ok()?;
    let k_t = Tensor::from_storage(
        Storage::from_cuda_slice(k_out_buf, k_out, dev.clone()), &[1, k_out]).ok()?;
    let v_t = Tensor::from_storage(
        Storage::from_cuda_slice(v_out_buf, v_out, dev), &[1, v_out]).ok()?;
    Some((q_t, k_t, v_t))
}
