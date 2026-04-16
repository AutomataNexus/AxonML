//! Weight abstraction supporting both pre-dequantized f32 and lazy quantized storage.
//!
//! A `Weight` holds either:
//! - `F32(Tensor<f32>)` — fully dequantized, pre-transposed, optionally on GPU.
//!   Fast matmul, high memory.
//! - `Quantized { data, shape, dtype }` — raw GGUF bytes on CPU. Dequantized
//!   on every matmul call (to scratch memory, then dropped). Slower but
//!   keeps a 27B model to ~10GB instead of ~50GB.
//!
//! The transpose is applied implicitly: quantized data is stored as the
//! physical GGUF layout (rows=out, cols=in) and dequantized into a
//! [out, in] scratch buffer, which is then used as an f32 tensor that
//! gets transposed to [in, out] for the matmul (same convention as the
//! pre-dequantized path).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use axonml_core::Device;
use axonml_quant::bitnet;
use axonml_tensor::Tensor;
use rayon::prelude::*;

use super::gguf::{self, GgmlType};

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
            Weight::Quantized { dims, .. } => vec![dims[0], dims[1]],
            #[cfg(feature = "cuda")]
            Weight::QuantizedGpu { dims, .. } => vec![dims[0], dims[1]],
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
        GgmlType::Q4K => dequantize_blocks_par(output, raw_data, n_elements, 256, 144, gguf::dequantize_q4_k),
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
