//! axonml-subbit - sub-bit vector-quantized (VQ) GPU primitives: fused VQ matmul, grouped MoE
//! experts, device router top-k, resident index assignment, offload tiles and NVFP4 paths.
#![allow(clippy::all, clippy::pedantic)]
#[cfg(feature = "cuda")]
mod imp {
    use axonml_core::backends::cuda::{CudaBackend, CudaError, PinnedBuffer, get_cuda_backend};
    use axonml_core::backends::cuda_kernels;
    use axonml_core::backends::cuda_pool::{
        pool_alloc, pool_alloc_uninit, pool_alloc_uninit_i32, pool_alloc_uninit_u32, pool_free,
        pool_free_i32, pool_free_u32,
    };
    use axonml_core::storage::Storage;
    use axonml_tensor::Tensor;
    use axonml_tensor::shape::contiguous_strides;
    use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock};
    const VQ_FUSED_MATMUL_PTX: &str = include_str!("../kernels/vq_fused_matmul.ptx");
    const VQ_ASSIGN_PTX: &str = include_str!("../kernels/vq_assign.ptx");
    const VQ_OFFLOAD_PTX: &str = include_str!("../kernels/vq_offload.ptx");
    const NVFP4_PTX: &str = include_str!("../kernels/nvfp4.ptx");
    struct SubbitKernels {
        funcs: HashMap<String, CudaFunction>,
        _mods: Vec<Arc<CudaModule>>,
    }
    static SUBBIT_KERNELS: OnceLock<SubbitKernels> = OnceLock::new();
    fn load_kernels() -> SubbitKernels {
        let backend = get_cuda_backend().expect("CUDA backend not available for axonml-subbit");
        let ctx = backend.context();
        let mut m = HashMap::new();
        let fused = ctx
            .load_module(Ptx::from_src(VQ_FUSED_MATMUL_PTX))
            .expect("load vq_fused_matmul");
        m.insert(
            "vq_fused_matmul_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_f32")
                .expect("vq_fused_matmul_f32"),
        );
        m.insert(
            "vq_fused_matmul_dx_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_dx_f32")
                .expect("vq_fused_matmul_dx_f32"),
        );
        m.insert(
            "vq_fused_matmul_dcb_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_dcb_f32")
                .expect("vq_fused_matmul_dcb_f32"),
        );
        m.insert(
            "vq_dcb_scatter_f32".to_string(),
            fused
                .load_function("vq_dcb_scatter_f32")
                .expect("vq_dcb_scatter_f32"),
        );
        m.insert(
            "vq_fused_matmul_tiled_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_tiled_f32")
                .expect("vq_fused_matmul_tiled_f32"),
        );
        m.insert(
            "vq_fused_matmul_tiled_warp_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_tiled_warp_f32")
                .expect("vq_fused_matmul_tiled_warp_f32"),
        );
        m.insert(
            "vq_fused_matmul_rb_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_rb_f32")
                .expect("vq_fused_matmul_rb_f32"),
        );
        m.insert(
            "vq_fused_matmul_dx_rb_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_dx_rb_f32")
                .expect("vq_fused_matmul_dx_rb_f32"),
        );
        m.insert(
            "vq_fused_matmul_dcb_rb_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_dcb_rb_f32")
                .expect("vq_fused_matmul_dcb_rb_f32"),
        );
        m.insert(
            "vq_fused_matmul_rb8_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_rb8_f32")
                .expect("vq_fused_matmul_rb8_f32"),
        );
        m.insert(
            "vq_fused_matmul_tc_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_tc_f32")
                .expect("vq_fused_matmul_tc_f32"),
        );
        m.insert(
            "vq_reconstruct_f32".to_string(),
            fused
                .load_function("vq_reconstruct_f32")
                .expect("vq_reconstruct_f32"),
        );
        m.insert(
            "vq_fused_matmul_ca_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_ca_f32")
                .expect("vq_fused_matmul_ca_f32"),
        );
        m.insert(
            "vq_moe_combine_f32".to_string(),
            fused
                .load_function("vq_moe_combine_f32")
                .expect("vq_moe_combine_f32"),
        );
        m.insert(
            "vq_router_topk_f32".to_string(),
            fused
                .load_function("vq_router_topk_f32")
                .expect("vq_router_topk_f32"),
        );
        m.insert(
            "vq_fused_matmul_grouped_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_grouped_f32")
                .expect("vq_fused_matmul_grouped_f32"),
        );
        m.insert(
            "vq_fused_matmul_grouped_warp_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_grouped_warp_f32")
                .expect("vq_fused_matmul_grouped_warp_f32"),
        );
        m.insert(
            "vq_fused_gate_up_swiglu_grouped_warp_f32".to_string(),
            fused
                .load_function("vq_fused_gate_up_swiglu_grouped_warp_f32")
                .expect("vq_fused_gate_up_swiglu_grouped_warp_f32"),
        );
        m.insert(
            "vq_fused_matmul_grouped_warp_f16cb".to_string(),
            fused
                .load_function("vq_fused_matmul_grouped_warp_f16cb")
                .expect("vq_fused_matmul_grouped_warp_f16cb"),
        );
        m.insert(
            "vq_fused_gate_up_swiglu_grouped_warp_f16cb".to_string(),
            fused
                .load_function("vq_fused_gate_up_swiglu_grouped_warp_f16cb")
                .expect("vq_fused_gate_up_swiglu_grouped_warp_f16cb"),
        );
        m.insert(
            "vq_fused_matmul_tiled_warp_f16cb".to_string(),
            fused
                .load_function("vq_fused_matmul_tiled_warp_f16cb")
                .expect("vq_fused_matmul_tiled_warp_f16cb"),
        );
        m.insert(
            "vq_router_gemv_f32".to_string(),
            fused
                .load_function("vq_router_gemv_f32")
                .expect("vq_router_gemv_f32"),
        );
        m.insert(
            "vq_decode_attn_f32".to_string(),
            fused
                .load_function("vq_decode_attn_f32")
                .expect("vq_decode_attn_f32"),
        );
        m.insert(
            "rope_split_halves_bhsd_devpos_f32".to_string(),
            fused
                .load_function("rope_split_halves_bhsd_devpos_f32")
                .expect("rope_split_halves_bhsd_devpos_f32"),
        );
        m.insert(
            "kv_scatter_devpos_f32".to_string(),
            fused
                .load_function("kv_scatter_devpos_f32")
                .expect("kv_scatter_devpos_f32"),
        );
        m.insert(
            "nh_conv_silu_f32".to_string(),
            fused
                .load_function("nh_conv_silu_f32")
                .expect("nh_conv_silu_f32"),
        );
        m.insert(
            "nh_ssd_scan_f32".to_string(),
            fused
                .load_function("nh_ssd_scan_f32")
                .expect("nh_ssd_scan_f32"),
        );
        m.insert(
            "nh_gated_gnorm_f32".to_string(),
            fused
                .load_function("nh_gated_gnorm_f32")
                .expect("nh_gated_gnorm_f32"),
        );
        m.insert(
            "nh_router_topk_f32".to_string(),
            fused
                .load_function("nh_router_topk_f32")
                .expect("nh_router_topk_f32"),
        );
        m.insert(
            "nh_dense_gemv_f32".to_string(),
            fused
                .load_function("nh_dense_gemv_f32")
                .expect("nh_dense_gemv_f32"),
        );
        m.insert(
            "nh_conv_silu_devpos_f32".to_string(),
            fused
                .load_function("nh_conv_silu_devpos_f32")
                .expect("nh_conv_silu_devpos_f32"),
        );
        m.insert(
            "nh_pos_incr_f32".to_string(),
            fused
                .load_function("nh_pos_incr_f32")
                .expect("nh_pos_incr_f32"),
        );
        m.insert(
            "nh_conv_silu_batch_f32".to_string(),
            fused
                .load_function("nh_conv_silu_batch_f32")
                .expect("nh_conv_silu_batch_f32"),
        );
        m.insert(
            "nh_ssd_scan_batch_f32".to_string(),
            fused
                .load_function("nh_ssd_scan_batch_f32")
                .expect("nh_ssd_scan_batch_f32"),
        );
        m.insert(
            "nh_ssd_scan_batch_v4_f32".to_string(),
            fused
                .load_function("nh_ssd_scan_batch_v4_f32")
                .expect("nh_ssd_scan_batch_v4_f32"),
        );
        m.insert(
            "vq_fused_matmul_warp_rc_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_warp_rc_f32")
                .expect("vq_fused_matmul_warp_rc_f32"),
        );
        m.insert(
            "nh_gated_gnorm_batch_f32".to_string(),
            fused
                .load_function("nh_gated_gnorm_batch_f32")
                .expect("nh_gated_gnorm_batch_f32"),
        );
        m.insert(
            "nh_rmsnorm_rows_f32".to_string(),
            fused
                .load_function("nh_rmsnorm_rows_f32")
                .expect("nh_rmsnorm_rows_f32"),
        );
        m.insert(
            "nh_row_repeat_f32".to_string(),
            fused
                .load_function("nh_row_repeat_f32")
                .expect("nh_row_repeat_f32"),
        );
        m.insert(
            "nh_moe_combine_batch_f32".to_string(),
            fused
                .load_function("nh_moe_combine_batch_f32")
                .expect("nh_moe_combine_batch_f32"),
        );
        m.insert(
            "nh_decode_attn_batch_f32".to_string(),
            fused
                .load_function("nh_decode_attn_batch_f32")
                .expect("nh_decode_attn_batch_f32"),
        );
        m.insert(
            "nh_kv_scatter_batch_f32".to_string(),
            fused
                .load_function("nh_kv_scatter_batch_f32")
                .expect("nh_kv_scatter_batch_f32"),
        );
        m.insert(
            "vq_fused_matmul_grouped_multirow_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_grouped_multirow_f32")
                .expect("vq_fused_matmul_grouped_multirow_f32"),
        );
        m.insert(
            "vq_fused_matmul_grouped_warp_multirow_f32".to_string(),
            fused
                .load_function("vq_fused_matmul_grouped_warp_multirow_f32")
                .expect("vq_fused_matmul_grouped_warp_multirow_f32"),
        );
        m.insert(
            "vq_fused_matmul_grouped_warp_f16cb_mlp4".to_string(),
            fused
                .load_function("vq_fused_matmul_grouped_warp_f16cb_mlp4")
                .expect("vq_fused_matmul_grouped_warp_f16cb_mlp4"),
        );
        m.insert(
            "vq_fused_matmul_grouped_warp_f16cb_c4".to_string(),
            fused
                .load_function("vq_fused_matmul_grouped_warp_f16cb_c4")
                .expect("vq_fused_matmul_grouped_warp_f16cb_c4"),
        );
        m.insert(
            "vq_fused_matmul_grouped_warp_f16cb_c8".to_string(),
            fused
                .load_function("vq_fused_matmul_grouped_warp_f16cb_c8")
                .expect("vq_fused_matmul_grouped_warp_f16cb_c8"),
        );
        let assign = ctx
            .load_module(Ptx::from_src(VQ_ASSIGN_PTX))
            .expect("load vq_assign");
        m.insert(
            "vq_assign_f32".to_string(),
            assign
                .load_function("vq_assign_f32")
                .expect("vq_assign_f32"),
        );
        let offload = ctx
            .load_module(Ptx::from_src(VQ_OFFLOAD_PTX))
            .expect("load vq_offload");
        for k in ["vq_normalize_f32", "vq_gather_denorm_f32"] {
            m.insert(k.to_string(), offload.load_function(k).expect(k));
        }
        SubbitKernels {
            funcs: m,
            _mods: vec![fused, assign, offload],
        }
    }
    /// Row-count ceiling for the warp-per-column GEMV (env `AXONML_VQ_WARP_ROWS`,
    /// default 16). Above it the 16x16 tile kernel amortizes better; at or below,
    /// the warp GEMV is the decode/multi-agent path.
    fn vq_warp_rows_max() -> usize {
        static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            // Measured crossover on the 30B (5070 Ti): the warp GEMV wins to n=4-6
            // (M=2: 36 ms vs 67 on the tile path) and the 16x16 tile amortizes better
            // by n=8 (102 vs 122 ms). 4 and 6 measured identical; 4 is the safe pick.
            std::env::var("AXONML_VQ_WARP_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4)
        })
    }

    fn subbit_func(name: &str) -> Result<CudaFunction, CudaError> {
        SUBBIT_KERNELS
            .get_or_init(load_kernels)
            .funcs
            .get(name)
            .cloned()
            .ok_or_else(|| CudaError::KernelNotFound(name.to_string()))
    }
    /// Per-call seed for the stochastic-rounding gradient quantizer — a process-wide
    /// counter so each quantize draws fresh rounding noise (same-seed calls would
    /// correlate the noise across steps and re-bias the estimator).
    fn nvfp4_sr_seed() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
        CTR.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
    }
    /// Lazy loader for the NVFP4 kernels (`.target sm_120a` — the module only loads on
    /// Blackwell). Separate from [`load_kernels`] so non-Blackwell devices keep every
    /// other subbit kernel; requesting an nvfp4 function there errors, nothing else breaks.
    fn nvfp4_func(name: &str) -> Result<CudaFunction, CudaError> {
        static NVFP4_KERNELS: OnceLock<Option<SubbitKernels>> = OnceLock::new();
        NVFP4_KERNELS
            .get_or_init(|| {
                let backend = get_cuda_backend()?;
                let module = backend
                    .context()
                    .load_module(Ptx::from_src(NVFP4_PTX))
                    .ok()?;
                let mut m = HashMap::new();
                for k in ["nvfp4_quant_f32", "nvfp4_quant_sr_f32", "nvfp4_gemm_sf2"] {
                    m.insert(k.to_string(), module.load_function(k).ok()?);
                }
                Some(SubbitKernels {
                    funcs: m,
                    _mods: vec![module],
                })
            })
            .as_ref()
            .and_then(|k| k.funcs.get(name).cloned())
            .ok_or_else(|| CudaError::KernelNotFound(format!("nvfp4 (sm_120a only): {name}")))
    }
    fn subbit_smem_flag() -> i32 {
        static FLAG: OnceLock<i32> = OnceLock::new();
        *FLAG.get_or_init(|| match std::env::var("PRG_SUBBIT_NO_SMEM") {
            Ok(v) if !v.is_empty() && v != "0" => 0,
            _ => 1,
        })
    }
    /// Warps-per-block (W) for the batch-1 warp-per-column decode GEMV. Overridable
    /// via `SUBBIT_TILED_WARP_W` for sweeping; default is the accepted best (16, from
    /// the W in {4,8,16,32} sweep on the 30B attention + 1.7B dense decode).
    fn subbit_tiled_warp_w() -> u32 {
        static W: OnceLock<u32> = OnceLock::new();
        *W.get_or_init(|| {
            std::env::var("SUBBIT_TILED_WARP_W")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&w| w >= 1 && w <= 16)
                .unwrap_or(16)
        })
    }
    /// Opt-in fp16-codebook fast path for the batch-1 decode GEMVs. Rounds the codebook
    /// to `__half` in shared memory (halved gather + smem footprint) while keeping x,
    /// scale and accumulation in fp32. NOT argmax-identical (it perturbs the greedy
    /// stream / FNV), so it is OFF by default and only enabled via `PRG_SUBBIT_F16_CB`.
    /// Output columns each warp handles in the grouped expert GEMV. The input row is
    /// the dominant traffic (out_f * in_f * 4 B, ~32x the packed indices), so sharing
    /// one loaded x vector across C columns cuts it C-fold. 0 = the classic
    /// one-column-per-warp kernel.
    /// Row+column blocked non-grouped GEMV for dim=8 packs. ON by default above the
    /// warp-GEMV row threshold, where the alternative is the element-at-a-time tile
    /// kernel: measured M=8 77.9 -> 69.3 ms, M=16 93.3 -> 89.9. `PRG_SUBBIT_RC=0`
    /// restores the tile path.
    fn subbit_rc() -> bool {
        static F: OnceLock<bool> = OnceLock::new();
        *F.get_or_init(|| !matches!(std::env::var("PRG_SUBBIT_RC"), Ok(ref v) if v == "0"))
    }
    fn subbit_cols() -> usize {
        static F: OnceLock<usize> = OnceLock::new();
        *F.get_or_init(|| {
            std::env::var("PRG_SUBBIT_COLS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        })
    }
    fn subbit_mlp4() -> bool {
        static F: OnceLock<bool> = OnceLock::new();
        *F.get_or_init(
            || matches!(std::env::var("PRG_SUBBIT_MLP4"), Ok(v) if !v.is_empty() && v != "0"),
        )
    }
    fn subbit_f16_cb() -> bool {
        static F: OnceLock<bool> = OnceLock::new();
        *F.get_or_init(
            || matches!(std::env::var("PRG_SUBBIT_F16_CB"), Ok(v) if !v.is_empty() && v != "0"),
        )
    }
    fn htod_u32_pinned(cuda: &CudaBackend, data: &[u32], dst: &mut CudaSlice<u32>) {
        use std::cell::RefCell;
        thread_local! { static STAGE: RefCell<Option<PinnedBuffer>> = const { RefCell::new(None) }; }
        let n = data.len();
        STAGE.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.as_ref().map_or(true, |p| p.len() < n) {
                *opt = Some(PinnedBuffer::alloc(n).expect("pinned u32 alloc"));
            }
            let pinned = opt.as_mut().unwrap();
            // SAFETY: the pinned buffer holds at least `n` page-locked f32s and is borrowed
            // exclusively here; u32 has the same size and alignment as f32, so the same `n`
            // elements viewed as u32 stay in bounds for the lifetime of this view.
            let pin_u32: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(pinned.as_slice_mut().as_mut_ptr().cast::<u32>(), n)
            };
            pin_u32.copy_from_slice(data);
            cuda.htod_into(&pin_u32[..n], dst).expect("htod u32 pinned");
            cuda.sync();
        });
    }

    // == structs ==
    /// GPU-resident per-tile VQ assignment + per-row scale for the offload lazy path.
    /// The assignment is identical every lazy step (cached), so it's uploaded ONCE
    /// per re-quant generation and reused — the lazy recon/scatter loop then does NO
    /// per-tile H2D (no cost, no sync → CUDA-graph capturable). Layer-owned.
    #[cfg(feature = "cuda")]
    pub struct ResidentAssign {
        assign: Vec<cudarc::driver::CudaSlice<u32>>,
        scale: Vec<cudarc::driver::CudaSlice<f32>>,
    }

    #[cfg(feature = "cuda")]
    impl ResidentAssign {
        /// Upload per-tile assignments (`u16` → `u32`) + per-tile row scales to GPU
        /// once. `scales[i]` is the row-scale slice for tile `i`.
        pub fn upload(tiles: &[Vec<u16>], scales: &[&[f32]]) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let mut assign = Vec::with_capacity(tiles.len());
            for t in tiles {
                let u32v: Vec<u32> = t.iter().map(|&a| u32::from(a)).collect();
                let mut s = axonml_core::backends::cuda_pool::pool_alloc_uninit_u32(u32v.len())
                    .expect("resident assign alloc");
                htod_u32_pinned(cuda, &u32v, &mut s);
                assign.push(s);
            }
            let mut scale = Vec::with_capacity(scales.len());
            for sc in scales {
                let mut s = pool_alloc_uninit(sc.len()).expect("resident scale alloc");
                cuda.htod_into(sc, &mut s).expect("resident scale htod");
                scale.push(s);
            }
            cuda.sync();
            Self { assign, scale }
        }

        /// Number of resident tiles.
        #[must_use]
        pub fn len(&self) -> usize {
            self.assign.len()
        }
        /// True if empty.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.assign.is_empty()
        }
    }

    /// A group of same-shaped sub-bit VQ experts with their packed weights
    /// concatenated into three device-resident buffers (indices `u16`→`u32`,
    /// codebooks, per-row scales), uploaded ONCE. `forward` runs the top-k selected
    /// experts' fused VQ GEMV for one decode row in a SINGLE grouped launch
    /// (`vq_fused_matmul_grouped_f32`, grid.z = group) instead of one launch per
    /// expert — the fix for the MoE decode path's ~1,152 tiny expert launches/token.
    ///
    /// Numerically it is byte-identical to running each expert through the per-expert
    /// tiled kernel: the grouped kernel's per-(group,row,col) arithmetic is the same
    /// tile loop / scale-fold / f32-accumulation order, just base-pointer-selected by
    /// `sel[g]`. The concatenation is in expert-id order, so `sel[g]` (the routed
    /// expert id) indexes directly.
    #[cfg(feature = "cuda")]
    pub struct VqGroupedExperts {
        idx: cudarc::driver::CudaSlice<u8>,
        cb: cudarc::driver::CudaSlice<f32>,
        scale: cudarc::driver::CudaSlice<f32>,
        out_f: usize,
        in_f: usize,
        dim: usize,
        idx_stride: usize,
        cb_stride: usize,
        scale_stride: usize,
    }

    #[cfg(feature = "cuda")]
    impl VqGroupedExperts {
        /// Upload the concatenated expert buffers to the device once. `idx_all` is the
        /// experts' `u8` indices back-to-back (kept 1 byte/index — valid for `k<=256`,
        /// the sub-bit production config — so all 128 experts fit in ~1/4 the VRAM of
        /// the widened `u32` form; `idx_stride` = per-expert index count =
        /// `out_f * in_f / dim`), `cb_all` the per-expert codebooks (`cb_stride` =
        /// `k * dim`), `scale_all` the per-expert row scales (`scale_stride` = `out_f`).
        #[allow(clippy::too_many_arguments)]
        #[must_use]
        pub fn upload(
            idx_all: &[u8],
            cb_all: &[f32],
            scale_all: &[f32],
            out_f: usize,
            in_f: usize,
            dim: usize,
            idx_stride: usize,
            cb_stride: usize,
            scale_stride: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let idx = cuda.htod_copy(idx_all).expect("grouped idx htod");
            let cb = cuda.htod_copy(cb_all).expect("grouped cb htod");
            let scale = cuda.htod_copy(scale_all).expect("grouped scale htod");
            cuda.sync();
            Self {
                idx,
                cb,
                scale,
                out_f,
                in_f,
                dim,
                idx_stride,
                cb_stride,
                scale_stride,
            }
        }

        /// Run the selected experts (`sel[g]` = routed expert id for group `g`) on the
        /// decode row(s) `x` in one grouped launch. When `x_broadcast` the SAME `x`
        /// row feeds every group (gate/up, `x` is `[1, in_f]`); otherwise `x` is
        /// `[G, in_f]` and group `g` uses row `g` (down, `x` is the per-expert SwiGLU
        /// activation). Returns `[G, out_f]` on `x`'s device.
        #[must_use]
        pub fn forward(&self, x: &Tensor<f32>, sel: &[i32], x_broadcast: bool) -> Tensor<f32> {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let xc = x.contiguous();
            let in_f = self.in_f;
            debug_assert_eq!(
                xc.shape()[xc.ndim() - 1],
                in_f,
                "grouped VQ forward: input in_features mismatch"
            );
            let g = sel.len();
            let vpr = in_f / self.dim;
            // sel is tiny (top_k ints); the async H2D + the launch are enqueued on the
            // one stream in order, and this scratch slice's free is stream-ordered
            // after the launch, so no host sync is needed on the decode hot path.
            let sel_g = cuda.htod_copy(sel).expect("grouped sel htod");
            let mut out = pool_alloc_uninit(g * self.out_f).expect("grouped out pool alloc");
            {
                let x_g = xc.as_cuda_slice_read();
                let x_stride = if x_broadcast { 0 } else { in_f };
                cuda.vq_fused_matmul_grouped_f32(
                    x_g.slice(),
                    &self.idx,
                    &self.cb,
                    &self.scale,
                    &sel_g,
                    &mut out,
                    g,
                    1,
                    in_f,
                    self.out_f,
                    self.dim,
                    vpr,
                    self.idx_stride,
                    self.cb_stride,
                    self.scale_stride,
                    x_stride,
                )
                .expect("vq_fused_matmul_grouped_f32");
            }
            let shape = [g, self.out_f];
            Tensor::from_storage(
                Storage::from_cuda_slice(out, g * self.out_f, xc.device()),
                &shape,
            )
            .expect("subbit from_storage")
        }

        /// Device-`sel` twin of [`forward`](Self::forward): the routed expert indices
        /// already live on-device (from [`Tensor::router_topk`]), so there is no
        /// per-call `sel` H2D and — crucially — no host round-trip for the router.
        /// `g` is the number of selected experts (`top_k`). Byte-identical to
        /// `forward` given the same `sel` values.
        pub fn forward_dev(
            &self,
            x: &Tensor<f32>,
            sel: &cudarc::driver::CudaSlice<i32>,
            g: usize,
            x_broadcast: bool,
        ) -> Tensor<f32> {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let xc = x.contiguous();
            let in_f = self.in_f;
            let vpr = in_f / self.dim;
            let mut out = pool_alloc_uninit(g * self.out_f).expect("grouped_dev out pool alloc");
            {
                let x_g = xc.as_cuda_slice_read();
                let x_stride = if x_broadcast { 0 } else { in_f };
                cuda.vq_fused_matmul_grouped_f32(
                    x_g.slice(),
                    &self.idx,
                    &self.cb,
                    &self.scale,
                    sel,
                    &mut out,
                    g,
                    1,
                    in_f,
                    self.out_f,
                    self.dim,
                    vpr,
                    self.idx_stride,
                    self.cb_stride,
                    self.scale_stride,
                    x_stride,
                )
                .expect("vq_fused_matmul_grouped_f32 (dev sel)");
            }
            let shape = [g, self.out_f];
            Tensor::from_storage(
                Storage::from_cuda_slice(out, g * self.out_f, xc.device()),
                &shape,
            )
            .expect("subbit from_storage")
        }

        /// Expert-major grouped forward: `sel_host` is the routed expert per group in
        /// group order, and the pairs are bucketed by expert so each expert's packed
        /// weights are walked ONCE for every group that selected it.
        ///
        /// This is the half of MoE batching that plain batching cannot do. The
        /// backbone amortizes across a batch for free, but the routed experts do not:
        /// the group-major kernel re-reads a 1.25 MB expert for every group, so M
        /// agents that happen to agree still pay M times. Bucketing collapses that to
        /// the number of DISTINCT experts the step actually needs.
        ///
        /// Rows are bit-identical to `forward_dev` — same per-row lane partition, same
        /// scale fold, same reduction tree.
        #[must_use]
        pub fn forward_dev_expert_major(
            &self,
            x: &Tensor<f32>,
            sel_host: &[i32],
            max_rows: usize,
        ) -> Tensor<f32> {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let xc = x.contiguous();
            let g = sel_host.len();
            let in_f = self.in_f;
            let vpr = in_f / self.dim;

            // Bucket group indices by expert. Buckets wider than the kernel's
            // accumulator budget are split, which costs one extra weight walk per
            // split rather than per group.
            let cap = max_rows.clamp(1, 16);
            let mut order: Vec<usize> = (0..g).collect();
            order.sort_by_key(|&i| sel_host[i]);
            let mut sel_u: Vec<i32> = Vec::new();
            let mut row_off: Vec<i32> = vec![0];
            let mut pair_g: Vec<i32> = Vec::with_capacity(g);
            let mut i = 0usize;
            while i < g {
                let e = sel_host[order[i]];
                let mut j = i;
                while j < g && sel_host[order[j]] == e {
                    j += 1;
                }
                let mut k = i;
                while k < j {
                    let end = (k + cap).min(j);
                    sel_u.push(e);
                    for &o in &order[k..end] {
                        pair_g.push(o as i32);
                    }
                    row_off.push(pair_g.len() as i32);
                    k = end;
                }
                i = j;
            }
            let u = sel_u.len();

            let sel_g = cuda.htod_copy(&sel_u).expect("sel_u htod");
            let off_g = cuda.htod_copy(&row_off).expect("row_off htod");
            let pair_gd = cuda.htod_copy(&pair_g).expect("pair_g htod");
            let mut out = pool_alloc_uninit(g * self.out_f).expect("expert-major out alloc");
            {
                let x_g = xc.as_cuda_slice_read();
                let func = subbit_func("vq_fused_matmul_grouped_warp_multirow_f32")
                    .expect("vq_fused_matmul_grouped_warp_multirow_f32");
                let warps: u32 = 8;
                let cfg = LaunchConfig {
                    grid_dim: ((self.out_f as u32).div_ceil(warps), 1, u as u32),
                    block_dim: (32, warps, 1),
                    shared_mem_bytes: 0,
                };
                let use_smem = subbit_smem_flag();
                // SAFETY: kernel `vq_fused_matmul_grouped_warp_multirow_f32` in kernels/vq_fused_matmul.ptx takes exactly these 16
                // arguments in this order, verified against the PTX signature by
                // tools/check_launches.py; the slice extents follow from the shapes derived above,
                // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
                // the backend's single stream.
                unsafe {
                    cuda.stream()
                        .launch_builder(&func)
                        .arg(x_g.slice())
                        .arg(&self.idx)
                        .arg(&self.cb)
                        .arg(&self.scale)
                        .arg(&sel_g)
                        .arg(&off_g)
                        .arg(&pair_gd)
                        .arg(&mut out)
                        .arg(&(in_f as i32))
                        .arg(&(self.out_f as i32))
                        .arg(&(self.dim as i32))
                        .arg(&(vpr as i32))
                        .arg(&(self.idx_stride as i32))
                        .arg(&(self.cb_stride as i32))
                        .arg(&(self.scale_stride as i32))
                        .arg(&use_smem)
                        .launch(cfg)
                        .expect("grouped warp multirow launch");
                }
                if std::env::var("AXONML_NH_MOE_DEDUP_SYNC").is_ok_and(|v| v != "0") {
                    cuda.stream().synchronize().expect("multirow sync probe");
                }
            }
            let shape = [g, self.out_f];
            Tensor::from_storage(
                Storage::from_cuda_slice(out, g * self.out_f, xc.device()),
                &shape,
            )
            .expect("expert-major from_storage")
        }

        /// Fused gate+up+SwiGLU for the decode MoE: computes `silu(gate·x)*(up·x)` for
        /// every selected expert in ONE launch, returning the `[g, inter]` activation
        /// directly. Replaces `gate.forward_dev(..) ; up.forward_dev(..) ; gate.swiglu(up)`
        /// (three launches + two DRAM intermediates) with a single grouped-warp kernel
        /// that runs the gate and up dots as two independent gather chains per warp.
        /// `gate` and `up` MUST share shape/dim/strides (asserted); `x` is the broadcast
        /// `[1, in_f]` decode row. Bit-identical to the unfused path.
        #[must_use]
        pub fn fused_gate_up_swiglu_dev(
            gate: &VqGroupedExperts,
            up: &VqGroupedExperts,
            x: &Tensor<f32>,
            sel: &cudarc::driver::CudaSlice<i32>,
            g: usize,
        ) -> Tensor<f32> {
            debug_assert_eq!(gate.in_f, up.in_f, "fused gate/up in_f mismatch");
            debug_assert_eq!(gate.out_f, up.out_f, "fused gate/up out_f mismatch");
            debug_assert_eq!(gate.dim, up.dim, "fused gate/up dim mismatch");
            debug_assert_eq!(
                gate.idx_stride, up.idx_stride,
                "fused gate/up idx_stride mismatch"
            );
            debug_assert_eq!(
                gate.cb_stride, up.cb_stride,
                "fused gate/up cb_stride mismatch"
            );
            debug_assert_eq!(
                gate.scale_stride, up.scale_stride,
                "fused gate/up scale_stride mismatch"
            );
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let xc = x.contiguous();
            let in_f = gate.in_f;
            let out_f = gate.out_f;
            let vpr = in_f / gate.dim;
            let mut act = pool_alloc_uninit(g * out_f).expect("fused gate_up act pool alloc");
            {
                let x_g = xc.as_cuda_slice_read();
                cuda.vq_fused_gate_up_swiglu_grouped_f32(
                    x_g.slice(),
                    &gate.idx,
                    &gate.cb,
                    &gate.scale,
                    &up.idx,
                    &up.cb,
                    &up.scale,
                    sel,
                    &mut act,
                    g,
                    1,
                    in_f,
                    out_f,
                    gate.dim,
                    vpr,
                    gate.idx_stride,
                    gate.cb_stride,
                    gate.scale_stride,
                    0, // x broadcast to every group (gate/up share one decode row)
                )
                .expect("vq_fused_gate_up_swiglu_grouped_f32");
            }
            let shape = [g, out_f];
            Tensor::from_storage(
                Storage::from_cuda_slice(act, g * out_f, xc.device()),
                &shape,
            )
            .expect("subbit from_storage")
        }
    }

    /// Device-resident MoE router selection: top-k expert indices (`i32`) and their
    /// weights (`f32`), produced by [`Tensor::router_topk`] and consumed by
    /// [`VqGroupedExperts::forward_dev`] + [`Tensor::vq_moe_combine_dev`] without any
    /// host round-trip.
    #[cfg(feature = "cuda")]
    pub struct RouterSel {
        // `Option` so `Drop` can move the slices back to the pool (routing to the
        // capture pen when a graph capture is active). Sourcing these from the pool
        // — rather than a raw `stream().alloc` — keeps them out of a captured graph
        // as MemAlloc/MemFree nodes.
        sel: Option<cudarc::driver::CudaSlice<i32>>,
        w: Option<cudarc::driver::CudaSlice<f32>>,
    }

    #[cfg(feature = "cuda")]
    impl RouterSel {
        #[must_use]
        pub fn sel(&self) -> &cudarc::driver::CudaSlice<i32> {
            self.sel.as_ref().expect("RouterSel.sel after drop")
        }
        #[must_use]
        pub fn w(&self) -> &cudarc::driver::CudaSlice<f32> {
            self.w.as_ref().expect("RouterSel.w after drop")
        }

        /// Copy the chosen expert ids back to the host. Small (`M * top_k`), and the
        /// scheduler needs them to co-schedule agents whose routing overlaps.
        #[must_use]
        pub fn to_host(&self) -> Vec<i32> {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            cuda.stream()
                .clone_dtoh(self.sel())
                .expect("route trace D2H")
        }
    }

    #[cfg(feature = "cuda")]
    impl Drop for RouterSel {
        fn drop(&mut self) {
            if let Some(sel) = self.sel.take() {
                pool_free_i32(sel);
            }
            if let Some(w) = self.w.take() {
                pool_free(w);
            }
        }
    }
    // SAFETY: holds only cudarc device slices that are read on the backend's single stream; the
    // struct is Arc-shared read-only between a layer and its graph node and never mutated
    // across threads.
    unsafe impl Send for ResidentAssign {}

    // SAFETY: see the `Send` impl above; shared references only ever read device handles.
    unsafe impl Sync for ResidentAssign {}

    impl std::fmt::Debug for ResidentAssign {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "ResidentAssign({} tiles)", self.assign.len())
        }
    }

    impl Drop for ResidentAssign {
        fn drop(&mut self) {
            for a in self.assign.drain(..) {
                axonml_core::backends::cuda_pool::pool_free_u32(a);
            }
            for s in self.scale.drain(..) {
                axonml_core::backends::cuda_pool::pool_free(s);
            }
        }
    }

    // SAFETY: holds only device slices read on the backend's single stream; the struct is
    // layer-owned and shared read-only across the forward, mirroring `ResidentAssign`.
    unsafe impl Send for VqGroupedExperts {}

    // SAFETY: see the `Send` impl above; shared references only ever read device handles.
    unsafe impl Sync for VqGroupedExperts {}

    impl std::fmt::Debug for VqGroupedExperts {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "VqGroupedExperts(out_f={}, in_f={}, dim={}, idx_stride={})",
                self.out_f, self.in_f, self.dim, self.idx_stride
            )
        }
    }

    // == backend ext ==
    pub trait SubbitBackendExt {
        /// Weight-offload VQ normalize: `out[i] = w[i] * (1/row_scale[row(i)])`,
        /// computing the per-row scale on-GPU so the tiled quantize uploads only the
        /// `[n_rows]` row scales instead of a full `[numel]` inv_scale buffer.
        fn vq_normalize_f32(
            &self,
            out: &mut CudaSlice<f32>,
            w: &CudaSlice<f32>,
            row_scale: &CudaSlice<f32>,
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Result<(), CudaError>;
        /// Fully self-contained fused VQ nearest-code assignment:
        /// `assign[j] = argmax_c(v_j·c − ½‖c‖²)` over the `k` codebook entries, one
        /// thread per weight-vector. Computes `−½‖c‖²` internally in shared memory —
        /// takes ONLY `normed` + `codebook`, materializes no `[n_vec, k]` scores
        /// matrix. Replaces the cuBLAS scores GEMM + `mul`/`sum_dim` + argmax.
        /// Fused VQ sub-bit matmul forward: `out[n,out_f] = x · recon^T`, recon
        /// gathered from (idx,cb,scale) with no dense weight materialized.
        fn vq_fused_matmul_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        /// Tiled (shared-memory) fused VQ matmul forward — high-throughput variant
        /// of `vq_fused_matmul_f32`. Same result, 16x16 smem GEMM with codebook gather.
        fn vq_fused_matmul_tiled_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            k: usize,
        ) -> Result<(), CudaError>;
        /// Warp-per-output-column fused VQ GEMV — batch-1 decode variant of
        /// `vq_fused_matmul_tiled_f32`. One warp reduces one output column via a
        /// shfl_down tree (32 in-flight gather streams/column, no per-k-tile sync);
        /// argmax-identical (FNV-gated), not bit-identical. Use only when n==1.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_tiled_warp_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            k: usize,
        ) -> Result<(), CudaError>;
        /// Grouped tiled fused VQ matmul forward — runs `n_groups` independent VQ
        /// GEMVs (one per selected MoE expert) in a SINGLE launch (grid.z = group).
        /// Per group the arithmetic is byte-identical to `vq_fused_matmul_tiled_f32`
        /// for that expert; the weights are selected from the concatenated per-expert
        /// buffers via `sel[g]`. `x_stride==0` broadcasts one input row to every group
        /// (gate/up); `x_stride==n*in_f` gives each group its own row (down).
        /// Row+column blocked non-grouped GEMV for dim=8 packs: one gathered codebook
        /// vector serves four rows, one loaded x vector serves four columns.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_warp_rc_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            k_codes: usize,
        ) -> Result<(), CudaError>;

        fn vq_fused_matmul_grouped_f32(
            &self,
            x: &CudaSlice<f32>,
            idx_all: &CudaSlice<u8>,
            cb_all: &CudaSlice<f32>,
            scale_all: &CudaSlice<f32>,
            sel: &CudaSlice<i32>,
            out: &mut CudaSlice<f32>,
            n_groups: usize,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            idx_stride: usize,
            cb_stride: usize,
            scale_stride: usize,
            x_stride: usize,
        ) -> Result<(), CudaError>;
        /// Fused gate+up+SwiGLU grouped GEMV: runs the MoE gate and up projections for
        /// all `n_groups` selected experts AND the SwiGLU elementwise in ONE launch,
        /// writing `act[g] = silu(gate[g])*up[g]` directly. Bit-identical to
        /// (grouped gate) -> (grouped up) -> (swiglu_f32); collapses 3 launches + the
        /// two `[G,out_f]` DRAM intermediates into one. gate/up share strides.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_gate_up_swiglu_grouped_f32(
            &self,
            x: &CudaSlice<f32>,
            g_idx_all: &CudaSlice<u8>,
            g_cb_all: &CudaSlice<f32>,
            g_scale_all: &CudaSlice<f32>,
            u_idx_all: &CudaSlice<u8>,
            u_cb_all: &CudaSlice<f32>,
            u_scale_all: &CudaSlice<f32>,
            sel: &CudaSlice<i32>,
            act: &mut CudaSlice<f32>,
            n_groups: usize,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            idx_stride: usize,
            cb_stride: usize,
            scale_stride: usize,
            x_stride: usize,
        ) -> Result<(), CudaError>;
        /// Fused MoE top-k combine: `out[h] = Σ_g rows[g,:]·w[g]` accumulated in g-order
        /// from zero. Byte-identical to a chain of `scaled_add_inplace_f32` launches, so
        /// it replaces the top_k per-expert scaled-adds of a MoE layer with one launch.
        fn vq_moe_combine_f32(
            &self,
            rows: &CudaSlice<f32>,
            w: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            h: usize,
            g_count: usize,
        ) -> Result<(), CudaError>;
        /// Device MoE router top-k: per row, softmax over `ne` router logits, pick
        /// the top_k experts, (optionally) renormalize — writing expert indices +
        /// weights to device buffers with NO host round-trip. One block per row.
        fn vq_router_topk_f32(
            &self,
            logits: &CudaSlice<f32>,
            sel: &mut CudaSlice<i32>,
            wout: &mut CudaSlice<f32>,
            rows: usize,
            ne: usize,
            top_k: usize,
            norm: bool,
        ) -> Result<(), CudaError>;
        /// Dense fp32 router GEMV: `out[n,N] = x[n,K] · w[K,N]` (row-major `w`), a small
        /// `[n,K]x[K,N]` matmul. Replaces the cuBLAS `matmul` on the MoE router decode
        /// path so the decode step is cuBLAS-free (CUDA-graph-capturable). One block per
        /// row; ascending-`k` f32 accumulation (FNV-gated argmax match to fp32 cuBLAS).
        fn vq_router_gemv_f32(
            &self,
            x: &CudaSlice<f32>,
            w: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            k: usize,
            n_out: usize,
        ) -> Result<(), CudaError>;
        /// Fused single-position decode attention over the fixed `[nkv,max_ctx,hd]` KV
        /// buffers with the additive device mask: per query head, `q·Kᵀ` (scaled) +
        /// mask → softmax → `·V`, GQA resolved in-kernel. One launch, no cuBLAS, no
        /// `repeat_kv`. Replaces the two SDPA matmuls + softmax on the decode path.
        #[allow(clippy::too_many_arguments)]
        fn vq_decode_attn_f32(
            &self,
            q: &CudaSlice<f32>,
            k: &CudaSlice<f32>,
            v: &CudaSlice<f32>,
            mask: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            nh: usize,
            nkv: usize,
            max_ctx: usize,
            hd: usize,
            scale: f32,
        ) -> Result<(), CudaError>;
        /// Device-pos split-halves RoPE: bit-identical to the eager
        /// `rope_split_halves_bhsd_f32` but reads the starting position from a device
        /// i32 scalar (`pos_ptr[0]`) instead of a launch-arg constant, so one captured
        /// CUDA graph replays RoPE correctly at every decode step. Shape
        /// `[bs, n_heads, seq, head_dim]`.
        #[allow(clippy::too_many_arguments)]
        fn rope_split_halves_bhsd_devpos_f32(
            &self,
            src: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            pos: &CudaSlice<f32>,
            seq: usize,
            n_heads: usize,
            head_dim: usize,
            theta: f32,
        ) -> Result<(), CudaError>;
        /// Device-pos KV scatter: write this step's `[nkv, t, hd]` K (or V) into the
        /// fixed `[nkv, max_ctx, hd]` cache at position `pos[0]` (device f32 scalar,
        /// cast to int in-kernel). Kernel twin of the `memcpy_2d_dtod` scatter —
        /// recomputes the destination from the device pos every launch, so a captured
        /// graph advances the KV correctly. Pure copy → byte-identical to the memcpy.
        fn kv_scatter_devpos_f32(
            &self,
            src: &CudaSlice<f32>,
            dst: &mut CudaSlice<f32>,
            pos: &CudaSlice<f32>,
            nkv: usize,
            t: usize,
            hd: usize,
            max_ctx: usize,
        ) -> Result<(), CudaError>;
        /// Register-blocked fused VQ matmul forward (64x64 tile, 4x4 microtile) —
        /// the high-throughput production forward. Same result as the naive variant.
        fn vq_fused_matmul_rb_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        /// Fused VQ matmul backward wrt input: `dx[n,in_f]`.
        fn vq_fused_matmul_dx_f32(
            &self,
            g: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            dx: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        /// 8x8-microtile / 128x128-tile fused VQ matmul forward — max-throughput
        /// variant (same result as `vq_fused_matmul_rb_f32`).
        fn vq_fused_matmul_rb8_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        /// cp.async-pipelined tf32 tensor-core fused VQ matmul forward (64x64 tile,
        /// codebook resident in smem, activation tile cp.async-loaded). Assumes k*dim<=4096.
        fn vq_fused_matmul_ca_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        /// Reconstruct the dense `recon[out_f,in_f]` from the packed weight, so a
        /// tf32 cuBLAS GEMM can run off it at tensor-core speed.
        fn vq_reconstruct_f32(
            &self,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            recon: &mut CudaSlice<f32>,
            out_f: usize,
            in_f: usize,
            dim: usize,
            vpr: usize,
            o_base: usize,
        ) -> Result<(), CudaError>;
        /// Scatter a precomputed dW tile (`g^T @ x`, rows [o_base, o_base+ot)) into the
        /// codebook grad — the second half of the tf32 backward (the GEMM runs on
        /// cuBLAS-tf32). Matches `vq_fused_matmul_dcb_f32`'s scatter exactly. `dcb`
        /// must be pre-zeroed; accumulates atomically across the (o,v)->code sharing.
        fn vq_dcb_scatter_f32(
            &self,
            dw: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            scale: &CudaSlice<f32>,
            dcb: &mut CudaSlice<f32>,
            ot: usize,
            in_f: usize,
            dim: usize,
            vpr: usize,
            o_base: usize,
        ) -> Result<(), CudaError>;
        /// tf32 tensor-core fused VQ matmul forward — highest throughput (approximate:
        /// tf32 ~10-bit mantissa). 4 warps/block, 32x32 output tile via wmma m16n16k8.
        fn vq_fused_matmul_tc_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        /// Register-blocked fused VQ matmul backward wrt input: `dx = grad_out @ recon`.
        fn vq_fused_matmul_dx_rb_f32(
            &self,
            g: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            dx: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        /// Fused VQ matmul backward wrt codebook: `dcb[k,dim]` (must be pre-zeroed).
        fn vq_fused_matmul_dcb_f32(
            &self,
            g: &CudaSlice<f32>,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            scale: &CudaSlice<f32>,
            dcb: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        /// Register-blocked fused VQ matmul backward wrt codebook: `G=gᵀ@x` tiled,
        /// scatter-added into `dcb[k,dim]` (must be pre-zeroed) via `idx`.
        fn vq_fused_matmul_dcb_rb_f32(
            &self,
            g: &CudaSlice<f32>,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            scale: &CudaSlice<f32>,
            dcb: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError>;
        fn vq_assign_f32(
            &self,
            normed: &CudaSlice<f32>,
            codebook: &CudaSlice<f32>,
            assign_out: &mut CudaSlice<f32>,
            n_vec: usize,
            dim: usize,
            k: usize,
        ) -> Result<(), CudaError>;
        /// Weight-offload VQ gather+denormalize: `recon[i] = codebook[assign[j]*dim +
        /// d] / (1/row_scale[row(j)])` (`j=i/dim`, `d=i%dim`), reconstructing the
        /// quantized tile from `[n_vec]` assignments + `[n_rows]` scales on-GPU
        /// instead of a full `[numel]` gather-index buffer.
        fn vq_gather_denorm_f32(
            &self,
            recon: &mut CudaSlice<f32>,
            codebook: &CudaSlice<f32>,
            assign: &CudaSlice<u32>,
            row_scale: &CudaSlice<f32>,
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Result<(), CudaError>;
        /// Stream-ordered (async, non-host-blocking) device-to-device copy with
        /// element offsets. Unlike `memcpy_dtod_f32` this
        /// does not synchronize the host — used to assemble tiled outputs on-device
        /// without a per-tile stall.
        fn memcpy_dtod_async_f32(
            &self,
            dst: &mut CudaSlice<f32>,
            dst_offset: usize,
            src: &CudaSlice<f32>,
            src_offset: usize,
            count: usize,
        ) -> Result<(), CudaError>;
        /// NVFP4 quantize (Blackwell only): dense f32 → e2m1 nibbles (2/byte, low-first)
        /// + one ue4m3 scale byte per 16-element block, byte-identical to the reference CPU
        /// `quantize_nvfp4`. `nblocks = numel/16`; buffers are u32-typed for the pool:
        /// `packed` holds `numel/8` words (= `numel/2` nibble-pair bytes), `scales`
        /// `numel/64` words (read by the GEMM as `u32[rows][K/64]`).
        fn nvfp4_quant_f32(
            &self,
            w: &CudaSlice<f32>,
            packed: &mut CudaSlice<u32>,
            scales: &mut CudaSlice<u32>,
            nblocks: usize,
        ) -> Result<(), CudaError>;
        /// `nvfp4_quant_f32` with STOCHASTIC ROUNDING of the e2m1 nibbles (round up
        /// with probability proportional to the position between grid points — an
        /// unbiased estimator). For GRADIENT operands in low-bit backward passes;
        /// round-to-nearest gradients bias the descent (the FP4_BWD 150-step failure).
        /// `seed` must vary per call (counter) so rounding noise decorrelates per step.
        fn nvfp4_quant_sr_f32(
            &self,
            w: &CudaSlice<f32>,
            packed: &mut CudaSlice<u32>,
            scales: &mut CudaSlice<u32>,
            nblocks: usize,
            seed: u64,
        ) -> Result<(), CudaError>;
        /// NVFP4 block-scaled tensor-core GEMM (Blackwell `mma.m16n8k64.kind::mxf4nvf4`):
        /// `c[m, ldc] += (a·sfa)(b·sfb)^T` into columns `[c0, c0+n_tile)`, both operands
        /// packed by `nvfp4_quant_f32`. Contract: `m%128==0`, `n_tile%64==0`,
        /// `k%256==0`; `c` pre-zeroed over the target columns (atomicAdd epilogue).
        #[allow(clippy::too_many_arguments)]
        fn nvfp4_gemm_sf2(
            &self,
            a: &CudaSlice<u32>,
            b: &CudaSlice<u32>,
            sfa: &CudaSlice<u32>,
            sfb: &CudaSlice<u32>,
            c: &mut CudaSlice<f32>,
            m: usize,
            n_tile: usize,
            k: usize,
            ldc: usize,
            c0: usize,
        ) -> Result<(), CudaError>;
    }

    impl SubbitBackendExt for CudaBackend {
        /// Weight-offload VQ normalize: `out[i] = w[i] * (1/row_scale[row(i)])`,
        /// computing the per-row scale on-GPU so the tiled quantize uploads only the
        /// `[n_rows]` row scales instead of a full `[numel]` inv_scale buffer.
        fn vq_normalize_f32(
            &self,
            out: &mut CudaSlice<f32>,
            w: &CudaSlice<f32>,
            row_scale: &CudaSlice<f32>,
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_normalize_f32")?;
            let n = n_vec * dim;
            let cfg = cuda_kernels::launch_config(n);
            // SAFETY: kernel `vq_normalize_f32` in kernels/vq_offload.ptx takes exactly these 6
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(w)
                    .arg(row_scale)
                    .arg(out)
                    .arg(&(n_vec as u32))
                    .arg(&(dim as u32))
                    .arg(&(vec_per_row as u32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Fully self-contained fused VQ nearest-code assignment:
        /// `assign[j] = argmax_c(v_j·c − ½‖c‖²)` over the `k` codebook entries, one
        /// thread per weight-vector. Computes `−½‖c‖²` internally in shared memory —
        /// takes ONLY `normed` + `codebook`, materializes no `[n_vec, k]` scores
        /// matrix. Replaces the cuBLAS scores GEMM + `mul`/`sum_dim` + argmax.
        /// Fused VQ sub-bit matmul forward: `out[n,out_f] = x · recon^T`, recon
        /// gathered from (idx,cb,scale) with no dense weight materialized.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_f32")?;
            let (bx, by) = (16u32, 16u32);
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(bx), (out_f as u32).div_ceil(by), 1),
                block_dim: (bx, by, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Tiled (shared-memory) fused VQ matmul forward — high-throughput variant
        /// of `vq_fused_matmul_f32`. Same result, 16x16 smem GEMM with codebook gather.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_tiled_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            k: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_tiled_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((out_f as u32).div_ceil(16), (n as u32).div_ceil(16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let use_smem = subbit_smem_flag();
            // SAFETY: kernel `vq_fused_matmul_tiled_f32` in kernels/vq_fused_matmul.ptx takes exactly these 12
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .arg(&(k as i32))
                    .arg(&use_smem)
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Warp-per-output-column fused VQ GEMV (batch-1 decode). Block (32, W); one
        /// warp reduces one output column via a shfl_down tree. grid.x = ceil(out_f/W),
        /// grid.y = rows (n==1 at decode). 32x the resident threads of the 16x16 tiled
        /// path with a coalesced idx read => hides the dependent gather latency the tiled
        /// GEMV can't at n==1 (where it wastes 15/16 threads + syncs per k-tile). Same
        /// packed weight & scale-into-B fold as `vq_fused_matmul_tiled_f32`; re-associated
        /// reduction (FNV-gated argmax-identical, not bit-identical). W via env sweep.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_tiled_warp_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            k: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func(if subbit_f16_cb() {
                "vq_fused_matmul_tiled_warp_f16cb"
            } else {
                "vq_fused_matmul_tiled_warp_f32"
            })?;
            let warps: u32 = subbit_tiled_warp_w();
            let cfg = LaunchConfig {
                grid_dim: ((out_f as u32).div_ceil(warps), n as u32, 1),
                block_dim: (32, warps, 1),
                shared_mem_bytes: 0,
            };
            let use_smem = subbit_smem_flag();
            // SAFETY: kernel `vq_fused_matmul_tiled_warp_f16cb` / `vq_fused_matmul_tiled_warp_f32` in kernels/vq_fused_matmul.ptx takes exactly these 12
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .arg(&(k as i32))
                    .arg(&use_smem)
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Grouped tiled fused VQ matmul forward — runs `n_groups` independent VQ
        /// GEMVs (one per selected MoE expert) in a SINGLE launch (grid.z = group).
        /// Per group the arithmetic is byte-identical to `vq_fused_matmul_tiled_f32`
        /// for that expert; the weights are selected from the concatenated per-expert
        /// buffers via `sel[g]`. `x_stride==0` broadcasts one input row to every group
        /// (gate/up); `x_stride==n*in_f` gives each group its own row (down).
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_warp_rc_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            k_codes: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_warp_rc_f32")?;
            let warps: u32 = 8;
            let cfg = LaunchConfig {
                grid_dim: (
                    (out_f as u32).div_ceil(warps * 4),
                    (n as u32).div_ceil(4),
                    1,
                ),
                block_dim: (32, warps, 1),
                shared_mem_bytes: 0,
            };
            let use_smem = subbit_smem_flag();
            // SAFETY: kernel `vq_fused_matmul_warp_rc_f32` in kernels/vq_fused_matmul.ptx takes exactly these 12
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .arg(&(k_codes as i32))
                    .arg(&use_smem)
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_grouped_f32(
            &self,
            x: &CudaSlice<f32>,
            idx_all: &CudaSlice<u8>,
            cb_all: &CudaSlice<f32>,
            scale_all: &CudaSlice<f32>,
            sel: &CudaSlice<i32>,
            out: &mut CudaSlice<f32>,
            n_groups: usize,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            idx_stride: usize,
            cb_stride: usize,
            scale_stride: usize,
            x_stride: usize,
        ) -> Result<(), CudaError> {
            let f16 = subbit_f16_cb();
            // The 4-way-MLP variant only implements the smem + dim==8 case; anything
            // else falls back to the general kernel.
            let smem_ok = f16 && dim == 8 && subbit_smem_flag() != 0 && cb_stride <= 2048;
            let cols = if smem_ok { subbit_cols() } else { 0 };
            let mlp4 = smem_ok && subbit_mlp4();
            let func = subbit_func(if cols == 8 {
                "vq_fused_matmul_grouped_warp_f16cb_c8"
            } else if cols == 4 {
                "vq_fused_matmul_grouped_warp_f16cb_c4"
            } else if mlp4 {
                "vq_fused_matmul_grouped_warp_f16cb_mlp4"
            } else if f16 {
                "vq_fused_matmul_grouped_warp_f16cb"
            } else {
                "vq_fused_matmul_grouped_warp_f32"
            })?;
            // Warp-per-output-column GEMV: block (32, W), one warp reduces one column via
            // a shfl_down tree. grid.x = ceil(out_f/W), grid.y = rows (n==1), grid.z =
            // expert group. 32x the resident threads of the thread-per-column path with a
            // coalesced idx read => hides the dependent gather latency that stalls the
            // batch-1 decode GEMV. Re-associates the reduction (not bit-identical) but is
            // argmax-identical on the decode stream (FNV-gated). Best of the sweep.
            let warps: u32 = 8;
            // c4 gives each warp FOUR columns, so the grid shrinks to match.
            let per_block = if cols > 0 { warps * cols as u32 } else { warps };
            let cfg = LaunchConfig {
                grid_dim: (
                    (out_f as u32).div_ceil(per_block),
                    n as u32,
                    n_groups as u32,
                ),
                block_dim: (32, warps, 1),
                shared_mem_bytes: 0,
            };
            let use_smem = subbit_smem_flag();
            // SAFETY: kernel `vq_fused_matmul_grouped_warp_f16cb_c8` / `vq_fused_matmul_grouped_warp_f16cb_c4` / `vq_fused_matmul_grouped_warp_f16cb_mlp4` / `vq_fused_matmul_grouped_warp_f16cb` / `vq_fused_matmul_grouped_warp_f32` in kernels/vq_fused_matmul.ptx takes exactly these 16
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx_all)
                    .arg(cb_all)
                    .arg(scale_all)
                    .arg(sel)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .arg(&(idx_stride as i32))
                    .arg(&(cb_stride as i32))
                    .arg(&(scale_stride as i32))
                    .arg(&(x_stride as i32))
                    .arg(&use_smem)
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn vq_fused_gate_up_swiglu_grouped_f32(
            &self,
            x: &CudaSlice<f32>,
            g_idx_all: &CudaSlice<u8>,
            g_cb_all: &CudaSlice<f32>,
            g_scale_all: &CudaSlice<f32>,
            u_idx_all: &CudaSlice<u8>,
            u_cb_all: &CudaSlice<f32>,
            u_scale_all: &CudaSlice<f32>,
            sel: &CudaSlice<i32>,
            act: &mut CudaSlice<f32>,
            n_groups: usize,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
            idx_stride: usize,
            cb_stride: usize,
            scale_stride: usize,
            x_stride: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func(if subbit_f16_cb() {
                "vq_fused_gate_up_swiglu_grouped_warp_f16cb"
            } else {
                "vq_fused_gate_up_swiglu_grouped_warp_f32"
            })?;
            // Same warp-per-column map as the grouped GEMV (block (32,W), grid
            // (ceil(out_f/W), n, groups)); each warp now reduces BOTH the gate and up
            // column and applies SwiGLU. Two independent gather->dot chains per warp
            // hide the batch-1 codebook-gather stall. Bit-identical to the 3-launch path.
            let warps: u32 = 8;
            let cfg = LaunchConfig {
                grid_dim: ((out_f as u32).div_ceil(warps), n as u32, n_groups as u32),
                block_dim: (32, warps, 1),
                shared_mem_bytes: 0,
            };
            let use_smem = subbit_smem_flag();
            // SAFETY: kernel `vq_fused_gate_up_swiglu_grouped_warp_f16cb` / `vq_fused_gate_up_swiglu_grouped_warp_f32` in kernels/vq_fused_matmul.ptx takes exactly these 19
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(g_idx_all)
                    .arg(g_cb_all)
                    .arg(g_scale_all)
                    .arg(u_idx_all)
                    .arg(u_cb_all)
                    .arg(u_scale_all)
                    .arg(sel)
                    .arg(act)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .arg(&(idx_stride as i32))
                    .arg(&(cb_stride as i32))
                    .arg(&(scale_stride as i32))
                    .arg(&(x_stride as i32))
                    .arg(&use_smem)
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Fused MoE top-k combine: `out[h] = Σ_g rows[g,:]·w[g]` accumulated in g-order
        /// from zero. Byte-identical to a chain of `scaled_add_inplace_f32` launches, so
        /// it replaces the top_k per-expert scaled-adds of a MoE layer with one launch.
        fn vq_moe_combine_f32(
            &self,
            rows: &CudaSlice<f32>,
            w: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            h: usize,
            g_count: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_moe_combine_f32")?;
            let block = 256u32;
            let cfg = LaunchConfig {
                grid_dim: ((h as u32).div_ceil(block), 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_moe_combine_f32` in kernels/vq_fused_matmul.ptx takes exactly these 5
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(rows)
                    .arg(w)
                    .arg(out)
                    .arg(&(h as i32))
                    .arg(&(g_count as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Device MoE router top-k: per row, softmax over `ne` router logits, pick
        /// the top_k experts, (optionally) renormalize — writing expert indices +
        /// weights to device buffers with NO host round-trip. One block per row.
        #[allow(clippy::too_many_arguments)]
        fn vq_router_topk_f32(
            &self,
            logits: &CudaSlice<f32>,
            sel: &mut CudaSlice<i32>,
            wout: &mut CudaSlice<f32>,
            rows: usize,
            ne: usize,
            top_k: usize,
            norm: bool,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_router_topk_f32")?;
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_router_topk_f32` in kernels/vq_fused_matmul.ptx takes exactly these 6
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(logits)
                    .arg(sel)
                    .arg(wout)
                    .arg(&(ne as i32))
                    .arg(&(top_k as i32))
                    .arg(&(norm as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Dense fp32 router GEMV: `out[n,N] = x[n,K] · w[K,N]` (row-major weight).
        /// One block per row; threads cover the `N` output columns (grid-stride when
        /// `N > block`). Coalesced weight loads (`w[k*N + c]` across consecutive `c`).
        fn vq_router_gemv_f32(
            &self,
            x: &CudaSlice<f32>,
            w: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            k: usize,
            n_out: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_router_gemv_f32")?;
            // One block per row, threads over the N columns (coalesced weight loads).
            // Simple 1D block: a 2D/1024-thread variant broke graph capture and did not
            // speed up the single-SM decode GEMV anyway.
            let block = (n_out as u32).min(256).max(1);
            let cfg = LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_router_gemv_f32` in kernels/vq_fused_matmul.ptx takes exactly these 6
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(w)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(k as i32))
                    .arg(&(n_out as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Fused decode attention (one block per query head; `hd`-wide reductions via
        /// warp shuffle). Dynamic shared mem = `(max_ctx + hd)` floats: the score row
        /// plus the cached q row.
        #[allow(clippy::too_many_arguments)]
        fn vq_decode_attn_f32(
            &self,
            q: &CudaSlice<f32>,
            k: &CudaSlice<f32>,
            v: &CudaSlice<f32>,
            mask: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            nh: usize,
            nkv: usize,
            max_ctx: usize,
            hd: usize,
            scale: f32,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_decode_attn_f32")?;
            let block = 128u32;
            let cfg = LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: ((max_ctx + hd) as u32) * 4,
            };
            // SAFETY: kernel `vq_decode_attn_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(q)
                    .arg(k)
                    .arg(v)
                    .arg(mask)
                    .arg(out)
                    .arg(&(nh as i32))
                    .arg(&(nkv as i32))
                    .arg(&(max_ctx as i32))
                    .arg(&(hd as i32))
                    .arg(&scale)
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn rope_split_halves_bhsd_devpos_f32(
            &self,
            src: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            pos: &CudaSlice<f32>,
            seq: usize,
            n_heads: usize,
            head_dim: usize,
            theta: f32,
        ) -> Result<(), CudaError> {
            let func = subbit_func("rope_split_halves_bhsd_devpos_f32")?;
            // grid = (seq, n_heads, bs=1), block = (head_dim/2). Matches the eager
            // rope_split_halves_bhsd launch geometry.
            let cfg = LaunchConfig {
                grid_dim: (seq as u32, n_heads as u32, 1),
                block_dim: ((head_dim / 2) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `rope_split_halves_bhsd_devpos_f32` in kernels/vq_fused_matmul.ptx takes exactly these 7
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(src)
                    .arg(out)
                    .arg(pos)
                    .arg(&(seq as u32))
                    .arg(&(n_heads as u32))
                    .arg(&(head_dim as u32))
                    .arg(&theta)
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        fn kv_scatter_devpos_f32(
            &self,
            src: &CudaSlice<f32>,
            dst: &mut CudaSlice<f32>,
            pos: &CudaSlice<f32>,
            nkv: usize,
            t: usize,
            hd: usize,
            max_ctx: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("kv_scatter_devpos_f32")?;
            let total = (nkv * t * hd) as u32;
            let block = 256u32;
            let grid = ((total + block - 1) / block).max(1);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `kv_scatter_devpos_f32` in kernels/vq_fused_matmul.ptx takes exactly these 7
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(src)
                    .arg(dst)
                    .arg(pos)
                    .arg(&(nkv as i32))
                    .arg(&(t as i32))
                    .arg(&(hd as i32))
                    .arg(&(max_ctx as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Register-blocked fused VQ matmul forward (64x64 tile, 4x4 microtile) —
        /// the high-throughput production forward. Same result as the naive variant.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_rb_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_rb_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((out_f as u32).div_ceil(64), (n as u32).div_ceil(64), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_rb_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Fused VQ matmul backward wrt input: `dx[n,in_f]`.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_dx_f32(
            &self,
            g: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            dx: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_dx_f32")?;
            let (bx, by) = (16u32, 16u32);
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(bx), (in_f as u32).div_ceil(by), 1),
                block_dim: (bx, by, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_dx_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(g)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(dx)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// 8x8-microtile / 128x128-tile fused VQ matmul forward — max-throughput
        /// variant (same result as `vq_fused_matmul_rb_f32`).
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_rb8_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_rb8_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((out_f as u32).div_ceil(128), (n as u32).div_ceil(128), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_rb8_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// cp.async-pipelined tf32 tensor-core fused VQ matmul forward (64x64 tile,
        /// codebook resident in smem, activation tile cp.async-loaded). Assumes k*dim<=4096.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_ca_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_ca_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((out_f as u32).div_ceil(64), (n as u32).div_ceil(64), 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_ca_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Reconstruct the dense `recon[out_f,in_f]` from the packed weight, so a
        /// tf32 cuBLAS GEMM can run off it at tensor-core speed.
        fn vq_reconstruct_f32(
            &self,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            recon: &mut CudaSlice<f32>,
            out_f: usize,
            in_f: usize,
            dim: usize,
            vpr: usize,
            o_base: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_reconstruct_f32")?;
            let total = (out_f * in_f) as u32;
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_reconstruct_f32` in kernels/vq_fused_matmul.ptx takes exactly these 9
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(recon)
                    .arg(&(out_f as i32))
                    .arg(&(in_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .arg(&(o_base as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// NVFP4 quantize (Blackwell only) — see trait docs; byte-identical to the CPU codec.
        fn nvfp4_quant_f32(
            &self,
            w: &CudaSlice<f32>,
            packed: &mut CudaSlice<u32>,
            scales: &mut CudaSlice<u32>,
            nblocks: usize,
        ) -> Result<(), CudaError> {
            let func = nvfp4_func("nvfp4_quant_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((nblocks as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nvfp4_quant_f32` in kernels/nvfp4.ptx takes exactly these 4
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(w)
                    .arg(packed)
                    .arg(scales)
                    .arg(&(nblocks as i64))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// NVFP4 stochastic-rounding quantize (Blackwell only) — see trait docs.
        fn nvfp4_quant_sr_f32(
            &self,
            w: &CudaSlice<f32>,
            packed: &mut CudaSlice<u32>,
            scales: &mut CudaSlice<u32>,
            nblocks: usize,
            seed: u64,
        ) -> Result<(), CudaError> {
            let func = nvfp4_func("nvfp4_quant_sr_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((nblocks as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nvfp4_quant_sr_f32` in kernels/nvfp4.ptx takes exactly these 5
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(w)
                    .arg(packed)
                    .arg(scales)
                    .arg(&(nblocks as i64))
                    .arg(&seed)
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// NVFP4 block-scaled tensor-core GEMM (Blackwell only) — see trait docs.
        #[allow(clippy::too_many_arguments)]
        fn nvfp4_gemm_sf2(
            &self,
            a: &CudaSlice<u32>,
            b: &CudaSlice<u32>,
            sfa: &CudaSlice<u32>,
            sfb: &CudaSlice<u32>,
            c: &mut CudaSlice<f32>,
            m: usize,
            n_tile: usize,
            k: usize,
            ldc: usize,
            c0: usize,
        ) -> Result<(), CudaError> {
            debug_assert!(
                m % 128 == 0 && n_tile % 64 == 0 && k % 256 == 0,
                "nvfp4_gemm_sf2 shape contract"
            );
            let func = nvfp4_func("nvfp4_gemm_sf2")?;
            let cfg = LaunchConfig {
                grid_dim: ((n_tile / 64) as u32, (m / 128) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nvfp4_gemm_sf2` in kernels/nvfp4.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(a)
                    .arg(b)
                    .arg(sfa)
                    .arg(sfb)
                    .arg(c)
                    .arg(&(m as i32))
                    .arg(&(n_tile as i32))
                    .arg(&(k as i32))
                    .arg(&(ldc as i32))
                    .arg(&(c0 as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Scatter a precomputed dW tile (`g^T @ x`, rows [o_base, o_base+ot)) into the
        /// codebook grad — the second half of the tf32 backward (the GEMM runs on
        /// cuBLAS-tf32). Matches `vq_fused_matmul_dcb_f32`'s scatter exactly. `dcb`
        /// must be pre-zeroed; accumulates atomically across the (o,v)->code sharing.
        #[allow(clippy::too_many_arguments)]
        fn vq_dcb_scatter_f32(
            &self,
            dw: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            scale: &CudaSlice<f32>,
            dcb: &mut CudaSlice<f32>,
            ot: usize,
            in_f: usize,
            dim: usize,
            vpr: usize,
            o_base: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_dcb_scatter_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((ot as u32).div_ceil(16), (vpr as u32).div_ceil(16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_dcb_scatter_f32` in kernels/vq_fused_matmul.ptx takes exactly these 9
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(dw)
                    .arg(idx)
                    .arg(scale)
                    .arg(dcb)
                    .arg(&(ot as i32))
                    .arg(&(in_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .arg(&(o_base as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// tf32 tensor-core fused VQ matmul forward — highest throughput (approximate:
        /// tf32 ~10-bit mantissa). 4 warps/block, 32x32 output tile via wmma m16n16k8.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_tc_f32(
            &self,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            out: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_tc_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((out_f as u32).div_ceil(64), (n as u32).div_ceil(64), 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_tc_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(x)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(out)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Register-blocked fused VQ matmul backward wrt input: `dx = grad_out @ recon`.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_dx_rb_f32(
            &self,
            g: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            cb: &CudaSlice<f32>,
            scale: &CudaSlice<f32>,
            dx: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_dx_rb_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((in_f as u32).div_ceil(64), (n as u32).div_ceil(64), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_dx_rb_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(g)
                    .arg(idx)
                    .arg(cb)
                    .arg(scale)
                    .arg(dx)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Fused VQ matmul backward wrt codebook: `dcb[k,dim]` (must be pre-zeroed).
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_dcb_f32(
            &self,
            g: &CudaSlice<f32>,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            scale: &CudaSlice<f32>,
            dcb: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_dcb_f32")?;
            let (bx, by) = (16u32, 16u32);
            let cfg = LaunchConfig {
                grid_dim: ((out_f as u32).div_ceil(bx), (vpr as u32).div_ceil(by), 1),
                block_dim: (bx, by, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_dcb_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(g)
                    .arg(x)
                    .arg(idx)
                    .arg(scale)
                    .arg(dcb)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Register-blocked fused VQ matmul backward wrt codebook: `G=gᵀ@x` tiled,
        /// scatter-added into `dcb[k,dim]` (must be pre-zeroed) via `idx`.
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_matmul_dcb_rb_f32(
            &self,
            g: &CudaSlice<f32>,
            x: &CudaSlice<f32>,
            idx: &CudaSlice<u32>,
            scale: &CudaSlice<f32>,
            dcb: &mut CudaSlice<f32>,
            n: usize,
            in_f: usize,
            out_f: usize,
            dim: usize,
            vpr: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_fused_matmul_dcb_rb_f32")?;
            let cfg = LaunchConfig {
                grid_dim: ((in_f as u32).div_ceil(64), (out_f as u32).div_ceil(64), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `vq_fused_matmul_dcb_rb_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(g)
                    .arg(x)
                    .arg(idx)
                    .arg(scale)
                    .arg(dcb)
                    .arg(&(n as i32))
                    .arg(&(in_f as i32))
                    .arg(&(out_f as i32))
                    .arg(&(dim as i32))
                    .arg(&(vpr as i32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        fn vq_assign_f32(
            &self,
            normed: &CudaSlice<f32>,
            codebook: &CudaSlice<f32>,
            assign_out: &mut CudaSlice<f32>,
            n_vec: usize,
            dim: usize,
            k: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_assign_f32")?;
            let block = 256u32;
            let grid = (n_vec as u32).div_ceil(block);
            let smem = ((k * dim + k) * std::mem::size_of::<f32>()) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: smem,
            };
            // SAFETY: kernel `vq_assign_f32` in kernels/vq_assign.ptx takes exactly these 6
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(normed)
                    .arg(codebook)
                    .arg(assign_out)
                    .arg(&(n_vec as u32))
                    .arg(&(dim as u32))
                    .arg(&(k as u32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Weight-offload VQ gather+denormalize: `recon[i] = codebook[assign[j]*dim +
        /// d] / (1/row_scale[row(j)])` (`j=i/dim`, `d=i%dim`), reconstructing the
        /// quantized tile from `[n_vec]` assignments + `[n_rows]` scales on-GPU
        /// instead of a full `[numel]` gather-index buffer.
        fn vq_gather_denorm_f32(
            &self,
            recon: &mut CudaSlice<f32>,
            codebook: &CudaSlice<f32>,
            assign: &CudaSlice<u32>,
            row_scale: &CudaSlice<f32>,
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Result<(), CudaError> {
            let func = subbit_func("vq_gather_denorm_f32")?;
            let n = n_vec * dim;
            let cfg = cuda_kernels::launch_config(n);
            // SAFETY: kernel `vq_gather_denorm_f32` in kernels/vq_offload.ptx takes exactly these 7
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                self.stream()
                    .launch_builder(&func)
                    .arg(codebook)
                    .arg(assign)
                    .arg(row_scale)
                    .arg(recon)
                    .arg(&(n_vec as u32))
                    .arg(&(dim as u32))
                    .arg(&(vec_per_row as u32))
                    .launch(cfg)
                    .map(|_| ())
                    .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }

        /// Stream-ordered (async, non-host-blocking) device-to-device copy with
        /// element offsets. Unlike [`memcpy_dtod_f32`](Self::memcpy_dtod_f32) this
        /// does not synchronize the host — used to assemble tiled outputs on-device
        /// without a per-tile stall.
        fn memcpy_dtod_async_f32(
            &self,
            dst: &mut CudaSlice<f32>,
            dst_offset: usize,
            src: &CudaSlice<f32>,
            src_offset: usize,
            count: usize,
        ) -> Result<(), CudaError> {
            assert!(
                src_offset + count <= src.len() && dst_offset + count <= dst.len(),
                "memcpy_dtod_async_f32: {count} at src {src_offset} (len {}) / dst {dst_offset} (len {})",
                src.len(),
                dst.len()
            );
            use cudarc::driver::DevicePtr as _;
            let (src_ptr, _guard_s) = src.device_ptr(&self.stream());
            let src_ptr = src_ptr
                + (src_offset * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
            use cudarc::driver::DevicePtrMut as _;
            let (dst_ptr, _guard_d) = dst.device_ptr_mut(&self.stream());
            let dst_ptr = dst_ptr
                + (dst_offset * std::mem::size_of::<f32>()) as cudarc::driver::sys::CUdeviceptr;
            let size = count * std::mem::size_of::<f32>();
            // SAFETY: both element ranges are bounds-checked above, the pointers are
            // offsets within the two borrowed slices, and the copy is queued on the
            // backend's single stream, so it is ordered after the producers of `src`.
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dst_ptr,
                    src_ptr,
                    size,
                    self.stream().cu_stream(),
                )
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
            }
            Ok(())
        }
    }

    // == tensor ext ==
    // ── NemotronH (Mamba2) fused decode step ──

    /// Device-resident per-layer Mamba2 decode state + constants: the depthwise-conv
    /// CIRCULAR ring (host tracks the slot), the SSD state, and the layer's small
    /// fp params — uploaded once, mutated in place every token by
    /// [`nh_mamba_decode`], so the token's mamba mixer never leaves the device
    /// between the in/out projections.
    pub struct NhMambaGpu {
        ring: CudaSlice<f32>,
        state: CudaSlice<f32>,
        /// Persistent per-call scratch (conv output / scan output): allocated once
        /// so the decode step does NO pool alloc/free — a hard requirement for
        /// CUDA-graph capture (a raw mid-capture free invalidates the capture).
        xbc: CudaSlice<f32>,
        y: CudaSlice<f32>,
        conv_w: CudaSlice<f32>,
        conv_b: CudaSlice<f32>,
        a_log: CudaSlice<f32>,
        dvec: CudaSlice<f32>,
        dt_bias: CudaSlice<f32>,
        norm_w: CudaSlice<f32>,
        pos: usize,
        k_conv: usize,
    }

    impl NhMambaGpu {
        /// Copy this sequence's ring and SSD state to the host, plus the ring index
        /// they are rotated for, so the state can be migrated into a batch slot.
        #[must_use]
        pub fn snapshot(&self) -> (Vec<f32>, Vec<f32>, usize) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let ring = cuda.stream().clone_dtoh(&self.ring).expect("ring D2H");
            let state = cuda.stream().clone_dtoh(&self.state).expect("state D2H");
            (ring, state, self.pos)
        }
    }

    // SAFETY: holds plain device handles (ring, state, scratch) whose mutation goes through
    // `&mut self` on the backend's single stream, so a move between threads cannot race
    // host memory and `&self` only reads handles.
    unsafe impl Send for NhMambaGpu {}
    // SAFETY: see the `Send` impl above; shared references only ever read device handles.
    unsafe impl Sync for NhMambaGpu {}

    impl NhMambaGpu {
        /// Upload the layer constants and zero the recurrent state.
        #[allow(clippy::too_many_arguments)]
        #[must_use]
        pub fn upload(
            conv_w: &[f32],
            conv_b: &[f32],
            a_log: &[f32],
            dvec: &[f32],
            dt_bias: &[f32],
            norm_w: &[f32],
            conv_dim: usize,
            k_conv: usize,
            state_len: usize,
            d_inner: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let up = |v: &[f32]| -> CudaSlice<f32> {
                let mut s = pool_alloc_uninit(v.len()).expect("nh const alloc");
                cuda.htod_into(v, &mut s).expect("nh const htod");
                s
            };
            let zeros = |n: usize| -> CudaSlice<f32> { up(&vec![0f32; n]) };
            let out = Self {
                ring: zeros(k_conv * conv_dim),
                state: zeros(state_len),
                xbc: zeros(conv_dim),
                y: zeros(d_inner),
                conv_w: up(conv_w),
                conv_b: up(conv_b),
                a_log: up(a_log),
                dvec: up(dvec),
                dt_bias: up(dt_bias),
                norm_w: up(norm_w),
                pos: 0,
                k_conv,
            };
            cuda.sync();
            out
        }

        /// Snapshot the recurrent state (ring + SSD) to host — used to make the
        /// non-idempotent SSD update safe across the graph-capture dance (reference
        /// run / pool prewarm / capture / verify all re-execute the same token).
        #[must_use]
        pub fn save(&self) -> (Vec<f32>, Vec<f32>) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let mut ring = vec![0f32; self.ring.len()];
            let mut state = vec![0f32; self.state.len()];
            cuda.dtoh_into_n(&self.ring, self.ring.len(), &mut ring)
                .expect("nh ring save");
            cuda.dtoh_into_n(&self.state, self.state.len(), &mut state)
                .expect("nh state save");
            (ring, state)
        }

        /// Restore a [`save`](Self::save) snapshot.
        pub fn load(&mut self, snap: &(Vec<f32>, Vec<f32>)) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            cuda.htod_into(&snap.0, &mut self.ring)
                .expect("nh ring load");
            cuda.htod_into(&snap.1, &mut self.state)
                .expect("nh state load");
        }

        /// Zero the recurrent state (new sequence).
        pub fn reset(&mut self) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let rz = vec![0f32; self.ring.len()];
            let sz = vec![0f32; self.state.len()];
            cuda.htod_into(&rz, &mut self.ring).expect("nh ring reset");
            cuda.htod_into(&sz, &mut self.state)
                .expect("nh state reset");
            self.pos = 0;
        }
    }

    /// Device-pos twin of [`nh_mamba_decode`]: the conv ring slot comes from the
    /// f32 scalar `pos[0]` (integer value; the kernel mods by K), so a CAPTURED
    /// decode graph advances the window across replays with the host only
    /// refreshing the pos scalar. `bufs.pos` is not used or advanced.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn nh_mamba_decode_devpos(
        zxbcdt: &Tensor<f32>,
        bufs: &mut NhMambaGpu,
        pos: &Tensor<f32>,
        d_inner: usize,
        conv_dim: usize,
        heads: usize,
        hd: usize,
        ns: usize,
        ngroups: usize,
        dt_min: f32,
        eps: f32,
    ) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let zc = zxbcdt.contiguous();
        let pc = pos.contiguous();
        let z_g = zc.as_cuda_slice_read();
        let p_g = pc.as_cuda_slice_read();
        let mut out = pool_alloc_uninit(d_inner).expect("nh out alloc");
        let k_conv = bufs.k_conv;
        {
            let func = subbit_func("nh_conv_silu_devpos_f32").expect("nh_conv_silu_devpos_f32");
            let cfg = LaunchConfig {
                grid_dim: ((conv_dim as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_conv_silu_devpos_f32` in kernels/vq_fused_matmul.ptx takes exactly these 9
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(z_g.slice())
                    .arg(&mut bufs.ring)
                    .arg(&bufs.conv_w)
                    .arg(&bufs.conv_b)
                    .arg(p_g.slice())
                    .arg(&mut bufs.xbc)
                    .arg(&(d_inner as i32))
                    .arg(&(conv_dim as i32))
                    .arg(&(k_conv as i32))
                    .launch(cfg)
                    .expect("nh_conv_silu_devpos launch");
            }
        }
        {
            let func = subbit_func("nh_ssd_scan_f32").expect("nh_ssd_scan_f32");
            let cfg = LaunchConfig {
                grid_dim: (heads as u32, 1, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_conv_silu_devpos_f32` / `nh_ssd_scan_f32` in kernels/vq_fused_matmul.ptx takes exactly these 14
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(z_g.slice())
                    .arg(&bufs.xbc)
                    .arg(&mut bufs.state)
                    .arg(&bufs.a_log)
                    .arg(&bufs.dvec)
                    .arg(&bufs.dt_bias)
                    .arg(&mut bufs.y)
                    .arg(&(d_inner as i32))
                    .arg(&(conv_dim as i32))
                    .arg(&(heads as i32))
                    .arg(&(hd as i32))
                    .arg(&(ns as i32))
                    .arg(&(ngroups as i32))
                    .arg(&dt_min)
                    .launch(cfg)
                    .expect("nh_ssd_scan launch");
            }
        }
        {
            let func = subbit_func("nh_gated_gnorm_f32").expect("nh_gated_gnorm_f32");
            let cfg = LaunchConfig {
                grid_dim: (ngroups as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_conv_silu_devpos_f32` / `nh_ssd_scan_f32` / `nh_gated_gnorm_f32` in kernels/vq_fused_matmul.ptx takes exactly these 7
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(&bufs.y)
                    .arg(z_g.slice())
                    .arg(&bufs.norm_w)
                    .arg(&mut out)
                    .arg(&(d_inner as i32))
                    .arg(&(ngroups as i32))
                    .arg(&eps)
                    .launch(cfg)
                    .expect("nh_gated_gnorm launch");
            }
        }
        let shape = [1usize, d_inner];
        Tensor::from_storage(Storage::from_cuda_slice(out, d_inner, zc.device()), &shape)
            .expect("nh mamba out from_storage")
    }

    /// One fused Mamba2 decode step on the device: conv+silu (circular ring), the
    /// per-head SSD recurrence (state updated in place), and the gated group-RMSNorm.
    /// `zxbcdt` is the GPU-resident in_proj output `[1, d_inner+conv_dim+heads]`;
    /// returns the normed `[1, d_inner]` mixer output on the device (out_proj input).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn nh_mamba_decode(
        zxbcdt: &Tensor<f32>,
        bufs: &mut NhMambaGpu,
        d_inner: usize,
        conv_dim: usize,
        heads: usize,
        hd: usize,
        ns: usize,
        ngroups: usize,
        dt_min: f32,
        eps: f32,
    ) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let zc = zxbcdt.contiguous();
        let z_g = zc.as_cuda_slice_read();
        let mut out = pool_alloc_uninit(d_inner).expect("nh out alloc");
        let k_conv = bufs.k_conv;
        {
            let func = subbit_func("nh_conv_silu_f32").expect("nh_conv_silu_f32");
            let cfg = LaunchConfig {
                grid_dim: ((conv_dim as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_conv_silu_f32` in kernels/vq_fused_matmul.ptx takes exactly these 9
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(z_g.slice())
                    .arg(&mut bufs.ring)
                    .arg(&bufs.conv_w)
                    .arg(&bufs.conv_b)
                    .arg(&mut bufs.xbc)
                    .arg(&(d_inner as i32))
                    .arg(&(conv_dim as i32))
                    .arg(&(k_conv as i32))
                    .arg(&(bufs.pos as i32))
                    .launch(cfg)
                    .expect("nh_conv_silu launch");
            }
        }
        {
            let func = subbit_func("nh_ssd_scan_f32").expect("nh_ssd_scan_f32");
            let cfg = LaunchConfig {
                grid_dim: (heads as u32, 1, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_conv_silu_f32` / `nh_ssd_scan_f32` in kernels/vq_fused_matmul.ptx takes exactly these 14
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(z_g.slice())
                    .arg(&bufs.xbc)
                    .arg(&mut bufs.state)
                    .arg(&bufs.a_log)
                    .arg(&bufs.dvec)
                    .arg(&bufs.dt_bias)
                    .arg(&mut bufs.y)
                    .arg(&(d_inner as i32))
                    .arg(&(conv_dim as i32))
                    .arg(&(heads as i32))
                    .arg(&(hd as i32))
                    .arg(&(ns as i32))
                    .arg(&(ngroups as i32))
                    .arg(&dt_min)
                    .launch(cfg)
                    .expect("nh_ssd_scan launch");
            }
        }
        {
            let func = subbit_func("nh_gated_gnorm_f32").expect("nh_gated_gnorm_f32");
            let cfg = LaunchConfig {
                grid_dim: (ngroups as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_conv_silu_f32` / `nh_ssd_scan_f32` / `nh_gated_gnorm_f32` in kernels/vq_fused_matmul.ptx takes exactly these 7
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(&bufs.y)
                    .arg(z_g.slice())
                    .arg(&bufs.norm_w)
                    .arg(&mut out)
                    .arg(&(d_inner as i32))
                    .arg(&(ngroups as i32))
                    .arg(&eps)
                    .launch(cfg)
                    .expect("nh_gated_gnorm launch");
            }
        }
        bufs.pos = (bufs.pos + 1) % k_conv;
        let shape = [1usize, d_inner];
        Tensor::from_storage(Storage::from_cuda_slice(out, d_inner, zc.device()), &shape)
            .expect("nh mamba out from_storage")
    }

    /// NemotronH sigmoid router on the device: top-k CHOSEN by `score + bias`,
    /// WEIGHTED by `score` (renormed when `norm`, then ×`scaling`) — the DeepSeek-
    /// style routing NemotronH uses, as device `sel`/`w` buffers so the MoE layer
    /// runs with zero host syncs. `bias` is the layer's `e_score_correction_bias`
    /// already on the device.
    #[must_use]
    pub fn nh_router_topk(
        logits: &Tensor<f32>,
        bias: &CudaSlice<f32>,
        ne: usize,
        top_k: usize,
        norm: bool,
        scaling: f32,
    ) -> RouterSel {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let lc = logits.contiguous();
        let rows = lc.numel() / ne;
        let n = rows * top_k;
        let mut sel = pool_alloc_uninit_i32(n).expect("nh router sel alloc");
        let mut w = pool_alloc_uninit(n).expect("nh router w alloc");
        {
            let logits_g = lc.as_cuda_slice_read();
            let func = subbit_func("nh_router_topk_f32").expect("nh_router_topk_f32");
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_router_topk_f32` in kernels/vq_fused_matmul.ptx takes exactly these 8
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(logits_g.slice())
                    .arg(bias)
                    .arg(&mut sel)
                    .arg(&mut w)
                    .arg(&(ne as i32))
                    .arg(&(top_k as i32))
                    .arg(&i32::from(norm))
                    .arg(&scaling)
                    .launch(cfg)
                    .expect("nh_router_topk launch");
            }
        }
        RouterSel {
            sel: Some(sel),
            w: Some(w),
        }
    }

    /// Own dense f32 GEMV: `x [1,K] · W[N,K]ᵀ → [1,N]` via the warp-per-row
    /// streaming kernel — no cuBLAS, so a decode step built on it CAPTURES.
    /// `w` is the UNTRANSPOSED dense mirror `[N, K]` (natural reconstruct layout).
    #[must_use]
    pub fn nh_dense_gemv(x: &Tensor<f32>, w: &Tensor<f32>) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let xc = x.contiguous();
        let wc = w.contiguous();
        let n = wc.shape()[0];
        let k = wc.shape()[1];
        debug_assert_eq!(xc.numel(), k, "nh_dense_gemv: x len vs K");
        let mut out = pool_alloc_uninit(n).expect("nh gemv out alloc");
        {
            let x_g = xc.as_cuda_slice_read();
            let w_g = wc.as_cuda_slice_read();
            let func = subbit_func("nh_dense_gemv_f32").expect("nh_dense_gemv_f32");
            let warps_per_block = 4u32;
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(warps_per_block), 1, 1),
                block_dim: (warps_per_block * 32, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_dense_gemv_f32` in kernels/vq_fused_matmul.ptx takes exactly these 5
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(x_g.slice())
                    .arg(w_g.slice())
                    .arg(&mut out)
                    .arg(&(n as i32))
                    .arg(&(k as i32))
                    .launch(cfg)
                    .expect("nh_dense_gemv launch");
            }
        }
        let shape = [1usize, n];
        Tensor::from_storage(Storage::from_cuda_slice(out, n, xc.device()), &shape)
            .expect("nh gemv from_storage")
    }

    // ── NemotronH batched (multi-agent) decode ──

    /// Per-LAYER recurrent state for M agent SLOTS, contiguous so one kernel can
    /// address every slot: `ring [M, K, conv_dim]`, `state [M, heads*hd*ns]`. The
    /// M=1 `NhMambaGpu` is the single-agent case of this; a multi-agent server
    /// allocates one of these per mamba layer and assigns each agent a slot.
    pub struct NhMambaBatch {
        ring: CudaSlice<f32>,
        state: CudaSlice<f32>,
        conv_w: CudaSlice<f32>,
        conv_b: CudaSlice<f32>,
        a_log: CudaSlice<f32>,
        dvec: CudaSlice<f32>,
        dt_bias: CudaSlice<f32>,
        norm_w: CudaSlice<f32>,
        xbc: CudaSlice<f32>,
        y: CudaSlice<f32>,
        slots: usize,
        k_conv: usize,
    }

    // SAFETY: holds plain device handles (per-slot ring, state, parameters) mutated only through
    // `&mut self` on the backend's single stream; `&self` only reads handles.
    unsafe impl Send for NhMambaBatch {}
    // SAFETY: see the `Send` impl above; shared references only ever read device handles.
    unsafe impl Sync for NhMambaBatch {}

    impl NhMambaBatch {
        /// Upload the layer constants once and zero `slots` copies of the state.
        #[allow(clippy::too_many_arguments)]
        #[must_use]
        pub fn upload(
            conv_w: &[f32],
            conv_b: &[f32],
            a_log: &[f32],
            dvec: &[f32],
            dt_bias: &[f32],
            norm_w: &[f32],
            conv_dim: usize,
            k_conv: usize,
            state_len: usize,
            d_inner: usize,
            slots: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let up = |v: &[f32]| -> CudaSlice<f32> {
                let mut s = pool_alloc_uninit(v.len()).expect("nh batch const alloc");
                cuda.htod_into(v, &mut s).expect("nh batch const htod");
                s
            };
            let zeros = |n: usize| -> CudaSlice<f32> { up(&vec![0f32; n]) };
            let out = Self {
                ring: zeros(slots * k_conv * conv_dim),
                state: zeros(slots * state_len),
                conv_w: up(conv_w),
                conv_b: up(conv_b),
                a_log: up(a_log),
                dvec: up(dvec),
                dt_bias: up(dt_bias),
                norm_w: up(norm_w),
                xbc: zeros(slots * conv_dim),
                y: zeros(slots * d_inner),
                slots,
                k_conv,
            };
            cuda.sync();
            out
        }

        /// Overwrite one slot's recurrent state from host buffers — the decode-side
        /// half of migrating a prefilled sequence into a batch slot. NOTE: the batch
        /// stores SSD state as `[head][state][dim]` (coalescing), not the
        /// `[head][dim][state]` a single-agent cache uses; a migration must transpose.
        pub fn load_slot(&mut self, slot: usize, ring: &[f32], state: &[f32], conv_dim: usize) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let rlen = self.k_conv * conv_dim;
            assert!(ring.len() == rlen, "ring length mismatch");
            let mut ring_view = self.ring.slice_mut(slot * rlen..(slot + 1) * rlen);
            cuda.stream()
                .memcpy_htod(ring, &mut ring_view)
                .expect("slot ring load");
            let slen = state.len();
            let mut st_view = self.state.slice_mut(slot * slen..(slot + 1) * slen);
            cuda.stream()
                .memcpy_htod(state, &mut st_view)
                .expect("slot state load");
        }

        /// Zero one slot's recurrent state (agent start / eviction).
        pub fn reset_slot(&mut self, slot: usize, conv_dim: usize, state_len: usize) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let rz = vec![0f32; self.k_conv * conv_dim];
            let sz = vec![0f32; state_len];
            let mut ring_view = self
                .ring
                .slice_mut(slot * self.k_conv * conv_dim..(slot + 1) * self.k_conv * conv_dim);
            cuda.stream()
                .memcpy_htod(&rz, &mut ring_view)
                .expect("slot ring reset");
            let mut st_view = self
                .state
                .slice_mut(slot * state_len..(slot + 1) * state_len);
            cuda.stream()
                .memcpy_htod(&sz, &mut st_view)
                .expect("slot state reset");
        }

        #[must_use]
        pub fn slots(&self) -> usize {
            self.slots
        }
    }

    /// One batched Mamba2 decode step for `m` agent slots: conv+silu, SSD recurrence
    /// (each slot's state advanced in place), gated group-RMSNorm — three launches
    /// TOTAL regardless of M, so the weight/constant reads are shared across agents.
    /// `zxbcdt` is `[m, d_inner+conv_dim+heads]` (the batched in_proj output).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn nh_mamba_decode_batch(
        zxbcdt: &Tensor<f32>,
        bufs: &mut NhMambaBatch,
        m: usize,
        ring_pos: &Tensor<f32>,
        d_inner: usize,
        conv_dim: usize,
        heads: usize,
        hd: usize,
        ns: usize,
        ngroups: usize,
        dt_min: f32,
        eps: f32,
        active: &Tensor<f32>,
    ) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let zc = zxbcdt.contiguous();
        let zx_stride = zc.shape()[zc.ndim() - 1];
        let z_g = zc.as_cuda_slice_read();
        let ac = active.contiguous();
        let a_g = ac.as_cuda_slice_read();
        let rc = ring_pos.contiguous();
        let r_g = rc.as_cuda_slice_read();
        let mut out = pool_alloc_uninit(m * d_inner).expect("nh batch out alloc");
        let k_conv = bufs.k_conv;
        {
            let func = subbit_func("nh_conv_silu_batch_f32").expect("nh_conv_silu_batch_f32");
            let cfg = LaunchConfig {
                grid_dim: ((conv_dim as u32).div_ceil(256), m as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_conv_silu_batch_f32` in kernels/vq_fused_matmul.ptx takes exactly these 12
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(z_g.slice())
                    .arg(&mut bufs.ring)
                    .arg(&bufs.conv_w)
                    .arg(&bufs.conv_b)
                    .arg(&mut bufs.xbc)
                    .arg(a_g.slice())
                    .arg(r_g.slice())
                    .arg(&(d_inner as i32))
                    .arg(&(conv_dim as i32))
                    .arg(&(k_conv as i32))
                    .arg(&(m as i32))
                    .arg(&(zx_stride as i32))
                    .launch(cfg)
                    .expect("nh_conv_silu_batch launch");
            }
        }
        {
            // float4 state when the dims allow it: same bytes, a quarter of the
            // memory instructions.
            // MEASURED NEUTRAL on sm_89 (92.9 vs 93.1 ms at M=16): after the layout fix
            // the scan is not instruction bound, so a quarter of the memory
            // instructions buys nothing. Opt-in, in case a part with lower
            // instruction throughput sees it differently.
            let v4 = hd % 4 == 0
                && heads % 4 == 0
                && matches!(std::env::var("PRG_SUBBIT_SSD_V4"), Ok(ref v) if v != "0");
            let func = subbit_func(if v4 {
                "nh_ssd_scan_batch_v4_f32"
            } else {
                "nh_ssd_scan_batch_f32"
            })
            .expect("nh_ssd_scan_batch");
            let cfg = if v4 {
                LaunchConfig {
                    grid_dim: ((heads / 4) as u32, m as u32, 1),
                    block_dim: ((hd / 4) as u32, 4, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                LaunchConfig {
                    grid_dim: (heads as u32, m as u32, 1),
                    block_dim: (hd as u32, 1, 1),
                    shared_mem_bytes: 0,
                }
            };
            // SAFETY: kernel `nh_conv_silu_batch_f32` / `nh_ssd_scan_batch_v4_f32` / `nh_ssd_scan_batch_f32` in kernels/vq_fused_matmul.ptx takes exactly these 17
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(z_g.slice())
                    .arg(&bufs.xbc)
                    .arg(&mut bufs.state)
                    .arg(&bufs.a_log)
                    .arg(&bufs.dvec)
                    .arg(&bufs.dt_bias)
                    .arg(&mut bufs.y)
                    .arg(a_g.slice())
                    .arg(&(d_inner as i32))
                    .arg(&(conv_dim as i32))
                    .arg(&(heads as i32))
                    .arg(&(hd as i32))
                    .arg(&(ns as i32))
                    .arg(&(ngroups as i32))
                    .arg(&dt_min)
                    .arg(&(m as i32))
                    .arg(&(zx_stride as i32))
                    .launch(cfg)
                    .expect("nh_ssd_scan_batch launch");
            }
        }
        {
            let func = subbit_func("nh_gated_gnorm_batch_f32").expect("nh_gated_gnorm_batch_f32");
            let cfg = LaunchConfig {
                grid_dim: (ngroups as u32, m as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_conv_silu_batch_f32` / `nh_ssd_scan_batch_v4_f32` / `nh_ssd_scan_batch_f32` / `nh_gated_gnorm_batch_f32` in kernels/vq_fused_matmul.ptx takes exactly these 10
            // arguments in this order (whichever this call selected), verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(&bufs.y)
                    .arg(z_g.slice())
                    .arg(&bufs.norm_w)
                    .arg(&mut out)
                    .arg(a_g.slice())
                    .arg(&(d_inner as i32))
                    .arg(&(ngroups as i32))
                    .arg(&eps)
                    .arg(&(m as i32))
                    .arg(&(zx_stride as i32))
                    .launch(cfg)
                    .expect("nh_gated_gnorm_batch launch");
            }
        }
        let shape = [m, d_inner];
        Tensor::from_storage(
            Storage::from_cuda_slice(out, m * d_inner, zc.device()),
            &shape,
        )
        .expect("nh batch mamba out")
    }

    /// Batched decode attention: M agents, each against its own KV in a slotted
    /// `[M, nkv, max_ctx, hd]` buffer, one launch. `q` is `[M, nh*hd]`; returns
    /// `[M, nh*hd]`. Byte-identical to the single-agent kernel at M=1.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn nh_decode_attn_batch(
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        mask: &Tensor<f32>,
        pos: &Tensor<f32>,
        m: usize,
        nh: usize,
        nkv: usize,
        max_ctx: usize,
        hd: usize,
        scale: f32,
    ) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let qc = q.contiguous();
        let kc = k.contiguous();
        let vc = v.contiguous();
        let mc = mask.contiguous();
        let qd = nh * hd;
        let mut out = pool_alloc_uninit(m * qd).expect("nh attn batch out alloc");
        {
            let q_g = qc.as_cuda_slice_read();
            let k_g = kc.as_cuda_slice_read();
            let v_g = vc.as_cuda_slice_read();
            let m_g = mc.as_cuda_slice_read();
            let pc2 = pos.contiguous();
            let p_g2 = pc2.as_cuda_slice_read();
            let func = subbit_func("nh_decode_attn_batch_f32").expect("nh_decode_attn_batch_f32");
            let threads = 256u32;
            let cfg = LaunchConfig {
                grid_dim: (nh as u32, m as u32, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: ((max_ctx + hd) * 4) as u32,
            };
            // SAFETY: kernel `nh_decode_attn_batch_f32` in kernels/vq_fused_matmul.ptx takes exactly these 12
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(q_g.slice())
                    .arg(k_g.slice())
                    .arg(v_g.slice())
                    .arg(m_g.slice())
                    .arg(p_g2.slice())
                    .arg(&mut out)
                    .arg(&(nh as i32))
                    .arg(&(nkv as i32))
                    .arg(&(max_ctx as i32))
                    .arg(&(hd as i32))
                    .arg(&scale)
                    .arg(&(m as i32))
                    .launch(cfg)
                    .expect("nh_decode_attn_batch launch");
            }
        }
        let shape = [m, qd];
        Tensor::from_storage(Storage::from_cuda_slice(out, m * qd, qc.device()), &shape)
            .expect("nh attn batch out")
    }

    /// Batched KV append at `pos` into a slotted `[M, nkv, max_ctx, hd]` buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn nh_kv_scatter_batch(
        src: &Tensor<f32>,
        dst: &Tensor<f32>,
        m: usize,
        nkv: usize,
        hd: usize,
        max_ctx: usize,
        pos: &Tensor<f32>,
        active: &Tensor<f32>,
    ) {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let sc = src.contiguous();
        let pc = pos.contiguous();
        let ac = active.contiguous();
        let width = nkv * hd;
        {
            let s_g = sc.as_cuda_slice_read();
            let p_g = pc.as_cuda_slice_read();
            let a_g = ac.as_cuda_slice_read();
            let mut d_guard = dst.as_cuda_slice_write();
            let func = subbit_func("nh_kv_scatter_batch_f32").expect("nh_kv_scatter_batch_f32");
            let cfg = LaunchConfig {
                grid_dim: ((width as u32).div_ceil(256), m as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_kv_scatter_batch_f32` in kernels/vq_fused_matmul.ptx takes exactly these 8
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(s_g.slice())
                    .arg(d_guard.slice_mut())
                    .arg(p_g.slice())
                    .arg(a_g.slice())
                    .arg(&(nkv as i32))
                    .arg(&(hd as i32))
                    .arg(&(max_ctx as i32))
                    .arg(&(m as i32))
                    .launch(cfg)
                    .expect("nh_kv_scatter_batch launch");
            }
        }
    }

    /// Repeat each row of `[m, n]` `rep` times → `[m*rep, n]`, so the grouped expert
    /// kernel (which indexes its input per GROUP) can serve M agents that each select
    /// `rep = top_k` experts.
    #[must_use]
    pub fn nh_row_repeat(x: &Tensor<f32>, rep: usize) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let xc = x.contiguous();
        let n = xc.shape()[xc.ndim() - 1];
        let m = xc.numel() / n;
        let rows = m * rep;
        let mut out = pool_alloc_uninit(rows * n).expect("nh row repeat alloc");
        {
            let x_g = xc.as_cuda_slice_read();
            let func = subbit_func("nh_row_repeat_f32").expect("nh_row_repeat_f32");
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(256), rows as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_row_repeat_f32` in kernels/vq_fused_matmul.ptx takes exactly these 5
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(x_g.slice())
                    .arg(&mut out)
                    .arg(&(n as i32))
                    .arg(&(rep as i32))
                    .arg(&(rows as i32))
                    .launch(cfg)
                    .expect("nh_row_repeat launch");
            }
        }
        let shape = [rows, n];
        Tensor::from_storage(Storage::from_cuda_slice(out, rows * n, xc.device()), &shape)
            .expect("nh row repeat out")
    }

    /// Batched MoE combine: `[m*top_k, h]` rows weighted by device `w` and summed
    /// within each agent's block → `[m, h]`.
    #[must_use]
    pub fn nh_moe_combine_batch(
        rows: &Tensor<f32>,
        w: &CudaSlice<f32>,
        m: usize,
        top_k: usize,
    ) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let rc = rows.contiguous();
        let h = rc.shape()[rc.ndim() - 1];
        let mut out = pool_alloc_uninit(m * h).expect("nh moe combine batch alloc");
        {
            let r_g = rc.as_cuda_slice_read();
            let func = subbit_func("nh_moe_combine_batch_f32").expect("nh_moe_combine_batch_f32");
            let cfg = LaunchConfig {
                grid_dim: ((h as u32).div_ceil(256), m as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_moe_combine_batch_f32` in kernels/vq_fused_matmul.ptx takes exactly these 6
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(r_g.slice())
                    .arg(w)
                    .arg(&mut out)
                    .arg(&(h as i32))
                    .arg(&(top_k as i32))
                    .arg(&(m as i32))
                    .launch(cfg)
                    .expect("nh_moe_combine_batch launch");
            }
        }
        let shape = [m, h];
        Tensor::from_storage(Storage::from_cuda_slice(out, m * h, rc.device()), &shape)
            .expect("nh moe combine batch out")
    }

    /// Row-wise RMSNorm over `[m, n]` (the generic `rms_norm` only handles a single
    /// row, since it requires weight-length == total length).
    #[must_use]
    pub fn nh_rmsnorm_rows(x: &Tensor<f32>, w: &Tensor<f32>, eps: f32) -> Tensor<f32> {
        let cuda = get_cuda_backend().expect("CUDA backend not available");
        let xc = x.contiguous();
        let wc = w.contiguous();
        let n = wc.numel();
        let m = xc.numel() / n;
        let mut out = pool_alloc_uninit(m * n).expect("nh rmsnorm rows alloc");
        {
            let x_g = xc.as_cuda_slice_read();
            let w_g = wc.as_cuda_slice_read();
            let func = subbit_func("nh_rmsnorm_rows_f32").expect("nh_rmsnorm_rows_f32");
            let cfg = LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: kernel `nh_rmsnorm_rows_f32` in kernels/vq_fused_matmul.ptx takes exactly these 6
            // arguments in this order, verified against the PTX signature by
            // tools/check_launches.py; the slice extents follow from the shapes derived above,
            // every buffer the kernel writes is passed as `&mut`, and the launch is ordered on
            // the backend's single stream.
            unsafe {
                cuda.stream()
                    .launch_builder(&func)
                    .arg(x_g.slice())
                    .arg(w_g.slice())
                    .arg(&mut out)
                    .arg(&(n as i32))
                    .arg(&eps)
                    .arg(&(m as i32))
                    .launch(cfg)
                    .expect("nh_rmsnorm_rows launch");
            }
        }
        let shape = [m, n];
        Tensor::from_storage(Storage::from_cuda_slice(out, m * n, xc.device()), &shape)
            .expect("nh rmsnorm rows out")
    }

    pub trait SubbitTensorExt: Sized {
        /// Device MoE router top-k: `self` = router logits `[rows, ne]` on GPU.
        /// Computes softmax over `ne` experts, selects top-`top_k`, optionally
        /// renormalizes — writing indices + weights to device buffers with NO host
        /// round-trip (the whole layer stays on the stream). Matches the host
        /// `topk_route` within the exp libm's last ULP.
        fn router_topk(&self, ne: usize, top_k: usize, norm: bool) -> RouterSel;
        /// Dense fp32 router GEMV on the decode path: `self` = normed hidden `[n, K]`,
        /// `w` = router weight `[K, N]` (row-major). Returns the router logits `[n, N]`
        /// via OUR kernel (no cuBLAS), so the MoE decode layer stays cuBLAS-free and the
        /// whole step is CUDA-graph-capturable. Argmax-matches the fp32 `matmul` (FNV).
        fn router_gemv(&self, w: &Self) -> Self;
        /// Fused single-position decode attention: `self` = q `[1, nh, 1, hd]` (contiguous
        /// → `[nh, hd]`), `k`/`v` = the fixed `[1, nkv, max_ctx, hd]` KV buffers, `mask` =
        /// the `[.., max_ctx]` additive causal mask. Returns the attention output as
        /// `[1, 1, nh*hd]` (ready for `o_proj`) computed by ONE kernel — q·Kᵀ + mask,
        /// softmax, ·V, GQA in-kernel — with no cuBLAS and no `repeat_kv`.
        #[allow(clippy::too_many_arguments)]
        fn decode_attn(
            &self,
            k: &Self,
            v: &Self,
            mask: &Self,
            nh: usize,
            nkv: usize,
            max_ctx: usize,
            hd: usize,
            scale: f32,
        ) -> Self;
        /// Device-`w` twin of [`vq_moe_combine`](Self::vq_moe_combine): the router
        /// weights already live on-device (from [`Tensor::router_topk`]), so no
        /// weight H2D. `self` = `[g, h]`; returns `[1, h] = Σ_g self[g,:]·w[g]`.
        fn vq_moe_combine_dev(&self, w: &cudarc::driver::CudaSlice<f32>, g: usize) -> Self;
        /// Fused MoE top-k combine: `self` is `[G, h]` (the top-k experts' `down`
        /// outputs in route order) and `w` the `[G]` router weights (route order).
        /// Returns `[1, h] = Σ_g self[g,:]·w[g]`, accumulated in g-order from zero —
        /// byte-identical to the sequential `acc += w[g]·down[g]` (`scaled_add_inplace_`)
        /// combine, but in ONE kernel launch instead of `G`.
        fn vq_moe_combine(&self, w: &[f32]) -> Self;
        /// Weight-offload VQ normalize: `self` is the `[n_vec, dim]` weight tile;
        /// returns `[n_vec, dim]` scaled by the per-row inverse scale, uploading only
        /// the `[n_rows]` row scales instead of a full `[numel]` inv_scale buffer.
        /// Bit-exact to `self * inv_scale` (`inv = 1/scale` via `div.rn`, `mul.f32`).
        fn vq_normalize_cuda(
            &self,
            row_scale: &[f32],
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Self;
        /// Fully self-contained fused VQ nearest-code assignment on-device: `self` is
        /// the `[n_vec, dim]` normalized weight, `codebook` is `[k, dim]`. Returns the
        /// `[n_vec]` nearest-code indices `argmax_c(v·c − ½‖c‖²)` in ONE of OUR
        /// kernels — no `[n_vec, k]` scores matrix, no cuBLAS, no external `‖c‖²`.
        fn vq_assign_cuda(&self, codebook: &Self, n_vec: usize, dim: usize, k: usize) -> Vec<u32>;
        /// Fused VQ matmul forward (register-blocked): `self` = x `[n,in_f]`; returns
        /// `[n,out_f] = x·recon^T` reconstructed on-device from the resident packed
        /// weight `(idx, codebook, scale)` — no dense weight materialized.
        fn vq_fused_fwd(
            &self,
            idx: &[u32],
            codebook: &Self,
            scale: &[f32],
            out_f: usize,
            dim: usize,
        ) -> Self;
        /// Fused VQ matmul forward off ALREADY-RESIDENT indices + scale: `self` = x
        /// `[n,in_f]`; returns `[n,out_f] = x·recon^T` reconstructed on-device from the
        /// packed weight `(idx, codebook, scale)` — same math as `vq_fused_fwd` but
        /// the `idx`/`scale` device buffers come from `resident` (uploaded ONCE via
        /// [`ResidentAssign::upload`] as a single tile: `resident.assign[0]` is the full
        /// `[out_f*vpr]` index array, `resident.scale[0]` the `[out_f]` row scale). No
        /// per-call H2D of the index array — the decode hot path re-uploaded ~index_bpw
        /// bits/weight of indices every token; this eliminates that. Inference-only:
        /// mirrors the non-TF32 (bit-exact-to-CPU) branch of `vq_fused_fwd`.
        fn vq_fused_fwd_resident(
            &self,
            resident: &ResidentAssign,
            codebook: &Self,
            out_f: usize,
            dim: usize,
        ) -> Self;
        /// Device-resident `cat` (NO host round-trip): concatenate GPU tensors along
        /// `dim` with device-to-device copies. Bit-identical to [`Tensor::cat`] — it is
        /// pure `memcpy` (no arithmetic), just assembled on-device. Each input's rows
        /// for a fixed outer index are contiguous in both source and destination, so it
        /// costs one `dtod` copy per (input, outer) — e.g. the decode KV append
        /// (`[1,nkv,seq,hd]` cat on dim=2) is `2*nkv` copies, keeping the growing K/V
        /// resident instead of the D2H+concat+H2D the generic host `cat` did per step.
        fn cat_gpu(tensors: &[&Self], dim: usize) -> Self;
        /// Fused VQ matmul backward: `self` = grad_out `[n,out_f]`; returns
        /// `(grad_x [n,in_f], grad_codebook [k,dim])` via the register-blocked kernels.
        fn vq_fused_bwd(
            &self,
            x: &Self,
            codebook: &Self,
            idx: &[u32],
            scale: &[f32],
            in_f: usize,
            dim: usize,
            k: usize,
        ) -> (Self, Self);
        /// Weight-offload VQ gather+denormalize: `self` is the `[k, dim]` codebook;
        /// returns the `[n_vec, dim]` reconstruction gathered by `assign` and scaled
        /// per row, uploading only `[n_vec]` assignments + `[n_rows]` scales instead
        /// of a full `[numel]` gather-index buffer. Bit-exact to
        /// `gather(codebook, expand(assign)).div(inv_scale)`.
        fn vq_gather_denorm_cuda(
            &self,
            assign: &[u32],
            row_scale: &[f32],
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Self;
        /// GPU-resident VQ gather+denorm: `self` is the `[k, dim]` codebook; reads the
        /// pre-uploaded assign + scale for `tile` from `resident` — NO per-tile H2D,
        /// NO sync (stream-ordered kernel only → CUDA-graph capturable). Bit-exact to
        /// `vq_gather_denorm_cuda`.
        fn vq_gather_denorm_resident(
            &self,
            resident: &ResidentAssign,
            tile: usize,
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Self;
        /// Device-to-device write of `self`'s contiguous data into `dst`'s storage
        /// starting at element `dst_elem_offset`, stream-ordered (no host sync).
        /// Used to assemble a tiled output on-device without a per-tile `to_vec`
        /// stall. `self` must be contiguous; `dst` must have room for `self.numel()`
        /// elements at the offset.
        fn dtod_write_at(&self, dst: &Self, dst_elem_offset: usize);
        /// Scatter this `[1, nkv, t, hd]` K or V tensor into a fixed
        /// `[1, nkv, max_ctx, hd]` cache buffer `dst` at sequence position `pos`,
        /// in ONE stream-ordered 2D copy (vs one copy per head). `self` is made
        /// contiguous first; each head's `t·hd` block lands at
        /// `head·max_ctx·hd + pos·hd`. Byte-identical to a per-head write loop, but
        /// captures as a single memcpy graph node.
        fn scatter_heads_at(
            &self,
            dst: &Self,
            pos: usize,
            nkv: usize,
            t: usize,
            hd: usize,
            max_ctx: usize,
        );
        /// Device-pos twin of `scatter_heads_at`: write `self` `[1, nkv, t, hd]`
        /// into `dst` `[1, nkv, max_ctx, hd]` at the position held in the device f32
        /// scalar `pos` (element 0, an integer value). A kernel (not a memcpy node),
        /// so a captured graph advances the KV correctly across decode steps.
        /// Byte-identical to the memcpy scatter for the same pos.
        fn scatter_heads_at_devpos(
            &self,
            dst: &Self,
            pos: &Self,
            nkv: usize,
            t: usize,
            hd: usize,
            max_ctx: usize,
        );
        /// Device-pos split-halves RoPE: rotate this `[1, n_heads, seq, head_dim]`
        /// tensor using the starting position in the device f32 scalar `pos` (element
        /// 0, an integer value) and `theta`. Returns the rotated tensor. Bit-identical
        /// to `apply_rope_split_halves_bhsd(.., offset=pos)` (same SFU approx math),
        /// but pos lives in a device buffer the graph reads, so one capture replays at
        /// every step.
        fn apply_rope_devpos(
            &self,
            pos: &Self,
            n_heads: usize,
            seq: usize,
            head_dim: usize,
            theta: f32,
        ) -> Self;
    }

    impl SubbitTensorExt for Tensor<f32> {
        /// Device MoE router top-k: `self` = router logits `[rows, ne]` on GPU.
        /// Computes softmax over `ne` experts, selects top-`top_k`, optionally
        /// renormalizes — writing indices + weights to device buffers with NO host
        /// round-trip (the whole layer stays on the stream). Matches the host
        /// `topk_route` within the exp libm's last ULP.
        fn router_topk(&self, ne: usize, top_k: usize, norm: bool) -> RouterSel {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let lc = self.contiguous();
            let rows = lc.numel() / ne;
            let n = rows * top_k;
            // Device output buffers; the kernel writes all `n` entries, so uninit is
            // fine. Pool-sourced (not a raw `stream().alloc`) so that under CUDA-graph
            // capture they resolve to pre-warmed pool hits — a pointer reconstruct, no
            // stream-ordered MemAlloc/MemFree graph node (those crash graph replay on
            // WSL). `RouterSel::drop` returns them to the pool / capture pen.
            let mut sel = pool_alloc_uninit_i32(n).expect("router sel pool alloc");
            let mut w = pool_alloc_uninit(n).expect("router w pool alloc");
            {
                let logits_g = lc.as_cuda_slice_read();
                cuda.vq_router_topk_f32(logits_g.slice(), &mut sel, &mut w, rows, ne, top_k, norm)
                    .expect("vq_router_topk_f32");
            }
            RouterSel {
                sel: Some(sel),
                w: Some(w),
            }
        }

        /// Dense fp32 router GEMV: `self` = normed hidden `[n, K]`, `w` = `[K, N]`
        /// (router weight, row-major). Returns `[n, N]` router logits via OUR kernel —
        /// no cuBLAS, capture-safe. Ascending-`k` f32 accumulation (FNV-gated).
        fn router_gemv(&self, w: &Self) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let xc = self.contiguous();
            let wc = w.contiguous();
            let n = xc.shape()[0];
            let kdim = xc.shape()[1];
            let n_out = wc.shape()[1];
            debug_assert_eq!(wc.shape()[0], kdim, "router_gemv: K mismatch");
            let mut out = pool_alloc_uninit(n * n_out).expect("router_gemv out pool alloc");
            {
                let x_g = xc.as_cuda_slice_read();
                let w_g = wc.as_cuda_slice_read();
                cuda.vq_router_gemv_f32(x_g.slice(), w_g.slice(), &mut out, n, kdim, n_out)
                    .expect("vq_router_gemv_f32");
            }
            let shape = [n, n_out];
            Tensor::from_storage(
                Storage::from_cuda_slice(out, n * n_out, xc.device()),
                &shape,
            )
            .expect("subbit from_storage")
        }

        /// Fused single-position decode attention over the fixed KV buffers with the
        /// additive device mask. `self` = q (contiguous `[nh, hd]`), `k`/`v` =
        /// `[1, nkv, max_ctx, hd]`, `mask` = `[.., max_ctx]`. Returns `[1, 1, nh*hd]`.
        fn decode_attn(
            &self,
            k: &Self,
            v: &Self,
            mask: &Self,
            nh: usize,
            nkv: usize,
            max_ctx: usize,
            hd: usize,
            scale: f32,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let qc = self.contiguous();
            let kc = k.contiguous();
            let vc = v.contiguous();
            let mc = mask.contiguous();
            let qd = nh * hd;
            let mut out = pool_alloc_uninit(qd).expect("decode_attn out pool alloc");
            {
                let q_g = qc.as_cuda_slice_read();
                let k_g = kc.as_cuda_slice_read();
                let v_g = vc.as_cuda_slice_read();
                let m_g = mc.as_cuda_slice_read();
                cuda.vq_decode_attn_f32(
                    q_g.slice(),
                    k_g.slice(),
                    v_g.slice(),
                    m_g.slice(),
                    &mut out,
                    nh,
                    nkv,
                    max_ctx,
                    hd,
                    scale,
                )
                .expect("vq_decode_attn_f32");
            }
            let shape = [1usize, 1usize, qd];
            Tensor::from_storage(Storage::from_cuda_slice(out, qd, qc.device()), &shape)
                .expect("subbit from_storage")
        }

        /// Device-`w` twin of [`vq_moe_combine`](Self::vq_moe_combine): the router
        /// weights already live on-device (from [`Tensor::router_topk`]), so no
        /// weight H2D. `self` = `[g, h]`; returns `[1, h] = Σ_g self[g,:]·w[g]`.
        fn vq_moe_combine_dev(&self, w: &cudarc::driver::CudaSlice<f32>, g: usize) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let rows = self.contiguous();
            let h = rows.shape()[1];
            let mut out = pool_alloc_uninit(h).expect("moe combine_dev out pool alloc");
            {
                let rows_g = rows.as_cuda_slice_read();
                cuda.vq_moe_combine_f32(rows_g.slice(), w, &mut out, h, g)
                    .expect("vq_moe_combine_f32 (dev w)");
            }
            let shape = [1usize, h];
            Tensor::from_storage(Storage::from_cuda_slice(out, h, rows.device()), &shape)
                .expect("subbit from_storage")
        }

        /// Fused MoE top-k combine: `self` is `[G, h]` (the top-k experts' `down`
        /// outputs in route order) and `w` the `[G]` router weights (route order).
        /// Returns `[1, h] = Σ_g self[g,:]·w[g]`, accumulated in g-order from zero —
        /// byte-identical to the sequential `acc += w[g]·down[g]` (`scaled_add_inplace_`)
        /// combine, but in ONE kernel launch instead of `G`.
        fn vq_moe_combine(&self, w: &[f32]) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let rows = self.contiguous();
            let g = rows.shape()[0];
            let h = rows.shape()[1];
            debug_assert_eq!(w.len(), g, "vq_moe_combine: weight count != rows");
            let w_g = cuda.htod_copy(w).expect("moe combine w htod");
            let mut out = pool_alloc_uninit(h).expect("moe combine out pool alloc");
            {
                let rows_g = rows.as_cuda_slice_read();
                cuda.vq_moe_combine_f32(rows_g.slice(), &w_g, &mut out, h, g)
                    .expect("vq_moe_combine_f32");
            }
            let shape = [1usize, h];
            Tensor::from_storage(Storage::from_cuda_slice(out, h, rows.device()), &shape)
                .expect("subbit from_storage")
        }

        /// Weight-offload VQ normalize: `self` is the `[n_vec, dim]` weight tile;
        /// returns `[n_vec, dim]` scaled by the per-row inverse scale, uploading only
        /// the `[n_rows]` row scales instead of a full `[numel]` inv_scale buffer.
        /// Bit-exact to `self * inv_scale` (`inv = 1/scale` via `div.rn`, `mul.f32`).
        fn vq_normalize_cuda(
            &self,
            row_scale: &[f32],
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let total = n_vec * dim;
            let mut scale_gpu = pool_alloc_uninit(row_scale.len()).expect("pool alloc row_scale");
            cuda.htod_into(row_scale, &mut scale_gpu)
                .expect("htod row_scale failed");
            let w_guard = self.as_cuda_slice_read();
            let mut out = pool_alloc_uninit(total).expect("pool alloc vq_normalize out");
            cuda.vq_normalize_f32(
                &mut out,
                w_guard.slice(),
                &scale_gpu,
                n_vec,
                dim,
                vec_per_row,
            )
            .expect("vq_normalize_f32 failed");
            axonml_core::backends::cuda_pool::pool_free(scale_gpu);
            let shape = [n_vec, dim];
            let storage = Storage::from_cuda_slice(out, total, self.device());
            Tensor::from_storage(storage, &shape).expect("subbit from_storage")
        }

        /// Fully self-contained fused VQ nearest-code assignment on-device: `self` is
        /// the `[n_vec, dim]` normalized weight, `codebook` is `[k, dim]`. Returns the
        /// `[n_vec]` nearest-code indices `argmax_c(v·c − ½‖c‖²)` in ONE of OUR
        /// kernels — no `[n_vec, k]` scores matrix, no cuBLAS, no external `‖c‖²`.
        fn vq_assign_cuda(&self, codebook: &Self, n_vec: usize, dim: usize, k: usize) -> Vec<u32> {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let mut out = pool_alloc_uninit(n_vec).expect("pool alloc vq_assign out");
            {
                let normed_g = self.as_cuda_slice_read();
                let cb_g = codebook.as_cuda_slice_read();
                cuda.vq_assign_f32(normed_g.slice(), cb_g.slice(), &mut out, n_vec, dim, k)
                    .expect("vq_assign_f32 failed");
            }
            // Copy back exactly n_vec (the pool may over-allocate `out`).
            let mut idx_f32 = vec![0f32; n_vec];
            cuda.dtoh_into_n(&out, n_vec, &mut idx_f32)
                .expect("vq_assign d2h");
            axonml_core::backends::cuda_pool::pool_free(out);
            idx_f32.iter().map(|&f| f as u32).collect()
        }

        /// Fused VQ matmul forward (register-blocked): `self` = x `[n,in_f]`; returns
        /// `[n,out_f] = x·recon^T` reconstructed on-device from the resident packed
        /// weight `(idx, codebook, scale)` — no dense weight materialized.
        #[cfg(feature = "cuda")]
        fn vq_fused_fwd(
            &self,
            idx: &[u32],
            codebook: &Self,
            scale: &[f32],
            out_f: usize,
            dim: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let xc = self.contiguous();
            let cbc = codebook.contiguous();
            let n = xc.shape()[0];
            let in_f = xc.shape()[1];
            let vpr = in_f / dim;
            let idx_g = cuda.htod_copy(idx).expect("htod idx");
            let sc_g = cuda.htod_copy(scale).expect("htod scale");
            // NVFP4 forward (AXONML_FP4=1, Blackwell): quantize x once + each recon tile to
            // block-scaled fp4 on-GPU and run the projection GEMM on the FP4 tensor cores
            // (~100 TOPS real-scale vs cuBLAS-tf32 ~26). Same tiled-reconstruct structure as
            // the tf32 path below but the GEMM lands straight into out[n,out_f] at a column
            // offset (no transposed assembly, no final transpose). Shape contract (n%128,
            // in_f%256, out_f%64) — falls through to tf32/fp32 when unmet. Forward-only:
            // backward stays on the existing exact path (STE tolerates fp4 forward noise;
            // that is the QAT premise).
            if std::env::var("AXONML_FP4").is_ok()
                && n % 128 == 0
                && in_f % 256 == 0
                && out_f % 64 == 0
            {
                let mut xq = pool_alloc_uninit_u32(n * in_f / 8).expect("fp4 pool xq");
                let mut xs = pool_alloc_uninit_u32(n * in_f / 64).expect("fp4 pool xs");
                {
                    let x_g = xc.as_cuda_slice_read();
                    cuda.nvfp4_quant_f32(x_g.slice(), &mut xq, &mut xs, n * in_f / 16)
                        .expect("nvfp4 quant x");
                }
                let mut out = pool_alloc_uninit(n * out_f).expect("fp4 pool out");
                cuda.memset_zeros_f32(&mut out).expect("fp4 zero out");
                let ot_max = ((((1usize << 30) / 4 / in_f.max(1)).max(64)) / 64) * 64;
                let mut o0 = 0;
                while o0 < out_f {
                    let ot = ot_max.min(out_f - o0);
                    let mut recon = pool_alloc_uninit(ot * in_f).expect("pool alloc recon tile");
                    {
                        let cb_g = cbc.as_cuda_slice_read();
                        cuda.vq_reconstruct_f32(
                            &idx_g,
                            cb_g.slice(),
                            &sc_g,
                            &mut recon,
                            ot,
                            in_f,
                            dim,
                            vpr,
                            o0,
                        )
                        .expect("vq_reconstruct_f32 tile");
                    }
                    let mut wq = pool_alloc_uninit_u32(ot * in_f / 8).expect("fp4 pool wq");
                    let mut ws = pool_alloc_uninit_u32(ot * in_f / 64).expect("fp4 pool ws");
                    cuda.nvfp4_quant_f32(&recon, &mut wq, &mut ws, ot * in_f / 16)
                        .expect("nvfp4 quant recon tile");
                    cuda.nvfp4_gemm_sf2(&xq, &wq, &xs, &ws, &mut out, n, ot, in_f, out_f, o0)
                        .expect("nvfp4_gemm_sf2 tile");
                    cuda.sync(); // GEMM done -> tile buffers recyclable next iter
                    pool_free(recon);
                    pool_free_u32(wq);
                    pool_free_u32(ws);
                    o0 += ot;
                }
                pool_free_u32(xq);
                pool_free_u32(xs);
                let shape = [n, out_f];
                return Tensor::from_storage(
                    Storage::from_cuda_slice(out, n * out_f, self.device()),
                    &shape,
                )
                .expect("subbit from_storage");
            }
            // TILED reconstruct + cuBLAS(tf32): assemble out^T[out_f, n] one row-tile at
            // a time off the resident packed weight, so a wide layer (mlp recon ~4 GB)
            // never materializes its full dense recon -> tensor-core GEMM, no OOM.
            // Under AXONML_TF32 cuBLAS runs the GEMMs on tf32 tensor cores (~1.7x fp32).
            if std::env::var("AXONML_TF32").is_ok() {
                let x_t = xc.transpose(0, 1).expect("x transpose"); // [in_f, n] view
                let ot_shape = [out_f, n];
                let out_t = Tensor::from_storage(
                    Storage::from_cuda_slice(
                        pool_alloc_uninit(out_f * n).expect("pool alloc out_t"),
                        out_f * n,
                        self.device(),
                    ),
                    &ot_shape,
                )
                .expect("subbit from_storage");
                let ot_max = ((1usize << 30) / 4 / in_f.max(1)).max(64); // recon tile <= ~1 GB
                let mut o0 = 0;
                while o0 < out_f {
                    let ot = ot_max.min(out_f - o0);
                    let mut recon = pool_alloc_uninit(ot * in_f).expect("pool alloc recon tile");
                    {
                        let cb_g = cbc.as_cuda_slice_read();
                        cuda.vq_reconstruct_f32(
                            &idx_g,
                            cb_g.slice(),
                            &sc_g,
                            &mut recon,
                            ot,
                            in_f,
                            dim,
                            vpr,
                            o0,
                        )
                        .expect("vq_reconstruct_f32 tile");
                    }
                    cuda.sync();
                    let rshape = [ot, in_f];
                    let recon_tile = Tensor::from_storage(
                        Storage::from_cuda_slice(recon, ot * in_f, self.device()),
                        &rshape,
                    )
                    .expect("subbit from_storage");
                    // out_t[o0..o0+ot, :] = recon_tile[ot,in_f] @ x^T[in_f,n]
                    let _ = recon_tile
                        .matmul_into_at(&x_t, &out_t, o0, 0.0)
                        .expect("matmul_into_at recon tile");
                    cuda.sync(); // GEMM done -> recon tile buffer recyclable next iter
                    o0 += ot;
                }
                return out_t
                    .transpose(0, 1)
                    .expect("out_t transpose")
                    .contiguous_gpu();
            }
            let mut out = pool_alloc_uninit(n * out_f).expect("pool alloc fused out");
            {
                let x_g = xc.as_cuda_slice_read();
                let cb_g = cbc.as_cuda_slice_read();
                // The register-blocked rb8 kernel hoists ONE codebook index per BK=16
                // k-tile (`code_s[o] = idx[o*vpr + k0/dim]`, then reads `cb[code*dim + kk]`
                // for kk in 0..16), so it is only correct when dim == 16 (BK == dim). For
                // any other vq dim (e.g. the dim=8 sub-bit packs) fall back to the tiled
                // kernel, which regathers the code per column (`idx[o*vpr + k/dim]`,
                // `cb[code*dim + k%dim]`) and is correct for all dims.
                let k_codes = cbc.numel() / dim; // codebook is [k, dim]
                if dim == 16 {
                    cuda.vq_fused_matmul_rb8_f32(
                        x_g.slice(),
                        &idx_g,
                        cb_g.slice(),
                        &sc_g,
                        &mut out,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                    )
                    .expect("vq_fused_matmul_rb8_f32");
                } else if n <= vq_warp_rows_max() {
                    // Small-batch decode (multi-agent): warp-per-output-column GEMV, one
                    // grid.y row per sequence — 32 in-flight gather streams/column and no
                    // per-k-tile sync, instead of the 16x16 tile that wastes 15/16 threads
                    // at these row counts. The kernel always indexed rows by blockIdx.y;
                    // it was merely GATED to n==1, which put M>1 agent batches on the slow
                    // tile path (measured: 18.2 ms/step at M=1 -> 67.4 at M=2).
                    // Argmax-identical (FNV-gated). Threshold: AXONML_VQ_WARP_ROWS.
                    cuda.vq_fused_matmul_tiled_warp_f32(
                        x_g.slice(),
                        &idx_g,
                        cb_g.slice(),
                        &sc_g,
                        &mut out,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                        k_codes,
                    )
                    .expect("vq_fused_matmul_tiled_warp_f32");
                } else {
                    cuda.vq_fused_matmul_tiled_f32(
                        x_g.slice(),
                        &idx_g,
                        cb_g.slice(),
                        &sc_g,
                        &mut out,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                        k_codes,
                    )
                    .expect("vq_fused_matmul_tiled_f32");
                }
            }
            // NO host sync here. `out` stays on-device, ordered on the default stream
            // after the fused kernel above; the consumer op (next matmul / elementwise /
            // the eventual logits D2H) runs after it via single-stream FIFO ordering with
            // no host round-trip. A `cuda.sync()` would only block the host early and
            // starve the GPU — the host only needs to sync before it READS device data,
            // which the D2H copies (router logits / final logits) do on their own.
            let shape = [n, out_f];
            Tensor::from_storage(
                Storage::from_cuda_slice(out, n * out_f, self.device()),
                &shape,
            )
            .expect("subbit from_storage")
        }

        /// Fused VQ matmul forward off ALREADY-RESIDENT indices + scale: `self` = x
        /// `[n,in_f]`; returns `[n,out_f] = x·recon^T` reconstructed on-device from the
        /// packed weight `(idx, codebook, scale)` — same math as `vq_fused_fwd` but
        /// the `idx`/`scale` device buffers come from `resident` (uploaded ONCE via
        /// [`ResidentAssign::upload`] as a single tile: `resident.assign[0]` is the full
        /// `[out_f*vpr]` index array, `resident.scale[0]` the `[out_f]` row scale). No
        /// per-call H2D of the index array — the decode hot path re-uploaded ~index_bpw
        /// bits/weight of indices every token; this eliminates that. Inference-only:
        /// mirrors the non-TF32 (bit-exact-to-CPU) branch of `vq_fused_fwd`.
        #[cfg(feature = "cuda")]
        fn vq_fused_fwd_resident(
            &self,
            resident: &ResidentAssign,
            codebook: &Self,
            out_f: usize,
            dim: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let xc = self.contiguous();
            let cbc = codebook.contiguous();
            let n = xc.shape()[0];
            let in_f = xc.shape()[1];
            let vpr = in_f / dim;
            // Resident device buffers — uploaded once, reused every decode step.
            let idx_g = &resident.assign[0];
            let sc_g = &resident.scale[0];
            let mut out = pool_alloc_uninit(n * out_f).expect("pool alloc fused out");
            {
                let x_g = xc.as_cuda_slice_read();
                let cb_g = cbc.as_cuda_slice_read();
                // Same kernel selection as `vq_fused_fwd`: rb8 only when dim==16 (BK==dim),
                // otherwise the tiled kernel (correct for all dims, e.g. dim=8 sub-bit).
                let k_codes = cbc.numel() / dim; // codebook is [k, dim]
                if dim == 16 {
                    cuda.vq_fused_matmul_rb8_f32(
                        x_g.slice(),
                        idx_g,
                        cb_g.slice(),
                        sc_g,
                        &mut out,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                    )
                    .expect("vq_fused_matmul_rb8_f32 resident");
                } else if n <= vq_warp_rows_max() {
                    // Decode hot path: warp-per-output-column GEMV (batch-1), see the
                    // matching branch in `vq_fused_fwd`. Argmax-identical (FNV-gated).
                    // MEASURED: still the best at these row counts — row+column blocking
                    // below wastes three of its four row slots here (M=1: 16.3 vs 17.2 ms).
                    cuda.vq_fused_matmul_tiled_warp_f32(
                        x_g.slice(),
                        idx_g,
                        cb_g.slice(),
                        sc_g,
                        &mut out,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                        k_codes,
                    )
                    .expect("vq_fused_matmul_tiled_warp_f32 resident");
                } else if dim == 8 && subbit_rc() {
                    // Above the warp threshold the alternative is the 16x16 tile kernel,
                    // which re-gathers the packed code for every (column, k) — eight times
                    // over at dim=8. Row+column blocking reads it once per eight k and
                    // shares one x load across four columns.
                    cuda.vq_fused_matmul_warp_rc_f32(
                        x_g.slice(),
                        idx_g,
                        cb_g.slice(),
                        sc_g,
                        &mut out,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                        k_codes,
                    )
                    .expect("vq_fused_matmul_warp_rc_f32 resident");
                } else {
                    cuda.vq_fused_matmul_tiled_f32(
                        x_g.slice(),
                        idx_g,
                        cb_g.slice(),
                        sc_g,
                        &mut out,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                        k_codes,
                    )
                    .expect("vq_fused_matmul_tiled_f32 resident");
                }
            }
            // NO host sync here — this is THE decode hot path (default, resident idx).
            // `out` is left on-device, ordered on the default stream after the kernel;
            // downstream device ops consume it in stream order without a host round-trip.
            // The idx/scale buffers are resident (not freed here) and `out` is owned by
            // the returned tensor, so nothing is reclaimed before the kernel runs. The
            // host only syncs where it actually reads device data (router/final logits
            // D2H), which is self-synchronizing. Batching the syncs this way lets the
            // whole token pipeline on one stream instead of 0-1% util stop-and-go.
            let shape = [n, out_f];
            Tensor::from_storage(
                Storage::from_cuda_slice(out, n * out_f, self.device()),
                &shape,
            )
            .expect("subbit from_storage")
        }

        /// Device-resident `cat` (NO host round-trip): concatenate GPU tensors along
        /// `dim` with device-to-device copies. Bit-identical to [`Tensor::cat`] — it is
        /// pure `memcpy` (no arithmetic), just assembled on-device. Each input's rows
        /// for a fixed outer index are contiguous in both source and destination, so it
        /// costs one `dtod` copy per (input, outer) — e.g. the decode KV append
        /// (`[1,nkv,seq,hd]` cat on dim=2) is `2*nkv` copies, keeping the growing K/V
        /// resident instead of the D2H+concat+H2D the generic host `cat` did per step.
        #[cfg(feature = "cuda")]
        fn cat_gpu(tensors: &[&Self], dim: usize) -> Self {
            assert!(!tensors.is_empty(), "cat_gpu requires at least one tensor");
            let ndim = tensors[0].ndim();
            assert!(dim < ndim, "cat_gpu dim out of range");
            // Own a contiguous device copy of each input so the copy loop can read a
            // flat, offset-addressable buffer (mirrors host `cat`'s `contiguous()`).
            let conts: Vec<Self> = tensors.iter().map(|t| t.contiguous_gpu()).collect();
            let total_dim: usize = conts.iter().map(|t| t.shape()[dim]).sum();
            let mut out_shape: Vec<usize> = conts[0].shape().to_vec();
            out_shape[dim] = total_dim;
            let outer_size: usize = out_shape[..dim].iter().product();
            let inner_size: usize = out_shape[dim + 1..].iter().product();
            let total_numel: usize = out_shape.iter().product();
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let mut out = pool_alloc_uninit(total_numel).expect("pool alloc cat_gpu out");
            let mut dim_offset = 0usize;
            for t in &conts {
                let tdim = t.shape()[dim];
                let block = tdim * inner_size; // contiguous run for one outer index
                let guard = t.as_cuda_slice_read();
                let src = guard.slice();
                for outer in 0..outer_size {
                    let src_off = outer * block;
                    let dst_off = outer * total_dim * inner_size + dim_offset * inner_size;
                    cuda.memcpy_dtod_async_f32(&mut out, dst_off, src, src_off, block)
                        .expect("cat_gpu dtod copy");
                }
                dim_offset += tdim;
            }
            // NO host sync here. The dtod copies above are enqueued on the default
            // stream and the returned tensor's `out` buffer is consumed by later device
            // ops in stream order; the `conts` scratch is pool-freed stream-ordered on
            // drop. On the decode path this runs twice per layer (KV K/V append) — a
            // per-call sync would drain the pipeline every layer for no correctness
            // reason (the growing K/V is only ever read back on-device by the attention
            // matmuls, never on the host mid-token).
            let shape = axonml_tensor::shape::Shape::from_slice(&out_shape);
            let _strides = contiguous_strides(&shape);
            Tensor::from_storage(
                Storage::from_cuda_slice(out, total_numel, conts[0].device()),
                &out_shape,
            )
            .expect("cat_gpu from_storage")
        }

        /// Fused VQ matmul backward: `self` = grad_out `[n,out_f]`; returns
        /// `(grad_x [n,in_f], grad_codebook [k,dim])` via the register-blocked kernels.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        fn vq_fused_bwd(
            &self,
            x: &Self,
            codebook: &Self,
            idx: &[u32],
            scale: &[f32],
            in_f: usize,
            dim: usize,
            k: usize,
        ) -> (Self, Self) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let gc = self.contiguous();
            let xc = x.contiguous();
            let cbc = codebook.contiguous();
            let n = gc.shape()[0];
            let out_f = gc.shape()[1];
            let vpr = in_f / dim;
            let idx_g = cuda.htod_copy(idx).expect("htod idx");
            let sc_g = cuda.htod_copy(scale).expect("htod scale");
            let mut dcb = cuda.htod_copy(&vec![0f32; k * dim]).expect("alloc grad_cb");
            let sx = [n, in_f];
            let scb = [k, dim];
            // dx: under AXONML_TF32 use the same tiled-reconstruct + cuBLAS-tf32 fast path
            // the FORWARD uses (~5x the rb kernel), so the backward's biggest GEMM
            // (dx = g @ W) runs on tensor cores. dcb keeps its fused scatter kernel.
            // dx[n,in_f] = g[n,out_f] @ W[out_f,in_f]; tile over out_f (the contraction):
            //   gT = g^T[out_f,n] contiguous; gT_tile = rows[o0..o0+ot] (contiguous view);
            //   W_tile = reconstruct[ot,in_f]; dx += gT_tile^T[n,ot] @ W_tile[ot,in_f].
            //
            // NVFP4 backward (AXONML_FP4_BWD=1, Blackwell): BOTH backward GEMMs on the FP4
            // tensor cores with transpose-quantized operands — dx += q(g-slice)·q(reconT)^T
            // (contract over out_f tiles; the atomicAdd epilogue accumulates across tiles),
            // dW_tile = q(gT-rows)·q(xT)^T (contract over n) feeding the unchanged codebook
            // scatter. Gradients pass through nvfp4 (per-16 e2m1+ue4m3) — a QAT-regime
            // experiment gated separately from the exact paths; falls through when the
            // shape contract (n%256, out_f%256, in_f%64) is unmet.
            let fp4_bwd = std::env::var("AXONML_FP4_BWD").is_ok()
                && n % 256 == 0
                && out_f % 256 == 0
                && in_f % 64 == 0;
            let grad_x = if fp4_bwd {
                let gt = gc.transpose(0, 1).expect("g transpose").contiguous_gpu(); // [out_f,n]
                let xt = xc.transpose(0, 1).expect("x transpose").contiguous_gpu(); // [in_f,n]
                let mut xtq = pool_alloc_uninit_u32(in_f * n / 8).expect("fp4 pool xtq");
                let mut xts = pool_alloc_uninit_u32(in_f * n / 64).expect("fp4 pool xts");
                {
                    let g = xt.as_cuda_slice_read();
                    cuda.nvfp4_quant_f32(g.slice(), &mut xtq, &mut xts, in_f * n / 16)
                        .expect("nvfp4 quant xT");
                }
                let mut dx = pool_alloc_uninit(n * in_f).expect("fp4 pool dx");
                cuda.memset_zeros_f32(&mut dx).expect("fp4 zero dx");
                let ot_max = ((((1usize << 30) / 4 / in_f.max(1)).max(256)) / 256) * 256;
                let mut o0 = 0;
                while o0 < out_f {
                    let ot = ot_max.min(out_f - o0);
                    let mut recon = pool_alloc_uninit(ot * in_f).expect("pool alloc recon tile");
                    {
                        let cb_g = cbc.as_cuda_slice_read();
                        cuda.vq_reconstruct_f32(
                            &idx_g,
                            cb_g.slice(),
                            &sc_g,
                            &mut recon,
                            ot,
                            in_f,
                            dim,
                            vpr,
                            o0,
                        )
                        .expect("vq_reconstruct_f32 fp4-bwd tile");
                    }
                    let rshape = [ot, in_f];
                    let recon_tile = Tensor::from_storage(
                        Storage::from_cuda_slice(recon, ot * in_f, self.device()),
                        &rshape,
                    )
                    .expect("subbit from_storage");
                    let recon_t = recon_tile
                        .transpose(0, 1)
                        .expect("recon T")
                        .contiguous_gpu(); // [in_f,ot]
                    let mut rq = pool_alloc_uninit_u32(in_f * ot / 8).expect("fp4 pool rq");
                    let mut rs = pool_alloc_uninit_u32(in_f * ot / 64).expect("fp4 pool rs");
                    {
                        let g = recon_t.as_cuda_slice_read();
                        cuda.nvfp4_quant_f32(g.slice(), &mut rq, &mut rs, in_f * ot / 16)
                            .expect("nvfp4 quant reconT");
                    }
                    let gsl = gc.narrow(1, o0, ot).expect("g k-slice").contiguous_gpu(); // [n,ot]
                    let mut gq = pool_alloc_uninit_u32(n * ot / 8).expect("fp4 pool gq");
                    let mut gs = pool_alloc_uninit_u32(n * ot / 64).expect("fp4 pool gs");
                    {
                        let g = gsl.as_cuda_slice_read();
                        cuda.nvfp4_quant_sr_f32(
                            g.slice(),
                            &mut gq,
                            &mut gs,
                            n * ot / 16,
                            nvfp4_sr_seed(),
                        )
                        .expect("nvfp4 sr quant g slice");
                    }
                    cuda.nvfp4_gemm_sf2(&gq, &rq, &gs, &rs, &mut dx, n, in_f, ot, in_f, 0)
                        .expect("nvfp4 dx tile");
                    let gt_tile = gt.narrow(0, o0, ot).expect("gt rows").contiguous_gpu(); // [ot,n]
                    let mut gtq = pool_alloc_uninit_u32(ot * n / 8).expect("fp4 pool gtq");
                    let mut gts = pool_alloc_uninit_u32(ot * n / 64).expect("fp4 pool gts");
                    {
                        let g = gt_tile.as_cuda_slice_read();
                        cuda.nvfp4_quant_sr_f32(
                            g.slice(),
                            &mut gtq,
                            &mut gts,
                            ot * n / 16,
                            nvfp4_sr_seed(),
                        )
                        .expect("nvfp4 sr quant gT rows");
                    }
                    let mut dwt = pool_alloc_uninit(ot * in_f).expect("fp4 pool dW tile");
                    cuda.memset_zeros_f32(&mut dwt).expect("fp4 zero dW");
                    cuda.nvfp4_gemm_sf2(&gtq, &xtq, &gts, &xts, &mut dwt, ot, in_f, n, in_f, 0)
                        .expect("nvfp4 dW tile");
                    cuda.vq_dcb_scatter_f32(&dwt, &idx_g, &sc_g, &mut dcb, ot, in_f, dim, vpr, o0)
                        .expect("vq_dcb_scatter_f32 fp4");
                    cuda.sync();
                    pool_free(dwt);
                    pool_free_u32(gtq);
                    pool_free_u32(gts);
                    pool_free_u32(gq);
                    pool_free_u32(gs);
                    pool_free_u32(rq);
                    pool_free_u32(rs);
                    o0 += ot;
                }
                pool_free_u32(xtq);
                pool_free_u32(xts);
                Tensor::from_storage(Storage::from_cuda_slice(dx, n * in_f, self.device()), &sx)
                    .expect("subbit from_storage")
            } else if std::env::var("AXONML_TF32").is_ok() {
                let gt = gc.transpose(0, 1).expect("g transpose").contiguous_gpu(); // [out_f,n]
                let dx_out = Tensor::from_storage(
                    Storage::from_cuda_slice(
                        pool_alloc_uninit(n * in_f).expect("pool alloc dx"),
                        n * in_f,
                        self.device(),
                    ),
                    &sx,
                )
                .expect("subbit from_storage");
                let ot_max = ((1usize << 30) / 4 / in_f.max(1)).max(64); // recon tile <= ~1 GB
                let mut o0 = 0;
                let mut first = true;
                while o0 < out_f {
                    let ot = ot_max.min(out_f - o0);
                    let mut recon = pool_alloc_uninit(ot * in_f).expect("pool alloc recon tile");
                    {
                        let cb_g = cbc.as_cuda_slice_read();
                        cuda.vq_reconstruct_f32(
                            &idx_g,
                            cb_g.slice(),
                            &sc_g,
                            &mut recon,
                            ot,
                            in_f,
                            dim,
                            vpr,
                            o0,
                        )
                        .expect("vq_reconstruct_f32 bwd tile");
                    }
                    cuda.sync();
                    let rshape = [ot, in_f];
                    let recon_tile = Tensor::from_storage(
                        Storage::from_cuda_slice(recon, ot * in_f, self.device()),
                        &rshape,
                    )
                    .expect("subbit from_storage");
                    let _gts = [ot, n];
                    let gt_tile = gt.narrow(0, o0, ot).expect("gt_tile narrow");
                    // dx[n,in_f] += gt_tile^T[n,ot] @ recon_tile[ot,in_f]  (beta accumulates)
                    let _ = gt_tile
                        .transpose(0, 1)
                        .expect("gt_tile T")
                        .matmul_into_at(&recon_tile, &dx_out, 0, if first { 0.0 } else { 1.0 })
                        .expect("dx tile matmul");
                    cuda.sync();
                    // dcb: dW_tile[ot,in_f] = gt_tile[ot,n] @ x[n,in_f] on cuBLAS-tf32,
                    // then scatter into the codebook grad (atomic, matches dcb_rb).
                    let dw_tile = Tensor::from_storage(
                        Storage::from_cuda_slice(
                            pool_alloc_uninit(ot * in_f).expect("pool alloc dW tile"),
                            ot * in_f,
                            self.device(),
                        ),
                        &rshape,
                    )
                    .expect("subbit from_storage");
                    let _ = gt_tile
                        .matmul_into_at(&xc, &dw_tile, 0, 0.0)
                        .expect("dW tile matmul");
                    cuda.sync();
                    {
                        let dw_g = dw_tile.as_cuda_slice_read();
                        cuda.vq_dcb_scatter_f32(
                            dw_g.slice(),
                            &idx_g,
                            &sc_g,
                            &mut dcb,
                            ot,
                            in_f,
                            dim,
                            vpr,
                            o0,
                        )
                        .expect("vq_dcb_scatter_f32");
                    }
                    cuda.sync();
                    first = false;
                    o0 += ot;
                }
                cuda.sync();
                dx_out
            } else {
                let mut dx = pool_alloc_uninit(n * in_f).expect("pool alloc grad_x");
                {
                    let g_g = gc.as_cuda_slice_read();
                    let x_g = xc.as_cuda_slice_read();
                    let cb_g = cbc.as_cuda_slice_read();
                    cuda.vq_fused_matmul_dx_rb_f32(
                        g_g.slice(),
                        &idx_g,
                        cb_g.slice(),
                        &sc_g,
                        &mut dx,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                    )
                    .expect("vq_fused_matmul_dx_rb_f32");
                    cuda.vq_fused_matmul_dcb_rb_f32(
                        g_g.slice(),
                        x_g.slice(),
                        &idx_g,
                        &sc_g,
                        &mut dcb,
                        n,
                        in_f,
                        out_f,
                        dim,
                        vpr,
                    )
                    .expect("vq_fused_matmul_dcb_rb_f32");
                }
                cuda.sync();
                Tensor::from_storage(Storage::from_cuda_slice(dx, n * in_f, self.device()), &sx)
                    .expect("subbit from_storage")
            };
            let grad_cb =
                Tensor::from_storage(Storage::from_cuda_slice(dcb, k * dim, self.device()), &scb)
                    .expect("subbit from_storage");
            (grad_x, grad_cb)
        }

        /// Weight-offload VQ gather+denormalize: `self` is the `[k, dim]` codebook;
        /// returns the `[n_vec, dim]` reconstruction gathered by `assign` and scaled
        /// per row, uploading only `[n_vec]` assignments + `[n_rows]` scales instead
        /// of a full `[numel]` gather-index buffer. Bit-exact to
        /// `gather(codebook, expand(assign)).div(inv_scale)`.
        fn vq_gather_denorm_cuda(
            &self,
            assign: &[u32],
            row_scale: &[f32],
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let total = n_vec * dim;
            let mut assign_gpu =
                axonml_core::backends::cuda_pool::pool_alloc_uninit_u32(assign.len())
                    .expect("pool alloc assign");
            htod_u32_pinned(cuda, assign, &mut assign_gpu);
            let mut scale_gpu = pool_alloc_uninit(row_scale.len()).expect("pool alloc row_scale");
            cuda.htod_into(row_scale, &mut scale_gpu)
                .expect("htod row_scale failed");
            let cb_guard = self.as_cuda_slice_read();
            let mut recon = pool_alloc_uninit(total).expect("pool alloc vq_gather_denorm out");
            cuda.vq_gather_denorm_f32(
                &mut recon,
                cb_guard.slice(),
                &assign_gpu,
                &scale_gpu,
                n_vec,
                dim,
                vec_per_row,
            )
            .expect("vq_gather_denorm_f32 failed");
            axonml_core::backends::cuda_pool::pool_free_u32(assign_gpu);
            axonml_core::backends::cuda_pool::pool_free(scale_gpu);
            let shape = [n_vec, dim];
            let storage = Storage::from_cuda_slice(recon, total, self.device());
            Tensor::from_storage(storage, &shape).expect("subbit from_storage")
        }

        /// GPU-resident VQ gather+denorm: `self` is the `[k, dim]` codebook; reads the
        /// pre-uploaded assign + scale for `tile` from `resident` — NO per-tile H2D,
        /// NO sync (stream-ordered kernel only → CUDA-graph capturable). Bit-exact to
        /// `vq_gather_denorm_cuda`.
        fn vq_gather_denorm_resident(
            &self,
            resident: &ResidentAssign,
            tile: usize,
            n_vec: usize,
            dim: usize,
            vec_per_row: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let total = n_vec * dim;
            let cb_guard = self.as_cuda_slice_read();
            let mut recon = pool_alloc_uninit(total).expect("pool alloc resident gather out");
            cuda.vq_gather_denorm_f32(
                &mut recon,
                cb_guard.slice(),
                &resident.assign[tile],
                &resident.scale[tile],
                n_vec,
                dim,
                vec_per_row,
            )
            .expect("vq_gather_denorm_f32 resident failed");
            let shape = [n_vec, dim];
            let storage = Storage::from_cuda_slice(recon, total, self.device());
            Tensor::from_storage(storage, &shape).expect("subbit from_storage")
        }

        /// Device-to-device write of `self`'s contiguous data into `dst`'s storage
        /// starting at element `dst_elem_offset`, stream-ordered (no host sync).
        /// Used to assemble a tiled output on-device without a per-tile `to_vec`
        /// stall. `self` must be contiguous; `dst` must have room for `self.numel()`
        /// elements at the offset.
        fn dtod_write_at(&self, dst: &Self, dst_elem_offset: usize) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let src_guard = self.as_cuda_slice_read();
            let mut dst_guard = dst.as_cuda_slice_write();
            cuda.memcpy_dtod_async_f32(
                dst_guard.slice_mut(),
                dst_elem_offset,
                src_guard.slice(),
                0,
                self.numel(),
            )
            .expect("dtod_write_at: async memcpy_dtod failed");
        }
        fn scatter_heads_at(
            &self,
            dst: &Self,
            pos: usize,
            nkv: usize,
            t: usize,
            hd: usize,
            max_ctx: usize,
        ) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let sc = self.contiguous(); // [1, nkv, t, hd] contiguous => [nkv][t*hd]
            let src_guard = sc.as_cuda_slice_read();
            let mut dst_guard = dst.as_cuda_slice_write();
            cuda.memcpy_2d_dtod_f32(
                dst_guard.slice_mut(),
                pos * hd,     // dst start within each head's [max_ctx, hd] region
                max_ctx * hd, // dst row pitch (one head)
                src_guard.slice(),
                0,      // src start
                t * hd, // src row pitch (contiguous)
                t * hd, // width per row (this token's t positions × hd)
                nkv,    // rows = heads
            )
            .expect("scatter_heads_at: memcpy_2d_dtod failed");
        }

        fn scatter_heads_at_devpos(
            &self,
            dst: &Self,
            pos: &Self,
            nkv: usize,
            t: usize,
            hd: usize,
            max_ctx: usize,
        ) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let sc = self.contiguous(); // [1, nkv, t, hd] contiguous => [nkv][t*hd]
            let src_guard = sc.as_cuda_slice_read();
            let pos_guard = pos.as_cuda_slice_read();
            let mut dst_guard = dst.as_cuda_slice_write();
            cuda.kv_scatter_devpos_f32(
                src_guard.slice(),
                dst_guard.slice_mut(),
                pos_guard.slice(),
                nkv,
                t,
                hd,
                max_ctx,
            )
            .expect("scatter_heads_at_devpos: kv_scatter_devpos_f32 failed");
        }

        fn apply_rope_devpos(
            &self,
            pos: &Self,
            n_heads: usize,
            seq: usize,
            head_dim: usize,
            theta: f32,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let src = self.contiguous();
            let n = src.numel();
            let mut out = pool_alloc_uninit(n).expect("apply_rope_devpos out pool alloc");
            {
                let src_guard = src.as_cuda_slice_read();
                let pos_guard = pos.as_cuda_slice_read();
                cuda.rope_split_halves_bhsd_devpos_f32(
                    src_guard.slice(),
                    &mut out,
                    pos_guard.slice(),
                    seq,
                    n_heads,
                    head_dim,
                    theta,
                )
                .expect("rope_split_halves_bhsd_devpos_f32 failed");
            }
            Tensor::from_storage(Storage::from_cuda_slice(out, n, src.device()), src.shape())
                .expect("apply_rope_devpos from_storage")
        }
    }

    // == embedding ext ==
    pub trait SubbitEmbeddingExt: Sized {
        /// GPU-resident scatter-add: uses the pre-uploaded assign for `tile` from
        /// `resident` as the indices — NO per-tile H2D, NO sync (capturable). `self`
        /// is grad_output. Bit-exact to `embedding_scatter_add_cuda`.
        fn embedding_scatter_add_resident(
            &self,
            resident: &ResidentAssign,
            tile: usize,
            num_embeddings: usize,
            emb_dim: usize,
        ) -> Self;
        /// Like `embedding_scatter_add_resident` but ACCUMULATES straight into the
        /// existing `out` buffer (the scatter is atomicAdd) — no per-tile allocation,
        /// no memset, no separate accumulate-add. Caller zeros `out` once, then every
        /// tile scatters into it. `self` is grad_output for the tile.
        fn embedding_scatter_add_resident_accum(
            &self,
            resident: &ResidentAssign,
            tile: usize,
            out: &Self,
            emb_dim: usize,
        );
    }

    impl SubbitEmbeddingExt for Tensor<f32> {
        /// GPU-resident scatter-add: uses the pre-uploaded assign for `tile` from
        /// `resident` as the indices — NO per-tile H2D, NO sync (capturable). `self`
        /// is grad_output. Bit-exact to `embedding_scatter_add_cuda`.
        fn embedding_scatter_add_resident(
            &self,
            resident: &ResidentAssign,
            tile: usize,
            num_embeddings: usize,
            emb_dim: usize,
        ) -> Self {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let idx_gpu = &resident.assign[tile];
            let num_indices = idx_gpu.len();
            let total_n = num_indices * emb_dim;
            let out_size = num_embeddings * emb_dim;
            let mut out = pool_alloc(out_size).expect("GPU pool alloc failed");
            cuda.memset_zeros_f32(&mut out)
                .expect("memset zeros failed");
            let grad = self.contiguous_gpu();
            let grad_guard = grad.as_cuda_slice_read();
            cuda.embedding_scatter_add_f32(grad_guard.slice(), idx_gpu, &mut out, total_n, emb_dim)
                .expect("CUDA embedding_scatter_add_f32 resident failed");
            let _shape = axonml_tensor::shape::Shape::from_slice(&[num_embeddings, emb_dim]);
            let storage = Storage::from_cuda_slice(out, out_size, self.device());
            Tensor::from_storage(storage, &[num_embeddings, emb_dim])
                .expect("embedding from_storage")
        }

        /// Like `embedding_scatter_add_resident` but ACCUMULATES straight into the
        /// existing `out` buffer (the scatter is atomicAdd) — no per-tile allocation,
        /// no memset, no separate accumulate-add. Caller zeros `out` once, then every
        /// tile scatters into it. `self` is grad_output for the tile.
        fn embedding_scatter_add_resident_accum(
            &self,
            resident: &ResidentAssign,
            tile: usize,
            out: &Self,
            emb_dim: usize,
        ) {
            let cuda = get_cuda_backend().expect("CUDA backend not available");
            let idx_gpu = &resident.assign[tile];
            let total_n = idx_gpu.len() * emb_dim;
            let grad = self.contiguous_gpu();
            let grad_guard = grad.as_cuda_slice_read();
            let mut out_guard = out.as_cuda_slice_write();
            cuda.embedding_scatter_add_f32(
                grad_guard.slice(),
                idx_gpu,
                out_guard.slice_mut(),
                total_n,
                emb_dim,
            )
            .expect("CUDA embedding_scatter_add_f32 accum failed");
        }
    }
}
#[cfg(feature = "cuda")]
pub use imp::*;
