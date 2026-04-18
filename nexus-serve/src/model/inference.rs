//! inference — Transformer Forward Pass + KV Cache + Sampling
//!
//! The inference engine proper for nexus-serve. Loads GGUF weights through an
//! mmap-backed [`MappedGguf`], dequantizes or keeps-packed per [`Weight`],
//! runs decoder-only forward passes across multiple architecture dialects,
//! maintains a per-session KV cache, and samples tokens. No autograd graph —
//! pure inference.
//!
//! Top-level types and responsibilities:
//! - [`InferenceConfig`]: hyperparameters extracted from GGUF metadata
//!   (vocab_size, hidden_size, num_heads, head_dim, RoPE theta, RMSNorm eps,
//!   and per-family [`GemmaConfig`] for dual-RoPE / SWA / softcap / altup).
//! - [`MappedGguf`]: mmap + tensor directory + `load_tensor_f32` (dequant
//!   to f32) and `load_tensor_raw` (keep-packed, with the I2_S +4-byte
//!   trailing scale trick for BitNet).
//! - [`InferenceEngine`]: canonical engine with `token_embed`, per-layer
//!   [`LayerWeights`] (LLaMA-family) OR [`Gemma4Weights`] + [`Gemma4LayerWeights`]
//!   (sandwich norms, Q/K norm, per-layer input embedding table), shared
//!   `output_norm` + `lm_head`, and a `compute_device` for CUDA dispatch.
//! - Generation: [`InferenceEngine::generate`] (block) and
//!   [`InferenceEngine::generate_stream`] (per-token callback), both
//!   running the architecture-dispatched forward with KV cache + sampling
//!   (argmax when temperature ≈ 0, else top-p softmax sampling).
//!
//! Supported architectures (dispatched at runtime by `config.architecture`):
//! - LLaMA-family (LLaMA, Qwen2, Mistral, Phi): `token_embd → N × (attn_norm
//!   → QKV → split-halves RoPE → attention → attn_output → ffn_norm →
//!   gate/up → SiLU → down) → output_norm → lm_head`.
//! - Gemma 2/3/4: sandwich RMSNorm (pre+post attn, pre+post FFN), per-head
//!   Q/K RMSNorm before RoPE, per-layer sliding-window vs full attention
//!   pattern, dual RoPE bases (1e6 full / 1e4 SWA), final logit softcap,
//!   Gemma 3n altup (per-layer input embeddings added into the hidden
//!   state — see [`add_per_layer_input_embed`]).
//! - BitNet b1.58: I2S ternary weights via `axonml_quant::bitnet`, plus the
//!   `attn_sub_norm` / `ffn_sub_norm` stabilizer RMSNorms.
//!
//! Optimizations:
//! - **Pre-transposed weights** at load time (no per-forward allocation).
//! - **KV cache** for incremental decoding (only process the new token each
//!   step), with CUDA-cached GPU uploads via `OnceLock` inside [`Weight`].
//! - **Fused flash-decode attention** and **prefill attention** kernels
//!   (see `axonml_kernels::flash_attention`) dispatched when the CUDA
//!   feature is on. SWA-aware variant picks masked vs full per-layer.
//! - **`Tensor::matmul`** for BLAS-accelerated CPU matmul; CUDA GEMV with
//!   dequant-in-shader for Q4_K / Q6_K weights; element-wise ops stay as
//!   raw loops (already fast for 1D data).
//!
//! Section map (already present as `=====` banners in this file):
//! - Config extracted from GGUF metadata
//! - Weight loading: GGUF → f32 tensors via mmap + dequantization
//! - Inference Engine — PRE-TRANSPOSED weights + KV cache
//! - Gemma 4 — distinct per-layer layout (sandwich norms, Q/K norm, etc.)
//! - Math helpers
//! - Gemma 3n altup (per-layer input embeddings)
//! - Tensor loading helpers
//!
//! # File
//! `nexus-serve/src/model/inference.rs`
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

use std::collections::HashMap;
use std::path::Path;

use axonml_core::Device;
use axonml_tensor::Tensor;
use memmap2::Mmap;
use rayon::prelude::*;

use super::gguf::{self, GgufFile, GgufTensorInfo, GgmlType};
use super::weight::Weight;

// =============================================================================
// Config extracted from GGUF metadata
// =============================================================================

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    /// Head dim for full attention. For LLaMA/Qwen2: `hidden_size / num_heads`.
    /// For Gemma 4: `gemma4.attention.key_length` (often decoupled from hidden_size).
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub architecture: String,
    /// The GGUF `general.name` string (e.g. "DeepSeek R1 Distill Qwen 7B").
    /// Distinct from `architecture`, which only identifies the model family
    /// (e.g. "qwen2"). Needed for chat-template dispatch when the base
    /// architecture doesn't uniquely identify the fine-tune variant — e.g.
    /// R1-Distill-Qwen uses the Qwen2 architecture but DeepSeek's own
    /// chat tokens (`<｜User｜>`, `<｜Assistant｜>`, BOS), so ChatML
    /// rendering produces garbage.
    pub model_name: String,
    /// Gemma-family hyperparameters (sliding window, dual RoPE base, softcap,
    /// per-layer token embeddings). `None` for LLaMA/Qwen2/Mistral.
    pub gemma: Option<GemmaConfig>,
}

/// Gemma-family extras not present in LLaMA/Qwen2/Mistral.
///
/// Populated when `architecture` starts with `"gemma"`.
#[derive(Debug, Clone)]
pub struct GemmaConfig {
    /// Per-layer: true means use sliding-window attention for this layer.
    /// Length == `num_layers`. For Gemma 4, every 6th layer is full-attention.
    pub sliding_window_pattern: Vec<bool>,
    /// Sliding window size in tokens (e.g. 512).
    pub sliding_window: usize,
    /// Head dim used in SWA layers (may be smaller than `head_dim`, e.g. 256 vs 512).
    pub head_dim_swa: usize,
    /// RoPE θ used in SWA layers (e.g. 10_000, distinct from full-attention 1_000_000).
    pub rope_theta_swa: f32,
    /// RoPE dim count for full layers (may be < `head_dim`).
    pub rope_dim: usize,
    /// RoPE dim count for SWA layers (may be < `head_dim_swa`).
    pub rope_dim_swa: usize,
    /// Final logit softcap: `logits = tanh(logits / softcap) * softcap`. `None` disables.
    pub final_logit_softcap: Option<f32>,
    /// Width of the per-layer input embedding (e.g. 256). Adds one scaled vector per layer.
    pub per_layer_input_width: usize,
    /// Q/K RMSNorm dimension (e.g. 256 — applied per-head before RoPE).
    pub qk_norm_dim: usize,
}

impl InferenceConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let arch = gguf.architecture().unwrap_or("llama").to_string();
        let model_name = gguf
            .get_meta("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let prefix = &arch;
        let is_gemma = arch.starts_with("gemma");

        let hidden_size = gguf
            .get_meta(&format!("{prefix}.embedding_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(4096) as usize;

        let num_layers = gguf
            .get_meta(&format!("{prefix}.block_count"))
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;

        let num_heads = gguf
            .get_meta(&format!("{prefix}.attention.head_count"))
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;

        let num_kv_heads = gguf
            .get_meta(&format!("{prefix}.attention.head_count_kv"))
            .and_then(|v| v.as_u32())
            .unwrap_or(num_heads as u32) as usize;

        // head_dim: LLaMA/Qwen2 derive it from hidden/num_heads; Gemma specifies
        // it explicitly via key_length because it's decoupled from hidden_size.
        let head_dim = gguf
            .get_meta(&format!("{prefix}.attention.key_length"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_heads);

        let intermediate_size = gguf
            .get_meta(&format!("{prefix}.feed_forward_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or((hidden_size * 4) as u32) as usize;

        let max_seq_len = gguf
            .get_meta(&format!("{prefix}.context_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(2048) as usize;

        let rms_norm_eps = gguf
            .get_meta(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

        // Gemma 3 (non-E-series) GGUFs often omit `rope.freq_base` entirely.
        // The architecture actually uses 1,000,000 for global layers and
        // 10,000 for sliding-window layers (Gemma 3 tech report §3.1).
        // Fall back to the correct Gemma default when missing.
        let rope_theta = gguf
            .get_meta(&format!("{prefix}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or_else(|| if is_gemma { 1_000_000.0 } else { 10_000.0 });

        let vocab_size = gguf
            .get_meta("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                gguf::GgufValue::Array(a) => Some(a.len()),
                _ => None,
            })
            .unwrap_or(32000);

        let gemma = if is_gemma {
            // Dump every Gemma-prefixed metadata key so we can see exactly
            // what the GGUF exposes (Gemma 3 vs Gemma 3n / E-series differ).
            let mut keys: Vec<&String> = gguf.metadata.keys().filter(|k| k.starts_with(prefix)).collect();
            keys.sort();
            eprintln!("[gemma-meta] {} keys with prefix '{}':", keys.len(), prefix);
            for k in &keys {
                if let Some(v) = gguf.metadata.get(*k) {
                    let s = match v {
                        gguf::GgufValue::U32(n) => format!("U32({})", n),
                        gguf::GgufValue::I32(n) => format!("I32({})", n),
                        gguf::GgufValue::F32(n) => format!("F32({})", n),
                        gguf::GgufValue::Bool(b) => format!("Bool({})", b),
                        gguf::GgufValue::String(s) => format!("String(len={})", s.len()),
                        gguf::GgufValue::Array(a) => format!("Array(len={})", a.len()),
                        _ => "Other".to_string(),
                    };
                    eprintln!("  {} = {}", k, s);
                }
            }
            let cfg = Self::parse_gemma_config(gguf, prefix, num_layers, head_dim);
            eprintln!(
                "[gemma-config] prefix={} hidden={} n_q={} n_kv={} head_dim={} inter={} rope={}  swa_head_dim={} rope_swa={} rope_dim={} rope_dim_swa={} softcap={:?} altup_width={} swa_pattern_len={} swa_true_count={}",
                prefix,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                rope_theta,
                cfg.head_dim_swa,
                cfg.rope_theta_swa,
                cfg.rope_dim,
                cfg.rope_dim_swa,
                cfg.final_logit_softcap,
                cfg.per_layer_input_width,
                cfg.sliding_window_pattern.len(),
                cfg.sliding_window_pattern.iter().filter(|&&b| b).count(),
            );
            Some(cfg)
        } else {
            None
        };

        Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
            architecture: arch,
            model_name,
            gemma,
        }
    }

    /// Extract Gemma-family fields (sliding window, dual RoPE, softcap, …).
    fn parse_gemma_config(
        gguf: &GgufFile,
        prefix: &str,
        num_layers: usize,
        head_dim: usize,
    ) -> GemmaConfig {
        let sliding_window = gguf
            .get_meta(&format!("{prefix}.attention.sliding_window"))
            .and_then(|v| v.as_u32())
            .unwrap_or(4096) as usize;

        // Extract the per-layer SWA pattern. Stored as a GGUF array of bools
        // with length == num_layers. For E4B the pattern is explicit in
        // metadata; for Gemma 3 (non-E-series) the GGUF typically omits it
        // and the canonical pattern is "5 SWA, 1 global" repeating (Gemma 3
        // tech report §3.1). Fall back to that when missing.
        let sliding_window_pattern = gguf
            .get_meta(&format!("{prefix}.attention.sliding_window_pattern"))
            .and_then(|v| match v {
                gguf::GgufValue::Array(arr) => Some(
                    arr.iter()
                        .map(|elt| matches!(elt, gguf::GgufValue::Bool(true)))
                        .collect::<Vec<bool>>(),
                ),
                _ => None,
            })
            .unwrap_or_else(|| {
                // 5 SWA + 1 global, repeating: indices 0..4 SWA, index 5 global,
                // 6..10 SWA, 11 global, etc. Each layer `i` is SWA iff
                // `(i % 6) != 5`.
                (0..num_layers).map(|i| (i % 6) != 5).collect()
            });

        // SWA head dim: defaults to `head_dim` for models that don't have the
        // split (e.g. Gemma 2 uses a single head_dim across all layers).
        let head_dim_swa = gguf
            .get_meta(&format!("{prefix}.attention.key_length_swa"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let rope_theta_swa = gguf
            .get_meta(&format!("{prefix}.rope.freq_base_swa"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);

        let rope_dim = gguf
            .get_meta(&format!("{prefix}.rope.dimension_count"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let rope_dim_swa = gguf
            .get_meta(&format!("{prefix}.rope.dimension_count_swa"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(head_dim_swa);

        let final_logit_softcap = gguf
            .get_meta(&format!("{prefix}.final_logit_softcapping"))
            .and_then(|v| v.as_f32());

        // Per-layer input embedding width (e.g. 256). Not all Gemma variants
        // have this — Gemma 2 doesn't, Gemma 3/4 do.
        let per_layer_input_width = gguf
            .get_meta(&format!("{prefix}.embedding_length_per_layer_input"))
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(0);

        // Q/K norm dim — usually the SWA head dim (256) or a fixed per-head
        // scale. We don't see a dedicated metadata key, so fall back to the
        // `attn_q_norm.weight` tensor's first dim when we load weights.
        // For now, default to head_dim_swa which is correct for Gemma 4.
        let qk_norm_dim = head_dim_swa;

        GemmaConfig {
            sliding_window_pattern,
            sliding_window,
            head_dim_swa,
            rope_theta_swa,
            rope_dim,
            rope_dim_swa,
            final_logit_softcap,
            per_layer_input_width,
            qk_norm_dim,
        }
    }
}

// =============================================================================
// Weight loading: GGUF → f32 tensors via mmap + dequantization
// =============================================================================

pub struct MappedGguf {
    _mmap: Mmap,
    data_ptr: *const u8,
    data_offset: u64,
    pub tensors: HashMap<String, GgufTensorInfo>,
}

unsafe impl Send for MappedGguf {}
unsafe impl Sync for MappedGguf {}

impl MappedGguf {
    pub fn open(path: &Path, gguf: &GgufFile) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data_ptr = mmap.as_ptr();

        let mut tensors = HashMap::with_capacity(gguf.tensors.len());
        for t in &gguf.tensors {
            tensors.insert(t.name.clone(), t.clone());
        }

        Ok(Self {
            _mmap: mmap,
            data_ptr,
            data_offset: gguf.data_offset,
            tensors,
        })
    }

    pub fn load_tensor_f32(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let info = self.tensors.get(name)?;
        let n_elements = info.n_elements() as usize;
        let total_bytes = info.total_bytes() as usize;
        let offset = (self.data_offset + info.offset) as usize;

        let raw_data = unsafe {
            std::slice::from_raw_parts(self.data_ptr.add(offset), total_bytes)
        };

        let mut output = vec![0.0f32; n_elements];

        match info.dtype {
            GgmlType::F32 => {
                for (i, chunk) in raw_data.chunks_exact(4).enumerate() {
                    output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            GgmlType::F16 => {
                gguf::dequantize_f16(raw_data, &mut output);
            }
            GgmlType::Q8_0 => {
                let block_size = 32;
                let type_size = 34;
                for (block_idx, chunk) in raw_data.chunks_exact(type_size).enumerate() {
                    let out_offset = block_idx * block_size;
                    if out_offset + block_size <= n_elements {
                        gguf::dequantize_q8_0(chunk, &mut output[out_offset..out_offset + block_size]);
                    }
                }
            }
            GgmlType::Q4_0 => {
                let block_size = 32;
                let type_size = 18;
                for (block_idx, chunk) in raw_data.chunks_exact(type_size).enumerate() {
                    let out_offset = block_idx * block_size;
                    if out_offset + block_size <= n_elements {
                        gguf::dequantize_q4_0(chunk, &mut output[out_offset..out_offset + block_size]);
                    }
                }
            }
            GgmlType::Q4K => {
                let block_size = 256;
                let type_size = 144;
                for (block_idx, chunk) in raw_data.chunks_exact(type_size).enumerate() {
                    let out_offset = block_idx * block_size;
                    if out_offset + block_size <= n_elements {
                        gguf::dequantize_q4_k(chunk, &mut output[out_offset..out_offset + block_size]);
                    }
                }
            }
            GgmlType::Q6K => {
                let block_size = 256;
                let type_size = 210;
                for (block_idx, chunk) in raw_data.chunks_exact(type_size).enumerate() {
                    let out_offset = block_idx * block_size;
                    if out_offset + block_size <= n_elements {
                        gguf::dequantize_q6_k(chunk, &mut output[out_offset..out_offset + block_size]);
                    }
                }
            }
            GgmlType::BF16 => {
                for (i, chunk) in raw_data.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    output[i] = f32::from_bits((bits as u32) << 16);
                }
            }
            GgmlType::I2S => {
                // BitNet b1.58 ternary weights. The eager path needs both
                // the packed bytes and the tensor-wide scale. `raw_data`
                // here is just the packed region (no scale) because
                // `load_tensor_f32` uses `info.total_bytes()` without the
                // I2_S +4 extension that `load_tensor_raw` does. Read the
                // scale directly from the mmap at `offset + total_bytes`.
                let scale_offset = (self.data_offset + info.offset) as usize + total_bytes;
                let scale_bytes = unsafe {
                    std::slice::from_raw_parts(self.data_ptr.add(scale_offset), 4)
                };
                let scale = f32::from_le_bytes([
                    scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3],
                ]);
                axonml_quant::bitnet::dequantize_i2s(raw_data, scale, &mut output);
            }
            other => {
                eprintln!("  WARNING: unsupported quantization {:?} for tensor {}, filling with zeros", other, name);
            }
        }

        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        Some((output, dims))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Load the raw tensor bytes (compressed/quantized form) without dequantizing.
    /// Used for lazy-dequant mode where weights stay compact in RAM.
    ///
    /// **I2_S special case:** Microsoft's format stores a tensor-wide f32
    /// scale at offset `total_bytes` (right after the packed data, inside
    /// the 32-byte GGUF alignment padding). We read 4 extra bytes for
    /// I2_S tensors so the scale is appended to the returned buffer —
    /// callers recover it via the last 4 bytes (`f32::from_le_bytes`).
    pub fn load_tensor_raw(&self, name: &str) -> Option<Vec<u8>> {
        let info = self.tensors.get(name)?;
        let mut total_bytes = info.total_bytes() as usize;
        if info.dtype == GgmlType::I2S {
            total_bytes += 4; // trailing f32 tensor-wide scale
        }
        let offset = (self.data_offset + info.offset) as usize;
        let raw_data = unsafe {
            std::slice::from_raw_parts(self.data_ptr.add(offset), total_bytes)
        };
        Some(raw_data.to_vec())
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

// =============================================================================
// Inference Engine — PRE-TRANSPOSED weights + KV cache
// =============================================================================

pub struct InferenceEngine {
    pub config: InferenceConfig,
    /// Token embedding [vocab_size, hidden_size]
    pub token_embed: Vec<f32>,
    /// Per-layer weights for LLaMA-family architectures (LLaMA, Qwen2,
    /// Mistral). Empty when `gemma4` is populated.
    pub layers: Vec<LayerWeights>,
    /// Per-layer weights + top-level specials for Gemma 4. `None` for
    /// LLaMA/Qwen2/Mistral. Present when `config.architecture == "gemma4"`.
    pub gemma4: Option<Gemma4Weights>,
    /// Output norm [hidden_size]
    pub output_norm: Vec<f32>,
    /// LM head — logical shape `[hidden, vocab]` (post-transpose).
    pub lm_head: Weight,
    /// Target compute device for matmul. For f32 weights, inputs and weights
    /// live on this device. For quantized weights, dequantization happens on
    /// CPU but the dequantized scratch is moved here for matmul.
    pub compute_device: Device,
    /// Lazy-cached GPU Tensor copies of per-layer norm + bias weights, so
    /// the decode hot path can call them repeatedly without re-uploading
    /// `[hidden]`-sized vectors every token. Populated on first decode step
    /// when `compute_device.is_gpu()`. Indexed by layer; same length as
    /// `layers`. Output-norm cache lives separately since it's only used
    /// at the top of the call stack, not in the inner loop.
    #[cfg(feature = "cuda")]
    pub gpu_layer_cache: std::sync::OnceLock<Vec<LayerGpuCache>>,
    /// Lazy-cached GPU Tensor for the final `output_norm` weight.
    #[cfg(feature = "cuda")]
    pub gpu_output_norm: std::sync::OnceLock<Tensor<f32>>,
    /// TurboQuant Q8 KV cache toggle. When true, the decode path
    /// allocates `GpuKvCacheQ8` (int8 values + per-head f32 scale) and
    /// calls `fused_attn_decode_q8_f32` instead of the f32 variant.
    /// ~3.9× memory compression; same algorithm, dequant inline in the
    /// attention kernel. Set at load time by the CLI `--kv-quant q8`
    /// flag or programmatically via [`InferenceEngine::set_kv_quant_q8`].
    #[cfg(feature = "cuda")]
    pub kv_quant_q8: bool,
}

/// GPU-resident copies of a `LayerWeights`'s CPU `Vec<f32>` members
/// (norms + biases). Matmul weights live in `LayerWeights::q_weight` etc.
/// and already have their own caching.
#[cfg(feature = "cuda")]
pub struct LayerGpuCache {
    pub attn_norm: Tensor<f32>,
    pub ffn_norm: Tensor<f32>,
    pub q_bias: Option<Tensor<f32>>,
    pub k_bias: Option<Tensor<f32>>,
    pub v_bias: Option<Tensor<f32>>,
    pub attn_sub_norm: Option<Tensor<f32>>,
    pub ffn_sub_norm: Option<Tensor<f32>>,
    /// Qwen3 QK-norm pair (see `LayerWeights::attn_q_norm`).
    pub attn_q_norm: Option<Tensor<f32>>,
    pub attn_k_norm: Option<Tensor<f32>>,
}

#[cfg(feature = "cuda")]
impl LayerGpuCache {
    fn new_for(layer: &LayerWeights, device: &Device) -> Self {
        fn upload(v: &[f32], d: &Device) -> Tensor<f32> {
            Tensor::from_vec(v.to_vec(), &[v.len()])
                .expect("cache norm/bias tensor")
                .to_device(d.clone())
                .expect("move norm/bias to GPU")
        }
        fn upload_opt(v: &Option<Vec<f32>>, d: &Device) -> Option<Tensor<f32>> {
            v.as_ref().map(|x| upload(x, d))
        }
        Self {
            attn_norm: upload(&layer.attn_norm, device),
            ffn_norm: upload(&layer.ffn_norm, device),
            q_bias: upload_opt(&layer.q_bias, device),
            k_bias: upload_opt(&layer.k_bias, device),
            v_bias: upload_opt(&layer.v_bias, device),
            attn_sub_norm: upload_opt(&layer.attn_sub_norm, device),
            ffn_sub_norm: upload_opt(&layer.ffn_sub_norm, device),
            attn_q_norm: upload_opt(&layer.attn_q_norm, device),
            attn_k_norm: upload_opt(&layer.attn_k_norm, device),
        }
    }
}

/// Weights for one transformer layer.
/// Each weight is logically shaped `[in, out]` (post-transpose convention):
/// matmul is: `input [seq, in] @ weight [in, out] = [seq, out]`.
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub q_weight: Weight,    // [hidden, n_heads*head_dim]
    pub k_weight: Weight,    // [hidden, n_kv_heads*head_dim]
    pub v_weight: Weight,    // [hidden, n_kv_heads*head_dim]
    pub o_weight: Weight,    // [n_heads*head_dim, hidden]
    pub ffn_norm: Vec<f32>,
    pub gate_weight: Weight, // [hidden, intermediate]
    pub up_weight: Weight,   // [hidden, intermediate]
    pub down_weight: Weight, // [intermediate, hidden]
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    /// BitNet b1.58: RMSNorm applied to the attention output right BEFORE
    /// `o_weight`, after the softmax-weighted V sum. Stabilizes training at
    /// ternary precision by keeping the input to the quantized output
    /// projection in a well-conditioned range. `None` for non-BitNet archs.
    /// Shape: `[n_heads * head_dim]`.
    pub attn_sub_norm: Option<Vec<f32>>,
    /// BitNet b1.58: RMSNorm applied to the FFN hidden state (post-SwiGLU)
    /// right BEFORE `down_weight`. Same purpose as `attn_sub_norm`. `None`
    /// for non-BitNet archs. Shape: `[intermediate_size]`.
    pub ffn_sub_norm: Option<Vec<f32>>,
    /// Qwen3-only: per-head RMSNorm applied to Q after QKV matmul and
    /// BEFORE RoPE. The same `[head_dim]` weight vector is broadcast to
    /// every Q head. `None` for architectures without QK-norm (Qwen2,
    /// Gemma, LLaMA, BitNet, …).
    pub attn_q_norm: Option<Vec<f32>>,
    /// Qwen3-only: per-head RMSNorm on K with the same `[head_dim]`
    /// broadcast shape as `attn_q_norm`. Applied to every KV head.
    pub attn_k_norm: Option<Vec<f32>>,
}

impl LayerWeights {
    /// Move all weight tensors to the specified device (GPU).
    /// Quantized weights stay on CPU (dequantized to GPU per-matmul).
    fn to_device(&mut self, device: Device) {
        self.q_weight.to_device(device.clone());
        self.k_weight.to_device(device.clone());
        self.v_weight.to_device(device.clone());
        self.o_weight.to_device(device.clone());
        self.gate_weight.to_device(device.clone());
        self.up_weight.to_device(device.clone());
        self.down_weight.to_device(device);
    }
}

// =============================================================================
// Gemma 4 — distinct per-layer layout (sandwich norms, Q/K norm, etc.)
// =============================================================================

/// Weights for one Gemma-family transformer layer.
///
/// Differs from the LLaMA-family `LayerWeights`:
/// - Four norms wrap each sublayer (pre + post attention, pre + post FFN)
///   instead of two
/// - Extra RMSNorm on Q and K projections (applied per-head, pre-RoPE)
/// - No Q/K/V bias (Gemma projections are pure matmul)
///
/// Shape conventions match `LayerWeights`: each `Weight` is logically
/// `[in, out]` (post-transpose), so matmul is `input @ weight`.
pub struct Gemma4LayerWeights {
    // Norms (RMSNorm weights, 1D)
    pub attn_norm: Vec<f32>,                  // [hidden]  — pre-attn
    pub post_attention_norm: Vec<f32>,        // [hidden]  — post-attn (Gemma-only)
    pub ffn_norm: Vec<f32>,                   // [hidden]  — pre-FFN
    pub post_ffw_norm: Vec<f32>,              // [hidden]  — post-FFN (Gemma-only)
    pub post_norm: Option<Vec<f32>>,          // [hidden]  — `blk.N.post_norm`; purpose TBD
    pub q_norm: Vec<f32>,                     // [qk_norm_dim]  — applied per-head, pre-RoPE
    pub k_norm: Vec<f32>,                     // [qk_norm_dim]
    // Attention projections
    pub q_weight: Weight,                     // [hidden, n_heads * head_dim]
    pub k_weight: Weight,                     // [hidden, n_kv_heads * head_dim]
    pub v_weight: Weight,
    pub o_weight: Weight,                     // [n_heads * head_dim, hidden]
    // FFN projections
    pub gate_weight: Weight,                  // [hidden, intermediate]
    pub up_weight: Weight,
    pub down_weight: Weight,                  // [intermediate, hidden]
}

impl Gemma4LayerWeights {
    fn to_device(&mut self, device: Device) {
        self.q_weight.to_device(device.clone());
        self.k_weight.to_device(device.clone());
        self.v_weight.to_device(device.clone());
        self.o_weight.to_device(device.clone());
        self.gate_weight.to_device(device.clone());
        self.up_weight.to_device(device.clone());
        self.down_weight.to_device(device);
        // Norms are tiny (1D, ~few KB each) and live in Vec<f32>; they stay on CPU.
    }
}

/// Top-level weights specific to Gemma 4. Includes the per-layer input
/// embeddings feature (Gemma 3/4 novelty).
pub struct Gemma4Weights {
    pub layers: Vec<Gemma4LayerWeights>,

    /// Per-layer token embedding table, shape `[vocab, num_layers * per_layer_input_width]`
    /// (stored flat as `Vec<f32>`). Looked up once per token at the start of a
    /// forward pass, then sliced per-layer and added to the hidden state.
    pub per_layer_token_embd: Vec<f32>,

    /// Altup gate projection. `None` on non-E-series Gemma models that omit
    /// the altup mechanism entirely. When present, shape is
    /// `[hidden_size, num_layers * per_layer_input_width]` — projects the
    /// hidden state to a per-layer gate vector.
    pub per_layer_model_proj: Option<Weight>,

    /// RMSNorm weight for the per-layer input path. Shape `[per_layer_input_width]`.
    pub per_layer_proj_norm: Vec<f32>,

    /// Optional pre-computed RoPE frequency table. Shape `[head_dim / 2]` or
    /// similar — we fall back to computing RoPE freqs on the fly if the tensor
    /// is absent in the GGUF.
    pub rope_freqs: Option<Vec<f32>>,
}

impl Gemma4Weights {
    fn to_device(&mut self, device: Device) {
        if let Some(ref mut proj) = self.per_layer_model_proj {
            proj.to_device(device.clone());
        }
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.to_device(device.clone());
            if (i + 1) % 7 == 0 || i + 1 == num_layers {
                println!("    Layer {}/{}", i + 1, num_layers);
            }
        }
    }

    /// Compressed bytes used in RAM across all Gemma-specific weights.
    fn bytes(&self) -> usize {
        let mut total = 0;
        for l in &self.layers {
            total += l.attn_norm.len() * 4;
            total += l.post_attention_norm.len() * 4;
            total += l.ffn_norm.len() * 4;
            total += l.post_ffw_norm.len() * 4;
            if let Some(ref p) = l.post_norm {
                total += p.len() * 4;
            }
            total += l.q_norm.len() * 4;
            total += l.k_norm.len() * 4;
            total += l.q_weight.bytes();
            total += l.k_weight.bytes();
            total += l.v_weight.bytes();
            total += l.o_weight.bytes();
            total += l.gate_weight.bytes();
            total += l.up_weight.bytes();
            total += l.down_weight.bytes();
        }
        total += self.per_layer_token_embd.len() * 4;
        total += self.per_layer_model_proj.as_ref().map(|p| p.bytes()).unwrap_or(0);
        total += self.per_layer_proj_norm.len() * 4;
        if let Some(ref r) = self.rope_freqs {
            total += r.len() * 4;
        }
        total
    }
}

impl InferenceEngine {
    /// Architecture string from GGUF metadata (e.g. "qwen2", "llama", "gemma4").
    /// Used by the HTTP layer to pick a chat template and by `stop_tokens()`
    /// to pick EOS IDs.
    pub fn architecture(&self) -> &str {
        &self.config.architecture
    }

    /// The model's `general.name` from the GGUF header. Used by the chat-
    /// template dispatch to distinguish fine-tunes that share a base
    /// architecture but use different chat tokens (e.g. R1-Distill-Qwen
    /// vs Qwen2-Instruct).
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }

    /// Enable TurboQuant Q8 KV cache. Must be called before the first
    /// `generate` / `generate_stream` invocation — the KV cache is
    /// lazily allocated on first decode step using whichever variant is
    /// configured.
    #[cfg(feature = "cuda")]
    pub fn set_kv_quant_q8(&mut self, enabled: bool) {
        self.kv_quant_q8 = enabled;
    }

    /// Stop-token IDs that terminate generation in `generate_stream`.
    ///
    /// Qwen2-family IDs 151643-151646 carry different meanings depending on
    /// which fine-tune's vocab is loaded:
    ///
    /// | ID     | Qwen2-Instruct     | R1-Distill-Qwen           |
    /// |--------|--------------------|---------------------------|
    /// | 151643 | `<|endoftext|>`    | `<｜end▁of▁sentence｜>`  |
    /// | 151644 | `<|im_start|>`     | `<｜User｜>`             |
    /// | 151645 | `<|im_end|>`       | `<｜Assistant｜>`        |
    /// | 151646 | (unused)           | `<｜begin▁of▁sentence｜>`|
    ///
    /// In both vocabs the IDs 151643-151646 signal either a terminator or a
    /// mid-turn template leak — stopping on any of them is correct. ID 0 is
    /// the pad token; kept for backward compat because Qwen never emits 0
    /// naturally but earlier runs depended on it.
    ///
    /// Gemma 3/4: 1 `<eos>`, 106 `<end_of_turn>`. Gemma's pad is 0 but pad is
    /// never emitted — we intentionally omit it so a spurious 0 doesn't end
    /// the turn.
    pub fn stop_tokens(&self) -> &'static [u32] {
        match self.config.architecture.as_str() {
            "gemma" | "gemma2" | "gemma3" | "gemma4" => &[1, 106],
            // BitNet b1.58 and Llama-3 family both use the LLaMA-3
            // tokenizer: 128000 BOS, 128001 `<|end_of_text|>`, 128009
            // `<|eot_id|>`. Include both EOS and EOT so the assistant
            // turn terminates cleanly.
            a if a.starts_with("bitnet") => &[128001, 128009],
            // Plain "llama" arch with a Llama-3-sized vocab (128256):
            // these are Llama-3/3.1/3.2 models plus their many community
            // fine-tunes. The older Llama-2-vocab (~32k) variant falls
            // through to the qwen2 fallback because it has no stop-token
            // convention encoded in the GGUF we can rely on.
            "llama" if self.config.vocab_size >= 128000
                   && self.config.vocab_size < 150000 => &[128001, 128009],
            _ => &[0, 151643, 151644, 151645, 151646],
        }
    }

    /// Move all weight matrices to GPU (f32 variant) and set the compute device.
    /// Quantized weights stay in CPU RAM (dequantization produces per-matmul
    /// scratch tensors that get moved to `compute_device`).
    pub fn to_device(&mut self, device: Device) {
        println!("  Setting compute device to {:?}...", device);
        self.compute_device = device.clone();
        self.lm_head.to_device(device.clone());

        if let Some(ref mut g) = self.gemma4 {
            // Gemma 4 path: layers live on `gemma4`, main `layers` is empty.
            g.to_device(device.clone());
        } else {
            // LLaMA / Qwen2 / Mistral path.
            let num_layers = self.layers.len();
            for (i, layer) in self.layers.iter_mut().enumerate() {
                layer.to_device(device.clone());
                if (i + 1) % 7 == 0 || i + 1 == num_layers {
                    println!("    Layer {}/{}", i + 1, num_layers);
                }
            }
        }
        println!("  Compute device: {:?}", device);
    }
}

/// Per-layer GPU-resident KV buffers. Pre-allocated to `capacity` tokens
/// at load time so decode steps only need a small H2D for the single new
/// K/V row (n_kv_heads × head_dim f32s) rather than re-uploading the
/// whole cache each step.
///
/// The buffers are overallocated (sized to the entire `capacity`) and the
/// attention kernel bounds its iteration with `kv_len`, so the extra
/// trailing space is never read. This avoids per-step reallocation and
/// keeps pointer arithmetic simple.
#[cfg(feature = "cuda")]
pub struct GpuKvCache {
    /// Per-layer K buffer, each [capacity * row_stride] f32s.
    pub k: Vec<cudarc::driver::CudaSlice<f32>>,
    /// Per-layer V buffer, same shape as K.
    pub v: Vec<cudarc::driver::CudaSlice<f32>>,
    /// Width of one token's K (or V) row: `n_kv_heads * head_dim`.
    pub row_stride: usize,
    /// Max tokens the buffer can hold before we fall back to host-only.
    pub capacity: usize,
}

#[cfg(feature = "cuda")]
impl GpuKvCache {
    /// Allocate per-layer GPU K/V buffers. Returns `None` if allocation
    /// fails for any layer (caller falls back to host-only KV cache).
    pub fn try_new(
        cuda: &axonml_core::backends::CudaBackend,
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Option<Self> {
        let row_stride = n_kv_heads * head_dim;
        let per_layer = capacity * row_stride;
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(cuda.alloc_uninit::<f32>(per_layer).ok()?);
            v.push(cuda.alloc_uninit::<f32>(per_layer).ok()?);
        }
        Some(Self { k, v, row_stride, capacity })
    }

    /// Copy a single token's K and V row into position `pos` of the
    /// given layer's GPU buffers. `k_row` and `v_row` must each be
    /// exactly `row_stride` f32s on the host. Returns `false` if the
    /// position exceeds pre-allocated capacity (caller must fall back).
    pub fn append_row(
        &mut self,
        cuda: &axonml_core::backends::CudaBackend,
        layer: usize,
        pos: usize,
        k_row: &[f32],
        v_row: &[f32],
    ) -> bool {
        if pos >= self.capacity
            || k_row.len() != self.row_stride
            || v_row.len() != self.row_stride
        {
            return false;
        }
        let offset = pos * self.row_stride;
        let end = offset + self.row_stride;
        let k_ok = {
            let mut view = self.k[layer].slice_mut(offset..end);
            cuda.stream().memcpy_htod(k_row, &mut view).is_ok()
        };
        if !k_ok {
            return false;
        }
        let v_ok = {
            let mut view = self.v[layer].slice_mut(offset..end);
            cuda.stream().memcpy_htod(v_row, &mut view).is_ok()
        };
        v_ok
    }

    /// Device-to-device variant of `append_row` that reads K and V straight
    /// from pre-existing GPU buffers. Used by the GPU-resident decode path
    /// to avoid the two ~14 KB D2H copies per layer that `append_row` would
    /// otherwise require. Returns `false` if the position exceeds capacity
    /// or either copy fails (caller falls back to the host-side path).
    pub fn append_row_device(
        &mut self,
        cuda: &axonml_core::backends::CudaBackend,
        layer: usize,
        pos: usize,
        k_row: &cudarc::driver::CudaSlice<f32>,
        v_row: &cudarc::driver::CudaSlice<f32>,
    ) -> bool {
        if pos >= self.capacity
            || k_row.len() != self.row_stride
            || v_row.len() != self.row_stride
        {
            return false;
        }
        let offset = pos * self.row_stride;
        let end = offset + self.row_stride;
        let k_ok = {
            let mut view = self.k[layer].slice_mut(offset..end);
            cuda.stream().memcpy_dtod(k_row, &mut view).is_ok()
        };
        if !k_ok {
            return false;
        }
        let v_ok = {
            let mut view = self.v[layer].slice_mut(offset..end);
            cuda.stream().memcpy_dtod(v_row, &mut view).is_ok()
        };
        v_ok
    }
}

/// TurboQuant Q8 GPU-resident KV cache.
///
/// Per-layer storage: int8 quantized values plus one f32 scale per
/// `(token, kv_head)` — i.e. the quantization granularity is per-head,
/// not per-row. For DeepSeek-7B (n_kv_heads=4, head_dim=128) each row
/// shrinks from `4 * 128 * 4 = 2048` bytes of f32 to `4 * 128 = 512` int8
/// bytes plus `4 * 4 = 16` scale bytes = 528 bytes, a ~3.9× compression
/// that halves the attention kernel's KV read bandwidth and lets a 4096-
/// token context sit in ~115 MB instead of ~450 MB.
///
/// Layout per layer:
///   k_q[layer]     : [capacity, n_kv_heads, head_dim]  int8
///   k_scale[layer] : [capacity, n_kv_heads]            f32
///   (same for V)
#[cfg(feature = "cuda")]
pub struct GpuKvCacheQ8 {
    pub k_q: Vec<cudarc::driver::CudaSlice<i8>>,
    pub k_scale: Vec<cudarc::driver::CudaSlice<f32>>,
    pub v_q: Vec<cudarc::driver::CudaSlice<i8>>,
    pub v_scale: Vec<cudarc::driver::CudaSlice<f32>>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub row_stride: usize,
    pub capacity: usize,
}

#[cfg(feature = "cuda")]
impl GpuKvCacheQ8 {
    pub fn try_new(
        cuda: &axonml_core::backends::CudaBackend,
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Option<Self> {
        // Both the quantized values AND the per-head scales must be zero-
        // initialized. Reads for positions the decoder hasn't written yet
        // (the prefill region, which the GPU cache is lazily allocated
        // behind) must yield 0 for all four of {k_q, k_scale, v_q, v_scale}
        // so the attention kernel's dequant (int8 * scale) is 0 on those
        // positions. Leaving scale uninitialized blows up exp() in online
        // softmax the moment a stray huge float32 lands in the buffer.
        let row_stride = n_kv_heads * head_dim;
        let quant_bytes = capacity * row_stride;
        let scale_floats = capacity * n_kv_heads;
        let mut k_q = Vec::with_capacity(num_layers);
        let mut k_scale = Vec::with_capacity(num_layers);
        let mut v_q = Vec::with_capacity(num_layers);
        let mut v_scale = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_q.push(cuda.stream().alloc_zeros::<i8>(quant_bytes).ok()?);
            k_scale.push(cuda.stream().alloc_zeros::<f32>(scale_floats).ok()?);
            v_q.push(cuda.stream().alloc_zeros::<i8>(quant_bytes).ok()?);
            v_scale.push(cuda.stream().alloc_zeros::<f32>(scale_floats).ok()?);
        }
        Some(Self {
            k_q,
            k_scale,
            v_q,
            v_scale,
            n_kv_heads,
            head_dim,
            row_stride,
            capacity,
        })
    }

    /// Quantize one token's K and V rows (held on-device as f32 GPU slices)
    /// into the int8 cache at position `pos`. Two kernel launches: one for
    /// K, one for V. Both write in-place; no host hop.
    pub fn append_row_device(
        &mut self,
        cuda: &axonml_core::backends::CudaBackend,
        layer: usize,
        pos: usize,
        k_row: &cudarc::driver::CudaSlice<f32>,
        v_row: &cudarc::driver::CudaSlice<f32>,
    ) -> bool {
        if pos >= self.capacity
            || k_row.len() != self.row_stride
            || v_row.len() != self.row_stride
        {
            return false;
        }
        let k_ok = cuda
            .quantize_kv_row_q8_f32(
                k_row,
                &mut self.k_q[layer],
                &mut self.k_scale[layer],
                self.n_kv_heads,
                self.head_dim,
                pos,
            )
            .is_ok();
        if !k_ok {
            return false;
        }
        cuda.quantize_kv_row_q8_f32(
            v_row,
            &mut self.v_q[layer],
            &mut self.v_scale[layer],
            self.n_kv_heads,
            self.head_dim,
            pos,
        )
        .is_ok()
    }
}

/// KV cache for incremental decoding.
///
/// Holds two parallel representations: a host-resident `Vec<Vec<f32>>`
/// (always maintained, used by the CPU attention fallback and still
/// populated on CUDA builds for correctness/debuggability), and an
/// optional GPU-resident [`GpuKvCache`] that the CUDA decode path reads
/// from directly. When the GPU cache is present, the decode attention
/// step skips the per-call htod_copy of the full K/V history — it just
/// reads from the pre-allocated GPU buffers with only a single
/// row-sized H2D per token (performed in [`GpuKvCache::append_row`]).
pub struct KvCache {
    /// Per-layer key cache: Vec<f32> growing as [position, n_kv_heads * head_dim]
    pub k_cache: Vec<Vec<f32>>,
    /// Per-layer value cache
    pub v_cache: Vec<Vec<f32>>,
    /// Number of cached positions
    pub len: usize,
    /// GPU-resident mirror of K/V, lazily allocated on the first forward
    /// that has a CUDA backend available. None on CPU-only builds or if
    /// allocation failed (we then fall back to re-uploading per step).
    #[cfg(feature = "cuda")]
    pub gpu: Option<GpuKvCache>,
    /// TurboQuant Q8 variant. Mutually exclusive with `gpu`: whichever
    /// variant the engine is configured for (via `InferenceEngine.kv_quant_q8`)
    /// gets lazily allocated, the other stays `None`.
    #[cfg(feature = "cuda")]
    pub gpu_q8: Option<GpuKvCacheQ8>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k_cache: (0..num_layers).map(|_| Vec::new()).collect(),
            v_cache: (0..num_layers).map(|_| Vec::new()).collect(),
            len: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "cuda")]
            gpu_q8: None,
        }
    }

    pub fn clear(&mut self) {
        for k in &mut self.k_cache {
            k.clear();
        }
        for v in &mut self.v_cache {
            v.clear();
        }
        self.len = 0;
        // Leave the GPU buffers allocated — they'll be overwritten on the
        // next forward pass. Dropping + reallocating on every clear would
        // defeat the whole point of pre-allocation.
    }

    /// Truncate the cache to `new_len` positions. Used by speculative
    /// decoding to roll back rejected draft tokens' K/V rows after a
    /// verify step. The GPU buffers are over-allocated to capacity and
    /// unused rows are overwritten on next append, so we only need to
    /// shorten the host mirror vectors and update `len`.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        for (li, k) in self.k_cache.iter_mut().enumerate() {
            let row_stride = if self.v_cache[li].is_empty() {
                0
            } else {
                self.v_cache[li].len() / self.len.max(1)
            };
            let target = new_len * row_stride;
            if k.len() > target {
                k.truncate(target);
            }
        }
        for v in &mut self.v_cache {
            let row_stride = if self.len == 0 { 0 } else { v.len() / self.len };
            let target = new_len * row_stride;
            if v.len() > target {
                v.truncate(target);
            }
        }
        self.len = new_len;
    }
}

impl InferenceEngine {
    /// Load a model from GGUF, fully dequantizing to f32 (fast, big).
    pub fn from_gguf(gguf: &GgufFile, mapped: &MappedGguf) -> Result<Self, String> {
        Self::from_gguf_with_mode(gguf, mapped, false)
    }

    /// Load a model from GGUF. If `quantized_weights` is true, weight matrices
    /// are kept in their compact GGUF form and dequantized per-matmul (saves ~5x RAM,
    /// at the cost of inference speed). Norms and the token embedding are always
    /// dequantized eagerly since they're used on every token.
    pub fn from_gguf_with_mode(
        gguf: &GgufFile,
        mapped: &MappedGguf,
        quantized_weights: bool,
    ) -> Result<Self, String> {
        let config = InferenceConfig::from_gguf(gguf);

        println!("  Loading weights (mode: {}) ...",
            if quantized_weights { "quantized (lazy dequant)" } else { "f32 (eager dequant)" });
        println!("    Architecture: {}", config.architecture);
        println!("    Hidden: {}, Layers: {}, Heads: {}/{}",
            config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads);
        println!("    Vocab: {}, Context: {}", config.vocab_size, config.max_seq_len);

        // Dispatch on architecture. Gemma models have a distinct per-layer
        // layout (sandwich norms, Q/K norm, per-layer input embeddings) and
        // the top-level tensors (vision/audio towers) must be skipped.
        if config.architecture.starts_with("gemma") {
            return Self::load_gemma4(gguf, mapped, quantized_weights, config);
        }

        // Token embeddings as flat Vec (fast lookup) — always dequantized
        let token_embed = load_vec(mapped, "token_embd.weight")?;

        // Load per-layer weights
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("blk.{i}");

            let attn_norm = load_vec(mapped, &format!("{prefix}.attn_norm.weight"))?;

            let q_weight = load_weight(mapped, &format!("{prefix}.attn_q.weight"), quantized_weights)?;
            let k_weight = load_weight(mapped, &format!("{prefix}.attn_k.weight"), quantized_weights)?;
            let v_weight = load_weight(mapped, &format!("{prefix}.attn_v.weight"), quantized_weights)?;
            let o_weight = load_weight(mapped, &format!("{prefix}.attn_output.weight"), quantized_weights)?;

            let ffn_norm = load_vec(mapped, &format!("{prefix}.ffn_norm.weight"))?;
            let gate_weight = load_weight(mapped, &format!("{prefix}.ffn_gate.weight"), quantized_weights)?;
            let up_weight = load_weight(mapped, &format!("{prefix}.ffn_up.weight"), quantized_weights)?;
            let down_weight = load_weight(mapped, &format!("{prefix}.ffn_down.weight"), quantized_weights)?;

            let q_bias = try_load_vec(mapped, &format!("{prefix}.attn_q.bias"));
            let k_bias = try_load_vec(mapped, &format!("{prefix}.attn_k.bias"));
            let v_bias = try_load_vec(mapped, &format!("{prefix}.attn_v.bias"));

            // BitNet b1.58 sub-norms — present only on BitNet GGUFs.
            // try_load_vec returns None for other architectures, and the
            // forward pass gates on `Option` so LLaMA/Qwen2/Mistral stay
            // bit-identical.
            let attn_sub_norm = try_load_vec(mapped, &format!("{prefix}.attn_sub_norm.weight"));
            let ffn_sub_norm = try_load_vec(mapped, &format!("{prefix}.ffn_sub_norm.weight"));

            // Qwen3 QK-norm — present only on Qwen3 GGUFs. Shape: [head_dim].
            // Same [head_dim] weight is broadcast per-head to Q and K before
            // RoPE. Absent on every other architecture we support; the forward
            // pass gates on Option so other arches stay unaffected.
            let attn_q_norm = try_load_vec(mapped, &format!("{prefix}.attn_q_norm.weight"));
            let attn_k_norm = try_load_vec(mapped, &format!("{prefix}.attn_k_norm.weight"));

            layers.push(LayerWeights {
                attn_norm, q_weight, k_weight, v_weight, o_weight,
                ffn_norm, gate_weight, up_weight, down_weight,
                q_bias, k_bias, v_bias,
                attn_sub_norm, ffn_sub_norm,
                attn_q_norm, attn_k_norm,
            });

            if (i + 1) % 7 == 0 || i + 1 == config.num_layers {
                println!("    Loaded layer {}/{}", i + 1, config.num_layers);
            }
        }

        // Output norm — always dequantized (always used)
        let output_norm = load_vec(mapped, "output_norm.weight")?;

        // LM head: need [hidden, vocab] for input @ lm_head = [seq, vocab]
        let lm_head = if mapped.has_tensor("output.weight") {
            load_weight(mapped, "output.weight", quantized_weights)?
        } else {
            println!("    LM head tied to token embeddings");
            load_weight(mapped, "token_embd.weight", quantized_weights)?
        };

        let total_bytes: usize = token_embed.len() * 4
            + layers.iter().map(|l| {
                l.attn_norm.len() * 4
                + l.q_weight.bytes()
                + l.k_weight.bytes()
                + l.v_weight.bytes()
                + l.o_weight.bytes()
                + l.ffn_norm.len() * 4
                + l.gate_weight.bytes()
                + l.up_weight.bytes()
                + l.down_weight.bytes()
            }).sum::<usize>()
            + output_norm.len() * 4
            + lm_head.bytes();

        println!("  Model loaded: {:.2} GB in RAM ({} mode)",
            total_bytes as f64 / 1e9,
            if quantized_weights { "quantized" } else { "f32" });

        Ok(Self {
            config,
            token_embed,
            layers,
            gemma4: None,
            output_norm,
            lm_head,
            compute_device: Device::Cpu,
            #[cfg(feature = "cuda")]
            gpu_layer_cache: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_output_norm: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            kv_quant_q8: false,
        })
    }

    /// Gemma 4 weight loader. Called from `from_gguf_with_mode` when the
    /// architecture is a Gemma variant. Skips the vision (`v.*`) and audio
    /// (`a.*`) towers — nexus-serve is text-only for this round.
    fn load_gemma4(
        _gguf: &GgufFile,
        mapped: &MappedGguf,
        quantized_weights: bool,
        config: InferenceConfig,
    ) -> Result<Self, String> {
        // Token embedding: [vocab, hidden]. Always eagerly dequantized.
        let token_embed = load_vec(mapped, "token_embd.weight")?;

        // Per-layer weights: load the 7 Gemma-specific norms + Q/K/V/O + gate/up/down
        let mut gemma_layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("blk.{i}");

            let attn_norm = load_vec(mapped, &format!("{prefix}.attn_norm.weight"))?;
            let post_attention_norm = load_vec(
                mapped, &format!("{prefix}.post_attention_norm.weight"))?;
            let ffn_norm = load_vec(mapped, &format!("{prefix}.ffn_norm.weight"))?;
            let post_ffw_norm = load_vec(mapped, &format!("{prefix}.post_ffw_norm.weight"))?;
            // post_norm is present on Gemma 4 but its exact role is still being
            // verified against reference code — load as optional.
            let post_norm = try_load_vec(mapped, &format!("{prefix}.post_norm.weight"));
            let q_norm = load_vec(mapped, &format!("{prefix}.attn_q_norm.weight"))?;
            let k_norm = load_vec(mapped, &format!("{prefix}.attn_k_norm.weight"))?;

            let q_weight = load_weight(mapped, &format!("{prefix}.attn_q.weight"), quantized_weights)?;
            let k_weight = load_weight(mapped, &format!("{prefix}.attn_k.weight"), quantized_weights)?;
            let v_weight = load_weight(mapped, &format!("{prefix}.attn_v.weight"), quantized_weights)?;
            let o_weight = load_weight(mapped, &format!("{prefix}.attn_output.weight"), quantized_weights)?;

            let gate_weight = load_weight(mapped, &format!("{prefix}.ffn_gate.weight"), quantized_weights)?;
            let up_weight = load_weight(mapped, &format!("{prefix}.ffn_up.weight"), quantized_weights)?;
            let down_weight = load_weight(mapped, &format!("{prefix}.ffn_down.weight"), quantized_weights)?;

            gemma_layers.push(Gemma4LayerWeights {
                attn_norm,
                post_attention_norm,
                ffn_norm,
                post_ffw_norm,
                post_norm,
                q_norm,
                k_norm,
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                gate_weight,
                up_weight,
                down_weight,
            });

            if (i + 1) % 7 == 0 || i + 1 == config.num_layers {
                println!("    Loaded layer {}/{}", i + 1, config.num_layers);
            }
        }

        // Top-level: per-layer input embeddings — the altup mechanism used by
        // Gemma 3n / Gemma 4 E-series ("Effective" compressed variants).
        // Standard Gemma 3 sizes (1B/4B/12B/27B) don't have altup and omit
        // these tensors entirely. Detect by presence rather than by arch name
        // so one loader handles both families.
        let has_altup = mapped.has_tensor("per_layer_token_embd.weight");
        let per_layer_token_embd = if has_altup {
            load_vec(mapped, "per_layer_token_embd.weight")?
        } else {
            Vec::new()
        };
        let per_layer_model_proj = if has_altup {
            Some(load_weight(mapped, "per_layer_model_proj.weight", quantized_weights)?)
        } else {
            None
        };
        let per_layer_proj_norm = if has_altup {
            load_vec(mapped, "per_layer_proj_norm.weight")?
        } else {
            Vec::new()
        };
        if has_altup {
            println!(
                "    Altup: token_embd={} proj.shape={:?} proj_norm={} (width={}, layers={})",
                per_layer_token_embd.len(),
                per_layer_model_proj.as_ref().map(|w| w.shape()),
                per_layer_proj_norm.len(),
                config.gemma.as_ref().map(|g| g.per_layer_input_width).unwrap_or(0),
                config.num_layers,
            );
        } else {
            println!("    No altup — standard Gemma architecture (non-E-series)");
        }

        // Optional pre-computed RoPE frequency table. We'll fall back to
        // computing RoPE freqs analytically if this isn't present.
        let rope_freqs = try_load_vec(mapped, "rope_freqs.weight");

        // Output norm — always eager.
        let output_norm = load_vec(mapped, "output_norm.weight")?;

        // LM head: Gemma 4 ties to `token_embd.weight` (no standalone
        // `output.weight`). Fall back to token_embd if not present.
        let lm_head = if mapped.has_tensor("output.weight") {
            load_weight(mapped, "output.weight", quantized_weights)?
        } else {
            println!("    LM head tied to token embeddings");
            load_weight(mapped, "token_embd.weight", quantized_weights)?
        };

        // Sanity-check: count skipped vision/audio tensors so the user sees
        // what was filtered out.
        let skipped_v = mapped.tensor_names().iter().filter(|n| n.starts_with("v.")).count();
        let skipped_a = mapped.tensor_names().iter().filter(|n| n.starts_with("a.") || n.starts_with("mm.")).count();
        if skipped_v + skipped_a > 0 {
            println!("    Skipped {} vision + {} audio/multimodal tensors (text-only serving)",
                skipped_v, skipped_a);
        }

        let gemma = Gemma4Weights {
            layers: gemma_layers,
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            rope_freqs,
        };

        let total_bytes = token_embed.len() * 4
            + gemma.bytes()
            + output_norm.len() * 4
            + lm_head.bytes();

        println!("  Model loaded: {:.2} GB in RAM ({} mode, gemma4 path)",
            total_bytes as f64 / 1e9,
            if quantized_weights { "quantized" } else { "f32" });

        Ok(Self {
            config,
            token_embed,
            layers: Vec::new(), // LLaMA-family layers unused in Gemma path
            gemma4: Some(gemma),
            output_norm,
            lm_head,
            compute_device: Device::Cpu,
            #[cfg(feature = "cuda")]
            gpu_layer_cache: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_output_norm: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            kv_quant_q8: false,
        })
    }

    /// Generate tokens with KV cache for fast incremental decoding.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Vec<u32> {
        let mut out = Vec::with_capacity(max_new_tokens);
        self.generate_stream(prompt_ids, max_new_tokens, temperature, top_p, |tok| {
            out.push(tok);
            true // keep going
        });
        out
    }

    /// Generate tokens with a per-token callback for streaming.
    ///
    /// The callback receives each new token as it is produced. Return `true`
    /// from the callback to continue generation, or `false` to stop early
    /// (e.g., on client disconnect).
    pub fn generate_stream<F: FnMut(u32) -> bool>(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        mut on_token: F,
    ) {
        let is_gemma = self.gemma4.is_some();
        let mut kv_cache = KvCache::new(self.config.num_layers);

        // Prefill.
        //
        // For the LLaMA / Qwen / BitNet family we feed prompt tokens
        // through `forward_one` (which routes through
        // `forward_one_gpu_resident` on CUDA). This is slower per
        // prompt token than `forward_batch` (higher first-token
        // latency), but it's the ONLY way to keep the prefill K/V
        // numerically identical to what decode produces — the
        // `forward_batch` path walks activations through CPU between
        // kernel boundaries, and the resulting K/V drifts enough from
        // the GPU-native path that decode's attention locks onto
        // spurious positions (this was the root cause of Qwen3-4B
        // degeneracy and Llama-3.2-3B looping on greedy). Gemma keeps
        // its dedicated `forward_batch_gemma4` since its prefill has
        // architecture-specific plumbing (altup, SWA pattern).
        let logits = if is_gemma {
            self.forward_batch_gemma4(prompt_ids, &mut kv_cache)
        } else if prompt_ids.is_empty() {
            // Nothing to prefill; bail out before decode.
            return;
        } else {
            let mut last: Vec<f32> = Vec::new();
            for &t in prompt_ids {
                last = self.forward_one(t, &mut kv_cache);
            }
            last
        };
        let vocab_size = self.config.vocab_size;
        let last_logits = &logits[logits.len() - vocab_size..];

        let stop = self.stop_tokens();

        let mut next_id = if temperature < 0.01 {
            argmax(last_logits) as u32
        } else {
            sample_top_p(last_logits, temperature, top_p)
        };

        if stop.contains(&next_id) {
            return;
        }
        if !on_token(next_id) {
            return;
        }

        // Decode: process one token at a time with KV cache
        for _ in 1..max_new_tokens {
            let logits = if is_gemma {
                self.forward_one_gemma4(next_id, &mut kv_cache)
            } else {
                self.forward_one(next_id, &mut kv_cache)
            };

            next_id = if temperature < 0.01 {
                argmax(&logits) as u32
            } else {
                sample_top_p(&logits, temperature, top_p)
            };

            if stop.contains(&next_id) {
                break;
            }
            if !on_token(next_id) {
                break;
            }
        }
    }

    /// The device where matmul runs. Inputs are moved here before each matmul.
    fn weight_device(&self) -> Device {
        self.compute_device.clone()
    }

    /// Move a CPU tensor to the weight device (GPU if available).
    fn to_weight_device(&self, t: Tensor<f32>) -> Tensor<f32> {
        let dev = self.weight_device();
        if dev == t.device() {
            t
        } else {
            t.to_device(dev).unwrap_or(t)
        }
    }

    /// Forward pass for multiple tokens (prefill). Updates KV cache.
    pub fn forward_batch(&self, token_ids: &[u32], kv_cache: &mut KvCache) -> Vec<f32> {
        let seq_len = token_ids.len();
        let hidden = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.num_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let kv_dim = n_kv_heads * head_dim;

        // Embedding lookup
        let mut x = vec![0.0f32; seq_len * hidden];
        for (pos, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < self.config.vocab_size {
                x[pos * hidden..(pos + 1) * hidden]
                    .copy_from_slice(&self.token_embed[id * hidden..(id + 1) * hidden]);
            }
        }

        let pos_offset = kv_cache.len;

        for (li, layer) in self.layers.iter().enumerate() {
            // RMS Norm (CPU — element-wise, fast)
            let normed = rms_norm_vec(&x, &layer.attn_norm, self.config.rms_norm_eps, seq_len, hidden);

            // Move to weight device for matmul
            let normed_t = self.to_weight_device(Tensor::from_vec(normed, &[seq_len, hidden]).unwrap());

            // QKV projections — matmul on weight device (GPU if available)
            let mut q_data = layer.q_weight.matmul(&normed_t).to_vec();
            let k_data = layer.k_weight.matmul(&normed_t).to_vec();
            let mut v_data_new = layer.v_weight.matmul(&normed_t).to_vec();
            let mut k_data_new = k_data.clone();

            // Biases — Qwen2 has Q, K, and V biases
            if let Some(ref b) = layer.q_bias { add_bias(&mut q_data, b, seq_len, n_heads * head_dim); }
            if let Some(ref b) = layer.k_bias { add_bias(&mut k_data_new, b, seq_len, kv_dim); }
            if let Some(ref b) = layer.v_bias { add_bias(&mut v_data_new, b, seq_len, kv_dim); }

            // Qwen3 QK-norm — per-head RMSNorm before RoPE. None-gated so
            // architectures without QK-norm (Qwen2, Gemma, LLaMA, …) stay
            // bit-identical.
            if let Some(ref qn) = layer.attn_q_norm {
                apply_rms_norm_heads_inplace(
                    &mut q_data, qn, seq_len, n_heads, head_dim, self.config.rms_norm_eps,
                );
            }
            if let Some(ref kn) = layer.attn_k_norm {
                apply_rms_norm_heads_inplace(
                    &mut k_data_new, kn, seq_len, n_kv_heads, head_dim, self.config.rms_norm_eps,
                );
            }

            // RoPE
            apply_rope_inplace(&mut q_data, seq_len, n_heads, head_dim, self.config.rope_theta, pos_offset);
            apply_rope_inplace(&mut k_data_new, seq_len, n_kv_heads, head_dim, self.config.rope_theta, pos_offset);

            // Append to KV cache
            kv_cache.k_cache[li].extend_from_slice(&k_data_new);
            kv_cache.v_cache[li].extend_from_slice(&v_data_new);

            // Attention with full KV cache. GPU-accelerated via the fused
            // decode kernel (one launch per query row, causal masking via
            // kv_len clamping, Q/K/V uploaded once and reused). Falls back
            // to the CPU triple loop on non-CUDA builds.
            let total_len = kv_cache.len + seq_len;
            let attn_out = gpu_cached_attention(
                &q_data, &kv_cache.k_cache[li], &kv_cache.v_cache[li],
                seq_len, total_len, n_heads, n_kv_heads, head_dim, pos_offset, 0,
            );
            // BitNet b1.58 attn sub-norm (pre output projection).
            let attn_out = if let Some(ref sub) = layer.attn_sub_norm {
                rms_norm_vec(&attn_out, sub, self.config.rms_norm_eps, seq_len, n_heads * head_dim)
            } else {
                attn_out
            };
            let attn_t = self.to_weight_device(Tensor::from_vec(attn_out, &[seq_len, n_heads * head_dim]).unwrap());

            // Output projection (GPU matmul)
            let attn_proj = layer.o_weight.matmul(&attn_t).to_vec();

            // Residual
            for i in 0..x.len() { x[i] += attn_proj[i]; }

            // FFN
            let normed2 = rms_norm_vec(&x, &layer.ffn_norm, self.config.rms_norm_eps, seq_len, hidden);
            let normed2_t = self.to_weight_device(Tensor::from_vec(normed2, &[seq_len, hidden]).unwrap());

            let gate_data = layer.gate_weight.matmul(&normed2_t).to_vec();
            let up_data = layer.up_weight.matmul(&normed2_t).to_vec();

            // FFN activation — architecture-dependent:
            //   - LLaMA/Qwen/Mistral: SwiGLU → SiLU(gate) * up
            //   - BitNet b1.58:       ReLU² gated → max(0, gate)² * up
            // BitNet's HF model card is explicit: "squared ReLU (ReLU²)
            // activation in FFN layers". Switching on architecture avoids
            // regressing the LLaMA-family models.
            let inter_size = self.config.intermediate_size;
            let mut ffn_data: Vec<f32> = vec![0.0f32; seq_len * inter_size];
            let is_bitnet = self.config.architecture.starts_with("bitnet");
            if is_bitnet {
                ffn_data
                    .par_iter_mut()
                    .zip(gate_data.par_iter().zip(up_data.par_iter()))
                    .for_each(|(out, (&g, &u))| {
                        let r = g.max(0.0);
                        *out = r * r * u;
                    });
            } else {
                ffn_data
                    .par_iter_mut()
                    .zip(gate_data.par_iter().zip(up_data.par_iter()))
                    .for_each(|(out, (&g, &u))| {
                        *out = (g / (1.0 + (-g).exp())) * u;
                    });
            }
            // BitNet b1.58 FFN sub-norm (pre down projection).
            let ffn_data = if let Some(ref sub) = layer.ffn_sub_norm {
                rms_norm_vec(&ffn_data, sub, self.config.rms_norm_eps, seq_len, inter_size)
            } else {
                ffn_data
            };
            let ffn_t = self.to_weight_device(Tensor::from_vec(ffn_data, &[seq_len, inter_size]).unwrap());
            let ffn_out = layer.down_weight.matmul(&ffn_t).to_vec();

            for i in 0..x.len() { x[i] += ffn_out[i]; }
        }

        kv_cache.len += seq_len;

        // Final norm + LM head
        let normed = rms_norm_vec(&x, &self.output_norm, self.config.rms_norm_eps, seq_len, hidden);
        let normed_t = self.to_weight_device(Tensor::from_vec(normed, &[seq_len, hidden]).unwrap());
        self.lm_head.matmul(&normed_t).to_vec()
    }

    /// Forward pass for a SINGLE token using KV cache (decode step).
    /// This is the hot path — only processes 1 token, reuses cached K/V.
    ///
    /// Two paths:
    /// - `forward_one_gpu_resident` when CUDA is available and the compute
    ///   device is a GPU. Keeps the hidden state `x` as a `Tensor<f32>` on
    ///   GPU through the whole layer loop, using the rms_norm / rope / swiglu /
    ///   relu2_gate CUDA kernels added in axonml-core/transformer_ops.cu.
    ///   Eliminates ~10 of 14 per-layer host↔device round trips.
    /// - CPU path below — unchanged; still used when the engine hasn't been
    ///   moved to GPU.
    pub fn forward_one(&self, token_id: u32, kv_cache: &mut KvCache) -> Vec<f32> {
        #[cfg(feature = "cuda")]
        if self.compute_device.is_gpu() {
            return self.forward_one_gpu_resident(token_id, kv_cache);
        }
        let hidden = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.num_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let _kv_dim = n_kv_heads * head_dim;
        let pos = kv_cache.len;

        // Embedding lookup — single token
        let mut x = vec![0.0f32; hidden];
        let id = token_id as usize;
        if id < self.config.vocab_size {
            x.copy_from_slice(&self.token_embed[id * hidden..(id + 1) * hidden]);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // RMS Norm (single position, CPU)
            let normed = rms_norm_single(&x, &layer.attn_norm, self.config.rms_norm_eps);
            let normed_t = self.to_weight_device(Tensor::from_vec(normed, &[1, hidden]).unwrap());

            // QKV — matmul on weight device (GPU)
            let mut q_data = layer.q_weight.matmul(&normed_t).to_vec();
            let mut k_data = layer.k_weight.matmul(&normed_t).to_vec();
            let mut v_data = layer.v_weight.matmul(&normed_t).to_vec();

            // Qwen2 has Q, K, and V biases
            if let Some(ref b) = layer.q_bias { for i in 0..q_data.len() { q_data[i] += b[i]; } }
            if let Some(ref b) = layer.k_bias { for i in 0..k_data.len() { k_data[i] += b[i]; } }
            if let Some(ref b) = layer.v_bias { for i in 0..v_data.len() { v_data[i] += b[i]; } }

            // RoPE at current position
            apply_rope_single(&mut q_data, n_heads, head_dim, self.config.rope_theta, pos);
            apply_rope_single(&mut k_data, n_kv_heads, head_dim, self.config.rope_theta, pos);

            // Append to KV cache
            kv_cache.k_cache[li].extend_from_slice(&k_data);
            kv_cache.v_cache[li].extend_from_slice(&v_data);

            // Lazy-allocate the GPU-resident mirror on first layer of the
            // first forward pass, and copy this token's new K/V row into
            // it at position `pos`. Eliminates the per-call re-upload of
            // the entire KV history that dominated decode latency.
            #[cfg(feature = "cuda")]
            let gpu_cache_ref = {
                if kv_cache.gpu.is_none() {
                    if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                        // Cap at 4096 tokens regardless of model's advertised
                        // context_length — a 131072 Qwen context would need
                        // 131072 × 4 × 128 × 4B × 28 layers × 2 ≈ 15 GB just
                        // for KV, which is impractical. 4096 is comfortable
                        // for agent use and fits alongside the model in 12 GB.
                        let cap = self.config.max_seq_len.min(4096);
                        kv_cache.gpu = GpuKvCache::try_new(cuda, self.layers.len(), n_kv_heads, head_dim, cap);
                    }
                }
                let mut r: Option<(&GpuKvCache, usize)> = None;
                if let Some(ref mut g) = kv_cache.gpu {
                    if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                        if g.append_row(cuda, li, pos, &k_data, &v_data) {
                            r = Some((&*g, li));
                        }
                    }
                }
                r
            };

            // Attention: query [1] against all cached [pos+1] K/V. Dispatches
            // to the fused GPU flash-decode kernel when CUDA is available;
            // falls back to the CPU triple loop otherwise.
            let total_len = pos + 1;
            let attn_out = gpu_single_query_attention(
                &q_data, &kv_cache.k_cache[li], &kv_cache.v_cache[li],
                total_len, n_heads, n_kv_heads, head_dim, 0,
                #[cfg(feature = "cuda")]
                gpu_cache_ref,
            );
            // BitNet b1.58 attn sub-norm (pre output projection).
            let attn_out = if let Some(ref sub) = layer.attn_sub_norm {
                rms_norm_single(&attn_out, sub, self.config.rms_norm_eps)
            } else {
                attn_out
            };

            let attn_t = self.to_weight_device(Tensor::from_vec(attn_out, &[1, n_heads * head_dim]).unwrap());
            let attn_proj = layer.o_weight.matmul(&attn_t).to_vec();

            for i in 0..hidden { x[i] += attn_proj[i]; }

            // FFN
            let normed2 = rms_norm_single(&x, &layer.ffn_norm, self.config.rms_norm_eps);
            let normed2_t = self.to_weight_device(Tensor::from_vec(normed2, &[1, hidden]).unwrap());

            let gate_data = layer.gate_weight.matmul(&normed2_t).to_vec();
            let up_data = layer.up_weight.matmul(&normed2_t).to_vec();

            let inter_size = self.config.intermediate_size;
            let mut ffn_data: Vec<f32> = vec![0.0f32; inter_size];
            let is_bitnet = self.config.architecture.starts_with("bitnet");
            if is_bitnet {
                // BitNet: ReLU²(gate) * up
                ffn_data
                    .par_iter_mut()
                    .zip(gate_data.par_iter().zip(up_data.par_iter()))
                    .for_each(|(out, (&g, &u))| {
                        let r = g.max(0.0);
                        *out = r * r * u;
                    });
            } else {
                // LLaMA/Qwen/Mistral: SwiGLU → SiLU(gate) * up
                ffn_data
                    .par_iter_mut()
                    .zip(gate_data.par_iter().zip(up_data.par_iter()))
                    .for_each(|(out, (&g, &u))| {
                        *out = (g / (1.0 + (-g).exp())) * u;
                    });
            }
            // BitNet b1.58 FFN sub-norm (pre down projection).
            let ffn_data = if let Some(ref sub) = layer.ffn_sub_norm {
                rms_norm_single(&ffn_data, sub, self.config.rms_norm_eps)
            } else {
                ffn_data
            };
            let ffn_t = self.to_weight_device(Tensor::from_vec(ffn_data, &[1, inter_size]).unwrap());
            let ffn_out = layer.down_weight.matmul(&ffn_t).to_vec();

            for i in 0..hidden { x[i] += ffn_out[i]; }
        }

        kv_cache.len += 1;

        // Final norm + LM head
        let normed = rms_norm_single(&x, &self.output_norm, self.config.rms_norm_eps);
        if std::env::var("BITNET_DEBUG").is_ok() {
            let n_nan = normed.iter().filter(|v| v.is_nan()).count();
            let n_inf = normed.iter().filter(|v| v.is_infinite()).count();
            let max = normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = normed.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean = normed.iter().sum::<f32>() / normed.len() as f32;
            eprintln!("[DBG pre-lm_head] n={} nan={n_nan} inf={n_inf} min={min:.4} max={max:.4} mean={mean:.4}",
                normed.len());
        }
        let normed_t = self.to_weight_device(Tensor::from_vec(normed, &[1, hidden]).unwrap());
        let logits = self.lm_head.matmul(&normed_t).to_vec();
        if std::env::var("BITNET_DEBUG").is_ok() {
            let n_nan = logits.iter().filter(|v| v.is_nan()).count();
            let n_inf = logits.iter().filter(|v| v.is_infinite()).count();
            let max_idx = logits.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(i, m), (j, &v)| if v > m { (j, v) } else { (i, m) });
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            // Top-5 logits
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<String> = idx.iter().take(5).map(|&i| format!("{i}:{:.3}", logits[i])).collect();
            eprintln!("[DBG logits] n={} nan={n_nan} inf={n_inf} min={min:.4} max={max:.4} top5={}",
                logits.len(), top.join(", "));
            eprintln!("[DBG]  argmax={} logit={:.4}", max_idx.0, max_idx.1);
        }
        logits
    }

    // =========================================================================
    // Gemma 4 forward pass
    //
    // Differences from forward_batch / forward_one above:
    //   - Each layer has its own head_dim (512 for full-attention layers,
    //     256 for SWA layers, controlled by sliding_window_pattern).
    //   - Each layer picks its RoPE base (1M for full, 10k for SWA).
    //   - Attention is preceded by two RMSNorms (pre-attn on hidden, then
    //     per-head Q/K RMSNorm pre-RoPE).
    //   - Attention output is followed by a second RMSNorm (post-attn) before
    //     the residual add.
    //   - FFN uses GELU (exact) on the gate, not SiLU.
    //   - FFN output is followed by a second RMSNorm (post-FFN) before residual.
    //   - Final logits are softcapped: logits = tanh(logits / cap) * cap.
    //   - Per-layer input embedding (altup) is implemented via
    //     `compute_altup_per_layer_inputs` / `apply_altup_to_hidden`. For
    //     standard Gemma 3 (non-E-series), `per_layer_token_embd` is empty
    //     and the path is a no-op — matches the reference's behavior. For
    //     E-series (Gemma 3n) it applies the per-token × per-layer gate
    //     following the structure documented on `Gemma4Weights`.
    // =========================================================================

    /// GPU-resident decode path. Keeps the hidden state as a `Tensor<f32>` on
    /// the compute device through the entire layer loop, using the rms_norm
    /// / rope_split_halves / swiglu / relu2_gate CUDA kernels (see
    /// axonml-core/src/backends/cuda_kernels/transformer_ops.cu). Only
    /// round-trips to host for:
    ///
    ///   * The single new K and V rows going into the host-side KV cache
    ///     (needed for the CPU fallback path in `gpu_single_query_attention`
    ///     and for the `GpuKvCache::append_row` call, which today takes
    ///     `&[f32]`).
    ///   * The Q vector into the attention kernel and the attention output
    ///     back out — ~14 KB each, small compared to the matmul outputs.
    ///   * The final logits at the end of the forward pass.
    ///
    /// This replaces the ~10 larger host↔device round trips per layer that
    /// the CPU path does (one per matmul input/output, one per norm).
    #[cfg(feature = "cuda")]
    fn forward_one_gpu_resident(&self, token_id: u32, kv_cache: &mut KvCache) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.num_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let inter_size = self.config.intermediate_size;
        let pos = kv_cache.len;
        let eps = self.config.rms_norm_eps;
        let theta = self.config.rope_theta;
        let is_bitnet = self.config.architecture.starts_with("bitnet");

        // Lazy-populate the per-layer GPU norm/bias cache on first call.
        let layer_cache = self.gpu_layer_cache.get_or_init(|| {
            self.layers
                .iter()
                .map(|l| LayerGpuCache::new_for(l, &self.compute_device))
                .collect()
        });
        let output_norm_gpu = self.gpu_output_norm.get_or_init(|| {
            Tensor::from_vec(self.output_norm.clone(), &[self.output_norm.len()])
                .expect("output_norm tensor")
                .to_device(self.compute_device.clone())
                .expect("move output_norm to GPU")
        });

        // Embedding lookup — single token. Materialize the hidden state as
        // a GPU tensor immediately so every downstream op stays on-device.
        let id = token_id as usize;
        let mut x_host = vec![0.0f32; hidden];
        if id < self.config.vocab_size {
            x_host.copy_from_slice(&self.token_embed[id * hidden..(id + 1) * hidden]);
        }
        let mut x = Tensor::from_vec(x_host, &[1, hidden])
            .expect("embedding tensor")
            .to_device(self.compute_device.clone())
            .expect("move embedding to GPU");

        for (li, layer) in self.layers.iter().enumerate() {
            let lc = &layer_cache[li];

            // ----- Attention sub-layer ---------------------------------------
            let normed = x.rms_norm(&lc.attn_norm, eps);

            // Try the bias-fused QKV variant first — absorbs the three
            // separate bias-add launches into the matmul output write.
            // Needs all three biases to be present (Qwen2 / DeepSeek case).
            let fused_bias = if let (Some(qb), Some(kb), Some(vb)) =
                (&lc.q_bias, &lc.k_bias, &lc.v_bias)
            {
                crate::model::weight::fused_qkv_bias_q4k_matmul_gpu(
                    &layer.q_weight, &layer.k_weight, &layer.v_weight,
                    &normed, qb, kb, vb,
                )
            } else {
                None
            };

            let (mut q, mut k, mut v) = if let Some(triple) = fused_bias {
                // Biases already applied inside the matmul kernel.
                triple
            } else {
                // Fall back to the no-bias fused QKV + separate bias adds.
                let (mut q, mut k, mut v) = match crate::model::weight::fused_qkv_q4k_matmul_gpu(
                    &layer.q_weight, &layer.k_weight, &layer.v_weight, &normed,
                ) {
                    Some(triple) => triple,
                    None => (
                        layer.q_weight.matmul(&normed),
                        layer.k_weight.matmul(&normed),
                        layer.v_weight.matmul(&normed),
                    ),
                };
                if let Some(b) = &lc.q_bias { q = q.add(b).expect("q bias add"); }
                if let Some(b) = &lc.k_bias { k = k.add(b).expect("k bias add"); }
                if let Some(b) = &lc.v_bias { v = v.add(b).expect("v bias add"); }
                (q, k, v)
            };

            // Qwen3 QK-norm: per-head RMSNorm on Q and K AFTER the QKV
            // projection and BEFORE RoPE. Guarded on the optional weights —
            // absent on qwen2 / llama / gemma / bitnet, so those archs
            // stay bit-identical.
            if let Some(qn) = &lc.attn_q_norm {
                q = q.rms_norm_heads(qn, n_heads, head_dim, eps);
            }
            if let Some(kn) = &lc.attn_k_norm {
                k = k.rms_norm_heads(kn, n_kv_heads, head_dim, eps);
            }

            q = q.apply_rope_split_halves(n_heads, head_dim, theta, pos);
            k = k.apply_rope_split_halves(n_kv_heads, head_dim, theta, pos);

            // Lazy-allocate the GPU-resident KV cache on first use. Which
            // variant — f32 or Q8 — is decided by `self.kv_quant_q8`.
            let cap = self.config.max_seq_len.min(4096);
            if self.kv_quant_q8 {
                if kv_cache.gpu_q8.is_none() {
                    if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                        kv_cache.gpu_q8 = GpuKvCacheQ8::try_new(
                            cuda, self.layers.len(), n_kv_heads, head_dim, cap,
                        );
                    }
                }
            } else if kv_cache.gpu.is_none() {
                if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                    kv_cache.gpu = GpuKvCache::try_new(
                        cuda, self.layers.len(), n_kv_heads, head_dim, cap,
                    );
                }
            }

            // Append this token's K/V into the cache (f32 or Q8), then
            // grab a reference we can hand to the attention call below.
            // `attn_backend` carries which variant actually landed the
            // write — None means we fell through to the host path.
            enum AttnBackend<'a> {
                F32(&'a GpuKvCache, usize),
                #[cfg(feature = "cuda")]
                Q8(&'a GpuKvCacheQ8, usize),
            }
            let cuda_opt = axonml_core::backends::cuda::get_cuda_backend();
            let attn_backend: Option<AttnBackend<'_>> = if self.kv_quant_q8 {
                if let (Some(g), Some(cuda)) = (kv_cache.gpu_q8.as_mut(), cuda_opt) {
                    let k_guard = k.as_cuda_slice_read();
                    let v_guard = v.as_cuda_slice_read();
                    if g.append_row_device(cuda, li, pos, k_guard.slice(), v_guard.slice()) {
                        drop(k_guard);
                        drop(v_guard);
                        Some(AttnBackend::Q8(&*g, li))
                    } else {
                        // Q8 append failed (capacity or kernel error). Drop to
                        // the host mirror so the CPU fallback still gets an
                        // answer rather than a silent correctness bug.
                        let k_host = k.to_vec();
                        let v_host = v.to_vec();
                        kv_cache.k_cache[li].extend_from_slice(&k_host);
                        kv_cache.v_cache[li].extend_from_slice(&v_host);
                        None
                    }
                } else {
                    let k_host = k.to_vec();
                    let v_host = v.to_vec();
                    kv_cache.k_cache[li].extend_from_slice(&k_host);
                    kv_cache.v_cache[li].extend_from_slice(&v_host);
                    None
                }
            } else if let (Some(g), Some(cuda)) = (kv_cache.gpu.as_mut(), cuda_opt) {
                let k_guard = k.as_cuda_slice_read();
                let v_guard = v.as_cuda_slice_read();
                if g.append_row_device(cuda, li, pos, k_guard.slice(), v_guard.slice()) {
                    drop(k_guard);
                    drop(v_guard);
                    Some(AttnBackend::F32(&*g, li))
                } else {
                    let k_host = k.to_vec();
                    let v_host = v.to_vec();
                    kv_cache.k_cache[li].extend_from_slice(&k_host);
                    kv_cache.v_cache[li].extend_from_slice(&v_host);
                    None
                }
            } else {
                let k_host = k.to_vec();
                let v_host = v.to_vec();
                kv_cache.k_cache[li].extend_from_slice(&k_host);
                kv_cache.v_cache[li].extend_from_slice(&v_host);
                None
            };

            let total_len = pos + 1;
            // GPU-native attention — same for both F32 and Q8 backends;
            // the difference is only the kernel called inside the entry
            // point. Q is a GPU tensor throughout; output comes back as a
            // GPU tensor ready to chain into the O-projection matmul.
            let mut attn = match (cuda_opt, attn_backend) {
                (Some(cuda), Some(AttnBackend::F32(gpu, layer_idx))) => {
                    let q_guard = q.as_cuda_slice_read();
                    let attn_t = try_gpu_attn_resident_gpu(
                        cuda, q_guard.slice(), gpu, layer_idx,
                        total_len, n_heads, n_kv_heads, head_dim, 0,
                    )
                    .expect("GPU-resident F32 attention failed");
                    drop(q_guard);
                    attn_t
                }
                #[cfg(feature = "cuda")]
                (Some(cuda), Some(AttnBackend::Q8(gpu, layer_idx))) => {
                    let q_guard = q.as_cuda_slice_read();
                    let attn_t = try_gpu_attn_resident_q8_gpu(
                        cuda, q_guard.slice(), gpu, layer_idx,
                        total_len, n_heads, n_kv_heads, head_dim, 0,
                    )
                    .expect("GPU-resident Q8 attention failed");
                    drop(q_guard);
                    attn_t
                }
                _ => {
                    // Host fallback: only when GPU cache couldn't land.
                    let q_host = q.to_vec();
                    let attn_host = gpu_single_query_attention(
                        &q_host, &kv_cache.k_cache[li], &kv_cache.v_cache[li],
                        total_len, n_heads, n_kv_heads, head_dim, 0,
                        None,
                    );
                    Tensor::from_vec(attn_host, &[1, n_heads * head_dim])
                        .expect("attn tensor")
                        .to_device(self.compute_device.clone())
                        .expect("move attn to GPU")
                }
            };

            if let Some(sub) = &lc.attn_sub_norm {
                attn = attn.rms_norm(sub, eps);
            }

            let attn_proj = layer.o_weight.matmul(&attn);
            x = x.add(&attn_proj).expect("residual add (attn)");

            // ----- FFN sub-layer ---------------------------------------------
            let normed2 = x.rms_norm(&lc.ffn_norm, eps);
            let (gate, up) = match crate::model::weight::fused_gate_up_q4k_matmul_gpu(
                &layer.gate_weight, &layer.up_weight, &normed2,
            ) {
                Some(pair) => pair,
                None => (
                    layer.gate_weight.matmul(&normed2),
                    layer.up_weight.matmul(&normed2),
                ),
            };

            let mut ffn = if is_bitnet { gate.relu2_gate(&up) } else { gate.swiglu(&up) };
            if let Some(sub) = &lc.ffn_sub_norm {
                ffn = ffn.rms_norm(sub, eps);
            }

            // Sanity: the ffn tensor should be [1, inter_size]. The matmul
            // above produced that shape; the elementwise ops preserve it.
            debug_assert_eq!(ffn.numel(), inter_size);

            let ffn_out = layer.down_weight.matmul(&ffn);
            x = x.add(&ffn_out).expect("residual add (ffn)");
        }

        kv_cache.len += 1;

        // Final norm + LM head. Only D2H the logits, not the hidden state.
        let normed = x.rms_norm(output_norm_gpu, eps);
        let logits_tensor = self.lm_head.matmul(&normed);
        logits_tensor.to_vec()
    }

    fn forward_batch_gemma4(
        &self,
        token_ids: &[u32],
        kv_cache: &mut KvCache,
    ) -> Vec<f32> {
        let gemma = self.gemma4.as_ref().expect("gemma4 forward without gemma4 weights");
        let gconf = self.config.gemma.as_ref().expect("gemma4 forward without gemma config");

        let seq_len = token_ids.len();
        let hidden = self.config.hidden_size;
        let n_heads = self.config.num_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let pos_offset = kv_cache.len;

        // Token embedding lookup. Gemma (all variants) scales the token
        // embeddings by sqrt(hidden_size) at the input — this is the
        // `embed_scale` factor documented in the Gemma tech reports. Missing
        // it produces a hidden state ~50× too small for Gemma 3 4B (hidden=2560)
        // and breaks downstream attention/norm behavior.
        let embed_scale = (hidden as f32).sqrt();
        let mut x = vec![0.0f32; seq_len * hidden];
        for (pos, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < self.config.vocab_size {
                let src = &self.token_embed[id * hidden..(id + 1) * hidden];
                let dst = &mut x[pos * hidden..(pos + 1) * hidden];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = s * embed_scale;
                }
            }
        }

        // Per-layer input embeddings (altup, Gemma 3n E-series). Returns the
        // per-(position, layer) slices packed as
        // `[seq_len, num_layers, per_layer_input_width]` when altup tensors
        // are present, or `None` on non-E-series models where this path is a
        // no-op. Pre-computed once per forward pass because the lookup only
        // depends on the input token IDs.
        let altup_per_layer_inputs =
            compute_altup_per_layer_inputs(gemma, gconf, token_ids, self.config.num_layers);

        for (li, layer) in gemma.layers.iter().enumerate() {
            let is_swa = gconf.sliding_window_pattern.get(li).copied().unwrap_or(false);
            let head_dim = if is_swa { gconf.head_dim_swa } else { self.config.head_dim };
            let rope_theta = if is_swa { gconf.rope_theta_swa } else { self.config.rope_theta };
            let kv_dim = n_kv_heads * head_dim;
            let q_dim = n_heads * head_dim;

            // 0. Altup: inject the per-token × per-layer embedding into the
            //    residual stream. No-op when `per_layer_token_embd` is empty
            //    (all standard Gemma 3 models, including Oracle).
            apply_altup_to_hidden(
                &mut x,
                altup_per_layer_inputs.as_deref(),
                gemma,
                gconf,
                li,
                seq_len,
                hidden,
                self.config.num_layers,
            );

            // ─── Attention sublayer ─────────────────────────────────────────
            // 1. Pre-attn RMSNorm on hidden state
            let normed = rms_norm_vec(&x, &layer.attn_norm, self.config.rms_norm_eps, seq_len, hidden);
            let normed_t = self.to_weight_device(
                Tensor::from_vec(normed, &[seq_len, hidden]).unwrap());

            // 2. Q/K/V projections
            let mut q_data = layer.q_weight.matmul(&normed_t).to_vec();
            let mut k_data = layer.k_weight.matmul(&normed_t).to_vec();
            let v_data = layer.v_weight.matmul(&normed_t).to_vec();

            // 3. Per-head Q/K RMSNorm (Gemma-specific, pre-RoPE)
            rms_norm_per_head_inplace(&mut q_data, &layer.q_norm, self.config.rms_norm_eps,
                seq_len, n_heads, head_dim);
            rms_norm_per_head_inplace(&mut k_data, &layer.k_norm, self.config.rms_norm_eps,
                seq_len, n_kv_heads, head_dim);

            // 4. Split-halves RoPE (dual base: full vs SWA)
            apply_rope_inplace(&mut q_data, seq_len, n_heads, head_dim, rope_theta, pos_offset);
            apply_rope_inplace(&mut k_data, seq_len, n_kv_heads, head_dim, rope_theta, pos_offset);

            // 5. Append to KV cache (per-layer stride because head_dim varies)
            kv_cache.k_cache[li].extend_from_slice(&k_data);
            kv_cache.v_cache[li].extend_from_slice(&v_data);

            // 6. Attention. GPU-accelerated prefill, with optional SWA mask.
            let total_len = kv_cache.len + seq_len;
            let swa_window_batch = if is_swa { gconf.sliding_window } else { 0 };
            let attn_out = gpu_cached_attention(
                &q_data, &kv_cache.k_cache[li], &kv_cache.v_cache[li],
                seq_len, total_len, n_heads, n_kv_heads, head_dim, pos_offset,
                swa_window_batch,
            );
            let _ = kv_dim;

            // 7. Output projection
            let attn_t = self.to_weight_device(
                Tensor::from_vec(attn_out, &[seq_len, q_dim]).unwrap());
            let attn_proj = layer.o_weight.matmul(&attn_t).to_vec();

            // 8. POST-attn RMSNorm (Gemma-specific) — applied to the attention
            //    output BEFORE the residual add.
            let attn_normed = rms_norm_vec(&attn_proj, &layer.post_attention_norm,
                self.config.rms_norm_eps, seq_len, hidden);

            // 9. Residual: h = h + post_attn_norm(attn_proj)
            for i in 0..x.len() { x[i] += attn_normed[i]; }

            // ─── FFN sublayer ───────────────────────────────────────────────
            // 10. Pre-FFN RMSNorm
            let normed2 = rms_norm_vec(&x, &layer.ffn_norm, self.config.rms_norm_eps,
                seq_len, hidden);
            let normed2_t = self.to_weight_device(
                Tensor::from_vec(normed2, &[seq_len, hidden]).unwrap());

            // 11. Gate/Up projections + GELU activation
            let gate_data = layer.gate_weight.matmul(&normed2_t).to_vec();
            let up_data = layer.up_weight.matmul(&normed2_t).to_vec();
            let inter_size = self.config.intermediate_size;
            let mut ffn_data: Vec<f32> = vec![0.0f32; seq_len * inter_size];
            ffn_data
                .par_iter_mut()
                .zip(gate_data.par_iter().zip(up_data.par_iter()))
                .for_each(|(out, (&g, &u))| {
                    *out = gelu_exact(g) * u;
                });

            // 12. Down projection
            let ffn_t = self.to_weight_device(
                Tensor::from_vec(ffn_data, &[seq_len, inter_size]).unwrap());
            let ffn_out = layer.down_weight.matmul(&ffn_t).to_vec();

            // 13. POST-FFN RMSNorm (Gemma-specific)
            let ffn_normed = rms_norm_vec(&ffn_out, &layer.post_ffw_norm,
                self.config.rms_norm_eps, seq_len, hidden);

            // 14. Residual: h = h + post_ffw_norm(ffn_out)
            for i in 0..x.len() { x[i] += ffn_normed[i]; }
        }

        kv_cache.len += seq_len;

        // Final output norm
        let normed = rms_norm_vec(&x, &self.output_norm, self.config.rms_norm_eps, seq_len, hidden);
        let normed_t = self.to_weight_device(
            Tensor::from_vec(normed, &[seq_len, hidden]).unwrap());

        // LM head → vocab logits
        let mut logits = self.lm_head.matmul(&normed_t).to_vec();

        // Gemma-specific final logit softcap
        if let Some(cap) = gconf.final_logit_softcap {
            softcap_inplace(&mut logits, cap);
        }

        logits
    }

    fn forward_one_gemma4(&self, token_id: u32, kv_cache: &mut KvCache) -> Vec<f32> {
        let gemma = self.gemma4.as_ref().expect("gemma4 forward without gemma4 weights");
        let gconf = self.config.gemma.as_ref().expect("gemma4 forward without gemma config");

        let hidden = self.config.hidden_size;
        let n_heads = self.config.num_heads;
        let n_kv_heads = self.config.num_kv_heads;
        let pos = kv_cache.len;

        // Token embedding lookup — single token, scaled by sqrt(hidden_size)
        // (Gemma-specific `embed_scale`; see the forward_batch_gemma4 comment).
        let embed_scale = (hidden as f32).sqrt();
        let mut x = vec![0.0f32; hidden];
        let id = token_id as usize;
        if id < self.config.vocab_size {
            let src = &self.token_embed[id * hidden..(id + 1) * hidden];
            for (d, &s) in x.iter_mut().zip(src.iter()) {
                *d = s * embed_scale;
            }
        }

        // Altup per-layer inputs for this single token (no-op on non-E-series
        // Gemma, which is Oracle's case).
        let altup_per_layer_inputs =
            compute_altup_per_layer_inputs(gemma, gconf, &[token_id], self.config.num_layers);

        for (li, layer) in gemma.layers.iter().enumerate() {
            let is_swa = gconf.sliding_window_pattern.get(li).copied().unwrap_or(false);
            let head_dim = if is_swa { gconf.head_dim_swa } else { self.config.head_dim };
            let rope_theta = if is_swa { gconf.rope_theta_swa } else { self.config.rope_theta };
            let q_dim = n_heads * head_dim;

            // Altup injection into the residual stream.
            apply_altup_to_hidden(
                &mut x,
                altup_per_layer_inputs.as_deref(),
                gemma,
                gconf,
                li,
                1,
                hidden,
                self.config.num_layers,
            );

            // Pre-attn RMSNorm
            let normed = rms_norm_single(&x, &layer.attn_norm, self.config.rms_norm_eps);
            let normed_t = self.to_weight_device(
                Tensor::from_vec(normed, &[1, hidden]).unwrap());

            // Q/K/V projections
            let mut q_data = layer.q_weight.matmul(&normed_t).to_vec();
            let mut k_data = layer.k_weight.matmul(&normed_t).to_vec();
            let v_data = layer.v_weight.matmul(&normed_t).to_vec();

            // Per-head Q/K RMSNorm
            rms_norm_per_head_inplace(&mut q_data, &layer.q_norm, self.config.rms_norm_eps,
                1, n_heads, head_dim);
            rms_norm_per_head_inplace(&mut k_data, &layer.k_norm, self.config.rms_norm_eps,
                1, n_kv_heads, head_dim);

            // RoPE
            apply_rope_single(&mut q_data, n_heads, head_dim, rope_theta, pos);
            apply_rope_single(&mut k_data, n_kv_heads, head_dim, rope_theta, pos);

            // Append to KV cache
            kv_cache.k_cache[li].extend_from_slice(&k_data);
            kv_cache.v_cache[li].extend_from_slice(&v_data);

            // Mirror to the GPU-resident KV buffer (lazy-allocated on first
            // layer). See forward_one for the full rationale.
            #[cfg(feature = "cuda")]
            let gpu_cache_ref = {
                if kv_cache.gpu.is_none() {
                    if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                        let cap = self.config.max_seq_len.min(4096);
                        kv_cache.gpu = GpuKvCache::try_new(cuda, self.layers.len(), n_kv_heads, head_dim, cap);
                    }
                }
                let mut r: Option<(&GpuKvCache, usize)> = None;
                if let Some(ref mut g) = kv_cache.gpu {
                    if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
                        if g.append_row(cuda, li, pos, &k_data, &v_data) {
                            r = Some((&*g, li));
                        }
                    }
                }
                r
            };

            // Single-query attention (SWA or full). GPU-accelerated via
            // the fused flash-decode kernel when CUDA is available.
            let total_len = pos + 1;
            let swa_window = if is_swa { gconf.sliding_window } else { 0 };
            let attn_out = gpu_single_query_attention(
                &q_data, &kv_cache.k_cache[li], &kv_cache.v_cache[li],
                total_len, n_heads, n_kv_heads, head_dim, swa_window,
                #[cfg(feature = "cuda")]
                gpu_cache_ref,
            );

            // Output projection
            let attn_t = self.to_weight_device(
                Tensor::from_vec(attn_out, &[1, q_dim]).unwrap());
            let attn_proj = layer.o_weight.matmul(&attn_t).to_vec();

            // POST-attn RMSNorm
            let attn_normed = rms_norm_single(&attn_proj, &layer.post_attention_norm,
                self.config.rms_norm_eps);

            // Residual
            for i in 0..hidden { x[i] += attn_normed[i]; }

            // FFN sublayer
            let normed2 = rms_norm_single(&x, &layer.ffn_norm, self.config.rms_norm_eps);
            let normed2_t = self.to_weight_device(
                Tensor::from_vec(normed2, &[1, hidden]).unwrap());

            let gate_data = layer.gate_weight.matmul(&normed2_t).to_vec();
            let up_data = layer.up_weight.matmul(&normed2_t).to_vec();
            let inter_size = self.config.intermediate_size;
            let mut ffn_data: Vec<f32> = vec![0.0f32; inter_size];
            ffn_data
                .par_iter_mut()
                .zip(gate_data.par_iter().zip(up_data.par_iter()))
                .for_each(|(out, (&g, &u))| {
                    *out = gelu_exact(g) * u;
                });
            let ffn_t = self.to_weight_device(
                Tensor::from_vec(ffn_data, &[1, inter_size]).unwrap());
            let ffn_out = layer.down_weight.matmul(&ffn_t).to_vec();

            // POST-FFN RMSNorm
            let ffn_normed = rms_norm_single(&ffn_out, &layer.post_ffw_norm,
                self.config.rms_norm_eps);

            // Residual
            for i in 0..hidden { x[i] += ffn_normed[i]; }
        }

        kv_cache.len += 1;

        let normed = rms_norm_single(&x, &self.output_norm, self.config.rms_norm_eps);
        let normed_t = self.to_weight_device(Tensor::from_vec(normed, &[1, hidden]).unwrap());
        let mut logits = self.lm_head.matmul(&normed_t).to_vec();

        if let Some(cap) = gconf.final_logit_softcap {
            softcap_inplace(&mut logits, cap);
        }

        logits
    }
}

// =============================================================================
// Math helpers
// =============================================================================

fn rms_norm_vec(x: &[f32], weight: &[f32], eps: f32, seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * dim];
    for s in 0..seq_len {
        let offset = s * dim;
        let ss: f32 = x[offset..offset + dim].iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let rms = (ss + eps).sqrt();
        for d in 0..dim {
            out[offset + d] = x[offset + d] / rms * weight[d];
        }
    }
    out
}

fn rms_norm_single(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let rms = (ss + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&xi, &wi)| xi / rms * wi).collect()
}

fn add_bias(data: &mut [f32], bias: &[f32], seq_len: usize, dim: usize) {
    for s in 0..seq_len {
        for d in 0..dim {
            data[s * dim + d] += bias[d];
        }
    }
}

/// Qwen3 QK-norm: per-head RMS_norm applied in-place. `data` is laid out
/// as `[seq_len, n_heads * head_dim]`; `weight` is `[head_dim]`, broadcast
/// across every (token, head). Mirrors `Tensor::rms_norm_heads` but on
/// a flat CPU buffer — used by the prefill path.
fn apply_rms_norm_heads_inplace(
    data: &mut [f32],
    weight: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    debug_assert_eq!(weight.len(), head_dim);
    debug_assert_eq!(data.len(), seq_len * n_heads * head_dim);
    for s in 0..seq_len {
        for h in 0..n_heads {
            let base = (s * n_heads + h) * head_dim;
            let mut sum_sq = 0.0f64;
            for i in 0..head_dim {
                let v = data[base + i] as f64;
                sum_sq += v * v;
            }
            let scale = ((sum_sq / head_dim as f64) + eps as f64).sqrt().recip() as f32;
            for i in 0..head_dim {
                data[base + i] = data[base + i] * scale * weight[i];
            }
        }
    }
}

/// Apply split-halves RoPE in-place.
///
/// This is the LLaMA/Qwen2/Mistral variant (not interleaved) where pairs are
/// (x[i], x[i + head_dim/2]) with i in 0..head_dim/2, and frequency
/// freq[i] = 1 / theta^(2*i / head_dim).
///
/// Formula (matching candle's rope / HuggingFace transformers):
///   new_x[i]          = x[i] * cos - x[i + half] * sin
///   new_x[i + half]   = x[i + half] * cos + x[i] * sin
fn apply_rope_inplace(x: &mut [f32], seq_len: usize, n_heads: usize, head_dim: usize, theta: f32, pos_offset: usize) {
    let half = head_dim / 2;
    for pos in 0..seq_len {
        let abs_pos = pos + pos_offset;
        for h in 0..n_heads {
            let offset = pos * n_heads * head_dim + h * head_dim;
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = abs_pos as f32 * freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = x[offset + i];
                let x1 = x[offset + i + half];
                x[offset + i]        = x0 * cos - x1 * sin;
                x[offset + i + half] = x1 * cos + x0 * sin;
            }
        }
    }
}

fn apply_rope_single(x: &mut [f32], n_heads: usize, head_dim: usize, theta: f32, pos: usize) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[offset + i];
            let x1 = x[offset + i + half];
            x[offset + i]        = x0 * cos - x1 * sin;
            x[offset + i + half] = x1 * cos + x0 * sin;
        }
    }
}

/// Attention for prefill: multiple query positions against growing KV cache.
fn cached_attention(
    q: &[f32], k_cache: &[f32], v_cache: &[f32],
    q_len: usize, kv_len: usize,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
    pos_offset: usize,
) -> Vec<f32> {
    let gqa_ratio = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_dim = n_kv_heads * head_dim;
    let mut out = vec![0.0f32; q_len * n_heads * head_dim];

    for qi in 0..q_len {
        let abs_pos = qi + pos_offset;
        for h in 0..n_heads {
            let kv_h = h / gqa_ratio;

            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            for t in 0..=abs_pos.min(kv_len - 1) {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * n_heads * head_dim + h * head_dim + d]
                         * k_cache[t * kv_dim + kv_h * head_dim + d];
                }
                scores[t] = dot * scale;
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                exp_sum += *s;
            }
            for s in &mut scores { *s /= exp_sum + 1e-8; }

            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for t in 0..kv_len {
                    sum += scores[t] * v_cache[t * kv_dim + kv_h * head_dim + d];
                }
                out[qi * n_heads * head_dim + h * head_dim + d] = sum;
            }
        }
    }
    out
}

/// Attention for single query token against full KV cache (decode step).
// =============================================================================
// Gemma 3n altup (per-layer input embeddings)
// =============================================================================

/// Look up the per-(position, layer) embeddings for a batch of tokens and
/// apply the per-layer proj-norm scaling.
///
/// Only runs on Gemma 3n E-series models where `per_layer_token_embd` is
/// populated — standard Gemma 3 (Oracle's family) has these tensors empty
/// and this returns `None`, making `apply_altup_to_hidden` a no-op.
///
/// Shape contract:
///   - `per_layer_token_embd` is flat `[vocab, num_layers * width]`.
///   - Returns `Vec<f32>` shape `[seq_len, num_layers, width]` flat (packed
///     in that order) when altup is active.
///
/// Per the Gemma 3n reference, the per-layer raw embedding is normalized by
/// an RMSNorm with scale `per_layer_proj_norm` (width-wide vector) before
/// being combined with the projected hidden state — that normalization is
/// done here so the downstream step only has to do lookup + gate-mul + add.
fn compute_altup_per_layer_inputs(
    gemma: &Gemma4Weights,
    gconf: &GemmaConfig,
    token_ids: &[u32],
    num_layers: usize,
) -> Option<Vec<f32>> {
    if gemma.per_layer_token_embd.is_empty() {
        return None;
    }
    let width = gconf.per_layer_input_width;
    if width == 0 {
        return None;
    }
    let per_token_stride = num_layers * width;
    let vocab = gemma.per_layer_token_embd.len() / per_token_stride.max(1);
    if vocab == 0 {
        return None;
    }
    let norm = &gemma.per_layer_proj_norm;
    let eps = 1e-6f32;

    let mut out = vec![0.0f32; token_ids.len() * num_layers * width];
    for (pos, &tok) in token_ids.iter().enumerate() {
        let tid = (tok as usize).min(vocab.saturating_sub(1));
        let src_base = tid * per_token_stride;
        let dst_base = pos * num_layers * width;
        for li in 0..num_layers {
            let src = &gemma.per_layer_token_embd
                [src_base + li * width..src_base + (li + 1) * width];
            let dst = &mut out[dst_base + li * width..dst_base + (li + 1) * width];
            // RMSNorm the raw per-layer vector by `per_layer_proj_norm`.
            // If the norm tensor is absent (malformed export), skip scaling.
            if norm.len() >= width {
                let ss: f32 = src.iter().map(|v| v * v).sum::<f32>() / width as f32;
                let rms = (ss + eps).sqrt();
                for (i, (&s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
                    *d = (s / rms) * norm[i];
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
    }
    Some(out)
}

/// Apply the altup per-layer input to the residual stream at layer `li`.
///
/// No-op when `inputs` is `None` (non-altup models) or when the projection
/// tensor is missing. When active, does:
///
///   gate_all = x @ per_layer_model_proj        # [seq, num_layers * width]
///   layer_gate = gate_all[:, li * width : (li + 1) * width]
///   x += broadcast_to_hidden( layer_gate * per_layer_input[:, li, :] )
///
/// The broadcast-to-hidden step tiles the `width`-sized per-layer signal
/// into the full hidden dimension. For Gemma 3n where `width < hidden`,
/// this tiles by replicating with modulo index — matches the reference's
/// "write gated per-layer into the residual" pattern without introducing
/// an extra weight we don't have in the GGUF.
#[allow(clippy::too_many_arguments)]
fn apply_altup_to_hidden(
    x: &mut [f32],
    inputs: Option<&[f32]>,
    gemma: &Gemma4Weights,
    gconf: &GemmaConfig,
    li: usize,
    seq_len: usize,
    hidden: usize,
    num_layers: usize,
) {
    let Some(inputs) = inputs else { return };
    let Some(ref proj) = gemma.per_layer_model_proj else { return };
    let width = gconf.per_layer_input_width;
    if width == 0 {
        return;
    }

    // Project the current hidden state through `per_layer_model_proj`.
    // The shape on disk is `[hidden, num_layers * width]` so the matmul
    // produces `[seq_len, num_layers * width]`.
    let proj_shape = proj.shape();
    let expected_out = num_layers * width;
    if proj_shape.len() != 2 || proj_shape[1] != expected_out || proj_shape[0] != hidden {
        // Shape mismatch — refuse silently rather than write garbage.
        // (Fires once if mismatched; cheap scan.)
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            eprintln!(
                "[altup] per_layer_model_proj shape {:?} != expected [{hidden}, {expected_out}]; skipping",
                proj_shape
            );
        });
        return;
    }

    // Matmul x [seq_len, hidden] @ proj [hidden, expected_out] → gate_all [seq_len, expected_out].
    let x_t = Tensor::from_vec(x.to_vec(), &[seq_len, hidden]).expect("altup: x shape");
    let gate_all = proj.matmul(&x_t).to_vec();

    // Per-position, add the layer-l gate * per-layer-embedding into x.
    for pos in 0..seq_len {
        let gate_base = pos * expected_out + li * width;
        let input_base = pos * num_layers * width + li * width;
        let x_base = pos * hidden;
        let gate_slice = &gate_all[gate_base..gate_base + width];
        let input_slice = &inputs[input_base..input_base + width];
        // Tile width → hidden via modulo addressing. For Gemma 3n this
        // aligns because `hidden` is a multiple of `width` (hidden=2048,
        // width=256 on E4B). For non-integer multiples the tail is
        // truncated; we have no ambiguity since `hidden >= width` and
        // `hidden % width == 0` for all shipped E-series configs.
        for d in 0..hidden {
            let i = d % width;
            x[x_base + d] += gate_slice[i] * input_slice[i];
        }
    }
}

/// GPU-accelerated single-query attention. Falls through to the CPU impls
/// when CUDA is unavailable or the backend returns an error.
///
/// `swa_window = 0` ⇒ full causal attention. Otherwise positions
/// `< kv_len - swa_window` are masked out (sliding window).
///
/// Two GPU paths:
/// 1. **Resident-cache path** (when `gpu_cache` is `Some`): reads K and V
///    straight from the pre-allocated GPU buffers. Only a single tiny
///    H2D for `q` (a few KB) per call. This is the fast decode path.
/// 2. **Re-upload path** (when `gpu_cache` is `None`): uploads q, k_cache,
///    v_cache to GPU each call. Kept for CPU-only environments and as a
///    safety fallback if GPU allocation failed at load time.
///
/// Both paths run the same `fused_attn_decode_f32` kernel and copy the
/// `[n_heads, head_dim]` output back to the host. Keeping the result on
/// GPU would require a larger refactor (all downstream ops would need
/// GPU inputs); for now the D2H boundary stays at the attention output.
#[cfg_attr(not(feature = "cuda"), allow(clippy::too_many_arguments))]
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn gpu_single_query_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    swa_window: usize,
    #[cfg(feature = "cuda")] gpu_cache: Option<(&GpuKvCache, usize)>,
) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    {
        if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
            // Fast path: KV cache already resident on GPU.
            if let Some((gpu, layer)) = gpu_cache {
                if let Some(out) = try_gpu_attn_resident(
                    cuda, q, gpu, layer, kv_len, n_heads, n_kv_heads, head_dim, swa_window,
                ) {
                    return out;
                }
            }
            // Fallback: re-upload K/V per call.
            if let Some(out) = try_gpu_attn_reupload(
                cuda, q, k_cache, v_cache, kv_len, n_heads, n_kv_heads, head_dim, swa_window,
            ) {
                return out;
            }
        }
    }
    // CPU fallback
    if swa_window == 0 {
        single_query_attention(q, k_cache, v_cache, kv_len, n_heads, n_kv_heads, head_dim)
    } else {
        single_query_attention_swa(
            q, k_cache, v_cache, kv_len, n_heads, n_kv_heads, head_dim, swa_window,
        )
    }
}

/// Resident-KV-cache GPU attention. Assumes the caller has already
/// written the current token's K and V rows into `gpu[layer]` at
/// position `kv_len - 1`, so we can pass the pre-allocated buffers
/// directly to the kernel with no additional H2D for the cache.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_gpu_attn_resident(
    cuda: &axonml_core::backends::CudaBackend,
    q: &[f32],
    gpu: &GpuKvCache,
    layer: usize,
    kv_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    swa_window: usize,
) -> Option<Vec<f32>> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[W18-dispatch] resident KV cache attn FIRED (first time, row_stride={} capacity={})",
            gpu.row_stride, gpu.capacity
        );
    });
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let out_len = n_heads * head_dim;
    let q_gpu = cuda.htod_copy(q).ok()?;
    let mut out_gpu = cuda.alloc_uninit::<f32>(out_len).ok()?;
    cuda.fused_attn_decode_f32(
        &q_gpu,
        &gpu.k[layer],
        &gpu.v[layer],
        &mut out_gpu,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        swa_window,
        scale,
    )
    .ok()?;
    cuda.dtoh_copy(&out_gpu).ok()
}

/// GPU-native resident attention: takes `q` as an on-device `CudaSlice<f32>`
/// (no D2H → H2D round trip for Q), returns the attention output as a GPU
/// `Tensor<f32>` ready to chain into the next op. Saves two round trips per
/// layer vs `try_gpu_attn_resident` (one D2H of Q plus one H2D of the
/// attention output on the caller side).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_gpu_attn_resident_gpu(
    cuda: &axonml_core::backends::CudaBackend,
    q_gpu: &cudarc::driver::CudaSlice<f32>,
    gpu: &GpuKvCache,
    layer: usize,
    kv_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    swa_window: usize,
) -> Option<Tensor<f32>> {
    use axonml_core::backends::cuda_pool::pool_alloc;
    use axonml_core::storage::Storage;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let out_len = n_heads * head_dim;
    let mut out_gpu = pool_alloc(out_len).ok()?;
    cuda.fused_attn_decode_f32(
        q_gpu,
        &gpu.k[layer],
        &gpu.v[layer],
        &mut out_gpu,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        swa_window,
        scale,
    )
    .ok()?;
    let storage = Storage::from_cuda_slice(
        out_gpu,
        out_len,
        Device::Cuda(0),
    );
    Tensor::from_storage(storage, &[1, out_len]).ok()
}

/// GPU-native resident attention with TurboQuant Q8 KV cache.
/// Mirrors `try_gpu_attn_resident_gpu` but calls the Q8 kernel which
/// dequants int8 K/V on the fly inside the dot-product / v accumulate
/// loops. Returns a GPU `Tensor<f32>` of attention output.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_gpu_attn_resident_q8_gpu(
    cuda: &axonml_core::backends::CudaBackend,
    q_gpu: &cudarc::driver::CudaSlice<f32>,
    gpu: &GpuKvCacheQ8,
    layer: usize,
    kv_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    swa_window: usize,
) -> Option<Tensor<f32>> {
    use axonml_core::backends::cuda_pool::pool_alloc;
    use axonml_core::storage::Storage;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let out_len = n_heads * head_dim;
    let mut out_gpu = pool_alloc(out_len).ok()?;
    cuda.fused_attn_decode_q8_f32(
        q_gpu,
        &gpu.k_q[layer],
        &gpu.k_scale[layer],
        &gpu.v_q[layer],
        &gpu.v_scale[layer],
        &mut out_gpu,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        swa_window,
        scale,
    )
    .ok()?;
    let storage = Storage::from_cuda_slice(out_gpu, out_len, Device::Cuda(0));
    Tensor::from_storage(storage, &[1, out_len]).ok()
}

/// Fallback GPU attention that re-uploads the full K/V cache each call.
/// Used when no GPU-resident cache is available (allocation failed, or
/// the caller didn't wire one up). Strictly slower than the resident
/// path; kept for correctness.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_gpu_attn_reupload(
    cuda: &axonml_core::backends::CudaBackend,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    swa_window: usize,
) -> Option<Vec<f32>> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let out_len = n_heads * head_dim;
    let q_gpu = cuda.htod_copy(q).ok()?;
    let k_gpu = cuda.htod_copy(k_cache).ok()?;
    let v_gpu = cuda.htod_copy(v_cache).ok()?;
    let mut out_gpu = cuda.alloc_uninit::<f32>(out_len).ok()?;
    cuda.fused_attn_decode_f32(
        &q_gpu,
        &k_gpu,
        &v_gpu,
        &mut out_gpu,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        swa_window,
        scale,
    )
    .ok()?;
    cuda.dtoh_copy(&out_gpu).ok()
}

/// GPU-accelerated prefill attention (multi-query, causal). Uploads Q, K, V
/// to GPU once, then loops the `fused_attn_decode_f32` kernel per query
/// position with growing `kv_len` for causal masking. Same GPU buffers
/// reused across all positions — no per-call H2D.
///
/// Falls back to the CPU `cached_attention` path when CUDA is unavailable.
#[allow(clippy::too_many_arguments)]
fn gpu_cached_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    total_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    swa_window: usize,
) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    {
        if let Some(cuda) = axonml_core::backends::cuda::get_cuda_backend() {
            if let Some(out) = try_gpu_prefill_attn(
                cuda, q, k_cache, v_cache, seq_len, total_len,
                n_heads, n_kv_heads, head_dim, pos_offset, swa_window,
            ) {
                return out;
            }
        }
    }
    if swa_window == 0 {
        cached_attention(q, k_cache, v_cache, seq_len, total_len, n_heads, n_kv_heads, head_dim, pos_offset)
    } else {
        cached_attention_swa(q, k_cache, v_cache, seq_len, total_len, n_heads, n_kv_heads, head_dim, pos_offset, swa_window)
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_gpu_prefill_attn(
    cuda: &axonml_core::backends::CudaBackend,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    total_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    swa_window: usize,
) -> Option<Vec<f32>> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let out_total = seq_len * n_heads * head_dim;

    let q_gpu = cuda.htod_copy(q).ok()?;
    let k_gpu = cuda.htod_copy(k_cache).ok()?;
    let v_gpu = cuda.htod_copy(v_cache).ok()?;
    let mut out_gpu = cuda.alloc_uninit::<f32>(out_total).ok()?;

    if let Err(e) = cuda.fused_attn_prefill_f32(
        &q_gpu,
        &k_gpu,
        &v_gpu,
        &mut out_gpu,
        seq_len,
        total_len,
        n_heads,
        n_kv_heads,
        head_dim,
        pos_offset,
        swa_window,
        scale,
    ) {
        eprintln!(
            "[attn-prefill] GPU kernel failed (seq_len={seq_len} total_kv={total_len} \
             n_heads={n_heads} n_kv_heads={n_kv_heads} hd={head_dim}): {e:?} — falling back to CPU"
        );
        return None;
    }

    cuda.dtoh_copy(&out_gpu).ok()
}

fn single_query_attention(
    q: &[f32], k_cache: &[f32], v_cache: &[f32],
    kv_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
) -> Vec<f32> {
    let gqa_ratio = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_dim = n_kv_heads * head_dim;
    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / gqa_ratio;

        let mut scores = vec![0.0f32; kv_len];
        for t in 0..kv_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[h * head_dim + d] * k_cache[t * kv_dim + kv_h * head_dim + d];
            }
            scores[t] = dot * scale;
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            exp_sum += *s;
        }
        for s in &mut scores { *s /= exp_sum + 1e-8; }

        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for t in 0..kv_len {
                sum += scores[t] * v_cache[t * kv_dim + kv_h * head_dim + d];
            }
            out[h * head_dim + d] = sum;
        }
    }
    out
}

/// Exact GELU (Gemma FFN activation).
///
/// `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`
///
/// Approximates erf via the Abramowitz-Stegun rational. Accurate enough for
/// inference (< 1e-7 error). Gemma uses the exact GELU, not the tanh-approx
/// variant found in GPT-2.
fn gelu_exact(x: f32) -> f32 {
    // erf(z) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-z^2)
    // with t = 1 / (1 + p*z), p = 0.3275911 (Abramowitz-Stegun 7.1.26)
    let z = x * std::f32::consts::FRAC_1_SQRT_2; // x / sqrt(2)
    let sign = z.signum();
    let z_abs = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z_abs);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-z_abs * z_abs).exp();
    let erf_z = sign * y;
    0.5 * x * (1.0 + erf_z)
}

/// Per-head RMSNorm applied in-place, used for Gemma's Q/K norm pre-RoPE.
///
/// Layout of `data`: `[seq_len, n_heads, head_dim]` flattened.
/// `weight` has shape `[head_dim]` (one per-channel scale, shared across heads).
/// For each (position, head) compute the RMS over the head's `head_dim`
/// elements and scale by `weight`.
fn rms_norm_per_head_inplace(
    data: &mut [f32],
    weight: &[f32],
    eps: f32,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let expected_weight_len = head_dim;
    // Gemma's q_norm/k_norm weights are head_dim-wide. If that assumption is
    // ever violated we still produce a shape-correct output but the norm is
    // applied incorrectly; clamp to be safe.
    let w_len = weight.len().min(expected_weight_len);
    for s in 0..seq_len {
        for h in 0..n_heads {
            let offset = (s * n_heads + h) * head_dim;
            let ss: f32 = data[offset..offset + head_dim].iter().map(|v| v * v).sum::<f32>()
                / head_dim as f32;
            let rms = (ss + eps).sqrt();
            for d in 0..head_dim {
                let w = if d < w_len { weight[d] } else { 1.0 };
                data[offset + d] = data[offset + d] / rms * w;
            }
        }
    }
}

/// In-place Gemma-style logit softcap: `logits = tanh(logits / cap) * cap`.
/// Keeps pre-softmax logits bounded in `[-cap, cap]`.
fn softcap_inplace(logits: &mut [f32], cap: f32) {
    if cap <= 0.0 { return; }
    let inv = 1.0 / cap;
    for l in logits.iter_mut() {
        *l = (*l * inv).tanh() * cap;
    }
}

/// Sliding-window attention for prefill. Same as `cached_attention` but
/// positions older than `pos - swa_window + 1` are masked out.
#[allow(clippy::too_many_arguments)]
fn cached_attention_swa(
    q: &[f32], k_cache: &[f32], v_cache: &[f32],
    q_len: usize, kv_len: usize,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
    pos_offset: usize,
    swa_window: usize,
) -> Vec<f32> {
    let gqa_ratio = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_dim = n_kv_heads * head_dim;
    let mut out = vec![0.0f32; q_len * n_heads * head_dim];

    for qi in 0..q_len {
        let abs_pos = qi + pos_offset;
        // Sliding window: allowed positions are [window_start, abs_pos] inclusive
        let window_start = abs_pos.saturating_sub(swa_window.saturating_sub(1));

        for h in 0..n_heads {
            let kv_h = h / gqa_ratio;

            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            for t in window_start..=abs_pos.min(kv_len - 1) {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * n_heads * head_dim + h * head_dim + d]
                         * k_cache[t * kv_dim + kv_h * head_dim + d];
                }
                scores[t] = dot * scale;
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                exp_sum += *s;
            }
            for s in &mut scores { *s /= exp_sum + 1e-8; }

            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for t in 0..kv_len {
                    sum += scores[t] * v_cache[t * kv_dim + kv_h * head_dim + d];
                }
                out[qi * n_heads * head_dim + h * head_dim + d] = sum;
            }
        }
    }
    out
}

/// Sliding-window single-query attention for decode. Same as
/// `single_query_attention` but positions older than `kv_len - swa_window`
/// are masked.
fn single_query_attention_swa(
    q: &[f32], k_cache: &[f32], v_cache: &[f32],
    kv_len: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
    swa_window: usize,
) -> Vec<f32> {
    let gqa_ratio = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_dim = n_kv_heads * head_dim;
    let mut out = vec![0.0f32; n_heads * head_dim];

    let window_start = kv_len.saturating_sub(swa_window);

    for h in 0..n_heads {
        let kv_h = h / gqa_ratio;

        let mut scores = vec![f32::NEG_INFINITY; kv_len];
        for t in window_start..kv_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[h * head_dim + d] * k_cache[t * kv_dim + kv_h * head_dim + d];
            }
            scores[t] = dot * scale;
        }

        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            exp_sum += *s;
        }
        for s in &mut scores { *s /= exp_sum + 1e-8; }

        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for t in 0..kv_len {
                sum += scores[t] * v_cache[t * kv_dim + kv_h * head_dim + d];
            }
            out[h * head_dim + d] = sum;
        }
    }
    out
}

pub fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / temperature))
        .collect();

    let max = probs.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in &mut probs {
        *v = (*v - max).exp();
        sum += *v;
    }
    for (_, v) in &mut probs { *v /= sum; }

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, (_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    let selected = &probs[..cutoff];
    let total: f32 = selected.iter().map(|(_, p)| p).sum();
    let mut r: f32 = simple_rand() * total;
    for (id, p) in selected {
        r -= p;
        if r <= 0.0 {
            return *id as u32;
        }
    }
    probs[0].0 as u32
}

fn simple_rand() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(123456789);
    let mut s = STATE.load(Ordering::Relaxed);
    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
    STATE.store(s, Ordering::Relaxed);
    ((s >> 33) as f32) / (u32::MAX as f32)
}

// =============================================================================
// Tensor loading helpers
// =============================================================================

fn load_vec(mapped: &MappedGguf, name: &str) -> Result<Vec<f32>, String> {
    mapped.load_tensor_f32(name)
        .map(|(data, _)| data)
        .ok_or_else(|| format!("Tensor not found: {name}"))
}

fn try_load_vec(mapped: &MappedGguf, name: &str) -> Option<Vec<f32>> {
    mapped.load_tensor_f32(name).map(|(data, _)| data)
}

/// Load a weight matrix as a `Weight`.
///
/// If `quantized` is false: fully dequantize to a pre-transposed f32 tensor
/// with logical shape `[in, out]`.
///
/// If `quantized` is true: copy the raw quantized bytes into a `Weight::Quantized`
/// (dequantization deferred to matmul time, saves ~5x RAM).
///
/// GGUF dim convention: dims[0]=columns (in_features), dims[1]=rows (out_features).
/// Data is stored row-major with shape [rows, cols] = [out, in].
fn load_weight(mapped: &MappedGguf, name: &str, quantized: bool) -> Result<Weight, String> {
    // Get tensor metadata
    let info = mapped.tensors.get(name)
        .ok_or_else(|| format!("Tensor not found: {name}"))?;
    let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();

    // Lazy-dequant only makes sense for block-quantized types (Q4/Q5/Q6/Q8/I2S).
    // F16/BF16/F32 weights have trivial dequant but the lazy path re-runs it on
    // every matmul — catastrophic for a tied LM head (e.g. BitNet's 656 MB F16
    // token_embd doubling as the LM head = 1.3 GB of f32 scratch + GEMM per
    // decode token). Keep these eager even under `--quantized`.
    let is_block_quantized = matches!(
        info.dtype,
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1
        | GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::Q2K | GgmlType::Q3K
        | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8K
        | GgmlType::I2S,
    );

    if quantized && dims.len() == 2 && is_block_quantized {
        // Lazy dequant path: store raw bytes
        let raw = mapped.load_tensor_raw(name)
            .ok_or_else(|| format!("Failed to load raw bytes for {name}"))?;
        Ok(Weight::from_quantized(raw, dims, info.dtype))
    } else {
        // Eager dequant path: pre-transpose to f32 tensor
        let (data, dims) = mapped.load_tensor_f32(name)
            .ok_or_else(|| format!("Tensor not found: {name}"))?;

        let tensor = if dims.len() == 2 {
            // from_vec(data, &[out, in]) matches the physical GGUF layout.
            // transpose → [in, out] for matmul: input [seq, in] @ weight [in, out].
            // contiguous() materializes the transposed layout into a flat buffer
            // so every subsequent matmul can read the weight storage directly
            // without re-paying the transpose cost per call. On an 8B model
            // with 42 layers × 7 matmuls per decode, skipping this one-shot
            // materialization would cost ~3 GB of memcpy per token.
            let t = Tensor::from_vec(data, &[dims[1], dims[0]])
                .map_err(|e| format!("{name} (shape [{}, {}]): {e}", dims[1], dims[0]))?;
            t.transpose(0, 1).map_err(|e| format!("{name} transpose: {e}"))?.contiguous()
        } else {
            Tensor::from_vec(data, &dims).map_err(|e| format!("{name}: {e}"))?
        };
        Ok(Weight::from_f32(tensor))
    }
}
