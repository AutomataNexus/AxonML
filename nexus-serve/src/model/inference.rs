//! Inference engine — dequantize GGUF weights, run transformer forward pass,
//! sample tokens. No autograd graph — pure inference.
//!
//! Supports LLaMA-family architectures (LLaMA, Qwen2, Mistral, Gemma, Phi)
//! which all share the same decoder-only transformer structure:
//!   token_embd → N × (attn_norm → QKV → RoPE → attention → attn_output →
//!                      ffn_norm → gate/up → SiLU → down) → output_norm → lm_head
//!
//! Key optimizations:
//! - **Pre-transposed weights** at load time (no per-forward allocation)
//! - **KV cache** for incremental decoding (only process new token each step)
//! - **Tensor::matmul** for BLAS-accelerated matrix multiplication
//! - Element-wise ops stay as raw loops (already fast for 1D data)

use std::collections::HashMap;
use std::path::Path;

use axonml_core::Device;
use axonml_tensor::Tensor;
use memmap2::Mmap;

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
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub architecture: String,
}

impl InferenceConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let arch = gguf.architecture().unwrap_or("llama").to_string();
        let prefix = &arch;

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

        let head_dim = hidden_size / num_heads;

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

        let rope_theta = gguf
            .get_meta(&format!("{prefix}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10000.0);

        let vocab_size = gguf
            .get_meta("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                gguf::GgufValue::Array(a) => Some(a.len()),
                _ => None,
            })
            .unwrap_or(32000);

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
    pub fn load_tensor_raw(&self, name: &str) -> Option<Vec<u8>> {
        let info = self.tensors.get(name)?;
        let total_bytes = info.total_bytes() as usize;
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
    /// Per-layer weights (pre-transposed for matmul)
    pub layers: Vec<LayerWeights>,
    /// Output norm [hidden_size]
    pub output_norm: Vec<f32>,
    /// LM head — logical shape `[hidden, vocab]` (post-transpose).
    pub lm_head: Weight,
    /// Target compute device for matmul. For f32 weights, inputs and weights
    /// live on this device. For quantized weights, dequantization happens on
    /// CPU but the dequantized scratch is moved here for matmul.
    pub compute_device: Device,
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

impl InferenceEngine {
    /// Move all weight matrices to GPU (f32 variant) and set the compute device.
    /// Quantized weights stay in CPU RAM (dequantization produces per-matmul
    /// scratch tensors that get moved to `compute_device`).
    pub fn to_device(&mut self, device: Device) {
        println!("  Setting compute device to {:?}...", device);
        self.compute_device = device.clone();
        self.lm_head.to_device(device.clone());
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.to_device(device.clone());
            if (i + 1) % 7 == 0 || i + 1 == num_layers {
                println!("    Layer {}/{}", i + 1, num_layers);
            }
        }
        println!("  Compute device: {:?}", device);
    }
}

/// KV cache for incremental decoding.
pub struct KvCache {
    /// Per-layer key cache: Vec<f32> growing as [position, n_kv_heads * head_dim]
    pub k_cache: Vec<Vec<f32>>,
    /// Per-layer value cache
    pub v_cache: Vec<Vec<f32>>,
    /// Number of cached positions
    pub len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k_cache: (0..num_layers).map(|_| Vec::new()).collect(),
            v_cache: (0..num_layers).map(|_| Vec::new()).collect(),
            len: 0,
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

            layers.push(LayerWeights {
                attn_norm, q_weight, k_weight, v_weight, o_weight,
                ffn_norm, gate_weight, up_weight, down_weight,
                q_bias, k_bias, v_bias,
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
            output_norm,
            lm_head,
            compute_device: Device::Cpu,
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
        let mut kv_cache = KvCache::new(self.config.num_layers);

        // Prefill: process entire prompt at once
        let logits = self.forward_batch(prompt_ids, &mut kv_cache);
        let vocab_size = self.config.vocab_size;
        let last_logits = &logits[logits.len() - vocab_size..];

        let mut next_id = if temperature < 0.01 {
            argmax(last_logits) as u32
        } else {
            sample_top_p(last_logits, temperature, top_p)
        };

        // Common EOS tokens: <|endoftext|> for LLaMA/Qwen, <|im_end|> for ChatML
        if next_id == 0 || next_id == 151643 || next_id == 151645 {
            return;
        }
        if !on_token(next_id) {
            return;
        }

        // Decode: process one token at a time with KV cache
        for _ in 1..max_new_tokens {
            let logits = self.forward_one(next_id, &mut kv_cache);

            next_id = if temperature < 0.01 {
                argmax(&logits) as u32
            } else {
                sample_top_p(&logits, temperature, top_p)
            };

            if next_id == 0 || next_id == 151643 || next_id == 151645 {
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
    fn forward_batch(&self, token_ids: &[u32], kv_cache: &mut KvCache) -> Vec<f32> {
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

            // RoPE
            apply_rope_inplace(&mut q_data, seq_len, n_heads, head_dim, self.config.rope_theta, pos_offset);
            apply_rope_inplace(&mut k_data_new, seq_len, n_kv_heads, head_dim, self.config.rope_theta, pos_offset);

            // Append to KV cache
            kv_cache.k_cache[li].extend_from_slice(&k_data_new);
            kv_cache.v_cache[li].extend_from_slice(&v_data_new);

            // Attention with full KV cache
            let total_len = kv_cache.len + seq_len;
            let attn_out = cached_attention(
                &q_data, &kv_cache.k_cache[li], &kv_cache.v_cache[li],
                seq_len, total_len, n_heads, n_kv_heads, head_dim, pos_offset,
            );
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

            // SiLU(gate) * up (CPU — element-wise)
            let inter_size = self.config.intermediate_size;
            let mut ffn_data = vec![0.0f32; seq_len * inter_size];
            for i in 0..ffn_data.len() {
                let g = gate_data[i];
                ffn_data[i] = (g / (1.0 + (-g).exp())) * up_data[i];
            }
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
    fn forward_one(&self, token_id: u32, kv_cache: &mut KvCache) -> Vec<f32> {
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

            // Attention: query [1] against all cached [pos+1] K/V
            let total_len = pos + 1;
            let attn_out = single_query_attention(
                &q_data, &kv_cache.k_cache[li], &kv_cache.v_cache[li],
                total_len, n_heads, n_kv_heads, head_dim,
            );

            let attn_t = self.to_weight_device(Tensor::from_vec(attn_out, &[1, n_heads * head_dim]).unwrap());
            let attn_proj = layer.o_weight.matmul(&attn_t).to_vec();

            for i in 0..hidden { x[i] += attn_proj[i]; }

            // FFN
            let normed2 = rms_norm_single(&x, &layer.ffn_norm, self.config.rms_norm_eps);
            let normed2_t = self.to_weight_device(Tensor::from_vec(normed2, &[1, hidden]).unwrap());

            let gate_data = layer.gate_weight.matmul(&normed2_t).to_vec();
            let up_data = layer.up_weight.matmul(&normed2_t).to_vec();

            let inter_size = self.config.intermediate_size;
            let mut ffn_data = vec![0.0f32; inter_size];
            for i in 0..inter_size {
                let g = gate_data[i];
                ffn_data[i] = (g / (1.0 + (-g).exp())) * up_data[i];
            }
            let ffn_t = self.to_weight_device(Tensor::from_vec(ffn_data, &[1, inter_size]).unwrap());
            let ffn_out = layer.down_weight.matmul(&ffn_t).to_vec();

            for i in 0..hidden { x[i] += ffn_out[i]; }
        }

        kv_cache.len += 1;

        // Final norm + LM head
        let normed = rms_norm_single(&x, &self.output_norm, self.config.rms_norm_eps);
        let normed_t = self.to_weight_device(Tensor::from_vec(normed, &[1, hidden]).unwrap());
        self.lm_head.matmul(&normed_t).to_vec()
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

fn argmax(data: &[f32]) -> usize {
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

    if quantized && dims.len() == 2 {
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
            let t = Tensor::from_vec(data, &[dims[1], dims[0]])
                .map_err(|e| format!("{name} (shape [{}, {}]): {e}", dims[1], dims[0]))?;
            t.transpose(0, 1).map_err(|e| format!("{name} transpose: {e}"))?
        } else {
            Tensor::from_vec(data, &dims).map_err(|e| format!("{name}: {e}"))?
        };
        Ok(Weight::from_f32(tensor))
    }
}
