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
    tensors: HashMap<String, GgufTensorInfo>,
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
    /// LM head [hidden_size, vocab_size] — PRE-TRANSPOSED
    pub lm_head_t: Tensor<f32>,
}

/// Weights for one transformer layer.
/// All weight matrices are PRE-TRANSPOSED at load time:
/// stored as [in_dim, out_dim] so matmul is: input @ weight = [seq, out_dim]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub q_weight_t: Tensor<f32>,  // [hidden, n_heads*head_dim] PRE-TRANSPOSED
    pub k_weight_t: Tensor<f32>,  // [hidden, n_kv_heads*head_dim]
    pub v_weight_t: Tensor<f32>,  // [hidden, n_kv_heads*head_dim]
    pub o_weight_t: Tensor<f32>,  // [n_heads*head_dim, hidden]
    pub ffn_norm: Vec<f32>,
    pub gate_weight_t: Tensor<f32>, // [hidden, intermediate]
    pub up_weight_t: Tensor<f32>,   // [hidden, intermediate]
    pub down_weight_t: Tensor<f32>, // [intermediate, hidden]
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
}

impl LayerWeights {
    /// Move all weight tensors to the specified device (GPU).
    fn to_device(&mut self, device: Device) {
        self.q_weight_t = self.q_weight_t.to_device(device.clone()).unwrap_or_else(|_| self.q_weight_t.clone());
        self.k_weight_t = self.k_weight_t.to_device(device.clone()).unwrap_or_else(|_| self.k_weight_t.clone());
        self.v_weight_t = self.v_weight_t.to_device(device.clone()).unwrap_or_else(|_| self.v_weight_t.clone());
        self.o_weight_t = self.o_weight_t.to_device(device.clone()).unwrap_or_else(|_| self.o_weight_t.clone());
        self.gate_weight_t = self.gate_weight_t.to_device(device.clone()).unwrap_or_else(|_| self.gate_weight_t.clone());
        self.up_weight_t = self.up_weight_t.to_device(device.clone()).unwrap_or_else(|_| self.up_weight_t.clone());
        self.down_weight_t = self.down_weight_t.to_device(device).unwrap_or_else(|_| self.down_weight_t.clone());
    }
}

impl InferenceEngine {
    /// Move all weight matrices to GPU. Norms stay on CPU (element-wise, fast).
    pub fn to_device(&mut self, device: Device) {
        println!("  Moving weights to {:?}...", device);
        self.lm_head_t = self.lm_head_t.to_device(device.clone()).unwrap_or_else(|_| self.lm_head_t.clone());
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.to_device(device.clone());
            if (i + 1) % 7 == 0 || i + 1 == num_layers {
                println!("    Moved layer {}/{}", i + 1, num_layers);
            }
        }
        println!("  Weights on {:?}", device);
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
    pub fn from_gguf(gguf: &GgufFile, mapped: &MappedGguf) -> Result<Self, String> {
        let config = InferenceConfig::from_gguf(gguf);

        println!("  Loading weights into f32 tensors...");
        println!("    Architecture: {}", config.architecture);
        println!("    Hidden: {}, Layers: {}, Heads: {}/{}",
            config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads);
        println!("    Vocab: {}, Context: {}", config.vocab_size, config.max_seq_len);

        // Token embeddings as flat Vec (fast lookup)
        let token_embed = load_vec(mapped, "token_embd.weight")?;

        // Load per-layer weights — PRE-TRANSPOSE at load time
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("blk.{i}");

            let attn_norm = load_vec(mapped, &format!("{prefix}.attn_norm.weight"))?;

            // Load + transpose weight matrices so matmul is: [seq, in] @ [in, out]
            let q_weight_t = load_and_transpose(mapped, &format!("{prefix}.attn_q.weight"))?;
            let k_weight_t = load_and_transpose(mapped, &format!("{prefix}.attn_k.weight"))?;
            let v_weight_t = load_and_transpose(mapped, &format!("{prefix}.attn_v.weight"))?;
            let o_weight_t = load_and_transpose(mapped, &format!("{prefix}.attn_output.weight"))?;

            let ffn_norm = load_vec(mapped, &format!("{prefix}.ffn_norm.weight"))?;
            let gate_weight_t = load_and_transpose(mapped, &format!("{prefix}.ffn_gate.weight"))?;
            let up_weight_t = load_and_transpose(mapped, &format!("{prefix}.ffn_up.weight"))?;
            let down_weight_t = load_and_transpose(mapped, &format!("{prefix}.ffn_down.weight"))?;

            let q_bias = try_load_vec(mapped, &format!("{prefix}.attn_q.bias"));
            let k_bias = try_load_vec(mapped, &format!("{prefix}.attn_k.bias"));
            let v_bias = try_load_vec(mapped, &format!("{prefix}.attn_v.bias"));

            layers.push(LayerWeights {
                attn_norm, q_weight_t, k_weight_t, v_weight_t, o_weight_t,
                ffn_norm, gate_weight_t, up_weight_t, down_weight_t,
                q_bias, k_bias, v_bias,
            });

            if (i + 1) % 7 == 0 || i + 1 == config.num_layers {
                println!("    Loaded layer {}/{}", i + 1, config.num_layers);
            }
        }

        // Output norm
        let output_norm = load_vec(mapped, "output_norm.weight")?;

        // LM head: need [hidden, vocab] for input @ lm_head = [seq, vocab]
        let lm_head_t = if mapped.has_tensor("output.weight") {
            load_and_transpose(mapped, "output.weight")?
        } else {
            // Tied embeddings: load_and_transpose handles the dim mapping
            println!("    LM head tied to token embeddings");
            load_and_transpose(mapped, "token_embd.weight")?
        };

        let total_bytes: usize = token_embed.len() * 4
            + layers.iter().map(|l| {
                l.attn_norm.len() * 4
                + l.q_weight_t.to_vec().len() * 4
                + l.k_weight_t.to_vec().len() * 4
                + l.v_weight_t.to_vec().len() * 4
                + l.o_weight_t.to_vec().len() * 4
                + l.ffn_norm.len() * 4
                + l.gate_weight_t.to_vec().len() * 4
                + l.up_weight_t.to_vec().len() * 4
                + l.down_weight_t.to_vec().len() * 4
            }).sum::<usize>()
            + output_norm.len() * 4;

        println!("  Model loaded: {:.1} GB f32 weights in RAM", total_bytes as f64 / 1e9);

        Ok(Self {
            config,
            token_embed,
            layers,
            output_norm,
            lm_head_t,
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
        let mut kv_cache = KvCache::new(self.config.num_layers);
        let mut generated = Vec::with_capacity(max_new_tokens);

        // Prefill: process entire prompt at once
        let logits = self.forward_batch(prompt_ids, &mut kv_cache);
        let vocab_size = self.config.vocab_size;
        let last_logits = &logits[logits.len() - vocab_size..];

        let mut next_id = if temperature < 0.01 {
            argmax(last_logits) as u32
        } else {
            sample_top_p(last_logits, temperature, top_p)
        };

        if next_id == 0 || next_id == 151643 || next_id == 151645 { // common EOS tokens
            return generated;
        }
        generated.push(next_id);

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
            generated.push(next_id);
        }

        generated
    }

    /// Detect the device weights are on.
    fn weight_device(&self) -> Device {
        self.lm_head_t.device()
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
            let mut q_data = normed_t.matmul(&layer.q_weight_t).unwrap().to_vec();
            let k_data = normed_t.matmul(&layer.k_weight_t).unwrap().to_vec();
            let mut v_data_new = normed_t.matmul(&layer.v_weight_t).unwrap().to_vec();
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
            let attn_proj = attn_t.matmul(&layer.o_weight_t).unwrap().to_vec();

            // Residual
            for i in 0..x.len() { x[i] += attn_proj[i]; }

            // FFN
            let normed2 = rms_norm_vec(&x, &layer.ffn_norm, self.config.rms_norm_eps, seq_len, hidden);
            let normed2_t = self.to_weight_device(Tensor::from_vec(normed2, &[seq_len, hidden]).unwrap());

            let gate_data = normed2_t.matmul(&layer.gate_weight_t).unwrap().to_vec();
            let up_data = normed2_t.matmul(&layer.up_weight_t).unwrap().to_vec();

            // SiLU(gate) * up (CPU — element-wise)
            let inter_size = self.config.intermediate_size;
            let mut ffn_data = vec![0.0f32; seq_len * inter_size];
            for i in 0..ffn_data.len() {
                let g = gate_data[i];
                ffn_data[i] = (g / (1.0 + (-g).exp())) * up_data[i];
            }
            let ffn_t = self.to_weight_device(Tensor::from_vec(ffn_data, &[seq_len, inter_size]).unwrap());
            let ffn_out = ffn_t.matmul(&layer.down_weight_t).unwrap().to_vec();

            for i in 0..x.len() { x[i] += ffn_out[i]; }
        }

        kv_cache.len += seq_len;

        // Final norm + LM head
        let normed = rms_norm_vec(&x, &self.output_norm, self.config.rms_norm_eps, seq_len, hidden);
        let normed_t = self.to_weight_device(Tensor::from_vec(normed, &[seq_len, hidden]).unwrap());
        normed_t.matmul(&self.lm_head_t).unwrap().to_vec()
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
            let mut q_data = normed_t.matmul(&layer.q_weight_t).unwrap().to_vec();
            let mut k_data = normed_t.matmul(&layer.k_weight_t).unwrap().to_vec();
            let mut v_data = normed_t.matmul(&layer.v_weight_t).unwrap().to_vec();

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
            let attn_proj = attn_t.matmul(&layer.o_weight_t).unwrap().to_vec();

            for i in 0..hidden { x[i] += attn_proj[i]; }

            // FFN
            let normed2 = rms_norm_single(&x, &layer.ffn_norm, self.config.rms_norm_eps);
            let normed2_t = self.to_weight_device(Tensor::from_vec(normed2, &[1, hidden]).unwrap());

            let gate_data = normed2_t.matmul(&layer.gate_weight_t).unwrap().to_vec();
            let up_data = normed2_t.matmul(&layer.up_weight_t).unwrap().to_vec();

            let inter_size = self.config.intermediate_size;
            let mut ffn_data = vec![0.0f32; inter_size];
            for i in 0..inter_size {
                let g = gate_data[i];
                ffn_data[i] = (g / (1.0 + (-g).exp())) * up_data[i];
            }
            let ffn_t = self.to_weight_device(Tensor::from_vec(ffn_data, &[1, inter_size]).unwrap());
            let ffn_out = ffn_t.matmul(&layer.down_weight_t).unwrap().to_vec();

            for i in 0..hidden { x[i] += ffn_out[i]; }
        }

        kv_cache.len += 1;

        // Final norm + LM head
        let normed = rms_norm_single(&x, &self.output_norm, self.config.rms_norm_eps);
        let normed_t = self.to_weight_device(Tensor::from_vec(normed, &[1, hidden]).unwrap());
        normed_t.matmul(&self.lm_head_t).unwrap().to_vec()
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

/// Load a weight matrix and transpose it for matmul: input @ weight_t = [seq, out]
///
/// GGUF dim convention: dims[0]=columns, dims[1]=rows, data row-major.
/// AxonML's Tensor::from_vec(data, &[A,B]) treats A as rows and B as cols (C-order).
///
/// So from_vec(data, &dims) with dims=[cols, rows] creates a tensor where the flat
/// buffer is interpreted as dims[0] rows of dims[1] cols — which effectively reads
/// the GGUF [rows, cols] physical layout as [cols, rows] logical. Then .transpose()
/// flips it to [rows, cols] = the correct weight orientation for matmul.
///
/// This was verified against llama.cpp output — the old [dims[1],dims[0]] version
/// double-transposed and produced incoherent output.
fn load_and_transpose(mapped: &MappedGguf, name: &str) -> Result<Tensor<f32>, String> {
    let (data, dims) = mapped.load_tensor_f32(name)
        .ok_or_else(|| format!("Tensor not found: {name}"))?;

    if dims.len() == 2 {
        // GGUF convention: dims[0] = columns (in_features), dims[1] = rows (out_features)
        // Data is row-major: [out, in] physically.
        // from_vec(data, &[out, in]) matches the physical layout.
        // transpose → [in, out] for matmul: input [seq, in] @ weight [in, out]
        let t = Tensor::from_vec(data, &[dims[1], dims[0]])
            .map_err(|e| format!("{name} (shape [{}, {}]): {e}", dims[1], dims[0]))?;
        t.transpose(0, 1).map_err(|e| format!("{name} transpose: {e}"))
    } else {
        Tensor::from_vec(data, &dims).map_err(|e| format!("{name}: {e}"))
    }
}
