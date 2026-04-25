//! Qwen3 → GGUF exporter for trained student models.
//!
//! Inverse of `gguf_loader`: takes a trained `Qwen3ForCausalLM`, writes a
//! GGUF v3 file that `nexus-serve` (and `spec_bench`) can load as a
//! draft. Tensor weights are written as F16 (half the bytes of F32,
//! supported by the existing `dequantize_f16` path in nexus-serve).
//! RMSNorm weights stay F32 because they're tiny and FP16 precision
//! around 1.0 ± ε can hurt convergence of the normalized scale.
//!
//! Optional tokenizer embedding: when `tokenizer_source` is set, the
//! exporter copies `tokenizer.ggml.*` + `general.file_type` metadata
//! from a reference GGUF (typically the teacher). Required for
//! spec_bench and nexus-serve to actually detokenize the student's
//! output. Without it the file still loads structurally but inference
//! needs a sidecar tokenizer.json.
//!
//! # Scope (deferred)
//! - Q4_K export. Requires calibration (super-block scales) and is a
//!   substantial compute / correctness effort. For a distillation
//!   draft that targets speculative decoding, F16 is the right
//!   first-cut: smaller than F32, loads fast, matches a real Qwen3
//!   GGUF's dtype mix for non-body-weight tensors.
//! - Q8_0 export (middle-ground quality/size). Same rationale as Q4_K.
//!
//! # File
//! `crates/axonml-llm/src/gguf_export.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Seek, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use axonml_nn::Module;
use axonml_tensor::Tensor;

use crate::gguf_loader::read_gguf_metadata_raw_bytes;
use crate::qwen3::{Qwen3Config, Qwen3ForCausalLM};

// =============================================================================
// GGUF constants (must match gguf_loader)
// =============================================================================

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const DATA_ALIGNMENT: u64 = 32;

// GGUF value type codes — match reader in gguf_loader. A few of these are
// defined for spec completeness; the current writer only emits U32/F32/STRING
// directly. `VTYPE_U64` and `VTYPE_BOOL` are referenced by the helper fns
// below which are kept for future emitters.
#[allow(dead_code)]
const VTYPE_U8: u32 = 0;
const VTYPE_U32: u32 = 4;
const VTYPE_F32: u32 = 6;
#[allow(dead_code)]
const VTYPE_BOOL: u32 = 7;
const VTYPE_STRING: u32 = 8;
#[allow(dead_code)]
const VTYPE_ARRAY: u32 = 9;
#[allow(dead_code)]
const VTYPE_U64: u32 = 10;

// GGML tensor type codes.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;

// =============================================================================
// HF → llama.cpp name translator (inverse of gguf_loader::ggml_to_hf_name)
// =============================================================================

/// Convert a HuggingFace-style tensor name into llama.cpp / GGUF naming.
/// Returns `None` for names we don't expect to emit (caller skips).
fn hf_to_ggml_name(hf_name: &str) -> Option<String> {
    match hf_name {
        "model.embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "model.norm.weight" => return Some("output_norm.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        _ => {}
    }

    let rest = hf_name.strip_prefix("model.layers.")?;
    let (idx_str, tail) = rest.split_once('.')?;
    let idx: usize = idx_str.parse().ok()?;

    let ggml_suffix = match tail {
        "input_layernorm.weight" => "attn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };

    Some(format!("blk.{idx}.{ggml_suffix}"))
}

// =============================================================================
// F32 → F16 encoding
// =============================================================================

/// Encode f32 → f16 (IEEE half), handling overflow / underflow / NaN.
/// Companion to `gguf_loader::f16_to_f32`.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // NaN or infinity.
        if frac != 0 {
            (sign << 15) | 0x7C00 | 0x0200 // quiet NaN
        } else {
            (sign << 15) | 0x7C00 // infinity
        }
    } else if exp > 112 + 30 {
        // Overflow → infinity.
        (sign << 15) | 0x7C00
    } else if exp < 112 - 10 {
        // Underflow → zero.
        sign << 15
    } else if exp < 112 {
        // Subnormal f16. Shift the fraction down so exponent is -14.
        let shift = 112 - exp + 1;
        let f = (frac | 0x0080_0000) >> shift;
        (sign << 15) | ((f >> 13) as u16)
    } else {
        let exp16 = (exp - 112) as u16;
        let frac16 = (frac >> 13) as u16;
        (sign << 15) | (exp16 << 10) | frac16
    }
}

// =============================================================================
// GGUF value writers
// =============================================================================

fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_u64::<LittleEndian>(s.len() as u64)?;
    w.write_all(s.as_bytes())
}

fn write_meta_string<W: Write>(w: &mut W, key: &str, value: &str) -> io::Result<()> {
    write_string(w, key)?;
    w.write_u32::<LittleEndian>(VTYPE_STRING)?;
    write_string(w, value)
}

fn write_meta_u32<W: Write>(w: &mut W, key: &str, value: u32) -> io::Result<()> {
    write_string(w, key)?;
    w.write_u32::<LittleEndian>(VTYPE_U32)?;
    w.write_u32::<LittleEndian>(value)
}

#[allow(dead_code)]
fn write_meta_u64<W: Write>(w: &mut W, key: &str, value: u64) -> io::Result<()> {
    write_string(w, key)?;
    w.write_u32::<LittleEndian>(VTYPE_U64)?;
    w.write_u64::<LittleEndian>(value)
}

fn write_meta_f32<W: Write>(w: &mut W, key: &str, value: f32) -> io::Result<()> {
    write_string(w, key)?;
    w.write_u32::<LittleEndian>(VTYPE_F32)?;
    w.write_f32::<LittleEndian>(value)
}

#[allow(dead_code)]
fn write_meta_bool<W: Write>(w: &mut W, key: &str, value: bool) -> io::Result<()> {
    write_string(w, key)?;
    w.write_u32::<LittleEndian>(VTYPE_BOOL)?;
    w.write_u8(u8::from(value))
}

// =============================================================================
// Tokenizer copy (from teacher GGUF)
// =============================================================================

/// Tokenizer metadata keys that must be preserved across GGUFs sharing
/// the same vocabulary. Derived from llama.cpp's tokenizer contract.
const TOKENIZER_META_KEYS: &[&str] = &[
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.chat_template",
];

// =============================================================================
// Tensor manifest
// =============================================================================

struct TensorEntry<'a> {
    ggml_name: String,
    /// Row-major physical shape `[out, in]` for 2D, `[n]` for 1D.
    shape: Vec<usize>,
    /// The trained parameter data (borrowed from the model via `.data()`).
    tensor: Tensor<f32>,
    /// `true` → store as F16; `false` → store as F32. Norms stay F32 to
    /// preserve their near-1.0 magnitudes across the f16 round-trip.
    half_precision: bool,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> TensorEntry<'a> {
    fn dtype_code(&self) -> u32 {
        if self.half_precision {
            GGML_F16
        } else {
            GGML_F32
        }
    }

    fn type_size(&self) -> usize {
        if self.half_precision { 2 } else { 4 }
    }

    fn byte_len(&self) -> usize {
        self.shape.iter().product::<usize>() * self.type_size()
    }

    /// GGUF dim convention: `[n_cols, n_rows, …]` — for a weight matrix
    /// with physical shape `[out, in]`, GGUF writes dims as `[in, out]`.
    fn gguf_dims(&self) -> Vec<u64> {
        match self.shape.len() {
            1 => vec![self.shape[0] as u64],
            2 => vec![self.shape[1] as u64, self.shape[0] as u64],
            _ => panic!("Unsupported tensor rank for GGUF export"),
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Export a trained `Qwen3ForCausalLM` to GGUF v3.
///
/// # Arguments
/// * `model` - The trained student.
/// * `config` - Qwen3 config used to build the student (dims, rope_theta, …).
/// * `output` - Destination path. Will be overwritten.
/// * `model_name` - Friendly name written to `general.name`.
/// * `tokenizer_source` - Optional path to a Qwen3-family GGUF whose
///   `tokenizer.ggml.*` metadata will be copied verbatim. Required for
///   nexus-serve / spec_bench to detokenize inference output.
///
/// Body weights (attention Q/K/V/O, MLP gate/up/down, embeddings,
/// LM head) are stored F16. RMSNorm weights (attn_norm, ffn_norm,
/// q_norm, k_norm, output_norm) stay F32 because F16 precision at
/// magnitudes near 1.0 can shift the normalized-scale statistics
/// enough to visibly hurt greedy quality.
pub fn export_qwen3_to_gguf(
    model: &Qwen3ForCausalLM,
    config: &Qwen3Config,
    output: &Path,
    model_name: &str,
    tokenizer_source: Option<&Path>,
) -> io::Result<()> {
    // ---- Build the tensor manifest from the model's parameters. ----
    let params = model.parameters();
    let mut manifest: Vec<TensorEntry<'_>> = Vec::new();

    // Match HF-style names by walking parameter iteration in the order
    // Qwen3ForCausalLM exposes them — see qwen3.rs::parameters().
    // We reconstruct the HF names by layer/component index.

    // Parameter ordering (from Qwen3::parameters + Qwen3ForCausalLM::parameters):
    //   embed_tokens                                  → model.embed_tokens.weight
    //   for each layer:
    //     self_attn.{q,k,v,o}_proj                    → model.layers.N.self_attn.*_proj.weight
    //     self_attn.q_norm, self_attn.k_norm          → model.layers.N.self_attn.{q,k}_norm.weight
    //     mlp.{gate,up,down}_proj                     → model.layers.N.mlp.*_proj.weight
    //     input_layernorm                             → model.layers.N.input_layernorm.weight
    //     post_attention_layernorm                    → model.layers.N.post_attention_layernorm.weight
    //   norm                                          → model.norm.weight
    //   lm_head (only if not tied)                    → lm_head.weight
    //
    // We can't reliably reconstruct the order without introspecting
    // the Parameter's name. Luckily Parameter is `named` — let's see.
    // For robustness, we assemble the expected list of HF names
    // explicitly and fetch each parameter by index.

    let expected_names = expected_hf_names(config);
    if params.len() < expected_names.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "parameter count mismatch: model has {} params, expected at least {}",
                params.len(),
                expected_names.len()
            ),
        ));
    }

    for (i, hf_name) in expected_names.iter().enumerate() {
        let p = &params[i];
        let data = p.data();
        let shape: Vec<usize> = data.shape().to_vec();
        let ggml_name = match hf_to_ggml_name(hf_name) {
            Some(n) => n,
            None => continue, // unexpected, skip
        };
        let half_precision = !is_norm_name(&ggml_name);
        manifest.push(TensorEntry {
            ggml_name,
            shape,
            tensor: data,
            half_precision,
            _marker: std::marker::PhantomData,
        });
    }

    // ---- Decide metadata. ----
    type MetaWriter = Box<dyn FnOnce(&mut File) -> io::Result<()> + Send>;
    let mut meta_writers: Vec<MetaWriter> = Vec::new();

    // General.
    meta_writers.push(Box::new({
        let name = model_name.to_string();
        move |w| {
            write_meta_string(w, "general.architecture", "qwen3")
                .and_then(|_| write_meta_string(w, "general.name", &name))
                .and_then(|_| write_meta_u32(w, "general.file_type", 1 /* f16 */))
        }
    }));

    // Qwen3 hyperparameters.
    let ctx_len = config.max_position_embeddings as u32;
    let hidden = config.hidden_size as u32;
    let inter = config.intermediate_size as u32;
    let n_layers = config.num_hidden_layers as u32;
    let n_heads = config.num_attention_heads as u32;
    let n_kv = config.num_key_value_heads as u32;
    let head_dim = config.head_dim as u32;
    let rms_eps = config.rms_norm_eps;
    let rope_theta = config.rope_theta;

    meta_writers.push(Box::new(move |w| {
        write_meta_u32(w, "qwen3.context_length", ctx_len)?;
        write_meta_u32(w, "qwen3.embedding_length", hidden)?;
        write_meta_u32(w, "qwen3.feed_forward_length", inter)?;
        write_meta_u32(w, "qwen3.block_count", n_layers)?;
        write_meta_u32(w, "qwen3.attention.head_count", n_heads)?;
        write_meta_u32(w, "qwen3.attention.head_count_kv", n_kv)?;
        write_meta_u32(w, "qwen3.attention.key_length", head_dim)?;
        write_meta_u32(w, "qwen3.attention.value_length", head_dim)?;
        write_meta_f32(w, "qwen3.attention.layer_norm_rms_epsilon", rms_eps)?;
        write_meta_f32(w, "qwen3.rope.freq_base", rope_theta)?;
        write_meta_u32(w, "general.alignment", DATA_ALIGNMENT as u32)
    }));

    // Tokenizer metadata passthrough: read the raw encoded bytes of
    // each tokenizer.ggml.* entry (plus chat_template + bos/eos ids)
    // from the source GGUF and splice them into our output. Works on
    // ANY value type (scalar, string, array of strings, array of f32,
    // etc.) because the bytes are already GGUF-encoded.
    let tokenizer_meta: HashMap<String, Vec<u8>> = match tokenizer_source {
        Some(src) => {
            let raw = read_gguf_metadata_raw_bytes(src, TOKENIZER_META_KEYS)?;
            if raw.is_empty() {
                eprintln!(
                    "[gguf-export] WARNING: --tokenizer-source {} has no tokenizer.ggml.* keys — exported file will lack tokenizer metadata.",
                    src.display()
                );
            }
            raw
        }
        None => HashMap::new(),
    };
    let has_tokenizer = !tokenizer_meta.is_empty();

    // ---- Count metadata entries for the header. ----
    //
    // Approach: write everything to an in-memory buffer first so we know
    // the exact byte layout before committing to disk. Simpler than a
    // two-pass header-patch.

    let mut header_buf: Vec<u8> = Vec::with_capacity(4096);

    // Magic + version.
    header_buf.write_u32::<LittleEndian>(GGUF_MAGIC)?;
    header_buf.write_u32::<LittleEndian>(GGUF_VERSION)?;
    // Tensor count.
    header_buf.write_u64::<LittleEndian>(manifest.len() as u64)?;

    // --- Emit metadata to a temp buffer so we can count entries. ---
    let mut meta_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut meta_count: u64 = 0;

    // General + qwen3.
    {
        let before = meta_buf.len();
        write_meta_string(&mut meta_buf, "general.architecture", "qwen3")?;
        write_meta_string(&mut meta_buf, "general.name", model_name)?;
        write_meta_u32(&mut meta_buf, "general.file_type", 1)?;
        write_meta_u32(&mut meta_buf, "general.alignment", DATA_ALIGNMENT as u32)?;
        meta_count += 4;

        write_meta_u32(&mut meta_buf, "qwen3.context_length", ctx_len)?;
        write_meta_u32(&mut meta_buf, "qwen3.embedding_length", hidden)?;
        write_meta_u32(&mut meta_buf, "qwen3.feed_forward_length", inter)?;
        write_meta_u32(&mut meta_buf, "qwen3.block_count", n_layers)?;
        write_meta_u32(&mut meta_buf, "qwen3.attention.head_count", n_heads)?;
        write_meta_u32(&mut meta_buf, "qwen3.attention.head_count_kv", n_kv)?;
        write_meta_u32(&mut meta_buf, "qwen3.attention.key_length", head_dim)?;
        write_meta_u32(&mut meta_buf, "qwen3.attention.value_length", head_dim)?;
        write_meta_f32(
            &mut meta_buf,
            "qwen3.attention.layer_norm_rms_epsilon",
            rms_eps,
        )?;
        write_meta_f32(&mut meta_buf, "qwen3.rope.freq_base", rope_theta)?;
        meta_count += 10;

        let _ = before;
    }

    // Tokenizer passthrough entries.
    for (key, raw_kv_bytes) in &tokenizer_meta {
        // `raw_kv_bytes` is the already-encoded key+value pair from the source file.
        meta_buf.extend_from_slice(raw_kv_bytes);
        meta_count += 1;
        let _ = key;
    }

    // Now we know the exact metadata count — patch the header and append meta.
    header_buf.write_u64::<LittleEndian>(meta_count)?;
    header_buf.extend_from_slice(&meta_buf);

    // --- Tensor directory. ---
    //
    // We write (name, dims, dtype, offset). Offset is relative to data_start,
    // which isn't known until after the directory's byte length is fixed.
    // Two-pass: build the directory with offset placeholders, compute data
    // start (header + directory, aligned to DATA_ALIGNMENT), then patch
    // each tensor's offset.

    let mut tensor_dir_buf: Vec<u8> = Vec::with_capacity(manifest.len() * 128);
    let mut data_offsets: Vec<usize> = Vec::with_capacity(manifest.len()); // within data region

    let mut running_offset: u64 = 0;
    for entry in &manifest {
        write_string(&mut tensor_dir_buf, &entry.ggml_name)?;
        let dims = entry.gguf_dims();
        tensor_dir_buf.write_u32::<LittleEndian>(dims.len() as u32)?;
        for d in &dims {
            tensor_dir_buf.write_u64::<LittleEndian>(*d)?;
        }
        tensor_dir_buf.write_u32::<LittleEndian>(entry.dtype_code())?;
        tensor_dir_buf.write_u64::<LittleEndian>(running_offset)?;
        data_offsets.push(running_offset as usize);

        // Advance running offset by tensor size, aligned per tensor.
        let bytes = entry.byte_len() as u64;
        running_offset += bytes;
        // GGUF aligns each tensor to DATA_ALIGNMENT within the data region.
        running_offset = running_offset.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
    }

    // ---- Write the file. ----
    let mut file = File::create(output)?;
    file.write_all(&header_buf)?;
    file.write_all(&tensor_dir_buf)?;

    // Pad to DATA_ALIGNMENT.
    let header_and_dir_len = file.stream_position()?;
    let data_start = header_and_dir_len.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
    let padding = (data_start - header_and_dir_len) as usize;
    if padding > 0 {
        file.write_all(&vec![0u8; padding])?;
    }

    // ---- Tensor data. ----
    for entry in &manifest {
        let data_f32 = entry.tensor.to_vec();
        if entry.half_precision {
            let mut buf: Vec<u8> = Vec::with_capacity(data_f32.len() * 2);
            for &v in &data_f32 {
                buf.write_u16::<LittleEndian>(f32_to_f16(v))?;
            }
            file.write_all(&buf)?;
        } else {
            let mut buf: Vec<u8> = Vec::with_capacity(data_f32.len() * 4);
            for &v in &data_f32 {
                buf.write_f32::<LittleEndian>(v)?;
            }
            file.write_all(&buf)?;
        }
        // Pad to alignment.
        let pos = file.stream_position()?;
        let aligned = pos.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
        let pad = (aligned - pos) as usize;
        if pad > 0 {
            file.write_all(&vec![0u8; pad])?;
        }
    }

    if !has_tokenizer {
        eprintln!("[gguf-export] WARNING: no --tokenizer-source; output lacks tokenizer metadata.");
        eprintln!(
            "[gguf-export] spec_bench / nexus-serve won't be able to detokenize output until a"
        );
        eprintln!(
            "[gguf-export] tokenizer sidecar is provided. Pass a pretrained Qwen3 GGUF path to"
        );
        eprintln!("[gguf-export] copy tokenizer metadata from.");
    }

    file.sync_all()?;
    Ok(())
}

/// Produce the HuggingFace tensor-name list in the exact order
/// `Qwen3::parameters` walks the model.
fn expected_hf_names(cfg: &Qwen3Config) -> Vec<String> {
    let mut names = Vec::new();
    names.push("model.embed_tokens.weight".to_string());
    for i in 0..cfg.num_hidden_layers {
        // Order matches Qwen3Attention::parameters → Qwen3DecoderLayer::parameters.
        names.push(format!("model.layers.{i}.self_attn.q_proj.weight"));
        names.push(format!("model.layers.{i}.self_attn.k_proj.weight"));
        names.push(format!("model.layers.{i}.self_attn.v_proj.weight"));
        names.push(format!("model.layers.{i}.self_attn.o_proj.weight"));
        names.push(format!("model.layers.{i}.self_attn.q_norm.weight"));
        names.push(format!("model.layers.{i}.self_attn.k_norm.weight"));
        // Qwen3DecoderLayer::parameters appends mlp after self_attn, then
        // input_layernorm, then post_attention_layernorm.
        names.push(format!("model.layers.{i}.mlp.gate_proj.weight"));
        names.push(format!("model.layers.{i}.mlp.up_proj.weight"));
        names.push(format!("model.layers.{i}.mlp.down_proj.weight"));
        names.push(format!("model.layers.{i}.input_layernorm.weight"));
        names.push(format!("model.layers.{i}.post_attention_layernorm.weight"));
    }
    names.push("model.norm.weight".to_string());
    if !cfg.tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }
    names
}

fn is_norm_name(ggml_name: &str) -> bool {
    ggml_name.ends_with("_norm.weight") || ggml_name == "output_norm.weight"
}

// =============================================================================
// RDT → GGUF exporter
// =============================================================================

/// Produce the ordered list of GGML tensor names for an `RDTForCausalLM`,
/// matching the walk order of `RDT::parameters` + `RDTForCausalLM::parameters`.
///
/// Shape:
///
/// ```text
///   token_embd.weight
///   prelude.blk.{i}.{attn_q,attn_k,attn_v,attn_output,attn_q_norm,
///                    attn_k_norm,ffn_gate,ffn_up,ffn_down,
///                    attn_norm,ffn_norm}.weight   (11 per layer × n_prelude)
///   core.blk.{i}.…                                 (11 per layer × n_core)
///   coda.blk.{i}.…                                 (11 per layer × n_coda)
///   output_norm.weight
///   output.weight                                   (lm_head, always present — RDT doesn't tie)
/// ```
///
/// The `prelude.blk.` / `core.blk.` / `coda.blk.` prefixes tell the GGUF
/// reader which stack each layer belongs to; nexus-serve's RDT load path
/// uses this to build the three separate layer stacks.
fn expected_rdt_ggml_names(cfg: &crate::rdt::RDTConfig) -> Vec<String> {
    let stack_layer_names = |stack: &str, n: usize| -> Vec<String> {
        let mut v = Vec::with_capacity(n * 11);
        for i in 0..n {
            // Order matches Qwen3Attention → Qwen3DecoderLayer::parameters.
            v.push(format!("{stack}.blk.{i}.attn_q.weight"));
            v.push(format!("{stack}.blk.{i}.attn_k.weight"));
            v.push(format!("{stack}.blk.{i}.attn_v.weight"));
            v.push(format!("{stack}.blk.{i}.attn_output.weight"));
            v.push(format!("{stack}.blk.{i}.attn_q_norm.weight"));
            v.push(format!("{stack}.blk.{i}.attn_k_norm.weight"));
            v.push(format!("{stack}.blk.{i}.ffn_gate.weight"));
            v.push(format!("{stack}.blk.{i}.ffn_up.weight"));
            v.push(format!("{stack}.blk.{i}.ffn_down.weight"));
            v.push(format!("{stack}.blk.{i}.attn_norm.weight"));
            v.push(format!("{stack}.blk.{i}.ffn_norm.weight"));
        }
        v
    };

    let mut names = Vec::new();
    names.push("token_embd.weight".to_string());
    names.extend(stack_layer_names("prelude", cfg.n_prelude));
    names.extend(stack_layer_names("core", cfg.n_core));
    names.extend(stack_layer_names("coda", cfg.n_coda));
    names.push("output_norm.weight".to_string());
    names.push("output.weight".to_string());
    names
}

/// Export a trained `RDTForCausalLM` to GGUF v3 under architecture id `rdt`.
///
/// Metadata written (design doc §4):
///
/// ```text
///   general.architecture          "rdt"
///   general.name                  <model_name>
///   general.file_type             1 (f16)
///   general.alignment             DATA_ALIGNMENT
///   rdt.context_length            u32
///   rdt.embedding_length          u32
///   rdt.feed_forward_length       u32
///   rdt.prelude.block_count       u32
///   rdt.core.block_count          u32
///   rdt.coda.block_count          u32
///   rdt.attention.head_count      u32
///   rdt.attention.head_count_kv   u32
///   rdt.attention.key_length      u32
///   rdt.attention.value_length    u32
///   rdt.attention.layer_norm_rms_epsilon  f32
///   rdt.rope.freq_base            f32
///   rdt.recurrent.k_default       u32
///   rdt.recurrent.k_min           u32
///   rdt.recurrent.k_max           u32
///   rdt.recurrent.alpha           f32
///   rdt.recurrent.beta            f32
/// ```
///
/// Plus optional tokenizer passthrough from `tokenizer_source` (same
/// mechanism as `export_qwen3_to_gguf`).
///
/// Body weights are stored F16 (same rationale as Qwen3 export);
/// RMSNorm weights stay F32 to preserve near-1.0 magnitudes.
pub fn export_rdt_to_gguf(
    model: &crate::rdt::RDTForCausalLM,
    output: &Path,
    model_name: &str,
    tokenizer_source: Option<&Path>,
) -> io::Result<()> {
    let cfg = model.config().clone();
    let params = model.parameters();
    let expected_names = expected_rdt_ggml_names(&cfg);

    if params.len() != expected_names.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "RDT parameter count mismatch: model has {} params, expected {} \
                 (check prelude={} core={} coda={} matches RDT::parameters order)",
                params.len(),
                expected_names.len(),
                cfg.n_prelude,
                cfg.n_core,
                cfg.n_coda,
            ),
        ));
    }

    // Build tensor manifest.
    let mut manifest: Vec<TensorEntry<'_>> = Vec::with_capacity(expected_names.len());
    for (i, ggml_name) in expected_names.iter().enumerate() {
        let p = &params[i];
        let data = p.data();
        let shape: Vec<usize> = data.shape().to_vec();
        let half_precision = !is_norm_name(ggml_name);
        manifest.push(TensorEntry {
            ggml_name: ggml_name.clone(),
            shape,
            tensor: data,
            half_precision,
            _marker: std::marker::PhantomData,
        });
    }

    // Tokenizer passthrough (optional).
    let tokenizer_meta: HashMap<String, Vec<u8>> = match tokenizer_source {
        Some(src) => {
            let raw = read_gguf_metadata_raw_bytes(src, TOKENIZER_META_KEYS)?;
            if raw.is_empty() {
                eprintln!(
                    "[gguf-export] WARNING: --tokenizer-source {} has no tokenizer.ggml.* keys.",
                    src.display()
                );
            }
            raw
        }
        None => HashMap::new(),
    };

    // ---- Header + metadata buffer. ----
    let mut header_buf: Vec<u8> = Vec::with_capacity(4096);
    header_buf.write_u32::<LittleEndian>(GGUF_MAGIC)?;
    header_buf.write_u32::<LittleEndian>(GGUF_VERSION)?;
    header_buf.write_u64::<LittleEndian>(manifest.len() as u64)?;

    let mut meta_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut meta_count: u64 = 0;

    // General.
    write_meta_string(&mut meta_buf, "general.architecture", "rdt")?;
    write_meta_string(&mut meta_buf, "general.name", model_name)?;
    write_meta_u32(&mut meta_buf, "general.file_type", 1)?;
    write_meta_u32(&mut meta_buf, "general.alignment", DATA_ALIGNMENT as u32)?;
    meta_count += 4;

    // Base transformer hyperparameters.
    write_meta_u32(
        &mut meta_buf,
        "rdt.context_length",
        cfg.base.max_position_embeddings as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "rdt.embedding_length",
        cfg.base.hidden_size as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "rdt.feed_forward_length",
        cfg.base.intermediate_size as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "rdt.attention.head_count",
        cfg.base.num_attention_heads as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "rdt.attention.head_count_kv",
        cfg.base.num_key_value_heads as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "rdt.attention.key_length",
        cfg.base.head_dim as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "rdt.attention.value_length",
        cfg.base.head_dim as u32,
    )?;
    write_meta_f32(
        &mut meta_buf,
        "rdt.attention.layer_norm_rms_epsilon",
        cfg.base.rms_norm_eps,
    )?;
    write_meta_f32(&mut meta_buf, "rdt.rope.freq_base", cfg.base.rope_theta)?;
    meta_count += 9;

    // RDT-specific layer splits.
    write_meta_u32(
        &mut meta_buf,
        "rdt.prelude.block_count",
        cfg.n_prelude as u32,
    )?;
    write_meta_u32(&mut meta_buf, "rdt.core.block_count", cfg.n_core as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.coda.block_count", cfg.n_coda as u32)?;
    meta_count += 3;

    // Recurrent update params.
    write_meta_u32(
        &mut meta_buf,
        "rdt.recurrent.k_default",
        cfg.k_default as u32,
    )?;
    write_meta_u32(&mut meta_buf, "rdt.recurrent.k_min", cfg.k_min as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.recurrent.k_max", cfg.k_max as u32)?;
    write_meta_f32(&mut meta_buf, "rdt.recurrent.alpha", cfg.alpha)?;
    write_meta_f32(&mut meta_buf, "rdt.recurrent.beta", cfg.beta)?;
    meta_count += 5;

    // Tokenizer passthrough.
    for raw_bytes in tokenizer_meta.values() {
        meta_buf.extend_from_slice(raw_bytes);
        meta_count += 1;
    }

    header_buf.write_u64::<LittleEndian>(meta_count)?;
    header_buf.extend_from_slice(&meta_buf);

    // ---- Tensor directory. ----
    let mut tensor_dir_buf: Vec<u8> = Vec::with_capacity(manifest.len() * 128);
    let mut running_offset: u64 = 0;
    for entry in &manifest {
        write_string(&mut tensor_dir_buf, &entry.ggml_name)?;
        let dims = entry.gguf_dims();
        tensor_dir_buf.write_u32::<LittleEndian>(dims.len() as u32)?;
        for d in &dims {
            tensor_dir_buf.write_u64::<LittleEndian>(*d)?;
        }
        tensor_dir_buf.write_u32::<LittleEndian>(entry.dtype_code())?;
        tensor_dir_buf.write_u64::<LittleEndian>(running_offset)?;

        let bytes = entry.byte_len() as u64;
        running_offset += bytes;
        running_offset = running_offset.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
    }

    // ---- Write file. ----
    let mut file = File::create(output)?;
    file.write_all(&header_buf)?;
    file.write_all(&tensor_dir_buf)?;

    let header_and_dir_len = file.stream_position()?;
    let data_start = header_and_dir_len.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
    let padding = (data_start - header_and_dir_len) as usize;
    if padding > 0 {
        file.write_all(&vec![0u8; padding])?;
    }

    for entry in &manifest {
        let data_f32 = entry.tensor.to_vec();
        if entry.half_precision {
            let mut buf: Vec<u8> = Vec::with_capacity(data_f32.len() * 2);
            for &v in &data_f32 {
                buf.write_u16::<LittleEndian>(f32_to_f16(v))?;
            }
            file.write_all(&buf)?;
        } else {
            let mut buf: Vec<u8> = Vec::with_capacity(data_f32.len() * 4);
            for &v in &data_f32 {
                buf.write_f32::<LittleEndian>(v)?;
            }
            file.write_all(&buf)?;
        }
        let pos = file.stream_position()?;
        let aligned = pos.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
        let pad = (aligned - pos) as usize;
        if pad > 0 {
            file.write_all(&vec![0u8; pad])?;
        }
    }

    if tokenizer_meta.is_empty() {
        eprintln!(
            "[gguf-export] WARNING: no --tokenizer-source; RDT output at {} lacks tokenizer metadata.",
            output.display()
        );
    }

    file.sync_all()?;
    Ok(())
}

// TODO: copy_tokenizer_metadata — round-trip parsed GgufValues from a
// source GGUF back out as encoded bytes. Requires exposing GgufValue +
// its writer from gguf_loader. Follow-up session.

// =============================================================================
// Trident → BitNet b1.58 GGUF exporter
// =============================================================================

const GGML_I2S: u32 = 36;
const I2S_BLOCK_SIZE: usize = 128;
const I2S_BYTES_PER_BLOCK: usize = 32;

/// One tensor entry for the Trident exporter — extends `TensorEntry`
/// with an `I2S` ternary variant (BitNet b1.58 dtype 36) on top of the
/// existing F32/F16 cases.
enum TridentTensorDType {
    F32,
    F16,
    /// 1.58-bit ternary, group-strided 2-bit blocks, trailing tensor-wide
    /// f32 scale. Matches `axonml_quant::bitnet::I2sBlock` byte layout.
    I2S,
}

struct TridentTensorEntry {
    ggml_name: String,
    /// Row-major physical shape `[out, in]` for 2D, `[n]` for 1D.
    shape: Vec<usize>,
    tensor: Tensor<f32>,
    dtype: TridentTensorDType,
}

impl TridentTensorEntry {
    fn dtype_code(&self) -> u32 {
        match self.dtype {
            TridentTensorDType::F32 => GGML_F32,
            TridentTensorDType::F16 => GGML_F16,
            TridentTensorDType::I2S => GGML_I2S,
        }
    }

    /// On-disk byte length INCLUDING the trailing tensor-wide scale for
    /// I2_S. `dequantize_i2s` in nexus-serve reads `total_bytes()` to
    /// load the packed region, then reads 4 more bytes from the GGUF
    /// alignment pad to recover the scale.
    fn byte_len(&self) -> usize {
        let n: usize = self.shape.iter().product();
        match self.dtype {
            TridentTensorDType::F32 => n * 4,
            TridentTensorDType::F16 => n * 2,
            TridentTensorDType::I2S => {
                let blocks = n / I2S_BLOCK_SIZE;
                blocks * I2S_BYTES_PER_BLOCK + 4
            }
        }
    }

    fn gguf_dims(&self) -> Vec<u64> {
        match self.shape.len() {
            1 => vec![self.shape[0] as u64],
            2 => vec![self.shape[1] as u64, self.shape[0] as u64],
            _ => panic!("Unsupported tensor rank for Trident GGUF export"),
        }
    }
}

/// Trident-coder-BPE tokenizer metadata extracted from a HuggingFace
/// `tokenizer.json` file (the format the trident-coder-bpe trainer
/// emits). Stripped down to exactly the fields BitNet b1.58 / llama.cpp
/// readers need.
struct TridentTokenizerMeta {
    /// `tokenizer.ggml.model` — usually `"gpt2"` for byte-level BPE.
    model: String,
    /// `tokenizer.ggml.pre` — pre-tokenizer family. Set to `"default"`
    /// when the tokenizer.json lists a ByteLevel pre-tokenizer; matches
    /// llama.cpp's expectation for GPT-2 / LLaMA-3 family.
    pre: String,
    /// `tokenizer.ggml.tokens` — vocab strings sorted by id.
    tokens: Vec<String>,
    /// `tokenizer.ggml.merges` — BPE merges, each `"piece1 piece2"`.
    merges: Vec<String>,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: Option<u32>,
}

/// Read a HuggingFace `tokenizer.json` (BPE + ByteLevel) and translate
/// it into the GGUF `tokenizer.ggml.*` schema.
///
/// Schema we expect (matches the trident-coder-bpe trainer output):
/// - `model.type == "BPE"`
/// - `model.vocab` = `{ token_string -> id }`
/// - `model.merges` = `[[piece1, piece2], ...]` OR `["piece1 piece2", ...]`
/// - `pre_tokenizer.type == "ByteLevel"`
/// - `added_tokens` carries the BOS/EOS specials at fixed ids
///
/// trident-coder-bpe convention (from
/// `/opt/AxonML/tokenizers/trident-coder-bpe`):
/// id 0 = `<|endoftext|>`, used as both BOS and EOS (GPT-2 style).
fn read_trident_bpe_tokenizer(path: &Path) -> io::Result<TridentTokenizerMeta> {
    let text = std::fs::read_to_string(path)?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tokenizer.json parse failed: {e}"),
        )
    })?;

    let model = v.get("model").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "tokenizer.json: missing `model`",
        )
    })?;
    let model_type = model.get("type").and_then(|x| x.as_str()).unwrap_or("");
    if model_type != "BPE" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tokenizer.json: model.type must be BPE, got {model_type}"),
        ));
    }

    // vocab → tokens sorted by id.
    let vocab_obj = model
        .get("vocab")
        .and_then(|x| x.as_object())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "tokenizer.json: missing model.vocab",
            )
        })?;
    let mut pairs: Vec<(String, u64)> = Vec::with_capacity(vocab_obj.len());
    for (tok, id_val) in vocab_obj {
        let id = id_val.as_u64().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("vocab id for {tok} not an integer"),
            )
        })?;
        pairs.push((tok.clone(), id));
    }
    pairs.sort_by_key(|(_, id)| *id);
    // Sanity: ids must be 0..vocab_size with no gaps.
    for (i, (_, id)) in pairs.iter().enumerate() {
        if *id != i as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("vocab id gap at index {i}: got id {}", id),
            ));
        }
    }
    let tokens: Vec<String> = pairs.into_iter().map(|(t, _)| t).collect();

    // merges → space-joined "p1 p2" strings (HF JSON gives 2-element
    // arrays; older / canonical files give pre-joined strings).
    let merges_val = model.get("merges").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "tokenizer.json: missing model.merges",
        )
    })?;
    let merges_arr = merges_val.as_array().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "tokenizer.json: model.merges not an array",
        )
    })?;
    let mut merges: Vec<String> = Vec::with_capacity(merges_arr.len());
    for m in merges_arr {
        if let Some(pair) = m.as_array() {
            if pair.len() == 2 {
                let a = pair[0].as_str().unwrap_or("");
                let b = pair[1].as_str().unwrap_or("");
                merges.push(format!("{a} {b}"));
                continue;
            }
        }
        if let Some(s) = m.as_str() {
            merges.push(s.to_string());
            continue;
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tokenizer.json: merge entry is neither [str, str] nor str",
        ));
    }

    // BOS / EOS / PAD via added_tokens lookup. Falls back to id 0 for
    // BOS+EOS if no `<|endoftext|>` is found (GPT-2 style default).
    let added: Vec<(u64, String)> = v
        .get("added_tokens")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let id = t.get("id")?.as_u64()?;
                    let content = t.get("content")?.as_str()?.to_string();
                    Some((id, content))
                })
                .collect()
        })
        .unwrap_or_default();

    let mut bos: Option<u32> = None;
    let mut eos: Option<u32> = None;
    let mut pad: Option<u32> = None;
    for (id, content) in &added {
        match content.as_str() {
            "<|endoftext|>" => {
                bos = Some(*id as u32);
                eos = Some(*id as u32);
            }
            "<s>" | "<|begin_of_text|>" | "<|user|>" => {
                if bos.is_none() {
                    bos = Some(*id as u32);
                }
            }
            "</s>" | "<|end_of_text|>" | "<|tool_end|>" => {
                if eos.is_none() {
                    eos = Some(*id as u32);
                }
            }
            "<|pad|>" | "<pad>" => {
                pad = Some(*id as u32);
            }
            _ => {}
        }
    }

    Ok(TridentTokenizerMeta {
        model: "gpt2".to_string(),
        pre: "default".to_string(),
        tokens,
        merges,
        bos_token_id: bos.unwrap_or(0),
        eos_token_id: eos.unwrap_or(0),
        pad_token_id: pad,
    })
}

/// Encode a `tokenizer.ggml.tokens` / `…merges` array of strings into
/// the GGUF metadata buffer (VTYPE_ARRAY of VTYPE_STRING).
fn write_meta_array_of_strings<W: Write>(
    w: &mut W,
    key: &str,
    values: &[String],
) -> io::Result<()> {
    write_string(w, key)?;
    w.write_u32::<LittleEndian>(VTYPE_ARRAY)?;
    w.write_u32::<LittleEndian>(VTYPE_STRING)?;
    w.write_u64::<LittleEndian>(values.len() as u64)?;
    for s in values {
        write_string(w, s)?;
    }
    Ok(())
}

/// Pack a row-major `[out, in]` f32 weight into BitNet I2_S bytes:
/// per-tensor absmean scale, then group-strided 2-bit codes (128 weights
/// per 32-byte block), then 4 trailing scale bytes.
fn pack_ternary_i2s(values: &[f32], _out_features: usize, in_features: usize) -> Vec<u8> {
    assert!(in_features.is_multiple_of(I2S_BLOCK_SIZE));

    // Tensor-wide absmean scale (BitNet b1.58 absmean rule).
    let n = values.len();
    let abs_sum: f32 = values.iter().map(|v| v.abs()).sum();
    let scale = (abs_sum / n as f32).max(1e-8);
    let inv_scale = 1.0 / scale;

    // Quantize to {-1, 0, +1}.
    let mut trits: Vec<i8> = Vec::with_capacity(n);
    for &v in values {
        let normalized = (v.abs() * inv_scale).round().min(1.0);
        let mag = normalized as i8;
        let trit = if v > 0.0 {
            mag
        } else if v < 0.0 {
            -mag
        } else {
            0
        };
        trits.push(trit);
    }

    // Pack each row into the BitNet I2_S group-strided 2-bit layout.
    // Encoding: 0 → -1, 1 → 0, 2 → +1.
    let n_blocks = n / I2S_BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * I2S_BYTES_PER_BLOCK + 4);
    for b in 0..n_blocks {
        let block = &trits[b * I2S_BLOCK_SIZE..(b + 1) * I2S_BLOCK_SIZE];
        let mut bytes = [0u8; I2S_BYTES_PER_BLOCK];
        for group_idx in 0..4 {
            let shift = (6 - 2 * group_idx) as u8;
            for group_pos in 0..32 {
                let trit = block[group_idx * 32 + group_pos];
                let code: u8 = if trit > 0 {
                    2
                } else if trit < 0 {
                    0
                } else {
                    1
                };
                bytes[group_pos] |= code << shift;
            }
        }
        out.extend_from_slice(&bytes);
    }

    // Trailing tensor-wide f32 scale (read from `offset + total_bytes`
    // by `MappedGguf::load_tensor_f32` in nexus-serve).
    out.extend_from_slice(&scale.to_le_bytes());
    out
}

/// Export a trained `TridentModel` as a BitNet b1.58 GGUF.
///
/// Produces a file `nexus-serve` can load via the existing I2_S dispatch
/// path. All ternary linear weights (Q/K/V/O projections + FFN gate/up/
/// down) are written as ggml dtype 36 (I2_S, 128-elem blocks, 32-byte
/// stride + trailing tensor-wide f32 scale, group-strided 2-bit codes).
/// RMSNorm weights stay F32. Embedding + LM head are F16 to match the
/// official `microsoft/bitnet-b1.58-2B-4T-gguf` layout.
///
/// Walks `model.parameters()` in the order produced by `TridentModel`'s
/// `parameters()` impl (embed_tokens → per-block (attn_norm, attn:
/// q/k/v/o[+sub_norm], mlp_norm, mlp: up[+gate]/down[+sub_norm]) →
/// final_norm → lm_head). If you change that ordering, update this
/// function in lockstep — the unit test
/// `axonml_llm::trident::tests::test_trident_1b_config_shapes` covers
/// the count but not the order.
pub fn export_trident_to_gguf(
    model: &crate::trident::TridentModel,
    output: &Path,
    model_name: &str,
    tokenizer_source: Option<&Path>,
) -> io::Result<()> {
    // Support either a sibling `.gguf` (pull tokenizer.ggml.* keys
    // verbatim — limited by reader-side knowledge of value type 21)
    // OR a HuggingFace `.json` (translated to ggml schema by
    // `read_trident_bpe_tokenizer`).  Auto-detect by extension.
    let tokenizer_json: Option<&Path> = tokenizer_source.filter(|p| {
        p.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
    });
    let tokenizer_gguf: Option<&Path> = tokenizer_source.filter(|p| {
        p.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    });
    if let Some(ts) = tokenizer_source {
        if tokenizer_json.is_none() && tokenizer_gguf.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "tokenizer-source must be a .json (HF tokenizer) or .gguf, got {}",
                    ts.display()
                ),
            ));
        }
    }
    let cfg = model.config().clone();
    let params = model.parameters();

    // Build the expected (ggml_name, dtype) sequence in the exact
    // order TridentModel::parameters() emits them.
    let mut expected: Vec<(String, TridentTensorDType)> = Vec::new();
    expected.push(("token_embd.weight".to_string(), TridentTensorDType::F16));
    for i in 0..cfg.num_layers {
        expected.push((format!("blk.{i}.attn_norm.weight"), TridentTensorDType::F32));
        expected.push((format!("blk.{i}.attn_q.weight"), TridentTensorDType::I2S));
        expected.push((format!("blk.{i}.attn_k.weight"), TridentTensorDType::I2S));
        expected.push((format!("blk.{i}.attn_v.weight"), TridentTensorDType::I2S));
        expected.push((
            format!("blk.{i}.attn_output.weight"),
            TridentTensorDType::I2S,
        ));
        if cfg.use_sub_ln {
            expected.push((
                format!("blk.{i}.attn_sub_norm.weight"),
                TridentTensorDType::F32,
            ));
        }
        expected.push((format!("blk.{i}.ffn_norm.weight"), TridentTensorDType::F32));
        expected.push((format!("blk.{i}.ffn_up.weight"), TridentTensorDType::I2S));
        if cfg.use_squared_relu {
            expected.push((format!("blk.{i}.ffn_gate.weight"), TridentTensorDType::I2S));
        }
        expected.push((format!("blk.{i}.ffn_down.weight"), TridentTensorDType::I2S));
        if cfg.use_sub_ln {
            expected.push((
                format!("blk.{i}.ffn_sub_norm.weight"),
                TridentTensorDType::F32,
            ));
        }
    }
    expected.push(("output_norm.weight".to_string(), TridentTensorDType::F32));
    expected.push(("output.weight".to_string(), TridentTensorDType::F16));

    if params.len() != expected.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Trident parameter count mismatch: model has {} params, exporter expects {} \
                 (config: layers={} sub_ln={} squared_relu={})",
                params.len(),
                expected.len(),
                cfg.num_layers,
                cfg.use_sub_ln,
                cfg.use_squared_relu,
            ),
        ));
    }

    let mut manifest: Vec<TridentTensorEntry> = Vec::with_capacity(expected.len());
    for (i, (ggml_name, dtype)) in expected.into_iter().enumerate() {
        let p = &params[i];
        let data = p.data();
        let shape: Vec<usize> = data.shape().to_vec();
        manifest.push(TridentTensorEntry {
            ggml_name,
            shape,
            tensor: data,
            dtype,
        });
    }

    // Tokenizer source can be either a HF tokenizer.json (preferred —
    // we own the encoding and emit a clean `tokenizer.ggml.*` block)
    // or a sibling .gguf (verbatim passthrough; constrained by what the
    // reader knows about value types).
    let tokenizer_json_meta: Option<TridentTokenizerMeta> = if let Some(p) = tokenizer_json {
        Some(read_trident_bpe_tokenizer(p)?)
    } else {
        None
    };
    let tokenizer_gguf_meta: HashMap<String, Vec<u8>> = if let Some(src) = tokenizer_gguf {
        let raw = read_gguf_metadata_raw_bytes(src, TOKENIZER_META_KEYS)?;
        if raw.is_empty() {
            eprintln!(
                "[gguf-export] WARNING: --tokenizer-source {} has no tokenizer.ggml.* keys.",
                src.display()
            );
        }
        raw
    } else {
        HashMap::new()
    };

    // ---- Header + metadata ----
    let mut header_buf: Vec<u8> = Vec::with_capacity(4096);
    header_buf.write_u32::<LittleEndian>(GGUF_MAGIC)?;
    header_buf.write_u32::<LittleEndian>(GGUF_VERSION)?;
    header_buf.write_u64::<LittleEndian>(manifest.len() as u64)?;

    let mut meta_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut meta_count: u64 = 0;

    write_meta_string(&mut meta_buf, "general.architecture", "bitnet-b1.58")?;
    write_meta_string(&mut meta_buf, "general.name", model_name)?;
    write_meta_u32(&mut meta_buf, "general.file_type", 1)?;
    write_meta_u32(&mut meta_buf, "general.alignment", DATA_ALIGNMENT as u32)?;
    meta_count += 4;

    write_meta_u32(
        &mut meta_buf,
        "bitnet-b1.58.context_length",
        cfg.max_seq_len as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "bitnet-b1.58.embedding_length",
        cfg.d_model as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "bitnet-b1.58.feed_forward_length",
        cfg.intermediate_size as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "bitnet-b1.58.block_count",
        cfg.num_layers as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "bitnet-b1.58.attention.head_count",
        cfg.num_heads as u32,
    )?;
    write_meta_u32(
        &mut meta_buf,
        "bitnet-b1.58.attention.head_count_kv",
        cfg.num_kv_heads as u32,
    )?;
    write_meta_f32(
        &mut meta_buf,
        "bitnet-b1.58.attention.layer_norm_rms_epsilon",
        cfg.rms_norm_eps,
    )?;
    write_meta_f32(&mut meta_buf, "bitnet-b1.58.rope.freq_base", cfg.rope_theta)?;
    meta_count += 8;

    // Tokenizer — preferred path: HF tokenizer.json → ggml schema.
    // Fallback path: verbatim passthrough from another GGUF.
    if let Some(tk) = &tokenizer_json_meta {
        write_meta_string(&mut meta_buf, "tokenizer.ggml.model", &tk.model)?;
        write_meta_string(&mut meta_buf, "tokenizer.ggml.pre", &tk.pre)?;
        write_meta_array_of_strings(&mut meta_buf, "tokenizer.ggml.tokens", &tk.tokens)?;
        write_meta_array_of_strings(&mut meta_buf, "tokenizer.ggml.merges", &tk.merges)?;
        write_meta_u32(
            &mut meta_buf,
            "tokenizer.ggml.bos_token_id",
            tk.bos_token_id,
        )?;
        write_meta_u32(
            &mut meta_buf,
            "tokenizer.ggml.eos_token_id",
            tk.eos_token_id,
        )?;
        meta_count += 6;
        if let Some(pad) = tk.pad_token_id {
            write_meta_u32(&mut meta_buf, "tokenizer.ggml.padding_token_id", pad)?;
            meta_count += 1;
        }
    } else {
        for (key, raw) in &tokenizer_gguf_meta {
            write_string(&mut meta_buf, key)?;
            meta_buf.write_all(raw)?;
            meta_count += 1;
        }
    }

    header_buf.write_u64::<LittleEndian>(meta_count)?;
    header_buf.extend_from_slice(&meta_buf);

    // ---- Tensor directory ----
    let mut tensor_dir_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut running_offset: u64 = 0;
    for entry in &manifest {
        write_string(&mut tensor_dir_buf, &entry.ggml_name)?;
        let dims = entry.gguf_dims();
        tensor_dir_buf.write_u32::<LittleEndian>(dims.len() as u32)?;
        for d in &dims {
            tensor_dir_buf.write_u64::<LittleEndian>(*d)?;
        }
        tensor_dir_buf.write_u32::<LittleEndian>(entry.dtype_code())?;
        tensor_dir_buf.write_u64::<LittleEndian>(running_offset)?;

        let bytes = entry.byte_len() as u64;
        running_offset += bytes;
        running_offset = running_offset.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
    }

    // ---- Write file ----
    let mut file = File::create(output)?;
    file.write_all(&header_buf)?;
    file.write_all(&tensor_dir_buf)?;

    let header_and_dir_len = file.stream_position()?;
    let data_start = header_and_dir_len.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
    let padding = (data_start - header_and_dir_len) as usize;
    if padding > 0 {
        file.write_all(&vec![0u8; padding])?;
    }

    for entry in &manifest {
        let data_f32 = entry.tensor.to_vec();
        match entry.dtype {
            TridentTensorDType::F32 => {
                let mut buf: Vec<u8> = Vec::with_capacity(data_f32.len() * 4);
                for &v in &data_f32 {
                    buf.write_f32::<LittleEndian>(v)?;
                }
                file.write_all(&buf)?;
            }
            TridentTensorDType::F16 => {
                let mut buf: Vec<u8> = Vec::with_capacity(data_f32.len() * 2);
                for &v in &data_f32 {
                    buf.write_u16::<LittleEndian>(f32_to_f16(v))?;
                }
                file.write_all(&buf)?;
            }
            TridentTensorDType::I2S => {
                // I2_S only valid for 2D weights with `in_features %
                // 128 == 0`. Trident's TernaryLinear shapes always
                // satisfy this for real configs (256 / 2048 / etc).
                if entry.shape.len() != 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "{}: I2_S export requires 2D, got rank {}",
                            entry.ggml_name,
                            entry.shape.len()
                        ),
                    ));
                }
                let out_features = entry.shape[0];
                let in_features = entry.shape[1];
                if !in_features.is_multiple_of(I2S_BLOCK_SIZE) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "{}: in_features={} must be a multiple of 128 for I2_S",
                            entry.ggml_name, in_features
                        ),
                    ));
                }
                let bytes = pack_ternary_i2s(&data_f32, out_features, in_features);
                file.write_all(&bytes)?;
            }
        }
        let pos = file.stream_position()?;
        let aligned = pos.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
        let pad = (aligned - pos) as usize;
        if pad > 0 {
            file.write_all(&vec![0u8; pad])?;
        }
    }

    if tokenizer_json_meta.is_none() && tokenizer_gguf_meta.is_empty() {
        eprintln!(
            "[gguf-export] WARNING: no --tokenizer-source; Trident output at {} lacks tokenizer metadata.",
            output.display()
        );
    }

    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod trident_export_tests {
    use super::*;

    #[test]
    fn export_trident_smoke_roundtrips_magic_and_arch() {
        // Build a tiny TridentModel with shapes valid for the BitNet
        // I2_S 128-block format (every TernaryLinear's `in_features`
        // must be a multiple of 128). Smoke/tiny configs use shapes
        // smaller than 128 so we hand-build a minimum-valid config.
        use crate::trident::{TridentConfig, TridentModel};
        use std::fs;

        let cfg = TridentConfig {
            vocab_size: 64,
            d_model: 128,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,        // kv_hidden = 4 * 32 = 128, multiple of 128 ✓
            intermediate_size: 256, // multiple of 128 ✓
            max_seq_len: 64,
            rms_norm_eps: 1e-5,
            use_rope: true,
            rope_theta: 500_000.0,
            use_squared_relu: true,
            use_sub_ln: true,
        };
        let model = TridentModel::new(&cfg);

        let tmp = std::env::temp_dir().join("trident_smoke_export.gguf");
        let _ = fs::remove_file(&tmp);
        export_trident_to_gguf(&model, &tmp, "trident-smoke", None)
            .expect("export_trident_to_gguf failed");

        let bytes = fs::read(&tmp).expect("read tmp gguf");
        assert!(
            bytes.len() > 64,
            "gguf should be > 64 bytes, got {}",
            bytes.len()
        );

        // GGUF magic + version.
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(magic, GGUF_MAGIC, "magic mismatch");
        assert_eq!(version, GGUF_VERSION, "version mismatch");

        // Architecture string somewhere in the header.
        let head = &bytes[..bytes.len().min(8192)];
        let needle = b"bitnet-b1.58";
        assert!(
            head.windows(needle.len()).any(|w| w == needle),
            "GGUF header missing 'bitnet-b1.58' architecture string",
        );

        let _ = fs::remove_file(&tmp);
    }
}
