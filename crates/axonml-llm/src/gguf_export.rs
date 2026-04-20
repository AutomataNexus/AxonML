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
                cfg.n_prelude, cfg.n_core, cfg.n_coda,
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
    write_meta_u32(&mut meta_buf, "rdt.context_length", cfg.base.max_position_embeddings as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.embedding_length", cfg.base.hidden_size as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.feed_forward_length", cfg.base.intermediate_size as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.attention.head_count", cfg.base.num_attention_heads as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.attention.head_count_kv", cfg.base.num_key_value_heads as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.attention.key_length", cfg.base.head_dim as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.attention.value_length", cfg.base.head_dim as u32)?;
    write_meta_f32(&mut meta_buf, "rdt.attention.layer_norm_rms_epsilon", cfg.base.rms_norm_eps)?;
    write_meta_f32(&mut meta_buf, "rdt.rope.freq_base", cfg.base.rope_theta)?;
    meta_count += 9;

    // RDT-specific layer splits.
    write_meta_u32(&mut meta_buf, "rdt.prelude.block_count", cfg.n_prelude as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.core.block_count", cfg.n_core as u32)?;
    write_meta_u32(&mut meta_buf, "rdt.coda.block_count", cfg.n_coda as u32)?;
    meta_count += 3;

    // Recurrent update params.
    write_meta_u32(&mut meta_buf, "rdt.recurrent.k_default", cfg.k_default as u32)?;
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
