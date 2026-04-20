//! GGUF → `Qwen3ForCausalLM` loader for training.
//!
//! Parses a GGUF v2/v3 file, dequantizes Qwen3-family tensors to f32,
//! and populates a trainable `Qwen3ForCausalLM`. Primary use: load a
//! pre-trained DeepSeek-R1-Distill-Qwen-7B (or any Qwen3-family) GGUF
//! as the frozen teacher in `train_draft_distill`, turning the current
//! fresh-init smoke trainer into a useful-draft trainer.
//!
//! # Scope
//! - GGUF v2/v3 header + tensor-directory parsing.
//! - Dequantization for the types actually present in Qwen3 GGUFs:
//!   Q4_K (body weights), Q6_K (embedding + LM head in Q4_K_M variant),
//!   F32 (some scales), F16 (other scales). Q8_0 and BF16 are included
//!   as a safety net because some community requants use them.
//! - llama.cpp → HuggingFace tensor-name mapping for Qwen3 (QK-norm
//!   aware).
//! - Architecture-metadata → `Qwen3Config` derivation.
//!
//! # Not in scope (deferred)
//! - Q2K / Q3K / Q5K / IQ* / I2S (not used by any mainstream Qwen3
//!   exports). Falls back to an error with the tensor's name + dtype.
//! - Memory-mapped I/O — we read tensor bytes off disk into a
//!   temporary buffer, dequantize, and drop the bytes. For a 500M-param
//!   student that's ~1 GB of transient memory; fine on the trainer box.
//!
//! # Naming convention bridge
//! GGUF (llama.cpp) uses `blk.N.attn_q.weight`, `token_embd.weight`,
//! `output_norm.weight`, etc. This module's HF-compatible shim
//! translates into `model.layers.N.self_attn.q_proj.weight`,
//! `model.embed_tokens.weight`, `model.norm.weight` so the existing
//! `Qwen3ForCausalLM::load_weights` (HF-style) consumes the result.
//!
//! # Tech debt
//! Duplicates ~300 lines of dequant + parser code with
//! `nexus-serve/src/model/gguf.rs`. The clean future move is a shared
//! `axonml-gguf` crate that both nexus-serve and axonml-llm depend on.
//! Noted; not blocking.
//!
//! # File
//! `crates/axonml-llm/src/gguf_loader.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt};

use axonml_tensor::Tensor;

use crate::qwen3::{Qwen3Config, Qwen3ForCausalLM};

// =============================================================================
// GGUF core types
// =============================================================================

const GGUF_MAGIC: u32 = 0x4655_4747; // 'GGUF' LE

#[derive(Debug, Clone)]
#[allow(dead_code)] // GGUF spec variants — not all are destructured in current code.
enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            Self::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q4K = 12,
    Q6K = 14,
    BF16 = 30,
    Unknown = 255,
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            30 => Self::BF16,
            _ => Self::Unknown,
        }
    }

    fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q6K => 210,
            Self::Unknown => 1,
        }
    }

    fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q6K => 256,
            Self::Unknown => 1,
        }
    }
}

#[derive(Debug, Clone)]
struct GgufTensorInfo {
    name: String,
    dims: Vec<u64>,
    dtype: GgmlType,
    /// Offset relative to the *tensor-data start*, not the file start.
    offset: u64,
}

impl GgufTensorInfo {
    fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    fn total_bytes(&self) -> usize {
        let n = self.n_elements();
        let bs = self.dtype.block_size();
        let ts = self.dtype.type_size();
        (n / bs) * ts
    }
}

struct GgufFile {
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
    tensor_index: HashMap<String, usize>,
    data_offset: u64,
    path: PathBuf,
}

impl GgufFile {
    fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;

        let magic = file.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Not a GGUF file (magic 0x{magic:08x})"),
            ));
        }
        let version = file.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version {version}"),
            ));
        }

        let n_tensors = file.read_u64::<LittleEndian>()?;
        let n_metadata = file.read_u64::<LittleEndian>()?;

        let mut metadata = HashMap::with_capacity(n_metadata as usize);
        for _ in 0..n_metadata {
            let key = read_gguf_string(&mut file)?;
            let value = read_gguf_value(&mut file)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut tensor_index = HashMap::with_capacity(n_tensors as usize);
        for i in 0..n_tensors {
            let name = read_gguf_string(&mut file)?;
            let n_dims = file.read_u32::<LittleEndian>()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(file.read_u64::<LittleEndian>()?);
            }
            let dtype = GgmlType::from_u32(file.read_u32::<LittleEndian>()?);
            let offset = file.read_u64::<LittleEndian>()?;
            tensor_index.insert(name.clone(), i as usize);
            tensors.push(GgufTensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }

        // Align data-start to the GGUF `general.alignment` boundary (32 default).
        let current_pos = file.stream_position()?;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        let data_offset = current_pos.div_ceil(alignment) * alignment;

        Ok(Self {
            metadata,
            tensors,
            tensor_index,
            data_offset,
            path: path.to_path_buf(),
        })
    }

    fn get_meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }
}

fn read_gguf_string(reader: &mut impl Read) -> io::Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_gguf_value(reader: &mut impl Read) -> io::Result<GgufValue> {
    let vtype = reader.read_u32::<LittleEndian>()?;
    match vtype {
        0 => Ok(GgufValue::U8(reader.read_u8()?)),
        1 => Ok(GgufValue::I8(reader.read_i8()?)),
        2 => Ok(GgufValue::U16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::I16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::U32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::I32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::F32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(GgufValue::String(read_gguf_string(reader)?)),
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                let v = match elem_type {
                    0 => GgufValue::U8(reader.read_u8()?),
                    1 => GgufValue::I8(reader.read_i8()?),
                    2 => GgufValue::U16(reader.read_u16::<LittleEndian>()?),
                    3 => GgufValue::I16(reader.read_i16::<LittleEndian>()?),
                    4 => GgufValue::U32(reader.read_u32::<LittleEndian>()?),
                    5 => GgufValue::I32(reader.read_i32::<LittleEndian>()?),
                    6 => GgufValue::F32(reader.read_f32::<LittleEndian>()?),
                    7 => GgufValue::Bool(reader.read_u8()? != 0),
                    8 => GgufValue::String(read_gguf_string(reader)?),
                    10 => GgufValue::U64(reader.read_u64::<LittleEndian>()?),
                    11 => GgufValue::I64(reader.read_i64::<LittleEndian>()?),
                    12 => GgufValue::F64(reader.read_f64::<LittleEndian>()?),
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Unknown array element type {elem_type}"),
                        ));
                    }
                };
                arr.push(v);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufValue::I64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufValue::F64(reader.read_f64::<LittleEndian>()?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown GGUF value type {vtype}"),
        )),
    }
}

// =============================================================================
// Dequantization (CPU scalar — these match `nexus-serve/src/model/gguf.rs`)
// =============================================================================

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal — normalize left until bit 10 (0x400) is set.
            let mut e = -14i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FF;
            let exp32 = (127 + e) as u32;
            f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::NAN
        }
    } else {
        let exp32 = exp + 112; // 127 - 15
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    // BF16 is the top 16 bits of an f32 — zero-pad the bottom 16 to reconstruct.
    f32::from_bits((bits as u32) << 16)
}

fn dequantize_q4_0(block: &[u8], output: &mut [f32]) {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..16 {
        let byte = block[2 + i];
        let lo = (byte & 0x0F) as i8 - 8;
        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
        output[i * 2] = lo as f32 * scale;
        output[i * 2 + 1] = hi as f32 * scale;
    }
}

fn dequantize_q8_0(block: &[u8], output: &mut [f32]) {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..32 {
        output[i] = block[2 + i] as i8 as f32 * scale;
    }
}

fn dequantize_q4_k(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    #[inline]
    fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    let mut is = 0usize;
    let mut q_off = 0usize;
    let mut out_idx = 0usize;

    for _ in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;

        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;

        for l in 0..32 {
            output[out_idx] = d1 * (qs[q_off + l] & 0xF) as f32 - min1;
            out_idx += 1;
        }
        for l in 0..32 {
            output[out_idx] = d2 * ((qs[q_off + l] >> 4) & 0xF) as f32 - min2;
            out_idx += 1;
        }

        q_off += 32;
        is += 2;
    }
}

fn dequantize_q6_k(block: &[u8], output: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let mut ql_off = 0usize;
    let mut qh_off = 128usize;
    let mut sc_off = 192usize;
    let mut y_off = 0usize;

    for _ in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((block[ql_off + l] & 0xF) | (((block[qh_off + l]) & 3) << 4)) as i8 - 32;
            let q2 =
                ((block[ql_off + l + 32] & 0xF) | (((block[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((block[ql_off + l] >> 4) | (((block[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 =
                ((block[ql_off + l + 32] >> 4) | (((block[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;

            output[y_off + l] = d * block[sc_off + is] as i8 as f32 * q1 as f32;
            output[y_off + l + 32] = d * block[sc_off + is + 2] as i8 as f32 * q2 as f32;
            output[y_off + l + 64] = d * block[sc_off + is + 4] as i8 as f32 * q3 as f32;
            output[y_off + l + 96] = d * block[sc_off + is + 6] as i8 as f32 * q4 as f32;
        }
        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
}

fn dequantize_to_f32(info: &GgufTensorInfo, raw: &[u8]) -> io::Result<Vec<f32>> {
    let n = info.n_elements();
    let bs = info.dtype.block_size();
    let ts = info.dtype.type_size();

    match info.dtype {
        GgmlType::F32 => {
            let mut out = vec![0f32; n];
            for (i, c) in raw.chunks_exact(4).take(n).enumerate() {
                out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            Ok(out)
        }
        GgmlType::F16 => {
            let mut out = vec![0f32; n];
            for (i, c) in raw.chunks_exact(2).take(n).enumerate() {
                out[i] = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
            Ok(out)
        }
        GgmlType::BF16 => {
            let mut out = vec![0f32; n];
            for (i, c) in raw.chunks_exact(2).take(n).enumerate() {
                out[i] = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
            Ok(out)
        }
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K => {
            let mut out = vec![0f32; n];
            let n_blocks = n / bs;
            for b in 0..n_blocks {
                let block = &raw[b * ts..(b + 1) * ts];
                let out_chunk = &mut out[b * bs..(b + 1) * bs];
                match info.dtype {
                    GgmlType::Q4_0 => dequantize_q4_0(block, out_chunk),
                    GgmlType::Q8_0 => dequantize_q8_0(block, out_chunk),
                    GgmlType::Q4K => dequantize_q4_k(block, out_chunk),
                    GgmlType::Q6K => dequantize_q6_k(block, out_chunk),
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        GgmlType::Unknown => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported dtype for tensor `{}`", info.name),
        )),
    }
}

fn read_tensor_bytes(
    file: &mut File,
    data_offset: u64,
    info: &GgufTensorInfo,
) -> io::Result<Vec<u8>> {
    let abs = data_offset + info.offset;
    file.seek(SeekFrom::Start(abs))?;
    let mut buf = vec![0u8; info.total_bytes()];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

// =============================================================================
// Qwen3 loader
// =============================================================================

/// llama.cpp → HuggingFace tensor-name translator for Qwen3.
///
/// Qwen3 GGUFs use the llama.cpp convention (`blk.N.attn_q.weight`,
/// `token_embd.weight`, …). Our `Qwen3ForCausalLM::load_weights` consumes
/// HuggingFace-style names (`model.layers.N.self_attn.q_proj.weight`, …).
/// This bridge produces the HF name for every tensor in a Qwen3 GGUF.
fn ggml_to_hf_name(ggml_name: &str) -> Option<String> {
    // Top-level tensors.
    match ggml_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        _ => {}
    }

    // Per-block tensors: `blk.N.<kind>.weight` → `model.layers.N.<hf_kind>.weight`.
    let rest = ggml_name.strip_prefix("blk.")?;
    let (idx_str, tail) = rest.split_once('.')?;
    let idx: usize = idx_str.parse().ok()?;

    let hf_suffix = match tail {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };

    Some(format!("model.layers.{idx}.{hf_suffix}"))
}

/// Build a `Qwen3Config` from a parsed GGUF's metadata.
///
/// Reads architecture-prefixed hyperparameters (`qwen3.*`). Derives
/// `head_dim` from explicit `qwen3.attention.key_length` if present,
/// otherwise from `hidden_size / num_attention_heads`. Derives
/// `tie_word_embeddings` from the presence of a standalone
/// `output.weight` tensor — tied when absent.
fn qwen3_config_from_gguf(gguf: &GgufFile) -> io::Result<Qwen3Config> {
    fn m_u32(g: &GgufFile, key: &str) -> io::Result<u32> {
        g.get_meta(key).and_then(GgufValue::as_u32).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing metadata `{key}`"),
            )
        })
    }
    fn m_u32_or(g: &GgufFile, key: &str, default: u32) -> u32 {
        g.get_meta(key)
            .and_then(GgufValue::as_u32)
            .unwrap_or(default)
    }
    fn m_f32_or(g: &GgufFile, key: &str, default: f32) -> f32 {
        g.get_meta(key)
            .and_then(GgufValue::as_f32)
            .unwrap_or(default)
    }

    let arch = gguf
        .get_meta("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or("")
        .to_string();
    // Accept qwen2 + qwen3 under the same loader (Qwen3ForCausalLM struct) —
    // the two archs share tensor layout (GQA, RoPE, SwiGLU, RMSNorm, `blk.N.*`
    // names). R1-Distill-Qwen series ships as `qwen2`; a hard qwen3 guard was
    // the blocker for Oracle teacher-distill loads (LESSONS L91). Prefix-
    // matching covers future variants (qwen3.5 etc).
    let arch_ok = arch == "qwen2"
        || arch.starts_with("qwen2")
        || arch == "qwen3"
        || arch.starts_with("qwen3");
    if !arch_ok {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("not a Qwen2/Qwen3 GGUF — architecture = `{arch}`"),
        ));
    }

    // llama.cpp convention is to prefix arch-specific metadata keys with the
    // architecture name. Look up each field under the file's actual arch
    // rather than assuming `qwen3.*`.
    let prefix = if arch.starts_with("qwen2") {
        "qwen2"
    } else {
        "qwen3"
    };
    let k_embed = format!("{prefix}.embedding_length");
    let k_ffn = format!("{prefix}.feed_forward_length");
    let k_blocks = format!("{prefix}.block_count");
    let k_heads = format!("{prefix}.attention.head_count");
    let k_heads_kv = format!("{prefix}.attention.head_count_kv");
    let k_keylen = format!("{prefix}.attention.key_length");
    let k_ctx = format!("{prefix}.context_length");
    let k_rms_eps = format!("{prefix}.attention.layer_norm_rms_epsilon");
    let k_rope = format!("{prefix}.rope.freq_base");

    let hidden_size = m_u32(gguf, &k_embed)? as usize;
    let intermediate = m_u32(gguf, &k_ffn)? as usize;
    let num_layers = m_u32(gguf, &k_blocks)? as usize;
    let n_heads = m_u32(gguf, &k_heads)? as usize;
    let n_kv_heads = m_u32_or(gguf, &k_heads_kv, n_heads as u32) as usize;

    // Prefer explicit key_length; fall back to hidden / n_heads.
    let head_dim = m_u32_or(gguf, &k_keylen, (hidden_size / n_heads) as u32) as usize;

    let context_len = m_u32_or(gguf, &k_ctx, 32_768) as usize;
    let rms_eps = m_f32_or(gguf, &k_rms_eps, 1e-6);
    let rope_theta = m_f32_or(gguf, &k_rope, 1_000_000.0);

    // Vocab size from the token embedding tensor's dims — dims[1] is the
    // num_embeddings in GGUF's (dim, n_emb) layout.
    let tok_embd = gguf
        .tensor("token_embd.weight")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing `token_embd.weight`"))?;
    if tok_embd.dims.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("`token_embd.weight` must be 2D, got {:?}", tok_embd.dims),
        ));
    }
    let vocab_size = tok_embd.dims[1] as usize;

    // LM head tying: if there's no `output.weight`, weights are tied to embeddings.
    let tie_word_embeddings = gguf.tensor("output.weight").is_none();

    Ok(Qwen3Config {
        vocab_size,
        hidden_size,
        intermediate_size: intermediate,
        num_hidden_layers: num_layers,
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv_heads,
        head_dim,
        max_position_embeddings: context_len,
        rms_norm_eps: rms_eps,
        rope_theta,
        attention_dropout: 0.0,
        hidden_dropout: 0.0,
        tie_word_embeddings,
    })
}

/// Read raw byte-encoded GGUF metadata entries for `keys` from `path`.
///
/// Returns `HashMap<key, encoded_kv_bytes>` where `encoded_kv_bytes` is
/// the complete GGUF-format serialization of the `(key_string_with_len,
/// value_type, value_body)` triple — exactly the bytes that would
/// appear in the metadata section of the source GGUF for that entry.
///
/// The returned bytes can be spliced verbatim into the metadata section
/// of a new GGUF. This is how the exporter copies tokenizer state
/// (`tokenizer.ggml.*`) without re-serializing every value type by hand.
///
/// Unknown / malformed values cause an `InvalidData` error rather than
/// silent skipping — a caller that requested a key needs to know if the
/// source file can't produce it.
pub fn read_gguf_metadata_raw_bytes(
    path: &Path,
    keys: &[&str],
) -> io::Result<HashMap<String, Vec<u8>>> {
    use std::io::{BufReader, Seek, SeekFrom};

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // --- Parse header up to the metadata table. ---
    let magic = reader.read_u32::<LittleEndian>()?;
    if magic != GGUF_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Not a GGUF file (magic 0x{magic:08x})"),
        ));
    }
    let version = reader.read_u32::<LittleEndian>()?;
    if !(2..=3).contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF version {version}"),
        ));
    }
    let _n_tensors = reader.read_u64::<LittleEndian>()?;
    let n_metadata = reader.read_u64::<LittleEndian>()?;

    let want: std::collections::HashSet<&str> = keys.iter().copied().collect();
    let mut result: HashMap<String, Vec<u8>> = HashMap::with_capacity(want.len());

    for _ in 0..n_metadata {
        let entry_start = reader.stream_position()?;

        // Parse key + value to advance the reader; we re-seek later to
        // read the bytes back if this entry matched.
        let key = read_gguf_string(&mut reader)?;
        let _value = read_gguf_value(&mut reader)?;
        let entry_end = reader.stream_position()?;

        if want.contains(key.as_str()) {
            let len = (entry_end - entry_start) as usize;
            let mut buf = vec![0u8; len];
            reader.seek(SeekFrom::Start(entry_start))?;
            reader.read_exact(&mut buf)?;
            // Reader is now at entry_end again, ready for the next iteration.
            result.insert(key, buf);
        }
    }

    Ok(result)
}

/// `(tokens, merges)` — tokens keyed by id; merges as `(left, right)` pairs.
pub type TokenizerData = (Vec<String>, Vec<(String, String)>);

/// Read Qwen-family GGUF tokenizer metadata: `(tokens, merges)`.
///
/// Returns the byte-level BPE vocabulary as `Vec<String>` (token[i] is
/// the string for token ID `i`) and the merge rules as
/// `Vec<(String, String)>`. Returns an empty merges vector for GGUFs
/// that don't ship merges (some SentencePiece exports omit them; in
/// that case the caller falls back to greedy-longest-match encoding).
pub fn read_gguf_tokenizer(path: &Path) -> io::Result<TokenizerData> {
    let gguf = GgufFile::open(path)?;

    let tokens: Vec<String> = match gguf.get_meta("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                GgufValue::String(s) => s.clone(),
                _ => String::new(),
            })
            .collect(),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing or malformed `tokenizer.ggml.tokens`",
            ));
        }
    };

    let merges: Vec<(String, String)> = match gguf.get_meta("tokenizer.ggml.merges") {
        Some(GgufValue::Array(arr)) => arr
            .iter()
            .filter_map(|v| match v {
                GgufValue::String(s) => {
                    let parts: Vec<&str> = s.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };

    Ok((tokens, merges))
}

/// Load a Qwen3-family GGUF into a fresh, trainable `Qwen3ForCausalLM`.
///
/// Steps:
/// 1. Parse GGUF header + tensor directory.
/// 2. Derive `Qwen3Config` from metadata; construct fresh model.
/// 3. For every GGUF tensor, dequantize to f32, build an AxonML `Tensor`
///    with the correct `[out, in]` shape, map the GGUF name to its HF
///    equivalent, and stash into a state dict.
/// 4. Call `Qwen3ForCausalLM::load_weights` with the state dict.
///
/// Returns the loaded model and the derived config (useful for the
/// caller to echo dims into logs / checkpoints).
pub fn load_qwen3_from_gguf(path: &Path) -> io::Result<(Qwen3ForCausalLM, Qwen3Config)> {
    let gguf = GgufFile::open(path)?;
    let config = qwen3_config_from_gguf(&gguf)?;
    let mut model = Qwen3ForCausalLM::new(&config);

    let mut state_dict: HashMap<String, Tensor<f32>> = HashMap::with_capacity(gguf.tensors.len());
    let mut file = File::open(&gguf.path)?;

    for info in &gguf.tensors {
        let hf_name = match ggml_to_hf_name(&info.name) {
            Some(name) => name,
            None => {
                // Unknown / unmapped tensor (e.g. tokenizer-related
                // metadata tensors sometimes present in GGUF). Skip
                // quietly — load_weights won't look for it.
                continue;
            }
        };

        let raw = read_tensor_bytes(&mut file, gguf.data_offset, info)?;
        let flat = dequantize_to_f32(info, &raw)?;

        // AxonML shape convention for 2D weight matrices is [out, in];
        // GGUF dims are [in, out] (dims[0] = n_cols = in, dims[1] = n_rows = out).
        // 1D tensors (bias, norm weight) stay as-is.
        let shape: Vec<usize> = match info.dims.len() {
            1 => vec![info.dims[0] as usize],
            2 => vec![info.dims[1] as usize, info.dims[0] as usize],
            n => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Unsupported tensor rank {n} for `{}`; expected 1 or 2",
                        info.name
                    ),
                ));
            }
        };

        let t = Tensor::from_vec(flat, &shape).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Tensor::from_vec failed for `{}`: {e:?}", info.name),
            )
        })?;
        state_dict.insert(hf_name, t);
    }

    let loaded = model.load_weights(&state_dict);
    if loaded == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "load_weights matched zero tensors — name mapping likely broken",
        ));
    }

    Ok((model, config))
}

// =============================================================================
// Tests (structural — don't need a real GGUF to run)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_to_hf_mapping() {
        assert_eq!(
            ggml_to_hf_name("token_embd.weight"),
            Some("model.embed_tokens.weight".to_string())
        );
        assert_eq!(
            ggml_to_hf_name("output_norm.weight"),
            Some("model.norm.weight".to_string())
        );
        assert_eq!(
            ggml_to_hf_name("output.weight"),
            Some("lm_head.weight".to_string())
        );
        assert_eq!(
            ggml_to_hf_name("blk.5.attn_q.weight"),
            Some("model.layers.5.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(
            ggml_to_hf_name("blk.12.attn_q_norm.weight"),
            Some("model.layers.12.self_attn.q_norm.weight".to_string())
        );
        assert_eq!(
            ggml_to_hf_name("blk.3.ffn_gate.weight"),
            Some("model.layers.3.mlp.gate_proj.weight".to_string())
        );
        assert_eq!(
            ggml_to_hf_name("blk.7.ffn_down.weight"),
            Some("model.layers.7.mlp.down_proj.weight".to_string())
        );
        // Unknown tensors map to None (caller skips them).
        assert!(ggml_to_hf_name("some.random.tensor").is_none());
    }

    #[test]
    fn test_f16_roundtrip() {
        // 1.0 in f16 is 0x3C00.
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0xBC00) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_roundtrip() {
        // 1.0 in bf16 is the top 16 bits of 1.0f32 = 0x3F800000 >> 16 = 0x3F80.
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }
}
