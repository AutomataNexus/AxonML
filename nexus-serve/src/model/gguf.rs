//! gguf — GGUF Container Parser + Block Dequant Helpers
//!
//! Parses the GGUF (GGML Universal Format) v2/v3 container — the on-disk
//! shape used by nexus-serve, llama.cpp, ollama, and most local LLM tools.
//! Streams the header only (metadata KV + tensor directory); weight bytes
//! stay on disk for later mmap in `inference::MappedGguf`.
//!
//! Types:
//! - [`GgufFile`]: parsed header (`version`, `n_tensors`, `metadata`,
//!   `tensors`, `data_offset`, `path`) plus helpers like `architecture`,
//!   `model_name`, `get_meta`, `total_tensor_bytes`, `summary`.
//! - [`GgufValue`]: tagged-union for every GGUF scalar/array value with
//!   coercion helpers (`as_u32`, `as_u64`, `as_f32`, `as_str`, `as_bool`).
//! - [`GgmlType`]: every GGML quant code including Q4_0/Q4_1, Q5_0/Q5_1,
//!   Q8_0/Q8_1, Q2K/Q3K/Q4K/Q5K/Q6K/Q8K, IQ* family, F16/BF16, and the
//!   BitNet b1.58 ternary `I2S` (dtype 36, 66-byte / 256-elem blocks).
//!   `type_size()` and `block_size()` report the on-disk layout.
//! - [`GgufTensorInfo`]: one tensor's name, shape, dtype, and byte offset;
//!   `n_elements()` and `total_bytes()` for size math.
//!
//! Private helpers [`read_gguf_string`] and [`read_gguf_value`] implement
//! the recursive string/scalar/array decoder.
//!
//! Dequantization kernels (CPU, scalar, used for both non-quantized weight
//! baking and reference checks against the CUDA kernels in
//! `axonml_quant`): [`dequantize_q4_0`], [`dequantize_q8_0`],
//! [`dequantize_q4_k`] (exact port of llama.cpp's `dequantize_row_q4_K` with
//! the `get_scale_min_k4` 6-bit packed scales), [`dequantize_q6_k`] (port
//! of `dequantize_row_q6_K` with ql/qh/sc unpacking), [`dequantize_f16`],
//! and the IEEE `f16_to_f32` converter (subnormal-safe, which was GGUF
//! inference bug #1 from memory — see `reference_gguf_inference_gotchas`).
//!
//! Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//!
//! # File
//! `nexus-serve/src/model/gguf.rs`
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
use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

// =============================================================================
// GGUF Magic + Versions
// =============================================================================

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as u32 little-endian (bytes: 47 47 55 46)

// =============================================================================
// GGUF Value Types
// =============================================================================

#[derive(Debug, Clone)]
pub enum GgufValue {
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
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            Self::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

// =============================================================================
// GGUF Quantization Types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    /// BitNet b1.58 ternary weights (Microsoft bitnet.cpp, dtype 36).
    /// 256-element blocks × (f16 scale + 64 data bytes) = 66 bytes per block.
    /// See `axonml_quant::bitnet` for the dequant + fused-matmul kernels.
    I2S = 36,
    Unknown = 255,
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            36 => Self::I2S,
            _ => Self::Unknown,
        }
    }

    /// Bytes per element for this quantization type (for non-block types).
    /// Block-quantized types return the block size.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,   // block of 32: 32*4/8 + 2 = 18 bytes
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,   // block of 32: 32 + 2 = 34 bytes
            Self::Q8_1 => 36,
            Self::Q2K => 256,
            Self::Q3K => 256,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::I2S => 32, // 128 × 2-bit trits, no per-block scale (one f32 per tensor at tail)
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::F64 => 8,
            _ => 1,
        }
    }

    /// Block size (number of elements per quantization block).
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1
            | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K
            | Self::Q6K | Self::Q8K => 256,
            Self::I2S => 128,
            _ => 1,
        }
    }
}

// =============================================================================
// GGUF Tensor Info
// =============================================================================

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    /// Total bytes for the quantized representation.
    pub fn total_bytes(&self) -> u64 {
        let n = self.n_elements();
        let bs = self.dtype.block_size() as u64;
        let ts = self.dtype.type_size() as u64;
        (n / bs) * ts
    }
}

// =============================================================================
// GGUF File
// =============================================================================

/// Parsed GGUF file header — metadata + tensor index (no weight data loaded).
pub struct GgufFile {
    pub version: u32,
    pub n_tensors: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset where tensor data begins in the file.
    pub data_offset: u64,
    /// Path to the GGUF file (for mmap).
    pub path: std::path::PathBuf,
}

impl GgufFile {
    /// Parse a GGUF file header. Does NOT load tensor data into memory.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;

        // Magic
        let magic = file.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Not a GGUF file (magic: 0x{:08x}, expected 0x{:08x})", magic, GGUF_MAGIC),
            ));
        }

        // Version
        let version = file.read_u32::<LittleEndian>()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version: {} (supported: 2, 3)", version),
            ));
        }

        // Counts
        let n_tensors = file.read_u64::<LittleEndian>()?;
        let n_metadata = file.read_u64::<LittleEndian>()?;

        // Metadata
        let mut metadata = HashMap::with_capacity(n_metadata as usize);
        for _ in 0..n_metadata {
            let key = read_gguf_string(&mut file)?;
            let value = read_gguf_value(&mut file)?;
            metadata.insert(key, value);
        }

        // Tensor info
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_gguf_string(&mut file)?;
            let n_dims = file.read_u32::<LittleEndian>()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(file.read_u64::<LittleEndian>()?);
            }
            let dtype = GgmlType::from_u32(file.read_u32::<LittleEndian>()?);
            let offset = file.read_u64::<LittleEndian>()?;
            tensors.push(GgufTensorInfo {
                name,
                n_dims,
                dims,
                dtype,
                offset,
            });
        }

        // Data starts at aligned position after header
        let current_pos = file.stream_position()?;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as u64;
        let data_offset = (current_pos + alignment - 1) / alignment * alignment;

        Ok(GgufFile {
            version,
            n_tensors,
            metadata,
            tensors,
            data_offset,
            path: path.to_path_buf(),
        })
    }

    /// Get a metadata value by key.
    pub fn get_meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get architecture name (e.g., "llama", "gemma", "qwen2").
    pub fn architecture(&self) -> Option<&str> {
        self.get_meta("general.architecture")?.as_str()
    }

    /// Get model name.
    pub fn model_name(&self) -> Option<&str> {
        self.get_meta("general.name")?.as_str()
    }

    /// Total size of all tensor data in bytes.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.total_bytes()).sum()
    }

    /// Print a summary of the GGUF file.
    pub fn summary(&self) {
        println!("GGUF v{}", self.version);
        if let Some(name) = self.model_name() {
            println!("  Model: {}", name);
        }
        if let Some(arch) = self.architecture() {
            println!("  Architecture: {}", arch);
        }
        println!("  Tensors: {}", self.n_tensors);
        println!("  Metadata keys: {}", self.metadata.len());
        println!(
            "  Total tensor data: {:.1} GB",
            self.total_tensor_bytes() as f64 / 1e9
        );
        println!("  Data offset: 0x{:x}", self.data_offset);

        // Print key hyperparameters
        for key in &[
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.vocab_size",
            "gemma.context_length",
            "gemma.embedding_length",
            "gemma.block_count",
            "qwen2.context_length",
            "qwen2.embedding_length",
            "qwen2.block_count",
        ] {
            if let Some(val) = self.get_meta(key) {
                println!("  {}: {:?}", key, val);
            }
        }
    }
}

// =============================================================================
// GGUF Parsing Helpers
// =============================================================================

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
            // Array
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                // Read each element as the declared type
                let val = match elem_type {
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
                            format!("Unknown array element type: {}", elem_type),
                        ))
                    }
                };
                arr.push(val);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufValue::I64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufValue::F64(reader.read_f64::<LittleEndian>()?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown GGUF value type: {}", vtype),
        )),
    }
}

// =============================================================================
// Dequantization
// =============================================================================

/// Dequantize a Q4_0 block (32 elements: 2-byte scale + 16 bytes of 4-bit pairs).
pub fn dequantize_q4_0(block: &[u8], output: &mut [f32]) {
    debug_assert!(block.len() >= 18);
    debug_assert!(output.len() >= 32);

    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    for i in 0..16 {
        let byte = block[2 + i];
        let lo = (byte & 0x0F) as i8 - 8;
        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
        output[i * 2] = lo as f32 * scale;
        output[i * 2 + 1] = hi as f32 * scale;
    }
}

/// Dequantize a Q8_0 block (32 elements: 2-byte scale + 32 bytes of int8).
pub fn dequantize_q8_0(block: &[u8], output: &mut [f32]) {
    debug_assert!(block.len() >= 34);
    debug_assert!(output.len() >= 32);

    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    for i in 0..32 {
        output[i] = block[2 + i] as i8 as f32 * scale;
    }
}

/// Convert f16 bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal f16: value = frac / 2^24
            // Normalize by shifting left until bit 10 (0x400) is set.
            // The subnormal f16 exponent is -14; each left shift decreases
            // the effective exponent by 1. Start e at -14 so that exp32 = 127 + e
            // gives the correct biased f32 exponent after normalization.
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

/// Dequantize a Q4_K block (256 elements per super-block).
///
/// Q4_K layout per super-block (144 bytes for 256 elements):
///   - 2 bytes: d (f16 super-block scale)
///   - 2 bytes: dmin (f16 super-block min)
///   - 12 bytes: packed scales+mins for 8 sub-blocks (6 bits each)
///   - 128 bytes: 4-bit quantized values (256 / 2 = 128 bytes)
///
/// Reference: llama.cpp ggml-quants.c dequantize_row_q4_K
pub fn dequantize_q4_k(block: &[u8], output: &mut [f32]) {
    if block.len() < 144 || output.len() < 256 {
        return;
    }

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    // Exact port of ggml dequantize_row_q4_K + get_scale_min_k4.
    //
    // Key: the loop processes 64 elements at a time. Within each 64-element chunk,
    // the LOW nibble of each byte uses scale (d1, m1) and the HIGH nibble uses a
    // DIFFERENT scale (d2, m2). The `is` index increments by 2 per 64-element chunk.
    let scales = &block[4..16]; // 12 bytes packed scales+mins
    let qs = &block[16..144];   // 128 bytes, 4-bit quantized values

    // get_scale_min_k4(j, scales) → (scale, min)
    #[inline]
    fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4),
            )
        }
    }


    let mut is = 0usize;
    let mut q_offset = 0usize; // offset into qs[]
    let mut out_idx = 0usize;  // offset into output[]

    // 4 chunks of 64 elements = 256 total
    for _chunk in 0..4 {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;

        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;

        // First 32 values: low nibbles with (d1, min1)
        for l in 0..32 {
            if out_idx < 256 {
                output[out_idx] = d1 * (qs[q_offset + l] & 0xF) as f32 - min1;
            }
            out_idx += 1;
        }
        // Next 32 values: high nibbles with (d2, min2)
        for l in 0..32 {
            if out_idx < 256 {
                output[out_idx] = d2 * ((qs[q_offset + l] >> 4) & 0xF) as f32 - min2;
            }
            out_idx += 1;
        }

        q_offset += 32;
        is += 2;
    }
}

/// Dequantize a Q6_K block (256 elements per super-block).
///
/// Exact port of ggml dequantize_row_q6_K.
///
/// Q6_K layout per super-block (210 bytes for 256 elements):
///   - 128 bytes (ql): low 4 bits of quantized values
///   - 64 bytes (qh): high 2 bits of quantized values
///   - 16 bytes (sc): int8 scales per 16-element group
///   - 2 bytes: d (f16 super-block scale)
pub fn dequantize_q6_k(block: &[u8], output: &mut [f32]) {
    if block.len() < 210 || output.len() < 256 {
        return;
    }

    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    let mut ql_off = 0usize;  // offset into ql (block[0..128])
    let mut qh_off = 128usize; // offset into qh (block[128..192])
    let mut sc_off = 192usize; // offset into sc (block[192..208])
    let mut y_off = 0usize;

    // 2 chunks of 128 elements = 256 total
    for _n in 0..2 {
        for l in 0..32 {
            let is = l / 16;

            let q1 = ((block[ql_off + l] & 0xF) | (((block[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
            let q2 = ((block[ql_off + l + 32] & 0xF) | (((block[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((block[ql_off + l] >> 4) | (((block[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((block[ql_off + l + 32] >> 4) | (((block[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;

            output[y_off + l]      = d * block[sc_off + is] as i8 as f32 * q1 as f32;
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

/// Dequantize F16 values to F32.
pub fn dequantize_f16(data: &[u8], output: &mut [f32]) {
    for (i, chunk) in data.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        output[i] = f16_to_f32(bits);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // -1.0
        assert!((f16_to_f32(0xBC00) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_q8_0() {
        let mut block = vec![0u8; 34];
        // Scale = 1.0 in f16
        block[0] = 0x00;
        block[1] = 0x3C;
        // Values: 1, 2, 3, ...
        for i in 0..32 {
            block[2 + i] = (i + 1) as u8;
        }
        let mut output = vec![0.0f32; 32];
        dequantize_q8_0(&block, &mut output);
        assert!((output[0] - 1.0).abs() < 0.01);
        assert!((output[31] - 32.0).abs() < 0.01);
    }
}
