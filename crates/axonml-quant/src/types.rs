//! Quantization Types
//!
//! Defines quantization formats and data structures.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::fmt;
use half::f16;

// =============================================================================
// Quantization Type Enum
// =============================================================================

/// Quantization type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    /// 8-bit quantization with per-block scale.
    /// Format: scale (f16) + 32 x int8
    Q8_0,

    /// 4-bit quantization with per-block scale.
    /// Format: scale (f16) + 16 x uint8 (two 4-bit values each)
    Q4_0,

    /// 4-bit quantization with per-block scale and min.
    /// Format: scale (f16) + min (f16) + 16 x uint8
    Q4_1,

    /// 5-bit quantization with per-block scale.
    Q5_0,

    /// 5-bit quantization with per-block scale and min.
    Q5_1,

    /// Half-precision (16-bit float).
    F16,

    /// Full precision (32-bit float).
    F32,
}

impl QuantType {
    /// Returns the block size for this quantization type.
    pub fn block_size(&self) -> usize {
        match self {
            QuantType::Q8_0 | QuantType::Q4_0 | QuantType::Q4_1 |
            QuantType::Q5_0 | QuantType::Q5_1 => 32,
            QuantType::F16 | QuantType::F32 => 1,
        }
    }

    /// Returns the number of bytes per block.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            QuantType::Q8_0 => 2 + 32,      // f16 scale + 32 int8
            QuantType::Q4_0 => 2 + 16,      // f16 scale + 16 bytes (32 x 4-bit)
            QuantType::Q4_1 => 4 + 16,      // f16 scale + f16 min + 16 bytes
            QuantType::Q5_0 => 2 + 20,      // f16 scale + 20 bytes (32 x 5-bit)
            QuantType::Q5_1 => 4 + 20,      // f16 scale + f16 min + 20 bytes
            QuantType::F16 => 2,
            QuantType::F32 => 4,
        }
    }

    /// Returns the bits per value.
    pub fn bits_per_value(&self) -> usize {
        match self {
            QuantType::Q8_0 => 8,
            QuantType::Q4_0 | QuantType::Q4_1 => 4,
            QuantType::Q5_0 | QuantType::Q5_1 => 5,
            QuantType::F16 => 16,
            QuantType::F32 => 32,
        }
    }

    /// Returns the compression ratio compared to F32.
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits_per_value() as f32
    }

    /// Returns true if this type uses block quantization.
    pub fn is_block_quantized(&self) -> bool {
        matches!(self, QuantType::Q8_0 | QuantType::Q4_0 | QuantType::Q4_1 |
                       QuantType::Q5_0 | QuantType::Q5_1)
    }

    /// Parses a quantization type from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "Q8_0" | "Q8" | "INT8" => Some(QuantType::Q8_0),
            "Q4_0" | "Q4" | "INT4" => Some(QuantType::Q4_0),
            "Q4_1" => Some(QuantType::Q4_1),
            "Q5_0" | "Q5" => Some(QuantType::Q5_0),
            "Q5_1" => Some(QuantType::Q5_1),
            "F16" | "FLOAT16" | "HALF" => Some(QuantType::F16),
            "F32" | "FLOAT32" | "FLOAT" => Some(QuantType::F32),
            _ => None,
        }
    }
}

impl fmt::Display for QuantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantType::Q8_0 => write!(f, "Q8_0"),
            QuantType::Q4_0 => write!(f, "Q4_0"),
            QuantType::Q4_1 => write!(f, "Q4_1"),
            QuantType::Q5_0 => write!(f, "Q5_0"),
            QuantType::Q5_1 => write!(f, "Q5_1"),
            QuantType::F16 => write!(f, "F16"),
            QuantType::F32 => write!(f, "F32"),
        }
    }
}

// =============================================================================
// Quantized Block Structures
// =============================================================================

/// A block of Q8_0 quantized data.
#[derive(Debug, Clone)]
pub struct Q8Block {
    /// Scale factor (stored as f16).
    pub scale: f16,
    /// Quantized values (32 x int8).
    pub data: [i8; 32],
}

impl Q8Block {
    /// Creates a new Q8 block.
    pub fn new(scale: f16, data: [i8; 32]) -> Self {
        Self { scale, data }
    }

    /// Returns the byte representation of this block.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(34);
        bytes.extend_from_slice(&self.scale.to_le_bytes());
        bytes.extend(self.data.iter().map(|&x| x as u8));
        bytes
    }

    /// Creates a block from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 34 {
            return None;
        }
        let scale = f16::from_le_bytes([bytes[0], bytes[1]]);
        let mut data = [0i8; 32];
        for (i, &b) in bytes[2..34].iter().enumerate() {
            data[i] = b as i8;
        }
        Some(Self { scale, data })
    }
}

/// A block of Q4_0 quantized data.
#[derive(Debug, Clone)]
pub struct Q4Block {
    /// Scale factor (stored as f16).
    pub scale: f16,
    /// Packed quantized values (16 bytes = 32 x 4-bit).
    pub data: [u8; 16],
}

impl Q4Block {
    /// Creates a new Q4 block.
    pub fn new(scale: f16, data: [u8; 16]) -> Self {
        Self { scale, data }
    }

    /// Extracts the 4-bit values as i8 (range -8 to 7).
    pub fn unpack(&self) -> [i8; 32] {
        let mut result = [0i8; 32];
        for i in 0..16 {
            let byte = self.data[i];
            result[i * 2] = ((byte & 0x0F) as i8) - 8;
            result[i * 2 + 1] = ((byte >> 4) as i8) - 8;
        }
        result
    }

    /// Packs 32 i8 values (-8 to 7 range) into 16 bytes.
    pub fn pack(values: &[i8; 32]) -> [u8; 16] {
        let mut data = [0u8; 16];
        for i in 0..16 {
            let low = ((values[i * 2] + 8) as u8) & 0x0F;
            let high = ((values[i * 2 + 1] + 8) as u8) & 0x0F;
            data[i] = low | (high << 4);
        }
        data
    }

    /// Returns the byte representation of this block.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(18);
        bytes.extend_from_slice(&self.scale.to_le_bytes());
        bytes.extend_from_slice(&self.data);
        bytes
    }

    /// Creates a block from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 18 {
            return None;
        }
        let scale = f16::from_le_bytes([bytes[0], bytes[1]]);
        let mut data = [0u8; 16];
        data.copy_from_slice(&bytes[2..18]);
        Some(Self { scale, data })
    }
}

/// A block of Q4_1 quantized data (with min value).
#[derive(Debug, Clone)]
pub struct Q4_1Block {
    /// Scale factor (stored as f16).
    pub scale: f16,
    /// Minimum value (stored as f16).
    pub min: f16,
    /// Packed quantized values (16 bytes = 32 x 4-bit).
    pub data: [u8; 16],
}

impl Q4_1Block {
    /// Creates a new Q4_1 block.
    pub fn new(scale: f16, min: f16, data: [u8; 16]) -> Self {
        Self { scale, min, data }
    }

    /// Extracts the 4-bit values as u8 (range 0 to 15).
    pub fn unpack(&self) -> [u8; 32] {
        let mut result = [0u8; 32];
        for i in 0..16 {
            let byte = self.data[i];
            result[i * 2] = byte & 0x0F;
            result[i * 2 + 1] = byte >> 4;
        }
        result
    }

    /// Returns the byte representation of this block.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&self.scale.to_le_bytes());
        bytes.extend_from_slice(&self.min.to_le_bytes());
        bytes.extend_from_slice(&self.data);
        bytes
    }
}

// =============================================================================
// Generic Quantized Block
// =============================================================================

/// Generic quantized block enum.
#[derive(Debug, Clone)]
pub enum QuantizedBlock {
    /// Q8_0 block.
    Q8(Q8Block),
    /// Q4_0 block.
    Q4(Q4Block),
    /// Q4_1 block.
    Q4_1(Q4_1Block),
    /// F16 values (block size 1).
    F16(Vec<f16>),
    /// F32 values (original).
    F32(Vec<f32>),
}

impl QuantizedBlock {
    /// Returns the quantization type of this block.
    pub fn quant_type(&self) -> QuantType {
        match self {
            QuantizedBlock::Q8(_) => QuantType::Q8_0,
            QuantizedBlock::Q4(_) => QuantType::Q4_0,
            QuantizedBlock::Q4_1(_) => QuantType::Q4_1,
            QuantizedBlock::F16(_) => QuantType::F16,
            QuantizedBlock::F32(_) => QuantType::F32,
        }
    }
}

// =============================================================================
// Quantized Tensor
// =============================================================================

/// A quantized tensor containing compressed weight data.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Quantization type.
    pub quant_type: QuantType,
    /// Quantized data blocks.
    pub blocks: Vec<QuantizedBlock>,
    /// Number of elements.
    pub numel: usize,
}

impl QuantizedTensor {
    /// Creates a new quantized tensor.
    pub fn new(shape: Vec<usize>, quant_type: QuantType, blocks: Vec<QuantizedBlock>) -> Self {
        let numel = shape.iter().product();
        Self {
            shape,
            quant_type,
            blocks,
            numel,
        }
    }

    /// Returns the memory size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * self.quant_type.bytes_per_block()
    }

    /// Returns the compression ratio compared to F32.
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.numel * 4;
        original_bytes as f32 / self.size_bytes() as f32
    }

    /// Returns the number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_properties() {
        assert_eq!(QuantType::Q8_0.block_size(), 32);
        assert_eq!(QuantType::Q4_0.block_size(), 32);
        assert_eq!(QuantType::F16.block_size(), 1);

        assert_eq!(QuantType::Q8_0.bits_per_value(), 8);
        assert_eq!(QuantType::Q4_0.bits_per_value(), 4);

        assert!(QuantType::Q8_0.is_block_quantized());
        assert!(!QuantType::F16.is_block_quantized());
    }

    #[test]
    fn test_quant_type_from_str() {
        assert_eq!(QuantType::from_str("Q8_0"), Some(QuantType::Q8_0));
        assert_eq!(QuantType::from_str("INT8"), Some(QuantType::Q8_0));
        assert_eq!(QuantType::from_str("Q4"), Some(QuantType::Q4_0));
        assert_eq!(QuantType::from_str("F16"), Some(QuantType::F16));
        assert_eq!(QuantType::from_str("invalid"), None);
    }

    #[test]
    fn test_q4_pack_unpack() {
        let values: [i8; 32] = [
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
        ];

        let packed = Q4Block::pack(&values);
        let block = Q4Block::new(f16::from_f32(1.0), packed);
        let unpacked = block.unpack();

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_q8_block() {
        let data = [0i8; 32];
        let block = Q8Block::new(f16::from_f32(0.5), data);
        let bytes = block.to_bytes();
        let restored = Q8Block::from_bytes(&bytes).unwrap();

        assert_eq!(block.scale, restored.scale);
        assert_eq!(block.data, restored.data);
    }
}
