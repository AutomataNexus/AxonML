//! q4k_block_test — Q4_K Dequant Golden-Block Test
//!
//! Single integration test [`test_dequantize_q4_k_block_8987`] that feeds a
//! hard-coded 144-byte Q4_K super-block (block 8987 from Qwen2.5 Coder 1.5B's
//! `blk.0.attn_q.weight`) into [`dequantize_q4_k`] and asserts that the first
//! 10 output values and the block-wide max-abs match a Python reference (via
//! ggml's canonical dequantize_row_q4_K). Guards against regressions in the
//! 6-bit packed scales/mins unpacking (`get_scale_min_k4`) and the
//! 64-element-chunk low/high nibble routing.
//!
//! Python reference values (from the same block bytes):
//!   First 10 values: [-0.01314712, 0.00270653, -0.0004642, 0.01221871,
//!                      0.00904799, 0.01538944, 0.00270653, 0.00587726,
//!                      0.00270653, 0.00270653]
//!   Block max abs: 0.036473
//!   Block mag: 0.2012
//!
//! # File
//! `nexus-serve/tests/q4k_block_test.rs`
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

use nexus_serve::model::gguf::dequantize_q4_k;

// =============================================================================
// Tests
// =============================================================================

#[test]
fn test_dequantize_q4_k_block_8987() {
    let block: [u8; 144] = [
        0xFF, 0x03, 0x20, 0x10, 0xF4, 0xFF, 0xEF, 0xEF,
        0xED, 0xAA, 0xDB, 0xE2, 0x8E, 0x8B, 0xFB, 0xEB,
        0x33, 0x98, 0xB7, 0x4B, 0x0A, 0x0C, 0xF8, 0x49,
        0x58, 0x28, 0x0B, 0x52, 0x4F, 0x48, 0x6A, 0x8F,
        0x2A, 0x2B, 0x5B, 0x7A, 0x35, 0x3B, 0x15, 0x92,
        0x26, 0xC5, 0x90, 0xD5, 0x8B, 0x07, 0xCA, 0x56,
        0xC6, 0xF1, 0xC7, 0xEB, 0x73, 0x71, 0xC7, 0x96,
        0x73, 0x74, 0x90, 0x75, 0x13, 0x52, 0x16, 0x39,
        0x03, 0xB6, 0xD5, 0x9D, 0x97, 0x94, 0x3B, 0x86,
        0x2A, 0x93, 0xA7, 0x44, 0xCE, 0x23, 0x13, 0x45,
        0x4C, 0x56, 0x26, 0x73, 0x33, 0x81, 0x6B, 0x07,
        0x1E, 0x17, 0x68, 0x83, 0x70, 0x56, 0x46, 0xBC,
        0x19, 0x38, 0x44, 0xE7, 0x6D, 0x9A, 0xB7, 0xA7,
        0x96, 0x54, 0x26, 0x78, 0x69, 0x66, 0x74, 0x88,
        0x66, 0x77, 0x68, 0xDB, 0xB7, 0x78, 0x78, 0xE7,
        0xB9, 0x2F, 0x63, 0x68, 0xC5, 0x09, 0x08, 0x96,
        0xED, 0xD9, 0xCD, 0x7A, 0xB7, 0x40, 0x7C, 0xA7,
        0x99, 0x2A, 0x7D, 0x59, 0x45, 0xC5, 0xA6, 0x5D,
    ];

    let expected_first_10: [f32; 10] = [
        -0.01314712, 0.00270653, -0.0004642, 0.01221871,
        0.00904799, 0.01538944, 0.00270653, 0.00587726,
        0.00270653, 0.00270653,
    ];
    let expected_max_abs: f32 = 0.036473;

    // Manually compute what d and dmin SHOULD be
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
    println!("d bits: {:#06x} = {}", d_bits, d_bits);
    println!("dmin bits: {:#06x} = {}", dmin_bits, dmin_bits);
    // Python got d=0.0000609756, dmin=0.0005035400

    let mut output = vec![0.0f32; 256];
    dequantize_q4_k(&block, &mut output);

    println!("Output first 10: {:?}", &output[..10]);
    println!("Output max abs: {}", output.iter().map(|v| v.abs()).fold(0.0f32, f32::max));
    println!("Output mag: {}", output.iter().map(|v| v * v).sum::<f32>().sqrt());

    for i in 0..10 {
        assert!(
            (output[i] - expected_first_10[i]).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i, output[i], expected_first_10[i]
        );
    }

    let actual_max = output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        (actual_max - expected_max_abs).abs() < 1e-4,
        "Max abs mismatch: got {}, expected {}",
        actual_max, expected_max_abs
    );
}
