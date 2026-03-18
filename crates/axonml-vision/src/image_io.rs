//! Image I/O — Load JPEG/PNG images into Tensors
//!
//! # File
//! `crates/axonml-vision/src/image_io.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_tensor::Tensor;
use std::path::Path;

// =============================================================================
// Image Loading
// =============================================================================

/// Load an image file (JPEG, PNG, BMP, etc.) into a [C, H, W] f32 tensor.
///
/// RGB images produce [3, H, W], grayscale produce [1, H, W].
/// Values are in [0.0, 1.0].
pub fn load_image<P: AsRef<Path>>(path: P) -> Result<Tensor<f32>, String> {
    let img = image::open(path.as_ref()).map_err(|e| format!("Failed to load image: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);

    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            data[0 * h * w + y * w + x] = pixel[0] as f32 / 255.0;
            data[1 * h * w + y * w + x] = pixel[1] as f32 / 255.0;
            data[2 * h * w + y * w + x] = pixel[2] as f32 / 255.0;
        }
    }

    Tensor::from_vec(data, &[3, h, w]).map_err(|e| format!("Tensor creation failed: {e}"))
}

/// Load an image and resize it to the given (height, width).
///
/// Returns a [3, target_h, target_w] f32 tensor in [0, 1].
pub fn load_image_resized<P: AsRef<Path>>(
    path: P,
    target_h: usize,
    target_w: usize,
) -> Result<Tensor<f32>, String> {
    let img = image::open(path.as_ref()).map_err(|e| format!("Failed to load image: {e}"))?;
    let resized = img.resize_exact(
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();

    let mut data = vec![0.0f32; 3 * target_h * target_w];
    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            data[0 * target_h * target_w + y * target_w + x] = pixel[0] as f32 / 255.0;
            data[1 * target_h * target_w + y * target_w + x] = pixel[1] as f32 / 255.0;
            data[2 * target_h * target_w + y * target_w + x] = pixel[2] as f32 / 255.0;
        }
    }

    Tensor::from_vec(data, &[3, target_h, target_w])
        .map_err(|e| format!("Tensor creation failed: {e}"))
}

/// Load an image and return both the tensor and the original dimensions.
///
/// Returns `(tensor [3, H, W], (original_height, original_width))`.
pub fn load_image_with_info<P: AsRef<Path>>(
    path: P,
) -> Result<(Tensor<f32>, (usize, usize)), String> {
    let img = image::open(path.as_ref()).map_err(|e| format!("Failed to load image: {e}"))?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = img.to_rgb8();

    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            data[0 * h * w + y * w + x] = pixel[0] as f32 / 255.0;
            data[1 * h * w + y * w + x] = pixel[1] as f32 / 255.0;
            data[2 * h * w + y * w + x] = pixel[2] as f32 / 255.0;
        }
    }

    let tensor =
        Tensor::from_vec(data, &[3, h, w]).map_err(|e| format!("Tensor creation failed: {e}"))?;
    Ok((tensor, (h, w)))
}

/// Create a tensor from raw RGB bytes in HWC layout.
///
/// Input: `&[u8]` of length `3 * h * w` in row-major HWC order.
/// Output: `Tensor<f32>` of shape [3, h, w] in [0, 1].
pub fn rgb_bytes_to_tensor(data: &[u8], h: usize, w: usize) -> Result<Tensor<f32>, String> {
    if data.len() != 3 * h * w {
        return Err(format!("Expected {} bytes, got {}", 3 * h * w, data.len()));
    }

    let mut chw = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            chw[0 * h * w + y * w + x] = data[idx] as f32 / 255.0;
            chw[1 * h * w + y * w + x] = data[idx + 1] as f32 / 255.0;
            chw[2 * h * w + y * w + x] = data[idx + 2] as f32 / 255.0;
        }
    }

    Tensor::from_vec(chw, &[3, h, w]).map_err(|e| format!("Tensor creation failed: {e}"))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_bytes_to_tensor() {
        // 2x2 red image
        let data = vec![
            255, 0, 0, 255, 0, 0, // row 0
            255, 0, 0, 255, 0, 0, // row 1
        ];
        let tensor = rgb_bytes_to_tensor(&data, 2, 2).unwrap();
        assert_eq!(tensor.shape(), &[3, 2, 2]);

        let vals = tensor.to_vec();
        // Red channel should be ~1.0
        assert!((vals[0] - 1.0).abs() < 1e-5);
        // Green and blue channels should be 0.0
        assert!((vals[4] - 0.0).abs() < 1e-5);
        assert!((vals[8] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_bytes_wrong_size() {
        let data = vec![0u8; 10];
        assert!(rgb_bytes_to_tensor(&data, 2, 2).is_err());
    }

    #[test]
    fn test_load_nonexistent() {
        assert!(load_image("/nonexistent/path.jpg").is_err());
    }
}
