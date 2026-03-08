//! Frame Preprocessing — Color Conversion, Resize, Normalize
//!
//! # File
//! `crates/axonml-vision/src/camera/preprocess.rs`
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

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use super::{FrameBuffer, PixelFormat};

// =============================================================================
// Color Conversion
// =============================================================================

/// Convert YUYV (YUV 4:2:2) frame data to RGB.
///
/// YUYV packs two pixels per 4 bytes: [Y0, U, Y1, V].
/// Each pair of pixels shares U and V chroma values.
///
/// # Arguments
/// - `yuyv`: Raw YUYV bytes (length = width * height * 2)
/// - `width`: Frame width
/// - `height`: Frame height
///
/// # Returns
/// RGB bytes (length = width * height * 3)
pub fn yuyv_to_rgb(yuyv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let num_pixels = (width * height) as usize;
    let mut rgb = vec![0u8; num_pixels * 3];

    // Process 2 pixels at a time (4 bytes YUYV → 6 bytes RGB)
    let num_macropixels = num_pixels / 2;
    for i in 0..num_macropixels {
        let base_yuyv = i * 4;
        let base_rgb = i * 6;

        if base_yuyv + 3 >= yuyv.len() {
            break;
        }

        let y0 = yuyv[base_yuyv] as f32;
        let u = yuyv[base_yuyv + 1] as f32 - 128.0;
        let y1 = yuyv[base_yuyv + 2] as f32;
        let v = yuyv[base_yuyv + 3] as f32 - 128.0;

        // BT.601 conversion
        rgb[base_rgb] = (y0 + 1.402 * v).clamp(0.0, 255.0) as u8;
        rgb[base_rgb + 1] = (y0 - 0.344136 * u - 0.714136 * v).clamp(0.0, 255.0) as u8;
        rgb[base_rgb + 2] = (y0 + 1.772 * u).clamp(0.0, 255.0) as u8;

        rgb[base_rgb + 3] = (y1 + 1.402 * v).clamp(0.0, 255.0) as u8;
        rgb[base_rgb + 4] = (y1 - 0.344136 * u - 0.714136 * v).clamp(0.0, 255.0) as u8;
        rgb[base_rgb + 5] = (y1 + 1.772 * u).clamp(0.0, 255.0) as u8;
    }

    rgb
}

/// Convert a `FrameBuffer` to RGB bytes, handling format conversion.
pub fn frame_to_rgb(frame: &FrameBuffer) -> Vec<u8> {
    match frame.format {
        PixelFormat::Rgb => frame.data.clone(),
        PixelFormat::Yuyv => yuyv_to_rgb(&frame.data, frame.width, frame.height),
        PixelFormat::Gray => {
            // Replicate grayscale to all 3 channels
            let mut rgb = vec![0u8; frame.data.len() * 3];
            for (i, &g) in frame.data.iter().enumerate() {
                rgb[i * 3] = g;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = g;
            }
            rgb
        }
        PixelFormat::Mjpeg => {
            // MJPEG decoding not implemented — return black frame
            vec![0u8; frame.num_pixels() * 3]
        }
    }
}

// =============================================================================
// Bilinear Resize
// =============================================================================

/// Bilinear resize of RGB image data.
///
/// # Arguments
/// - `rgb`: Input RGB bytes in HWC layout (length = src_h * src_w * 3)
/// - `src_w`, `src_h`: Source dimensions
/// - `dst_w`, `dst_h`: Target dimensions
///
/// # Returns
/// Resized RGB bytes (length = dst_h * dst_w * 3)
pub fn resize_bilinear(
    rgb: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    let mut out = vec![0u8; dw * dh * 3];

    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    for dy in 0..dh {
        for dx in 0..dw {
            let src_x = (dx as f32 + 0.5) * scale_x - 0.5;
            let src_y = (dy as f32 + 0.5) * scale_y - 0.5;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let sample = |iy: i32, ix: i32, c: usize| -> f32 {
                let iy = iy.clamp(0, sh as i32 - 1) as usize;
                let ix = ix.clamp(0, sw as i32 - 1) as usize;
                rgb[(iy * sw + ix) * 3 + c] as f32
            };

            let dst_idx = (dy * dw + dx) * 3;
            for c in 0..3 {
                let v = sample(y0, x0, c) * (1.0 - fx) * (1.0 - fy)
                    + sample(y0, x1, c) * fx * (1.0 - fy)
                    + sample(y1, x0, c) * (1.0 - fx) * fy
                    + sample(y1, x1, c) * fx * fy;
                out[dst_idx + c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

// =============================================================================
// Normalize
// =============================================================================

/// Normalize RGB bytes to ImageNet-standard float tensor [1, 3, H, W].
///
/// Converts HWC u8 → NCHW f32, applies per-channel normalization:
///   pixel = (pixel / 255.0 - mean) / std
///
/// # Arguments
/// - `rgb`: RGB bytes in HWC layout (length = H * W * 3)
/// - `width`, `height`: Image dimensions
/// - `mean`: Per-channel mean [R, G, B]
/// - `std`: Per-channel std [R, G, B]
pub fn normalize_imagenet(
    rgb: &[u8],
    width: u32,
    height: u32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Variable {
    let w = width as usize;
    let h = height as usize;
    let mut nchw = vec![0.0f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            for c in 0..3 {
                let val = rgb[src + c] as f32 / 255.0;
                nchw[c * h * w + y * w + x] = (val - mean[c]) / std[c];
            }
        }
    }

    let tensor = Tensor::from_vec(nchw, &[1, 3, h, w]).unwrap();
    Variable::new(tensor, false)
}

/// Full preprocessing: FrameBuffer → normalized [1, 3, target_h, target_w] Variable.
///
/// Steps: format convert → resize → normalize (ImageNet defaults).
pub fn preprocess_frame(
    frame: &FrameBuffer,
    target_w: u32,
    target_h: u32,
) -> Variable {
    let rgb = frame_to_rgb(frame);
    let resized = resize_bilinear(&rgb, frame.width, frame.height, target_w, target_h);
    normalize_imagenet(
        &resized,
        target_w,
        target_h,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuyv_to_rgb_basic() {
        // 2x1 YUYV frame: Y0=128, U=128, Y1=128, V=128 → neutral gray
        let yuyv = vec![128u8, 128, 128, 128];
        let rgb = yuyv_to_rgb(&yuyv, 2, 1);
        assert_eq!(rgb.len(), 6);
        // With U=0, V=0 (after -128), should be ~128 for all channels
        assert!((rgb[0] as i32 - 128).unsigned_abs() < 3);
        assert!((rgb[1] as i32 - 128).unsigned_abs() < 3);
        assert!((rgb[2] as i32 - 128).unsigned_abs() < 3);
    }

    #[test]
    fn test_yuyv_to_rgb_size() {
        let w = 16u32;
        let h = 8u32;
        let yuyv = vec![128u8; (w * h * 2) as usize];
        let rgb = yuyv_to_rgb(&yuyv, w, h);
        assert_eq!(rgb.len(), (w * h * 3) as usize);
    }

    #[test]
    fn test_resize_bilinear_identity() {
        let rgb = vec![100u8; 4 * 4 * 3]; // 4x4 uniform
        let resized = resize_bilinear(&rgb, 4, 4, 4, 4);
        assert_eq!(resized.len(), 4 * 4 * 3);
        for &v in &resized {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_resize_bilinear_downsample() {
        let rgb = vec![200u8; 8 * 8 * 3];
        let resized = resize_bilinear(&rgb, 8, 8, 4, 4);
        assert_eq!(resized.len(), 4 * 4 * 3);
        for &v in &resized {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_resize_bilinear_upsample() {
        let rgb = vec![150u8; 2 * 2 * 3];
        let resized = resize_bilinear(&rgb, 2, 2, 8, 8);
        assert_eq!(resized.len(), 8 * 8 * 3);
        for &v in &resized {
            assert_eq!(v, 150);
        }
    }

    #[test]
    fn test_normalize_imagenet_shape() {
        let rgb = vec![128u8; 32 * 32 * 3];
        let var = normalize_imagenet(&rgb, 32, 32, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        assert_eq!(var.shape(), vec![1, 3, 32, 32]);
    }

    #[test]
    fn test_normalize_imagenet_values() {
        // All zeros → (0/255 - mean) / std
        let rgb = vec![0u8; 4 * 4 * 3];
        let var = normalize_imagenet(&rgb, 4, 4, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        let data = var.data().to_vec();
        for &v in &data {
            assert!((v - (-1.0)).abs() < 1e-5, "Expected -1.0, got {v}");
        }
    }

    #[test]
    fn test_preprocess_frame() {
        let frame = FrameBuffer::new(vec![128u8; 64 * 48 * 3], 64, 48, PixelFormat::Rgb);
        let var = preprocess_frame(&frame, 32, 32);
        assert_eq!(var.shape(), vec![1, 3, 32, 32]);
    }

    #[test]
    fn test_frame_to_rgb_gray() {
        let frame = FrameBuffer::new(vec![200u8; 4 * 4], 4, 4, PixelFormat::Gray);
        let rgb = frame_to_rgb(&frame);
        assert_eq!(rgb.len(), 4 * 4 * 3);
        for chunk in rgb.chunks(3) {
            assert_eq!(chunk, &[200, 200, 200]);
        }
    }
}
