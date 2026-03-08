//! Camera Capture and Inference Pipeline
//!
//! # File
//! `crates/axonml-vision/src/camera/mod.rs`
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

pub mod file;
pub mod pipeline;
pub mod preprocess;

#[cfg(target_os = "linux")]
pub mod v4l2;

// =============================================================================
// Core Types
// =============================================================================

/// Pixel format of captured frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// RGB 8-bit per channel, packed [R,G,B,R,G,B,...].
    Rgb,
    /// YUYV (YUV 4:2:2) packed format — 2 pixels per 4 bytes.
    Yuyv,
    /// MJPEG compressed frame.
    Mjpeg,
    /// Grayscale 8-bit.
    Gray,
}

/// Configuration for camera capture.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Capture width in pixels.
    pub width: u32,
    /// Capture height in pixels.
    pub height: u32,
    /// Preferred pixel format.
    pub format: PixelFormat,
    /// Target frames per second (hint to driver).
    pub fps: u32,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            format: PixelFormat::Yuyv,
            fps: 30,
        }
    }
}

/// A raw captured frame from a camera backend.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    /// Raw pixel data.
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format of the data.
    pub format: PixelFormat,
    /// Monotonic timestamp in microseconds (if available).
    pub timestamp_us: u64,
}

impl FrameBuffer {
    /// Create a new frame buffer.
    pub fn new(data: Vec<u8>, width: u32, height: u32, format: PixelFormat) -> Self {
        Self {
            data,
            width,
            height,
            format,
            timestamp_us: 0,
        }
    }

    /// Number of pixels in the frame.
    pub fn num_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Errors from camera capture.
#[derive(Debug)]
pub enum CaptureError {
    /// Device could not be opened.
    DeviceNotFound(String),
    /// Format negotiation failed.
    FormatError(String),
    /// Frame grab failed.
    CaptureError(String),
    /// Device is not open.
    NotOpen,
    /// I/O error.
    IoError(std::io::Error),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceNotFound(s) => write!(f, "Device not found: {s}"),
            Self::FormatError(s) => write!(f, "Format error: {s}"),
            Self::CaptureError(s) => write!(f, "Capture error: {s}"),
            Self::NotOpen => write!(f, "Device not open"),
            Self::IoError(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for CaptureError {}

impl From<std::io::Error> for CaptureError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Trait for camera capture backends.
///
/// Implementations provide hardware-specific frame acquisition from
/// V4L2 devices, files, or other sources.
pub trait CaptureBackend {
    /// Open the capture device with the given configuration.
    fn open(&mut self, config: &CaptureConfig) -> Result<(), CaptureError>;

    /// Grab a single frame. Returns `None` if no frame is available.
    fn grab_frame(&mut self) -> Result<FrameBuffer, CaptureError>;

    /// Check if the device is currently open.
    fn is_open(&self) -> bool;

    /// Close the capture device and release resources.
    fn close(&mut self);

    /// Get the actual capture dimensions (may differ from requested).
    fn resolution(&self) -> (u32, u32);
}

// =============================================================================
// Re-exports
// =============================================================================

pub use file::FileBackend;
pub use pipeline::{DetectionModel, InferencePipeline, PipelineStats};
pub use preprocess::{normalize_imagenet, resize_bilinear, yuyv_to_rgb};

#[cfg(target_os = "linux")]
pub use v4l2::V4L2Backend;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_config_default() {
        let cfg = CaptureConfig::default();
        assert_eq!(cfg.width, 640);
        assert_eq!(cfg.height, 480);
        assert_eq!(cfg.format, PixelFormat::Yuyv);
        assert_eq!(cfg.fps, 30);
    }

    #[test]
    fn test_frame_buffer_creation() {
        let fb = FrameBuffer::new(vec![0u8; 640 * 480 * 3], 640, 480, PixelFormat::Rgb);
        assert_eq!(fb.num_pixels(), 640 * 480);
        assert_eq!(fb.data.len(), 640 * 480 * 3);
    }

    #[test]
    fn test_pixel_format_equality() {
        assert_eq!(PixelFormat::Rgb, PixelFormat::Rgb);
        assert_ne!(PixelFormat::Rgb, PixelFormat::Yuyv);
    }
}
