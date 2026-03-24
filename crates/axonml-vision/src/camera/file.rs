//! File Backend — Read frames from disk for testing and replay
//!
//! # File
//! `crates/axonml-vision/src/camera/file.rs`
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

use super::{CaptureBackend, CaptureConfig, CaptureError, FrameBuffer, PixelFormat};

// =============================================================================
// FileBackend
// =============================================================================

/// Camera backend that reads frames from disk.
///
/// Useful for deterministic testing, offline replay, and benchmarking.
/// Frames are stored as raw RGB bytes and looped when exhausted.
pub struct FileBackend {
    frames: Vec<FrameBuffer>,
    current: usize,
    is_open: bool,
    loop_frames: bool,
    width: u32,
    height: u32,
}

impl FileBackend {
    /// Create a file backend with pre-loaded frames.
    pub fn new(frames: Vec<FrameBuffer>) -> Self {
        let (w, h) = frames.first().map_or((0, 0), |f| (f.width, f.height));
        Self {
            frames,
            current: 0,
            is_open: false,
            loop_frames: true,
            width: w,
            height: h,
        }
    }

    /// Create a file backend with a single synthetic frame (for testing).
    pub fn synthetic(width: u32, height: u32, value: u8) -> Self {
        let data = vec![value; (width * height * 3) as usize];
        let frame = FrameBuffer::new(data, width, height, PixelFormat::Rgb);
        Self::new(vec![frame])
    }

    /// Create a file backend with N synthetic frames of incrementing values.
    pub fn synthetic_sequence(width: u32, height: u32, num_frames: usize) -> Self {
        let frames: Vec<_> = (0..num_frames)
            .map(|i| {
                let val = ((i * 20) % 256) as u8;
                let data = vec![val; (width * height * 3) as usize];
                let mut fb = FrameBuffer::new(data, width, height, PixelFormat::Rgb);
                fb.timestamp_us = i as u64 * 33333; // ~30fps timestamps
                fb
            })
            .collect();
        Self::new(frames)
    }

    /// Set whether frames should loop when exhausted. Default: true.
    pub fn set_loop(&mut self, loop_frames: bool) {
        self.loop_frames = loop_frames;
    }

    /// Get the number of loaded frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
}

impl CaptureBackend for FileBackend {
    fn open(&mut self, _config: &CaptureConfig) -> Result<(), CaptureError> {
        if self.frames.is_empty() {
            return Err(CaptureError::DeviceNotFound("No frames loaded".into()));
        }
        self.is_open = true;
        self.current = 0;
        Ok(())
    }

    fn grab_frame(&mut self) -> Result<FrameBuffer, CaptureError> {
        if !self.is_open {
            return Err(CaptureError::NotOpen);
        }

        if self.current >= self.frames.len() {
            if self.loop_frames {
                self.current = 0;
            } else {
                return Err(CaptureError::CaptureError("No more frames".into()));
            }
        }

        let frame = self.frames[self.current].clone();
        self.current += 1;
        Ok(frame)
    }

    fn is_open(&self) -> bool {
        self.is_open
    }

    fn close(&mut self) {
        self.is_open = false;
        self.current = 0;
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_backend_synthetic() {
        let mut backend = FileBackend::synthetic(32, 32, 128);
        assert_eq!(backend.num_frames(), 1);
        assert!(!backend.is_open());

        backend.open(&CaptureConfig::default()).unwrap();
        assert!(backend.is_open());

        let frame = backend.grab_frame().unwrap();
        assert_eq!(frame.width, 32);
        assert_eq!(frame.height, 32);
        assert_eq!(frame.format, PixelFormat::Rgb);
        assert_eq!(frame.data[0], 128);
    }

    #[test]
    fn test_file_backend_loops() {
        let mut backend = FileBackend::synthetic_sequence(8, 8, 3);
        backend.open(&CaptureConfig::default()).unwrap();

        // Read all 3 frames
        let f1 = backend.grab_frame().unwrap();
        let f2 = backend.grab_frame().unwrap();
        let f3 = backend.grab_frame().unwrap();

        assert_eq!(f1.data[0], 0);
        assert_eq!(f2.data[0], 20);
        assert_eq!(f3.data[0], 40);

        // Should loop back
        let f4 = backend.grab_frame().unwrap();
        assert_eq!(f4.data[0], 0);
    }

    #[test]
    fn test_file_backend_no_loop() {
        let mut backend = FileBackend::synthetic_sequence(8, 8, 2);
        backend.set_loop(false);
        backend.open(&CaptureConfig::default()).unwrap();

        backend.grab_frame().unwrap();
        backend.grab_frame().unwrap();

        assert!(backend.grab_frame().is_err());
    }

    #[test]
    fn test_file_backend_not_open() {
        let mut backend = FileBackend::synthetic(8, 8, 0);
        assert!(backend.grab_frame().is_err());
    }

    #[test]
    fn test_file_backend_close_resets() {
        let mut backend = FileBackend::synthetic_sequence(8, 8, 3);
        backend.open(&CaptureConfig::default()).unwrap();
        backend.grab_frame().unwrap();
        backend.grab_frame().unwrap();

        backend.close();
        assert!(!backend.is_open());

        backend.open(&CaptureConfig::default()).unwrap();
        let f = backend.grab_frame().unwrap();
        assert_eq!(f.data[0], 0); // Reset to first frame
    }

    #[test]
    fn test_file_backend_resolution() {
        let backend = FileBackend::synthetic(320, 240, 0);
        assert_eq!(backend.resolution(), (320, 240));
    }

    #[test]
    fn test_file_backend_timestamps() {
        let mut backend = FileBackend::synthetic_sequence(4, 4, 5);
        backend.open(&CaptureConfig::default()).unwrap();

        let f0 = backend.grab_frame().unwrap();
        let f1 = backend.grab_frame().unwrap();
        assert_eq!(f0.timestamp_us, 0);
        assert_eq!(f1.timestamp_us, 33333);
    }
}
