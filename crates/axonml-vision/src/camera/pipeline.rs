//! Inference Pipeline — Generic capture → preprocess → detect loop
//!
//! # File
//! `crates/axonml-vision/src/camera/pipeline.rs`
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

use super::preprocess::preprocess_frame;
use super::{CaptureBackend, CaptureConfig, CaptureError};

use std::time::Instant;

// =============================================================================
// Detection Model Trait
// =============================================================================

/// Trait for models that can run detection on preprocessed frames.
///
/// Generic over output type to support different detection architectures
/// (object detection, face detection, depth estimation, etc.).
pub trait DetectionModel {
    /// The output type produced by detection.
    type Output;

    /// Run detection on a preprocessed input tensor [1, C, H, W].
    fn detect(&mut self, input: &Variable) -> Self::Output;

    /// Get the expected input dimensions (width, height).
    fn input_size(&self) -> (u32, u32);
}

// =============================================================================
// Pipeline Statistics
// =============================================================================

/// Real-time performance statistics for the inference pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total frames processed.
    pub frames_processed: u64,
    /// Cumulative preprocessing time in seconds.
    pub total_preprocess_time: f64,
    /// Cumulative inference time in seconds.
    pub total_inference_time: f64,
    /// Timestamp of pipeline start.
    start_time: Instant,
}

impl PipelineStats {
    fn new() -> Self {
        Self {
            frames_processed: 0,
            total_preprocess_time: 0.0,
            total_inference_time: 0.0,
            start_time: Instant::now(),
        }
    }

    /// Frames per second (wall clock).
    pub fn fps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.frames_processed as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Average preprocessing time per frame in milliseconds.
    pub fn avg_preprocess_ms(&self) -> f64 {
        if self.frames_processed > 0 {
            (self.total_preprocess_time / self.frames_processed as f64) * 1000.0
        } else {
            0.0
        }
    }

    /// Average inference time per frame in milliseconds.
    pub fn avg_inference_ms(&self) -> f64 {
        if self.frames_processed > 0 {
            (self.total_inference_time / self.frames_processed as f64) * 1000.0
        } else {
            0.0
        }
    }

    /// Wall-clock elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

// =============================================================================
// Inference Pipeline
// =============================================================================

/// Generic inference pipeline that connects a camera backend to a detection model.
///
/// Runs: capture → preprocess → detect → callback for each frame.
///
/// # Type Parameters
/// - `M`: Detection model implementing `DetectionModel`
/// - `B`: Camera backend implementing `CaptureBackend`
pub struct InferencePipeline<M, B> {
    model: M,
    backend: B,
    stats: PipelineStats,
}

impl<M, B> InferencePipeline<M, B>
where
    M: DetectionModel,
    B: CaptureBackend,
{
    /// Create a new inference pipeline.
    pub fn new(model: M, backend: B) -> Self {
        Self {
            model,
            backend,
            stats: PipelineStats::new(),
        }
    }

    /// Open the camera and prepare for inference.
    pub fn start(&mut self, config: &CaptureConfig) -> Result<(), CaptureError> {
        self.backend.open(config)?;
        self.stats = PipelineStats::new();
        Ok(())
    }

    /// Process a single frame: capture → preprocess → detect.
    ///
    /// Returns the detection output from the model.
    pub fn step(&mut self) -> Result<M::Output, CaptureError> {
        let frame = self.backend.grab_frame()?;

        let (target_w, target_h) = self.model.input_size();

        let t0 = Instant::now();
        let input = preprocess_frame(&frame, target_w, target_h);
        let preprocess_time = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let output = self.model.detect(&input);
        let inference_time = t1.elapsed().as_secs_f64();

        self.stats.frames_processed += 1;
        self.stats.total_preprocess_time += preprocess_time;
        self.stats.total_inference_time += inference_time;

        Ok(output)
    }

    /// Run the pipeline for N frames, calling a callback for each result.
    pub fn run_n<F>(&mut self, n: usize, mut callback: F) -> Result<(), CaptureError>
    where
        F: FnMut(M::Output, &PipelineStats),
    {
        for _ in 0..n {
            let output = self.step()?;
            callback(output, &self.stats);
        }
        Ok(())
    }

    /// Stop the pipeline and close the camera.
    pub fn stop(&mut self) {
        self.backend.close();
    }

    /// Get current pipeline statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get a mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get a reference to the model.
    pub fn model(&self) -> &M {
        &self.model
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::file::FileBackend;

    /// Dummy detection model for testing.
    struct DummyDetector {
        call_count: usize,
    }

    impl DummyDetector {
        fn new() -> Self {
            Self { call_count: 0 }
        }
    }

    impl DetectionModel for DummyDetector {
        type Output = usize;

        fn detect(&mut self, input: &Variable) -> usize {
            assert_eq!(input.shape()[0], 1);
            assert_eq!(input.shape()[1], 3);
            self.call_count += 1;
            self.call_count
        }

        fn input_size(&self) -> (u32, u32) {
            (32, 32)
        }
    }

    #[test]
    fn test_pipeline_basic() {
        let backend = FileBackend::synthetic(64, 48, 128);
        let model = DummyDetector::new();
        let mut pipeline = InferencePipeline::new(model, backend);

        pipeline.start(&CaptureConfig::default()).unwrap();
        let result = pipeline.step().unwrap();
        assert_eq!(result, 1);

        assert_eq!(pipeline.stats().frames_processed, 1);
        assert!(pipeline.stats().avg_preprocess_ms() > 0.0);
        pipeline.stop();
    }

    #[test]
    fn test_pipeline_run_n() {
        let backend = FileBackend::synthetic_sequence(32, 32, 10);
        let model = DummyDetector::new();
        let mut pipeline = InferencePipeline::new(model, backend);

        pipeline.start(&CaptureConfig::default()).unwrap();

        let mut results = Vec::new();
        pipeline
            .run_n(5, |output, _stats| {
                results.push(output);
            })
            .unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results, vec![1, 2, 3, 4, 5]);
        assert_eq!(pipeline.stats().frames_processed, 5);
        pipeline.stop();
    }

    #[test]
    fn test_pipeline_fps_tracking() {
        let backend = FileBackend::synthetic_sequence(16, 16, 10);
        let model = DummyDetector::new();
        let mut pipeline = InferencePipeline::new(model, backend);

        pipeline.start(&CaptureConfig::default()).unwrap();

        for _ in 0..10 {
            pipeline.step().unwrap();
        }

        assert_eq!(pipeline.stats().frames_processed, 10);
        assert!(pipeline.stats().fps() > 0.0);
        assert!(pipeline.stats().elapsed_secs() > 0.0);
        pipeline.stop();
    }

    #[test]
    fn test_pipeline_stats_fresh() {
        let stats = PipelineStats::new();
        assert_eq!(stats.frames_processed, 0);
        assert_eq!(stats.fps(), 0.0);
        assert_eq!(stats.avg_preprocess_ms(), 0.0);
        assert_eq!(stats.avg_inference_ms(), 0.0);
    }
}
