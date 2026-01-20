//! Training module for AxonML Server
//!
//! Provides training run tracking and real-time metrics streaming.

pub mod tracker;
pub mod websocket;

pub use tracker::TrainingTracker;
pub use websocket::MetricsStreamer;
