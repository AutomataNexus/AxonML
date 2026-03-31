//! Checkpoint - Training State Persistence
//!
//! # File
//! `crates/axonml-serialize/src/checkpoint.rs`
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

use crate::StateDict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// TrainingState
// =============================================================================

/// Training state for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingState {
    /// Current epoch.
    pub epoch: usize,
    /// Current step within epoch.
    pub step: usize,
    /// Global step count.
    pub global_step: usize,
    /// Best metric value seen so far.
    pub best_metric: Option<f32>,
    /// Name of the best metric.
    pub best_metric_name: Option<String>,
    /// Training loss history (last N values).
    pub loss_history: Vec<f32>,
    /// Validation loss history.
    pub val_loss_history: Vec<f32>,
    /// Learning rate history.
    pub lr_history: Vec<f32>,
    /// Custom metrics.
    pub custom_metrics: HashMap<String, Vec<f32>>,
}

impl TrainingState {
    /// Create a new training state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a training loss value (capped at last 1000).
    pub fn record_loss(&mut self, loss: f32) {
        self.loss_history.push(loss);
        if self.loss_history.len() > 1000 {
            self.loss_history.drain(..self.loss_history.len() - 1000);
        }
    }

    /// Record a validation loss value (capped at last 1000).
    pub fn record_val_loss(&mut self, loss: f32) {
        self.val_loss_history.push(loss);
        if self.val_loss_history.len() > 1000 {
            self.val_loss_history.drain(..self.val_loss_history.len() - 1000);
        }
    }

    /// Record learning rate (capped at last 1000).
    pub fn record_lr(&mut self, lr: f32) {
        self.lr_history.push(lr);
        if self.lr_history.len() > 1000 {
            self.lr_history.drain(..self.lr_history.len() - 1000);
        }
    }

    /// Record a custom metric (capped at last 1000 per metric).
    pub fn record_metric(&mut self, name: &str, value: f32) {
        let history = self.custom_metrics.entry(name.to_string()).or_default();
        history.push(value);
        if history.len() > 1000 {
            history.drain(..history.len() - 1000);
        }
    }

    /// Update best metric if improved.
    pub fn update_best(&mut self, name: &str, value: f32, higher_is_better: bool) -> bool {
        let improved = match self.best_metric {
            None => true,
            Some(best) => {
                if higher_is_better {
                    value > best
                } else {
                    value < best
                }
            }
        };

        if improved {
            self.best_metric = Some(value);
            self.best_metric_name = Some(name.to_string());
        }

        improved
    }

    /// Get the average loss over recent values.
    #[must_use]
    pub fn avg_loss(&self, n: usize) -> Option<f32> {
        if self.loss_history.is_empty() {
            return None;
        }
        let start = self.loss_history.len().saturating_sub(n);
        let slice = &self.loss_history[start..];
        Some(slice.iter().sum::<f32>() / slice.len() as f32)
    }

    /// Increment epoch.
    pub fn next_epoch(&mut self) {
        self.epoch += 1;
        self.step = 0;
    }

    /// Increment step.
    pub fn next_step(&mut self) {
        self.step += 1;
        self.global_step += 1;
    }
}

// =============================================================================
// Checkpoint
// =============================================================================

/// A complete training checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Model state dictionary.
    pub model_state: StateDict,
    /// Optimizer state dictionary.
    pub optimizer_state: StateDict,
    /// Training state.
    pub training_state: TrainingState,
    /// Random number generator state (for reproducibility).
    pub rng_state: Option<Vec<u8>>,
    /// Configuration used for training.
    pub config: HashMap<String, String>,
    /// Axonml version.
    pub axonml_version: String,
    /// Timestamp when checkpoint was created.
    pub timestamp: String,
}

impl Checkpoint {
    /// Create a new checkpoint builder.
    #[must_use]
    pub fn builder() -> CheckpointBuilder {
        CheckpointBuilder::new()
    }

    /// Get the epoch from this checkpoint.
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.training_state.epoch
    }

    /// Get the global step from this checkpoint.
    #[must_use]
    pub fn global_step(&self) -> usize {
        self.training_state.global_step
    }

    /// Get the best metric value.
    #[must_use]
    pub fn best_metric(&self) -> Option<f32> {
        self.training_state.best_metric
    }
}

// =============================================================================
// CheckpointBuilder
// =============================================================================

/// Builder for creating checkpoints.
pub struct CheckpointBuilder {
    model_state: Option<StateDict>,
    optimizer_state: Option<StateDict>,
    training_state: TrainingState,
    rng_state: Option<Vec<u8>>,
    config: HashMap<String, String>,
}

impl CheckpointBuilder {
    /// Create a new checkpoint builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_state: None,
            optimizer_state: None,
            training_state: TrainingState::new(),
            rng_state: None,
            config: HashMap::new(),
        }
    }

    /// Set the model state.
    #[must_use]
    pub fn model_state(mut self, state: StateDict) -> Self {
        self.model_state = Some(state);
        self
    }

    /// Set the optimizer state.
    #[must_use]
    pub fn optimizer_state(mut self, state: StateDict) -> Self {
        self.optimizer_state = Some(state);
        self
    }

    /// Set the training state.
    #[must_use]
    pub fn training_state(mut self, state: TrainingState) -> Self {
        self.training_state = state;
        self
    }

    /// Set the RNG state.
    #[must_use]
    pub fn rng_state(mut self, state: Vec<u8>) -> Self {
        self.rng_state = Some(state);
        self
    }

    /// Add a configuration value.
    #[must_use]
    pub fn config(mut self, key: &str, value: &str) -> Self {
        self.config.insert(key.to_string(), value.to_string());
        self
    }

    /// Set the epoch.
    #[must_use]
    pub fn epoch(mut self, epoch: usize) -> Self {
        self.training_state.epoch = epoch;
        self
    }

    /// Set the global step.
    #[must_use]
    pub fn global_step(mut self, step: usize) -> Self {
        self.training_state.global_step = step;
        self
    }

    /// Build the checkpoint.
    #[must_use]
    pub fn build(self) -> Checkpoint {
        Checkpoint {
            model_state: self.model_state.unwrap_or_default(),
            optimizer_state: self.optimizer_state.unwrap_or_default(),
            training_state: self.training_state,
            rng_state: self.rng_state,
            config: self.config,
            axonml_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono_timestamp(),
        }
    }
}

impl Default for CheckpointBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Utilities
// =============================================================================

fn chrono_timestamp() -> String {
    // ISO 8601-ish timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Convert unix timestamp to UTC date-time string
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Days since 1970-01-01
    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }

    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining_days < md as i64 {
            m = i + 1;
            break;
        }
        remaining_days -= md as i64;
    }
    let d = remaining_days + 1;

    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorData;

    #[test]
    fn test_training_state_basic() {
        let mut state = TrainingState::new();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);

        state.next_step();
        assert_eq!(state.step, 1);
        assert_eq!(state.global_step, 1);

        state.next_epoch();
        assert_eq!(state.epoch, 1);
        assert_eq!(state.step, 0);
    }

    #[test]
    fn test_training_state_loss_recording() {
        let mut state = TrainingState::new();

        state.record_loss(1.0);
        state.record_loss(0.8);
        state.record_loss(0.6);

        assert_eq!(state.loss_history.len(), 3);
        let avg = state.avg_loss(2).unwrap();
        assert!((avg - 0.7).abs() < 1e-5, "Expected ~0.7, got {avg}");
    }

    #[test]
    fn test_training_state_best_metric() {
        let mut state = TrainingState::new();

        // Lower is better (like loss)
        assert!(state.update_best("loss", 1.0, false));
        assert!(!state.update_best("loss", 1.5, false));
        assert!(state.update_best("loss", 0.5, false));
        assert_eq!(state.best_metric, Some(0.5));

        // Higher is better (like accuracy)
        let mut state2 = TrainingState::new();
        assert!(state2.update_best("accuracy", 0.8, true));
        assert!(!state2.update_best("accuracy", 0.7, true));
        assert!(state2.update_best("accuracy", 0.9, true));
        assert_eq!(state2.best_metric, Some(0.9));
    }

    #[test]
    fn test_checkpoint_builder() {
        let mut model_state = StateDict::new();
        model_state.insert(
            "weight".to_string(),
            TensorData {
                shape: vec![10, 5],
                values: vec![0.0; 50],
            },
        );

        let checkpoint = Checkpoint::builder()
            .model_state(model_state)
            .epoch(5)
            .global_step(1000)
            .config("learning_rate", "0.001")
            .build();

        assert_eq!(checkpoint.epoch(), 5);
        assert_eq!(checkpoint.global_step(), 1000);
        assert!(checkpoint.config.contains_key("learning_rate"));
    }

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = Checkpoint::builder().epoch(10).global_step(5000).build();

        // Serialize
        let bytes = bincode::serialize(&checkpoint).unwrap();
        assert!(!bytes.is_empty());

        // Deserialize
        let restored: Checkpoint = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.epoch(), 10);
        assert_eq!(restored.global_step(), 5000);
    }
}
