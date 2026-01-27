//! High-Level Training Utilities
//!
//! Provides a `Trainer` class for simplified model training, similar to
//! PyTorch Lightning or Hugging Face Trainer.
//!
//! # Example
//! ```rust,ignore
//! use axonml::trainer::{Trainer, TrainingConfig};
//!
//! let trainer = Trainer::new(model, optimizer)
//!     .config(TrainingConfig::new().epochs(10).batch_size(32))
//!     .callbacks(vec![EarlyStopping::new(5)])
//!     .fit(&train_dataset, Some(&val_dataset));
//! ```
//!
//! @version 0.1.0

#[cfg(feature = "core")]
use axonml_tensor::Tensor;

#[cfg(feature = "nn")]
use axonml_nn::Parameter;

// =============================================================================
// Training Configuration
// =============================================================================

/// Configuration for training.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Gradient clipping max norm (None = no clipping)
    pub gradient_clip_norm: Option<f32>,
    /// Number of gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Logging frequency (steps)
    pub log_every: usize,
    /// Evaluation frequency (epochs)
    pub eval_every: usize,
    /// Save checkpoints
    pub save_checkpoints: bool,
    /// Checkpoint directory
    pub checkpoint_dir: String,
    /// Use mixed precision training
    pub mixed_precision: bool,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            learning_rate: 1e-3,
            gradient_clip_norm: None,
            gradient_accumulation_steps: 1,
            log_every: 100,
            eval_every: 1,
            save_checkpoints: false,
            checkpoint_dir: "checkpoints".to_string(),
            mixed_precision: false,
            seed: None,
        }
    }
}

impl TrainingConfig {
    /// Creates a new training configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set number of epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Builder: set batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Builder: set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder: set gradient clipping.
    pub fn gradient_clip_norm(mut self, max_norm: f32) -> Self {
        self.gradient_clip_norm = Some(max_norm);
        self
    }

    /// Builder: set gradient accumulation steps.
    pub fn gradient_accumulation_steps(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = steps.max(1);
        self
    }

    /// Builder: set logging frequency.
    pub fn log_every(mut self, steps: usize) -> Self {
        self.log_every = steps;
        self
    }

    /// Builder: enable mixed precision.
    pub fn mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Builder: set seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

// =============================================================================
// Training State
// =============================================================================

/// Current training state.
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch (0-indexed)
    pub epoch: usize,
    /// Global step count
    pub global_step: usize,
    /// Best validation metric
    pub best_metric: f32,
    /// Training loss history
    pub train_losses: Vec<f32>,
    /// Validation loss history
    pub val_losses: Vec<f32>,
    /// Learning rate history
    pub lr_history: Vec<f32>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            best_metric: f32::INFINITY,
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            lr_history: Vec::new(),
        }
    }
}

impl TrainingState {
    /// Creates a new training state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current epoch (1-indexed for display).
    pub fn current_epoch(&self) -> usize {
        self.epoch + 1
    }

    /// Returns average training loss for current epoch.
    pub fn avg_train_loss(&self) -> f32 {
        if self.train_losses.is_empty() {
            0.0
        } else {
            self.train_losses.iter().sum::<f32>() / self.train_losses.len() as f32
        }
    }

    /// Returns the last validation loss.
    pub fn last_val_loss(&self) -> Option<f32> {
        self.val_losses.last().copied()
    }
}

// =============================================================================
// Training Metrics
// =============================================================================

/// Metrics collected during training.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Loss value
    pub loss: f32,
    /// Accuracy (if applicable)
    pub accuracy: Option<f32>,
    /// Additional metrics
    pub extras: std::collections::HashMap<String, f32>,
}

impl TrainingMetrics {
    /// Creates metrics with just loss.
    pub fn new(loss: f32) -> Self {
        Self {
            loss,
            accuracy: None,
            extras: std::collections::HashMap::new(),
        }
    }

    /// Adds accuracy metric.
    pub fn with_accuracy(mut self, accuracy: f32) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    /// Adds a custom metric.
    pub fn with_metric(mut self, name: &str, value: f32) -> Self {
        self.extras.insert(name.to_string(), value);
        self
    }
}

// =============================================================================
// Callback Trait
// =============================================================================

/// Callback for training events.
pub trait Callback: Send {
    /// Called at the start of training.
    fn on_train_begin(&mut self, _state: &TrainingState) {}

    /// Called at the end of training.
    fn on_train_end(&mut self, _state: &TrainingState) {}

    /// Called at the start of an epoch.
    fn on_epoch_begin(&mut self, _epoch: usize, _state: &TrainingState) {}

    /// Called at the end of an epoch.
    fn on_epoch_end(&mut self, _epoch: usize, _state: &TrainingState) -> bool {
        true // Continue training
    }

    /// Called after each training step.
    fn on_step_end(&mut self, _step: usize, _metrics: &TrainingMetrics, _state: &TrainingState) {}

    /// Called after validation.
    fn on_validation_end(&mut self, _metrics: &TrainingMetrics, _state: &TrainingState) {}
}

// =============================================================================
// Early Stopping Callback
// =============================================================================

/// Early stopping callback.
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    counter: usize,
    best_loss: f32,
    mode: String,
}

impl EarlyStopping {
    /// Creates a new early stopping callback.
    pub fn new(patience: usize) -> Self {
        Self {
            patience,
            min_delta: 0.0,
            counter: 0,
            best_loss: f32::INFINITY,
            mode: "min".to_string(),
        }
    }

    /// Sets minimum delta for improvement.
    pub fn min_delta(mut self, delta: f32) -> Self {
        self.min_delta = delta;
        self
    }

    /// Sets mode ("min" or "max").
    pub fn mode(mut self, mode: &str) -> Self {
        self.mode = mode.to_string();
        self
    }
}

impl Callback for EarlyStopping {
    fn on_epoch_end(&mut self, _epoch: usize, state: &TrainingState) -> bool {
        let current = state.val_losses.last().copied().unwrap_or(f32::INFINITY);

        let improved = if self.mode == "min" {
            current < self.best_loss - self.min_delta
        } else {
            current > self.best_loss + self.min_delta
        };

        if improved {
            self.best_loss = current;
            self.counter = 0;
        } else {
            self.counter += 1;
        }

        self.counter < self.patience
    }
}

// =============================================================================
// Progress Logger Callback
// =============================================================================

/// Simple progress logging callback.
pub struct ProgressLogger {
    log_every: usize,
}

impl ProgressLogger {
    /// Creates a new progress logger.
    pub fn new(log_every: usize) -> Self {
        Self { log_every }
    }
}

impl Callback for ProgressLogger {
    fn on_epoch_begin(&mut self, epoch: usize, _state: &TrainingState) {
        println!("Epoch {}", epoch + 1);
    }

    fn on_step_end(&mut self, step: usize, metrics: &TrainingMetrics, _state: &TrainingState) {
        if step % self.log_every == 0 {
            print!("  Step {}: loss = {:.4}", step, metrics.loss);
            if let Some(acc) = metrics.accuracy {
                print!(", accuracy = {:.2}%", acc * 100.0);
            }
            println!();
        }
    }

    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> bool {
        println!(
            "Epoch {} complete: avg_loss = {:.4}",
            epoch + 1,
            state.avg_train_loss()
        );
        if let Some(val_loss) = state.last_val_loss() {
            println!("  Validation loss: {:.4}", val_loss);
        }
        true
    }
}

// =============================================================================
// Training History
// =============================================================================

/// Complete training history.
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Training losses per epoch
    pub train_loss: Vec<f32>,
    /// Validation losses per epoch
    pub val_loss: Vec<f32>,
    /// Learning rates per epoch
    pub learning_rates: Vec<f32>,
    /// Training duration in seconds
    pub duration_secs: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Whether training completed successfully
    pub completed: bool,
}

impl TrainingHistory {
    /// Creates an empty history.
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            learning_rates: Vec::new(),
            duration_secs: 0.0,
            epochs_completed: 0,
            completed: false,
        }
    }

    /// Returns the best training loss.
    pub fn best_train_loss(&self) -> Option<f32> {
        self.train_loss.iter().cloned().reduce(f32::min)
    }

    /// Returns the best validation loss.
    pub fn best_val_loss(&self) -> Option<f32> {
        self.val_loss.iter().cloned().reduce(f32::min)
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Clips gradients by global norm.
#[cfg(feature = "nn")]
pub fn clip_grad_norm(parameters: &[Parameter], max_norm: f32) -> f32 {
    let mut total_norm_sq = 0.0f32;

    for param in parameters {
        if let Some(grad) = param.grad() {
            let grad_vec = grad.to_vec();
            total_norm_sq += grad_vec.iter().map(|x| x * x).sum::<f32>();
        }
    }

    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for param in parameters {
            if let Some(grad) = param.grad() {
                let clipped: Vec<f32> = grad.to_vec().iter().map(|x| x * clip_coef).collect();
                #[cfg(feature = "core")]
                {
                    let clipped_tensor = Tensor::from_vec(clipped, grad.shape()).unwrap();
                    param.variable().set_grad(clipped_tensor);
                }
            }
        }
    }

    total_norm
}

/// Computes accuracy for classification.
#[cfg(feature = "core")]
pub fn compute_accuracy(predictions: &Tensor<f32>, targets: &Tensor<f32>) -> f32 {
    let pred_vec = predictions.to_vec();
    let target_vec = targets.to_vec();

    // Assume predictions are logits [batch, num_classes] and targets are indices
    let batch_size = predictions.shape()[0];
    let num_classes = if predictions.shape().len() > 1 {
        predictions.shape()[1]
    } else {
        1
    };

    let mut correct = 0;

    for b in 0..batch_size {
        // Find argmax of predictions
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let idx = b * num_classes + c;
            if pred_vec[idx] > max_val {
                max_val = pred_vec[idx];
                max_idx = c;
            }
        }

        // Compare with target
        let target = target_vec[b] as usize;
        if max_idx == target {
            correct += 1;
        }
    }

    correct as f32 / batch_size as f32
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::new()
            .epochs(20)
            .batch_size(64)
            .learning_rate(0.01)
            .gradient_clip_norm(1.0);

        assert_eq!(config.epochs, 20);
        assert_eq!(config.batch_size, 64);
        assert!((config.learning_rate - 0.01).abs() < 1e-6);
        assert_eq!(config.gradient_clip_norm, Some(1.0));
    }

    #[test]
    fn test_training_state() {
        let mut state = TrainingState::new();
        state.train_losses.push(0.5);
        state.train_losses.push(0.3);

        assert!((state.avg_train_loss() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_early_stopping() {
        let mut callback = EarlyStopping::new(3);
        let mut state = TrainingState::new();

        // Improving
        state.val_losses.push(1.0);
        assert!(callback.on_epoch_end(0, &state));

        state.val_losses.push(0.8);
        assert!(callback.on_epoch_end(1, &state));

        // Not improving
        state.val_losses.push(0.9);
        assert!(callback.on_epoch_end(2, &state)); // counter = 1

        state.val_losses.push(0.85);
        assert!(callback.on_epoch_end(3, &state)); // counter = 2

        state.val_losses.push(0.82);
        assert!(!callback.on_epoch_end(4, &state)); // counter = 3, stop
    }

    #[test]
    fn test_training_metrics() {
        let metrics = TrainingMetrics::new(0.5)
            .with_accuracy(0.9)
            .with_metric("f1", 0.85);

        assert!((metrics.loss - 0.5).abs() < 1e-6);
        assert_eq!(metrics.accuracy, Some(0.9));
        assert_eq!(metrics.extras.get("f1"), Some(&0.85));
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();
        history.train_loss = vec![0.5, 0.3, 0.2];
        history.val_loss = vec![0.6, 0.4, 0.35];

        assert_eq!(history.best_train_loss(), Some(0.2));
        assert_eq!(history.best_val_loss(), Some(0.35));
    }

    #[cfg(feature = "core")]
    #[test]
    fn test_compute_accuracy() {
        use axonml_tensor::Tensor;

        // 2 samples, 3 classes
        // Sample 0: [0.1, 0.8, 0.1] -> predicted class 1
        // Sample 1: [0.9, 0.05, 0.05] -> predicted class 0
        let predictions = Tensor::from_vec(vec![0.1, 0.8, 0.1, 0.9, 0.05, 0.05], &[2, 3]).unwrap();

        // Targets: [1, 0] (both correct)
        let targets = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();

        let accuracy = compute_accuracy(&predictions, &targets);
        assert!((accuracy - 1.0).abs() < 1e-6);
    }
}
