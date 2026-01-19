//! Config - Configuration File Handling
//!
//! Handles parsing and validation of Axonml configuration files.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{CliError, CliResult};

// =============================================================================
// Project Configuration
// =============================================================================

/// Project configuration (axonml.toml)
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ProjectConfig {
    /// Project metadata
    pub project: ProjectMeta,

    /// Training configuration
    #[serde(default)]
    pub training: TrainingConfig,

    /// Model configuration
    #[serde(default)]
    pub model: ModelConfig,

    /// Data configuration
    #[serde(default)]
    pub data: DataConfig,
}

/// Project metadata
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ProjectMeta {
    /// Project name
    pub name: String,

    /// Project version
    #[serde(default = "default_version")]
    pub version: String,

    /// Project description
    #[serde(default)]
    pub description: String,

    /// Authors
    #[serde(default)]
    pub authors: Vec<String>,
}

fn default_version() -> String {
    "0.1.0".to_string()
}

// =============================================================================
// Training Configuration
// =============================================================================

/// Training configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    /// Batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Learning rate
    #[serde(default = "default_lr")]
    pub learning_rate: f64,

    /// Optimizer configuration
    #[serde(default)]
    pub optimizer: OptimizerConfig,

    /// Learning rate scheduler
    #[serde(default)]
    pub scheduler: Option<SchedulerConfig>,

    /// Device to train on
    #[serde(default = "default_device")]
    pub device: String,

    /// Random seed
    #[serde(default)]
    pub seed: Option<u64>,

    /// Checkpoint save frequency (in epochs)
    #[serde(default = "default_checkpoint_freq")]
    pub checkpoint_frequency: usize,

    /// Output directory
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Number of data loading workers
    #[serde(default = "default_workers")]
    pub num_workers: usize,

    /// Enable gradient clipping
    #[serde(default)]
    pub gradient_clip: Option<f64>,

    /// Mixed precision training
    #[serde(default)]
    pub mixed_precision: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            learning_rate: default_lr(),
            optimizer: OptimizerConfig::default(),
            scheduler: None,
            device: default_device(),
            seed: None,
            checkpoint_frequency: default_checkpoint_freq(),
            output_dir: default_output_dir(),
            num_workers: default_workers(),
            gradient_clip: None,
            mixed_precision: false,
        }
    }
}

fn default_epochs() -> usize {
    10
}
fn default_batch_size() -> usize {
    32
}
fn default_lr() -> f64 {
    0.001
}
fn default_device() -> String {
    "cpu".to_string()
}
fn default_checkpoint_freq() -> usize {
    1
}
fn default_output_dir() -> String {
    "./output".to_string()
}
fn default_workers() -> usize {
    4
}

// =============================================================================
// Optimizer Configuration
// =============================================================================

/// Optimizer configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type (sgd, adam, adamw, rmsprop)
    #[serde(default = "default_optimizer")]
    pub name: String,

    /// Weight decay
    #[serde(default)]
    pub weight_decay: f64,

    /// Momentum (for SGD)
    #[serde(default)]
    pub momentum: f64,

    /// Beta1 (for Adam)
    #[serde(default = "default_beta1")]
    pub beta1: f64,

    /// Beta2 (for Adam)
    #[serde(default = "default_beta2")]
    pub beta2: f64,

    /// Epsilon (for Adam)
    #[serde(default = "default_eps")]
    pub eps: f64,

    /// Nesterov momentum (for SGD)
    #[serde(default)]
    pub nesterov: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            name: default_optimizer(),
            weight_decay: 0.0,
            momentum: 0.0,
            beta1: default_beta1(),
            beta2: default_beta2(),
            eps: default_eps(),
            nesterov: false,
        }
    }
}

fn default_optimizer() -> String {
    "adam".to_string()
}
fn default_beta1() -> f64 {
    0.9
}
fn default_beta2() -> f64 {
    0.999
}
fn default_eps() -> f64 {
    1e-8
}

// =============================================================================
// Scheduler Configuration
// =============================================================================

/// Learning rate scheduler configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler type (step, cosine, exponential, plateau)
    pub name: String,

    /// Step size (for `StepLR`)
    #[serde(default)]
    pub step_size: Option<usize>,

    /// Gamma (decay factor)
    #[serde(default = "default_gamma")]
    pub gamma: f64,

    /// Milestones (for `MultiStepLR`)
    #[serde(default)]
    pub milestones: Vec<usize>,

    /// `T_max` (for `CosineAnnealing`)
    #[serde(default)]
    pub t_max: Option<usize>,

    /// Minimum learning rate
    #[serde(default)]
    pub eta_min: f64,

    /// Warmup epochs
    #[serde(default)]
    pub warmup_epochs: usize,
}

fn default_gamma() -> f64 {
    0.1
}

// =============================================================================
// Model Configuration
// =============================================================================

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelConfig {
    /// Model architecture name
    #[serde(default)]
    pub architecture: String,

    /// Path to model definition file
    #[serde(default)]
    pub path: Option<String>,

    /// Number of input features/channels
    #[serde(default)]
    pub input_size: Option<usize>,

    /// Number of output classes
    #[serde(default)]
    pub num_classes: Option<usize>,

    /// Hidden layer sizes
    #[serde(default)]
    pub hidden_sizes: Vec<usize>,

    /// Dropout probability
    #[serde(default)]
    pub dropout: f64,

    /// Pretrained weights path
    #[serde(default)]
    pub pretrained: Option<String>,
}

// =============================================================================
// Data Configuration
// =============================================================================

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataConfig {
    /// Path to training data
    #[serde(default)]
    pub train_path: Option<String>,

    /// Path to validation data
    #[serde(default)]
    pub val_path: Option<String>,

    /// Path to test data
    #[serde(default)]
    pub test_path: Option<String>,

    /// Data format (csv, json, images, etc.)
    #[serde(default)]
    pub format: String,

    /// Train/validation split ratio
    #[serde(default = "default_val_split")]
    pub val_split: f64,

    /// Enable data augmentation
    #[serde(default)]
    pub augmentation: bool,

    /// Shuffle training data
    #[serde(default = "default_shuffle")]
    pub shuffle: bool,

    /// Normalize data
    #[serde(default)]
    pub normalize: bool,

    /// Normalization mean
    #[serde(default)]
    pub mean: Vec<f64>,

    /// Normalization std
    #[serde(default)]
    pub std: Vec<f64>,
}

fn default_val_split() -> f64 {
    0.1
}
fn default_shuffle() -> bool {
    true
}

// =============================================================================
// Configuration Loading
// =============================================================================

impl ProjectConfig {
    /// Load configuration from a TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> CliResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: ProjectConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> CliResult<()> {
        let content =
            toml::to_string_pretty(self).map_err(|e| CliError::Serialization(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create a default configuration with the given project name
    pub fn new(name: &str) -> Self {
        Self {
            project: ProjectMeta {
                name: name.to_string(),
                version: "0.1.0".to_string(),
                description: String::new(),
                authors: vec![],
            },
            training: TrainingConfig::default(),
            model: ModelConfig::default(),
            data: DataConfig::default(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProjectConfig::new("test-project");
        assert_eq!(config.project.name, "test-project");
        assert_eq!(config.training.epochs, 10);
        assert_eq!(config.training.batch_size, 32);
    }

    #[test]
    fn test_config_serialization() {
        let config = ProjectConfig::new("test-project");
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: ProjectConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.project.name, "test-project");
    }
}
