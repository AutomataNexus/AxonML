//! HVAC Multi-Horizon Predictor - Native AxonML Implementation
//!
//! This example demonstrates building and training an HVAC failure prediction
//! model entirely within AxonML, without external dependencies.
//!
//! Model Architecture:
//! - Input projection: Linear(28 -> 128) + LayerNorm + ReLU
//! - Temporal encoder: 2-layer GRU with hidden_size=128
//! - Mean pooling for sequence aggregation
//! - 3 prediction heads (5min, 15min, 30min horizons)
//! - Each head: Linear(128->128) -> ReLU -> Linear(128->64) -> ReLU -> Linear(64->20)
//!
//! Usage: cargo run --example hvac_model

use axonml::autograd::Variable;
use axonml::nn::{Dropout, LayerNorm, Linear, Module, Parameter, ReLU, Softmax, GRU};
use axonml::tensor::Tensor;

// =============================================================================
// Model Configuration
// =============================================================================

/// HVAC Model Configuration
#[derive(Debug, Clone)]
pub struct HvacConfig {
    /// Number of input features (sensor readings)
    pub num_features: usize,
    /// Sequence length (time steps)
    pub seq_len: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of GRU layers
    pub num_layers: usize,
    /// Number of failure classes
    pub num_classes: usize,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for HvacConfig {
    fn default() -> Self {
        Self {
            num_features: 28,
            seq_len: 120,
            hidden_size: 128,
            num_layers: 2,
            num_classes: 20,
            dropout: 0.1,
        }
    }
}

// =============================================================================
// Prediction Head
// =============================================================================

/// Classification head for a single prediction horizon
pub struct PredictionHead {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    relu: ReLU,
    dropout: Dropout,
}

impl PredictionHead {
    pub fn new(hidden_size: usize, num_classes: usize, dropout: f32) -> Self {
        Self {
            fc1: Linear::new(hidden_size, hidden_size),
            fc2: Linear::new(hidden_size, 64),
            fc3: Linear::new(64, num_classes),
            relu: ReLU,
            dropout: Dropout::new(dropout),
        }
    }
}

impl Module for PredictionHead {
    fn forward(&self, x: &Variable) -> Variable {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(&x);
        let x = self.dropout.forward(&x);
        let x = self.fc2.forward(&x);
        let x = self.relu.forward(&x);
        let x = self.dropout.forward(&x);
        self.fc3.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

// =============================================================================
// HVAC Multi-Horizon Predictor
// =============================================================================

/// HVAC Multi-Horizon Failure Predictor
///
/// Predicts potential HVAC system failures at 3 time horizons:
/// - Imminent: 5 minutes
/// - Warning: 15 minutes
/// - Early: 30 minutes
pub struct HvacPredictor {
    config: HvacConfig,

    // Input projection
    input_proj: Linear,
    input_norm: LayerNorm,
    input_relu: ReLU,

    // Temporal encoder
    gru: GRU,

    // Prediction heads
    head_imminent: PredictionHead, // 5 min
    head_warning: PredictionHead,  // 15 min
    head_early: PredictionHead,    // 30 min

    // Output
    softmax: Softmax,
}

/// Output from the HVAC predictor
#[derive(Debug)]
pub struct HvacOutput {
    /// Imminent (5 min) class predictions
    pub imminent_logits: Variable,
    /// Warning (15 min) class predictions
    pub warning_logits: Variable,
    /// Early (30 min) class predictions
    pub early_logits: Variable,
}

impl HvacPredictor {
    /// Creates a new HVAC predictor with the given configuration
    pub fn new(config: HvacConfig) -> Self {
        Self {
            input_proj: Linear::new(config.num_features, config.hidden_size),
            input_norm: LayerNorm::new(vec![config.hidden_size]),
            input_relu: ReLU,
            gru: GRU::new(config.hidden_size, config.hidden_size, config.num_layers),
            head_imminent: PredictionHead::new(
                config.hidden_size,
                config.num_classes,
                config.dropout,
            ),
            head_warning: PredictionHead::new(
                config.hidden_size,
                config.num_classes,
                config.dropout,
            ),
            head_early: PredictionHead::new(config.hidden_size, config.num_classes, config.dropout),
            softmax: Softmax::new(-1),
            config,
        }
    }

    /// Mean pooling over sequence dimension
    fn mean_pool(&self, x: &Variable) -> Variable {
        let data = x.data();
        let shape = data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden = shape[2];

        // Reshape to [batch * seq, hidden] then back
        let values = data.to_vec();

        // Calculate mean over sequence dimension
        let mut pooled = vec![0.0f32; batch_size * hidden];
        for b in 0..batch_size {
            for h in 0..hidden {
                let mut sum = 0.0;
                for s in 0..seq_len {
                    let idx = b * seq_len * hidden + s * hidden + h;
                    sum += values[idx];
                }
                pooled[b * hidden + h] = sum / seq_len as f32;
            }
        }

        let pooled_tensor = Tensor::from_vec(pooled, &[batch_size, hidden])
            .expect("Failed to create pooled tensor");
        Variable::new(pooled_tensor, x.requires_grad())
    }

    /// Forward pass returning logits for all 3 horizons
    pub fn forward_multi(&self, x: &Variable) -> HvacOutput {
        let x_data = x.data();
        let shape = x_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        drop(x_data); // Release borrow

        // Input projection: [batch, seq, features] -> [batch, seq, hidden]
        // Reshape for linear: [batch * seq, features]
        let x_flat = x.reshape(&[batch_size * seq_len, self.config.num_features]);
        let proj = self.input_proj.forward(&x_flat);
        let proj = self.input_norm.forward(&proj);
        let proj = self.input_relu.forward(&proj);
        let proj = proj.reshape(&[batch_size, seq_len, self.config.hidden_size]);

        // GRU encoding: [batch, seq, hidden] -> [batch, seq, hidden]
        let encoded = self.gru.forward(&proj);

        // Mean pooling: [batch, seq, hidden] -> [batch, hidden]
        let pooled = self.mean_pool(&encoded);

        // Prediction heads
        let imminent_logits = self.head_imminent.forward(&pooled);
        let warning_logits = self.head_warning.forward(&pooled);
        let early_logits = self.head_early.forward(&pooled);

        HvacOutput {
            imminent_logits,
            warning_logits,
            early_logits,
        }
    }

    /// Get predicted classes (argmax of logits)
    pub fn predict(&self, x: &Variable) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let output = self.forward_multi(x);

        let imminent_probs = self.softmax.forward(&output.imminent_logits);
        let warning_probs = self.softmax.forward(&output.warning_logits);
        let early_probs = self.softmax.forward(&output.early_logits);

        (
            argmax_batch(&imminent_probs),
            argmax_batch(&warning_probs),
            argmax_batch(&early_probs),
        )
    }

    /// Returns the model configuration
    pub fn config(&self) -> &HvacConfig {
        &self.config
    }

    /// Returns the number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| p.variable().data().numel())
            .sum()
    }
}

impl Module for HvacPredictor {
    fn forward(&self, x: &Variable) -> Variable {
        // Return concatenated logits for all horizons
        let output = self.forward_multi(x);
        // For single output, return imminent predictions
        output.imminent_logits
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.input_proj.parameters();
        params.extend(self.input_norm.parameters());
        params.extend(self.gru.parameters());
        params.extend(self.head_imminent.parameters());
        params.extend(self.head_warning.parameters());
        params.extend(self.head_early.parameters());
        params
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get argmax for each sample in batch
fn argmax_batch(x: &Variable) -> Vec<usize> {
    let data = x.data();
    let shape = data.shape();
    let batch_size = shape[0];
    let num_classes = shape[1];
    let values = data.to_vec();

    let mut results = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let start = b * num_classes;
        let end = start + num_classes;
        let slice = &values[start..end];

        let mut max_idx = 0;
        let mut max_val = slice[0];
        for (i, &v) in slice.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        results.push(max_idx);
    }
    results
}

/// Failure type names
pub const FAILURE_TYPES: [&str; 20] = [
    "normal",
    "pump_failure_hw_5",
    "pump_failure_hw_6",
    "pump_failure_cw_3",
    "pump_failure_cw_4",
    "pump_failure_2pipe_a",
    "pump_failure_2pipe_b",
    "pressure_low_hw",
    "pressure_high_hw",
    "pressure_low_cw",
    "pressure_high_cw",
    "temp_anomaly_hw_supply",
    "temp_anomaly_cw_supply",
    "temp_anomaly_space",
    "valve_stuck_1_3",
    "valve_stuck_2_3",
    "vfd_fault",
    "sensor_drift",
    "chiller_fault",
    "interlock_violation",
];

/// Feature names for the 28 sensor inputs
pub const FEATURE_NAMES: [&str; 28] = [
    "hw_pump_5_current",
    "hw_pump_6_current",
    "cw_pump_3_current",
    "cw_pump_4_current",
    "2pipe_pump_a_current",
    "2pipe_pump_b_current",
    "hw_supply_4pipe_temp",
    "cw_supply_4pipe_temp",
    "hw_supply_2pipe_temp",
    "cw_return_2pipe_temp",
    "outdoor_air_temp",
    "mech_room_temp",
    "space_sensor_1_temp",
    "space_sensor_2_temp",
    "hw_pressure_4pipe",
    "cw_pressure_4pipe",
    "hw_pump_5_vfd_speed",
    "hw_pump_6_vfd_speed",
    "cw_pump_3_vfd_speed",
    "cw_pump_4_vfd_speed",
    "2pipe_pump_a_vfd_speed",
    "2pipe_pump_b_vfd_speed",
    "steam_valve_1_3_pos",
    "steam_valve_2_3_pos",
    "summer_winter_mode",
    "hw_lead_pump_id",
    "cw_lead_pump_id",
    "2pipe_lead_pump_id",
];

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     HVAC Multi-Horizon Predictor - AxonML Native           ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Create model with default config
    let config = HvacConfig::default();
    println!("Model Configuration:");
    println!("  Input features: {}", config.num_features);
    println!("  Sequence length: {}", config.seq_len);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  GRU layers: {}", config.num_layers);
    println!("  Output classes: {}", config.num_classes);
    println!("  Dropout: {}", config.dropout);
    println!();

    let model = HvacPredictor::new(config.clone());
    println!("Model created!");
    println!("  Total parameters: {}", model.num_parameters());
    println!();

    // Create sample input
    let batch_size = 2;
    let mut input_data = vec![0.5f32; batch_size * config.seq_len * config.num_features];

    // Simulate normal HVAC readings
    for b in 0..batch_size {
        for t in 0..config.seq_len {
            let base = (b * config.seq_len + t) * config.num_features;
            // Pump currents ~25A (normalized)
            for i in 0..6 {
                input_data[base + i] = 0.5;
            }
            // Temperatures (normalized)
            input_data[base + 6] = 0.83; // HW supply ~180F
            input_data[base + 7] = 0.375; // CW supply ~55F
                                          // VFD speeds ~60%
            for i in 16..22 {
                input_data[base + i] = 0.6;
            }
        }
    }

    let input = Tensor::from_vec(
        input_data,
        &[batch_size, config.seq_len, config.num_features],
    )
    .expect("Failed to create input tensor");

    let input_var = Variable::new(input, false);
    println!("Input shape: {:?}", input_var.data().shape());

    // Run inference
    println!();
    println!("Running inference...");
    let (imminent, warning, early) = model.predict(&input_var);

    println!();
    println!("Predictions:");
    println!("────────────────────────────────────────────────────────────");
    for b in 0..batch_size {
        println!("Sample {}:", b);
        println!(
            "  5 min (Imminent): {} - {}",
            imminent[b], FAILURE_TYPES[imminent[b]]
        );
        println!(
            "  15 min (Warning): {} - {}",
            warning[b], FAILURE_TYPES[warning[b]]
        );
        println!(
            "  30 min (Early):   {} - {}",
            early[b], FAILURE_TYPES[early[b]]
        );
    }
    println!("────────────────────────────────────────────────────────────");
    println!();
    println!("Model ready for training with your HVAC sensor data!");
}
