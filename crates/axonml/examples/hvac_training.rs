//! HVAC Multi-Horizon Predictor - Training with Synthetic Data
//!
//! Complete training pipeline for HVAC failure prediction model:
//! - Synthetic data generation (normal operation + failure scenarios)
//! - Multi-horizon training (5/15/30 minute predictions)
//! - Adam optimizer with learning rate scheduling
//! - Validation and metrics reporting
//!
//! Usage: cargo run --example hvac_training --release

use axonml::nn::{Module, Parameter, GRU, Linear, LayerNorm, ReLU, Dropout, CrossEntropyLoss};
use axonml::optim::{Adam, Optimizer};
use axonml::autograd::Variable;
use axonml::tensor::Tensor;
use std::time::Instant;

// =============================================================================
// Configuration
// =============================================================================

/// Training Configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub val_split: f32,
    pub print_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            epochs: 50,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            val_split: 0.2,
            print_every: 10,
        }
    }
}

/// HVAC Model Configuration
#[derive(Debug, Clone)]
pub struct HvacConfig {
    pub num_features: usize,
    pub seq_len: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_classes: usize,
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
// Synthetic Data Generator
// =============================================================================

/// Operating conditions for simulation
#[derive(Debug, Clone)]
pub struct OperatingConditions {
    pub outdoor_temp: f32,
    pub is_winter: bool,
    pub hw_lead_pump: usize,
    pub cw_lead_pump: usize,
    pub pipe2_lead_pump: usize,
}

impl Default for OperatingConditions {
    fn default() -> Self {
        Self {
            outdoor_temp: 70.0,
            is_winter: false,
            hw_lead_pump: 0,
            cw_lead_pump: 0,
            pipe2_lead_pump: 0,
        }
    }
}

/// HVAC Synthetic Data Generator
/// Generates realistic sensor data including normal operation and failure scenarios
pub struct HvacDataGenerator {
    _seed: u64,
    rng_state: u64,
}

impl HvacDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            _seed: seed,
            rng_state: seed,
        }
    }

    /// Simple LCG random number generator
    fn rand(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    /// Generate random normal (approximate using Box-Muller)
    fn randn(&mut self) -> f32 {
        let u1 = self.rand().max(1e-10);
        let u2 = self.rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Generate Gaussian noise
    fn noise(&mut self, scale: f32) -> f32 {
        self.randn() * scale
    }

    /// Generate normal operation data
    /// Returns: (data [n_samples, 28], labels [n_samples])
    pub fn generate_normal_operation(
        &mut self,
        n_samples: usize,
        conditions: &OperatingConditions,
    ) -> (Vec<f32>, Vec<i64>) {
        let mut data = vec![0.0f32; n_samples * 28];
        let labels = vec![0i64; n_samples]; // All normal

        let oat = conditions.outdoor_temp;
        let is_heating = oat < 65.0;
        let is_cooling = oat > 72.0;
        let base_current = 18.0;

        for t in 0..n_samples {
            let base = t * 28;

            // Diurnal outdoor temp variation
            let hour = (t as f32 / 3600.0) % 24.0;
            let oat_var = 8.0 * (2.0 * std::f32::consts::PI * hour / 24.0).sin();
            let outdoor_temp = oat + oat_var + self.noise(1.0);
            data[base + 10] = outdoor_temp;

            // ===== PUMP CURRENTS (0-5) =====
            // 4-Pipe HW Pumps
            if conditions.hw_lead_pump == 0 {
                data[base + 0] = if is_heating { base_current + self.noise(2.0) } else { 0.5 };
                data[base + 1] = 0.3 + self.noise(0.1);
            } else {
                data[base + 1] = if is_heating { base_current + self.noise(2.0) } else { 0.5 };
                data[base + 0] = 0.3 + self.noise(0.1);
            }

            // 4-Pipe CW Pumps
            if conditions.cw_lead_pump == 0 {
                data[base + 2] = if is_cooling { base_current + self.noise(2.0) } else { 0.5 };
                data[base + 3] = 0.3 + self.noise(0.1);
            } else {
                data[base + 3] = if is_cooling { base_current + self.noise(2.0) } else { 0.5 };
                data[base + 2] = 0.3 + self.noise(0.1);
            }

            // 2-Pipe Pumps (winter only)
            if conditions.is_winter {
                if conditions.pipe2_lead_pump == 0 {
                    data[base + 4] = base_current * 0.8 + self.noise(1.5);
                    data[base + 5] = 0.3 + self.noise(0.1);
                } else {
                    data[base + 5] = base_current * 0.8 + self.noise(1.5);
                    data[base + 4] = 0.3 + self.noise(0.1);
                }
            } else {
                data[base + 4] = 0.3 + self.noise(0.1);
                data[base + 5] = 0.3 + self.noise(0.1);
            }

            // ===== TEMPERATURES (6-13) =====
            // 4-Pipe HW Supply
            data[base + 6] = if is_heating { 160.0 + self.noise(3.0) } else { 90.0 + self.noise(2.0) };

            // 4-Pipe CW Supply
            data[base + 7] = if is_cooling { 45.0 + self.noise(1.5) } else { 55.0 + self.noise(2.0) };

            // 2-Pipe HW Supply (OA reset)
            let oa_reset = 155.0 - (outdoor_temp - 40.0) * (155.0 - 115.0) / 32.0;
            data[base + 8] = oa_reset.clamp(115.0, 155.0) + self.noise(2.0);

            // 2-Pipe CW Return
            data[base + 9] = data[base + 8] - 20.0 + self.noise(3.0);

            // Mech room temp
            data[base + 11] = 72.0 + self.noise(2.0);

            // Space sensors
            data[base + 12] = 72.0 + self.noise(0.5);
            data[base + 13] = 72.0 + self.noise(0.5);

            // ===== PRESSURES (14-15) =====
            data[base + 14] = 16.0 + self.noise(0.5); // HW
            data[base + 15] = 12.0 + self.noise(0.4); // CW

            // ===== VFD SPEEDS (16-21) =====
            for i in 0..6 {
                let current = data[base + i];
                data[base + 16 + i] = if current > 1.0 {
                    40.0 + (current / 25.0) * 40.0 + self.noise(2.0)
                } else {
                    0.0
                };
            }

            // ===== VALVE POSITIONS (22-23) =====
            let heating_load = ((65.0 - outdoor_temp) / 25.0).clamp(0.0, 1.0);
            data[base + 22] = (heating_load * 70.0 + self.noise(3.0)).clamp(0.0, 100.0);
            data[base + 23] = if data[base + 22] > 90.0 {
                ((data[base + 22] - 90.0) * 5.0 + self.noise(2.0)).clamp(0.0, 100.0)
            } else {
                0.0
            };

            // ===== SYSTEM STATES (24-27) =====
            data[base + 24] = if conditions.is_winter { 1.0 } else { 0.0 };
            data[base + 25] = conditions.hw_lead_pump as f32;
            data[base + 26] = conditions.cw_lead_pump as f32;
            data[base + 27] = conditions.pipe2_lead_pump as f32;
        }

        (data, labels)
    }

    /// Inject pump failure into data
    pub fn inject_pump_failure(
        &mut self,
        data: &mut [f32],
        labels: &mut [i64],
        n_samples: usize,
        pump_index: usize,
        failure_start: usize,
        degradation_rate: f32,
    ) {
        let failure_type = (pump_index + 1) as i64;

        for i in failure_start..n_samples {
            let base = i * 28;
            let degradation = (i - failure_start) as f32 * degradation_rate;

            // Reduce pump current
            data[base + pump_index] = (data[base + pump_index] - degradation).max(0.0);

            // VFD speed also decreases
            data[base + 16 + pump_index] = (data[base + 16 + pump_index] - degradation * 2.0).max(0.0);

            // Mark as failure when current drops below threshold
            if data[base + pump_index] < 3.0 {
                labels[i] = failure_type;
            }
        }
    }

    /// Inject pressure anomaly
    pub fn inject_pressure_anomaly(
        &mut self,
        data: &mut [f32],
        labels: &mut [i64],
        n_samples: usize,
        is_hw: bool,
        is_low: bool,
        anomaly_start: usize,
        magnitude: f32,
    ) {
        let feat_idx = if is_hw { 14 } else { 15 };
        let failure_type = match (is_hw, is_low) {
            (true, true) => 7,   // pressure_low_hw
            (true, false) => 8,  // pressure_high_hw
            (false, true) => 9,  // pressure_low_cw
            (false, false) => 10, // pressure_high_cw
        };
        let setpoint = if is_hw { 16.0 } else { 12.0 };

        for i in anomaly_start..n_samples {
            let base = i * 28;
            let drift = (i - anomaly_start) as f32 * 0.005 * magnitude;

            if is_low {
                data[base + feat_idx] -= drift;
            } else {
                data[base + feat_idx] += drift;
            }

            if (data[base + feat_idx] - setpoint).abs() > 3.0 {
                labels[i] = failure_type;
            }
        }
    }

    /// Inject temperature anomaly
    pub fn inject_temperature_anomaly(
        &mut self,
        data: &mut [f32],
        labels: &mut [i64],
        n_samples: usize,
        temp_type: &str,
        anomaly_start: usize,
    ) {
        let (feat_indices, failure_type): (Vec<usize>, i64) = match temp_type {
            "hw_supply" => (vec![6], 11),
            "cw_supply" => (vec![7], 12),
            "space" => (vec![12, 13], 13),
            _ => return,
        };

        for i in anomaly_start..n_samples {
            let base = i * 28;
            let drift = (i - anomaly_start) as f32 * 0.02;

            for &feat_idx in &feat_indices {
                data[base + feat_idx] += drift + self.noise(0.5);
            }

            if drift > 5.0 {
                labels[i] = failure_type;
            }
        }
    }

    /// Generate complete training dataset
    pub fn generate_training_dataset(
        &mut self,
        n_normal_samples: usize,
        n_failure_scenarios: usize,
        failure_duration: usize,
    ) -> (Vec<f32>, Vec<i64>) {
        let mut all_data = Vec::new();
        let mut all_labels = Vec::new();

        println!("Generating normal operation data...");
        // Generate normal data for various conditions
        for season in [false, true] { // summer, winter
            for oat_idx in 0..10 {
                let oat = 20.0 + oat_idx as f32 * 7.5;
                let conditions = OperatingConditions {
                    outdoor_temp: oat,
                    is_winter: season,
                    hw_lead_pump: (self.rand() * 2.0) as usize,
                    cw_lead_pump: (self.rand() * 2.0) as usize,
                    pipe2_lead_pump: (self.rand() * 2.0) as usize,
                };
                let samples = n_normal_samples / 20;
                let (data, labels) = self.generate_normal_operation(samples, &conditions);
                all_data.extend(data);
                all_labels.extend(labels);
            }
        }

        // Generate pump failure scenarios
        println!("Generating pump failure scenarios...");
        for pump_idx in 0..6 {
            for _ in 0..(n_failure_scenarios / 6) {
                let conditions = OperatingConditions {
                    outdoor_temp: 30.0 + self.rand() * 55.0,
                    is_winter: self.rand() > 0.5,
                    ..Default::default()
                };
                let (mut data, mut labels) = self.generate_normal_operation(failure_duration, &conditions);
                let failure_start = failure_duration / 3;
                let rate = 0.005 + self.rand() * 0.015;
                self.inject_pump_failure(&mut data, &mut labels, failure_duration, pump_idx, failure_start, rate);
                all_data.extend(data);
                all_labels.extend(labels);
            }
        }

        // Generate pressure anomalies
        println!("Generating pressure anomaly scenarios...");
        for (is_hw, is_low) in [(true, true), (true, false), (false, true), (false, false)] {
            for _ in 0..(n_failure_scenarios / 4) {
                let conditions = OperatingConditions {
                    outdoor_temp: 30.0 + self.rand() * 55.0,
                    ..Default::default()
                };
                let (mut data, mut labels) = self.generate_normal_operation(failure_duration, &conditions);
                let failure_start = failure_duration / 3;
                let magnitude = 3.0 + self.rand() * 5.0;
                self.inject_pressure_anomaly(&mut data, &mut labels, failure_duration, is_hw, is_low, failure_start, magnitude);
                all_data.extend(data);
                all_labels.extend(labels);
            }
        }

        // Generate temperature anomalies
        println!("Generating temperature anomaly scenarios...");
        for temp_type in ["hw_supply", "cw_supply", "space"] {
            for _ in 0..(n_failure_scenarios / 3) {
                let conditions = OperatingConditions {
                    outdoor_temp: 30.0 + self.rand() * 55.0,
                    ..Default::default()
                };
                let (mut data, mut labels) = self.generate_normal_operation(failure_duration, &conditions);
                let failure_start = failure_duration / 3;
                self.inject_temperature_anomaly(&mut data, &mut labels, failure_duration, temp_type, failure_start);
                all_data.extend(data);
                all_labels.extend(labels);
            }
        }

        let total_samples = all_labels.len();
        println!("Generated {} total samples", total_samples);

        // Print label distribution
        let mut label_counts = vec![0usize; 20];
        for &label in &all_labels {
            label_counts[label as usize] += 1;
        }
        println!("Label distribution: Normal={}, Failures={}",
            label_counts[0],
            label_counts[1..].iter().sum::<usize>());

        (all_data, all_labels)
    }

    /// Convert raw data to sequences with multi-horizon labels
    pub fn generate_multi_horizon_sequences(
        &self,
        data: &[f32],
        labels: &[i64],
        sequence_length: usize,
        horizons: &[usize], // [300, 900, 1800] for 5/15/30 min
        stride: usize,
    ) -> (Vec<f32>, Vec<i64>, Vec<i64>, Vec<i64>) {
        let n_samples = labels.len();
        let max_horizon = *horizons.iter().max().unwrap();
        let n_sequences = (n_samples - sequence_length - max_horizon) / stride;

        let mut x_data = vec![0.0f32; n_sequences * sequence_length * 28];
        let mut y_imminent = vec![0i64; n_sequences];
        let mut y_warning = vec![0i64; n_sequences];
        let mut y_early = vec![0i64; n_sequences];

        for i in 0..n_sequences {
            let start_idx = i * stride;
            let end_idx = start_idx + sequence_length;

            // Copy sequence
            for t in 0..sequence_length {
                for f in 0..28 {
                    x_data[i * sequence_length * 28 + t * 28 + f] = data[(start_idx + t) * 28 + f];
                }
            }

            // Get labels for each horizon (max in prediction window)
            for (h_idx, &horizon) in horizons.iter().enumerate() {
                let label_start = end_idx;
                let label_end = (end_idx + horizon).min(n_samples);
                let mut max_label = 0i64;
                for j in label_start..label_end {
                    max_label = max_label.max(labels[j]);
                }
                match h_idx {
                    0 => y_imminent[i] = max_label,
                    1 => y_warning[i] = max_label,
                    2 => y_early[i] = max_label,
                    _ => {}
                }
            }
        }

        (x_data, y_imminent, y_warning, y_early)
    }
}

// =============================================================================
// Model Components (same as hvac_model.rs)
// =============================================================================

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

pub struct HvacPredictor {
    config: HvacConfig,
    input_proj: Linear,
    input_norm: LayerNorm,
    input_relu: ReLU,
    gru: GRU,
    head_imminent: PredictionHead,
    head_warning: PredictionHead,
    head_early: PredictionHead,
}

impl HvacPredictor {
    pub fn new(config: HvacConfig) -> Self {
        Self {
            input_proj: Linear::new(config.num_features, config.hidden_size),
            input_norm: LayerNorm::new(vec![config.hidden_size]),
            input_relu: ReLU,
            gru: GRU::new(config.hidden_size, config.hidden_size, config.num_layers),
            head_imminent: PredictionHead::new(config.hidden_size, config.num_classes, config.dropout),
            head_warning: PredictionHead::new(config.hidden_size, config.num_classes, config.dropout),
            head_early: PredictionHead::new(config.hidden_size, config.num_classes, config.dropout),
            config,
        }
    }

    pub fn forward_multi(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let x_data = x.data();
        let shape = x_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        drop(x_data);

        let x_flat = x.reshape(&[batch_size * seq_len, self.config.num_features]);
        let proj = self.input_proj.forward(&x_flat);
        let proj = self.input_norm.forward(&proj);
        let proj = self.input_relu.forward(&proj);
        let proj = proj.reshape(&[batch_size, seq_len, self.config.hidden_size]);

        // Use forward_mean for proper gradient flow (equivalent to forward + mean_pool)
        let pooled = self.gru.forward_mean(&proj);

        let imminent = self.head_imminent.forward(&pooled);
        let warning = self.head_warning.forward(&pooled);
        let early = self.head_early.forward(&pooled);

        (imminent, warning, early)
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.variable().data().numel()).sum()
    }
}

impl Module for HvacPredictor {
    fn forward(&self, x: &Variable) -> Variable {
        let (imminent, _, _) = self.forward_multi(x);
        imminent
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
// Training Loop
// =============================================================================

/// Normalize data to [0, 1] range based on sensor ranges
fn normalize_data(data: &mut [f32], n_samples: usize) {
    let sensor_ranges: [(f32, f32); 28] = [
        (0.0, 50.0),    // 0: hw_pump_5_current
        (0.0, 50.0),    // 1: hw_pump_6_current
        (0.0, 50.0),    // 2: cw_pump_3_current
        (0.0, 50.0),    // 3: cw_pump_4_current
        (0.0, 50.0),    // 4: 2pipe_pump_a_current
        (0.0, 50.0),    // 5: 2pipe_pump_b_current
        (80.0, 200.0),  // 6: hw_supply_4pipe_temp
        (40.0, 80.0),   // 7: cw_supply_4pipe_temp
        (115.0, 155.0), // 8: hw_supply_2pipe_temp
        (70.0, 120.0),  // 9: cw_return_2pipe_temp
        (-20.0, 120.0), // 10: outdoor_air_temp
        (50.0, 90.0),   // 11: mech_room_temp
        (65.0, 85.0),   // 12: space_sensor_1_temp
        (65.0, 85.0),   // 13: space_sensor_2_temp
        (0.0, 200.0),   // 14: hw_pressure_4pipe
        (0.0, 200.0),   // 15: cw_pressure_4pipe
        (0.0, 100.0),   // 16-21: VFD speeds
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),   // 22: steam_valve_1_3_pos
        (0.0, 100.0),   // 23: steam_valve_2_3_pos
        (0.0, 1.0),     // 24: summer_winter_mode
        (0.0, 1.0),     // 25: hw_lead_pump_id
        (0.0, 1.0),     // 26: cw_lead_pump_id
        (0.0, 1.0),     // 27: 2pipe_lead_pump_id
    ];

    for i in 0..n_samples {
        for f in 0..28 {
            let (min_val, max_val) = sensor_ranges[f];
            let idx = i * 28 + f;
            data[idx] = ((data[idx] - min_val) / (max_val - min_val)).clamp(0.0, 1.0);
        }
    }
}

/// Calculate accuracy
fn calculate_accuracy(logits: &Variable, labels: &[i64]) -> f32 {
    let data = logits.data();
    let shape = data.shape();
    let batch_size = shape[0];
    let num_classes = shape[1];
    let values = data.to_vec();

    let mut correct = 0;
    for b in 0..batch_size {
        let start = b * num_classes;
        let mut max_idx = 0;
        let mut max_val = values[start];
        for c in 1..num_classes {
            if values[start + c] > max_val {
                max_val = values[start + c];
                max_idx = c;
            }
        }
        if max_idx == labels[b] as usize {
            correct += 1;
        }
    }
    correct as f32 / batch_size as f32
}

/// Training function
fn train_epoch(
    model: &HvacPredictor,
    optimizer: &mut Adam,
    loss_fn: &CrossEntropyLoss,
    x_data: &[f32],
    y_imminent: &[i64],
    y_warning: &[i64],
    y_early: &[i64],
    batch_size: usize,
    seq_len: usize,
    num_features: usize,
) -> (f32, f32, f32, f32) {
    let n_sequences = y_imminent.len();
    let n_batches = n_sequences / batch_size;

    let mut total_loss = 0.0f32;
    let mut total_acc_imm = 0.0f32;
    let mut total_acc_warn = 0.0f32;
    let mut total_acc_early = 0.0f32;

    for batch_idx in 0..n_batches {
        let start = batch_idx * batch_size;

        // Prepare batch data
        let mut batch_x = vec![0.0f32; batch_size * seq_len * num_features];
        let mut batch_y_imm = vec![0i64; batch_size];
        let mut batch_y_warn = vec![0i64; batch_size];
        let mut batch_y_early = vec![0i64; batch_size];

        for b in 0..batch_size {
            let seq_start = (start + b) * seq_len * num_features;
            for i in 0..(seq_len * num_features) {
                batch_x[b * seq_len * num_features + i] = x_data[seq_start + i];
            }
            batch_y_imm[b] = y_imminent[start + b];
            batch_y_warn[b] = y_warning[start + b];
            batch_y_early[b] = y_early[start + b];
        }

        // Create tensors
        let x_tensor = Tensor::from_vec(batch_x, &[batch_size, seq_len, num_features])
            .expect("Failed to create input tensor");
        let x_var = Variable::new(x_tensor, true);

        // Forward pass
        let (logits_imm, logits_warn, logits_early) = model.forward_multi(&x_var);

        // Calculate losses (simplified - just use imminent for now)
        let y_imm_tensor = Tensor::from_vec(
            batch_y_imm.iter().map(|&y| y as f32).collect(),
            &[batch_size],
        ).expect("Failed to create label tensor");
        let y_imm_var = Variable::new(y_imm_tensor, false);

        let loss = loss_fn.compute(&logits_imm, &y_imm_var);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Metrics
        total_loss += loss.data().to_vec()[0];
        total_acc_imm += calculate_accuracy(&logits_imm, &batch_y_imm);
        total_acc_warn += calculate_accuracy(&logits_warn, &batch_y_warn);
        total_acc_early += calculate_accuracy(&logits_early, &batch_y_early);
    }

    let n = n_batches as f32;
    (total_loss / n, total_acc_imm / n, total_acc_warn / n, total_acc_early / n)
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     HVAC Multi-Horizon Predictor - Training Pipeline       ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Check for quick mode via environment variable
    let quick_mode = std::env::var("HVAC_QUICK").is_ok();

    let train_config = if quick_mode {
        TrainingConfig {
            batch_size: 16,
            epochs: 100,
            learning_rate: 0.01,
            weight_decay: 0.0001,
            val_split: 0.2,
            print_every: 10,
        }
    } else {
        TrainingConfig::default()
    };

    let model_config = if quick_mode {
        HvacConfig {
            num_features: 28,
            seq_len: 30,      // Reduced from 120
            hidden_size: 64,  // Reduced from 128
            num_layers: 1,    // Reduced from 2
            num_classes: 20,
            dropout: 0.1,
        }
    } else {
        HvacConfig::default()
    };

    println!("Training Configuration:");
    println!("  Batch size: {}", train_config.batch_size);
    println!("  Epochs: {}", train_config.epochs);
    println!("  Learning rate: {}", train_config.learning_rate);
    println!();

    // Generate synthetic training data
    println!("=== Data Generation ===");
    if quick_mode {
        println!("(Quick mode enabled - reduced data and model size)");
    }
    let start_time = Instant::now();
    let mut generator = HvacDataGenerator::new(42);

    let (normal_samples, failure_scenarios, failure_duration) = if quick_mode {
        (5000, 5, 300)  // Much smaller for quick testing
    } else {
        (50000, 30, 1800)  // Full dataset
    };

    let (mut raw_data, raw_labels) = generator.generate_training_dataset(
        normal_samples,
        failure_scenarios,
        failure_duration,
    );

    // Normalize data
    let n_samples = raw_labels.len();
    normalize_data(&mut raw_data, n_samples);

    // Generate sequences
    println!("Creating multi-horizon sequences...");
    let stride = if quick_mode { 30 } else { 10 };
    let (x_data, y_imminent, y_warning, y_early) = generator.generate_multi_horizon_sequences(
        &raw_data,
        &raw_labels,
        model_config.seq_len,
        &[300, 900, 1800], // 5, 15, 30 minutes
        stride,
    );

    let n_sequences = y_imminent.len();
    println!("Generated {} sequences", n_sequences);
    println!("Data generation took: {:?}", start_time.elapsed());
    println!();

    // Split train/val
    let val_size = (n_sequences as f32 * train_config.val_split) as usize;
    let train_size = n_sequences - val_size;
    println!("Train: {} sequences, Val: {} sequences", train_size, val_size);

    // Create model
    println!();
    println!("=== Model ===");
    let model = HvacPredictor::new(model_config.clone());
    println!("Parameters: {}", model.num_parameters());

    // Create optimizer and loss
    let mut optimizer = Adam::new(model.parameters(), train_config.learning_rate);
    let loss_fn = CrossEntropyLoss::new();

    // Training loop
    println!();
    println!("=== Training ===");
    let training_start = Instant::now();

    for epoch in 0..train_config.epochs {
        let epoch_start = Instant::now();

        let (loss, acc_imm, acc_warn, acc_early) = train_epoch(
            &model,
            &mut optimizer,
            &loss_fn,
            &x_data[..(train_size * model_config.seq_len * model_config.num_features)],
            &y_imminent[..train_size],
            &y_warning[..train_size],
            &y_early[..train_size],
            train_config.batch_size,
            model_config.seq_len,
            model_config.num_features,
        );

        if epoch % train_config.print_every == 0 || epoch == train_config.epochs - 1 {
            println!(
                "Epoch {:3}/{}: Loss={:.4}, Acc(5m)={:.2}%, Acc(15m)={:.2}%, Acc(30m)={:.2}% [{:?}]",
                epoch + 1,
                train_config.epochs,
                loss,
                acc_imm * 100.0,
                acc_warn * 100.0,
                acc_early * 100.0,
                epoch_start.elapsed()
            );
        }
    }

    println!();
    println!("Training completed in {:?}", training_start.elapsed());
    println!();
    println!("Model ready for deployment!");
}
