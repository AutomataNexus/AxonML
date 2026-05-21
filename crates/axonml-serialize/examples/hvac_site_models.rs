//! HVAC Site Models — Train + Export LSTM and GRU Predictors for 3 Sites
//!
//! End-to-end pipeline that:
//!   1. Builds LSTM-based and GRU-based `HvacPredictor` models for chosen
//!      sites (fcog-mechroom, taylor-greenhouse, peabody-mechroom).
//!   2. Trains each on synthetic sensor data (same generator pattern as
//!      `gru_train_and_export.rs`).
//!   3. Saves each as a v3 `.axonml` bundle with an embedded `BundleGraph`
//!      representing the architecture using ONNX-compatible ops.
//!
//! ## Usage
//!
//! ```bash
//! # All sites, all architectures:
//! cargo run --release --example hvac_site_models -p axonml-serialize -- all all
//!
//! # Single site, single arch:
//! cargo run --release --example hvac_site_models -p axonml-serialize -- \
//!     fcog-mechroom lstm
//!
//! # Single site, both archs:
//! cargo run --release --example hvac_site_models -p axonml-serialize -- \
//!     taylor-greenhouse all
//! ```
//!
//! ## Output bundles
//!
//! `/opt/AxonML-Hailo/hailo8/edgemodels/bundles/`
//!   - `fcog_mechroom_lstm.axonml`
//!   - `fcog_mechroom_gru.axonml`
//!   - `taylor_greenhouse_lstm.axonml`
//!   - `taylor_greenhouse_gru.axonml`
//!   - `peabody_mechroom_lstm.axonml`
//!   - `peabody_mechroom_gru.axonml`
//!
//! # File
//! `crates/axonml-serialize/examples/hvac_site_models.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! May 2, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use axonml::TrainingMonitor;
use axonml::autograd::Variable;
use axonml::nn::{
    CrossEntropyLoss, Dropout, GRU, LSTM, LayerNorm, Linear, Module, Parameter, ReLU,
};
use axonml::optim::{Adam, Optimizer};
use axonml::tensor::Tensor;

use axonml_serialize::{
    BundleGraph, Checkpoint, CheckpointBuilder, ModelBundle, StateDict, TensorData,
    load_checkpoint, save_bundle, save_checkpoint,
};

use std::path::PathBuf;
use std::time::Instant;

// =============================================================================
// Site Configuration
// =============================================================================

#[derive(Debug, Clone)]
struct SiteConfig {
    slug: &'static str,
    num_features: usize,
    num_classes: usize,
    hidden_size: usize,
    num_layers: usize,
}

const SEQ_LEN: usize = 60;
const DROPOUT: f32 = 0.1;

const SITES: &[SiteConfig] = &[
    // FCOG Mechroom: 2 chillers, 4 boilers, 6 pumps, 4 VFDs
    // Features: 6 pump amps, 2 compressor amps, 2 CW temps, 2 HW temps,
    //   4 setpoints, 12 enable commands, 4 VFD speed refs, 3 lead/lag/failover,
    //   12 runtimes, 12 user commands = 57 features
    SiteConfig {
        slug: "fcog-mechroom",
        num_features: 57,
        num_classes: 12,
        hidden_size: 128,
        num_layers: 2,
    },
    // Taylor Greenhouse: supply fan, 2 exhaust fans, 4 unit heaters,
    //   2 wall water coils, 2 space temps, 2 RH
    // Features: 1 supply fan speed, 2 exhaust fan status, 4 heater enable,
    //   4 heater status, 2 coil valve position, 2 space temps, 2 RH,
    //   1 OAT, 2 setpoints = 22 features
    SiteConfig {
        slug: "taylor-greenhouse",
        num_features: 22,
        num_classes: 8,
        hidden_size: 64,
        num_layers: 2,
    },
    // Peabody Mechroom: 3 cooling towers, 3 CW pumps on VFDs, 2 boilers,
    //   heat exchanger, condenser loop temps, boiler loop temps,
    //   3 heat pump loop pumps, 2 HW pumps
    // Features: 3 CT amps, 3 CW pump amps, 3 VFD speed refs,
    //   2 boiler enable/status, 1 HX valve, 2 condenser temps,
    //   2 boiler temps, 3 HP pump on/off, 2 HW pump on/off,
    //   3 CT fan status, 4 setpoints, 3 runtimes = 34 features
    SiteConfig {
        slug: "peabody-mechroom",
        num_features: 34,
        num_classes: 10,
        hidden_size: 96,
        num_layers: 2,
    },
    // Peabody Boiler HP Loop Injection: 2 comfort boilers (lead/lag),
    //   2 HW circulation pumps (lead/lag), 3 HP loop pumps (3-way lead/lag),
    //   1 injection valve (0-10V modulating), 1 bypass valve (on/off triac)
    //   MegaBAS 0: AI1-AI2 HW pump CTs, AI3-AI4 boiler loop supply/return,
    //     AI5 OAT, AI6-AI7 per-boiler supply temps, TR1-TR4 boiler/pump enables,
    //     AO1 injection valve
    //   MegaBAS 1: AI1-AI2+AI5 HP pump CTs, AI3-AI4 HP loop supply/return,
    //     TR1-TR3 HP pump enables, TR4 bypass valve
    // Features: 5 pump CTs (HWP1, HWP2, HPP1, HPP2, HPP3),
    //   7 temps (boiler supply, boiler return, OAT, boiler1 supply, boiler2 supply,
    //     HP loop supply, HP loop return),
    //   1 injection valve position, 2 boiler enables, 2 HW pump enables,
    //   3 HP pump enables, 1 bypass valve = 21 features
    SiteConfig {
        slug: "peabody-boiler-hp-loop",
        num_features: 21,
        num_classes: 8,
        hidden_size: 96,
        num_layers: 2,
    },
];

// =============================================================================
// Architecture selector
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arch {
    Lstm,
    Gru,
}

impl Arch {
    fn as_str(self) -> &'static str {
        match self {
            Arch::Lstm => "lstm",
            Arch::Gru => "gru",
        }
    }
}

// =============================================================================
// Synthetic Data Generator (identical to gru_train_and_export.rs)
// =============================================================================

struct DataGenerator {
    rng_state: u64,
}

impl DataGenerator {
    fn new(seed: u64) -> Self {
        Self { rng_state: seed }
    }

    fn rand(&mut self) -> f32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    fn randn(&mut self) -> f32 {
        let u1 = self.rand().max(1e-10);
        let u2 = self.rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Generate synthetic sensor data for any site configuration.
    ///
    /// Produces `n_samples` rows of `num_features` sensor channels, with
    /// labels in `[0, num_classes)`. Normal operation is class 0; fault
    /// classes are injected at random intervals with gradual degradation.
    fn generate_dataset(
        &mut self,
        num_features: usize,
        num_classes: usize,
        n_samples: usize,
    ) -> (Vec<f32>, Vec<i64>) {
        let mut data = vec![0.0f32; n_samples * num_features];
        let mut labels = vec![0i64; n_samples];

        for t in 0..n_samples {
            let base = t * num_features;
            for f in 0..num_features {
                let phase = (t as f32 / 200.0 + f as f32 * 0.3).sin() * 0.3;
                let trend = (t as f32 / (n_samples as f32)) * 0.1;
                data[base + f] = 0.5 + phase + trend + self.randn() * 0.05;
                data[base + f] = data[base + f].clamp(0.0, 1.0);
            }
        }

        let fault_classes = num_classes - 1;
        if fault_classes > 0 {
            let target_fault_pct = 0.60;
            let fault_samples = (n_samples as f32 * target_fault_pct) as usize;
            let per_class = fault_samples / fault_classes;
            let segment_len = per_class / 3;
            for cls in 1..num_classes {
                for rep in 0..3 {
                    let start = ((self.rand() * 0.7 + 0.1) * n_samples as f32) as usize;
                    let end = (start + segment_len).min(n_samples);
                    let feat_primary = (cls - 1) % num_features;
                    let feat_secondary = (cls * 3 + rep) % num_features;
                    for t in start..end {
                        let base = t * num_features;
                        let degradation = (t - start) as f32 / segment_len.max(1) as f32;
                        data[base + feat_primary] += degradation * 0.8 + self.randn() * 0.05;
                        data[base + feat_secondary] += degradation * 0.3 + self.randn() * 0.03;
                        data[base + feat_primary] = data[base + feat_primary].clamp(-0.5, 2.5);
                        data[base + feat_secondary] = data[base + feat_secondary].clamp(-0.5, 2.5);
                        if degradation > 0.15 {
                            labels[t] = cls as i64;
                        }
                    }
                }
            }
        }

        (data, labels)
    }

    /// Slide windows over raw data to produce (batch, seq_len, features) sequences
    /// with multi-horizon labels (imminent / warning / early).
    fn make_sequences(
        &self,
        data: &[f32],
        labels: &[i64],
        num_features: usize,
        seq_len: usize,
        stride: usize,
    ) -> (Vec<f32>, Vec<i64>, Vec<i64>, Vec<i64>) {
        let n_samples = labels.len();
        let horizons = [50, 150, 300];
        let max_horizon = horizons[2];
        if n_samples <= seq_len + max_horizon {
            return (vec![], vec![], vec![], vec![]);
        }
        let n_sequences = (n_samples - seq_len - max_horizon) / stride;

        let mut x = vec![0.0f32; n_sequences * seq_len * num_features];
        let mut y_imm = vec![0i64; n_sequences];
        let mut y_warn = vec![0i64; n_sequences];
        let mut y_early = vec![0i64; n_sequences];

        for i in 0..n_sequences {
            let start = i * stride;
            let end = start + seq_len;
            for t in 0..seq_len {
                for f in 0..num_features {
                    x[i * seq_len * num_features + t * num_features + f] =
                        data[(start + t) * num_features + f];
                }
            }
            for (h_idx, &horizon) in horizons.iter().enumerate() {
                let mut max_label = 0i64;
                let label_end = (end + horizon).min(n_samples);
                for j in end..label_end {
                    max_label = max_label.max(labels[j]);
                }
                match h_idx {
                    0 => y_imm[i] = max_label,
                    1 => y_warn[i] = max_label,
                    2 => y_early[i] = max_label,
                    _ => {}
                }
            }
        }

        (x, y_imm, y_warn, y_early)
    }
}

// =============================================================================
// Shared: PredictionHead
// =============================================================================

struct PredictionHead {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    relu: ReLU,
    dropout: Dropout,
}

impl PredictionHead {
    fn new(hidden_size: usize, num_classes: usize, dropout: f32) -> Self {
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
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p.extend(self.fc3.parameters());
        p
    }
}

// =============================================================================
// Model: LSTM-Based HVAC Predictor
// =============================================================================

struct HvacLstmPredictor {
    num_features: usize,
    hidden_size: usize,
    input_proj: Linear,
    input_norm: LayerNorm,
    input_relu: ReLU,
    lstm: LSTM,
    head_imminent: PredictionHead,
    head_warning: PredictionHead,
    head_early: PredictionHead,
}

impl HvacLstmPredictor {
    fn new(cfg: &SiteConfig) -> Self {
        Self {
            num_features: cfg.num_features,
            hidden_size: cfg.hidden_size,
            input_proj: Linear::new(cfg.num_features, cfg.hidden_size),
            input_norm: LayerNorm::new(vec![cfg.hidden_size]),
            input_relu: ReLU,
            lstm: LSTM::new(cfg.hidden_size, cfg.hidden_size, cfg.num_layers),
            head_imminent: PredictionHead::new(cfg.hidden_size, cfg.num_classes, DROPOUT),
            head_warning: PredictionHead::new(cfg.hidden_size, cfg.num_classes, DROPOUT),
            head_early: PredictionHead::new(cfg.hidden_size, cfg.num_classes, DROPOUT),
        }
    }

    fn forward_multi(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let x_data = x.data();
        let shape = x_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        drop(x_data);

        // Project each timestep: (B*S, F) -> (B*S, H)
        let x_flat = x.reshape(&[batch_size * seq_len, self.num_features]);
        let proj = self.input_proj.forward(&x_flat);
        let proj = self.input_norm.forward(&proj);
        let proj = self.input_relu.forward(&proj);
        // Restore sequence shape: (B, S, H)
        let proj = proj.reshape(&[batch_size, seq_len, self.hidden_size]);

        // LSTM forward returns [batch, seq, hidden]; take the last timestep
        let lstm_out = self.lstm.forward(&proj);
        let last = lstm_out.select(1, seq_len - 1);

        let imminent = self.head_imminent.forward(&last);
        let warning = self.head_warning.forward(&last);
        let early = self.head_early.forward(&last);

        (imminent, warning, early)
    }
}

impl Module for HvacLstmPredictor {
    fn forward(&self, x: &Variable) -> Variable {
        let (imm, _, _) = self.forward_multi(x);
        imm
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.input_proj.parameters();
        p.extend(self.input_norm.parameters());
        p.extend(self.lstm.parameters());
        p.extend(self.head_imminent.parameters());
        p.extend(self.head_warning.parameters());
        p.extend(self.head_early.parameters());
        p
    }
}

// =============================================================================
// Model: GRU-Based HVAC Predictor
// =============================================================================

struct HvacGruPredictor {
    num_features: usize,
    hidden_size: usize,
    input_proj: Linear,
    input_norm: LayerNorm,
    input_relu: ReLU,
    gru: GRU,
    head_imminent: PredictionHead,
    head_warning: PredictionHead,
    head_early: PredictionHead,
}

impl HvacGruPredictor {
    fn new(cfg: &SiteConfig) -> Self {
        Self {
            num_features: cfg.num_features,
            hidden_size: cfg.hidden_size,
            input_proj: Linear::new(cfg.num_features, cfg.hidden_size),
            input_norm: LayerNorm::new(vec![cfg.hidden_size]),
            input_relu: ReLU,
            gru: GRU::new(cfg.hidden_size, cfg.hidden_size, cfg.num_layers),
            head_imminent: PredictionHead::new(cfg.hidden_size, cfg.num_classes, DROPOUT),
            head_warning: PredictionHead::new(cfg.hidden_size, cfg.num_classes, DROPOUT),
            head_early: PredictionHead::new(cfg.hidden_size, cfg.num_classes, DROPOUT),
        }
    }

    fn forward_multi(&self, x: &Variable) -> (Variable, Variable, Variable) {
        let x_data = x.data();
        let shape = x_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        drop(x_data);

        // Project each timestep: (B*S, F) -> (B*S, H)
        let x_flat = x.reshape(&[batch_size * seq_len, self.num_features]);
        let proj = self.input_proj.forward(&x_flat);
        let proj = self.input_norm.forward(&proj);
        let proj = self.input_relu.forward(&proj);
        // Restore sequence shape: (B, S, H)
        let proj = proj.reshape(&[batch_size, seq_len, self.hidden_size]);

        // GRU forward_mean returns [batch, hidden] directly (mean over timesteps)
        let pooled = self.gru.forward_mean(&proj);

        let imminent = self.head_imminent.forward(&pooled);
        let warning = self.head_warning.forward(&pooled);
        let early = self.head_early.forward(&pooled);

        (imminent, warning, early)
    }
}

impl Module for HvacGruPredictor {
    fn forward(&self, x: &Variable) -> Variable {
        let (imm, _, _) = self.forward_multi(x);
        imm
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.input_proj.parameters();
        p.extend(self.input_norm.parameters());
        p.extend(self.gru.parameters());
        p.extend(self.head_imminent.parameters());
        p.extend(self.head_warning.parameters());
        p.extend(self.head_early.parameters());
        p
    }
}

// =============================================================================
// Training helpers
// =============================================================================

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

// Training loop for LSTM variant
fn train_lstm(
    model: &HvacLstmPredictor,
    cfg: &SiteConfig,
    x_data: &[f32],
    y_imm: &[i64],
    epochs: usize,
    batch_size: usize,
) {
    let n_sequences = y_imm.len();
    let n_batches = n_sequences / batch_size;
    if n_batches == 0 {
        println!("  warning: not enough sequences to form a batch, skipping training");
        return;
    }

    let param_count: usize = model
        .parameters()
        .iter()
        .map(|p| p.variable().data().numel())
        .sum();
    let monitor = TrainingMonitor::new(&format!("{}_lstm", cfg.slug), param_count)
        .total_epochs(epochs)
        .batch_size(batch_size)
        .launch();
    println!("  monitor: http://localhost:{}", monitor.port());

    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::new();

    for epoch in 0..epochs {
        let t0 = Instant::now();
        let mut total_loss = 0.0f32;
        let mut total_acc = 0.0f32;

        for b in 0..n_batches {
            let start = b * batch_size;

            let mut bx = vec![0.0f32; batch_size * SEQ_LEN * cfg.num_features];
            let mut by = vec![0i64; batch_size];
            for i in 0..batch_size {
                let seq_off = (start + i) * SEQ_LEN * cfg.num_features;
                let dst_off = i * SEQ_LEN * cfg.num_features;
                bx[dst_off..dst_off + SEQ_LEN * cfg.num_features]
                    .copy_from_slice(&x_data[seq_off..seq_off + SEQ_LEN * cfg.num_features]);
                by[i] = y_imm[start + i];
            }

            let x_t =
                Tensor::from_vec(bx, &[batch_size, SEQ_LEN, cfg.num_features]).expect("tensor");
            let x_v = Variable::new(x_t, true);

            let (logits, _, _) = model.forward_multi(&x_v);

            let y_t = Tensor::from_vec(by.iter().map(|&y| y as f32).collect(), &[batch_size])
                .expect("label tensor");
            let y_v = Variable::new(y_t, false);

            let loss = loss_fn.compute(&logits, &y_v);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.data().to_vec()[0];
            total_acc += calculate_accuracy(&logits, &by);
        }

        let avg_loss = total_loss / n_batches as f32;
        let avg_acc = total_acc / n_batches as f32 * 100.0;
        monitor.log_epoch(epoch, avg_loss, None, vec![("acc", avg_acc)]);

        if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
            let mut sd = StateDict::new();
            for p in model.parameters() {
                let d = p.variable().data();
                sd.insert(p.name().to_string(), TensorData::from_tensor(&d));
            }
            let ckpt = CheckpointBuilder::new()
                .model_state(sd)
                .epoch(epoch)
                .build();
            let ckpt_path = format!(
                "/tmp/hvac_ckpts/{}_lstm/epoch_{:04}.ckpt",
                cfg.slug,
                epoch + 1
            );
            std::fs::create_dir_all(format!("/tmp/hvac_ckpts/{}_lstm", cfg.slug)).ok();
            if save_checkpoint(&ckpt, &ckpt_path).is_ok() {
                println!("  checkpoint: {ckpt_path}");
            }
        }

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!(
                "  epoch {}/{}: loss={:.4}, acc={:.1}% [{:?}]",
                epoch + 1,
                epochs,
                avg_loss,
                avg_acc,
                t0.elapsed()
            );
        }
    }
    monitor.set_status("complete");
}

// Training loop for GRU variant
fn train_gru(
    model: &HvacGruPredictor,
    cfg: &SiteConfig,
    x_data: &[f32],
    y_imm: &[i64],
    epochs: usize,
    batch_size: usize,
) {
    let n_sequences = y_imm.len();
    let n_batches = n_sequences / batch_size;
    if n_batches == 0 {
        println!("  warning: not enough sequences to form a batch, skipping training");
        return;
    }

    let param_count: usize = model
        .parameters()
        .iter()
        .map(|p| p.variable().data().numel())
        .sum();
    let monitor = TrainingMonitor::new(&format!("{}_gru", cfg.slug), param_count)
        .total_epochs(epochs)
        .batch_size(batch_size)
        .launch();
    println!("  monitor: http://localhost:{}", monitor.port());

    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::new();

    for epoch in 0..epochs {
        let t0 = Instant::now();
        let mut total_loss = 0.0f32;
        let mut total_acc = 0.0f32;

        for b in 0..n_batches {
            let start = b * batch_size;

            let mut bx = vec![0.0f32; batch_size * SEQ_LEN * cfg.num_features];
            let mut by = vec![0i64; batch_size];
            for i in 0..batch_size {
                let seq_off = (start + i) * SEQ_LEN * cfg.num_features;
                let dst_off = i * SEQ_LEN * cfg.num_features;
                bx[dst_off..dst_off + SEQ_LEN * cfg.num_features]
                    .copy_from_slice(&x_data[seq_off..seq_off + SEQ_LEN * cfg.num_features]);
                by[i] = y_imm[start + i];
            }

            let x_t =
                Tensor::from_vec(bx, &[batch_size, SEQ_LEN, cfg.num_features]).expect("tensor");
            let x_v = Variable::new(x_t, true);

            let (logits, _, _) = model.forward_multi(&x_v);

            let y_t = Tensor::from_vec(by.iter().map(|&y| y as f32).collect(), &[batch_size])
                .expect("label tensor");
            let y_v = Variable::new(y_t, false);

            let loss = loss_fn.compute(&logits, &y_v);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.data().to_vec()[0];
            total_acc += calculate_accuracy(&logits, &by);
        }

        let avg_loss = total_loss / n_batches as f32;
        let avg_acc = total_acc / n_batches as f32 * 100.0;
        monitor.log_epoch(epoch, avg_loss, None, vec![("acc", avg_acc)]);

        if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
            let mut sd = StateDict::new();
            for p in model.parameters() {
                let d = p.variable().data();
                sd.insert(p.name().to_string(), TensorData::from_tensor(&d));
            }
            let ckpt = CheckpointBuilder::new()
                .model_state(sd)
                .epoch(epoch)
                .build();
            let ckpt_path = format!(
                "/tmp/hvac_ckpts/{}_gru/epoch_{:04}.ckpt",
                cfg.slug,
                epoch + 1
            );
            std::fs::create_dir_all(format!("/tmp/hvac_ckpts/{}_gru", cfg.slug)).ok();
            if save_checkpoint(&ckpt, &ckpt_path).is_ok() {
                println!("  checkpoint: {ckpt_path}");
            }
        }

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!(
                "  epoch {}/{}: loss={:.4}, acc={:.1}% [{:?}]",
                epoch + 1,
                epochs,
                avg_loss,
                avg_acc,
                t0.elapsed()
            );
        }
    }
    monitor.set_status("complete");
}

// =============================================================================
// BundleGraph Builders
// =============================================================================

/// Deterministic Kaiming-uniform init seeded by `seed`.
fn init_kaiming(n: usize, fan_in: usize, seed: u64) -> Vec<f32> {
    let k = (2.0 / fan_in as f64).sqrt() as f32;
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u32;
        let f = (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        out.push(f * k);
    }
    out
}

/// Build the shared input-projection + normalization subgraph.
///
/// Adds: reshape_to_2d, Gemm(input_proj), BatchNorm+Relu, reshape_to_3d.
/// Returns the name of the 3-D projected output tensor `(B, S, H)`.
fn build_input_proj_subgraph(
    g: &mut BundleGraph,
    f: i64,
    h: i64,
    seq: i64,
    seed_base: u64,
) -> String {
    let proj_w_name = "input_proj.weight";
    let proj_b_name = "input_proj.bias";
    g.add_initializer(
        proj_w_name,
        vec![h, f],
        init_kaiming((h * f) as usize, f as usize, seed_base + 1),
    );
    g.add_initializer(proj_b_name, vec![h], vec![0.0; h as usize]);

    // Reshape (B, S, F) -> (B*S, F)
    let reshape_1_shape = "reshape_1_shape";
    g.add_initializer(reshape_1_shape, vec![2], vec![-1.0, f as f32]);
    g.add_node(
        "reshape_to_2d",
        "Reshape",
        serde_json::Value::Null,
        vec!["input", reshape_1_shape],
        vec!["input_2d"],
    );

    // Gemm: (B*S, F) x (H, F)^T -> (B*S, H)
    g.add_node(
        "input_proj",
        "Gemm",
        serde_json::json!({"alpha": 1.0, "beta": 1.0, "trans_a": false, "trans_b": true}),
        vec!["input_2d", proj_w_name, proj_b_name],
        vec!["proj_out"],
    );

    // BatchNorm + Relu on (B*S, H)
    let bn_w = "proj_bn.weight";
    let bn_b = "proj_bn.bias";
    let bn_m = "proj_bn.running_mean";
    let bn_v = "proj_bn.running_var";
    g.add_initializer(bn_w, vec![h], vec![1.0; h as usize]);
    g.add_initializer(bn_b, vec![h], vec![0.0; h as usize]);
    g.add_initializer(bn_m, vec![h], vec![0.0; h as usize]);
    g.add_initializer(bn_v, vec![h], vec![1.0; h as usize]);
    g.add_node(
        "proj_bn",
        "BatchNorm",
        serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        vec!["proj_out", bn_w, bn_b, bn_m, bn_v],
        vec!["proj_bn_out"],
    );
    g.add_node(
        "proj_relu",
        "Relu",
        serde_json::Value::Null,
        vec!["proj_bn_out"],
        vec!["proj_relu_out"],
    );

    // Reshape (B*S, H) -> (B, S, H)
    let reshape_3d_shape = "reshape_3d_shape";
    g.add_initializer(reshape_3d_shape, vec![3], vec![-1.0, seq as f32, h as f32]);
    g.add_node(
        "reshape_to_3d",
        "Reshape",
        serde_json::Value::Null,
        vec!["proj_relu_out", reshape_3d_shape],
        vec!["proj_3d"],
    );

    "proj_3d".to_string()
}

/// Add the three classification heads to `g` reading from `pooled_flat`.
fn build_classification_heads(g: &mut BundleGraph, n_cls: i64, h: i64, seed_base: u64) {
    for (head, out_name) in &[
        ("imminent", "imminent_logits"),
        ("warning", "warning_logits"),
        ("early", "early_logits"),
    ] {
        let w = format!("head_{head}.weight");
        let b = format!("head_{head}.bias");
        g.add_initializer(
            &w,
            vec![n_cls, h],
            init_kaiming(
                (n_cls * h) as usize,
                h as usize,
                seed_base + 100 + head.len() as u64,
            ),
        );
        g.add_initializer(&b, vec![n_cls], vec![0.0; n_cls as usize]);
        g.add_node(
            &format!("head_{head}"),
            "Gemm",
            serde_json::json!({"alpha": 1.0, "beta": 1.0, "trans_a": false, "trans_b": true}),
            vec!["pooled_flat", &w, &b],
            vec![*out_name],
        );
    }
}

/// Build a BundleGraph for the LSTM predictor architecture.
///
/// Graph flow:
///   1. Input: (batch, seq_len, features)
///   2. Reshape + Gemm (input projection): features -> hidden
///   3. BatchNorm + Relu on projected
///   4. Reshape back to (B, S, H); Transpose to (S, B, H) for LSTM input
///   5. LSTM layers (chained): (S, 1, B, H) output per layer
///   6. Squeeze + select last timestep -> (B, H)
///   7. Three Gemm heads: hidden -> classes (imminent, warning, early)
fn build_lstm_graph(cfg: &SiteConfig, seed_base: u64) -> BundleGraph {
    let mut g = BundleGraph::new();
    let f = cfg.num_features as i64;
    let h = cfg.hidden_size as i64;
    let n_cls = cfg.num_classes as i64;
    let seq = SEQ_LEN as i64;

    // --- Graph I/O ---
    g.add_input("input", vec![-1, seq, f]);
    g.add_output("imminent_logits", vec![-1, n_cls]);
    g.add_output("warning_logits", vec![-1, n_cls]);
    g.add_output("early_logits", vec![-1, n_cls]);

    // --- 1-3. Input projection subgraph -> (B, S, H) ---
    build_input_proj_subgraph(&mut g, f, h, seq, seed_base);

    // --- 4. Transpose (B, S, H) -> (S, B, H) for LSTM ---
    g.add_node(
        "transpose_to_sbh",
        "Transpose",
        serde_json::json!({"perm": [1, 0, 2]}),
        vec!["proj_3d"],
        vec!["lstm_input"],
    );

    // --- 5. LSTM layers (chained) ---
    // ONNX LSTM inputs: X (S, B, input_size), W (num_dir, 4*H, input_size),
    //                   R (num_dir, 4*H, H), B_lstm (num_dir, 8*H)
    let mut prev_lstm_output = "lstm_input".to_string();
    let mut prev_input_dim = h;

    for layer in 0..cfg.num_layers {
        let lstm_w = format!("lstm_{layer}.W");
        let lstm_r = format!("lstm_{layer}.R");
        let lstm_b = format!("lstm_{layer}.B");
        let lstm_out_y = format!("lstm_{layer}_Y");
        let lstm_out_yh = format!("lstm_{layer}_Y_h");
        let lstm_out_yc = format!("lstm_{layer}_Y_c");

        // W: (1, 4*H, input_dim)  R: (1, 4*H, H)  B: (1, 8*H)
        let w_size = (4 * h * prev_input_dim) as usize;
        let r_size = (4 * h * h) as usize;
        let b_size = (8 * h) as usize;

        g.add_initializer(
            &lstm_w,
            vec![1, 4 * h, prev_input_dim],
            init_kaiming(
                w_size,
                prev_input_dim as usize,
                seed_base + 10 + layer as u64 * 3,
            ),
        );
        g.add_initializer(
            &lstm_r,
            vec![1, 4 * h, h],
            init_kaiming(r_size, h as usize, seed_base + 11 + layer as u64 * 3),
        );
        g.add_initializer(&lstm_b, vec![1, 8 * h], vec![0.0; b_size]);

        let node_name = format!("lstm_layer_{layer}");
        g.add_node(
            &node_name,
            "LSTM",
            serde_json::json!({
                "hidden_size": h,
                "direction": "forward"
            }),
            vec![&prev_lstm_output, &lstm_w, &lstm_r, &lstm_b],
            vec![&lstm_out_y, &lstm_out_yh, &lstm_out_yc],
        );

        // Y shape: (S, 1, B, H) -> squeeze num_directions dim -> (S, B, H)
        let squeezed = format!("lstm_{layer}_squeezed");
        g.add_node(
            &format!("squeeze_dir_{layer}"),
            "Squeeze",
            serde_json::json!({"axes": [1]}),
            vec![&lstm_out_y],
            vec![&squeezed],
        );

        prev_lstm_output = squeezed;
        prev_input_dim = h;
    }

    // --- 6. Select last timestep: (S, B, H) -> transpose (B, S, H) -> select S-1 -> (B, H) ---
    // Transpose (S, B, H) -> (B, S, H) first so we can use Gather on axis 1
    g.add_node(
        "transpose_to_bsh",
        "Transpose",
        serde_json::json!({"perm": [1, 0, 2]}),
        vec![&prev_lstm_output],
        vec!["lstm_bsh"],
    );

    // Gather last timestep index
    let last_idx_name = "last_seq_idx";
    g.add_initializer(last_idx_name, vec![1], vec![(SEQ_LEN - 1) as f32]);
    g.add_node(
        "gather_last_timestep",
        "Gather",
        serde_json::json!({"axis": 1}),
        vec!["lstm_bsh", last_idx_name],
        vec!["pooled_flat"],
    );

    // --- 7. Classification heads ---
    build_classification_heads(&mut g, n_cls, h, seed_base);

    g
}

/// Build a BundleGraph for the GRU predictor architecture.
///
/// Graph flow:
///   1. Input: (batch, seq_len, features)
///   2. Reshape + Gemm (input projection): features -> hidden
///   3. BatchNorm + Relu on projected
///   4. Reshape back to (B, S, H); Transpose to (S, B, H) for GRU input
///   5. GRU layers (chained): (S, 1, B, H) output per layer
///   6. Squeeze + GlobalAvgPool -> (B, H)
///   7. Three Gemm heads: hidden -> classes (imminent, warning, early)
fn build_gru_graph(cfg: &SiteConfig, seed_base: u64) -> BundleGraph {
    let mut g = BundleGraph::new();
    let f = cfg.num_features as i64;
    let h = cfg.hidden_size as i64;
    let n_cls = cfg.num_classes as i64;
    let seq = SEQ_LEN as i64;

    // --- Graph I/O ---
    g.add_input("input", vec![-1, seq, f]);
    g.add_output("imminent_logits", vec![-1, n_cls]);
    g.add_output("warning_logits", vec![-1, n_cls]);
    g.add_output("early_logits", vec![-1, n_cls]);

    // --- 1-3. Input projection subgraph -> (B, S, H) ---
    build_input_proj_subgraph(&mut g, f, h, seq, seed_base);

    // --- 4. Transpose (B, S, H) -> (S, B, H) for GRU ---
    g.add_node(
        "transpose_to_sbh",
        "Transpose",
        serde_json::json!({"perm": [1, 0, 2]}),
        vec!["proj_3d"],
        vec!["gru_input"],
    );

    // --- 5. GRU layers (chained) ---
    // ONNX GRU inputs: X (S, B, input_size), W (num_dir, 3*H, input_size),
    //                   R (num_dir, 3*H, H), B_gru (num_dir, 6*H)
    let mut prev_gru_output = "gru_input".to_string();
    let mut prev_input_dim = h;

    for layer in 0..cfg.num_layers {
        let gru_w = format!("gru_{layer}.W");
        let gru_r = format!("gru_{layer}.R");
        let gru_b = format!("gru_{layer}.B");
        let gru_out_y = format!("gru_{layer}_Y");
        let gru_out_yh = format!("gru_{layer}_Y_h");

        // W: (1, 3*H, input_dim)  R: (1, 3*H, H)  B: (1, 6*H)
        let w_size = (3 * h * prev_input_dim) as usize;
        let r_size = (3 * h * h) as usize;
        let b_size = (6 * h) as usize;

        g.add_initializer(
            &gru_w,
            vec![1, 3 * h, prev_input_dim],
            init_kaiming(
                w_size,
                prev_input_dim as usize,
                seed_base + 10 + layer as u64 * 3,
            ),
        );
        g.add_initializer(
            &gru_r,
            vec![1, 3 * h, h],
            init_kaiming(r_size, h as usize, seed_base + 11 + layer as u64 * 3),
        );
        g.add_initializer(&gru_b, vec![1, 6 * h], vec![0.0; b_size]);

        let node_name = format!("gru_layer_{layer}");
        g.add_node(
            &node_name,
            "GRU",
            serde_json::json!({
                "hidden_size": h,
                "direction": "forward",
                "linear_before_reset": 1
            }),
            vec![&prev_gru_output, &gru_w, &gru_r, &gru_b],
            vec![&gru_out_y, &gru_out_yh],
        );

        // Y shape: (S, 1, B, H) -> squeeze num_directions dim -> (S, B, H)
        let squeezed = format!("gru_{layer}_squeezed");
        g.add_node(
            &format!("squeeze_dir_{layer}"),
            "Squeeze",
            serde_json::json!({"axes": [1]}),
            vec![&gru_out_y],
            vec![&squeezed],
        );

        prev_gru_output = squeezed;
        prev_input_dim = h;
    }

    // --- 6. Pool over time: mean of (S, B, H) along dim 0 -> (B, H) ---
    // Transpose (S, B, H) -> (B, H, S) for GlobalAvgPool
    let pool_transpose_out = "pool_transpose";
    g.add_node(
        "transpose_for_pool",
        "Transpose",
        serde_json::json!({"perm": [1, 2, 0]}),
        vec![&prev_gru_output],
        vec![pool_transpose_out],
    );
    // Reshape (B, H, S) -> (B, H, S, 1) then GlobalAvgPool -> (B, H, 1, 1)
    g.add_initializer(
        "pool_4d_shape",
        vec![4],
        vec![-1.0, h as f32, seq as f32, 1.0],
    );
    g.add_node(
        "reshape_to_4d",
        "Reshape",
        serde_json::Value::Null,
        vec![pool_transpose_out, "pool_4d_shape"],
        vec!["pool_4d"],
    );
    g.add_node(
        "global_avg_pool",
        "GlobalAvgPool",
        serde_json::Value::Null,
        vec!["pool_4d"],
        vec!["pooled"],
    );
    // Flatten (B, H, 1, 1) -> (B, H)
    g.add_node(
        "flatten_pool",
        "Flatten",
        serde_json::json!({"axis": 1}),
        vec!["pooled"],
        vec!["pooled_flat"],
    );

    // --- 7. Classification heads ---
    build_classification_heads(&mut g, n_cls, h, seed_base);

    g
}

// =============================================================================
// Per-site-per-arch runner
// =============================================================================

fn run_one(cfg: &SiteConfig, arch: Arch, out_dir: &PathBuf) {
    println!("--------------------------------------------------------------");
    println!(
        "site: {}  arch: {}  (features={}, classes={}, hidden={}, layers={})",
        cfg.slug,
        arch.as_str(),
        cfg.num_features,
        cfg.num_classes,
        cfg.hidden_size,
        cfg.num_layers
    );
    println!("--------------------------------------------------------------");

    // --- 1. Generate synthetic data ---
    println!("generating synthetic data...");
    let mut rng = DataGenerator::new(42 + cfg.num_features as u64);
    let n_samples = 20000;
    let (data, labels) = rng.generate_dataset(cfg.num_features, cfg.num_classes, n_samples);

    let stride = 8;
    let (x_data, y_imm, y_warn, y_early) =
        rng.make_sequences(&data, &labels, cfg.num_features, SEQ_LEN, stride);

    let n_seq = y_imm.len();
    println!("  {} sequences from {} raw samples", n_seq, n_samples);
    let _ = (&y_warn, &y_early); // reserved for future multi-horizon loss

    // --- 2. Build, count params, train ---
    let seed_base = 0x7A0_000 + cfg.num_features as u64 + arch.as_str().len() as u64 * 0x1000;

    let (n_params, graph, arch_type_str) = match arch {
        Arch::Lstm => {
            println!("building LSTM model...");
            let model = HvacLstmPredictor::new(cfg);
            let np: usize = model
                .parameters()
                .iter()
                .map(|p| p.variable().data().numel())
                .sum();
            println!("  parameters: {}", np);
            println!("training (30 epochs, batch=32)...");
            train_lstm(&model, cfg, &x_data, &y_imm, 100, 32);
            println!("building BundleGraph (LSTM)...");
            let g = build_lstm_graph(cfg, seed_base);
            (np, g, "lstm_multi_horizon")
        }
        Arch::Gru => {
            println!("building GRU model...");
            let model = HvacGruPredictor::new(cfg);
            let np: usize = model
                .parameters()
                .iter()
                .map(|p| p.variable().data().numel())
                .sum();
            println!("  parameters: {}", np);
            println!("training (30 epochs, batch=32)...");
            train_gru(&model, cfg, &x_data, &y_imm, 100, 32);
            println!("building BundleGraph (GRU)...");
            let g = build_gru_graph(cfg, seed_base);
            (np, g, "gru_multi_horizon")
        }
    };

    let total_params: usize = graph.initializers.values().map(|t| t.data.len()).sum();
    println!(
        "  graph: {} nodes, {} initializers ({} params)",
        graph.nodes.len(),
        graph.initializers.len(),
        total_params,
    );

    // --- 3. Save bundle ---
    let slug_file = cfg.slug.replace('-', "_");
    let bundle_path = out_dir.join(format!("{}_{}.axonml", slug_file, arch.as_str()));

    let bundle = ModelBundle::new(
        &format!("hvac_{}_predictor_{}", arch.as_str(), cfg.slug),
        cfg.num_features,
        Vec::new(),
    )
    .with_hyperparam("location_slug", cfg.slug)
    .with_hyperparam("architecture_type", arch_type_str)
    .with_hyperparam("seq_len", SEQ_LEN as i64)
    .with_hyperparam("num_features", cfg.num_features as i64)
    .with_hyperparam("hidden_size", cfg.hidden_size as i64)
    .with_hyperparam("num_layers", cfg.num_layers as i64)
    .with_hyperparam("num_classes", cfg.num_classes as i64)
    .with_hyperparam("dropout", DROPOUT as f64)
    .with_hyperparam("model_params", n_params as i64)
    .with_hyperparam(
        "note",
        format!(
            "{} multi-horizon predictor; total_graph_params={total_params}",
            arch.as_str().to_uppercase()
        ),
    )
    .with_graph(graph);

    save_bundle(&bundle, &bundle_path).expect("save_bundle failed");
    let file_size = std::fs::metadata(&bundle_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("  saved: {} ({} bytes)", bundle_path.display(), file_size);
    println!();
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("==============================================================");
    println!("  HVAC Site Models — Train + Export LSTM & GRU .axonml Bundles");
    println!("==============================================================");
    println!();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hvac_site_models <site|all> <arch|all>");
        eprintln!("       site: fcog-mechroom | taylor-greenhouse | peabody-mechroom | all");
        eprintln!("       arch: lstm | gru | all");
        eprintln!();
        eprintln!("known sites:");
        for s in SITES {
            eprintln!(
                "  {:28} features={:2} classes={:2} hidden={:3} layers={}",
                s.slug, s.num_features, s.num_classes, s.hidden_size, s.num_layers
            );
        }
        eprintln!();
        eprintln!("output directory: /opt/AxonML-Hailo/hailo8/edgemodels/bundles/");
        std::process::exit(2);
    }

    let target_site = &args[1];
    let target_arch = &args[2];

    let out_dir = PathBuf::from("/opt/AxonML-Hailo/hailo8/edgemodels/bundles");
    std::fs::create_dir_all(&out_dir).expect("mkdir -p out_dir");

    let sites_to_build: Vec<&SiteConfig> = if target_site == "all" {
        SITES.iter().collect()
    } else {
        SITES
            .iter()
            .filter(|s| s.slug == target_site.as_str())
            .collect()
    };

    if sites_to_build.is_empty() {
        eprintln!("no site matched '{target_site}'");
        eprintln!("known slugs:");
        for s in SITES {
            eprintln!("  {}", s.slug);
        }
        std::process::exit(1);
    }

    let archs_to_build: Vec<Arch> = match target_arch.as_str() {
        "lstm" => vec![Arch::Lstm],
        "gru" => vec![Arch::Gru],
        "all" => vec![Arch::Lstm, Arch::Gru],
        other => {
            eprintln!("unknown arch '{other}'; use lstm | gru | all");
            std::process::exit(1);
        }
    };

    let total = sites_to_build.len() * archs_to_build.len();
    println!(
        "building {} model(s): {} site(s) × {} arch(s)",
        total,
        sites_to_build.len(),
        archs_to_build.len()
    );
    println!("output dir: {}", out_dir.display());
    println!();

    let mut built = 0;
    for cfg in &sites_to_build {
        for &arch in &archs_to_build {
            run_one(cfg, arch, &out_dir);
            built += 1;
        }
    }

    println!("==============================================================");
    println!("  done — {} model(s) built", built);
    println!("==============================================================");
}
