//! Train Panoptes — Facility-Wide Anomaly Detection Training Driver
//!
//! Three-phase training pipeline for the Panoptes facility-wide anomaly
//! detection model on physics-informed synthetic data from the Heritage Pointe
//! of Warren BAS control logic. The crate-internal `WarrenSimulator` produces
//! `FacilitySnapshot`s for 59 equipment slots; this binary drives them through
//! `Panoptes::forward_snapshot` and `Panoptes::forward_temporal` while training
//! against equipment-level fault target vectors via `MSELoss`.
//!
//! Phases:
//! 1. **Normal-only** (`PHASE1_EPOCHS`): teach the model that healthy snapshots
//!    map to all-zero equipment scores, using `Adam` at `LR`.
//! 2. **Normal + faults** (`PHASE2_EPOCHS`): interleave normal snapshots with
//!    faulted ones whose `affected` indices set their target to 1.0
//!    (`PanoptesTrainingData::fault_target`). LR is halved for finer fitting.
//! 3. **Temporal** (`PHASE3_EPOCHS`): generate `TEMPORAL_WINDOW`-length
//!    sequences of snapshots (1 hour at 5-minute spacing) covering normal
//!    drift trajectories and fault sequences with mid-window onset, and train
//!    the temporal head via `forward_temporal`. LR scaled to 0.3x.
//!
//! Validation is run with a different random seed via `evaluate_normal`,
//! `evaluate_mixed`, and `evaluate_temporal_mixed`. After training, the script
//! prints per-sample facility scores plus affected-vs-unaffected score gaps and
//! the alert count produced by `PanoptesOutput::from_scores` against the
//! Warren `FacilityConfig` at threshold 0.3.
//!
//! Run with: `cargo run --release -p axonml-hvac --example train_panoptes`.
//!
//! # File
//! `crates/axonml-hvac/examples/train_panoptes.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use axonml_autograd::Variable;
use axonml_hvac::panoptes::*;
use axonml_hvac::panoptes_datagen::*;
use axonml_nn::MSELoss;
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::Tensor;
use std::time::Instant;

// =============================================================================
// Configuration
// =============================================================================

const NUM_EQUIPMENT: usize = 59;
const NORMAL_SAMPLES: usize = 2000;
const FAULT_SAMPLES: usize = 1000;
const PHASE1_EPOCHS: usize = 30;
const PHASE2_EPOCHS: usize = 20;
const PHASE3_EPOCHS: usize = 15;
const TEMPORAL_WINDOW: usize = 12; // 12 snapshots = 1 hour at 5-min intervals
const TEMPORAL_NORMAL_SEQS: usize = 100;
const TEMPORAL_FAULT_SEQS: usize = 80;
const BATCH_SIZE: usize = 16;
const LR: f32 = 1e-3;
const SEED: u64 = 42;

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     PANOPTES — Facility-Wide Anomaly Detection Training     ║");
    println!("║     Heritage Pointe of Warren (59 equipment)                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // -------------------------------------------------------------------------
    // Generate training data
    // -------------------------------------------------------------------------
    println!("[data] Generating physics-informed training data...");
    let t0 = Instant::now();

    let sim = WarrenSimulator::new(SEED);
    let normal_train = sim.generate_normal(NORMAL_SAMPLES);
    let fault_data = sim.generate_with_faults(FAULT_SAMPLES, 1.0);

    // Validation set (different seed)
    let val_sim = WarrenSimulator::new(SEED + 999);
    let normal_val = val_sim.generate_normal(200);
    let fault_val = val_sim.generate_with_faults(100, 1.0);

    println!("  Normal train: {} samples", normal_train.len());
    println!("  Fault train:  {} samples", fault_data.len());
    println!("  Normal val:   {} samples", normal_val.len());
    println!("  Fault val:    {} samples", fault_val.len());
    println!("  Generated in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    // -------------------------------------------------------------------------
    // Create model
    // -------------------------------------------------------------------------
    let model = Panoptes::new(NUM_EQUIPMENT);
    println!("[model] Panoptes created");
    println!("  Equipment slots: {NUM_EQUIPMENT}");
    println!("  Parameters: {}", model.num_parameters());
    println!("  Embed dim: {EMBED_DIM}");
    println!();

    let mse = MSELoss::new();

    // Zero target for normal operation
    let zero_target = Variable::new(
        Tensor::from_vec(vec![0.0; NUM_EQUIPMENT], &[1, NUM_EQUIPMENT]).unwrap(),
        false,
    );

    // -------------------------------------------------------------------------
    // Phase 1: Learn normal operation
    // -------------------------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════");
    println!(" PHASE 1: Learning Normal Operation ({PHASE1_EPOCHS} epochs)");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  {:>5}  {:>12}  {:>12}  {:>8}",
        "Epoch", "Train Loss", "Val Loss", "Time"
    );
    println!("  {:-<5}  {:-<12}  {:-<12}  {:-<8}", "", "", "", "");

    let params = model.parameters();
    let mut optimizer = Adam::new(params, LR);

    for epoch in 1..=PHASE1_EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0;

        // Train on normal data: target = all zeros
        for batch_start in (0..normal_train.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(normal_train.len());

            for i in batch_start..batch_end {
                optimizer.zero_grad();

                let (equip_scores, _) = model.forward_snapshot(&normal_train[i]);
                let loss = mse.compute(&equip_scores, &zero_target);
                let loss_val = loss.data().to_vec()[0];
                epoch_loss += loss_val;
                batch_count += 1;

                if loss.requires_grad() {
                    loss.backward();
                    optimizer.step();
                }
            }
        }

        // Validation
        let val_loss = evaluate_normal(&model, &normal_val, &mse, &zero_target);

        let avg_loss = epoch_loss / batch_count as f32;
        let elapsed = epoch_start.elapsed().as_secs_f32();

        println!(
            "  {:>5}  {:>12.6}  {:>12.6}  {:>6.1}s",
            epoch, avg_loss, val_loss, elapsed
        );
    }

    println!();

    // -------------------------------------------------------------------------
    // Phase 2: Learn fault signatures
    // -------------------------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════");
    println!(" PHASE 2: Learning Fault Signatures ({PHASE2_EPOCHS} epochs)");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  {:>5}  {:>12}  {:>12}  {:>12}  {:>8}",
        "Epoch", "Normal Loss", "Fault Loss", "Val Loss", "Time"
    );
    println!(
        "  {:-<5}  {:-<12}  {:-<12}  {:-<12}  {:-<8}",
        "", "", "", "", ""
    );

    // Reset optimizer with lower LR for phase 2
    let params = model.parameters();
    let mut optimizer = Adam::new(params, LR * 0.5);

    for epoch in 1..=PHASE2_EPOCHS {
        let epoch_start = Instant::now();
        let mut normal_loss_sum = 0.0f32;
        let mut fault_loss_sum = 0.0f32;
        let mut normal_count = 0;
        let mut fault_count = 0;

        // Interleave normal + fault samples
        let normal_per_epoch = NORMAL_SAMPLES / 2; // Use half of normal data
        let fault_per_epoch = fault_data.len();

        // Normal samples: target = zeros
        for i in 0..normal_per_epoch.min(normal_train.len()) {
            optimizer.zero_grad();
            let (equip_scores, _) = model.forward_snapshot(&normal_train[i]);
            let loss = mse.compute(&equip_scores, &zero_target);
            normal_loss_sum += loss.data().to_vec()[0];
            normal_count += 1;

            if loss.requires_grad() {
                loss.backward();
                optimizer.step();
            }
        }

        // Fault samples: target = 1.0 for affected equipment
        for i in 0..fault_per_epoch {
            let (ref snap, ref _fault, ref affected) = fault_data[i];

            let target_vec = PanoptesTrainingData::fault_target(NUM_EQUIPMENT, affected);
            let fault_target = Variable::new(
                Tensor::from_vec(target_vec, &[1, NUM_EQUIPMENT]).unwrap(),
                false,
            );

            optimizer.zero_grad();
            let (equip_scores, _) = model.forward_snapshot(snap);
            let loss = mse.compute(&equip_scores, &fault_target);
            fault_loss_sum += loss.data().to_vec()[0];
            fault_count += 1;

            if loss.requires_grad() {
                loss.backward();
                optimizer.step();
            }
        }

        // Validation
        let val_loss = evaluate_mixed(&model, &normal_val, &fault_val, &mse, &zero_target);

        let avg_normal = normal_loss_sum / normal_count.max(1) as f32;
        let avg_fault = fault_loss_sum / fault_count.max(1) as f32;
        let elapsed = epoch_start.elapsed().as_secs_f32();

        println!(
            "  {:>5}  {:>12.6}  {:>12.6}  {:>12.6}  {:>6.1}s",
            epoch, avg_normal, avg_fault, val_loss, elapsed
        );
    }

    println!();

    // -------------------------------------------------------------------------
    // Phase 3: Temporal training
    // -------------------------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════");
    println!(" PHASE 3: Temporal Training ({PHASE3_EPOCHS} epochs, window={TEMPORAL_WINDOW})");
    println!("═══════════════════════════════════════════════════════════════");

    // Generate temporal sequences
    println!("[data] Generating temporal sequences...");
    let t0 = Instant::now();

    // Normal temporal sequences: varied starting OAT, slow drift
    let mut normal_seqs: Vec<Vec<FacilitySnapshot>> = Vec::new();
    for i in 0..TEMPORAL_NORMAL_SEQS {
        let start_oat = -5.0 + (i as f32 / TEMPORAL_NORMAL_SEQS as f32) * 100.0;
        let drift = if start_oat < 50.0 { 0.2 } else { -0.1 }; // warming up or cooling down
        let seq_sim = WarrenSimulator::new(SEED + 5000 + i as u64);
        let seq = seq_sim.generate_temporal_sequence(TEMPORAL_WINDOW, start_oat, drift);
        normal_seqs.push(seq);
    }

    // Fault temporal sequences: fault injected mid-sequence
    let mut fault_seqs: Vec<(Vec<FacilitySnapshot>, usize, FaultType, Vec<usize>)> = Vec::new();
    for i in 0..TEMPORAL_FAULT_SEQS {
        let start_oat = -5.0 + (i as f32 / TEMPORAL_FAULT_SEQS as f32) * 100.0;
        let drift = 0.1;
        let seq_sim = WarrenSimulator::new(SEED + 8000 + i as u64);
        let seq_data =
            seq_sim.generate_temporal_with_fault(TEMPORAL_WINDOW, start_oat, drift, i as u64);
        fault_seqs.push(seq_data);
    }

    // Validation temporal sequences
    let mut val_normal_seqs: Vec<Vec<FacilitySnapshot>> = Vec::new();
    for i in 0..20 {
        let start_oat = 10.0 + (i as f32 / 20.0) * 80.0;
        let seq_sim = WarrenSimulator::new(SEED + 9000 + i as u64);
        let seq = seq_sim.generate_temporal_sequence(TEMPORAL_WINDOW, start_oat, 0.15);
        val_normal_seqs.push(seq);
    }

    let mut val_fault_seqs: Vec<(Vec<FacilitySnapshot>, usize, FaultType, Vec<usize>)> = Vec::new();
    for i in 0..20 {
        let start_oat = 10.0 + (i as f32 / 20.0) * 80.0;
        let seq_sim = WarrenSimulator::new(SEED + 9500 + i as u64);
        let seq_data =
            seq_sim.generate_temporal_with_fault(TEMPORAL_WINDOW, start_oat, 0.1, i as u64);
        val_fault_seqs.push(seq_data);
    }

    println!("  Normal temporal seqs: {}", normal_seqs.len());
    println!("  Fault temporal seqs:  {}", fault_seqs.len());
    println!("  Val normal seqs:      {}", val_normal_seqs.len());
    println!("  Val fault seqs:       {}", val_fault_seqs.len());
    println!("  Window size: {TEMPORAL_WINDOW} snapshots (1 hour)");
    println!("  Generated in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    println!(
        "  {:>5}  {:>12}  {:>12}  {:>12}  {:>8}",
        "Epoch", "Normal Loss", "Fault Loss", "Val Loss", "Time"
    );
    println!(
        "  {:-<5}  {:-<12}  {:-<12}  {:-<12}  {:-<8}",
        "", "", "", "", ""
    );

    // Lower LR for temporal fine-tuning
    let params = model.parameters();
    let mut optimizer = Adam::new(params, LR * 0.3);

    for epoch in 1..=PHASE3_EPOCHS {
        let epoch_start = Instant::now();
        let mut normal_loss_sum = 0.0f32;
        let mut fault_loss_sum = 0.0f32;
        let mut normal_count = 0;
        let mut fault_count = 0;

        // Normal temporal sequences: target = all zeros
        for seq in &normal_seqs {
            optimizer.zero_grad();
            let (equip_scores, _) = model.forward_temporal(seq);
            let loss = mse.compute(&equip_scores, &zero_target);
            normal_loss_sum += loss.data().to_vec()[0];
            normal_count += 1;

            if loss.requires_grad() {
                loss.backward();
                optimizer.step();
            }
        }

        // Fault temporal sequences: target = 1.0 for affected equipment
        for (seq, _onset, _fault, affected) in &fault_seqs {
            let target_vec = PanoptesTrainingData::fault_target(NUM_EQUIPMENT, affected);
            let fault_target = Variable::new(
                Tensor::from_vec(target_vec, &[1, NUM_EQUIPMENT]).unwrap(),
                false,
            );

            optimizer.zero_grad();
            let (equip_scores, _) = model.forward_temporal(seq);
            let loss = mse.compute(&equip_scores, &fault_target);
            fault_loss_sum += loss.data().to_vec()[0];
            fault_count += 1;

            if loss.requires_grad() {
                loss.backward();
                optimizer.step();
            }
        }

        // Validation
        let val_loss = evaluate_temporal_mixed(
            &model,
            &val_normal_seqs,
            &val_fault_seqs,
            &mse,
            &zero_target,
        );

        let avg_normal = normal_loss_sum / normal_count.max(1) as f32;
        let avg_fault = fault_loss_sum / fault_count.max(1) as f32;
        let elapsed = epoch_start.elapsed().as_secs_f32();

        println!(
            "  {:>5}  {:>12.6}  {:>12.6}  {:>12.6}  {:>6.1}s",
            epoch, avg_normal, avg_fault, val_loss, elapsed
        );
    }

    println!();

    // -------------------------------------------------------------------------
    // Final evaluation
    // -------------------------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════");
    println!(" FINAL EVALUATION");
    println!("═══════════════════════════════════════════════════════════════");

    // Test on normal data — scores should be near zero
    let config = FacilityConfig::warren();
    println!("\n  Normal operation (should be low scores):");
    for i in [0, 50, 100, 150] {
        if i >= normal_val.len() {
            break;
        }
        let (equip_scores, fac_score) = model.forward_snapshot(&normal_val[i]);
        let scores = equip_scores.data().to_vec();
        let fac = fac_score.data().to_vec()[0];
        let max_score = scores.iter().cloned().fold(0.0f32, f32::max);
        let avg_score: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
        println!(
            "    Sample {i:>3}: facility={fac:.4}, avg_equip={avg_score:.4}, max_equip={max_score:.4}"
        );
    }

    // Test on fault data — affected equipment should have higher scores
    println!("\n  Fault samples (affected equipment should score higher):");
    for i in 0..5.min(fault_val.len()) {
        let (ref snap, ref fault, ref affected) = fault_val[i];
        let (equip_scores, fac_score) = model.forward_snapshot(snap);
        let scores = equip_scores.data().to_vec();
        let fac = fac_score.data().to_vec()[0];

        let output = PanoptesOutput::from_scores(&scores, fac, &config, 0.3);

        // Get scores for affected vs unaffected
        let affected_avg: f32 = if !affected.is_empty() {
            affected
                .iter()
                .filter(|&&s| s < scores.len())
                .map(|&s| scores[s])
                .sum::<f32>()
                / affected.len() as f32
        } else {
            0.0
        };

        println!("    Fault {:?}:", fault);
        println!(
            "      facility={fac:.4}, affected_avg={affected_avg:.4}, alerts={}",
            output.alerts.len()
        );
    }

    // Temporal evaluation
    println!("\n  Temporal normal (should be low scores):");
    for i in 0..3.min(val_normal_seqs.len()) {
        let (equip_scores, fac_score) = model.forward_temporal(&val_normal_seqs[i]);
        let scores = equip_scores.data().to_vec();
        let fac = fac_score.data().to_vec()[0];
        let max_score = scores.iter().cloned().fold(0.0f32, f32::max);
        let avg_score: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
        println!(
            "    Seq {i:>3}: facility={fac:.4}, avg_equip={avg_score:.4}, max_equip={max_score:.4}"
        );
    }

    println!("\n  Temporal fault (fault injected mid-sequence):");
    for i in 0..5.min(val_fault_seqs.len()) {
        let (ref seq, onset, ref fault, ref affected) = val_fault_seqs[i];
        let (equip_scores, fac_score) = model.forward_temporal(seq);
        let scores = equip_scores.data().to_vec();
        let fac = fac_score.data().to_vec()[0];

        let affected_avg: f32 = if !affected.is_empty() {
            affected
                .iter()
                .filter(|&&s| s < scores.len())
                .map(|&s| scores[s])
                .sum::<f32>()
                / affected.len() as f32
        } else {
            0.0
        };
        let unaffected_avg: f32 = {
            let unaffected: Vec<f32> = scores
                .iter()
                .enumerate()
                .filter(|(idx, _)| !affected.contains(idx))
                .map(|(_, &s)| s)
                .collect();
            if unaffected.is_empty() {
                0.0
            } else {
                unaffected.iter().sum::<f32>() / unaffected.len() as f32
            }
        };

        let output = PanoptesOutput::from_scores(&scores, fac, &config, 0.3);
        println!(
            "    Fault {:?} (onset step {onset}/{TEMPORAL_WINDOW}):",
            fault
        );
        println!(
            "      facility={fac:.4}, affected={affected_avg:.4}, unaffected={unaffected_avg:.4}, alerts={}",
            output.alerts.len()
        );
    }

    println!();
    println!("Training complete.");
}

// =============================================================================
// Evaluation helpers
// =============================================================================

fn evaluate_normal(
    model: &Panoptes,
    val_data: &[FacilitySnapshot],
    mse: &MSELoss,
    zero_target: &Variable,
) -> f32 {
    let mut total_loss = 0.0f32;
    for snap in val_data {
        let (equip_scores, _) = model.forward_snapshot(snap);
        let loss = mse.compute(&equip_scores, zero_target);
        total_loss += loss.data().to_vec()[0];
    }
    total_loss / val_data.len() as f32
}

fn evaluate_mixed(
    model: &Panoptes,
    normal_val: &[FacilitySnapshot],
    fault_val: &[(FacilitySnapshot, FaultType, Vec<usize>)],
    mse: &MSELoss,
    zero_target: &Variable,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut count = 0;

    for snap in normal_val {
        let (equip_scores, _) = model.forward_snapshot(snap);
        let loss = mse.compute(&equip_scores, zero_target);
        total_loss += loss.data().to_vec()[0];
        count += 1;
    }

    for (snap, _, affected) in fault_val {
        let target_vec = PanoptesTrainingData::fault_target(NUM_EQUIPMENT, affected);
        let fault_target = Variable::new(
            Tensor::from_vec(target_vec, &[1, NUM_EQUIPMENT]).unwrap(),
            false,
        );
        let (equip_scores, _) = model.forward_snapshot(snap);
        let loss = mse.compute(&equip_scores, &fault_target);
        total_loss += loss.data().to_vec()[0];
        count += 1;
    }

    total_loss / count as f32
}

fn evaluate_temporal_mixed(
    model: &Panoptes,
    normal_seqs: &[Vec<FacilitySnapshot>],
    fault_seqs: &[(Vec<FacilitySnapshot>, usize, FaultType, Vec<usize>)],
    mse: &MSELoss,
    zero_target: &Variable,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut count = 0;

    for seq in normal_seqs {
        let (equip_scores, _) = model.forward_temporal(seq);
        let loss = mse.compute(&equip_scores, zero_target);
        total_loss += loss.data().to_vec()[0];
        count += 1;
    }

    for (seq, _, _, affected) in fault_seqs {
        let target_vec = PanoptesTrainingData::fault_target(NUM_EQUIPMENT, affected);
        let fault_target = Variable::new(
            Tensor::from_vec(target_vec, &[1, NUM_EQUIPMENT]).unwrap(),
            false,
        );
        let (equip_scores, _) = model.forward_temporal(seq);
        let loss = mse.compute(&equip_scores, &fault_target);
        total_loss += loss.data().to_vec()[0];
        count += 1;
    }

    total_loss / count as f32
}
