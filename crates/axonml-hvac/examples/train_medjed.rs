//! Train Medjed — Per-Equipment-Type Predictive Failure Detector
//!
//! Same Sentinel architecture as Nehebkau but trained with loss weighted
//! toward the 6 predictive failure horizons (30min/4hr/12hr/3d/7d/14d)
//! rather than the binary anomaly flag.
//!
//! Usage: cargo run --release -p axonml-hvac --example train_medjed [--type boiler]
//!
//! Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use axonml_autograd::Variable;
use axonml_hvac::sentinel::*;
use axonml_hvac::sentinel_datagen::*;
use axonml_nn::{Module, MSELoss};
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::Tensor;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::time::Instant;

const NORMAL_SAMPLES: usize = 16000;
const FAULT_SAMPLES_PER_TYPE: usize = 3000;
const R1_EPOCHS: usize = 80;
const R2_EPOCHS: usize = 60;
const BATCH_SIZE: usize = 64;
const BASE_R1_LR: f32 = 2e-3;
const BASE_R2_LR: f32 = 2e-4;
const VAL_SPLIT: f32 = 0.15;

// Medjed weights prediction horizons 5x more than anomaly flag
const ANOMALY_WEIGHT: f32 = 0.2;
const HORIZON_WEIGHT: f32 = 1.0;

fn weighted_mse(output: &Variable, target: &Variable) -> Variable {
    let diff = output.sub_var(target);
    let sq = diff.mul_var(&diff);
    let bs = output.shape()[0];
    let mut w_data = Vec::with_capacity(bs * NUM_OUTPUTS);
    for _ in 0..bs {
        w_data.push(ANOMALY_WEIGHT);
        for _ in 1..NUM_OUTPUTS {
            w_data.push(HORIZON_WEIGHT);
        }
    }
    let weights = Variable::new(Tensor::from_vec(w_data, &[bs, NUM_OUTPUTS]).unwrap(), false);
    sq.mul_var(&weights).mean()
}

fn evaluate(model: &Sentinel, samples: &[SentinelSample], train_count: usize, n_feat: usize) -> (f32, f32, f32) {
    let mut val_loss = 0.0f32;
    let mut val_batches = 0;
    let mut correct_anomaly = 0usize;
    let mut total_anomaly = 0usize;
    let mut horizon_mae_sum = 0.0f32;
    let mut horizon_count = 0usize;

    for batch_start in (train_count..samples.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(samples.len());
        let bs = batch_end - batch_start;
        if bs == 0 { continue; }

        let mut input_data = Vec::with_capacity(bs * TIMESTEPS * n_feat);
        let mut target_data = Vec::with_capacity(bs * NUM_OUTPUTS);
        for i in batch_start..batch_end {
            input_data.extend_from_slice(&samples[i].data);
            target_data.extend_from_slice(&samples[i].labels);
        }

        let input = Variable::new(Tensor::from_vec(input_data, &[bs, TIMESTEPS, n_feat]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(target_data.clone(), &[bs, NUM_OUTPUTS]).unwrap(), false);
        let output = model.forward(&input);

        let loss = weighted_mse(&output, &target);
        val_loss += loss.data().to_vec()[0];
        val_batches += 1;

        let out_vec = output.data().to_vec();
        for i in 0..bs {
            let pred_anom = out_vec[i * NUM_OUTPUTS] > 0.5;
            let actual_anom = target_data[i * NUM_OUTPUTS] > 0.5;
            if pred_anom == actual_anom { correct_anomaly += 1; }
            total_anomaly += 1;
            for h in 1..NUM_OUTPUTS {
                horizon_mae_sum += (out_vec[i * NUM_OUTPUTS + h] - target_data[i * NUM_OUTPUTS + h]).abs();
                horizon_count += 1;
            }
        }
    }

    let avg_val = val_loss / val_batches.max(1) as f32;
    let accuracy = 100.0 * correct_anomaly as f32 / total_anomaly.max(1) as f32;
    let horizon_mae = horizon_mae_sum / horizon_count.max(1) as f32;
    (avg_val, accuracy, horizon_mae)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let filter_type = args.iter().position(|a| a == "--type")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let types: Vec<EquipmentType> = if let Some(t) = filter_type {
        EquipmentType::all().iter().filter(|e| e.name() == t).copied().collect()
    } else {
        EquipmentType::all().to_vec()
    };

    for eq_type in &types {
        train_one(*eq_type);
    }
}

fn train_one(eq_type: EquipmentType) {
    let start = Instant::now();
    let n_feat = eq_type.sensor_count();
    eprintln!("\n==============================================================");
    eprintln!("  MEDJED — {} ({} sensors)", eq_type.name().to_uppercase(), n_feat);
    eprintln!("  Predictive Failure Detector (30m/4h/12h/3d/7d/14d)");
    eprintln!("==============================================================");

    let datagen = SentinelDatagen::new(eq_type, 77); // different seed than nehebkau
    let mut samples = datagen.generate(NORMAL_SAMPLES, FAULT_SAMPLES_PER_TYPE);
    let mut shuffle_rng = rand::rngs::StdRng::seed_from_u64(77 + eq_type.sensor_count() as u64);
    samples.shuffle(&mut shuffle_rng);
    let train_count = samples.len() - (samples.len() as f32 * VAL_SPLIT) as usize;

    let model = Sentinel::new(eq_type);
    eprintln!("  Samples: {} | Parameters: {}", samples.len(), model.param_count());

    let lr_scale = (10.0 / n_feat as f32).min(1.5);
    let r1_lr = BASE_R1_LR * lr_scale;
    let r2_lr = BASE_R2_LR * lr_scale;

    let params = model.parameters();
    let mut optimizer = Adam::new(params.clone(), r1_lr);
    let mut best_acc = 0.0f32;
    let mut best_mae = 1.0f32;

    eprintln!("  --- Round 1: {} epochs, lr={:.5} ---", R1_EPOCHS, r1_lr);

    for epoch in 0..(R1_EPOCHS + R2_EPOCHS) {
        if epoch == R1_EPOCHS {
            eprintln!("  --- Round 2: {} epochs, lr={:.5} ---", R2_EPOCHS, r2_lr);
            optimizer = Adam::new(params.clone(), r2_lr);
        }

        let mut epoch_loss = 0.0f32;
        let mut batch_count = 0;

        for batch_start in (0..train_count).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(train_count);
            let bs = batch_end - batch_start;
            if bs == 0 { continue; }

            let mut input_data = Vec::with_capacity(bs * TIMESTEPS * n_feat);
            let mut target_data = Vec::with_capacity(bs * NUM_OUTPUTS);
            for i in batch_start..batch_end {
                input_data.extend_from_slice(&samples[i].data);
                target_data.extend_from_slice(&samples[i].labels);
            }

            let input = Variable::new(Tensor::from_vec(input_data, &[bs, TIMESTEPS, n_feat]).unwrap(), true);
            let target = Variable::new(Tensor::from_vec(target_data, &[bs, NUM_OUTPUTS]).unwrap(), false);

            let output = model.forward(&input);
            let loss = weighted_mse(&output, &target);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.data().to_vec()[0];
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count.max(1) as f32;

        if (epoch + 1) % 10 == 0 || epoch == 0 || epoch == R1_EPOCHS {
            let (avg_val, accuracy, horizon_mae) = evaluate(&model, &samples, train_count, n_feat);
            if accuracy > best_acc { best_acc = accuracy; }
            if horizon_mae < best_mae { best_mae = horizon_mae; }
            let round = if epoch < R1_EPOCHS { 1 } else { 2 };
            let ep = if epoch < R1_EPOCHS { epoch + 1 } else { epoch - R1_EPOCHS + 1 };
            let total = if epoch < R1_EPOCHS { R1_EPOCHS } else { R2_EPOCHS };
            eprintln!("  R{} Epoch {:3}/{}: train={:.4} val={:.4} acc={:.1}% horizon_mae={:.3}",
                round, ep, total, avg_loss, avg_val, accuracy, horizon_mae);
        }
    }

    let elapsed = start.elapsed();
    eprintln!("  Training: {:.1}s | Best acc: {:.1}% | Best horizon MAE: {:.3}", elapsed.as_secs_f64(), best_acc, best_mae);

    let out_dir = format!("sentinel_output/{}", eq_type.name());
    std::fs::create_dir_all(&out_dir).ok();
    let model_path = format!("{}/medjed_{}.axonml", out_dir, eq_type.name());
    match axonml_serialize::save_model(&model, &model_path) {
        Ok(()) => eprintln!("  Saved: {}", model_path),
        Err(e) => eprintln!("  Save failed: {}", e),
    }
    eprintln!("  Done: {}", eq_type.name());
}
