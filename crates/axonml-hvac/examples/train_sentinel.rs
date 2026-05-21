//! Train Sentinel — Per-Equipment-Type Anomaly Detection + Predictive Failure
//!
//! Trains one Sentinel model per HVAC equipment type on synthetic data.
//! Exports each trained model to .axonml bundle + ONNX for NexusFoundry compilation.
//!
//! Usage: cargo run --release -p axonml-hvac --example train_sentinel [--type boiler]
//!
//! Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use axonml_autograd::Variable;
use axonml_hvac::sentinel::*;
use axonml_hvac::sentinel_datagen::*;
use axonml_nn::{Module, MSELoss};
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::Tensor;
use axonml_serialize::{ModelBundle, BundleGraph, GraphNode, NamedTensor, save_bundle};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;
use std::time::Instant;

const NORMAL_SAMPLES: usize = 16000;
const FAULT_SAMPLES_PER_TYPE: usize = 3000;
const R1_EPOCHS: usize = 80;
const R2_EPOCHS: usize = 60;
const BATCH_SIZE: usize = 64;
const BASE_R1_LR: f32 = 2e-3;
const BASE_R2_LR: f32 = 2e-4;
const VAL_SPLIT: f32 = 0.15;

fn evaluate(model: &Sentinel, samples: &[SentinelSample], train_count: usize, n_feat: usize, loss_fn: &MSELoss) -> (f32, f32) {
    let mut val_loss = 0.0f32;
    let mut val_batches = 0;
    let mut correct = 0usize;
    let mut total = 0usize;

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
        let target = Variable::new(Tensor::from_vec(target_data, &[bs, NUM_OUTPUTS]).unwrap(), false);
        let output = model.forward(&input);
        let loss = loss_fn.compute(&output, &target);
        val_loss += loss.data().to_vec()[0];
        val_batches += 1;

        let out_vec = output.data().to_vec();
        let tgt_vec = target.data().to_vec();
        for i in 0..bs {
            let pred = out_vec[i * NUM_OUTPUTS] > 0.5;
            let actual = tgt_vec[i * NUM_OUTPUTS] > 0.5;
            if pred == actual { correct += 1; }
            total += 1;
        }
    }
    (val_loss / val_batches.max(1) as f32, 100.0 * correct as f32 / total.max(1) as f32)
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

    if types.is_empty() {
        eprintln!("Unknown equipment type. Available: boiler, chiller, pump, ahu, zone_reheat, heat_pump");
        return;
    }

    for eq_type in &types {
        train_one(*eq_type);
    }
}

fn train_one(eq_type: EquipmentType) {
    let start = Instant::now();
    eprintln!("\n==============================================================");
    eprintln!("  SENTINEL — {} ({} sensors)", eq_type.name().to_uppercase(), eq_type.sensor_count());
    eprintln!("  Anomaly Detection + Predictive Failure (30m/4h/12h/3d/7d/14d)");
    eprintln!("==============================================================");

    // Generate data and shuffle for balanced train/val split
    let datagen = SentinelDatagen::new(eq_type, 42);
    let mut samples = datagen.generate(NORMAL_SAMPLES, FAULT_SAMPLES_PER_TYPE);
    let n_feat = eq_type.sensor_count();

    eprintln!("  Samples: {} ({} normal + {} faulted)", samples.len(),
        samples.iter().filter(|s| s.fault_name == "normal").count(),
        samples.iter().filter(|s| s.fault_name != "normal").count());

    let mut shuffle_rng = rand::rngs::StdRng::seed_from_u64(42 + eq_type.sensor_count() as u64);
    samples.shuffle(&mut shuffle_rng);

    // Split train/val
    let val_count = (samples.len() as f32 * VAL_SPLIT) as usize;
    let train_count = samples.len() - val_count;

    // Create model
    let model = Sentinel::new(eq_type);
    eprintln!("  Parameters: {}", model.param_count());

    let loss_fn = MSELoss::new();

    // === ROUND 1: Initial training ===
    // Scale LR inversely with sensor count — bigger models need lower LR
    let lr_scale = (10.0 / n_feat as f32).min(1.5); // cap at 1.5x to prevent divergence on small models
    let r1_lr = BASE_R1_LR * lr_scale;
    let r2_lr = BASE_R2_LR * lr_scale;
    eprintln!("  --- Round 1: {} epochs, lr={:.5} (scaled for {} sensors) ---", R1_EPOCHS, r1_lr, n_feat);
    let params = model.parameters();
    let mut optimizer = Adam::new(params.clone(), r1_lr);
    let mut best_acc = 0.0f32;

    for epoch in 0..(R1_EPOCHS + R2_EPOCHS) {
        // Switch to round 2 LR
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

            let input = Variable::new(Tensor::from_vec(
                input_data, &[bs, TIMESTEPS, n_feat]).unwrap(), true);
            let target = Variable::new(Tensor::from_vec(
                target_data, &[bs, NUM_OUTPUTS]).unwrap(), false);

            let output = model.forward(&input);
            let loss = loss_fn.compute(&output, &target);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.data().to_vec()[0];
            batch_count += 1;
        }

        let avg_loss = epoch_loss / batch_count.max(1) as f32;

        if (epoch + 1) % 10 == 0 || epoch == 0 || epoch == R1_EPOCHS {
            let (avg_val, accuracy) = evaluate(&model, &samples, train_count, n_feat, &loss_fn);
            if accuracy > best_acc { best_acc = accuracy; }
            let round = if epoch < R1_EPOCHS { 1 } else { 2 };
            let epoch_in_round = if epoch < R1_EPOCHS { epoch + 1 } else { epoch - R1_EPOCHS + 1 };
            let total_in_round = if epoch < R1_EPOCHS { R1_EPOCHS } else { R2_EPOCHS };
            eprintln!("  R{} Epoch {:3}/{}: train={:.4} val={:.4} acc={:.1}%",
                round, epoch_in_round, total_in_round, avg_loss, avg_val, accuracy);
        }
    }

    let elapsed = start.elapsed();
    eprintln!("  Training: {:.1}s | Best accuracy: {:.1}%", elapsed.as_secs_f64(), best_acc);

    // Save model as bundle with graph for ONNX export
    let out_dir = format!("sentinel_output/{}", eq_type.name());
    std::fs::create_dir_all(&out_dir).ok();

    let mut graph = BundleGraph::new();
    graph.add_input("input", vec![1, TIMESTEPS as i64, n_feat as i64]);
    graph.add_output("output", vec![1, NUM_OUTPUTS as i64]);

    let mut add_init = |name: &str, shape: Vec<i64>, data: Vec<f32>| {
        graph.initializers.insert(name.into(), NamedTensor { shape, dtype: "f32".into(), data });
    };

    // Conv1d weights/biases: AxonML Conv1d stores as (out_ch, in_ch, kernel)
    add_init("conv1.weight", vec![64, n_feat as i64, 3], model.conv1.weight.data().to_vec());
    add_init("conv1.bias", vec![64], model.conv1.bias.as_ref().unwrap().data().to_vec());
    add_init("bn1.weight", vec![64], model.bn1.weight.data().to_vec());
    add_init("bn1.bias", vec![64], model.bn1.bias.data().to_vec());
    add_init("bn1.running_mean", vec![64], model.bn1.running_mean().to_vec());
    add_init("bn1.running_var", vec![64], model.bn1.running_var().to_vec());
    add_init("conv2.weight", vec![128, 64, 3], model.conv2.weight.data().to_vec());
    add_init("conv2.bias", vec![128], model.conv2.bias.as_ref().unwrap().data().to_vec());
    add_init("bn2.weight", vec![128], model.bn2.weight.data().to_vec());
    add_init("bn2.bias", vec![128], model.bn2.bias.data().to_vec());
    add_init("bn2.running_mean", vec![128], model.bn2.running_mean().to_vec());
    add_init("bn2.running_var", vec![128], model.bn2.running_var().to_vec());
    add_init("conv3.weight", vec![64, 128, 3], model.conv3.weight.data().to_vec());
    add_init("conv3.bias", vec![64], model.conv3.bias.as_ref().unwrap().data().to_vec());
    add_init("fc1.weight", vec![64, 128], model.fc1.weight.data().to_vec());
    add_init("fc1.bias", vec![64], model.fc1.bias.as_ref().unwrap().data().to_vec());
    add_init("fc2.weight", vec![NUM_OUTPUTS as i64, 64], model.fc2.weight.data().to_vec());
    add_init("fc2.bias", vec![NUM_OUTPUTS as i64], model.fc2.bias.as_ref().unwrap().data().to_vec());

    // Transpose: (B, T, C) → (B, C, T)
    graph.nodes.push(GraphNode { name: "transpose".into(), op: "Transpose".into(),
        attrs: serde_json::json!({"perm": [0, 2, 1]}), inputs: vec!["input".into()], outputs: vec!["t0".into()] });
    // Conv1d → Relu → BN × 3
    graph.nodes.push(GraphNode { name: "conv1".into(), op: "Conv".into(),
        attrs: serde_json::json!({"kernel_shape": [3], "strides": [1], "pads": [0, 0]}),
        inputs: vec!["t0".into(), "conv1.weight".into(), "conv1.bias".into()], outputs: vec!["c1".into()] });
    graph.nodes.push(GraphNode { name: "relu1".into(), op: "Relu".into(),
        attrs: serde_json::json!({}), inputs: vec!["c1".into()], outputs: vec!["r1".into()] });
    graph.nodes.push(GraphNode { name: "bn1".into(), op: "BatchNormalization".into(),
        attrs: serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        inputs: vec!["r1".into(), "bn1.weight".into(), "bn1.bias".into(), "bn1.running_mean".into(), "bn1.running_var".into()],
        outputs: vec!["b1".into()] });
    graph.nodes.push(GraphNode { name: "conv2".into(), op: "Conv".into(),
        attrs: serde_json::json!({"kernel_shape": [3], "strides": [1], "pads": [0, 0]}),
        inputs: vec!["b1".into(), "conv2.weight".into(), "conv2.bias".into()], outputs: vec!["c2".into()] });
    graph.nodes.push(GraphNode { name: "relu2".into(), op: "Relu".into(),
        attrs: serde_json::json!({}), inputs: vec!["c2".into()], outputs: vec!["r2".into()] });
    graph.nodes.push(GraphNode { name: "bn2".into(), op: "BatchNormalization".into(),
        attrs: serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        inputs: vec!["r2".into(), "bn2.weight".into(), "bn2.bias".into(), "bn2.running_mean".into(), "bn2.running_var".into()],
        outputs: vec!["b2".into()] });
    graph.nodes.push(GraphNode { name: "conv3".into(), op: "Conv".into(),
        attrs: serde_json::json!({"kernel_shape": [3], "strides": [1], "pads": [0, 0]}),
        inputs: vec!["b2".into(), "conv3.weight".into(), "conv3.bias".into()], outputs: vec!["c3".into()] });
    graph.nodes.push(GraphNode { name: "relu3".into(), op: "Relu".into(),
        attrs: serde_json::json!({}), inputs: vec!["c3".into()], outputs: vec!["r3".into()] });
    add_init("flatten_shape", vec![2], vec![1.0, 128.0]);
    graph.nodes.push(GraphNode { name: "flatten".into(), op: "Reshape".into(),
        attrs: serde_json::json!({}),
        inputs: vec!["r3".into(), "flatten_shape".into()], outputs: vec!["flat".into()] });
    // Dense layers (no dropout in inference)
    graph.nodes.push(GraphNode { name: "fc1".into(), op: "Gemm".into(),
        attrs: serde_json::json!({"transB": 1}),
        inputs: vec!["flat".into(), "fc1.weight".into(), "fc1.bias".into()], outputs: vec!["g1".into()] });
    graph.nodes.push(GraphNode { name: "relu4".into(), op: "Relu".into(),
        attrs: serde_json::json!({}), inputs: vec!["g1".into()], outputs: vec!["r4".into()] });
    graph.nodes.push(GraphNode { name: "fc2".into(), op: "Gemm".into(),
        attrs: serde_json::json!({"transB": 1}),
        inputs: vec!["r4".into(), "fc2.weight".into(), "fc2.bias".into()], outputs: vec!["g2".into()] });
    graph.nodes.push(GraphNode { name: "sigmoid".into(), op: "Sigmoid".into(),
        attrs: serde_json::json!({}), inputs: vec!["g2".into()], outputs: vec!["output".into()] });

    let mut hparams = HashMap::new();
    hparams.insert("timesteps".into(), serde_json::json!(TIMESTEPS));
    hparams.insert("num_outputs".into(), serde_json::json!(NUM_OUTPUTS));
    hparams.insert("equipment_type".into(), serde_json::json!(eq_type.name()));
    hparams.insert("sensor_count".into(), serde_json::json!(n_feat));
    hparams.insert("best_accuracy".into(), serde_json::json!(best_acc));

    let bundle = ModelBundle {
        architecture: format!("sentinel_{}", eq_type.name()),
        input_features: n_feat,
        hyperparameters: hparams,
        weights: Vec::new(),
        norm_means: Vec::new(),
        norm_stds: Vec::new(),
        anomaly_threshold: Some(0.5),
        graph: Some(graph),
    };

    let model_path = format!("{}/nehebkau_{}", out_dir, eq_type.name());
    match save_bundle(&bundle, &model_path) {
        Ok(p) => eprintln!("  Saved bundle: {}", p.display()),
        Err(e) => eprintln!("  Save failed: {}", e),
    }

    eprintln!("  Input: [1, {}, {}] → Output: [1, {}]", TIMESTEPS, n_feat, NUM_OUTPUTS);
    eprintln!("  Best accuracy: {:.1}%", best_acc);
    eprintln!("  Done: {}", eq_type.name());
}
