//! Repack existing nehebkau .axonml weight files into bundle format for ONNX export.
//! No retraining — just loads weights and re-saves with graph info.

use axonml_hvac::sentinel::*;
use axonml_nn::Module;
use axonml_serialize::{ModelBundle, BundleGraph, GraphNode, NamedTensor, save_bundle};
use std::collections::HashMap;

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
        repack(*eq_type);
    }
}

fn repack(eq_type: EquipmentType) {
    let n_feat = eq_type.sensor_count();
    let model = Sentinel::new(eq_type);

    let weight_path = format!("sentinel_output/{}/nehebkau_{}.axonml", eq_type.name(), eq_type.name());
    match axonml_serialize::load_model(&model, &weight_path) {
        Ok(n) => eprintln!("  Loaded {} params from {}", n, weight_path),
        Err(e) => { eprintln!("  SKIP {}: {}", eq_type.name(), e); return; }
    }

    let mut graph = BundleGraph::new();
    // Input already in NCHW (app layer transposes before sending to NPU)
    graph.add_input("input", vec![1, n_feat as i64, 1, TIMESTEPS as i64]);
    graph.add_output("output", vec![1, NUM_OUTPUTS as i64]);

    let mut add_init = |name: &str, shape: Vec<i64>, data: Vec<f32>| {
        graph.initializers.insert(name.into(), NamedTensor { shape, dtype: "f32".into(), data });
    };

    // Conv weights stored as 4D for DFC compatibility: (out, in, 1, k)
    add_init("conv1.weight", vec![64, n_feat as i64, 1, 3], model.conv1.weight.data().to_vec());
    add_init("conv1.bias", vec![64], model.conv1.bias.as_ref().unwrap().data().to_vec());
    add_init("bn1.weight", vec![64], model.bn1.weight.data().to_vec());
    add_init("bn1.bias", vec![64], model.bn1.bias.data().to_vec());
    add_init("bn1.running_mean", vec![64], model.bn1.running_mean().to_vec());
    add_init("bn1.running_var", vec![64], model.bn1.running_var().to_vec());
    add_init("conv2.weight", vec![128, 64, 1, 3], model.conv2.weight.data().to_vec());
    add_init("conv2.bias", vec![128], model.conv2.bias.as_ref().unwrap().data().to_vec());
    add_init("bn2.weight", vec![128], model.bn2.weight.data().to_vec());
    add_init("bn2.bias", vec![128], model.bn2.bias.data().to_vec());
    add_init("bn2.running_mean", vec![128], model.bn2.running_mean().to_vec());
    add_init("bn2.running_var", vec![128], model.bn2.running_var().to_vec());
    add_init("conv3.weight", vec![64, 128, 1, 3], model.conv3.weight.data().to_vec());
    add_init("conv3.bias", vec![64], model.conv3.bias.as_ref().unwrap().data().to_vec());
    // Gemm weights stored as (in, out) for transB=0 — DFC reads dim[1] as out_features
    let fc1_w = model.fc1.weight.data().to_vec(); // [64, 128] row-major
    let mut fc1_t = vec![0.0f32; 128 * 64];
    for r in 0..64 { for c in 0..128 { fc1_t[c * 64 + r] = fc1_w[r * 128 + c]; } }
    add_init("fc1.weight", vec![128, 64], fc1_t);
    add_init("fc1.bias", vec![64], model.fc1.bias.as_ref().unwrap().data().to_vec());
    let fc2_w = model.fc2.weight.data().to_vec(); // [7, 64] row-major
    let mut fc2_t = vec![0.0f32; 64 * NUM_OUTPUTS];
    for r in 0..NUM_OUTPUTS { for c in 0..64 { fc2_t[c * NUM_OUTPUTS + r] = fc2_w[r * 64 + c]; } }
    add_init("fc2.weight", vec![64, NUM_OUTPUTS as i64], fc2_t);
    add_init("fc2.bias", vec![NUM_OUTPUTS as i64], model.fc2.bias.as_ref().unwrap().data().to_vec());

    // Input is (1, n_feat, 1, 8) — already NCHW, Conv2d (1,3) kernels
    graph.nodes.push(GraphNode { name: "conv1".into(), op: "Conv".into(),
        attrs: serde_json::json!({"kernel_shape": [1, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}),
        inputs: vec!["input".into(), "conv1.weight".into(), "conv1.bias".into()], outputs: vec!["c1".into()] });
    graph.nodes.push(GraphNode { name: "relu1".into(), op: "Relu".into(),
        attrs: serde_json::json!({}), inputs: vec!["c1".into()], outputs: vec!["r1".into()] });
    graph.nodes.push(GraphNode { name: "bn1".into(), op: "BatchNormalization".into(),
        attrs: serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        inputs: vec!["r1".into(), "bn1.weight".into(), "bn1.bias".into(), "bn1.running_mean".into(), "bn1.running_var".into()],
        outputs: vec!["b1".into()] });
    graph.nodes.push(GraphNode { name: "conv2".into(), op: "Conv".into(),
        attrs: serde_json::json!({"kernel_shape": [1, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}),
        inputs: vec!["b1".into(), "conv2.weight".into(), "conv2.bias".into()], outputs: vec!["c2".into()] });
    graph.nodes.push(GraphNode { name: "relu2".into(), op: "Relu".into(),
        attrs: serde_json::json!({}), inputs: vec!["c2".into()], outputs: vec!["r2".into()] });
    graph.nodes.push(GraphNode { name: "bn2".into(), op: "BatchNormalization".into(),
        attrs: serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        inputs: vec!["r2".into(), "bn2.weight".into(), "bn2.bias".into(), "bn2.running_mean".into(), "bn2.running_var".into()],
        outputs: vec!["b2".into()] });
    graph.nodes.push(GraphNode { name: "conv3".into(), op: "Conv".into(),
        attrs: serde_json::json!({"kernel_shape": [1, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]}),
        inputs: vec!["b2".into(), "conv3.weight".into(), "conv3.bias".into()], outputs: vec!["c3".into()] });
    graph.nodes.push(GraphNode { name: "relu3".into(), op: "Relu".into(),
        attrs: serde_json::json!({}), inputs: vec!["c3".into()], outputs: vec!["r3".into()] });
    // Flatten (1, 64, 1, 2) → (1, 128)
    add_init("flatten_shape", vec![2], vec![1.0, 128.0]);
    graph.nodes.push(GraphNode { name: "flatten".into(), op: "Reshape".into(),
        attrs: serde_json::json!({}),
        inputs: vec!["r3".into(), "flatten_shape".into()], outputs: vec!["flat".into()] });
    graph.nodes.push(GraphNode { name: "fc1".into(), op: "Gemm".into(),
        attrs: serde_json::json!({}),
        inputs: vec!["flat".into(), "fc1.weight".into(), "fc1.bias".into()], outputs: vec!["g1".into()] });
    graph.nodes.push(GraphNode { name: "relu4".into(), op: "Relu".into(),
        attrs: serde_json::json!({}), inputs: vec!["g1".into()], outputs: vec!["r4".into()] });
    graph.nodes.push(GraphNode { name: "fc2".into(), op: "Gemm".into(),
        attrs: serde_json::json!({}),
        inputs: vec!["r4".into(), "fc2.weight".into(), "fc2.bias".into()], outputs: vec!["g2".into()] });
    graph.nodes.push(GraphNode { name: "sigmoid".into(), op: "Sigmoid".into(),
        attrs: serde_json::json!({}), inputs: vec!["g2".into()], outputs: vec!["output".into()] });

    let mut hparams = HashMap::new();
    hparams.insert("timesteps".into(), serde_json::json!(TIMESTEPS));
    hparams.insert("num_outputs".into(), serde_json::json!(NUM_OUTPUTS));
    hparams.insert("equipment_type".into(), serde_json::json!(eq_type.name()));
    hparams.insert("sensor_count".into(), serde_json::json!(n_feat));

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

    let out_path = format!("sentinel_output/{}/nehebkau_{}_bundle", eq_type.name(), eq_type.name());
    match save_bundle(&bundle, &out_path) {
        Ok(p) => eprintln!("  Bundle: {} → {}", eq_type.name(), p.display()),
        Err(e) => eprintln!("  FAILED {}: {}", eq_type.name(), e),
    }
}
