//! Build a Mnemosyne v2 face-embedding bundle using only NexusFoundry-supported
//! IrOps (Conv2d / BatchNorm / Relu / GlobalAvgPool / Gemm).
//!
//! v1 used a custom temporal-crystallization GRU+CNN head — no graph-translatable
//! to Hailo silicon. v2 keeps just the encoder pipeline and emits a 128-d
//! L2-norm embedding (L2-norm itself is host-side post-NPU).
//!
//! Architecture: 4× (Conv2d-stride2 + BN + ReLU) → GlobalAvgPool → Linear(48→128)
//!
//!   [B, 3, 64, 64]
//!     conv1 (3→16, 3x3, s=2, p=1)  → BN → ReLU   [B, 16, 32, 32]
//!     conv2 (16→24, 3x3, s=2, p=1) → BN → ReLU   [B, 24, 16, 16]
//!     conv3 (24→32, 3x3, s=2, p=1) → BN → ReLU   [B, 32, 8, 8]
//!     conv4 (32→48, 3x3, s=2, p=1) → BN → ReLU   [B, 48, 4, 4]
//!   GlobalAvgPool                                [B, 48, 1, 1]
//!   Gemm  (48 → 128, transB=true)                [B, 128]
//!
//! Weights are random-init (this binary's purpose is pipeline validation —
//! shipping a Hailo-compilable graph through the NF→ONNX→DFC→HEF flow with
//! the right shape, layer count, and op coverage). For real face-recognition
//! accuracy, run training (not in this example) and replace the initializers
//! before save_bundle.
//!
//! Usage:
//!   cargo run --release --example mnemosyne_v2_bundle -p axonml-serialize -- \
//!       <output.axonml>

use std::path::PathBuf;

use axonml_serialize::{BundleGraph, ModelBundle, save_bundle};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/mnemosyne_v2.axonml".to_string());

    let mut graph = BundleGraph::new();

    // Input/output declarations
    graph.add_input("input", vec![-1, 3, 64, 64]);
    graph.add_output("embedding", vec![-1, 128]);

    // ----- Conv stack -----
    let convs = [
        ("conv1", 3, 16, "input", "conv1_out"),
        ("conv2", 16, 24, "relu1_out", "conv2_out"),
        ("conv3", 24, 32, "relu2_out", "conv3_out"),
        ("conv4", 32, 48, "relu3_out", "conv4_out"),
    ];

    for (i, (name, in_c, out_c, in_act, out_act)) in convs.iter().enumerate() {
        let bn_name = format!("bn{}", i + 1);
        let relu_name = format!("relu{}", i + 1);
        let bn_out = format!("bn{}_out", i + 1);
        let relu_out = format!("relu{}_out", i + 1);

        // Conv weight: [out_c, in_c, 3, 3]
        let cw_name = format!("{name}.weight");
        let cb_name = format!("{name}.bias");
        let cw_n = (out_c * in_c * 3 * 3) as usize;
        graph.add_initializer(
            &cw_name,
            vec![*out_c as i64, *in_c as i64, 3, 3],
            // small Kaiming-ish: uniform in [-k, k], k = sqrt(2 / fan_in)
            init_kaiming(cw_n, (in_c * 3 * 3) as usize, i as u64),
        );
        graph.add_initializer(&cb_name, vec![*out_c as i64], vec![0.0; *out_c as usize]);

        // BN params: weight=1, bias=0, running_mean=0, running_var=1
        let bn_w = format!("{bn_name}.weight");
        let bn_b = format!("{bn_name}.bias");
        let bn_m = format!("{bn_name}.running_mean");
        let bn_v = format!("{bn_name}.running_var");
        graph.add_initializer(&bn_w, vec![*out_c as i64], vec![1.0; *out_c as usize]);
        graph.add_initializer(&bn_b, vec![*out_c as i64], vec![0.0; *out_c as usize]);
        graph.add_initializer(&bn_m, vec![*out_c as i64], vec![0.0; *out_c as usize]);
        graph.add_initializer(&bn_v, vec![*out_c as i64], vec![1.0; *out_c as usize]);

        graph.add_node(
            name,
            "Conv2d",
            serde_json::json!({
                "kernel_shape": [3, 3],
                "strides": [2, 2],
                "pads": [1, 1, 1, 1],
                "dilations": [1, 1],
                "group": 1,
            }),
            vec![in_act, &cw_name, &cb_name],
            vec![out_act],
        );
        graph.add_node(
            &bn_name,
            "BatchNorm",
            serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
            vec![out_act, &bn_w, &bn_b, &bn_m, &bn_v],
            vec![&bn_out],
        );
        graph.add_node(
            &relu_name,
            "Relu",
            serde_json::Value::Null,
            vec![&bn_out],
            vec![&relu_out],
        );
    }

    // ----- GlobalAvgPool: [B, 48, 4, 4] → [B, 48, 1, 1] -----
    graph.add_node(
        "gap",
        "GlobalAvgPool",
        serde_json::Value::Null,
        vec!["relu4_out"],
        vec!["pooled"],
    );

    // ----- Flatten [B, 48, 1, 1] → [B, 48], then Gemm → [B, 128] -----
    // DFC parser rejects Gemm on rank-4 input; Flatten(axis=1) is the canonical bridge.
    graph.add_node(
        "flatten",
        "Flatten",
        serde_json::json!({"axis": 1}),
        vec!["pooled"],
        vec!["pooled_flat"],
    );
    graph.add_initializer("fc.weight", vec![128, 48], init_kaiming(128 * 48, 48, 99));
    graph.add_initializer("fc.bias", vec![128], vec![0.0; 128]);
    graph.add_node(
        "fc",
        "Gemm",
        serde_json::json!({"alpha": 1.0, "beta": 1.0, "trans_a": false, "trans_b": true}),
        vec!["pooled_flat", "fc.weight", "fc.bias"],
        vec!["embedding"],
    );

    let bundle = ModelBundle::new("mnemosyne_v2", 3, Vec::new())
        .with_hyperparam("input_h", 64)
        .with_hyperparam("input_w", 64)
        .with_hyperparam("embedding_dim", 128)
        .with_hyperparam(
            "notes",
            "v2 face encoder — drops v1's GRU temporal head; \
                                    NF-only ops; L2-norm runs host-side post-NPU",
        )
        .with_graph(graph);

    let final_path = save_bundle(&bundle, PathBuf::from(&out)).expect("save_bundle failed");
    let total_params: usize = bundle
        .graph
        .as_ref()
        .unwrap()
        .initializers
        .values()
        .map(|t| t.data.len())
        .sum();
    println!("wrote bundle: {}", final_path.display());
    println!(
        "file size: {} bytes",
        std::fs::metadata(&final_path).unwrap().len()
    );
    println!(
        "compute nodes: {}",
        bundle.graph.as_ref().unwrap().nodes.len()
    );
    println!(
        "initializers: {}",
        bundle.graph.as_ref().unwrap().initializers.len()
    );
    println!("total params: {total_params}");
}

/// Tiny LCG-based pseudo-random init in `[-k, k]` where `k = sqrt(2 / fan_in)`.
/// Deterministic per `seed` so consecutive runs produce byte-identical bundles.
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
        // map u32 → [-1, 1)
        let f = (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        out.push(f * k);
    }
    out
}
