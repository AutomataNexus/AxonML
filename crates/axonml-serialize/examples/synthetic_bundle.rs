//! Build a tiny Conv2d → BatchNorm → Relu → GlobalAvgPool → Gemm bundle
//! and save it to `.axonml` for the NexusFoundry smoke compile.
//!
//! Usage:
//!   cargo run --example synthetic_bundle -p axonml-serialize -- <output_path>
//!
//! The bundle uses initializer values seeded by index×0.01 for conv weights
//! and identity-ish parameters for batchnorm — no training, just enough
//! structure for NexusFoundry to walk the graph and emit a non-trivial HEF.
//!
//! Architecture matches `nexusfoundry-cli/tests/e2e_pipeline::build_synthetic_ir`
//! so the smoke compile exercises the same op set those tests already cover.

use std::path::PathBuf;

use axonml_serialize::{BundleGraph, ModelBundle, save_bundle};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/synthetic_bundle.axonml".to_string());

    let in_c: i64 = 3;
    let conv_c: i64 = 16;
    let h: i64 = 32;
    let w: i64 = 32;
    let n_classes: i64 = 10;

    let mut graph = BundleGraph::new();

    // Graph I/O
    graph.add_input("input", vec![-1, in_c, h, w]);
    graph.add_output("logits", vec![-1, n_classes]);

    // Conv2d weights (16, 3, 3, 3) + bias (16)
    let cw_n = (conv_c * in_c * 3 * 3) as usize;
    graph.add_initializer(
        "conv.weight",
        vec![conv_c, in_c, 3, 3],
        (0..cw_n).map(|i| (i as f32) * 0.01).collect(),
    );
    graph.add_initializer("conv.bias", vec![conv_c], vec![0.0; conv_c as usize]);

    // BatchNorm γ,β,μ,σ²  — identity-ish init
    graph.add_initializer("bn.weight", vec![conv_c], vec![1.0; conv_c as usize]);
    graph.add_initializer("bn.bias", vec![conv_c], vec![0.0; conv_c as usize]);
    graph.add_initializer("bn.running_mean", vec![conv_c], vec![0.0; conv_c as usize]);
    graph.add_initializer("bn.running_var", vec![conv_c], vec![1.0; conv_c as usize]);

    // Linear (Gemm) weights (n_classes, conv_c) + bias (n_classes)
    let fc_n = (n_classes * conv_c) as usize;
    graph.add_initializer(
        "fc.weight",
        vec![n_classes, conv_c],
        (0..fc_n).map(|i| (i as f32) * 0.001).collect(),
    );
    graph.add_initializer(
        "fc.bias",
        vec![n_classes],
        vec![0.0; n_classes as usize],
    );

    // Compute nodes
    graph.add_node(
        "conv1",
        "Conv2d",
        serde_json::json!({
            "kernel_shape": [3, 3],
            "strides": [1, 1],
            "pads": [1, 1, 1, 1],
            "dilations": [1, 1],
            "group": 1,
        }),
        vec!["input", "conv.weight", "conv.bias"],
        vec!["conv_out"],
    );
    graph.add_node(
        "bn1",
        "BatchNorm",
        serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        vec![
            "conv_out",
            "bn.weight",
            "bn.bias",
            "bn.running_mean",
            "bn.running_var",
        ],
        vec!["bn_out"],
    );
    graph.add_node(
        "relu1",
        "Relu",
        serde_json::Value::Null,
        vec!["bn_out"],
        vec!["relu_out"],
    );
    graph.add_node(
        "gap1",
        "GlobalAvgPool",
        serde_json::Value::Null,
        vec!["relu_out"],
        vec!["pooled"],
    );
    graph.add_node(
        "fc1",
        "Gemm",
        serde_json::json!({
            "alpha": 1.0,
            "beta": 1.0,
            "trans_a": false,
            "trans_b": true,
        }),
        vec!["pooled", "fc.weight", "fc.bias"],
        vec!["logits"],
    );

    let bundle = ModelBundle::new("synthetic_smoke_cnn", in_c as usize, Vec::new())
        .with_hyperparam("input_h", h)
        .with_hyperparam("input_w", w)
        .with_hyperparam("num_classes", n_classes)
        .with_graph(graph);

    let final_path = save_bundle(&bundle, PathBuf::from(&out)).expect("save_bundle failed");
    println!("wrote bundle: {}", final_path.display());
    println!(
        "size: {} bytes",
        std::fs::metadata(&final_path).unwrap().len()
    );
}
