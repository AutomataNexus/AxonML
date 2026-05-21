//! Prometheus — Sparse Autoencoder for anomaly detection / feature extraction
//!
//! Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! Architecture: Conv2d encoder → bottleneck → Conv2d decoder
//!
//!   Encoder:
//!     [B, 1, 64, 64]
//!       conv1  (1→16, 3x3, s=2, p=1)  → BN → ReLU    [B, 16, 32, 32]
//!       conv2  (16→32, 3x3, s=2, p=1) → BN → ReLU    [B, 32, 16, 16]
//!       conv3  (32→64, 3x3, s=2, p=1) → BN → ReLU    [B, 64, 8, 8]
//!     GlobalAvgPool → Flatten                          [B, 64]
//!     Gemm (64→16) → ReLU                              [B, 16]  (sparse bottleneck)
//!
//!   Decoder:
//!     Gemm (16→64)  → ReLU                             [B, 64]
//!     Gemm (64→256) → ReLU                             [B, 256]
//!     Gemm (256→1024) → ReLU                           [B, 1024]
//!     Gemm (1024→4096)                                 [B, 4096]  (= 1×64×64 reconstructed)
//!
//!   Output: [B, 4096] flattened reconstruction
//!
//! The 16-dim bottleneck enforces sparsity through dimensionality reduction.
//! Anomaly score = reconstruction MSE (computed host-side).
//!
//! All ops are NexusFoundry-compilable: Conv2d, BatchNorm, Relu, GlobalAvgPool,
//! Flatten, Gemm.
//!
//! Usage:
//!   cargo run --release --example prometheus_sae_bundle -p axonml-serialize -- \
//!       <output.axonml>

use axonml_serialize::{BundleGraph, ModelBundle, save_bundle};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "models/prometheus_sae.axonml".to_string());

    let mut graph = BundleGraph::new();

    // ═══════════════════════════════════════════
    // Graph I/O
    // ═══════════════════════════════════════════

    graph.add_input("input", vec![-1, 1, 64, 64]);
    graph.add_output("reconstruction", vec![-1, 4096]);

    // ═══════════════════════════════════════════
    // Encoder: 3× (Conv2d-s2 + BN + ReLU)
    // ═══════════════════════════════════════════

    let enc_convs: [(&str, i64, i64, &str, &str); 3] = [
        ("enc_conv1", 1, 16, "input", "enc_conv1_out"),
        ("enc_conv2", 16, 32, "enc_relu1_out", "enc_conv2_out"),
        ("enc_conv3", 32, 64, "enc_relu2_out", "enc_conv3_out"),
    ];

    for (i, (name, in_c, out_c, in_act, out_act)) in enc_convs.iter().enumerate() {
        let bn = format!("enc_bn{}", i + 1);
        let relu = format!("enc_relu{}", i + 1);
        let bn_out = format!("enc_bn{}_out", i + 1);
        let relu_out = format!("enc_relu{}_out", i + 1);

        let cw = format!("{name}.weight");
        let cb = format!("{name}.bias");
        let n = (out_c * in_c * 3 * 3) as usize;
        graph.add_initializer(
            &cw,
            vec![*out_c, *in_c, 3, 3],
            init_kaiming(n, (*in_c * 9) as usize, i as u64),
        );
        graph.add_initializer(&cb, vec![*out_c], vec![0.0; *out_c as usize]);

        graph.add_initializer(
            &format!("{bn}.weight"),
            vec![*out_c],
            vec![1.0; *out_c as usize],
        );
        graph.add_initializer(
            &format!("{bn}.bias"),
            vec![*out_c],
            vec![0.0; *out_c as usize],
        );
        graph.add_initializer(
            &format!("{bn}.running_mean"),
            vec![*out_c],
            vec![0.0; *out_c as usize],
        );
        graph.add_initializer(
            &format!("{bn}.running_var"),
            vec![*out_c],
            vec![1.0; *out_c as usize],
        );

        graph.add_node(name, "Conv2d",
            serde_json::json!({"kernel_shape":[3,3],"strides":[2,2],"pads":[1,1,1,1],"dilations":[1,1],"group":1}),
            vec![in_act, &cw, &cb], vec![out_act]);
        graph.add_node(
            &bn,
            "BatchNorm",
            serde_json::json!({"epsilon":1e-5,"momentum":0.1}),
            vec![
                out_act,
                &format!("{bn}.weight"),
                &format!("{bn}.bias"),
                &format!("{bn}.running_mean"),
                &format!("{bn}.running_var"),
            ],
            vec![&bn_out],
        );
        graph.add_node(
            &relu,
            "Relu",
            serde_json::Value::Null,
            vec![&bn_out],
            vec![&relu_out],
        );
    }

    // ═══════════════════════════════════════════
    // Bottleneck: GlobalAvgPool → Flatten → Gemm(64→16) → ReLU
    // ═══════════════════════════════════════════

    graph.add_node(
        "enc_gap",
        "GlobalAvgPool",
        serde_json::Value::Null,
        vec!["enc_relu3_out"],
        vec!["enc_pooled"],
    );
    graph.add_node(
        "enc_flatten",
        "Flatten",
        serde_json::json!({"axis":1}),
        vec!["enc_pooled"],
        vec!["enc_flat"],
    );

    graph.add_initializer(
        "bottleneck.weight",
        vec![16, 64],
        init_kaiming(16 * 64, 64, 10),
    );
    graph.add_initializer("bottleneck.bias", vec![16], vec![0.0; 16]);
    graph.add_node(
        "bottleneck",
        "Gemm",
        serde_json::json!({"alpha":1.0,"beta":1.0,"trans_a":false,"trans_b":true}),
        vec!["enc_flat", "bottleneck.weight", "bottleneck.bias"],
        vec!["latent_raw"],
    );
    graph.add_node(
        "bottleneck_relu",
        "Relu",
        serde_json::Value::Null,
        vec!["latent_raw"],
        vec!["latent"],
    );

    // ═══════════════════════════════════════════
    // Decoder: Gemm stack back to 4096
    // ═══════════════════════════════════════════

    let dec_layers: [(&str, i64, i64, &str, &str); 4] = [
        ("dec_fc1", 16, 64, "latent", "dec_fc1_out"),
        ("dec_fc2", 64, 256, "dec_relu1_out", "dec_fc2_out"),
        ("dec_fc3", 256, 1024, "dec_relu2_out", "dec_fc3_out"),
        ("dec_fc4", 1024, 4096, "dec_relu3_out", "dec_fc4_out"),
    ];

    for (i, (name, in_f, out_f, in_act, out_act)) in dec_layers.iter().enumerate() {
        let w = format!("{name}.weight");
        let b = format!("{name}.bias");
        graph.add_initializer(
            &w,
            vec![*out_f, *in_f],
            init_kaiming((*out_f * *in_f) as usize, *in_f as usize, 20 + i as u64),
        );
        graph.add_initializer(&b, vec![*out_f], vec![0.0; *out_f as usize]);

        graph.add_node(
            name,
            "Gemm",
            serde_json::json!({"alpha":1.0,"beta":1.0,"trans_a":false,"trans_b":true}),
            vec![in_act, &w, &b],
            vec![out_act],
        );

        if i < 3 {
            let relu = format!("dec_relu{}", i + 1);
            let relu_out = format!("dec_relu{}_out", i + 1);
            graph.add_node(
                &relu,
                "Relu",
                serde_json::Value::Null,
                vec![out_act],
                vec![&relu_out],
            );
        }
    }

    // Final output: rename dec_fc4_out → reconstruction
    graph.add_node(
        "output_identity",
        "Identity",
        serde_json::Value::Null,
        vec!["dec_fc4_out"],
        vec!["reconstruction"],
    );

    // ═══════════════════════════════════════════
    // Bundle
    // ═══════════════════════════════════════════

    let bundle = ModelBundle::new("prometheus_sae", 1, Vec::new())
        .with_hyperparam("input_h", 64)
        .with_hyperparam("input_w", 64)
        .with_hyperparam("latent_dim", 16)
        .with_hyperparam(
            "description",
            "Prometheus Sparse Autoencoder — Conv2d encoder, \
            16-dim sparse bottleneck, Gemm decoder. Anomaly score = reconstruction MSE.",
        )
        .with_graph(graph);

    let final_path = save_bundle(&bundle, PathBuf::from(&out)).expect("save_bundle failed");
    let g = bundle.graph.as_ref().unwrap();
    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();

    eprintln!("  Prometheus SAE — Sparse Autoencoder");
    eprintln!("  Architecture: Conv2d encoder → 16-dim bottleneck → Gemm decoder");
    eprintln!("  Input: [B, 1, 64, 64] → Output: [B, 4096]");
    eprintln!("  Compute nodes: {}", g.nodes.len());
    eprintln!("  Initializers: {}", g.initializers.len());
    eprintln!(
        "  Total params: {total_params} ({:.1} KB)",
        total_params as f64 * 4.0 / 1024.0
    );
    eprintln!("  Bundle: {}", final_path.display());
    eprintln!(
        "  Size: {} bytes",
        std::fs::metadata(&final_path).unwrap().len()
    );
}

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
