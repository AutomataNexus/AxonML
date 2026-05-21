//! HVAC controller v3 — TCN-style, NF-supported ops only, Hailo + CPU.
//!
//! Replaces the v1/v2 LSTM-Autoencoder + Multi-Horizon-GRU pairs with a clean
//! Conv-only architecture that compiles to Hailo silicon AND runs on CPU via
//! AxonML's existing inference path. One binary builds both halves of any
//! location's pair from MODELS.md config; per-location configs are baked
//! below from the EdgeModels MODELS.md master table.
//!
//! ## Architectural correspondence
//!
//! v1/v2 pair:                          v3 pair:
//!   Anomaly:  LSTM-Autoencoder            Anomaly:  TCN-Autoencoder
//!   Predictor: Multi-Horizon GRU          Predictor: TCN multi-head classifier
//!
//! Inputs match exactly (sequence_length × num_features). Outputs match: 3
//! horizon-specific class-probability heads + 1 health-score head from the
//! predictor; reconstruction tensor + MSE-on-host from the anomaly half.
//!
//! ## Op coverage (cross-checked against NexusFoundry IrOp set)
//!
//! - Conv2d (with H=1 to stand in for Conv1d on Hailo)
//! - TransposedConv2d (decoder upsampling in the autoencoder)
//! - BatchNorm + Relu
//! - GlobalAvgPool
//! - Gemm
//!
//! No Slice / Gather / ArgMax / Erf / LayerNorm — all the ops that broke the
//! 2026-01-17 DFC parse attempt on hvac_multi_horizon_predictor_v2.onnx
//! (see /mnt/d/Projects/EdgeModels/warren/innis/inference/hailo_sdk.client.log).
//! ArgMax + softmax over horizon-class logits live on host CPU (sub-µs);
//! reconstruction MSE for the autoencoder lives on host too.
//!
//! ## Usage
//!
//! ```bash
//! # Build all 33 locations:
//! cargo run --release --example hvac_v3_bundle -p axonml-serialize -- all  /tmp/hvac_v3
//!
//! # Build a single location pair:
//! cargo run --release --example hvac_v3_bundle -p axonml-serialize -- warren-innis  /tmp/hvac_v3
//! ```
//!
//! Output per location: two .axonml bundles
//!   <out>/<location>_anomaly.axonml      (Nyx-equivalent — TCN AE)
//!   <out>/<location>_predictor.axonml    (Chronos-equivalent — TCN classifier)
//!
//! Both use random Kaiming-initialized weights — for production accuracy,
//! retrain via the AxonML trainer that mirrors these graphs (Phase 2d/2e).

use std::path::PathBuf;

use axonml_serialize::{BundleGraph, ModelBundle, save_bundle};

// ---------------------------------------------------------------------------
// Per-location configuration — baked from EdgeModels MODELS.md
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LocConfig {
    /// Slug, e.g. "warren-innis"
    slug: &'static str,
    /// Anomaly-side model name (e.g. "Nyx")
    anomaly_name: &'static str,
    /// Predictor-side model name (e.g. "Chronos")
    predictor_name: &'static str,
    /// Number of sensor channels
    n_features: i64,
    /// Number of failure classes
    n_classes: i64,
    /// Anomaly hidden / latent dims (H/L from MODELS.md)
    anomaly_hidden: i64,
    anomaly_latent: i64,
    /// Predictor hidden / attention dims (H/A from MODELS.md)
    predictor_hidden: i64,
    predictor_attn: i64,
}

const ANOM_SEQ: i64 = 60; // 60 samples @ 1 Hz = 1 min  — fixed across all locations
const PRED_SEQ: i64 = 120; // 120 samples @ 1 Hz = 2 min — fixed
const N_HORIZONS: i64 = 3; // 5 / 15 / 30 min — fixed

const LOCATIONS: &[LocConfig] = &[
    // Warren
    LocConfig {
        slug: "warren-innis",
        anomaly_name: "Nyx",
        predictor_name: "Chronos",
        n_features: 28,
        n_classes: 20,
        anomaly_hidden: 64,
        anomaly_latent: 16,
        predictor_hidden: 128,
        predictor_attn: 64,
    },
    LocConfig {
        slug: "warren-ahu1",
        anomaly_name: "Aether",
        predictor_name: "Moros",
        n_features: 11,
        n_classes: 11,
        anomaly_hidden: 32,
        anomaly_latent: 8,
        predictor_hidden: 64,
        predictor_attn: 32,
    },
    LocConfig {
        slug: "warren-ahu2",
        anomaly_name: "Phanes",
        predictor_name: "Hecate",
        n_features: 16,
        n_classes: 13,
        anomaly_hidden: 48,
        anomaly_latent: 12,
        predictor_hidden: 96,
        predictor_attn: 48,
    },
    LocConfig {
        slug: "warren-ahu4",
        anomaly_name: "Nyctos",
        predictor_name: "Cassandra",
        n_features: 11,
        n_classes: 10,
        anomaly_hidden: 32,
        anomaly_latent: 8,
        predictor_hidden: 64,
        predictor_attn: 32,
    },
    LocConfig {
        slug: "warren-ahu7",
        anomaly_name: "Poseidon",
        predictor_name: "Triton",
        n_features: 8,
        n_classes: 9,
        anomaly_hidden: 32,
        anomaly_latent: 8,
        predictor_hidden: 64,
        predictor_attn: 32,
    },
    LocConfig {
        slug: "warren-chapel",
        anomaly_name: "Apollo",
        predictor_name: "Sibyl",
        n_features: 27,
        n_classes: 14,
        anomaly_hidden: 64,
        anomaly_latent: 16,
        predictor_hidden: 128,
        predictor_attn: 64,
    },
    LocConfig {
        slug: "warren-cove",
        anomaly_name: "Demeter",
        predictor_name: "Persephone",
        n_features: 15,
        n_classes: 12,
        anomaly_hidden: 48,
        anomaly_latent: 12,
        predictor_hidden: 96,
        predictor_attn: 48,
    },
    LocConfig {
        slug: "warren-a-basement",
        anomaly_name: "Athena",
        predictor_name: "Ares",
        n_features: 29,
        n_classes: 16,
        anomaly_hidden: 64,
        anomaly_latent: 16,
        predictor_hidden: 128,
        predictor_attn: 64,
    },
    LocConfig {
        slug: "warren-activity-rooms",
        anomaly_name: "Hermes",
        predictor_name: "Tyche",
        n_features: 15,
        n_classes: 13,
        anomaly_hidden: 48,
        anomaly_latent: 12,
        predictor_hidden: 96,
        predictor_attn: 48,
    },
    LocConfig {
        slug: "warren-chompson-steambundle",
        anomaly_name: "Pyrrha",
        predictor_name: "Clotho",
        n_features: 5,
        n_classes: 8,
        anomaly_hidden: 32,
        anomaly_latent: 8,
        predictor_hidden: 64,
        predictor_attn: 32,
    },
    LocConfig {
        slug: "warren-executive-offices",
        anomaly_name: "Calliope",
        predictor_name: "Lachesis",
        n_features: 8,
        n_classes: 10,
        anomaly_hidden: 32,
        anomaly_latent: 8,
        predictor_hidden: 64,
        predictor_attn: 32,
    },
    LocConfig {
        slug: "warren-fahl-steambundle",
        anomaly_name: "Kratos",
        predictor_name: "Atropos",
        n_features: 6,
        n_classes: 9,
        anomaly_hidden: 32,
        anomaly_latent: 8,
        predictor_hidden: 64,
        predictor_attn: 32,
    },
    LocConfig {
        slug: "warren-innis-mechroom",
        anomaly_name: "Daedalus",
        predictor_name: "Pythia",
        n_features: 17,
        n_classes: 15,
        anomaly_hidden: 48,
        anomaly_latent: 12,
        predictor_hidden: 96,
        predictor_attn: 48,
    },
    // (Add remaining 20 locations here as MODELS.md is consolidated. Pattern is:
    //  small locations → 32/8 + 64/32, medium → 48/12 + 96/48, large → 64/16 + 128/64.)
];

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

/// Build a TCN-style anomaly autoencoder bundle for the given location.
///
/// Input  shape: `(batch, n_features, 1, ANOM_SEQ)` fp32 — sensor sequence reshaped to NCHW.
/// Output shape: `(batch, n_features, 1, ANOM_SEQ)` fp32 — reconstructed sensor sequence.
/// Anomaly score = MSE(input - output) on host, threshold per location.
fn build_anomaly_bundle(c: &LocConfig, seed_base: u64) -> ModelBundle {
    let mut g = BundleGraph::new();
    let f = c.n_features;
    let h = c.anomaly_hidden;
    let l = c.anomaly_latent;

    g.add_input("sensor_seq", vec![-1, f, 1, ANOM_SEQ]);
    g.add_output("reconstructed", vec![-1, f, 1, ANOM_SEQ]);

    // Encoder: f → h → h → l
    add_conv_bn_relu(
        &mut g,
        "enc1",
        f,
        h,
        /*stride*/ 2,
        "sensor_seq",
        "enc1_out",
        seed_base + 1,
    );
    add_conv_bn_relu(
        &mut g,
        "enc2",
        h,
        h,
        /*stride*/ 2,
        "enc1_out_relu",
        "enc2_out",
        seed_base + 2,
    );
    add_conv_pointwise_bn_relu(
        &mut g,
        "bottleneck",
        h,
        l,
        "enc2_out_relu",
        "bottleneck_out",
        seed_base + 3,
    );

    // Decoder: l → h → h → f
    // DFC 5.3.0's compiler stage chokes on TransposedConv2d (TypeError: NoneType
    // in compiled-graph emit). Replace with Resize(scale=2) + Conv2d, which is
    // structurally equivalent (nearest-neighbor upsample then learnable refine)
    // and emits cleanly through every DFC stage.
    add_conv_pointwise_bn_relu(
        &mut g,
        "dec0",
        l,
        h,
        "bottleneck_out_relu",
        "dec0_out",
        seed_base + 4,
    );

    // Upsample × 2 via Resize, then refine with Conv2d-BN-Relu
    add_resize_2x_w(&mut g, "up1", "dec0_out_relu", "up1_out");
    add_conv_bn_relu_same(&mut g, "dec1", h, h, "up1_out", "dec1_out", seed_base + 5);

    // Final upsample × 2 + linear-output Conv to reconstruct
    add_resize_2x_w(&mut g, "up2", "dec1_out_relu", "up2_out");
    add_conv_linear_same(
        &mut g,
        "dec2",
        h,
        f,
        "up2_out",
        "reconstructed",
        seed_base + 6,
    );

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(
        &format!("hvac_v3_anomaly_{}", c.slug),
        f as usize,
        Vec::new(),
    )
    .with_hyperparam("location_slug", c.slug)
    .with_hyperparam("model_role", c.anomaly_name)
    .with_hyperparam("seq_len", ANOM_SEQ)
    .with_hyperparam("n_features", f)
    .with_hyperparam("hidden", h)
    .with_hyperparam("latent", l)
    .with_hyperparam(
        "note",
        format!("v3 TCN-AE replacing v1/v2 LSTM-AE; total_params={total_params}"),
    )
    .with_graph(g)
}

/// Build a TCN-style multi-horizon predictor bundle for the given location.
///
/// Input  shape: `(batch, n_features, 1, PRED_SEQ)` fp32.
/// Outputs: 3 horizon classification heads `(batch, n_classes)` + 1 health regression `(batch, 1)`.
/// Argmax / softmax over class logits run on host (sub-µs).
fn build_predictor_bundle(c: &LocConfig, seed_base: u64) -> ModelBundle {
    let mut g = BundleGraph::new();
    let f = c.n_features;
    let h = c.predictor_hidden;
    let a = c.predictor_attn;
    let n_cls = c.n_classes;

    g.add_input("sensor_seq", vec![-1, f, 1, PRED_SEQ]);
    g.add_output("imminent_logits", vec![-1, n_cls]);
    g.add_output("warning_logits", vec![-1, n_cls]);
    g.add_output("early_logits", vec![-1, n_cls]);
    g.add_output("health_score", vec![-1, 1]);

    // Encoder: f → h → h → a   (3 stride-2 stages = 8× temporal downsampling)
    add_conv_bn_relu(
        &mut g,
        "enc1",
        f,
        h,
        2,
        "sensor_seq",
        "enc1_out",
        seed_base + 1,
    );
    add_conv_bn_relu(
        &mut g,
        "enc2",
        h,
        h,
        2,
        "enc1_out_relu",
        "enc2_out",
        seed_base + 2,
    );
    add_conv_bn_relu(
        &mut g,
        "enc3",
        h,
        a,
        2,
        "enc2_out_relu",
        "enc3_out",
        seed_base + 3,
    );

    // Pool to per-channel scalar:  [B, a, 1, T/8]  →  [B, a, 1, 1]
    g.add_node(
        "gap",
        "GlobalAvgPool",
        serde_json::Value::Null,
        vec!["enc3_out_relu"],
        vec!["pooled"],
    );

    // Flatten [B, a, 1, 1] → [B, a]. DFC parser rejects Gemm directly on rank-4 input;
    // Flatten(axis=1) is the canonical NF/DFC-friendly bridge.
    g.add_node(
        "flatten",
        "Flatten",
        serde_json::json!({"axis": 1}),
        vec!["pooled"],
        vec!["pooled_flat"],
    );

    // 4 heads, each Gemm(a → out)
    for (head, out_dim, out_name) in &[
        ("imminent", n_cls, "imminent_logits"),
        ("warning", n_cls, "warning_logits"),
        ("early", n_cls, "early_logits"),
        ("health", 1i64, "health_score"),
    ] {
        let w = format!("head_{head}.weight");
        let b = format!("head_{head}.bias");
        g.add_initializer(
            &w,
            vec![*out_dim, a],
            init_kaiming(
                (*out_dim as usize) * (a as usize),
                a as usize,
                seed_base + 100 + head.len() as u64,
            ),
        );
        g.add_initializer(&b, vec![*out_dim], vec![0.0; *out_dim as usize]);
        g.add_node(
            &format!("head_{head}"),
            "Gemm",
            serde_json::json!({"alpha": 1.0, "beta": 1.0, "trans_a": false, "trans_b": true}),
            vec!["pooled_flat", &w, &b],
            vec![*out_name],
        );
    }

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(
        &format!("hvac_v3_predictor_{}", c.slug),
        f as usize,
        Vec::new(),
    )
    .with_hyperparam("location_slug", c.slug)
    .with_hyperparam("model_role", c.predictor_name)
    .with_hyperparam("seq_len", PRED_SEQ)
    .with_hyperparam("n_features", f)
    .with_hyperparam("hidden", h)
    .with_hyperparam("attn", a)
    .with_hyperparam("n_classes", n_cls)
    .with_hyperparam("n_horizons", N_HORIZONS)
    .with_hyperparam(
        "note",
        format!("v3 TCN-MultiHorizon replacing v1/v2 GRU; total_params={total_params}"),
    )
    .with_graph(g)
}

// ---------------------------------------------------------------------------
// Layer building blocks (Conv2d/BN/Relu fuses cleanly via NF's conv_bn_relu_fusion pass)
// ---------------------------------------------------------------------------

fn add_conv_bn_relu(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    stride: i64,
    in_act: &str,
    out_act: &str,
    seed: u64,
) {
    let cw = format!("{name}.conv.weight");
    let cb = format!("{name}.conv.bias");
    let bn_w = format!("{name}.bn.weight");
    let bn_b = format!("{name}.bn.bias");
    let bn_m = format!("{name}.bn.running_mean");
    let bn_v = format!("{name}.bn.running_var");
    let bn_out = format!("{name}_bn_out");
    let relu_out = format!("{out_act}_relu");

    // Conv weight: [out_c, in_c, 1, 3]
    let w_n = (out_c * in_c * 1 * 3) as usize;
    g.add_initializer(
        &cw,
        vec![out_c, in_c, 1, 3],
        init_kaiming(w_n, (in_c * 3) as usize, seed),
    );
    g.add_initializer(&cb, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_w, vec![out_c], vec![1.0; out_c as usize]);
    g.add_initializer(&bn_b, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_m, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_v, vec![out_c], vec![1.0; out_c as usize]);

    g.add_node(
        &format!("{name}_conv"),
        "Conv2d",
        serde_json::json!({
            "kernel_shape": [1, 3],
            "strides": [1, stride],
            "pads": [0, 1, 0, 1],
            "dilations": [1, 1],
            "group": 1,
        }),
        vec![in_act, &cw, &cb],
        vec![out_act],
    );
    g.add_node(
        &format!("{name}_bn"),
        "BatchNorm",
        serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        vec![out_act, &bn_w, &bn_b, &bn_m, &bn_v],
        vec![&bn_out],
    );
    g.add_node(
        &format!("{name}_relu"),
        "Relu",
        serde_json::Value::Null,
        vec![&bn_out],
        vec![&relu_out],
    );
}

/// 1×1 pointwise Conv2d + BN + Relu — used at the bottleneck where we squeeze channels.
fn add_conv_pointwise_bn_relu(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    in_act: &str,
    out_act: &str,
    seed: u64,
) {
    let cw = format!("{name}.conv.weight");
    let cb = format!("{name}.conv.bias");
    let bn_w = format!("{name}.bn.weight");
    let bn_b = format!("{name}.bn.bias");
    let bn_m = format!("{name}.bn.running_mean");
    let bn_v = format!("{name}.bn.running_var");
    let bn_out = format!("{name}_bn_out");
    let relu_out = format!("{out_act}_relu");

    g.add_initializer(
        &cw,
        vec![out_c, in_c, 1, 1],
        init_kaiming((out_c * in_c) as usize, in_c as usize, seed),
    );
    g.add_initializer(&cb, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_w, vec![out_c], vec![1.0; out_c as usize]);
    g.add_initializer(&bn_b, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_m, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_v, vec![out_c], vec![1.0; out_c as usize]);

    g.add_node(&format!("{name}_conv"), "Conv2d",
        serde_json::json!({"kernel_shape": [1, 1], "strides": [1, 1], "pads": [0, 0, 0, 0], "dilations": [1, 1], "group": 1}),
        vec![in_act, &cw, &cb], vec![out_act]);
    g.add_node(
        &format!("{name}_bn"),
        "BatchNorm",
        serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        vec![out_act, &bn_w, &bn_b, &bn_m, &bn_v],
        vec![&bn_out],
    );
    g.add_node(
        &format!("{name}_relu"),
        "Relu",
        serde_json::Value::Null,
        vec![&bn_out],
        vec![&relu_out],
    );
}

/// TransposedConv2d (decoder upsample) + BN + Relu.
fn add_tconv_bn_relu(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    stride: i64,
    in_act: &str,
    out_act: &str,
    seed: u64,
) {
    let cw = format!("{name}.tconv.weight");
    let cb = format!("{name}.tconv.bias");
    let bn_w = format!("{name}.bn.weight");
    let bn_b = format!("{name}.bn.bias");
    let bn_m = format!("{name}.bn.running_mean");
    let bn_v = format!("{name}.bn.running_var");
    let bn_out = format!("{name}_bn_out");
    let relu_out = format!("{out_act}_relu");

    // TransposedConv weight layout in ONNX: [in_c, out_c, kH, kW]
    let w_n = (in_c * out_c * 1 * 3) as usize;
    g.add_initializer(
        &cw,
        vec![in_c, out_c, 1, 3],
        init_kaiming(w_n, (in_c * 3) as usize, seed),
    );
    g.add_initializer(&cb, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_w, vec![out_c], vec![1.0; out_c as usize]);
    g.add_initializer(&bn_b, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_m, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_v, vec![out_c], vec![1.0; out_c as usize]);

    g.add_node(
        &format!("{name}_tconv"),
        "TransposedConv2d",
        serde_json::json!({
            "kernel_shape": [1, 3],
            "strides": [1, stride],
            "pads": [0, 1, 0, 1],
            "dilations": [1, 1],
            "group": 1,
        }),
        vec![in_act, &cw, &cb],
        vec![out_act],
    );
    g.add_node(
        &format!("{name}_bn"),
        "BatchNorm",
        serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        vec![out_act, &bn_w, &bn_b, &bn_m, &bn_v],
        vec![&bn_out],
    );
    g.add_node(
        &format!("{name}_relu"),
        "Relu",
        serde_json::Value::Null,
        vec![&bn_out],
        vec![&relu_out],
    );
}

/// Resize × 2 along the W dimension only (we keep H=1 for our 1D-as-2D layout).
/// Emits a Resize node with explicit `scales` initializer of `[1, 1, 1, 2]`.
fn add_resize_2x_w(g: &mut BundleGraph, name: &str, in_act: &str, out_act: &str) {
    // ONNX Resize wants `roi` and `scales` as inputs (opset 13+).
    // Empty roi (scalar empty) + scales [1,1,1,2] is the standard nearest-neighbor up-2x pattern.
    let roi_name = format!("{name}.roi");
    let scales_name = format!("{name}.scales");
    g.add_initializer(&roi_name, vec![0], Vec::<f32>::new());
    g.add_initializer(&scales_name, vec![4], vec![1.0, 1.0, 1.0, 2.0]);
    g.add_node(
        name,
        "Resize",
        serde_json::json!({
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        }),
        vec![in_act, &roi_name, &scales_name],
        vec![out_act],
    );
}

/// Standard Conv2d (1×3 kernel, stride 1, pad 1 → "same" padding) + BN + Relu.
/// Used for decoder refinement after upsampling.
fn add_conv_bn_relu_same(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    in_act: &str,
    out_act: &str,
    seed: u64,
) {
    let cw = format!("{name}.conv.weight");
    let cb = format!("{name}.conv.bias");
    let bn_w = format!("{name}.bn.weight");
    let bn_b = format!("{name}.bn.bias");
    let bn_m = format!("{name}.bn.running_mean");
    let bn_v = format!("{name}.bn.running_var");
    let bn_out = format!("{name}_bn_out");
    let relu_out = format!("{out_act}_relu");

    let w_n = (out_c * in_c * 1 * 3) as usize;
    g.add_initializer(
        &cw,
        vec![out_c, in_c, 1, 3],
        init_kaiming(w_n, (in_c * 3) as usize, seed),
    );
    g.add_initializer(&cb, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_w, vec![out_c], vec![1.0; out_c as usize]);
    g.add_initializer(&bn_b, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_m, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_v, vec![out_c], vec![1.0; out_c as usize]);

    g.add_node(
        &format!("{name}_conv"),
        "Conv2d",
        serde_json::json!({
            "kernel_shape": [1, 3],
            "strides": [1, 1],
            "pads": [0, 1, 0, 1],
            "dilations": [1, 1],
            "group": 1,
        }),
        vec![in_act, &cw, &cb],
        vec![out_act],
    );
    g.add_node(
        &format!("{name}_bn"),
        "BatchNorm",
        serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        vec![out_act, &bn_w, &bn_b, &bn_m, &bn_v],
        vec![&bn_out],
    );
    g.add_node(
        &format!("{name}_relu"),
        "Relu",
        serde_json::Value::Null,
        vec![&bn_out],
        vec![&relu_out],
    );
}

/// Linear-output Conv2d (no BN/Relu) — reconstruction head emits raw signal.
fn add_conv_linear_same(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    in_act: &str,
    out_act: &str,
    seed: u64,
) {
    let cw = format!("{name}.conv.weight");
    let cb = format!("{name}.conv.bias");
    let w_n = (out_c * in_c * 1 * 3) as usize;
    g.add_initializer(
        &cw,
        vec![out_c, in_c, 1, 3],
        init_kaiming(w_n, (in_c * 3) as usize, seed),
    );
    g.add_initializer(&cb, vec![out_c], vec![0.0; out_c as usize]);
    g.add_node(
        &format!("{name}_conv"),
        "Conv2d",
        serde_json::json!({
            "kernel_shape": [1, 3],
            "strides": [1, 1],
            "pads": [0, 1, 0, 1],
            "dilations": [1, 1],
            "group": 1,
        }),
        vec![in_act, &cw, &cb],
        vec![out_act],
    );
}

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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hvac_v3_bundle <slug|all> <out_dir>");
        eprintln!("       hvac_v3_bundle all /tmp/hvac_v3");
        eprintln!("       hvac_v3_bundle warren-innis /tmp/hvac_v3");
        std::process::exit(2);
    }
    let target = &args[1];
    let out_dir = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out_dir).expect("mkdir -p out_dir");

    let to_build: Vec<&LocConfig> = if target == "all" {
        LOCATIONS.iter().collect()
    } else {
        LOCATIONS.iter().filter(|c| c.slug == target).collect()
    };
    if to_build.is_empty() {
        eprintln!("no location matched '{target}'");
        eprintln!("known slugs:");
        for c in LOCATIONS {
            eprintln!("  {}  ({} / {})", c.slug, c.anomaly_name, c.predictor_name);
        }
        std::process::exit(1);
    }

    let mut total_anomaly = 0usize;
    let mut total_predictor = 0usize;
    for (i, c) in to_build.iter().enumerate() {
        let seed = (i as u64).wrapping_mul(1_000_001) + 0xc0ffee;

        let anom = build_anomaly_bundle(c, seed);
        let pred = build_predictor_bundle(c, seed + 500_000);

        let anom_path = out_dir.join(format!("{}_anomaly.axonml", c.slug));
        let pred_path = out_dir.join(format!("{}_predictor.axonml", c.slug));
        save_bundle(&anom, &anom_path).expect("save anomaly bundle");
        save_bundle(&pred, &pred_path).expect("save predictor bundle");

        let a_params: usize = anom
            .graph
            .as_ref()
            .unwrap()
            .initializers
            .values()
            .map(|t| t.data.len())
            .sum();
        let p_params: usize = pred
            .graph
            .as_ref()
            .unwrap()
            .initializers
            .values()
            .map(|t| t.data.len())
            .sum();
        total_anomaly += a_params;
        total_predictor += p_params;

        println!(
            "{:32}  anomaly={:>7}p ({} nodes)  predictor={:>7}p ({} nodes)",
            c.slug,
            a_params,
            anom.graph.as_ref().unwrap().nodes.len(),
            p_params,
            pred.graph.as_ref().unwrap().nodes.len(),
        );
    }
    println!("---");
    println!(
        "built {} location(s) — anomaly Σ={total_anomaly}p, predictor Σ={total_predictor}p",
        to_build.len()
    );
}
