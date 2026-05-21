//! New-site HVAC model bundles — 5 distinct graph topologies for Hailo + CPU.
//!
//! Each model uses Conv2d with H=1 to represent Conv1d on Hailo silicon.
//! Input shapes are `[-1, n_features, 1, 120]` (NCHW, H=1).
//!
//! ## Models
//!
//! 1. **taylor_greenhouse**  — Multi-branch (3 parallel sensor-group paths → Concat → fusion)
//! 2. **taylor_natorium**    — Deep residual (4 dilated residual blocks)
//! 3. **taylor_chiller**     — Bottleneck (4 squeeze-expand blocks)
//! 4. **peabody_cooling_towers** — Multi-scale depthwise (3 parallel DW conv scales)
//! 5. **peabody_boilers**    — Progressive compression + dilated temporal
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example new_sites_bundle -p axonml-serialize -- /tmp/new_site_bundles
//! ```
//!
//! # File
//! `crates/axonml-serialize/examples/new_sites_bundle.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 29, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::path::PathBuf;

use axonml_serialize::{BundleGraph, ModelBundle, save_bundle};

const SEQ_LEN: i64 = 120;

// ============================================================================
// Kaiming init (deterministic, seeded)
// ============================================================================

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

// ============================================================================
// Layer helpers
// ============================================================================

/// Conv2d (kernel [1,K], stride S, dilation D, group G) + optional BN + optional ReLU.
/// Computes "same" padding for the W dimension: pad = D * (K - 1) / 2.
fn add_conv_bn_relu_full(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    kernel: i64,
    stride: i64,
    dilation: i64,
    group: i64,
    in_act: &str,
    out_prefix: &str,
    bn: bool,
    relu: bool,
    seed: u64,
) -> String {
    let cw = format!("{name}.conv.weight");
    let cb = format!("{name}.conv.bias");
    let pad = dilation * (kernel - 1) / 2;

    // Conv weight: [out_c, in_c / group, 1, K]
    let in_c_per_group = in_c / group;
    let w_n = (out_c * in_c_per_group * 1 * kernel) as usize;
    g.add_initializer(
        &cw,
        vec![out_c, in_c_per_group, 1, kernel],
        init_kaiming(w_n, (in_c_per_group * kernel) as usize, seed),
    );
    g.add_initializer(&cb, vec![out_c], vec![0.0; out_c as usize]);

    let conv_out = format!("{out_prefix}_conv");
    g.add_node(
        &format!("{name}_conv"),
        "Conv2d",
        serde_json::json!({
            "kernel_shape": [1, kernel],
            "strides": [1, stride],
            "pads": [0, pad, 0, pad],
            "dilations": [1, dilation],
            "group": group,
        }),
        vec![in_act, &cw, &cb],
        vec![&conv_out],
    );

    let mut last = conv_out.clone();

    if bn {
        let bn_w = format!("{name}.bn.weight");
        let bn_b = format!("{name}.bn.bias");
        let bn_m = format!("{name}.bn.running_mean");
        let bn_v = format!("{name}.bn.running_var");
        g.add_initializer(&bn_w, vec![out_c], vec![1.0; out_c as usize]);
        g.add_initializer(&bn_b, vec![out_c], vec![0.0; out_c as usize]);
        g.add_initializer(&bn_m, vec![out_c], vec![0.0; out_c as usize]);
        g.add_initializer(&bn_v, vec![out_c], vec![1.0; out_c as usize]);

        let bn_out = format!("{out_prefix}_bn");
        g.add_node(
            &format!("{name}_bn"),
            "BatchNorm",
            serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
            vec![&last, &bn_w, &bn_b, &bn_m, &bn_v],
            vec![&bn_out],
        );
        last = bn_out;
    }

    if relu {
        let relu_out = format!("{out_prefix}_relu");
        g.add_node(
            &format!("{name}_relu"),
            "Relu",
            serde_json::Value::Null,
            vec![&last],
            vec![&relu_out],
        );
        last = relu_out;
    }

    last
}

/// Convenience: Conv + BN + ReLU, stride 1, dilation 1, group 1, kernel K
fn add_cbr(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    kernel: i64,
    in_act: &str,
    out_prefix: &str,
    seed: u64,
) -> String {
    add_conv_bn_relu_full(
        g, name, in_c, out_c, kernel, 1, 1, 1, in_act, out_prefix, true, true, seed,
    )
}

/// Conv + BN (no ReLU) — for pre-residual branches
fn add_cb(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    kernel: i64,
    dilation: i64,
    in_act: &str,
    out_prefix: &str,
    seed: u64,
) -> String {
    add_conv_bn_relu_full(
        g, name, in_c, out_c, kernel, 1, dilation, 1, in_act, out_prefix, true, false, seed,
    )
}

/// 1x1 pointwise Conv + BN + ReLU
fn add_pw_cbr(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    in_act: &str,
    out_prefix: &str,
    seed: u64,
) -> String {
    add_conv_bn_relu_full(
        g, name, in_c, out_c, 1, 1, 1, 1, in_act, out_prefix, true, true, seed,
    )
}

/// 1x1 pointwise Conv + BN (no ReLU)
fn add_pw_cb(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    in_act: &str,
    out_prefix: &str,
    seed: u64,
) -> String {
    add_conv_bn_relu_full(
        g, name, in_c, out_c, 1, 1, 1, 1, in_act, out_prefix, true, false, seed,
    )
}

/// Depthwise Conv + BN + ReLU (group = in_c = out_c)
fn add_dw_cbr(
    g: &mut BundleGraph,
    name: &str,
    channels: i64,
    kernel: i64,
    in_act: &str,
    out_prefix: &str,
    seed: u64,
) -> String {
    add_conv_bn_relu_full(
        g, name, channels, channels, kernel, 1, 1, channels, in_act, out_prefix, true, true, seed,
    )
}

/// Add residual connection (element-wise Add)
fn add_residual(g: &mut BundleGraph, name: &str, a: &str, b: &str, out: &str) {
    g.add_node(name, "Add", serde_json::Value::Null, vec![a, b], vec![out]);
}

/// GAP + Flatten
fn add_gap_flatten(g: &mut BundleGraph, prefix: &str, in_act: &str, out: &str) {
    let gap_out = format!("{prefix}_gap");
    g.add_node(
        &format!("{prefix}_gap"),
        "GlobalAvgPool",
        serde_json::Value::Null,
        vec![in_act],
        vec![&gap_out],
    );
    g.add_node(
        &format!("{prefix}_flatten"),
        "Flatten",
        serde_json::json!({"axis": 1}),
        vec![&gap_out],
        vec![out],
    );
}

/// Gemm head: flat_in → n_out
fn add_gemm_head(
    g: &mut BundleGraph,
    name: &str,
    in_dim: i64,
    out_dim: i64,
    in_act: &str,
    out_name: &str,
    seed: u64,
) {
    let w = format!("{name}.weight");
    let b = format!("{name}.bias");
    g.add_initializer(
        &w,
        vec![out_dim, in_dim],
        init_kaiming((out_dim * in_dim) as usize, in_dim as usize, seed),
    );
    g.add_initializer(&b, vec![out_dim], vec![0.0; out_dim as usize]);
    g.add_node(
        name,
        "Gemm",
        serde_json::json!({"alpha": 1.0, "beta": 1.0, "trans_a": false, "trans_b": true}),
        vec![in_act, &w, &b],
        vec![out_name],
    );
}

// ============================================================================
// Model 1: taylor_greenhouse (multi-branch)
// 14 features, 3 heads x 10 outputs
// ============================================================================

fn build_taylor_greenhouse(seed_base: u64) -> ModelBundle {
    let mut g = BundleGraph::new();
    let n_features: i64 = 14;

    g.add_input("sensor_seq", vec![-1, n_features, 1, SEQ_LEN]);
    g.add_output("head_temp", vec![-1, 10]);
    g.add_output("head_humid", vec![-1, 10]);
    g.add_output("head_env", vec![-1, 10]);

    // We need to split the input into 3 branches by channel.
    // Since ONNX Slice is problematic on Hailo, we use 3 separate Conv layers
    // that each read from the full input. The first conv in each branch acts as
    // a channel selector + feature extractor.

    // Branch 1 (temp sensors, first 4 channels conceptually): Conv[14->32, k3]
    let b1 = add_cbr(
        &mut g,
        "branch1",
        n_features,
        32,
        3,
        "sensor_seq",
        "b1",
        seed_base + 1,
    );

    // Branch 2 (humidity sensors): Conv[14->24, k5]
    let b2 = add_conv_bn_relu_full(
        &mut g,
        "branch2",
        n_features,
        24,
        5,
        1,
        1,
        1,
        "sensor_seq",
        "b2",
        true,
        true,
        seed_base + 2,
    );

    // Branch 3 (environmental): Conv[14->32, k3]
    let b3 = add_cbr(
        &mut g,
        "branch3",
        n_features,
        32,
        3,
        "sensor_seq",
        "b3",
        seed_base + 3,
    );

    // Concat branches (32+24+32=88 channels)
    g.add_node(
        "concat",
        "Concat",
        serde_json::json!({"axis": 1}),
        vec![&b1, &b2, &b3],
        vec!["concat_out"],
    );

    // Fusion layers
    let fuse1 = add_cbr(
        &mut g,
        "fuse1",
        88,
        64,
        3,
        "concat_out",
        "fuse1",
        seed_base + 4,
    );
    let fuse2 = add_conv_bn_relu_full(
        &mut g,
        "fuse2",
        64,
        48,
        5,
        1,
        1,
        1,
        &fuse1,
        "fuse2",
        true,
        true,
        seed_base + 5,
    );

    // GAP + Flatten
    add_gap_flatten(&mut g, "pool", &fuse2, "flat");

    // 3 Gemm heads (48 -> 10 each)
    add_gemm_head(
        &mut g,
        "head_temp",
        48,
        10,
        "flat",
        "head_temp",
        seed_base + 10,
    );
    add_gemm_head(
        &mut g,
        "head_humid",
        48,
        10,
        "flat",
        "head_humid",
        seed_base + 11,
    );
    add_gemm_head(
        &mut g,
        "head_env",
        48,
        10,
        "flat",
        "head_env",
        seed_base + 12,
    );

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new("taylor_greenhouse", n_features as usize, Vec::new())
        .with_hyperparam("seq_len", SEQ_LEN)
        .with_hyperparam("n_features", n_features)
        .with_hyperparam("topology", "multi_branch")
        .with_hyperparam("n_heads", 3)
        .with_hyperparam("n_outputs_per_head", 10)
        .with_hyperparam(
            "note",
            format!("Multi-branch 3-path sensor fusion; total_params={total_params}"),
        )
        .with_graph(g)
}

// ============================================================================
// Model 2: taylor_natorium (deep residual)
// 18 features, 3 heads x 12 outputs
// ============================================================================

fn build_taylor_natorium(seed_base: u64) -> ModelBundle {
    let mut g = BundleGraph::new();
    let n_features: i64 = 18;
    let hidden: i64 = 64;

    g.add_input("sensor_seq", vec![-1, n_features, 1, SEQ_LEN]);
    g.add_output("head_a", vec![-1, 12]);
    g.add_output("head_b", vec![-1, 12]);
    g.add_output("head_c", vec![-1, 12]);

    // Stem: Conv[18->64, k3] + BN + ReLU
    let stem = add_cbr(
        &mut g,
        "stem",
        n_features,
        hidden,
        3,
        "sensor_seq",
        "stem",
        seed_base + 1,
    );

    // 4 residual blocks with increasing dilation (1, 2, 4, 8)
    let dilations = [1, 2, 4, 8];
    let mut last = stem;

    for (i, &d) in dilations.iter().enumerate() {
        let blk = format!("res{i}");
        let seed = seed_base + 10 + (i as u64) * 10;

        // Conv[64->64, k3, dilation=d] + BN + ReLU
        let c1 = add_conv_bn_relu_full(
            &mut g,
            &format!("{blk}_c1"),
            hidden,
            hidden,
            3,
            1,
            d,
            1,
            &last,
            &format!("{blk}_c1"),
            true,
            true,
            seed,
        );

        // Conv[64->64, k3] + BN (no relu yet — add after residual)
        let c2 = add_cb(
            &mut g,
            &format!("{blk}_c2"),
            hidden,
            hidden,
            3,
            1,
            &c1,
            &format!("{blk}_c2"),
            seed + 1,
        );

        // Residual Add
        let add_out = format!("{blk}_add");
        add_residual(&mut g, &format!("{blk}_add"), &last, &c2, &add_out);

        // ReLU after add
        let relu_out = format!("{blk}_out");
        g.add_node(
            &format!("{blk}_relu"),
            "Relu",
            serde_json::Value::Null,
            vec![&add_out],
            vec![&relu_out],
        );

        last = relu_out;
    }

    // GAP + Flatten
    add_gap_flatten(&mut g, "pool", &last, "flat");

    // 3 Gemm heads (64 -> 12 each)
    add_gemm_head(
        &mut g,
        "head_a",
        hidden,
        12,
        "flat",
        "head_a",
        seed_base + 100,
    );
    add_gemm_head(
        &mut g,
        "head_b",
        hidden,
        12,
        "flat",
        "head_b",
        seed_base + 101,
    );
    add_gemm_head(
        &mut g,
        "head_c",
        hidden,
        12,
        "flat",
        "head_c",
        seed_base + 102,
    );

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new("taylor_natorium", n_features as usize, Vec::new())
        .with_hyperparam("seq_len", SEQ_LEN)
        .with_hyperparam("n_features", n_features)
        .with_hyperparam("topology", "deep_residual")
        .with_hyperparam("hidden", hidden)
        .with_hyperparam("n_residual_blocks", 4)
        .with_hyperparam("dilations", serde_json::json!([1, 2, 4, 8]))
        .with_hyperparam("n_heads", 3)
        .with_hyperparam("n_outputs_per_head", 12)
        .with_hyperparam(
            "note",
            format!("Deep residual with dilated convs; total_params={total_params}"),
        )
        .with_graph(g)
}

// ============================================================================
// Model 3: taylor_chiller (bottleneck)
// 22 features, 2 heads: 8+8 outputs
// ============================================================================

fn build_taylor_chiller(seed_base: u64) -> ModelBundle {
    let mut g = BundleGraph::new();
    let n_features: i64 = 22;
    let wide: i64 = 96;
    let narrow: i64 = 32;

    g.add_input("sensor_seq", vec![-1, n_features, 1, SEQ_LEN]);
    g.add_output("head_perf", vec![-1, 8]);
    g.add_output("head_fault", vec![-1, 8]);

    // Stem: Conv[22->96, k3] + BN + ReLU
    let stem = add_cbr(
        &mut g,
        "stem",
        n_features,
        wide,
        3,
        "sensor_seq",
        "stem",
        seed_base + 1,
    );

    // 4 bottleneck blocks
    let mut last = stem;
    for i in 0..4 {
        let blk = format!("bneck{i}");
        let seed = seed_base + 10 + (i as u64) * 10;

        // Conv1x1[96->32] + BN + ReLU (squeeze)
        let c1 = add_pw_cbr(
            &mut g,
            &format!("{blk}_sq"),
            wide,
            narrow,
            &last,
            &format!("{blk}_sq"),
            seed,
        );

        // Conv[32->32, k5] + BN + ReLU
        let c2 = add_conv_bn_relu_full(
            &mut g,
            &format!("{blk}_mid"),
            narrow,
            narrow,
            5,
            1,
            1,
            1,
            &c1,
            &format!("{blk}_mid"),
            true,
            true,
            seed + 1,
        );

        // Conv1x1[32->96] + BN (no relu — add after residual)
        let c3 = add_pw_cb(
            &mut g,
            &format!("{blk}_ex"),
            narrow,
            wide,
            &c2,
            &format!("{blk}_ex"),
            seed + 2,
        );

        // Residual Add
        let add_out = format!("{blk}_add");
        add_residual(&mut g, &format!("{blk}_add"), &last, &c3, &add_out);

        // ReLU after add
        let relu_out = format!("{blk}_out");
        g.add_node(
            &format!("{blk}_relu"),
            "Relu",
            serde_json::Value::Null,
            vec![&add_out],
            vec![&relu_out],
        );

        last = relu_out;
    }

    // GAP + Flatten
    add_gap_flatten(&mut g, "pool", &last, "flat");

    // 2 Gemm heads (96 -> 8 each)
    add_gemm_head(
        &mut g,
        "head_perf",
        wide,
        8,
        "flat",
        "head_perf",
        seed_base + 100,
    );
    add_gemm_head(
        &mut g,
        "head_fault",
        wide,
        8,
        "flat",
        "head_fault",
        seed_base + 101,
    );

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new("taylor_chiller", n_features as usize, Vec::new())
        .with_hyperparam("seq_len", SEQ_LEN)
        .with_hyperparam("n_features", n_features)
        .with_hyperparam("topology", "bottleneck")
        .with_hyperparam("wide", wide)
        .with_hyperparam("narrow", narrow)
        .with_hyperparam("n_bottleneck_blocks", 4)
        .with_hyperparam("n_heads", 2)
        .with_hyperparam("n_outputs_per_head", 8)
        .with_hyperparam(
            "note",
            format!("Bottleneck squeeze-expand residual; total_params={total_params}"),
        )
        .with_graph(g)
}

// ============================================================================
// Model 4: peabody_cooling_towers (multi-scale depthwise)
// 16 features, 2 heads x 10 outputs
// ============================================================================

fn build_peabody_cooling_towers(seed_base: u64) -> ModelBundle {
    let mut g = BundleGraph::new();
    let n_features: i64 = 16;
    let hidden: i64 = 72;

    g.add_input("sensor_seq", vec![-1, n_features, 1, SEQ_LEN]);
    g.add_output("head_eff", vec![-1, 10]);
    g.add_output("head_health", vec![-1, 10]);

    // Stem: Conv[16->72, k3] + BN + ReLU
    let stem = add_cbr(
        &mut g,
        "stem",
        n_features,
        hidden,
        3,
        "sensor_seq",
        "stem",
        seed_base + 1,
    );

    // 3 multi-scale blocks
    let kernels = [3i64, 7, 15];
    let mut last = stem;

    for i in 0..3 {
        let blk = format!("ms{i}");
        let seed = seed_base + 10 + (i as u64) * 10;

        // 3 parallel depthwise Conv [groups=72, k=3/7/15]
        let dw_outs: Vec<String> = kernels
            .iter()
            .enumerate()
            .map(|(j, &k)| {
                add_dw_cbr(
                    &mut g,
                    &format!("{blk}_dw{j}"),
                    hidden,
                    k,
                    &last,
                    &format!("{blk}_dw{j}"),
                    seed + j as u64,
                )
            })
            .collect();

        // Concat(72*3=216)
        let concat_out = format!("{blk}_cat");
        let dw_refs: Vec<&str> = dw_outs.iter().map(|s| s.as_str()).collect();
        g.add_node(
            &format!("{blk}_concat"),
            "Concat",
            serde_json::json!({"axis": 1}),
            dw_refs,
            vec![&concat_out],
        );

        // Pointwise Conv[216->72] + BN + ReLU
        let pw = add_pw_cbr(
            &mut g,
            &format!("{blk}_pw"),
            hidden * 3,
            hidden,
            &concat_out,
            &format!("{blk}_pw"),
            seed + 5,
        );

        // Residual Add + (no extra relu — pw already has relu, Add preserves it)
        let add_out = format!("{blk}_out");
        add_residual(&mut g, &format!("{blk}_add"), &last, &pw, &add_out);

        last = add_out;
    }

    // Final reduction: Conv[72->48] + BN + ReLU
    let red = add_cbr(
        &mut g,
        "reduce",
        hidden,
        48,
        3,
        &last,
        "reduce",
        seed_base + 50,
    );

    // GAP + Flatten
    add_gap_flatten(&mut g, "pool", &red, "flat");

    // 2 Gemm heads (48 -> 10 each)
    add_gemm_head(
        &mut g,
        "head_eff",
        48,
        10,
        "flat",
        "head_eff",
        seed_base + 100,
    );
    add_gemm_head(
        &mut g,
        "head_health",
        48,
        10,
        "flat",
        "head_health",
        seed_base + 101,
    );

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new("peabody_cooling_towers", n_features as usize, Vec::new())
        .with_hyperparam("seq_len", SEQ_LEN)
        .with_hyperparam("n_features", n_features)
        .with_hyperparam("topology", "multi_scale_depthwise")
        .with_hyperparam("hidden", hidden)
        .with_hyperparam("dw_kernels", serde_json::json!([3, 7, 15]))
        .with_hyperparam("n_ms_blocks", 3)
        .with_hyperparam("n_heads", 2)
        .with_hyperparam("n_outputs_per_head", 10)
        .with_hyperparam(
            "note",
            format!("Multi-scale depthwise separable; total_params={total_params}"),
        )
        .with_graph(g)
}

// ============================================================================
// Model 5: peabody_boilers (progressive compression + dilated temporal)
// 20 features, 3 heads x 10 outputs
// ============================================================================

fn build_peabody_boilers(seed_base: u64) -> ModelBundle {
    let mut g = BundleGraph::new();
    let n_features: i64 = 20;
    let wide: i64 = 80;
    let narrow: i64 = 24;

    g.add_input("sensor_seq", vec![-1, n_features, 1, SEQ_LEN]);
    g.add_output("head_eff", vec![-1, 10]);
    g.add_output("head_safety", vec![-1, 10]);
    g.add_output("head_maint", vec![-1, 10]);

    // Stem: Conv[20->80, k5] + BN + ReLU
    let stem = add_conv_bn_relu_full(
        &mut g,
        "stem",
        n_features,
        wide,
        5,
        1,
        1,
        1,
        "sensor_seq",
        "stem",
        true,
        true,
        seed_base + 1,
    );

    // 3 bottleneck blocks (80->24->80)
    let mut last = stem;
    for i in 0..3 {
        let blk = format!("bneck{i}");
        let seed = seed_base + 10 + (i as u64) * 10;

        // Conv1x1[80->24] + BN + ReLU
        let c1 = add_pw_cbr(
            &mut g,
            &format!("{blk}_sq"),
            wide,
            narrow,
            &last,
            &format!("{blk}_sq"),
            seed,
        );

        // Conv[24->24, k5] + BN + ReLU
        let c2 = add_conv_bn_relu_full(
            &mut g,
            &format!("{blk}_mid"),
            narrow,
            narrow,
            5,
            1,
            1,
            1,
            &c1,
            &format!("{blk}_mid"),
            true,
            true,
            seed + 1,
        );

        // Conv1x1[24->80] + BN (no relu)
        let c3 = add_pw_cb(
            &mut g,
            &format!("{blk}_ex"),
            narrow,
            wide,
            &c2,
            &format!("{blk}_ex"),
            seed + 2,
        );

        // Residual + ReLU
        let add_out = format!("{blk}_add");
        add_residual(&mut g, &format!("{blk}_add"), &last, &c3, &add_out);
        let relu_out = format!("{blk}_out");
        g.add_node(
            &format!("{blk}_relu"),
            "Relu",
            serde_json::Value::Null,
            vec![&add_out],
            vec![&relu_out],
        );
        last = relu_out;
    }

    // Dilated temporal: Conv[80->80, k7, d=2] + BN + ReLU
    let dt1 = add_conv_bn_relu_full(
        &mut g,
        "dil1",
        wide,
        wide,
        7,
        1,
        2,
        1,
        &last,
        "dil1",
        true,
        true,
        seed_base + 50,
    );

    // Conv[80->64, k5, d=2] + BN + ReLU
    let dt2 = add_conv_bn_relu_full(
        &mut g,
        "dil2",
        wide,
        64,
        5,
        1,
        2,
        1,
        &dt1,
        "dil2",
        true,
        true,
        seed_base + 51,
    );

    // GAP + Flatten
    add_gap_flatten(&mut g, "pool", &dt2, "flat");

    // 3 Gemm heads (64 -> 10 each)
    add_gemm_head(
        &mut g,
        "head_eff",
        64,
        10,
        "flat",
        "head_eff",
        seed_base + 100,
    );
    add_gemm_head(
        &mut g,
        "head_safety",
        64,
        10,
        "flat",
        "head_safety",
        seed_base + 101,
    );
    add_gemm_head(
        &mut g,
        "head_maint",
        64,
        10,
        "flat",
        "head_maint",
        seed_base + 102,
    );

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new("peabody_boilers", n_features as usize, Vec::new())
        .with_hyperparam("seq_len", SEQ_LEN)
        .with_hyperparam("n_features", n_features)
        .with_hyperparam("topology", "progressive_compression_dilated")
        .with_hyperparam("wide", wide)
        .with_hyperparam("narrow", narrow)
        .with_hyperparam("n_bottleneck_blocks", 3)
        .with_hyperparam("n_heads", 3)
        .with_hyperparam("n_outputs_per_head", 10)
        .with_hyperparam(
            "note",
            format!("Progressive compression + dilated temporal; total_params={total_params}"),
        )
        .with_graph(g)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: new_sites_bundle <out_dir>");
        eprintln!("       new_sites_bundle /tmp/new_site_bundles");
        std::process::exit(2);
    }
    let out_dir = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out_dir).expect("mkdir -p out_dir");

    let models: Vec<(&str, ModelBundle)> = vec![
        ("taylor_greenhouse", build_taylor_greenhouse(0xbeef_0001)),
        ("taylor_natorium", build_taylor_natorium(0xbeef_0002)),
        ("taylor_chiller", build_taylor_chiller(0xbeef_0003)),
        (
            "peabody_cooling_towers",
            build_peabody_cooling_towers(0xbeef_0004),
        ),
        ("peabody_boilers", build_peabody_boilers(0xbeef_0005)),
    ];

    for (slug, bundle) in &models {
        let path = out_dir.join(format!("{slug}.axonml"));
        save_bundle(bundle, &path).expect("save_bundle failed");

        let g = bundle.graph.as_ref().unwrap();
        let n_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
        let n_nodes = g.nodes.len();
        let n_init = g.initializers.len();
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

        println!(
            "{slug:30}  params={n_params:>8}  nodes={n_nodes:>3}  initializers={n_init:>3}  file={size:>10} bytes"
        );
    }
    println!("---");
    println!("built {} bundles in {}", models.len(), out_dir.display());
}
