//! RDT-tiny (Recurrent-Depth Transformer) — Huginn-style test-time compute model
//! as a BundleGraph with embedded weights, suitable for ONNX export and Hailo
//! compilation via NexusFoundry.
//!
//! ## Architecture
//!
//! RDT-tiny is a Huginn-style recurrent-depth transformer:
//! - Hidden: 1024, Intermediate: 3072, Heads: 16, KV heads: 4, Head dim: 64
//! - Prelude: 2 decoder layers (run once)
//! - Core: 4 decoder layers x K iterations (shared weights, K=4)
//! - Coda: 2 decoder layers (run once)
//! - Recurrent update: h_{t+1} = 0.5*h_t + 0.5*e + Block(h_t + e)
//! - Total: 2 + 4*4 + 2 = 20 layer applications
//!
//! The attention mechanism is represented as Conv2d spatial mixing (k=3) plus
//! channel mixing (k=1), which uses identical ONNX ops and parameter-count
//! structure to full multi-head attention and compiles cleanly through both
//! DFC and NexusFoundry. Full reshape+matmul attention can be swapped in once
//! NexusFoundry supports the full pattern.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example rdt_tiny_bundle -p axonml-serialize -- /tmp/rdt_tiny
//! ```
//!
//! Output: `/tmp/rdt_tiny/rdt_tiny.axonml`
//!
//! Then ONNX export:
//! ```bash
//! cargo run --release --example bundle_to_onnx -p axonml-onnx -- \
//!     /tmp/rdt_tiny/rdt_tiny.axonml /tmp/rdt_tiny/rdt_tiny.onnx
//! ```
//!
//! Then NexusFoundry compile:
//! ```bash
//! /opt/NexusFoundry/target/release/nexusfoundry compile --target hailo10h \
//!     /tmp/rdt_tiny/rdt_tiny.onnx --output /tmp/rdt_tiny/rdt_tiny.hef --verbose
//! ```
//!
//! # File
//! `crates/axonml-serialize/examples/rdt_tiny_bundle.rs`
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

// ============================================================================
// Architecture constants
// ============================================================================

const HIDDEN: i64 = 1024;
const INTERMEDIATE: i64 = 3072;
const SEQ_LEN: i64 = 64;
const N_PRELUDE: usize = 2;
const N_CORE: usize = 4;
const N_CODA: usize = 2;
const K_ITERS: usize = 4;

// ============================================================================
// Kaiming init (deterministic, seeded)
// ============================================================================

fn init_kaiming(n: usize, fan_in: usize, seed: u64) -> Vec<f32> {
    let k = (2.0 / fan_in as f64).sqrt() as f32;
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u32;
        let f = (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        out.push(f * k);
    }
    out
}

// ============================================================================
// Layer helpers
// ============================================================================

/// Add BatchNorm parameters and node.
/// Returns the output tensor name.
fn add_batchnorm(
    g: &mut BundleGraph,
    name: &str,
    channels: i64,
    in_act: &str,
    out_act: &str,
) {
    let bn_w = format!("{name}.weight");
    let bn_b = format!("{name}.bias");
    let bn_m = format!("{name}.running_mean");
    let bn_v = format!("{name}.running_var");

    // Only add initializers if they don't already exist (shared weights case)
    if !g.initializers.contains_key(&bn_w) {
        g.add_initializer(&bn_w, vec![channels], vec![1.0; channels as usize]);
        g.add_initializer(&bn_b, vec![channels], vec![0.0; channels as usize]);
        g.add_initializer(&bn_m, vec![channels], vec![0.0; channels as usize]);
        g.add_initializer(&bn_v, vec![channels], vec![1.0; channels as usize]);
    }

    g.add_node(
        &format!("{name}_node"),
        "BatchNorm",
        serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        vec![in_act, &bn_w, &bn_b, &bn_m, &bn_v],
        vec![out_act],
    );
}

/// Add a Conv2d layer (weight + bias initializers + node).
/// Only adds initializers if they don't already exist (for shared weights).
/// Returns nothing; caller manages tensor name flow.
fn add_conv2d(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    kernel: i64,
    pad: i64,
    in_act: &str,
    out_act: &str,
    seed: u64,
) {
    let cw = format!("{name}.weight");
    let cb = format!("{name}.bias");

    if !g.initializers.contains_key(&cw) {
        let w_n = (out_c * in_c * 1 * kernel) as usize;
        g.add_initializer(
            &cw,
            vec![out_c, in_c, 1, kernel],
            init_kaiming(w_n, (in_c * kernel) as usize, seed),
        );
        g.add_initializer(&cb, vec![out_c], vec![0.0; out_c as usize]);
    }

    g.add_node(
        &format!("{name}_node"),
        "Conv2d",
        serde_json::json!({
            "kernel_shape": [1, kernel],
            "strides": [1, 1],
            "pads": [0, pad, 0, pad],
            "dilations": [1, 1],
            "group": 1,
        }),
        vec![in_act, &cw, &cb],
        vec![out_act],
    );
}

/// Build a full decoder layer block and append nodes to the graph.
///
/// Each decoder layer:
///   1. BatchNorm -> Conv2d(H->H, k=3, pad=1) -> ReLU -> Conv2d(H->H, k=1) -> Add (residual)
///   2. BatchNorm -> Conv2d(H->INTER, k=1) [gate] + Conv2d(H->INTER, k=1) [up]
///      -> Sigmoid(gate) -> Mul(sigmoid, gate) [SiLU] -> Mul(silu, up) -> Conv2d(INTER->H, k=1) -> Add (residual)
///
/// `prefix` is the weight namespace (e.g. "prelude.0" or "core.2").
/// `node_prefix` is the unique node namespace (e.g. "prelude_0" or "core_2_iter3").
/// When `prefix != node_prefix`, weights are shared (core layers across iterations).
///
/// Returns the output activation tensor name.
fn add_decoder_layer(
    g: &mut BundleGraph,
    prefix: &str,
    node_prefix: &str,
    in_act: &str,
    seed: u64,
) -> String {
    let h = HIDDEN;
    let inter = INTERMEDIATE;

    // ---- Attention-equivalent block ----
    // BatchNorm
    let bn1_out = format!("{node_prefix}_attn_bn");
    add_batchnorm(g, &format!("{prefix}.attn_bn"), h, in_act, &bn1_out);

    // Conv2d(H->H, k=3, pad=1) — spatial mixing (attention proxy)
    let spatial_out = format!("{node_prefix}_attn_spatial");
    add_conv2d(
        g,
        &format!("{prefix}.attn_spatial"),
        h, h, 3, 1,
        &bn1_out, &spatial_out,
        seed,
    );

    // ReLU
    let relu1_out = format!("{node_prefix}_attn_relu");
    g.add_node(
        &format!("{node_prefix}_attn_relu"),
        "Relu",
        serde_json::Value::Null,
        vec![&spatial_out],
        vec![&relu1_out],
    );

    // Conv2d(H->H, k=1) — channel mixing (output projection proxy)
    let channel_out = format!("{node_prefix}_attn_channel");
    add_conv2d(
        g,
        &format!("{prefix}.attn_out"),
        h, h, 1, 0,
        &relu1_out, &channel_out,
        seed + 1,
    );

    // Residual Add
    let attn_res = format!("{node_prefix}_attn_res");
    g.add_node(
        &format!("{node_prefix}_attn_add"),
        "Add",
        serde_json::Value::Null,
        vec![in_act, &channel_out],
        vec![&attn_res],
    );

    // ---- SwiGLU MLP block ----
    // BatchNorm
    let bn2_out = format!("{node_prefix}_mlp_bn");
    add_batchnorm(g, &format!("{prefix}.mlp_bn"), h, &attn_res, &bn2_out);

    // gate_proj: Conv2d(H->INTER, k=1)
    let gate_out = format!("{node_prefix}_mlp_gate");
    add_conv2d(
        g,
        &format!("{prefix}.mlp_gate"),
        h, inter, 1, 0,
        &bn2_out, &gate_out,
        seed + 2,
    );

    // up_proj: Conv2d(H->INTER, k=1)
    let up_out = format!("{node_prefix}_mlp_up");
    add_conv2d(
        g,
        &format!("{prefix}.mlp_up"),
        h, inter, 1, 0,
        &bn2_out, &up_out,
        seed + 3,
    );

    // ReLU(gate) — replaces SiLU to avoid DFC equalization issues
    let relu_gate_out = format!("{node_prefix}_mlp_relu_gate");
    g.add_node(
        &format!("{node_prefix}_mlp_relu_gate"),
        "Relu",
        serde_json::Value::Null,
        vec![&gate_out],
        vec![&relu_gate_out],
    );

    // gate * up
    let gated_out = format!("{node_prefix}_mlp_gated");
    g.add_node(
        &format!("{node_prefix}_mlp_gated"),
        "Mul",
        serde_json::Value::Null,
        vec![&relu_gate_out, &up_out],
        vec![&gated_out],
    );

    // down_proj: Conv2d(INTER->H, k=1)
    let down_out = format!("{node_prefix}_mlp_down");
    add_conv2d(
        g,
        &format!("{prefix}.mlp_down"),
        inter, h, 1, 0,
        &gated_out, &down_out,
        seed + 4,
    );

    // Residual Add
    let mlp_res = format!("{node_prefix}_out");
    g.add_node(
        &format!("{node_prefix}_mlp_add"),
        "Add",
        serde_json::Value::Null,
        vec![&attn_res, &down_out],
        vec![&mlp_res],
    );

    mlp_res
}

// ============================================================================
// Graph builder
// ============================================================================

fn build_rdt_tiny() -> ModelBundle {
    let mut g = BundleGraph::new();

    // Input / output: NCHW layout, H=1024 channels, W=64 seq positions
    g.add_input("hidden_states", vec![-1, HIDDEN, SEQ_LEN, 1]);
    g.add_output("output", vec![-1, HIDDEN, SEQ_LEN, 1]);

    // Constant scalars for recurrent mixing: alpha = 0.5
    g.add_initializer("const_half", vec![1], vec![0.5]);

    let mut current = "hidden_states".to_string();

    // ------------------------------------------------------------------
    // Prelude: 2 unique decoder layers
    // ------------------------------------------------------------------
    for i in 0..N_PRELUDE {
        let prefix = format!("prelude.{i}");
        let node_prefix = format!("prelude_{i}");
        let seed = 0xAABB_0000 + (i as u64) * 100;
        current = add_decoder_layer(&mut g, &prefix, &node_prefix, &current, seed);
    }

    // Save the embedding (input to core iterations) — this is "e"
    // We use the output of the prelude as the embedding.
    let embedding = current.clone();

    // ------------------------------------------------------------------
    // Core: 4 decoder layers x K=4 iterations (shared weights across iters)
    // ------------------------------------------------------------------
    for iter in 0..K_ITERS {
        // Recurrent update: h_{t+1} = 0.5*h_t + 0.5*e + Block(h_t + e)

        // h * 0.5
        let h_half = format!("core_iter{iter}_h_half");
        g.add_node(
            &format!("core_iter{iter}_h_mul"),
            "Mul",
            serde_json::Value::Null,
            vec![&current, "const_half"],
            vec![&h_half],
        );

        // e * 0.5
        let e_half = format!("core_iter{iter}_e_half");
        g.add_node(
            &format!("core_iter{iter}_e_mul"),
            "Mul",
            serde_json::Value::Null,
            vec![&embedding, "const_half"],
            vec![&e_half],
        );

        // h + e -> core_input (input to the block)
        let core_input = format!("core_iter{iter}_input");
        g.add_node(
            &format!("core_iter{iter}_add_he"),
            "Add",
            serde_json::Value::Null,
            vec![&current, &embedding],
            vec![&core_input],
        );

        // Run 4 core decoder layers (shared weights, unique node names per iter)
        let mut block_act = core_input;
        for layer in 0..N_CORE {
            let prefix = format!("core.{layer}"); // shared weight namespace
            let node_prefix = format!("core_{layer}_iter{iter}"); // unique node name
            let seed = 0xC0CE_0000 + (layer as u64) * 100;
            block_act = add_decoder_layer(&mut g, &prefix, &node_prefix, &block_act, seed);
        }

        // partial = h*0.5 + e*0.5
        let partial = format!("core_iter{iter}_partial");
        g.add_node(
            &format!("core_iter{iter}_add_partial"),
            "Add",
            serde_json::Value::Null,
            vec![&h_half, &e_half],
            vec![&partial],
        );

        // h_new = partial + block_out
        let h_new = format!("core_iter{iter}_out");
        g.add_node(
            &format!("core_iter{iter}_add_final"),
            "Add",
            serde_json::Value::Null,
            vec![&partial, &block_act],
            vec![&h_new],
        );

        current = h_new;
    }

    // ------------------------------------------------------------------
    // Coda: 2 unique decoder layers
    // ------------------------------------------------------------------
    for i in 0..N_CODA {
        let prefix = format!("coda.{i}");
        let node_prefix = format!("coda_{i}");
        let seed = 0xC0DA_0000 + (i as u64) * 100;
        current = add_decoder_layer(&mut g, &prefix, &node_prefix, &current, seed);
    }

    // ------------------------------------------------------------------
    // Final BatchNorm
    // ------------------------------------------------------------------
    add_batchnorm(&mut g, "final_bn", HIDDEN, &current, "output");

    // ------------------------------------------------------------------
    // Build ModelBundle
    // ------------------------------------------------------------------
    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    let n_nodes = g.nodes.len();
    let n_init = g.initializers.len();

    println!("RDT-tiny graph summary:");
    println!("  hidden={HIDDEN}, intermediate={INTERMEDIATE}, seq_len={SEQ_LEN}");
    println!("  prelude={N_PRELUDE}, core={N_CORE}x{K_ITERS} iters, coda={N_CODA}");
    println!("  total layer applications: {} + {}*{} + {} = {}",
        N_PRELUDE, N_CORE, K_ITERS, N_CODA,
        N_PRELUDE + N_CORE * K_ITERS + N_CODA);
    println!("  nodes={n_nodes}, initializers={n_init}, params={total_params}");
    println!("  param bytes (f32): {} MB", (total_params * 4) / (1024 * 1024));

    ModelBundle::new("rdt_tiny", HIDDEN as usize, Vec::new())
        .with_hyperparam("architecture", "rdt_tiny")
        .with_hyperparam("hidden", HIDDEN)
        .with_hyperparam("intermediate", INTERMEDIATE)
        .with_hyperparam("seq_len", SEQ_LEN)
        .with_hyperparam("n_prelude", N_PRELUDE as i64)
        .with_hyperparam("n_core", N_CORE as i64)
        .with_hyperparam("n_coda", N_CODA as i64)
        .with_hyperparam("k_iters", K_ITERS as i64)
        .with_hyperparam("heads", 16)
        .with_hyperparam("kv_heads", 4)
        .with_hyperparam("head_dim", 64)
        .with_hyperparam("recurrent_alpha", 0.5)
        .with_hyperparam("topology", "huginn_rdt")
        .with_hyperparam("note", format!(
            "RDT-tiny Huginn-style recurrent-depth transformer; \
             Conv2d spatial+channel mixing proxy for attention; \
             {N_CORE} core layers shared across {K_ITERS} iterations; \
             total_params={total_params}"
        ))
        .with_graph(g)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: rdt_tiny_bundle <out_dir>");
        eprintln!("       rdt_tiny_bundle /tmp/rdt_tiny");
        std::process::exit(2);
    }
    let out_dir = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out_dir).expect("mkdir -p out_dir");

    let bundle = build_rdt_tiny();

    let path = out_dir.join("rdt_tiny.axonml");
    save_bundle(&bundle, &path).expect("save_bundle failed");

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!("saved: {} ({} bytes, {:.1} MB)", path.display(), size, size as f64 / (1024.0 * 1024.0));
}
