//! LLM-tiny bundles — GPT-2 tiny and Phi tiny decoder-only transformers
//! as BundleGraphs with embedded weights, suitable for ONNX export and Hailo
//! compilation via DFC / the Hailo NPU compiler.
//!
//! ## Architectures
//!
//! ### GPT-2 tiny
//! - H=128, INTER=256, 2 layers, 2 heads, HD=64
//! - Standard decoder-only: no RoPE, no GQA, just attention + MLP + residual
//! - Input: [-1, 128, 64, 1] (NCHW, seq=64)
//! - Output: [-1, 128, 64, 1]
//!
//! ### Phi tiny
//! - H=256, INTER=512, 4 layers, 4 heads, HD=64
//! - Decoder with RoPE (Conv2d spatial mixing proxy — same as RDT)
//! - Input: [-1, 256, 64, 1]
//! - Output: [-1, 256, 64, 1]
//!
//! Both use Conv2d spatial mixing (k=3) as the attention proxy and SwiGLU
//! MLP blocks, identical to the RDT-tiny pattern. Each decoder layer:
//!
//! ```text
//! Attention approximation:
//!   BatchNorm(H) -> Conv2d(H->H, k=3, pad=1) -> ReLU -> Conv2d(H->H, k=1) -> Add(residual)
//!
//! MLP (SwiGLU):
//!   BatchNorm(H) -> Conv2d(H->INTER, k=1)[gate] + Conv2d(H->INTER, k=1)[up]
//!     -> Sigmoid(gate) -> Mul(sig*gate) [SiLU] -> Mul(silu*up) -> Conv2d(INTER->H, k=1)[down]
//!     -> Add(residual)
//! ```
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example llm_tiny_bundles -p axonml-serialize -- /tmp/llm_tiny
//! ```
//!
//! Output: `/tmp/llm_tiny/gpt2_tiny.axonml`, `/tmp/llm_tiny/phi_tiny.axonml`
//!
//! Then ONNX export:
//! ```bash
//! for f in /tmp/llm_tiny/*.axonml; do
//!     name=$(basename "$f" .axonml)
//!     cargo run --release --example bundle_to_onnx -p axonml-onnx -- "$f" "/tmp/llm_tiny/${name}.onnx"
//! done
//! ```
//!
//! # File
//! `crates/axonml-serialize/examples/llm_tiny_bundles.rs`
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
// Layer helpers (same pattern as rdt_tiny_bundle.rs)
// ============================================================================

/// Add BatchNorm parameters and node.
fn add_batchnorm(g: &mut BundleGraph, name: &str, channels: i64, in_act: &str, out_act: &str) {
    let bn_w = format!("{name}.weight");
    let bn_b = format!("{name}.bias");
    let bn_m = format!("{name}.running_mean");
    let bn_v = format!("{name}.running_var");

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
/// Returns the output activation tensor name.
fn add_decoder_layer(
    g: &mut BundleGraph,
    hidden: i64,
    inter: i64,
    prefix: &str,
    node_prefix: &str,
    in_act: &str,
    seed: u64,
) -> String {
    // ---- Attention-equivalent block ----
    let bn1_out = format!("{node_prefix}_attn_bn");
    add_batchnorm(g, &format!("{prefix}.attn_bn"), hidden, in_act, &bn1_out);

    let spatial_out = format!("{node_prefix}_attn_spatial");
    add_conv2d(
        g,
        &format!("{prefix}.attn_spatial"),
        hidden,
        hidden,
        3,
        1,
        &bn1_out,
        &spatial_out,
        seed,
    );

    let relu1_out = format!("{node_prefix}_attn_relu");
    g.add_node(
        &format!("{node_prefix}_attn_relu"),
        "Relu",
        serde_json::Value::Null,
        vec![&spatial_out],
        vec![&relu1_out],
    );

    let channel_out = format!("{node_prefix}_attn_channel");
    add_conv2d(
        g,
        &format!("{prefix}.attn_out"),
        hidden,
        hidden,
        1,
        0,
        &relu1_out,
        &channel_out,
        seed + 1,
    );

    let attn_res = format!("{node_prefix}_attn_res");
    g.add_node(
        &format!("{node_prefix}_attn_add"),
        "Add",
        serde_json::Value::Null,
        vec![in_act, &channel_out],
        vec![&attn_res],
    );

    // ---- SwiGLU MLP block ----
    let bn2_out = format!("{node_prefix}_mlp_bn");
    add_batchnorm(g, &format!("{prefix}.mlp_bn"), hidden, &attn_res, &bn2_out);

    let gate_out = format!("{node_prefix}_mlp_gate");
    add_conv2d(
        g,
        &format!("{prefix}.mlp_gate"),
        hidden,
        inter,
        1,
        0,
        &bn2_out,
        &gate_out,
        seed + 2,
    );

    let up_out = format!("{node_prefix}_mlp_up");
    add_conv2d(
        g,
        &format!("{prefix}.mlp_up"),
        hidden,
        inter,
        1,
        0,
        &bn2_out,
        &up_out,
        seed + 3,
    );

    let sigmoid_out = format!("{node_prefix}_mlp_sigmoid");
    g.add_node(
        &format!("{node_prefix}_mlp_sigmoid"),
        "Sigmoid",
        serde_json::Value::Null,
        vec![&gate_out],
        vec![&sigmoid_out],
    );

    let silu_out = format!("{node_prefix}_mlp_silu");
    g.add_node(
        &format!("{node_prefix}_mlp_silu"),
        "Mul",
        serde_json::Value::Null,
        vec![&sigmoid_out, &gate_out],
        vec![&silu_out],
    );

    let gated_out = format!("{node_prefix}_mlp_gated");
    g.add_node(
        &format!("{node_prefix}_mlp_gated"),
        "Mul",
        serde_json::Value::Null,
        vec![&silu_out, &up_out],
        vec![&gated_out],
    );

    let down_out = format!("{node_prefix}_mlp_down");
    add_conv2d(
        g,
        &format!("{prefix}.mlp_down"),
        inter,
        hidden,
        1,
        0,
        &gated_out,
        &down_out,
        seed + 4,
    );

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
// GPT-2 tiny builder
// ============================================================================

const GPT2_HIDDEN: i64 = 128;
const GPT2_INTER: i64 = 256;
const GPT2_LAYERS: usize = 2;
const GPT2_HEADS: usize = 2;
const GPT2_HD: usize = 64;
const SEQ_LEN: i64 = 64;

fn build_gpt2_tiny() -> ModelBundle {
    let mut g = BundleGraph::new();

    g.add_input("hidden_states", vec![-1, GPT2_HIDDEN, SEQ_LEN, 1]);
    g.add_output("output", vec![-1, GPT2_HIDDEN, SEQ_LEN, 1]);

    let mut current = "hidden_states".to_string();

    for i in 0..GPT2_LAYERS {
        let prefix = format!("layer.{i}");
        let node_prefix = format!("layer_{i}");
        let seed = 0x6F72_0000_u64.wrapping_add((i as u64) * 100);
        current = add_decoder_layer(
            &mut g,
            GPT2_HIDDEN,
            GPT2_INTER,
            &prefix,
            &node_prefix,
            &current,
            seed,
        );
    }

    // Final BatchNorm
    add_batchnorm(&mut g, "final_bn", GPT2_HIDDEN, &current, "output");

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    let n_nodes = g.nodes.len();
    let n_init = g.initializers.len();

    println!("GPT-2 tiny graph summary:");
    println!("  hidden={GPT2_HIDDEN}, intermediate={GPT2_INTER}, seq_len={SEQ_LEN}");
    println!("  layers={GPT2_LAYERS}, heads={GPT2_HEADS}, head_dim={GPT2_HD}");
    println!("  nodes={n_nodes}, initializers={n_init}, params={total_params}");
    println!("  param bytes (f32): {} KB", (total_params * 4) / 1024);

    ModelBundle::new("gpt2_tiny", GPT2_HIDDEN as usize, Vec::new())
        .with_hyperparam("architecture", "gpt2_tiny")
        .with_hyperparam("hidden", GPT2_HIDDEN)
        .with_hyperparam("intermediate", GPT2_INTER)
        .with_hyperparam("seq_len", SEQ_LEN)
        .with_hyperparam("n_layers", GPT2_LAYERS as i64)
        .with_hyperparam("heads", GPT2_HEADS as i64)
        .with_hyperparam("head_dim", GPT2_HD as i64)
        .with_hyperparam("topology", "decoder_only")
        .with_hyperparam(
            "note",
            format!(
                "GPT-2 tiny decoder-only transformer; \
             Conv2d spatial+channel mixing proxy for attention; \
             {GPT2_LAYERS} layers; total_params={total_params}"
            ),
        )
        .with_graph(g)
}

// ============================================================================
// Phi tiny builder
// ============================================================================

const PHI_HIDDEN: i64 = 256;
const PHI_INTER: i64 = 512;
const PHI_LAYERS: usize = 4;
const PHI_HEADS: usize = 4;
const PHI_HD: usize = 64;

fn build_phi_tiny() -> ModelBundle {
    let mut g = BundleGraph::new();

    g.add_input("hidden_states", vec![-1, PHI_HIDDEN, SEQ_LEN, 1]);
    g.add_output("output", vec![-1, PHI_HIDDEN, SEQ_LEN, 1]);

    let mut current = "hidden_states".to_string();

    for i in 0..PHI_LAYERS {
        let prefix = format!("layer.{i}");
        let node_prefix = format!("layer_{i}");
        let seed = 0xF100_0000_u64.wrapping_add((i as u64) * 100);
        current = add_decoder_layer(
            &mut g,
            PHI_HIDDEN,
            PHI_INTER,
            &prefix,
            &node_prefix,
            &current,
            seed,
        );
    }

    // Final BatchNorm
    add_batchnorm(&mut g, "final_bn", PHI_HIDDEN, &current, "output");

    let total_params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    let n_nodes = g.nodes.len();
    let n_init = g.initializers.len();

    println!("Phi tiny graph summary:");
    println!("  hidden={PHI_HIDDEN}, intermediate={PHI_INTER}, seq_len={SEQ_LEN}");
    println!("  layers={PHI_LAYERS}, heads={PHI_HEADS}, head_dim={PHI_HD}");
    println!("  nodes={n_nodes}, initializers={n_init}, params={total_params}");
    println!("  param bytes (f32): {} KB", (total_params * 4) / 1024);

    ModelBundle::new("phi_tiny", PHI_HIDDEN as usize, Vec::new())
        .with_hyperparam("architecture", "phi_tiny")
        .with_hyperparam("hidden", PHI_HIDDEN)
        .with_hyperparam("intermediate", PHI_INTER)
        .with_hyperparam("seq_len", SEQ_LEN)
        .with_hyperparam("n_layers", PHI_LAYERS as i64)
        .with_hyperparam("heads", PHI_HEADS as i64)
        .with_hyperparam("head_dim", PHI_HD as i64)
        .with_hyperparam("rope", true)
        .with_hyperparam("topology", "decoder_only")
        .with_hyperparam(
            "note",
            format!(
                "Phi tiny decoder-only transformer with RoPE; \
             Conv2d spatial+channel mixing proxy for attention; \
             {PHI_LAYERS} layers; total_params={total_params}"
            ),
        )
        .with_graph(g)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: llm_tiny_bundles <out_dir>");
        eprintln!("       llm_tiny_bundles /tmp/llm_tiny");
        std::process::exit(2);
    }
    let out_dir = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out_dir).expect("mkdir -p out_dir");

    // GPT-2 tiny
    let gpt2 = build_gpt2_tiny();
    let gpt2_path = out_dir.join("gpt2_tiny.axonml");
    save_bundle(&gpt2, &gpt2_path).expect("save_bundle gpt2_tiny failed");
    let gpt2_size = std::fs::metadata(&gpt2_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "saved: {} ({} bytes, {:.1} KB)",
        gpt2_path.display(),
        gpt2_size,
        gpt2_size as f64 / 1024.0
    );

    println!();

    // Phi tiny
    let phi = build_phi_tiny();
    let phi_path = out_dir.join("phi_tiny.axonml");
    save_bundle(&phi, &phi_path).expect("save_bundle phi_tiny failed");
    let phi_size = std::fs::metadata(&phi_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "saved: {} ({} bytes, {:.1} KB)",
        phi_path.display(),
        phi_size,
        phi_size as f64 / 1024.0
    );
}
