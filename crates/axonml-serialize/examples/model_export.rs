//! NexusFoundry Model Export — Generate fresh .axonml bundles for ALL model families.
//!
//! Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//! Date: 2026-04-30
//!
//! Creates proper BundleGraph-format .axonml files with Kaiming-initialized
//! weights for every AutomataNexus model architecture. These replace legacy
//! state-dict-only checkpoints and can be exported to ONNX via bundle_to_onnx,
//! then compiled to HEF via NexusFoundry.
//!
//! Usage:
//!   cargo run --release --example model_export -p axonml-serialize -- <arch> <output_dir>
//!   cargo run --release --example model_export -p axonml-serialize -- all /tmp/models
//!
//! Supported architectures:
//!   HVAC/Apollo:  boreas, aquilo, naiad, vulcan, zephyrus, colossus, gaia, apollo
//!   Biometrics:   mnemosyne, argus, ariadne
//!   NLP/Audio:    sentinel, nabu, birdclef, atlas
//!   Transformer:  trident-tcn, qwen3-4l, rdt-tiny
//!   Genomics:     olympus-position, olympus-pvalue, olympus-meta,
//!                 olympus-gene, olympus-npz, olympus-combined

use axonml_serialize::{BundleGraph, ModelBundle, save_bundle};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: model_export <arch|all> <output_dir>");
        eprintln!("architectures: boreas aquilo naiad vulcan zephyrus colossus gaia apollo");
        eprintln!("               mnemosyne argus ariadne sentinel nabu birdclef atlas");
        eprintln!("               trident-tcn qwen3-4l rdt-tiny");
        eprintln!("               olympus-position olympus-pvalue olympus-meta");
        eprintln!("               olympus-gene olympus-npz olympus-combined");
        eprintln!("               all");
        std::process::exit(2);
    }
    let arch = &args[1];
    let out_dir = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out_dir).unwrap();

    let archs: Vec<&str> = if arch == "all" {
        vec![
            "boreas",
            "aquilo",
            "naiad",
            "vulcan",
            "zephyrus",
            "colossus",
            "gaia",
            "apollo",
            "mnemosyne",
            "argus",
            "ariadne",
            "sentinel",
            "nabu",
            "birdclef",
            "atlas",
            "qwen3-4l",
            "rdt-tiny",
            "gpt2-tiny",
            "bert-tiny",
            "phi-tiny",
            "mistral-tiny",
            "mamba-ssm",
            "trident",
            "trident-1b",
            "trident-tcn",
            "hydra",
            "hydra-300m",
            "chimera",
            "chimera-2b",
            "olympus-position",
            "olympus-pvalue",
            "olympus-meta",
            "olympus-gene",
            "olympus-npz",
            "olympus-combined",
        ]
    } else {
        vec![arch.as_str()]
    };

    for name in &archs {
        let bundle = build_model(name);
        let path = out_dir.join(format!("{name}.axonml"));
        save_bundle(&bundle, &path).expect("save_bundle failed");
        let params: usize = bundle
            .graph
            .as_ref()
            .map(|g| g.initializers.values().map(|t| t.data.len()).sum())
            .unwrap_or(0);
        eprintln!("  {name}: {params} params → {}", path.display());
    }
    eprintln!("done: {} models exported", archs.len());
}

fn build_model(name: &str) -> ModelBundle {
    match name {
        // Apollo/HVAC diagnostic suite — TCN-based temporal models
        "boreas" => {
            build_hvac_diagnostic("boreas", "Refrigeration Diagnostics", 7, 16, 1, 5, 8, 80)
        }
        "aquilo" => {
            build_hvac_diagnostic("aquilo", "Cooling Tower Diagnostics", 6, 12, 1, 4, 6, 60)
        }
        "naiad" => build_hvac_diagnostic("naiad", "Water Systems Diagnostics", 8, 14, 1, 6, 8, 80),
        "vulcan" => {
            build_hvac_diagnostic("vulcan", "Boiler/Furnace Diagnostics", 9, 18, 1, 7, 10, 80)
        }
        "zephyrus" => {
            build_hvac_diagnostic("zephyrus", "Air Handler Diagnostics", 11, 20, 1, 8, 11, 120)
        }
        "colossus" => build_hvac_diagnostic(
            "colossus",
            "Chiller Plant Diagnostics",
            12,
            22,
            1,
            9,
            12,
            120,
        ),
        "gaia" => build_hvac_diagnostic("gaia", "Geothermal Diagnostics", 10, 16, 1, 6, 8, 80),
        "apollo" => {
            build_hvac_diagnostic("apollo", "Unified HVAC Diagnostics", 15, 24, 1, 10, 14, 120)
        }

        // Biometrics — vision CNN
        "mnemosyne" => build_vision_cnn("mnemosyne", "Face Recognition", 3, 112, 112, 128, 6),
        "argus" => build_vision_cnn("argus", "Iris Recognition", 1, 64, 512, 64, 4),
        "ariadne" => build_vision_cnn("ariadne", "Fingerprint Recognition", 1, 128, 128, 128, 5),

        // NLP/Audio/Other
        "sentinel" => build_mlp("sentinel", "Equipment Health MLP", 32, &[64, 32, 16], 1),
        "nabu" => build_mlp("nabu", "Akkadian NLP Classifier", 256, &[128, 64], 42),
        "birdclef" => build_vision_cnn(
            "birdclef",
            "BirdClef SedNet 234-species",
            1,
            128,
            256,
            234,
            5,
        ),
        "atlas" => build_mlp("atlas", "Toyota GR Racing Predictor", 64, &[128, 64, 32], 8),

        // Transformer / LLM
        // Standard LLM architectures
        "qwen3-4l" => build_transformer("qwen3-4l", "Qwen3 4-Layer Custom Attention", 32, 64, 4),
        "rdt-tiny" => build_transformer("rdt-tiny", "RDT-tiny Huginn Recurrent-Depth", 64, 128, 2),
        "gpt2-tiny" => build_transformer("gpt2-tiny", "GPT-2 Tiny Decoder-Only", 128, 256, 4),
        "bert-tiny" => build_transformer("bert-tiny", "BERT Tiny Encoder", 128, 256, 2),
        "phi-tiny" => build_transformer("phi-tiny", "Phi Tiny Dense Attention", 128, 256, 4),
        "mistral-tiny" => build_transformer("mistral-tiny", "Mistral Tiny GQA", 128, 256, 4),
        "mamba-ssm" => build_tcn_transformer("mamba-ssm", "Mamba SSM Conv1d Backbone", 64, 128, 6),

        // NOVEL AutomataNexus architectures (Andrew Jewell Sr.)
        "trident" => build_transformer(
            "trident",
            "Trident 1.58-bit Ternary LM (BitNet b1.58)",
            64,
            128,
            4,
        ),
        "trident-1b" => build_transformer(
            "trident-1b",
            "Trident 1B Ternary (d=512, 12L)",
            512,
            1024,
            12,
        ),
        "hydra" => build_hydra("hydra", "Hydra SSM+Windowed Attention Hybrid", 64, 128, 4),
        "hydra-300m" => build_hydra(
            "hydra-300m",
            "Hydra 300M (d=768, 12L, S6+LocalAttn)",
            768,
            1536,
            12,
        ),
        "chimera" => build_chimera(
            "chimera",
            "Chimera MoE+DiffAttn (8 experts, top-2)",
            64,
            128,
            4,
            8,
        ),
        "chimera-2b" => build_chimera(
            "chimera-2b",
            "Chimera 2B (d=512, 16L, 8 experts)",
            512,
            1024,
            16,
            8,
        ),
        "trident-tcn" => {
            build_tcn_transformer("trident-tcn", "Trident TCN 1.58-bit Temporal", 32, 64, 4)
        }

        // Alzheimer's OLYMPUS genomic experts
        "olympus-position" => build_mlp(
            "olympus-position",
            "OLYMPUS Position Expert",
            128,
            &[256, 128, 64],
            2,
        ),
        "olympus-pvalue" => build_mlp(
            "olympus-pvalue",
            "OLYMPUS P-value Expert",
            64,
            &[128, 64],
            2,
        ),
        "olympus-meta" => build_mlp("olympus-meta", "OLYMPUS Meta Expert", 96, &[192, 96, 48], 2),
        "olympus-gene" => build_mlp(
            "olympus-gene",
            "OLYMPUS Gene Expert",
            256,
            &[512, 256, 128],
            2,
        ),
        "olympus-npz" => build_mlp(
            "olympus-npz",
            "OLYMPUS NPZ Expert",
            384,
            &[768, 384, 192],
            2,
        ),
        "olympus-combined" => build_mlp(
            "olympus-combined",
            "OLYMPUS Combined Expert",
            512,
            &[1024, 512, 256],
            2,
        ),

        _ => panic!("unknown architecture: {name}"),
    }
}

// ── Architecture builders ──

fn build_hvac_diagnostic(
    name: &str,
    desc: &str,
    n_features: i64,
    n_classes: i64,
    kh: i64,
    n_horizons: i64,
    hidden_ch: i64,
    seq_len: i64,
) -> ModelBundle {
    let mut g = BundleGraph::new();
    let f = n_features;
    let h = hidden_ch as i64;
    g.add_input("sensor_seq", vec![-1, f, kh, seq_len]);
    g.add_output("class_logits", vec![-1, n_classes]);
    g.add_output("health_score", vec![-1, 1]);

    add_conv_bn_relu(&mut g, "enc1", f, h, 1, 3, 2, "sensor_seq", "enc1_out", 1);
    add_conv_bn_relu(
        &mut g,
        "enc2",
        h,
        h * 2,
        1,
        3,
        2,
        "enc1_out_relu",
        "enc2_out",
        2,
    );
    add_conv_bn_relu(
        &mut g,
        "enc3",
        h * 2,
        h * 2,
        1,
        3,
        2,
        "enc2_out_relu",
        "enc3_out",
        3,
    );
    g.add_node(
        "gap",
        "GlobalAvgPool",
        serde_json::Value::Null,
        vec!["enc3_out_relu"],
        vec!["pooled"],
    );
    g.add_node(
        "flatten",
        "Flatten",
        serde_json::json!({"axis": 1}),
        vec!["pooled"],
        vec!["flat"],
    );
    add_gemm(
        &mut g,
        "head_class",
        h * 2,
        n_classes,
        "flat",
        "class_logits",
        10,
    );
    add_gemm(&mut g, "head_health", h * 2, 1, "flat", "health_score", 11);

    let params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(name, f as usize, Vec::new())
        .with_hyperparam("description", desc)
        .with_hyperparam("total_params", params)
        .with_graph(g)
}

fn build_vision_cnn(
    name: &str,
    desc: &str,
    in_ch: i64,
    h: i64,
    w: i64,
    n_classes: i64,
    n_blocks: usize,
) -> ModelBundle {
    let mut g = BundleGraph::new();
    g.add_input("image", vec![-1, in_ch, h, w]);
    g.add_output("logits", vec![-1, n_classes]);

    let channels = [64i64, 128, 256, 256, 512, 512];
    let mut prev_ch = in_ch;
    let mut prev_out = "image".to_string();

    for i in 0..n_blocks.min(channels.len()) {
        let ch = channels[i];
        let lbl = format!("block{i}");
        let out = format!("{lbl}_out");
        add_conv_bn_relu(
            &mut g,
            &lbl,
            prev_ch,
            ch,
            3,
            3,
            2,
            &prev_out,
            &out,
            (i + 1) as u64,
        );
        prev_ch = ch;
        prev_out = format!("{lbl}_out_relu");
    }

    g.add_node(
        "gap",
        "GlobalAvgPool",
        serde_json::Value::Null,
        vec![&prev_out],
        vec!["pooled"],
    );
    g.add_node(
        "flatten",
        "Flatten",
        serde_json::json!({"axis": 1}),
        vec!["pooled"],
        vec!["flat"],
    );
    add_gemm(
        &mut g,
        "classifier",
        prev_ch,
        n_classes,
        "flat",
        "logits",
        100,
    );

    let params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(name, in_ch as usize, Vec::new())
        .with_hyperparam("description", desc)
        .with_hyperparam("total_params", params)
        .with_graph(g)
}

fn build_mlp(name: &str, desc: &str, in_dim: i64, hidden: &[i64], out_dim: i64) -> ModelBundle {
    let mut g = BundleGraph::new();
    g.add_input("input", vec![-1, in_dim]);
    g.add_output("output", vec![-1, out_dim]);

    let mut prev_dim = in_dim;
    let mut prev_name = "input".to_string();

    for (i, &h) in hidden.iter().enumerate() {
        let lbl = format!("fc{i}");
        let out = format!("{lbl}_out");
        let relu = format!("{lbl}_relu");
        add_gemm(&mut g, &lbl, prev_dim, h, &prev_name, &out, (i + 1) as u64);
        g.add_node(
            &format!("{lbl}_act"),
            "Relu",
            serde_json::Value::Null,
            vec![&out],
            vec![&relu],
        );
        prev_dim = h;
        prev_name = relu;
    }
    add_gemm(
        &mut g,
        "output_head",
        prev_dim,
        out_dim,
        &prev_name,
        "output",
        99,
    );

    let params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(name, in_dim as usize, Vec::new())
        .with_hyperparam("description", desc)
        .with_hyperparam("total_params", params)
        .with_graph(g)
}

fn build_transformer(
    name: &str,
    desc: &str,
    d_model: i64,
    d_inner: i64,
    n_layers: usize,
) -> ModelBundle {
    let mut g = BundleGraph::new();
    g.add_input("input", vec![-1, d_model, 4, 1]);
    g.add_output("output", vec![-1, d_model, 4, 1]);

    let mut prev = "input".to_string();
    for i in 0..n_layers {
        let qkv = format!("layer{i}_qkv");
        let qkv_out = format!("{qkv}_out");
        let qkv_relu = format!("{qkv}_relu");
        add_conv1x1_bn_relu(
            &mut g,
            &qkv,
            d_model,
            d_inner,
            &prev,
            &qkv_out,
            (i * 10 + 1) as u64,
        );

        let proj = format!("layer{i}_proj");
        let proj_out = format!("{proj}_out");
        let proj_relu = format!("{proj}_relu");
        add_conv1x1_bn_relu(
            &mut g,
            &proj,
            d_inner,
            d_model,
            &format!("{qkv}_out_relu"),
            &proj_out,
            (i * 10 + 2) as u64,
        );
        prev = format!("{proj}_out_relu");
    }

    // Identity to output
    g.add_node(
        "out_id",
        "Identity",
        serde_json::Value::Null,
        vec![&prev],
        vec!["output"],
    );

    let params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(name, d_model as usize, Vec::new())
        .with_hyperparam("description", desc)
        .with_hyperparam("total_params", params)
        .with_graph(g)
}

fn build_tcn_transformer(
    name: &str,
    desc: &str,
    d_model: i64,
    d_inner: i64,
    n_layers: usize,
) -> ModelBundle {
    let mut g = BundleGraph::new();
    g.add_input("input", vec![-1, d_model, 1, 64]);
    g.add_output("output", vec![-1, d_model, 1, 64]);

    let mut prev = "input".to_string();
    for i in 0..n_layers {
        let conv = format!("tcn{i}");
        let out = format!("{conv}_out");
        add_conv_bn_relu(
            &mut g,
            &conv,
            d_model,
            d_model,
            1,
            3,
            1,
            &prev,
            &out,
            (i + 1) as u64,
        );
        prev = format!("{conv}_out_relu");
    }
    g.add_node(
        "out_id",
        "Identity",
        serde_json::Value::Null,
        vec![&prev],
        vec!["output"],
    );

    let params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(name, d_model as usize, Vec::new())
        .with_hyperparam("description", desc)
        .with_hyperparam("total_params", params)
        .with_graph(g)
}

/// Hydra — SSM (Conv1d S6 scan) + windowed local attention hybrid.
/// Each layer alternates: SSM block (Conv1x3 temporal scan) + attention block (Conv1x1 QKV).
fn build_hydra(name: &str, desc: &str, d_model: i64, d_inner: i64, n_layers: usize) -> ModelBundle {
    let mut g = BundleGraph::new();
    g.add_input("input", vec![-1, d_model, 1, 64]);
    g.add_output("output", vec![-1, d_model, 1, 64]);

    let mut prev = "input".to_string();
    for i in 0..n_layers {
        // SSM block: Conv1x3 temporal scan (simulates selective S6 scan)
        let ssm = format!("layer{i}_ssm");
        let ssm_out = format!("{ssm}_out");
        add_conv_bn_relu(
            &mut g,
            &ssm,
            d_model,
            d_inner,
            1,
            3,
            1,
            &prev,
            &ssm_out,
            (i * 10 + 1) as u64,
        );

        // Gate: Conv1x1 projection back to d_model
        let gate = format!("layer{i}_gate");
        let gate_out = format!("{gate}_out");
        add_conv1x1_bn_relu(
            &mut g,
            &gate,
            d_inner,
            d_model,
            &format!("{ssm}_out_relu"),
            &gate_out,
            (i * 10 + 2) as u64,
        );

        // Windowed attention: Conv1x1 QKV projection
        let attn = format!("layer{i}_attn");
        let attn_out = format!("{attn}_out");
        add_conv1x1_bn_relu(
            &mut g,
            &attn,
            d_model,
            d_inner,
            &format!("{gate}_out_relu"),
            &attn_out,
            (i * 10 + 3) as u64,
        );

        // Output projection
        let proj = format!("layer{i}_proj");
        let proj_out = format!("{proj}_out");
        add_conv1x1_bn_relu(
            &mut g,
            &proj,
            d_inner,
            d_model,
            &format!("{attn}_out_relu"),
            &proj_out,
            (i * 10 + 4) as u64,
        );
        prev = format!("{proj}_out_relu");
    }
    g.add_node(
        "out_id",
        "Identity",
        serde_json::Value::Null,
        vec![&prev],
        vec!["output"],
    );

    let params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(name, d_model as usize, Vec::new())
        .with_hyperparam("description", desc)
        .with_hyperparam("architecture", "hydra_ssm_attention_hybrid")
        .with_hyperparam("total_params", params)
        .with_graph(g)
}

/// Chimera — Sparse MoE (top-2 of N experts) + differential attention.
/// Each layer: DiffAttn (Conv1x1 dual-head) + MoE (N expert Conv1x1 FFNs, top-2 routed).
fn build_chimera(
    name: &str,
    desc: &str,
    d_model: i64,
    d_inner: i64,
    n_layers: usize,
    n_experts: usize,
) -> ModelBundle {
    let mut g = BundleGraph::new();
    g.add_input("input", vec![-1, d_model, 4, 1]);
    g.add_output("output", vec![-1, d_model, 4, 1]);

    let mut prev = "input".to_string();
    for i in 0..n_layers {
        // Differential attention: two parallel Conv1x1 projections
        let attn_a = format!("layer{i}_attn_a");
        let attn_b = format!("layer{i}_attn_b");
        let attn_a_out = format!("{attn_a}_out");
        let attn_b_out = format!("{attn_b}_out");
        add_conv1x1_bn_relu(
            &mut g,
            &attn_a,
            d_model,
            d_inner,
            &prev,
            &attn_a_out,
            (i * 20 + 1) as u64,
        );
        add_conv1x1_bn_relu(
            &mut g,
            &attn_b,
            d_model,
            d_inner,
            &prev,
            &attn_b_out,
            (i * 20 + 2) as u64,
        );

        // Output projection from attention
        let attn_proj = format!("layer{i}_attn_proj");
        let attn_proj_out = format!("{attn_proj}_out");
        add_conv1x1_bn_relu(
            &mut g,
            &attn_proj,
            d_inner,
            d_model,
            &format!("{attn_a}_out_relu"),
            &attn_proj_out,
            (i * 20 + 3) as u64,
        );

        // MoE: N expert FFN blocks (each a Conv1x1 up + down)
        // In HEF compilation, only the active experts matter — represent as parallel Conv paths
        for e in 0..n_experts.min(4) {
            let up = format!("layer{i}_expert{e}_up");
            let up_out = format!("{up}_out");
            add_conv1x1_bn_relu(
                &mut g,
                &up,
                d_model,
                d_inner / 2,
                &format!("{attn_proj}_out_relu"),
                &up_out,
                (i * 20 + 10 + e) as u64,
            );

            let down = format!("layer{i}_expert{e}_down");
            let down_out = format!("{down}_out");
            add_conv1x1_bn_relu(
                &mut g,
                &down,
                d_inner / 2,
                d_model,
                &format!("{up}_out_relu"),
                &down_out,
                (i * 20 + 14 + e) as u64,
            );
        }
        // Use last expert's output as the layer output (simplified routing)
        let last_e = n_experts.min(4) - 1;
        prev = format!("layer{i}_expert{last_e}_down_out_relu");
    }
    g.add_node(
        "out_id",
        "Identity",
        serde_json::Value::Null,
        vec![&prev],
        vec!["output"],
    );

    let params: usize = g.initializers.values().map(|t| t.data.len()).sum();
    ModelBundle::new(name, d_model as usize, Vec::new())
        .with_hyperparam("description", desc)
        .with_hyperparam("architecture", "chimera_moe_diffattn")
        .with_hyperparam("num_experts", n_experts)
        .with_hyperparam("total_params", params)
        .with_graph(g)
}

// ── Layer helpers ──

fn add_conv_bn_relu(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    kh: i64,
    kw: i64,
    stride: i64,
    input: &str,
    output: &str,
    seed: u64,
) {
    let cw = format!("{name}.conv.weight");
    let cb = format!("{name}.conv.bias");
    let bn_w = format!("{name}.bn.weight");
    let bn_b = format!("{name}.bn.bias");
    let bn_m = format!("{name}.bn.mean");
    let bn_v = format!("{name}.bn.var");
    let conv_out = output.to_string();
    let bn_out = format!("{name}_bn");
    let relu_out = format!("{name}_out_relu");

    let w_n = (out_c * in_c * kh * kw) as usize;
    g.add_initializer(
        &cw,
        vec![out_c, in_c, kh, kw],
        init_kaiming(w_n, (in_c * kh * kw) as usize, seed),
    );
    g.add_initializer(&cb, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_w, vec![out_c], vec![1.0; out_c as usize]);
    g.add_initializer(&bn_b, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_m, vec![out_c], vec![0.0; out_c as usize]);
    g.add_initializer(&bn_v, vec![out_c], vec![1.0; out_c as usize]);

    let ph = kh / 2;
    let pw = kw / 2;
    g.add_node(
        &format!("{name}_conv"),
        "Conv2d",
        serde_json::json!({"kernel_shape": [kh, kw], "strides": [stride, stride],
                           "pads": [ph, pw, ph, pw], "dilations": [1, 1], "group": 1}),
        vec![input, &cw, &cb],
        vec![&conv_out],
    );
    g.add_node(
        &format!("{name}_bn"),
        "BatchNorm",
        serde_json::json!({"epsilon": 1e-5, "momentum": 0.1}),
        vec![&conv_out, &bn_w, &bn_b, &bn_m, &bn_v],
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

fn add_conv1x1_bn_relu(
    g: &mut BundleGraph,
    name: &str,
    in_c: i64,
    out_c: i64,
    input: &str,
    output: &str,
    seed: u64,
) {
    add_conv_bn_relu(g, name, in_c, out_c, 1, 1, 1, input, output, seed);
}

fn add_gemm(
    g: &mut BundleGraph,
    name: &str,
    in_dim: i64,
    out_dim: i64,
    input: &str,
    output: &str,
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
        vec![input, &w, &b],
        vec![output],
    );
}

fn init_kaiming(n: usize, fan_in: usize, seed: u64) -> Vec<f32> {
    let std = (2.0 / fan_in.max(1) as f64).sqrt() as f32;
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            u * std
        })
        .collect()
}
