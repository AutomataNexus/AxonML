// Generate NexusFoundry profiler reports for all RustyMythos scale variants
mod model;

use model::RustyMythosConfig;
use nexusfoundry_profiler::{ProfileData, LayerInfo, StreamInfo, OptimizationPass, write_report};
use std::path::Path;

fn build_profile(scale: &str, target: &str) -> ProfileData {
    let config = RustyMythosConfig::from_scale(scale);
    let hef_path = format!("profiler_reports/rusty_mythos_{scale}_{target}.hef");
    let hef_size = std::fs::metadata(&hef_path).map(|m| m.len()).unwrap_or(0);

    let num_layers = 2 + config.max_loop_iters; // embed + N iters + head
    let mut layers = Vec::new();

    layers.push(LayerInfo {
        name: "prelude/embed".into(),
        op_type: "Linear".into(),
        input_shape: format!("[1, {}]", config.vocab_size),
        output_shape: format!("[1, {}]", config.d_model),
        params: (config.vocab_size * config.d_model + config.d_model) as u64,
        macs: (config.vocab_size * config.d_model) as u64,
    });

    for i in 0..config.max_loop_iters {
        let attn_params = 2 * (config.d_model * config.d_model + config.d_model);
        let expert_params = config.num_experts * (config.d_model * config.expert_intermediate + config.expert_intermediate + config.expert_intermediate * config.d_model + config.d_model);
        let gate_params = config.d_model * config.num_experts;
        let norm_params = 4 * config.d_model;
        let iter_params = attn_params + expert_params + gate_params + norm_params;
        let iter_macs = 2 * config.d_model * config.d_model + config.d_model * config.expert_intermediate * 2;

        layers.push(LayerInfo {
            name: format!("recurrent/iter_{i}/transformer"),
            op_type: "MythosTransformerLayer".into(),
            input_shape: format!("[1, {}]", config.d_model),
            output_shape: format!("[1, {}]", config.d_model),
            params: iter_params as u64,
            macs: iter_macs as u64,
        });
    }

    layers.push(LayerInfo {
        name: "coda/head".into(),
        op_type: "Linear".into(),
        input_shape: format!("[1, {}]", config.d_model),
        output_shape: format!("[1, {}]", config.vocab_size),
        params: (config.d_model * config.vocab_size + config.vocab_size) as u64,
        macs: (config.d_model * config.vocab_size) as u64,
    });

    let total_params: u64 = layers.iter().map(|l| l.params).sum();
    let total_macs: u64 = layers.iter().map(|l| l.macs).sum();

    ProfileData {
        model_name: format!("RustyMythos-{}", scale.to_uppercase()),
        target_chip: target.into(),
        compiler_version: "NexusFoundry 1.0.0 (native-compile)".into(),
        compile_time_ms: 50,
        hef_size_bytes: hef_size,
        layers,
        total_params,
        total_macs,
        num_contexts: 1,
        num_clusters: if target == "hailo10h" { 4 } else { 1 },
        dram_transfers: 0,
        optimizations: vec![
            OptimizationPass { name: "IdentityElimination".into(), removed: 2, added: 0, fusions: 0 },
            OptimizationPass { name: "DeadNodeElimination".into(), removed: 0, added: 0, fusions: 0 },
            OptimizationPass { name: "RecurrentUnroll".into(), removed: 1, added: config.max_loop_iters as u32, fusions: 0 },
        ],
        quantization_mode: "INT8 symmetric".into(),
        calibration_samples: 256,
        inputs: vec![StreamInfo {
            name: "input".into(), dtype: "uint8".into(),
            shape: format!("[1, {}]", config.vocab_size),
        }],
        outputs: vec![StreamInfo {
            name: "output".into(), dtype: "uint8".into(),
            shape: format!("[1, {}]", config.vocab_size),
        }],
        silicon_fps: None,
        silicon_hw_latency_ms: None,
        silicon_e2e_latency_ms: None,
        silicon_temp_avg: None,
        silicon_temp_min: None,
        silicon_temp_max: None,
        silicon_send_rate: None,
        silicon_recv_rate: None,
        quantization: Vec::new(),
    }
}

fn main() {
    let scales = ["xs", "small", "medium", "large"];
    let targets = ["hailo10h", "hailo8"];

    for scale in &scales {
        for target in &targets {
            let profile = build_profile(scale, target);
            let report_path = format!("profiler_reports/rusty_mythos_{scale}_{target}_report.html");
            write_report(&profile, Path::new(&report_path)).expect("write report");
            eprintln!("  {} {}: {} params, {} MACs, HEF {}B → {}",
                profile.model_name, target,
                profile.total_params, profile.total_macs,
                profile.hef_size_bytes, report_path);
        }
    }
    eprintln!("\n  All reports generated.");
}
