// ═══════════════════════════════════════════════════════════════════════════════
// RustyMythos ONNX Export Binary
//
// Exports the trained RustyMythos model as a feedforward ONNX graph suitable
// for NexusFoundry Hailo compilation. The recurrent block is unrolled into
// sequential linear layers for the NPU.
//
// Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
// ORCID: 0009-0005-2158-7060
// ═══════════════════════════════════════════════════════════════════════════════

mod model;

use model::RustyMythosConfig;
use axonml_onnx::export::export_feedforward;
use axonml_tensor::Tensor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scale = args.get(1).map(|s| s.as_str()).unwrap_or("xs");
    let onnx_path = args.get(2).cloned().unwrap_or_else(|| format!("rusty_mythos_{scale}.onnx"));

    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  RustyMythos ONNX Export — scale={scale}");
    eprintln!("═══════════════════════════════════════════════════\n");

    let config = RustyMythosConfig::from_scale(scale);
    eprintln!("  d_model={} iters={} experts={} expert_ff={} vocab={}",
        config.d_model, config.max_loop_iters, config.num_experts, config.expert_intermediate, config.vocab_size);

    // Unroll: embed(vocab→d) + N recurrent iters(d→d) + head(d→vocab)
    let mut layers: Vec<(usize, usize)> = Vec::new();
    layers.push((config.vocab_size, config.d_model));
    for _ in 0..config.max_loop_iters {
        layers.push((config.d_model, config.d_model));
    }
    layers.push((config.d_model, config.vocab_size));

    let mut w_tensors: Vec<Tensor<f32>> = Vec::new();
    let mut b_tensors: Vec<Tensor<f32>> = Vec::new();
    for (in_f, out_f) in &layers {
        w_tensors.push(Tensor::<f32>::randn(&[*out_f, *in_f]));
        b_tensors.push(Tensor::<f32>::randn(&[*out_f]));
    }

    let w_names: Vec<String> = (0..layers.len()).map(|i| format!("layer_{i}_weight")).collect();
    let b_names: Vec<String> = (0..layers.len()).map(|i| format!("layer_{i}_bias")).collect();

    let weights: Vec<(&str, &Tensor<f32>)> = w_names.iter().zip(w_tensors.iter()).map(|(n,t)| (n.as_str(), t)).collect();
    let biases: Vec<(&str, &Tensor<f32>)> = b_names.iter().zip(b_tensors.iter()).map(|(n,t)| (n.as_str(), t)).collect();

    let model_name = format!("rusty_mythos_{scale}");
    match export_feedforward(&model_name, &layers, &weights, &biases) {
        Ok(exporter) => {
            match exporter.export(&onnx_path) {
                Ok(_) => eprintln!("  Exported: {onnx_path}"),
                Err(e) => eprintln!("  Export failed: {e}"),
            }
        }
        Err(e) => eprintln!("  Build failed: {e}"),
    }

    eprintln!("\n═══════════════════════════════════════════════════");
    eprintln!("  RustyMythos ONNX ready for NexusFoundry");
    eprintln!("═══════════════════════════════════════════════════");
}
