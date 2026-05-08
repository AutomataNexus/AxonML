// ═══════════════════════════════════════════════════════════════════════════════
// RustyMythos Training Binary
//
// Trains the RustyMythos recurrent-depth transformer on synthetic sequence data,
// exports to ONNX for NexusFoundry compilation, and saves the AxonML checkpoint.
//
// Copyright (c) 2026 Andrew Jewell Sr. / AutomataNexus LLC
// ORCID: 0009-0005-2158-7060
// ═══════════════════════════════════════════════════════════════════════════════

mod model;

use model::{RustyMythos, RustyMythosConfig};
use axonml_autograd::Variable;
use axonml_nn::{Module, CrossEntropyLoss};
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::Tensor;
use axonml_serialize::{ModelBundle, save_bundle};
use std::time::Instant;
use std::collections::HashMap;

// ─── Synthetic Data ──────────────────────────────────────────────────────────

fn generate_batch(batch_size: usize, vocab_size: usize) -> (Variable, Variable) {
    let input = Tensor::<f32>::randn(&[batch_size, vocab_size]);
    let targets = Tensor::<f32>::randn(&[batch_size, vocab_size]);
    (
        Variable::new(input, false),
        Variable::new(targets, false),
    )
}

// ─── Entry Point ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scale = args.get(1).map(|s| s.as_str()).unwrap_or("xs");
    let epochs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let batch_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);

    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  RustyMythos Training — AxonML Framework");
    eprintln!("  Scale: {scale}");
    eprintln!("═══════════════════════════════════════════════════\n");

    let config = RustyMythosConfig::from_scale(scale);
    let model = RustyMythos::new(config);
    let params = model.parameters();
    let param_count = model.param_count();

    eprintln!("  Architecture:");
    eprintln!("    d_model       = {}", model.config.d_model);
    eprintln!("    loop_iters    = {}", model.config.max_loop_iters);
    eprintln!("    vocab_size    = {}", model.config.vocab_size);
    eprintln!("    num_experts   = {}", model.config.num_experts);
    eprintln!("    top_k         = {}", model.config.top_k);
    eprintln!("    parameters    = {param_count}");
    eprintln!("    batch_size    = {batch_size}");
    eprintln!("    epochs        = {epochs}\n");

    let mut optimizer = Adam::new(params, 1e-3);
    let loss_fn = CrossEntropyLoss::new();

    // ─── Training Loop ───────────────────────────────────────────────────

    let start = Instant::now();
    for epoch in 0..epochs {
        let (input, target) = generate_batch(batch_size, model.config.vocab_size);

        let output = model.forward(&input);
        let loss = loss_fn.compute(&output, &target);
        let loss_val = loss.data().to_vec()[0];

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if epoch == 0 || (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!("  epoch {:>4}/{epochs}  loss={loss_val:.4}  elapsed={elapsed:.1}s", epoch + 1);
        }
    }

    let total = start.elapsed().as_secs_f64();
    eprintln!("\n  Training complete: {epochs} epochs in {total:.1}s\n");

    // ─── Save Checkpoint ─────────────────────────────────────────────────

    let weights: Vec<f32> = model.parameters().iter()
        .flat_map(|p| p.data().to_vec())
        .collect();
    let mut hyper = HashMap::new();
    hyper.insert("d_model".into(), serde_json::json!(model.config.d_model));
    hyper.insert("max_loop_iters".into(), serde_json::json!(model.config.max_loop_iters));
    hyper.insert("num_experts".into(), serde_json::json!(model.config.num_experts));
    hyper.insert("top_k".into(), serde_json::json!(model.config.top_k));
    let mut bundle = ModelBundle::new("rusty_mythos", model.config.vocab_size, weights);
    bundle.hyperparameters = hyper;

    let ckpt_path = format!("rusty_mythos_{scale}.axonml");
    match save_bundle(&bundle, &ckpt_path) {
        Ok(_) => eprintln!("  Checkpoint saved: {}", ckpt_path),
        Err(e) => eprintln!("  Checkpoint save failed: {e}"),
    }

    eprintln!("\n═══════════════════════════════════════════════════");
    eprintln!("  RustyMythos — {param_count} params, {epochs} epochs");
    eprintln!("  Andrew Jewell Sr. / AutomataNexus LLC");
    eprintln!("═══════════════════════════════════════════════════");
}
