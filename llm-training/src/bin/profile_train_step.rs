//! Profile a single Qwen3 forward + backward + optimizer step.
//!
//! Goal: bucket the 45s/step distillation wall-clock into forward vs
//! backward vs optimizer, and (with `AXONML_PROFILE_BACKWARD=1` set)
//! print per-backward-op timing so we know which GradFn to fix next.
//!
//! This is NOT a full distill — no teacher, no dataset, just random
//! token IDs into a fresh tiny-Qwen3 and a shifted-CE loss. The point
//! is to make one step fast to iterate on, not to train a model.
//!
//! Run (with CUDA):
//!   cargo run --release --features cuda --bin profile_train_step
//!
//! Run with per-backward-op breakdown:
//!   AXONML_PROFILE_BACKWARD=1 \
//!     cargo run --release --features cuda --bin profile_train_step

use std::time::Instant;

use axonml_core::Device;
use axonml_llm::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use axonml_optim::{AdamW, Optimizer};
use axonml_tensor::Tensor;
use llm_training::shifted_cross_entropy;

fn main() {
    let bs = 4;
    let seq = 512;

    // Qwen3-0.6B shape — the real distill target.
    let mut cfg = Qwen3Config::qwen3_0_6b();
    // Keep vocab manageable so random labels don't dominate CE cost.
    cfg.vocab_size = 1024;
    cfg.max_position_embeddings = seq;

    let device = pick_device();
    println!("profile_train_step — Qwen3-0.6B-shaped, bs={bs} seq={seq}, device={device:?}");
    println!();

    let t_build = Instant::now();
    let model = Qwen3ForCausalLM::new(&cfg);
    for p in model.parameters() {
        p.to_device(device);
    }
    println!(
        "  model init + to_device   {:.2} ms",
        t_build.elapsed().as_secs_f64() * 1000.0
    );

    let params = model.parameters();
    let mut optimizer = AdamW::new(params, 1e-4);

    // Random token IDs. u32 Tensors stay on CPU (GPU path is f32-only).
    let ids: Vec<u32> = (0..bs * seq)
        .map(|i| (i as u32) % cfg.vocab_size as u32)
        .collect();
    let input_ids = Tensor::<u32>::from_vec(ids.clone(), &[bs, seq]).unwrap();
    let labels = Tensor::<u32>::from_vec(ids, &[bs, seq]).unwrap();

    // Warm-up step — first step does one-time CUDA initialization, PTX
    // parsing, pool priming; throws off the next-step measurement.
    println!("\n--- warm-up step (not counted) ---");
    run_step(&model, &mut optimizer, &input_ids, &labels, device);

    // Hot step — all the numbers we care about.
    println!("\n--- hot step ---");
    print_pool_stats("before hot step");
    let breakdown = run_step(&model, &mut optimizer, &input_ids, &labels, device);
    print_pool_stats("after hot step");
    println!();
    println!(
        "  forward                  {:>8.1} ms",
        breakdown.forward_ms
    );
    println!("  loss (shifted CE)        {:>8.1} ms", breakdown.loss_ms);
    println!(
        "  backward                 {:>8.1} ms",
        breakdown.backward_ms
    );
    println!("  optimizer.step           {:>8.1} ms", breakdown.optim_ms);
    println!("  ───────────────────────────────────");
    println!(
        "  TOTAL                    {:>8.1} ms",
        breakdown.forward_ms + breakdown.loss_ms + breakdown.backward_ms + breakdown.optim_ms
    );
    println!();
    println!(
        "Set AXONML_PROFILE_BACKWARD=1 for per-backward-op breakdown. Current: {}",
        if std::env::var("AXONML_PROFILE_BACKWARD").is_ok() {
            "ON"
        } else {
            "OFF"
        }
    );
}

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::Cuda(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

struct Breakdown {
    forward_ms: f64,
    loss_ms: f64,
    backward_ms: f64,
    optim_ms: f64,
}

fn run_step(
    model: &Qwen3ForCausalLM,
    optimizer: &mut AdamW,
    input_ids: &Tensor<u32>,
    labels: &Tensor<u32>,
    device: Device,
) -> Breakdown {
    let _ = device;
    optimizer.zero_grad();
    // Drain any leftover GPU work before we start measuring.
    sync();

    // Forward
    let t_fwd = Instant::now();
    let logits = model.forward_ids(input_ids);
    sync();
    let forward_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;

    // Shifted CE loss: drop last logit position, use [1..] labels.
    let t_loss = Instant::now();
    let loss = shifted_cross_entropy(&logits, labels);
    sync();
    let loss_ms = t_loss.elapsed().as_secs_f64() * 1000.0;

    // Backward
    let t_bwd = Instant::now();
    loss.backward();
    sync();
    let backward_ms = t_bwd.elapsed().as_secs_f64() * 1000.0;

    // Optimizer step
    let t_opt = Instant::now();
    optimizer.step();
    sync();
    let optim_ms = t_opt.elapsed().as_secs_f64() * 1000.0;

    Breakdown {
        forward_ms,
        loss_ms,
        backward_ms,
        optim_ms,
    }
}

#[inline]
fn sync() {
    #[cfg(feature = "cuda")]
    {
        let _ = axonml_core::backends::cuda::cuda_sync();
    }
}

fn print_pool_stats(label: &str) {
    #[cfg(feature = "cuda")]
    {
        let pool = axonml_core::backends::cuda_pool::get_memory_pool();
        let (hits, misses, returns, bytes) = pool.stats();
        println!(
            "  [pool {label}] hits={hits} misses={misses} returns={returns} pooled={}MB",
            bytes / (1024 * 1024)
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = label;
    }
}
