//! Trident 1.58-bit SLM Training Example
//!
//! Trains a small Trident model (BitNet b1.58 ternary weights) on synthetic
//! next-token prediction data. Reports loss, perplexity, weight sparsity,
//! and compression ratio per epoch.
//!
//! Usage:
//!   cargo run --release --example train_trident -p axonml-llm
//!   cargo run --release --example train_trident -p axonml-llm -- --monitor
//!
//! # File
//! `crates/axonml-llm/examples/train_trident.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 19, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::env;
use std::time::Instant;

use axonml_llm::trident::{TridentConfig, TridentModel};
use axonml_nn::Module;
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::Tensor;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// =============================================================================
// Configuration
// =============================================================================

const VOCAB_SIZE: usize = 1000;
const MAX_SEQ_LEN: usize = 32;
const D_MODEL: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_LAYERS: usize = 2;
const INTERMEDIATE_SIZE: usize = 256;

const NUM_TRAIN: usize = 200;
const BATCH_SIZE: usize = 4;
const NUM_EPOCHS: usize = 10;
const LEARNING_RATE: f32 = 0.001;

// =============================================================================
// Synthetic data generation
// =============================================================================

/// Generate synthetic token sequences for language modeling.
///
/// Creates sequences with simple patterns:
/// - Tokens cycle through small ranges to give the model learnable structure
/// - Token 0 is reserved (padding)
fn generate_sequences(num_samples: usize, seq_len: usize, rng: &mut StdRng) -> Vec<Vec<u32>> {
    let mut sequences = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        // Create a patterned sequence: pick a base token and create an
        // incrementing pattern with noise, giving the model something learnable
        let base = rng.gen_range(1..VOCAB_SIZE as u32 / 2);
        let step = rng.gen_range(1u32..5);
        let seq: Vec<u32> = (0..seq_len)
            .map(|i| {
                let token = base + (i as u32 * step);
                let noise: u32 = rng.gen_range(0..3);
                ((token + noise) % (VOCAB_SIZE as u32 - 1)) + 1
            })
            .collect();
        sequences.push(seq);
    }
    sequences
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    let use_monitor = args.iter().any(|a| a == "--monitor");

    println!("=== Trident 1.58-bit SLM Training ===");
    println!("  BitNet b1.58 ternary weights {{-1, 0, +1}}");
    println!();

    // ---- Model configuration ----
    let config = TridentConfig {
        vocab_size: VOCAB_SIZE,
        d_model: D_MODEL,
        num_layers: NUM_LAYERS,
        num_heads: NUM_HEADS,
        intermediate_size: INTERMEDIATE_SIZE,
        max_seq_len: MAX_SEQ_LEN,
        rms_norm_eps: 1e-6,
    };

    let model = TridentModel::new(&config);

    let param_count: usize = model.parameters().iter().map(|p| p.data().numel()).sum();
    let fp32_mb = param_count as f32 * 4.0 / (1024.0 * 1024.0);

    println!("Model Configuration:");
    println!("  d_model          : {}", D_MODEL);
    println!("  heads            : {}", NUM_HEADS);
    println!("  layers           : {}", NUM_LAYERS);
    println!("  intermediate     : {}", INTERMEDIATE_SIZE);
    println!("  vocab            : {}", VOCAB_SIZE);
    println!("  max_seq_len      : {}", MAX_SEQ_LEN);
    println!("  parameters       : {}", param_count);
    println!("  fp32 size        : {:.2} MB", fp32_mb);
    println!();

    // Storage analysis
    let ternary_bytes = config.ternary_storage_bytes();
    let fp32_bytes = config.fp32_storage_bytes();
    let compression = fp32_bytes as f32 / ternary_bytes as f32;
    println!("Storage Analysis:");
    println!(
        "  fp32 storage     : {:.2} MB",
        fp32_bytes as f32 / (1024.0 * 1024.0)
    );
    println!(
        "  ternary storage  : {:.2} MB",
        ternary_bytes as f32 / (1024.0 * 1024.0)
    );
    println!("  compression      : {:.1}x", compression);
    println!();

    println!("Training:");
    println!("  samples          : {}", NUM_TRAIN);
    println!("  batch_size       : {}", BATCH_SIZE);
    println!("  epochs           : {}", NUM_EPOCHS);
    println!("  learning_rate    : {}", LEARNING_RATE);
    println!("  optimizer        : Adam (STE for ternary weights)");
    println!();

    // ---- Monitor (optional) ----
    let monitor = if use_monitor {
        let m = axonml::TrainingMonitor::new("Trident-1.58bit", param_count)
            .total_epochs(NUM_EPOCHS)
            .batch_size(BATCH_SIZE)
            .launch();
        println!("Monitor : http://127.0.0.1:{}", m.port());
        println!();
        Some(m)
    } else {
        None
    };

    // ---- Generate synthetic data ----
    let mut rng = StdRng::seed_from_u64(42);
    let train_seqs = generate_sequences(NUM_TRAIN, MAX_SEQ_LEN, &mut rng);

    // ---- Optimizer (Adam on shadow weights — STE gradient flows through) ----
    let mut optimizer = Adam::new(model.parameters(), LEARNING_RATE);

    // ---- Training loop ----
    let total_start = Instant::now();

    println!(
        "{:<6} {:<10} {:<10} {:<10} {:<10} {:<8}",
        "Epoch", "Loss", "PPL", "Sparsity", "CompRatio", "Time"
    );
    println!("{}", "-".repeat(60));

    for epoch in 1..=NUM_EPOCHS {
        let epoch_start = Instant::now();

        let mut epoch_loss = 0.0f32;
        let mut epoch_tokens = 0usize;
        let num_batches = NUM_TRAIN.div_ceil(BATCH_SIZE);

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(NUM_TRAIN);
            let bs = end - start;

            // Build input tensor [bs, seq_len]
            let mut token_data = Vec::with_capacity(bs * MAX_SEQ_LEN);
            for seq in &train_seqs[start..end] {
                token_data.extend_from_slice(seq);
            }
            let input_ids =
                Tensor::<u32>::from_vec(token_data.clone(), &[bs, MAX_SEQ_LEN]).unwrap();
            let labels = Tensor::<u32>::from_vec(token_data, &[bs, MAX_SEQ_LEN]).unwrap();

            // Forward + loss
            optimizer.zero_grad();
            let (_logits, loss) = model.forward_with_loss(&input_ids, &labels);
            let loss_val = loss.data().to_vec()[0];

            // Backward + step (STE: gradients pass through quantization)
            loss.backward();
            optimizer.step();

            let predicted_tokens = bs * (MAX_SEQ_LEN - 1);
            epoch_loss += loss_val * predicted_tokens as f32;
            epoch_tokens += predicted_tokens;
        }

        let avg_loss = epoch_loss / epoch_tokens as f32;
        let perplexity = avg_loss.exp().min(99999.0);
        let sparsity = model.average_sparsity();
        let epoch_time = epoch_start.elapsed();

        println!(
            "{:<6} {:<10.4} {:<10.2} {:<10.1}% {:<10.1}x {:<8.1}s",
            format!("{}/{}", epoch, NUM_EPOCHS),
            avg_loss,
            perplexity,
            sparsity * 100.0,
            compression,
            epoch_time.as_secs_f32()
        );

        if let Some(ref m) = monitor {
            m.log_epoch(
                epoch,
                avg_loss,
                None,
                vec![
                    ("perplexity", perplexity),
                    ("sparsity", sparsity),
                    ("compression_ratio", compression),
                ],
            );
        }
    }

    let total_time = total_start.elapsed();
    println!();
    println!("Training complete in {:.1}s", total_time.as_secs_f32());
    println!();

    // ---- Final report ----
    println!("=== Final Model Statistics ===");
    let final_sparsity = model.average_sparsity();
    println!(
        "Weight sparsity    : {:.1}% zeros in ternary representation",
        final_sparsity * 100.0
    );
    println!(
        "Memory savings     : {:.1}x compression (ternary vs fp32)",
        compression
    );
    println!("Inference benefit  : Matmul reduces to add/sub (no FP multiply)");

    if let Some(ref m) = monitor {
        m.set_status("complete");
        println!();
        println!(
            "Monitor still running at http://127.0.0.1:{} (Ctrl+C to exit)",
            m.port()
        );
        std::thread::park();
    }
}
