//! Chimera Training Example — MoE + Differential Attention SLM
//!
//! Trains a small Chimera model on synthetic next-token prediction data.
//! Demonstrates sparse MoE routing with load balancing loss and differential
//! attention noise cancellation.
//!
//! Reports: loss, perplexity, expert utilization, active params per token,
//! and lambda evolution across training.
//!
//! Usage:
//!   cargo run --release --example train_chimera -p axonml-llm

use std::time::Instant;

use axonml_llm::chimera::{ChimeraConfig, ChimeraModel};
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
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const INTERMEDIATE_SIZE: usize = 256;

const NUM_TRAIN: usize = 200;
const BATCH_SIZE: usize = 4;
const NUM_EPOCHS: usize = 8;
const LEARNING_RATE: f32 = 0.0003;

// =============================================================================
// Synthetic Data
// =============================================================================

/// Generate synthetic token sequences with simple patterns.
///
/// Creates sequences where each token is (prev_token + offset) % vocab_size,
/// making next-token prediction learnable. This tests whether the model can
/// learn systematic token relationships through the MoE + DiffAttn pipeline.
fn generate_patterned_sequences(
    num_samples: usize,
    seq_len: usize,
    rng: &mut StdRng,
) -> Vec<Vec<u32>> {
    let mut sequences = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let start = rng.gen_range(1..VOCAB_SIZE as u32);
        let offset = rng.gen_range(1..10u32);
        let seq: Vec<u32> = (0..seq_len)
            .map(|i| (start + offset * i as u32) % VOCAB_SIZE as u32)
            .collect();
        sequences.push(seq);
    }
    sequences
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("=== Chimera: MoE + Differential Attention SLM ===");
    println!();

    // ---- Model configuration ----
    let config = ChimeraConfig {
        vocab_size: VOCAB_SIZE,
        d_model: D_MODEL,
        num_layers: NUM_LAYERS,
        num_heads: NUM_HEADS,
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        intermediate_size: INTERMEDIATE_SIZE,
        max_seq_len: MAX_SEQ_LEN,
        rms_norm_eps: 1e-5,
        lambda_init: 0.05,
        load_balance_weight: 0.01,
    };

    let model = ChimeraModel::new(&config);

    let total_params = model.total_param_count();
    let active_params = config.estimate_active_params();

    println!("Architecture:");
    println!("  d_model       = {D_MODEL}");
    println!("  layers        = {NUM_LAYERS}");
    println!("  heads         = {NUM_HEADS}");
    println!("  experts       = {NUM_EXPERTS} (top-{TOP_K} active)");
    println!("  intermediate  = {INTERMEDIATE_SIZE}");
    println!("  vocab_size    = {VOCAB_SIZE}");
    println!();
    println!("Parameters:");
    println!("  Total params  = {total_params}");
    println!("  Active/token  = {active_params} ({:.1}%)", 100.0 * active_params as f64 / total_params as f64);
    println!();
    println!("Training:");
    println!("  Sequences     = {NUM_TRAIN}");
    println!("  Batch size    = {BATCH_SIZE}");
    println!("  Epochs        = {NUM_EPOCHS}");
    println!("  Learning rate = {LEARNING_RATE}");
    println!();

    // ---- Generate data ----
    let mut rng = StdRng::seed_from_u64(42);
    let train_seqs = generate_patterned_sequences(NUM_TRAIN, MAX_SEQ_LEN, &mut rng);

    // ---- Optimizer ----
    let mut optimizer = Adam::new(model.parameters(), LEARNING_RATE);

    // ---- Training loop ----
    let total_start = Instant::now();

    println!("{:<6} {:<10} {:<10} {:<8} {:<30} {}",
             "Epoch", "Loss", "PPL", "Time", "Expert Util (layer 0)", "Lambda[0]");
    println!("{}", "-".repeat(85));

    for epoch in 1..=NUM_EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_tokens = 0usize;
        let num_batches = (NUM_TRAIN + BATCH_SIZE - 1) / BATCH_SIZE;

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

            // Backward + step
            loss.backward();
            optimizer.step();

            let predicted_tokens = bs * (MAX_SEQ_LEN - 1);
            epoch_loss += loss_val * predicted_tokens as f32;
            epoch_tokens += predicted_tokens;
        }

        let avg_loss = epoch_loss / epoch_tokens as f32;
        let perplexity = avg_loss.exp().min(99999.0);
        let epoch_time = epoch_start.elapsed();

        // Expert utilization for first layer
        let util = model.expert_utilization();
        let layer0_util = if let Some((_, counts)) = util.first() {
            counts
                .iter()
                .map(|c| format!("{c:>3}"))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            "N/A".to_string()
        };

        // Lambda values
        let lambdas = model.lambda_values();
        let lambda0 = lambdas.first().copied().unwrap_or(0.0);

        println!(
            "{epoch:>4}/{NUM_EPOCHS}  {avg_loss:<10.4} {perplexity:<10.2} {:<8.1}s {:<30} {lambda0:.4}",
            epoch_time.as_secs_f32(),
            layer0_util,
        );
    }

    let total_time = total_start.elapsed();
    println!();
    println!("Training complete in {:.1}s", total_time.as_secs_f32());

    // ---- Final statistics ----
    println!();
    println!("=== Final Model Statistics ===");
    println!();

    // Expert utilization across all layers
    let util = model.expert_utilization();
    println!("Expert utilization (last batch):");
    for (layer, counts) in &util {
        let total: usize = counts.iter().sum();
        let max_count = counts.iter().max().copied().unwrap_or(0);
        let min_count = counts.iter().min().copied().unwrap_or(0);
        let balance = if max_count > 0 {
            min_count as f32 / max_count as f32
        } else {
            0.0
        };
        println!(
            "  Layer {layer:>2}: {:?}  (balance ratio: {balance:.2})",
            counts
        );
        let _ = total;
    }

    println!();
    println!("Lambda values (learnable noise cancellation):");
    let lambdas = model.lambda_values();
    for (i, l) in lambdas.iter().enumerate() {
        let delta = l - config.lambda_init;
        let direction = if delta > 0.001 {
            " (increased)"
        } else if delta < -0.001 {
            " (decreased)"
        } else {
            " (stable)"
        };
        println!("  Layer {i:>2}: {l:.6}{direction}");
    }

    println!();
    println!("Sparsity summary:");
    println!("  Total parameters:  {total_params}");
    println!("  Active per token:  {active_params}");
    println!(
        "  Compute savings:   {:.1}%",
        100.0 * (1.0 - active_params as f64 / total_params as f64)
    );
}
