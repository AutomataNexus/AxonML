//! GPT-2 Language Modeling Training Example
//!
//! Trains a small GPT-2 model on synthetic next-token prediction data.
//! Demonstrates GPT2LMHead with CrossEntropyLoss and Adam optimizer,
//! reporting both loss and perplexity per epoch.
//!
//! Usage:
//!   cargo run --release --example train_gpt2 -p axonml-llm
//!   cargo run --release --example train_gpt2 -p axonml-llm -- --monitor

use std::env;
use std::time::Instant;

use axonml_llm::{GPT2Config, GPT2LMHead};
use axonml_nn::Module;
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::Tensor;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// =============================================================================
// Configuration
// =============================================================================

const VOCAB_SIZE: usize = 1000;
const MAX_SEQ_LEN: usize = 64;
const D_MODEL: usize = 128;
const NUM_HEADS: usize = 4;
const NUM_LAYERS: usize = 2;

const NUM_TRAIN: usize = 500;
const BATCH_SIZE: usize = 8;
const NUM_EPOCHS: usize = 10;
const LEARNING_RATE: f32 = 0.0001;

// =============================================================================
// Synthetic data generation
// =============================================================================

/// Generate synthetic token sequences for language modeling.
///
/// Each sequence is a random series of token IDs in [1, VOCAB_SIZE).
/// Token 0 is reserved (padding). The target for each sequence is the
/// input shifted right by one position (next-token prediction).
fn generate_sequences(num_samples: usize, seq_len: usize, rng: &mut StdRng) -> Vec<Vec<u32>> {
    let mut sequences = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let seq: Vec<u32> = (0..seq_len)
            .map(|_| rng.gen_range(1..VOCAB_SIZE as u32))
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

    println!("=== GPT-2 Language Modeling Training ===");
    println!();

    let device_name = "CPU";
    println!("Device  : {device_name}");

    // ---- Model configuration ----
    let config = GPT2Config {
        vocab_size: VOCAB_SIZE,
        n_ctx: MAX_SEQ_LEN,
        n_embd: D_MODEL,
        n_layer: NUM_LAYERS,
        n_head: NUM_HEADS,
        dropout: 0.1,
        attn_dropout: 0.1,
        resid_dropout: 0.1,
        layer_norm_eps: 1e-5,
        activation: "gelu".to_string(),
        bos_token_id: 0,
        eos_token_id: 0,
    };

    let mut model = GPT2LMHead::new(&config);

    let param_count: usize = model.parameters().iter().map(|p| p.data().numel()).sum();
    println!("Params  : {param_count}");
    println!(
        "Config  : d_model={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}, vocab={VOCAB_SIZE}"
    );
    println!("Data    : {NUM_TRAIN} sequences, batch_size={BATCH_SIZE}");
    println!("Epochs  : {NUM_EPOCHS}, lr={LEARNING_RATE}");
    println!();

    // ---- Monitor (optional) ----
    let monitor = if use_monitor {
        let m = axonml::TrainingMonitor::new("GPT2-LM", param_count)
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

    // ---- Optimizer ----
    let mut optimizer = Adam::new(model.parameters(), LEARNING_RATE);

    // ---- Training loop ----
    let total_start = Instant::now();

    for epoch in 1..=NUM_EPOCHS {
        model.train();
        let epoch_start = Instant::now();

        let mut epoch_loss = 0.0_f32;
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

            // Labels are the same sequence (forward_with_loss handles the shift)
            let labels = Tensor::<u32>::from_vec(token_data, &[bs, MAX_SEQ_LEN]).unwrap();

            // Forward + loss (internally shifts logits/labels for next-token prediction)
            optimizer.zero_grad();
            let (_logits, loss) = model.forward_with_loss(&input_ids, &labels);
            let loss_val = loss.data().to_vec()[0];

            // Backward + step
            loss.backward();
            optimizer.step();

            // Accumulate loss weighted by number of predicted tokens
            let predicted_tokens = bs * (MAX_SEQ_LEN - 1);
            epoch_loss += loss_val * predicted_tokens as f32;
            epoch_tokens += predicted_tokens;
        }

        let avg_loss = epoch_loss / epoch_tokens as f32;
        let perplexity = avg_loss.exp();
        let epoch_time = epoch_start.elapsed();

        println!(
            "Epoch {epoch:>2}/{NUM_EPOCHS}  loss={avg_loss:.4}  ppl={perplexity:.2}  [{:.1}s]",
            epoch_time.as_secs_f32()
        );

        if let Some(ref m) = monitor {
            m.log_epoch(epoch, avg_loss, None, vec![("perplexity", perplexity)]);
        }
    }

    let total_time = total_start.elapsed();
    println!();
    println!("Training complete in {:.1}s", total_time.as_secs_f32());

    if let Some(ref m) = monitor {
        m.set_status("complete");
        println!(
            "Monitor still running at http://127.0.0.1:{} (Ctrl+C to exit)",
            m.port()
        );
        std::thread::park();
    }
}
