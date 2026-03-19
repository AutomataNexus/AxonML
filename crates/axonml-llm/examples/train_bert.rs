//! BERT Binary Classification Training Example
//!
//! Trains a small BERT model on synthetic binary-classification data
//! (sentiment-like task). Demonstrates BertForSequenceClassification with
//! CrossEntropyLoss and Adam optimizer.
//!
//! Usage:
//!   cargo run --release --example train_bert -p axonml-llm
//!   cargo run --release --example train_bert -p axonml-llm -- --monitor

use std::env;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_llm::{BertConfig, BertForSequenceClassification};
use axonml_nn::{CrossEntropyLoss, Module};
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
const INTERMEDIATE_SIZE: usize = 256;
const NUM_CLASSES: usize = 2;

const NUM_TRAIN: usize = 500;
const NUM_TEST: usize = 100;
const BATCH_SIZE: usize = 16;
const NUM_EPOCHS: usize = 10;
const LEARNING_RATE: f32 = 0.0001;

// =============================================================================
// Synthetic data generation
// =============================================================================

/// Generate synthetic token sequences with binary labels.
///
/// Label 0: tokens drawn mostly from vocab range [0, 500)
/// Label 1: tokens drawn mostly from vocab range [500, 1000)
///
/// This gives the model a learnable signal to separate the two classes.
fn generate_data(
    num_samples: usize,
    seq_len: usize,
    rng: &mut StdRng,
) -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut sequences = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let label: u32 = rng.gen_range(0..NUM_CLASSES as u32);
        let mut seq = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            let token = if label == 0 {
                // Mostly low-range tokens with some noise
                if rng.gen::<f32>() < 0.8 {
                    rng.gen_range(0..500)
                } else {
                    rng.gen_range(500..VOCAB_SIZE as u32)
                }
            } else {
                // Mostly high-range tokens with some noise
                if rng.gen::<f32>() < 0.8 {
                    rng.gen_range(500..VOCAB_SIZE as u32)
                } else {
                    rng.gen_range(0..500)
                }
            };
            seq.push(token);
        }
        sequences.push(seq);
        labels.push(label);
    }

    (sequences, labels)
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    let use_monitor = args.iter().any(|a| a == "--monitor");

    println!("=== BERT Binary Classification Training ===");
    println!();

    // ---- Device selection ----
    // CUDA support requires building with --features cuda on axonml-core.
    // This example runs on CPU by default; GPU tensors are used automatically
    // when the model/data reside on a CUDA device.
    let device_name = "CPU";
    println!("Device  : {device_name}");

    // ---- Model configuration ----
    let config = BertConfig {
        vocab_size: VOCAB_SIZE,
        hidden_size: D_MODEL,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: NUM_HEADS,
        intermediate_size: INTERMEDIATE_SIZE,
        hidden_act: "gelu".to_string(),
        hidden_dropout_prob: 0.1,
        attention_probs_dropout_prob: 0.1,
        max_position_embeddings: MAX_SEQ_LEN,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
    };

    let mut model = BertForSequenceClassification::new(&config, NUM_CLASSES);

    let param_count: usize = model.parameters().iter().map(|p| p.data().numel()).sum();
    println!("Params  : {param_count}");
    println!("Config  : d_model={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}");
    println!("Data    : {NUM_TRAIN} train / {NUM_TEST} test, batch_size={BATCH_SIZE}");
    println!("Epochs  : {NUM_EPOCHS}, lr={LEARNING_RATE}");
    println!();

    // ---- Monitor (optional) ----
    let monitor = if use_monitor {
        let m = axonml::TrainingMonitor::new("BERT-cls", param_count)
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
    let (train_seqs, train_labels) = generate_data(NUM_TRAIN, MAX_SEQ_LEN, &mut rng);
    let (test_seqs, test_labels) = generate_data(NUM_TEST, MAX_SEQ_LEN, &mut rng);

    // ---- Optimizer and loss ----
    let mut optimizer = Adam::new(model.parameters(), LEARNING_RATE);
    let criterion = CrossEntropyLoss::new();

    // ---- Training loop ----
    let total_start = Instant::now();

    for epoch in 1..=NUM_EPOCHS {
        model.train();
        let epoch_start = Instant::now();

        let mut epoch_loss = 0.0_f32;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;
        let num_batches = (NUM_TRAIN + BATCH_SIZE - 1) / BATCH_SIZE;

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(NUM_TRAIN);
            let bs = end - start;

            // Flatten token sequences into a single tensor [bs, seq_len]
            let mut token_data = Vec::with_capacity(bs * MAX_SEQ_LEN);
            for seq in &train_seqs[start..end] {
                token_data.extend_from_slice(seq);
            }
            let input_ids =
                Tensor::<u32>::from_vec(token_data, &[bs, MAX_SEQ_LEN]).unwrap();

            // Labels as f32 for CrossEntropyLoss target
            let label_data: Vec<f32> =
                train_labels[start..end].iter().map(|&l| l as f32).collect();
            let targets = Variable::new(
                Tensor::<f32>::from_vec(label_data, &[bs]).unwrap(),
                false,
            );

            // Forward
            optimizer.zero_grad();
            let logits = model
                .forward_classification(&input_ids)
                .expect("forward_classification failed");

            // Loss
            let loss = criterion.compute(&logits, &targets);
            let loss_val = loss.data().to_vec()[0];

            // Backward
            loss.backward();
            optimizer.step();

            // Accuracy
            let logits_data = logits.data();
            let logits_vec = logits_data.to_vec();
            for i in 0..bs {
                let pred = if logits_vec[i * NUM_CLASSES + 1] > logits_vec[i * NUM_CLASSES] {
                    1u32
                } else {
                    0u32
                };
                if pred == train_labels[start + i] {
                    epoch_correct += 1;
                }
            }

            epoch_loss += loss_val * bs as f32;
            epoch_total += bs;
        }

        let train_loss = epoch_loss / epoch_total as f32;
        let train_acc = epoch_correct as f32 / epoch_total as f32 * 100.0;
        let epoch_time = epoch_start.elapsed();

        // ---- Evaluation ----
        model.eval();
        let mut test_loss = 0.0_f32;
        let mut test_correct = 0usize;
        let test_batches = (NUM_TEST + BATCH_SIZE - 1) / BATCH_SIZE;

        for batch_idx in 0..test_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(NUM_TEST);
            let bs = end - start;

            let mut token_data = Vec::with_capacity(bs * MAX_SEQ_LEN);
            for seq in &test_seqs[start..end] {
                token_data.extend_from_slice(seq);
            }
            let input_ids =
                Tensor::<u32>::from_vec(token_data, &[bs, MAX_SEQ_LEN]).unwrap();

            let label_data: Vec<f32> =
                test_labels[start..end].iter().map(|&l| l as f32).collect();
            let targets = Variable::new(
                Tensor::<f32>::from_vec(label_data, &[bs]).unwrap(),
                false,
            );

            let logits = model
                .forward_classification(&input_ids)
                .expect("forward_classification failed");
            let loss = criterion.compute(&logits, &targets);
            test_loss += loss.data().to_vec()[0] * bs as f32;

            let logits_data = logits.data();
            let logits_vec = logits_data.to_vec();
            for i in 0..bs {
                let pred = if logits_vec[i * NUM_CLASSES + 1] > logits_vec[i * NUM_CLASSES] {
                    1u32
                } else {
                    0u32
                };
                if pred == test_labels[start + i] {
                    test_correct += 1;
                }
            }
        }

        let val_loss = test_loss / NUM_TEST as f32;
        let test_acc = test_correct as f32 / NUM_TEST as f32 * 100.0;

        println!(
            "Epoch {epoch:>2}/{NUM_EPOCHS}  train_loss={train_loss:.4}  train_acc={train_acc:.1}%  \
             val_loss={val_loss:.4}  val_acc={test_acc:.1}%  [{:.1}s]",
            epoch_time.as_secs_f32()
        );

        if let Some(ref m) = monitor {
            m.log_epoch(
                epoch,
                train_loss,
                Some(val_loss),
                vec![("train_acc", train_acc), ("val_acc", test_acc)],
            );
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
        // Keep the process alive so the dashboard remains accessible
        std::thread::park();
    }
}
