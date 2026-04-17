//! ResNet-18 Training on Synthetic CIFAR-10 — Reference Image-Classification Example
//!
//! End-to-end training script for `ResNet::resnet18` on the `SyntheticCIFAR`
//! dataset (programmatically generated stand-in for CIFAR-10 to keep the example
//! offline). Demonstrates the standard supervised image-classification flow on
//! AxonML: dataset construction, `DataLoader` batching, GPU placement, Adam +
//! `CrossEntropyLoss`, per-epoch train/test accuracy reporting, optional
//! browser-based `TrainingMonitor`, and best/final-model checkpointing.
//!
//! Pieces:
//! - `detect_device()` — probe CUDA, fall back to CPU.
//! - `argmax_batch()` — argmax over an `[N, C]` flattened logits buffer.
//! - `onehot_to_indices()` — convert one-hot `[N, C]` targets to class index `[N]`.
//! - `main()` — builds train/test `SyntheticCIFAR::cifar10` datasets, wraps them
//!   in `DataLoader`s, instantiates `ResNet::resnet18(NUM_CLASSES)`, optimizes
//!   with `Adam` (lr=1e-3) + `CrossEntropyLoss`, then runs a `NUM_EPOCHS` loop
//!   reshaping each batch to `[N, 3, 32, 32]`, computing per-batch loss/accuracy,
//!   evaluating in `no_grad` after each epoch, and saving the best test-accuracy
//!   checkpoint plus a final checkpoint at the end.
//!
//! # File
//! `crates/axonml-vision/examples/train_resnet.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use axonml::monitor::TrainingMonitor;
use axonml_autograd::{Variable, no_grad};
use axonml_core::Device;
use axonml_data::DataLoader;
use axonml_nn::{CrossEntropyLoss, Module};
use axonml_optim::{Adam, Optimizer};
use axonml_serialize::save_model;
use axonml_tensor::Tensor;
use axonml_vision::datasets::SyntheticCIFAR;
use axonml_vision::models::ResNet;

use std::time::Instant;

// =============================================================================
// Configuration
// =============================================================================

const NUM_CLASSES: usize = 10;
const BATCH_SIZE: usize = 32;
const NUM_EPOCHS: usize = 20;
const LEARNING_RATE: f32 = 0.001;
const NUM_TRAIN: usize = 2000;
const NUM_TEST: usize = 400;
const CHECKPOINT_DIR: &str = "checkpoints/resnet18_cifar10";

// =============================================================================
// Device Detection
// =============================================================================

fn detect_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        let test = Tensor::<f32>::from_vec(vec![0.0], &[1]).unwrap();
        if test.to_device(Device::Cuda(0)).is_ok() {
            return Device::Cuda(0);
        }
    }
    Device::Cpu
}

// =============================================================================
// Helpers
// =============================================================================

/// Compute argmax over logits for a batch [N, C] returning predicted class per sample.
fn argmax_batch(logits: &[f32], num_classes: usize) -> Vec<usize> {
    let batch_size = logits.len() / num_classes;
    let mut preds = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let offset = i * num_classes;
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            if logits[offset + c] > best_val {
                best_val = logits[offset + c];
                best_idx = c;
            }
        }
        preds.push(best_idx);
    }
    preds
}

/// Extract class indices from one-hot encoded targets [N, C].
fn onehot_to_indices(targets_onehot: &[f32], num_classes: usize) -> Vec<usize> {
    argmax_batch(targets_onehot, num_classes)
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let use_monitor = std::env::args().any(|a| a == "--monitor");

    println!("=== AxonML - ResNet-18 Training on CIFAR-10 ===\n");

    // -------------------------------------------------------------------------
    // Setup: device, dataset, model, optimizer
    // -------------------------------------------------------------------------

    // Device
    let device = detect_device();
    println!("Device: {:?}", device);

    // Dataset
    println!(
        "\n1. Creating SyntheticCIFAR-10 dataset ({} train, {} test)...",
        NUM_TRAIN, NUM_TEST
    );
    let train_dataset = SyntheticCIFAR::cifar10(NUM_TRAIN);
    let test_dataset = SyntheticCIFAR::cifar10(NUM_TEST);

    // DataLoader
    println!("2. Creating DataLoader (batch_size={})...", BATCH_SIZE);
    let train_loader = DataLoader::new(train_dataset, BATCH_SIZE);
    let test_loader = DataLoader::new(test_dataset, BATCH_SIZE);
    println!("   Training batches: {}", train_loader.len());
    println!("   Test batches:     {}", test_loader.len());

    // Model
    println!("3. Creating ResNet-18 ({} classes)...", NUM_CLASSES);
    let mut model = ResNet::resnet18(NUM_CLASSES);
    model.train();
    model.to_device(device);

    let params = model.parameters();
    let total_params: usize = params
        .iter()
        .map(|p| p.variable().data().to_vec().len())
        .sum();
    println!(
        "   Parameters: {} tensors ({} total weights)",
        params.len(),
        total_params
    );

    // Optimizer + Loss
    println!(
        "4. Creating Adam optimizer (lr={}) + CrossEntropyLoss...",
        LEARNING_RATE
    );
    let mut optimizer = Adam::new(params, LEARNING_RATE);
    let criterion = CrossEntropyLoss::new();

    // Monitor (optional -- launches browser dashboard)
    let monitor = if use_monitor {
        Some(
            TrainingMonitor::new("ResNet-18 CIFAR-10", total_params)
                .total_epochs(NUM_EPOCHS)
                .batch_size(BATCH_SIZE)
                .launch(),
        )
    } else {
        None
    };

    // Checkpoint dir
    std::fs::create_dir_all(CHECKPOINT_DIR).ok();

    // -------------------------------------------------------------------------
    // Training loop
    // -------------------------------------------------------------------------

    // Training
    println!("5. Training for {} epochs...\n", NUM_EPOCHS);
    let train_start = Instant::now();
    let mut best_test_acc = 0.0f32;

    for epoch in 0..NUM_EPOCHS {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut batch_count = 0usize;

        model.train();

        for batch in train_loader.iter() {
            let bs = batch.data.shape()[0];

            // Reshape to [N, 3, 32, 32]
            let input_data = batch.data.to_vec();
            let input_tensor = Tensor::from_vec(input_data, &[bs, 3, 32, 32]).unwrap();
            let input = Variable::new(
                if device.is_gpu() {
                    input_tensor.to_device(device).unwrap()
                } else {
                    input_tensor
                },
                true,
            );

            // Target: one-hot [N, 10] -> class indices [N]
            let target_onehot = batch.targets.to_vec();
            let target_indices = onehot_to_indices(&target_onehot, NUM_CLASSES);
            let target_f32: Vec<f32> = target_indices.iter().map(|&i| i as f32).collect();
            let target_tensor = Tensor::from_vec(target_f32, &[bs]).unwrap();
            let target = Variable::new(
                if device.is_gpu() {
                    target_tensor.to_device(device).unwrap()
                } else {
                    target_tensor
                },
                false,
            );

            // Forward
            let output = model.forward(&input);

            // Loss
            let loss = criterion.compute(&output, &target);
            let loss_val = loss.data().to_vec()[0];
            total_loss += loss_val;
            batch_count += 1;

            // Training accuracy
            let out_data = output.data().to_vec();
            let preds = argmax_batch(&out_data, NUM_CLASSES);
            for (pred, &true_label) in preds.iter().zip(target_indices.iter()) {
                if *pred == true_label {
                    correct += 1;
                }
                total += 1;
            }

            // Backward + step
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }

        let epoch_time = epoch_start.elapsed();
        let avg_loss = total_loss / batch_count as f32;
        let train_acc = 100.0 * correct as f32 / total as f32;
        let samples_per_sec = total as f64 / epoch_time.as_secs_f64();

        // ---------------------------------------------------------------------
        // Per-epoch test evaluation
        // ---------------------------------------------------------------------

        // Test evaluation
        model.eval();
        let (test_correct, test_total) = no_grad(|| {
            let mut tc = 0usize;
            let mut tt = 0usize;

            for batch in test_loader.iter() {
                let bs = batch.data.shape()[0];

                let input_data = batch.data.to_vec();
                let input_tensor = Tensor::from_vec(input_data, &[bs, 3, 32, 32]).unwrap();
                let input = Variable::new(
                    if device.is_gpu() {
                        input_tensor.to_device(device).unwrap()
                    } else {
                        input_tensor
                    },
                    false,
                );

                let output = model.forward(&input);
                let out_data = output.data().to_vec();
                let preds = argmax_batch(&out_data, NUM_CLASSES);

                let target_onehot = batch.targets.to_vec();
                let true_labels = onehot_to_indices(&target_onehot, NUM_CLASSES);

                for (pred, true_label) in preds.iter().zip(true_labels.iter()) {
                    if pred == true_label {
                        tc += 1;
                    }
                    tt += 1;
                }
            }

            (tc, tt)
        });

        let test_acc = 100.0 * test_correct as f32 / test_total as f32;

        println!(
            "   Epoch {:2}/{}: Loss={:.4}  TrainAcc={:.1}%  TestAcc={:.1}%  ({:.0} samples/s, {:.2}s)",
            epoch + 1,
            NUM_EPOCHS,
            avg_loss,
            train_acc,
            test_acc,
            samples_per_sec,
            epoch_time.as_secs_f64(),
        );

        // Log to monitor
        if let Some(ref mon) = monitor {
            mon.log_epoch(
                epoch,
                avg_loss,
                None,
                vec![("train_acc", train_acc), ("test_acc", test_acc)],
            );
        }

        // Checkpoint on best test accuracy
        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            let path = format!("{}/best_model.axonml", CHECKPOINT_DIR);
            match save_model(&model, &path) {
                Ok(()) => println!(
                    "   -> Saved best model (test_acc={:.1}%) to {}",
                    test_acc, path
                ),
                Err(e) => println!("   -> Warning: could not save checkpoint: {}", e),
            }
        }
    }

    let total_time = train_start.elapsed();

    // =========================================================================
    // Finalization
    // =========================================================================

    // Final checkpoint
    let final_path = format!("{}/final_model.axonml", CHECKPOINT_DIR);
    match save_model(&model, &final_path) {
        Ok(()) => println!("\n   Saved final model to {}", final_path),
        Err(e) => println!("\n   Warning: could not save final model: {}", e),
    }

    if let Some(ref mon) = monitor {
        mon.set_status("complete");
    }

    println!("\n=== Training Complete! ===");
    println!("   Device:             {:?}", device);
    println!("   Total time:         {:.2}s", total_time.as_secs_f64());
    println!("   Best test accuracy: {:.2}%", best_test_acc);
    println!("   Checkpoints:        {}/", CHECKPOINT_DIR);
}
