//! MNIST Training Example — LeNet on SyntheticMNIST with GPU Support
//!
//! Demonstrates a complete training pipeline using the AxonML framework:
//! device detection (CUDA or CPU), `SyntheticMNIST` dataset creation with
//! configurable train/test sizes, batched `DataLoader` iteration, `LeNet`
//! model construction with parameter counting and device placement, `Adam`
//! optimizer with `CrossEntropyLoss`, a full training loop (forward pass,
//! one-hot to class-index target conversion, loss computation, backward
//! pass, optimizer step) with per-epoch loss/accuracy/throughput reporting,
//! and a `no_grad` evaluation pass over the test set with final accuracy.
//!
//! # File
//! `crates/axonml/examples/mnist_training.rs`
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

use axonml::prelude::*;
use std::time::Instant;

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    println!("=== AxonML - MNIST Training (LeNet) ===\n");

    // -------------------------------------------------------------------------
    // Device Detection
    // -------------------------------------------------------------------------

    // Detect device
    #[cfg(feature = "cuda")]
    let device = {
        let cuda = Device::Cuda(0);
        if cuda.is_available() {
            println!("GPU detected: using CUDA device 0");
            cuda
        } else {
            println!("CUDA feature enabled but no GPU available, using CPU");
            Device::Cpu
        }
    };
    #[cfg(not(feature = "cuda"))]
    let device = {
        println!("Using CPU (compile with --features cuda for GPU)");
        Device::Cpu
    };

    // -------------------------------------------------------------------------
    // Dataset and DataLoader Setup
    // -------------------------------------------------------------------------

    // 1. Create dataset
    let num_train = 2000;
    let num_test = 400;
    println!("\n1. Creating SyntheticMNIST dataset ({num_train} train, {num_test} test)...");
    let train_dataset = SyntheticMNIST::new(num_train);
    let test_dataset = SyntheticMNIST::new(num_test);

    // 2. Create DataLoader
    let batch_size = 64;
    println!("2. Creating DataLoader (batch_size={batch_size})...");
    let train_loader = DataLoader::new(train_dataset, batch_size);
    let test_loader = DataLoader::new(test_dataset, batch_size);
    println!("   Training batches: {}", train_loader.len());

    // -------------------------------------------------------------------------
    // Model, Optimizer, and Loss
    // -------------------------------------------------------------------------

    // 3. Create LeNet model and move to device
    println!("3. Creating LeNet model...");
    let model = LeNet::new();
    model.to_device(device);
    let params = model.parameters();
    let total_params: usize = params
        .iter()
        .map(|p| p.variable().data().to_vec().len())
        .sum();
    println!(
        "   Parameters: {} ({} total weights)",
        params.len(),
        total_params
    );
    println!("   Device: {:?}", device);

    // 4. Create optimizer and loss
    println!("4. Creating Adam optimizer (lr=0.001) + CrossEntropyLoss...");
    let mut optimizer = Adam::new(params, 0.001);
    let criterion = CrossEntropyLoss::new();

    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------

    // 5. Training loop
    let epochs = 10;
    println!("5. Training for {epochs} epochs...\n");

    let train_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut batch_count = 0;

        for batch in train_loader.iter() {
            let bs = batch.data.shape()[0];

            // Reshape to [N, 1, 28, 28] and create Variable
            let input_data = batch.data.to_vec();
            let input_tensor = Tensor::from_vec(input_data, &[bs, 1, 28, 28]).unwrap();
            let input = Variable::new(
                if device.is_gpu() {
                    input_tensor.to_device(device).unwrap()
                } else {
                    input_tensor
                },
                true,
            );

            // Target: convert one-hot [N, 10] to class indices [N]
            let target_onehot = batch.targets.to_vec();
            let mut target_indices = vec![0.0f32; bs];
            for i in 0..bs {
                let offset = i * 10;
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for c in 0..10 {
                    if target_onehot[offset + c] > max_val {
                        max_val = target_onehot[offset + c];
                        max_idx = c;
                    }
                }
                target_indices[i] = max_idx as f32;
            }
            let target_tensor = Tensor::from_vec(target_indices.clone(), &[bs]).unwrap();
            let target = Variable::new(
                if device.is_gpu() {
                    target_tensor.to_device(device).unwrap()
                } else {
                    target_tensor
                },
                false,
            );

            // Forward pass
            let output = model.forward(&input);

            // Cross-entropy loss
            let loss = criterion.compute(&output, &target);

            let loss_val = loss.data().to_vec()[0];
            total_loss += loss_val;
            batch_count += 1;

            // Compute training accuracy
            let out_data = output.data().to_vec();
            for i in 0..bs {
                let offset = i * 10;
                let mut pred = 0;
                let mut pred_val = f32::NEG_INFINITY;
                for c in 0..10 {
                    if out_data[offset + c] > pred_val {
                        pred_val = out_data[offset + c];
                        pred = c;
                    }
                }
                if pred == target_indices[i] as usize {
                    correct += 1;
                }
                total += 1;
            }

            // Backward pass
            loss.backward();

            // Update weights
            optimizer.step();
            optimizer.zero_grad();
        }

        let epoch_time = epoch_start.elapsed();
        let avg_loss = total_loss / batch_count as f32;
        let accuracy = 100.0 * correct as f32 / total as f32;
        let samples_per_sec = total as f64 / epoch_time.as_secs_f64();

        println!(
            "   Epoch {:2}/{}: Loss={:.4}  Acc={:.1}%  ({:.0} samples/s, {:.2}s)",
            epoch + 1,
            epochs,
            avg_loss,
            accuracy,
            samples_per_sec,
            epoch_time.as_secs_f64(),
        );
    }

    let train_time = train_start.elapsed();
    println!("\n   Total training time: {:.2}s", train_time.as_secs_f64());

    // -------------------------------------------------------------------------
    // Test Evaluation
    // -------------------------------------------------------------------------

    // 6. Test evaluation
    println!("\n6. Evaluating on test set...");

    // Disable gradient computation for evaluation
    let (correct, total) = no_grad(|| {
        let mut correct = 0usize;
        let mut total = 0usize;

        for batch in test_loader.iter() {
            let bs = batch.data.shape()[0];

            let input_data = batch.data.to_vec();
            let input_tensor = Tensor::from_vec(input_data, &[bs, 1, 28, 28]).unwrap();
            let input = Variable::new(
                if device.is_gpu() {
                    input_tensor.to_device(device).unwrap()
                } else {
                    input_tensor
                },
                false,
            );

            let target_onehot = batch.targets.to_vec();
            let output = model.forward(&input);
            let out_data = output.data().to_vec();

            for i in 0..bs {
                // Prediction: argmax of output
                let offset = i * 10;
                let mut pred = 0;
                let mut pred_val = f32::NEG_INFINITY;
                for c in 0..10 {
                    if out_data[offset + c] > pred_val {
                        pred_val = out_data[offset + c];
                        pred = c;
                    }
                }

                // True label: argmax of one-hot target
                let mut true_label = 0;
                let mut true_val = f32::NEG_INFINITY;
                for c in 0..10 {
                    if target_onehot[i * 10 + c] > true_val {
                        true_val = target_onehot[i * 10 + c];
                        true_label = c;
                    }
                }

                if pred == true_label {
                    correct += 1;
                }
                total += 1;
            }
        }

        (correct, total)
    });

    let test_accuracy = 100.0 * correct as f32 / total as f32;
    println!(
        "   Test Accuracy: {}/{} ({:.2}%)",
        correct, total, test_accuracy
    );

    println!("\n=== Training Complete! ===");
    println!("   Device: {:?}", device);
    println!("   Final test accuracy: {:.2}%", test_accuracy);
}
