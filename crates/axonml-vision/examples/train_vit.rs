//! Vision Transformer (ViT) Training on Synthetic CIFAR-like Data
//!
//! # File
//! `crates/axonml-vision/examples/train_vit.rs`
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

use axonml::monitor::TrainingMonitor;
use axonml_autograd::Variable;
use axonml_core::Device;
use axonml_nn::{CrossEntropyLoss, Module};
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::Tensor;
use axonml_vision::models::transformer::VisionTransformer;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

// =============================================================================
// Config
// =============================================================================

const IMAGE_SIZE: usize = 32;
const PATCH_SIZE: usize = 8;
const IN_CHANNELS: usize = 3;
const NUM_CLASSES: usize = 10;
const D_MODEL: usize = 128;
const NHEAD: usize = 4;
const NUM_LAYERS: usize = 4;
const DIM_FF: usize = 256;
const DROPOUT: f32 = 0.1;

const NUM_TRAIN: usize = 512;
const NUM_VAL: usize = 128;
const BATCH_SIZE: usize = 32;
const NUM_EPOCHS: usize = 20;
const LR: f32 = 1e-4;

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
// Synthetic Data
// =============================================================================

/// Generate synthetic CIFAR-like data: random 3x32x32 images with random labels.
fn generate_synthetic_data(num_samples: usize, rng: &mut StdRng) -> (Vec<Vec<f32>>, Vec<usize>) {
    let pixels = IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE;
    let mut images = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let img: Vec<f32> = (0..pixels).map(|_| rng.gen_range(-1.0..1.0)).collect();
        images.push(img);
        labels.push(rng.gen_range(0..NUM_CLASSES));
    }
    (images, labels)
}

/// Build a batched tensor pair from a slice of images and labels.
fn make_batch(
    images: &[Vec<f32>],
    labels: &[usize],
    start: usize,
    batch_size: usize,
) -> (Variable, Variable) {
    let end = (start + batch_size).min(images.len());
    let bs = end - start;
    let pixels = IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE;

    let mut img_data = Vec::with_capacity(bs * pixels);
    let mut lbl_data = Vec::with_capacity(bs);
    for i in start..end {
        img_data.extend_from_slice(&images[i]);
        lbl_data.push(labels[i] as f32);
    }

    let img_tensor =
        Tensor::from_vec(img_data, &[bs, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]).unwrap();
    let lbl_tensor = Tensor::from_vec(lbl_data, &[bs]).unwrap();

    (
        Variable::new(img_tensor, true),
        Variable::new(lbl_tensor, false),
    )
}

// =============================================================================
// Accuracy
// =============================================================================

fn compute_accuracy(logits: &Variable, labels: &Variable) -> f32 {
    let logits_data = logits.data().to_vec();
    let labels_data = labels.data().to_vec();
    let batch_size = logits.shape()[0];
    let num_classes = logits.shape()[1];

    let mut correct = 0usize;
    for b in 0..batch_size {
        let mut best_class = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let score = logits_data[b * num_classes + c];
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }
        if best_class == labels_data[b] as usize {
            correct += 1;
        }
    }
    correct as f32 / batch_size as f32
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_monitor = args.iter().any(|a| a == "--monitor");

    let device = detect_device();
    println!("===== ViT Training on Synthetic CIFAR-10 =====");
    println!("Device:       {:?}", device);
    println!("Image size:   {}x{}", IMAGE_SIZE, IMAGE_SIZE);
    println!("Patch size:   {}", PATCH_SIZE);
    println!("d_model:      {}", D_MODEL);
    println!("Heads:        {}", NHEAD);
    println!("Layers:       {}", NUM_LAYERS);
    println!("Batch size:   {}", BATCH_SIZE);
    println!("Epochs:       {}", NUM_EPOCHS);
    println!("LR:           {}", LR);
    println!();

    // Build model
    let mut vit = VisionTransformer::new(
        IMAGE_SIZE,
        PATCH_SIZE,
        IN_CHANNELS,
        NUM_CLASSES,
        D_MODEL,
        NHEAD,
        NUM_LAYERS,
        DIM_FF,
        DROPOUT,
    );
    let params = vit.parameters();
    let param_count: usize = params.iter().map(|p| p.data().numel()).sum();
    println!("Parameters:   {}", param_count);

    // Optimizer and loss
    let mut optimizer = Adam::new(params, LR);
    let criterion = CrossEntropyLoss::new();

    // Generate synthetic data
    let mut rng = StdRng::seed_from_u64(42);
    let (train_images, train_labels) = generate_synthetic_data(NUM_TRAIN, &mut rng);
    let (val_images, val_labels) = generate_synthetic_data(NUM_VAL, &mut rng);
    println!(
        "Train samples: {}  Val samples: {}",
        train_images.len(),
        val_images.len()
    );
    println!();

    // Optional monitor
    let monitor = if use_monitor {
        Some(
            TrainingMonitor::new("ViT-CIFAR10", param_count)
                .total_epochs(NUM_EPOCHS)
                .batch_size(BATCH_SIZE)
                .launch(),
        )
    } else {
        None
    };

    // Training loop
    vit.train();
    let num_batches = (NUM_TRAIN + BATCH_SIZE - 1) / BATCH_SIZE;

    for epoch in 1..=NUM_EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;
            let (images, labels) = make_batch(&train_images, &train_labels, start, BATCH_SIZE);
            let bs = images.shape()[0];

            optimizer.zero_grad();

            let logits = vit.forward(&images);
            let loss = criterion.compute(&logits, &labels);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.data().to_vec()[0] * bs as f32;

            // Accuracy
            let acc = compute_accuracy(&logits, &labels);
            epoch_correct += (acc * bs as f32) as usize;
            epoch_total += bs;
        }

        let avg_loss = epoch_loss / epoch_total as f32;
        let train_acc = epoch_correct as f32 / epoch_total as f32 * 100.0;

        // Validation
        vit.eval();
        let mut val_loss = 0.0f32;
        let mut val_correct = 0usize;
        let mut val_total = 0usize;
        let val_batches = (NUM_VAL + BATCH_SIZE - 1) / BATCH_SIZE;

        for batch_idx in 0..val_batches {
            let start = batch_idx * BATCH_SIZE;
            let (images, labels) = make_batch(&val_images, &val_labels, start, BATCH_SIZE);
            let bs = images.shape()[0];

            let logits = vit.forward(&images);
            let loss = criterion.compute(&logits, &labels);

            val_loss += loss.data().to_vec()[0] * bs as f32;
            let acc = compute_accuracy(&logits, &labels);
            val_correct += (acc * bs as f32) as usize;
            val_total += bs;
        }

        let avg_val_loss = val_loss / val_total as f32;
        let val_acc = val_correct as f32 / val_total as f32 * 100.0;

        vit.train();

        let elapsed = epoch_start.elapsed().as_secs_f32();
        println!(
            "Epoch {:2}/{} | loss: {:.4} | acc: {:5.1}% | val_loss: {:.4} | val_acc: {:5.1}% | {:.1}s",
            epoch, NUM_EPOCHS, avg_loss, train_acc, avg_val_loss, val_acc, elapsed
        );

        if let Some(ref mon) = monitor {
            mon.log_epoch(
                epoch,
                avg_loss,
                Some(avg_val_loss),
                vec![("train_acc", train_acc), ("val_acc", val_acc)],
            );
        }
    }

    if let Some(ref mon) = monitor {
        mon.set_status("complete");
    }

    println!();
    println!("Training complete.");
}
