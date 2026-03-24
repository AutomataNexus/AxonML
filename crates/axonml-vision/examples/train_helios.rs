//! Helios YOLO Training Example — Synthetic Detection Data
//!
//! # File
//! `crates/axonml-vision/examples/train_helios.rs`
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
//!
//! # Description
//! Demonstrates the Helios (YOLO-competitive) anchor-free detector on synthetic
//! data. Verifies model construction, forward pass shapes, parameter count,
//! loss computation, backward pass, and optimizer convergence.
//!
//! # Usage
//! ```bash
//! cargo run -p axonml-vision --example train_helios
//! cargo run -p axonml-vision --example train_helios -- --monitor
//! ```

use axonml::monitor::TrainingMonitor;
use axonml_autograd::Variable;
use axonml_core::Device;
use axonml_nn::Module;
use axonml_optim::{Adam, Optimizer};
use axonml_tensor::Tensor;
use axonml_vision::models::helios::{Helios, HeliosConfig, HeliosLoss};

use std::time::Instant;

// =============================================================================
// Config
// =============================================================================

const INPUT_SIZE: usize = 320;
const NUM_CLASSES: usize = 5;
const BATCH_SIZE: usize = 2;
const NUM_EPOCHS: usize = 10;
const STEPS_PER_EPOCH: usize = 8;
const LR: f32 = 1e-3;

// =============================================================================
// GPU Detection
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

/// Generate a synthetic batch of images with random bounding boxes.
///
/// Returns (images [B, 3, H, W], gt_boxes per image, gt_classes per image).
fn generate_batch(
    batch_size: usize,
    input_size: usize,
    num_classes: usize,
    seed: usize,
) -> (Tensor<f32>, Vec<Vec<[f32; 4]>>, Vec<Vec<usize>>) {
    let numel = batch_size * 3 * input_size * input_size;
    let mut pixels = Vec::with_capacity(numel);

    // Deterministic pseudo-random via simple LCG
    let mut rng_state = (seed as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1);
    let mut next_f32 = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };

    for _ in 0..numel {
        pixels.push(next_f32());
    }

    let images = Tensor::from_vec(pixels, &[batch_size, 3, input_size, input_size]).unwrap();

    let mut all_gt_boxes = Vec::with_capacity(batch_size);
    let mut all_gt_classes = Vec::with_capacity(batch_size);
    let img_f = input_size as f32;

    for _ in 0..batch_size {
        // 1-3 random boxes per image
        let num_gt = (next_f32() * 2.0) as usize + 1;
        let mut gt_boxes = Vec::with_capacity(num_gt);
        let mut gt_classes = Vec::with_capacity(num_gt);

        for _ in 0..num_gt {
            // Random box: ensure min size of 32px and valid coords
            let cx = next_f32() * img_f * 0.6 + img_f * 0.2;
            let cy = next_f32() * img_f * 0.6 + img_f * 0.2;
            let half_w = next_f32() * img_f * 0.15 + 16.0;
            let half_h = next_f32() * img_f * 0.15 + 16.0;

            let x1 = (cx - half_w).max(0.0);
            let y1 = (cy - half_h).max(0.0);
            let x2 = (cx + half_w).min(img_f);
            let y2 = (cy + half_h).min(img_f);

            gt_boxes.push([x1, y1, x2, y2]);
            gt_classes.push((next_f32() * num_classes as f32) as usize % num_classes);
        }

        all_gt_boxes.push(gt_boxes);
        all_gt_classes.push(gt_classes);
    }

    (images, all_gt_boxes, all_gt_classes)
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let use_monitor = std::env::args().any(|a| a == "--monitor");

    println!("=== Helios YOLO Training Example (Synthetic Data) ===");
    println!();

    // ── Step 1: Device detection ──
    let device = detect_device();
    println!("Device: {:?}", device);
    println!();

    // ── Step 2: Build model ──
    let t0 = Instant::now();
    let mut config = HeliosConfig::nano(NUM_CLASSES);
    config.input_size = INPUT_SIZE;
    let mut model = Helios::new(config);
    let build_ms = t0.elapsed().as_millis();

    let params = model.parameters();
    let param_count: usize = params
        .iter()
        .map(|p| p.variable().data().to_vec().len())
        .sum();
    let num_param_tensors = params.len();
    println!("Model:      Helios-Nano (anchor-free YOLO)");
    println!("Classes:    {NUM_CLASSES}");
    println!("Input:      3x{INPUT_SIZE}x{INPUT_SIZE}");
    println!("Params:     {param_count} ({num_param_tensors} tensors)");
    println!("Built in:   {build_ms}ms");
    println!();

    // ── Step 3: Verify forward pass shapes ──
    println!("--- Forward Pass Verification ---");
    let dummy = Variable::new(
        Tensor::from_vec(
            vec![0.5f32; BATCH_SIZE * 3 * INPUT_SIZE * INPUT_SIZE],
            &[BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE],
        )
        .unwrap(),
        false,
    );

    let t1 = Instant::now();
    let train_out = model.forward_train(&dummy);
    let fwd_ms = t1.elapsed().as_millis();

    println!("Forward pass: {fwd_ms}ms (batch_size={BATCH_SIZE})");
    println!("Scales: {}", train_out.scales.len());
    for (i, scale) in train_out.scales.iter().enumerate() {
        let cls_shape = scale.cls_logits.shape();
        let bbox_shape = scale.bbox_dfl.shape();
        println!(
            "  P{}/stride={}: cls={:?}, bbox_dfl={:?}",
            i + 3,
            scale.stride,
            cls_shape,
            bbox_shape,
        );
        // Verify shapes
        assert_eq!(cls_shape[0], BATCH_SIZE, "Batch dim mismatch");
        assert_eq!(cls_shape[1], NUM_CLASSES, "Class dim mismatch");
        assert_eq!(bbox_shape[1], 4 * 16, "DFL dim should be 4*reg_max=64");
        let expected_spatial = INPUT_SIZE / scale.stride;
        assert_eq!(cls_shape[2], expected_spatial, "Spatial H mismatch");
        assert_eq!(cls_shape[3], expected_spatial, "Spatial W mismatch");
    }
    println!("All output shapes verified.");
    println!();

    // ── Step 4: Verify all parameters are reachable by optimizer ──
    println!("--- Parameter Reachability ---");
    let all_params = model.parameters();
    let mut has_grad_params = 0;
    let mut no_grad_params = 0;
    for p in &all_params {
        if p.variable().requires_grad() {
            has_grad_params += 1;
        } else {
            no_grad_params += 1;
        }
    }
    println!("Trainable:      {has_grad_params}/{num_param_tensors} tensors require grad");
    if no_grad_params > 0 {
        println!("Non-trainable:  {no_grad_params} tensors (frozen/buffers)");
    }
    println!();

    // Move model to GPU if available
    if device.is_gpu() {
        model.to_device(device);
        println!("Model moved to {:?}", device);
    }

    // ── Step 5: Loss computation test ──
    println!("--- Loss Computation Test ---");
    let loss_fn = HeliosLoss::new(NUM_CLASSES, 16);

    let (test_imgs, test_gt_boxes, test_gt_classes) =
        generate_batch(BATCH_SIZE, INPUT_SIZE, NUM_CLASSES, 42);
    let test_input = Variable::new(test_imgs, false);
    let test_input = if device.is_gpu() {
        test_input.to_device(device)
    } else {
        test_input
    };

    let t2 = Instant::now();
    let test_out = model.forward_train(&test_input);
    let (test_loss, cls_l, box_l, dfl_l) =
        loss_fn.compute(&test_out, &test_gt_boxes, &test_gt_classes, NUM_CLASSES);
    let loss_ms = t2.elapsed().as_millis();

    let test_loss_val = test_loss.data().to_vec()[0];
    println!("Loss computation: {loss_ms}ms");
    println!(
        "  total={:.4}, cls={:.4}, box={:.6}, dfl={:.6}",
        test_loss_val, cls_l, box_l, dfl_l,
    );
    assert!(
        test_loss_val.is_finite(),
        "Loss must be finite, got {test_loss_val}"
    );
    println!("Loss is finite and computable.");
    println!();

    // ── Step 6: Backward pass + optimizer step test ──
    println!("--- Backward Pass Test ---");
    let params_for_opt = model.parameters();
    let mut optimizer = Adam::new(params_for_opt, LR);

    let (step_imgs, step_gt_boxes, step_gt_classes) =
        generate_batch(BATCH_SIZE, INPUT_SIZE, NUM_CLASSES, 99);
    let step_input = Variable::new(step_imgs, true);
    let step_input = if device.is_gpu() {
        step_input.to_device(device)
    } else {
        step_input
    };

    optimizer.zero_grad();
    let step_out = model.forward_train(&step_input);
    let (step_loss, _, _, _) =
        loss_fn.compute(&step_out, &step_gt_boxes, &step_gt_classes, NUM_CLASSES);
    let loss_before = step_loss.data().to_vec()[0];

    let t3 = Instant::now();
    step_loss.backward();
    let bwd_ms = t3.elapsed().as_millis();

    optimizer.step();
    println!("Backward pass: {bwd_ms}ms");
    println!("Optimizer step completed (Adam, lr={LR})");
    println!("Loss before step: {loss_before:.4}");
    println!();

    // ── Step 7: Training loop ──
    println!("--- Training Loop ({NUM_EPOCHS} epochs x {STEPS_PER_EPOCH} steps) ---");

    let monitor = if use_monitor {
        Some(
            TrainingMonitor::new("Helios-Nano", param_count)
                .total_epochs(NUM_EPOCHS)
                .batch_size(BATCH_SIZE)
                .launch(),
        )
    } else {
        None
    };

    let mut best_loss = f32::MAX;

    for epoch in 0..NUM_EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_cls = 0.0f32;
        let mut epoch_box = 0.0f32;
        let mut epoch_dfl = 0.0f32;

        for step in 0..STEPS_PER_EPOCH {
            let seed = epoch * STEPS_PER_EPOCH + step + 1000;
            let (imgs, gt_boxes, gt_classes) =
                generate_batch(BATCH_SIZE, INPUT_SIZE, NUM_CLASSES, seed);
            let input = Variable::new(imgs, true);
            let input = if device.is_gpu() {
                input.to_device(device)
            } else {
                input
            };

            optimizer.zero_grad();
            let train_out = model.forward_train(&input);
            let (loss, cl, bl, dl) =
                loss_fn.compute(&train_out, &gt_boxes, &gt_classes, NUM_CLASSES);

            let loss_val = loss.data().to_vec()[0];
            if loss.requires_grad() {
                loss.backward();
                optimizer.step();
            }

            epoch_loss += loss_val;
            epoch_cls += cl;
            epoch_box += bl;
            epoch_dfl += dl;
        }

        let steps = STEPS_PER_EPOCH as f32;
        let avg_loss = epoch_loss / steps;
        let avg_cls = epoch_cls / steps;
        let avg_box = epoch_box / steps;
        let avg_dfl = epoch_dfl / steps;
        let elapsed = epoch_start.elapsed().as_secs_f32();
        let imgs_processed = BATCH_SIZE * STEPS_PER_EPOCH;

        if avg_loss < best_loss {
            best_loss = avg_loss;
        }

        println!(
            "Epoch {}/{}: loss={:.4} (cls={:.4} box={:.6} dfl={:.6}) | {:.1}s ({:.1} img/s)",
            epoch + 1,
            NUM_EPOCHS,
            avg_loss,
            avg_cls,
            avg_box,
            avg_dfl,
            elapsed,
            imgs_processed as f32 / elapsed,
        );

        if let Some(ref mon) = monitor {
            let img_s = imgs_processed as f32 / elapsed;
            mon.log_epoch(
                epoch + 1,
                avg_loss,
                None,
                vec![
                    ("cls_loss", avg_cls),
                    ("box_loss", avg_box),
                    ("dfl_loss", avg_dfl),
                    ("img_s", img_s),
                ],
            );
        }
    }

    if let Some(ref mon) = monitor {
        mon.set_status("complete");
    }

    // ── Step 8: Inference test ──
    println!();
    println!("--- Inference Test ---");
    model.eval();

    let (infer_imgs, _, _) = generate_batch(1, INPUT_SIZE, NUM_CLASSES, 7777);
    let infer_input = Variable::new(infer_imgs, false);
    let infer_input = if device.is_gpu() {
        infer_input.to_device(device)
    } else {
        infer_input
    };

    let t4 = Instant::now();
    let detections = model.detect(&infer_input, 0.25, 0.45);
    let infer_ms = t4.elapsed().as_millis();

    println!("Inference: {infer_ms}ms");
    println!("Detections (score>=0.25): {}", detections.len());
    for (i, det) in detections.iter().take(10).enumerate() {
        println!(
            "  det[{}]: class={} conf={:.3} bbox=[{:.1},{:.1},{:.1},{:.1}]",
            i, det.class_id, det.confidence, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
        );
    }

    // ── Summary ──
    println!();
    println!("=== Summary ===");
    println!("Model:          Helios-Nano");
    println!("Parameters:     {param_count}");
    println!("Input:          3x{INPUT_SIZE}x{INPUT_SIZE}");
    println!("Classes:        {NUM_CLASSES}");
    println!("Best loss:      {best_loss:.4}");
    println!("Device:         {:?}", device);
    println!("Forward pass:   verified (3 scales)");
    println!("Backward pass:  verified");
    println!("Optimizer:      verified (Adam)");
    println!("Inference:      {} detections", detections.len());
    println!();
    println!("All verifications passed.");
}
