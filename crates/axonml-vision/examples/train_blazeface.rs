//! BlazeFace Training on WIDER FACE — with checkpointing
//!
//! # File
//! `crates/axonml-vision/examples/train_blazeface.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml::monitor::TrainingMonitor;
use axonml_autograd::Variable;
use axonml_core::Device;
use axonml_nn::{Module, SmoothL1Loss};
use axonml_optim::{Adam, Optimizer};
use axonml_serialize::{load_state_dict, save_model};
use axonml_tensor::Tensor;
use axonml_vision::datasets::WiderFaceDataset;
use axonml_vision::losses::FocalLoss;
use axonml_vision::models::blazeface::BlazeFace;

use rayon::prelude::*;
use std::time::Instant;

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
// Config
// =============================================================================

const INPUT_SIZE: usize = 128;
const NUM_SAMPLES: usize = 12000;
const BATCH_SIZE: usize = 32;
const NUM_EPOCHS: usize = 80;
const LR: f32 = 1e-3;
const CONF_THRESHOLD: f32 = 0.10;
const NMS_THRESHOLD: f32 = 0.3;
const CHECKPOINT_DIR: &str = "checkpoints";
const DATASET_ROOT: &str = "/opt/datasets/wider_face";
const LOG_INTERVAL: usize = 100;

// =============================================================================
// Anchor ←→ GT Matching
// =============================================================================

fn iou_box_anchor(gt: &[f32; 4], anchor: &[f32; 4]) -> f32 {
    let (acx, acy, aw, ah) = (anchor[0], anchor[1], anchor[2], anchor[3]);
    let ax1 = acx - aw / 2.0;
    let ay1 = acy - ah / 2.0;
    let ax2 = acx + aw / 2.0;
    let ay2 = acy + ah / 2.0;

    let ix1 = gt[0].max(ax1);
    let iy1 = gt[1].max(ay1);
    let ix2 = gt[2].min(ax2);
    let iy2 = gt[3].min(ay2);

    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1]);
    let area_a = aw * ah;
    let union = area_gt + area_a - inter;

    if union > 0.0 { inter / union } else { 0.0 }
}

fn assign_anchors(
    gt_boxes: &[[f32; 4]],
    anchors: &[[f32; 4]],
    input_size: f32,
) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
    let num_anchors = anchors.len();
    let mut cls_target = vec![0.0f32; num_anchors];
    let mut bbox_target = vec![0.0f32; num_anchors * 4];
    let mut positive = vec![false; num_anchors];

    let gt_pixel: Vec<[f32; 4]> = gt_boxes
        .iter()
        .map(|b| {
            [
                b[0] * input_size,
                b[1] * input_size,
                b[2] * input_size,
                b[3] * input_size,
            ]
        })
        .collect();

    if gt_pixel.is_empty() {
        return (cls_target, bbox_target, positive);
    }

    // Match anchors to GT by best IoU
    for (ai, anchor) in anchors.iter().enumerate() {
        let mut best_iou = 0.0f32;
        let mut best_gt = 0usize;
        for (gi, gt) in gt_pixel.iter().enumerate() {
            let iou = iou_box_anchor(gt, anchor);
            if iou > best_iou {
                best_iou = iou;
                best_gt = gi;
            }
        }

        if best_iou > 0.35 {
            positive[ai] = true;
            cls_target[ai] = 1.0;
            encode_bbox(&gt_pixel[best_gt], anchor, &mut bbox_target, ai);
        }
    }

    // Ensure every GT matches at least one anchor
    for gt in &gt_pixel {
        let mut best_iou = 0.0f32;
        let mut best_ai = 0usize;
        for (ai, anchor) in anchors.iter().enumerate() {
            let iou = iou_box_anchor(gt, anchor);
            if iou > best_iou {
                best_iou = iou;
                best_ai = ai;
            }
        }
        if !positive[best_ai] {
            positive[best_ai] = true;
            cls_target[best_ai] = 1.0;
            encode_bbox(gt, &anchors[best_ai], &mut bbox_target, best_ai);
        }
    }

    (cls_target, bbox_target, positive)
}

fn encode_bbox(gt: &[f32; 4], anchor: &[f32; 4], target: &mut [f32], ai: usize) {
    let gt_cx = (gt[0] + gt[2]) / 2.0;
    let gt_cy = (gt[1] + gt[3]) / 2.0;
    let gt_w = (gt[2] - gt[0]).max(1.0);
    let gt_h = (gt[3] - gt[1]).max(1.0);
    let (acx, acy, aw, ah) = (anchor[0], anchor[1], anchor[2], anchor[3]);
    target[ai * 4] = (gt_cx - acx) / aw;
    target[ai * 4 + 1] = (gt_cy - acy) / ah;
    target[ai * 4 + 2] = (gt_w / aw).ln();
    target[ai * 4 + 3] = (gt_h / ah).ln();
}

// =============================================================================
// Checkpoint helpers
// =============================================================================

fn save_checkpoint(model: &BlazeFace, epoch: usize, loss: f32) {
    std::fs::create_dir_all(CHECKPOINT_DIR).ok();

    let path = format!("{}/blazeface_epoch_{}.axonml", CHECKPOINT_DIR, epoch);
    match save_model(model, &path) {
        Ok(()) => println!("  Checkpoint saved: {path} (loss={loss:.4})"),
        Err(e) => eprintln!("  Failed to save checkpoint: {e}"),
    }

    // Also save as "latest" for easy resume
    let latest = format!("{}/blazeface_latest.axonml", CHECKPOINT_DIR);
    std::fs::copy(&path, &latest).ok();
}

fn save_best(model: &BlazeFace, loss: f32) {
    std::fs::create_dir_all(CHECKPOINT_DIR).ok();
    let path = format!("{}/blazeface_best.axonml", CHECKPOINT_DIR);
    match save_model(model, &path) {
        Ok(()) => println!("  Best model saved: {path} (loss={loss:.4})"),
        Err(e) => eprintln!("  Failed to save best model: {e}"),
    }
}

fn load_weights(model: &mut BlazeFace) -> Option<usize> {
    let latest = format!("{}/blazeface_latest.axonml", CHECKPOINT_DIR);
    if !std::path::Path::new(&latest).exists() {
        println!("No checkpoint found at {latest}");
        return None;
    }

    let state_dict = match load_state_dict(&latest) {
        Ok(sd) => sd,
        Err(e) => {
            eprintln!("Failed to load checkpoint: {e}");
            return None;
        }
    };

    // Load weights into model parameters using indexed keys (param_0, param_1, ...)
    let params = model.parameters();

    let mut loaded = 0;
    for (i, param) in params.iter().enumerate() {
        let key = format!("param_{i}");
        if let Some(entry) = state_dict.get(&key) {
            let tensor = match entry.data.to_tensor() {
                Ok(t) => t,
                Err(_) => continue,
            };
            param.update_data(tensor);
            loaded += 1;
        }
    }

    println!("Loaded {loaded}/{} parameters from {latest}", params.len());

    // Try to detect epoch from latest checkpoint by scanning directory
    let mut max_epoch = 0usize;
    if let Ok(entries) = std::fs::read_dir(CHECKPOINT_DIR) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(rest) = name.strip_prefix("blazeface_epoch_") {
                if let Some(num_str) = rest.strip_suffix(".axonml") {
                    if let Ok(n) = num_str.parse::<usize>() {
                        max_epoch = max_epoch.max(n);
                    }
                }
            }
        }
    }

    Some(max_epoch)
}

// =============================================================================
// Training
// =============================================================================

// =============================================================================
// (monitor is now provided by axonml::monitor::TrainingMonitor)
// =============================================================================

fn main() {
    let resume = std::env::args().any(|a| a == "--resume");
    let use_monitor = std::env::args().any(|a| a == "--monitor");

    println!("=== BlazeFace Training on WIDER FACE ===");
    println!("Input size: {INPUT_SIZE}x{INPUT_SIZE}");
    println!("Samples:    {NUM_SAMPLES}");
    println!("Batch size: {BATCH_SIZE}");
    println!("Epochs:     {NUM_EPOCHS}");
    println!("LR:         {LR}");
    println!("Resume:     {resume}");
    println!();

    // Load dataset
    let t0 = Instant::now();
    let dataset = WiderFaceDataset::new(DATASET_ROOT, "train", (INPUT_SIZE, INPUT_SIZE))
        .expect("Failed to load WIDER FACE dataset");
    println!(
        "Dataset loaded: {} entries ({:.1}s)",
        dataset.len(),
        t0.elapsed().as_secs_f32()
    );

    let num_samples = NUM_SAMPLES.min(dataset.len());

    // Detect GPU
    let device = detect_device();
    println!("Device:     {:?}", device);

    // Create model
    let mut model = BlazeFace::new();
    let param_count: usize = model
        .parameters()
        .iter()
        .map(|p| p.variable().data().to_vec().len())
        .sum();
    println!("Model params: {param_count}");

    // Move model to GPU if available
    if device.is_gpu() {
        model.to_device(device);
        println!("Model moved to GPU");
    }

    // Resume from checkpoint if requested
    let start_epoch = if resume {
        match load_weights(&mut model) {
            Some(e) => {
                println!("Resuming from epoch {e}");
                if device.is_gpu() {
                    model.to_device(device);
                }
                e
            }
            None => 0,
        }
    } else {
        0
    };

    model.train();

    // Generate anchors
    let anchors = BlazeFace::generate_anchors(INPUT_SIZE);
    println!("Anchors: {}", anchors.len());
    println!();

    // Launch browser monitor if requested
    let monitor = if use_monitor {
        Some(
            TrainingMonitor::new("BlazeFace", param_count)
                .total_epochs(NUM_EPOCHS)
                .batch_size(BATCH_SIZE)
                .launch(),
        )
    } else {
        None
    };

    // Optimizer (fresh — Adam state not saved, but weights carry learned representations)
    let params = model.parameters();
    let mut optimizer = Adam::new(params, LR);

    let focal_loss = FocalLoss::with_params(0.75, 2.0); // alpha=0.75 upweights positives (faces)
    let smooth_l1 = SmoothL1Loss::new();

    let mut best_loss = f32::MAX;

    // Training loop
    for epoch in start_epoch..NUM_EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_cls_loss = 0.0f32;
        let mut epoch_bbox_loss = 0.0f32;
        let mut epoch_total_loss = 0.0f32;
        let mut num_pos_total = 0usize;
        let mut samples_ok = 0usize;

        let mut i = 0;
        while i < num_samples {
            // Collect a batch of samples — parallel data loading + preprocessing
            let batch_end = (i + BATCH_SIZE).min(num_samples);
            let batch_items: Vec<_> = (i..batch_end)
                .into_par_iter()
                .filter_map(|j| {
                    let (image, gt_boxes) = dataset.get(j)?;
                    let (cls_target, bbox_target, positive) =
                        assign_anchors(&gt_boxes, &anchors, INPUT_SIZE as f32);
                    Some((image, cls_target, bbox_target, positive))
                })
                .collect();
            i = batch_end;

            let mut batch_images = Vec::with_capacity(batch_items.len());
            let mut batch_cls_targets = Vec::with_capacity(batch_items.len());
            let mut batch_bbox_targets = Vec::with_capacity(batch_items.len());
            let mut batch_positives = Vec::with_capacity(batch_items.len());
            for (image, cls_target, bbox_target, positive) in batch_items {
                batch_images.push(image);
                batch_cls_targets.push(cls_target);
                batch_bbox_targets.push(bbox_target);
                batch_positives.push(positive);
            }

            let actual_batch = batch_images.len();
            if actual_batch == 0 {
                continue;
            }

            // Stack images into [B, 3, H, W] batch
            let img_numel = 3 * INPUT_SIZE * INPUT_SIZE;
            let mut batch_data = Vec::with_capacity(actual_batch * img_numel);
            for img in &batch_images {
                batch_data.extend_from_slice(&img.to_vec());
            }
            let batch_tensor =
                Tensor::from_vec(batch_data, &[actual_batch, 3, INPUT_SIZE, INPUT_SIZE]).unwrap();
            let batch_input = if device.is_gpu() {
                Variable::new(batch_tensor.to_device(device).unwrap(), true)
            } else {
                Variable::new(batch_tensor, true)
            };

            // Forward: [B, num_anchors, ...] outputs
            let (cls_logits, bbox_preds) = model.forward_train(&batch_input);

            // Compute per-sample losses and accumulate
            let mut batch_loss = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);
            let mut batch_cls_val = 0.0f32;
            let mut batch_bbox_val = 0.0f32;

            for s in 0..actual_batch {
                let num_pos = batch_positives[s].iter().filter(|&&v| v).count();
                num_pos_total += num_pos;

                // Extract this sample's predictions from batch
                let cls_s = cls_logits.narrow(0, s, 1).reshape(&[anchors.len()]);
                let bbox_s = bbox_preds.narrow(0, s, 1).reshape(&[anchors.len(), 4]);

                // Cls loss
                let cls_tgt = Variable::new(
                    Tensor::from_vec(batch_cls_targets[s].clone(), &[anchors.len()]).unwrap(),
                    false,
                );
                let cls_loss = focal_loss.compute(&cls_s, &cls_tgt);

                // Bbox loss (masked, graph-connected)
                let bbox_loss = if num_pos > 0 {
                    let bbox_tgt_var = Variable::new(
                        Tensor::from_vec(batch_bbox_targets[s].clone(), &[anchors.len(), 4])
                            .unwrap(),
                        false,
                    );
                    let mask_data: Vec<f32> = batch_positives[s]
                        .iter()
                        .map(|&p| if p { 1.0 } else { 0.0 })
                        .collect();
                    let mask = Variable::new(
                        Tensor::from_vec(mask_data, &[anchors.len(), 1]).unwrap(),
                        false,
                    );
                    let masked_pred = bbox_s.mul_var(&mask);
                    let masked_tgt = bbox_tgt_var.mul_var(&mask);
                    let raw_loss = smooth_l1.compute(&masked_pred, &masked_tgt);
                    raw_loss.mul_scalar(anchors.len() as f32 / num_pos as f32)
                } else {
                    Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false)
                };

                // Upweight cls loss 4x to balance with bbox (bbox gradient is ~4x stronger)
                let cls_loss_weighted = cls_loss.mul_scalar(4.0);
                batch_cls_val += cls_loss_weighted.data().to_vec()[0];
                batch_bbox_val += bbox_loss.data().to_vec()[0];
                batch_loss = batch_loss.add_var(&cls_loss_weighted).add_var(&bbox_loss);
            }

            // Average over batch
            let avg_factor = 1.0 / actual_batch as f32;
            let total_loss = batch_loss.mul_scalar(avg_factor);
            let total_val = total_loss.data().to_vec()[0];

            if total_loss.requires_grad() {
                optimizer.zero_grad();
                total_loss.backward();
                optimizer.step();
            }

            epoch_cls_loss += batch_cls_val * avg_factor;
            epoch_bbox_loss += batch_bbox_val * avg_factor;
            epoch_total_loss += total_val;
            samples_ok += 1; // count batch steps

            if i % (LOG_INTERVAL * BATCH_SIZE) < BATCH_SIZE {
                let steps = samples_ok.max(1) as f32;
                println!(
                    "  [{}/{}] loss: {:.4} (cls: {:.4}, bbox: {:.4}), pos: {}",
                    i,
                    num_samples,
                    epoch_total_loss / steps,
                    epoch_cls_loss / steps,
                    epoch_bbox_loss / steps,
                    num_pos_total,
                );
            }
        }

        if samples_ok == 0 {
            continue;
        }

        let steps = samples_ok.max(1) as f32;
        let avg_loss = epoch_total_loss / steps;
        let elapsed = epoch_start.elapsed().as_secs_f32();
        let images_processed = (samples_ok * BATCH_SIZE).min(num_samples);
        println!(
            "Epoch {}/{}: loss={:.4} (cls={:.4} bbox={:.4}) | pos={} | {:.1}s ({:.1} img/s, batch={})",
            epoch + 1,
            NUM_EPOCHS,
            avg_loss,
            epoch_cls_loss / steps,
            epoch_bbox_loss / steps,
            num_pos_total,
            elapsed,
            images_processed as f32 / elapsed,
            BATCH_SIZE,
        );

        // Save checkpoint every epoch
        save_checkpoint(&model, epoch + 1, avg_loss);

        // Track best
        if avg_loss < best_loss {
            best_loss = avg_loss;
            save_best(&model, avg_loss);
        }

        // Update monitor
        if let Some(ref mon) = monitor {
            let img_s = images_processed as f32 / elapsed;
            mon.log_epoch(
                epoch + 1,
                avg_loss,
                None,
                vec![
                    ("cls_loss", epoch_cls_loss / steps),
                    ("bbox_loss", epoch_bbox_loss / steps),
                    ("img_s", img_s),
                ],
            );
        }
    }

    if let Some(ref mon) = monitor {
        mon.set_status("complete");
    }

    // Inference test
    println!("\n=== Inference Test (conf={CONF_THRESHOLD}, nms={NMS_THRESHOLD}) ===");
    model.eval();
    let mut total_gt = 0usize;
    let mut total_det = 0usize;
    let mut total_tp = 0usize;
    let test_indices = [0, 5, 10, 20, 50, 100, 200, 500];
    for sample_idx in test_indices {
        if sample_idx >= dataset.len() {
            continue;
        }
        if let Some((test_img, gt_boxes)) = dataset.get(sample_idx) {
            let img_t = test_img.unsqueeze(0).unwrap();
            let img_t = if device.is_gpu() {
                img_t.to_device(device).unwrap()
            } else {
                img_t
            };
            let input = Variable::new(img_t, false);
            let dets = model.detect(&input, CONF_THRESHOLD, NMS_THRESHOLD);

            // Count true positives (IoU > 0.3 match)
            let mut tp = 0usize;
            let mut matched = vec![false; gt_boxes.len()];
            for det in &dets {
                let db = det.bbox; // [x1, y1, x2, y2] in pixels
                for (gi, gt) in gt_boxes.iter().enumerate() {
                    if matched[gi] {
                        continue;
                    }
                    // GT is normalized [0,1] — convert to pixels
                    let gx1 = gt[0] * INPUT_SIZE as f32;
                    let gy1 = gt[1] * INPUT_SIZE as f32;
                    let gx2 = gt[2] * INPUT_SIZE as f32;
                    let gy2 = gt[3] * INPUT_SIZE as f32;
                    // Direct box IoU (both in x1,y1,x2,y2 pixel format)
                    let ix1 = db[0].max(gx1);
                    let iy1 = db[1].max(gy1);
                    let ix2 = db[2].min(gx2);
                    let iy2 = db[3].min(gy2);
                    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
                    let area_d = (db[2] - db[0]).max(0.0) * (db[3] - db[1]).max(0.0);
                    let area_g = (gx2 - gx1).max(0.0) * (gy2 - gy1).max(0.0);
                    let union = area_d + area_g - inter;
                    let iou = if union > 0.0 { inter / union } else { 0.0 };
                    if iou > 0.3 {
                        tp += 1;
                        matched[gi] = true;
                        break;
                    }
                }
            }

            total_gt += gt_boxes.len();
            total_det += dets.len();
            total_tp += tp;
            println!(
                "Sample {}: GT={} faces, Detected={}, TP={} | conf>={CONF_THRESHOLD}",
                sample_idx,
                gt_boxes.len(),
                dets.len(),
                tp,
            );
            for (i, det) in dets.iter().take(5).enumerate() {
                println!(
                    "  det[{}]: [{:.1},{:.1},{:.1},{:.1}] conf={:.3}",
                    i, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.confidence,
                );
            }
        }
    }

    let precision = if total_det > 0 {
        total_tp as f32 / total_det as f32
    } else {
        0.0
    };
    let recall = if total_gt > 0 {
        total_tp as f32 / total_gt as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    println!("\nMetrics over {} test samples:", test_indices.len());
    println!(
        "  Precision: {:.1}% ({}/{})",
        precision * 100.0,
        total_tp,
        total_det
    );
    println!(
        "  Recall:    {:.1}% ({}/{})",
        recall * 100.0,
        total_tp,
        total_gt
    );
    println!("  F1:        {:.1}%", f1 * 100.0);

    println!("\nTraining complete. Best loss: {best_loss:.4}");
    println!("Checkpoints: {CHECKPOINT_DIR}/");
    println!("  blazeface_best.axonml   — best model weights");
    println!("  blazeface_latest.axonml — latest for --resume");
}
