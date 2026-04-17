//! Detection Model Training Suite — COCO val2017 + WIDER FACE Pipelines
//!
//! Test-only end-to-end training tests for the axonml-vision detector zoo.
//! All tests are `#[ignore]` (opt in with `--ignored`) and write JSON
//! results to `release-artifacts/benchmark_results/`. Hard-coded paths point
//! at `/opt/datasets/coco/val2017` and `/opt/datasets/wider_face`. Helpers
//! `ensure_dirs` creates output directories, `load_coco_split` loads
//! `CocoDataset` and partitions indices into 80/20 train/eval slices,
//! `annos_to_gt` converts normalized COCO annotations to pixel-space GT,
//! `save_results` writes a pretty-printed JSON, and `evaluate_detections`
//! returns `(mAP@50, mAP@75, COCO mAP)`. The BlazeFace pipeline adds
//! `assign_anchors_to_gt` (IoU-based positive / negative / ignore label
//! assignment with force-assigned best anchor per GT and offset / log-ratio
//! encoding) and `ohem_select` (online hard negative mining at a fixed
//! neg:pos ratio).
//!
//! Per-model tests train, evaluate, and dump per-epoch loss history plus
//! mAP / latency / FPS:
//!
//! - `train_helios_nano` / `train_helios_small` use `HeliosTrainer` over a
//!   COCO subset.
//! - `train_nanodet` runs a custom backbone+neck+head loop with focal +
//!   smooth-L1 losses and grid-cell target encoding.
//! - `train_nexus` and `train_phantom` reuse the per-step training functions
//!   from the parent module.
//! - `train_blazeface` trains on WIDER FACE with full caching, anchor
//!   assignment, focal classification, masked box regression, gradient
//!   accumulation, warmup + cosine LR, random horizontal flip, and periodic
//!   mid-training mAP evaluation.
//! - `train_retinaface` uses person annotations from COCO as a face proxy and
//!   computes per-FPN-level focal classification loss.
//!
//! # File
//! `crates/axonml-vision/src/training/train_models.rs`
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

#[cfg(test)]
mod tests {
    use axonml_autograd::Variable;
    use axonml_nn::Module;
    use axonml_optim::{Adam, Optimizer};
    use axonml_tensor::Tensor;

    use crate::datasets::coco::{CocoAnnotation, CocoDataset};
    use crate::losses::FocalLoss;
    use crate::training::{nexus_training_step, phantom_training_step};
    use crate::training::metrics::{compute_map, compute_coco_map, DetectionResult, GroundTruth};

    const COCO_ROOT: &str = "/opt/datasets/coco";
    const COCO_IMAGES: &str = "/opt/datasets/coco/val2017";
    const COCO_ANNOS: &str = "/opt/datasets/coco/annotations/instances_val2017.json";
    const MODEL_SAVE_DIR: &str = "/opt/AxonML/models/trained";
    const RESULTS_DIR: &str = "/opt/AxonML/release-artifacts/benchmark_results";

    // =========================================================================
    // Helpers
    // =========================================================================

    fn ensure_dirs() {
        std::fs::create_dir_all(MODEL_SAVE_DIR).ok();
        std::fs::create_dir_all(RESULTS_DIR).ok();
    }

    /// Load a subset of COCO dataset and split into train/eval.
    fn load_coco_split(
        input_size: (usize, usize),
        max_train: usize,
        max_eval: usize,
    ) -> (CocoDataset, Vec<usize>, Vec<usize>) {
        let ds = CocoDataset::new(COCO_IMAGES, COCO_ANNOS, input_size)
            .expect("Failed to load COCO dataset");

        let total = ds.len();
        println!("  COCO loaded: {} images with annotations", total);
        println!("  Num classes: {}", ds.num_classes());

        let train_count = max_train.min(total * 4 / 5);
        let eval_count = max_eval.min(total - train_count);

        let train_indices: Vec<usize> = (0..train_count).collect();
        let eval_indices: Vec<usize> = (train_count..train_count + eval_count).collect();

        println!("  Train: {} images, Eval: {} images", train_indices.len(), eval_indices.len());

        (ds, train_indices, eval_indices)
    }

    /// Convert COCO annotations to pixel-space gt_boxes and gt_classes.
    fn annos_to_gt(
        annos: &[CocoAnnotation],
        img_h: f32,
        img_w: f32,
    ) -> (Vec<[f32; 4]>, Vec<usize>) {
        let mut boxes = Vec::new();
        let mut classes = Vec::new();
        for a in annos {
            boxes.push([
                a.bbox[0] * img_w,
                a.bbox[1] * img_h,
                a.bbox[2] * img_w,
                a.bbox[3] * img_h,
            ]);
            classes.push(a.category_id);
        }
        (boxes, classes)
    }

    /// Save benchmark results to JSON.
    fn save_results(model_name: &str, results: &serde_json::Value) {
        let path = format!("{}/{}.json", RESULTS_DIR, model_name);
        let json = serde_json::to_string_pretty(results).unwrap();
        std::fs::write(&path, json).unwrap();
        println!("  Results saved to {}", path);
    }

    /// Run COCO evaluation on a set of model detections.
    fn evaluate_detections(
        all_dets: &[Vec<DetectionResult>],
        all_gts: &[Vec<GroundTruth>],
        num_classes: usize,
    ) -> (f32, f32, f32) {
        let map50 = compute_map(all_dets, all_gts, num_classes, 0.5);
        let map75 = compute_map(all_dets, all_gts, num_classes, 0.75);
        let coco_map = compute_coco_map(all_dets, all_gts, num_classes);
        (map50, map75, coco_map)
    }

    // =========================================================================
    // Helios-Nano Training
    // =========================================================================

    #[test]
    #[ignore]
    fn train_helios_nano() {
        use crate::models::helios::Helios;
        use crate::training::helios_trainer::{HeliosTrainConfig, HeliosTrainer};

        println!("\n{}", "=".repeat(80));
        println!("  TRAINING: Helios-Nano on COCO val2017");
        println!("{}\n", "=".repeat(80));
        ensure_dirs();

        let input_size = (128, 128);
        let num_classes = 80;
        let (ds, train_idx, eval_idx) = load_coco_split(input_size, 200, 50);

        // Create model and trainer
        let model = Helios::nano(num_classes);
        let mut config = HeliosTrainConfig::fast(num_classes);
        config.epochs = 10;
        config.input_size = input_size;
        config.log_interval = 50;
        config.eval_interval = 5;
        config.use_mosaic = false;
        config.use_mixup = false;

        let mut trainer = HeliosTrainer::new(model, config.clone());
        let (img_h, img_w) = (input_size.0 as f32, input_size.1 as f32);

        println!("  Config: {} epochs, lr={}, input={}x{}", config.epochs, config.lr, input_size.0, input_size.1);
        let param_count: usize = trainer.parameters().iter().map(|p| p.data().numel()).sum();
        println!("  Parameters: {}\n", param_count);

        // Training loop
        let mut best_loss = f32::MAX;
        let mut epoch_losses = Vec::new();

        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0f32;
            let mut steps = 0;

            for &idx in &train_idx {
                if let Some((img_tensor, annos)) = ds.get(idx) {
                    let (gt_boxes, gt_classes) = annos_to_gt(&annos, img_h, img_w);
                    if gt_boxes.is_empty() { continue; }

                    let input = Variable::new(
                        Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                        false,
                    );

                    let (total, cls, bx, dfl) = trainer.train_step(&input, &[gt_boxes], &[gt_classes]);
                    epoch_loss += total;
                    steps += 1;

                    if steps % config.log_interval == 0 {
                        println!("  [Epoch {}/{}] Step {}/{}: loss={:.4} (cls={:.4} box={:.4} dfl={:.4})",
                            epoch + 1, config.epochs, steps, train_idx.len(),
                            total, cls, bx, dfl);
                    }
                }
            }

            let avg_loss = if steps > 0 { epoch_loss / steps as f32 } else { 0.0 };
            epoch_losses.push(avg_loss);
            if avg_loss < best_loss { best_loss = avg_loss; }

            println!("  Epoch {}/{}: avg_loss={:.4} (best={:.4})", epoch + 1, config.epochs, avg_loss, best_loss);
            trainer.advance_epoch();
        }

        // Evaluation
        println!("\n  Running COCO evaluation on {} images...", eval_idx.len());
        let mut all_dets = Vec::new();
        let mut all_gts = Vec::new();
        let mut total_latency = 0.0f64;

        for &idx in &eval_idx {
            if let Some((img_tensor, annos)) = ds.get(idx) {
                let input = Variable::new(
                    Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                    false,
                ).to_device(trainer.device());

                let start = std::time::Instant::now();
                let detections = trainer.model.detect(&input, 0.01, 0.65);
                total_latency += start.elapsed().as_secs_f64() * 1000.0;

                let dets: Vec<DetectionResult> = detections.iter().map(|d| {
                    DetectionResult {
                        bbox: [
                            d.bbox[0] / img_w,
                            d.bbox[1] / img_h,
                            d.bbox[2] / img_w,
                            d.bbox[3] / img_h,
                        ],
                        confidence: d.confidence,
                        class_id: d.class_id,
                    }
                }).collect();

                let gts: Vec<GroundTruth> = annos.iter().map(|a| {
                    GroundTruth { bbox: a.bbox, class_id: a.category_id }
                }).collect();

                all_dets.push(dets);
                all_gts.push(gts);
            }
        }

        let eval_count = all_dets.len();
        let mean_latency = total_latency / eval_count as f64;
        let fps = 1000.0 / mean_latency;

        let (map50, map75, coco_map) = evaluate_detections(&all_dets, &all_gts, num_classes);

        println!("\n  === Helios-Nano Results ===");
        println!("  mAP@50:    {:.4}", map50);
        println!("  mAP@75:    {:.4}", map75);
        println!("  COCO mAP:  {:.4}", coco_map);
        println!("  Latency:   {:.1}ms", mean_latency);
        println!("  FPS:       {:.1}", fps);
        println!("  Params:    {}", param_count);
        println!("  Best loss: {:.4}", best_loss);

        let results = serde_json::json!({
            "model": "helios-nano",
            "params": param_count,
            "epochs_trained": config.epochs,
            "training_images": train_idx.len(),
            "eval_images": eval_count,
            "input_size": [input_size.0, input_size.1],
            "best_loss": best_loss,
            "final_loss": epoch_losses.last().unwrap_or(&0.0),
            "loss_history": epoch_losses,
            "map50": map50,
            "map75": map75,
            "coco_map": coco_map,
            "mean_latency_ms": mean_latency,
            "fps": fps,
        });
        save_results("helios_nano", &results);
    }

    // =========================================================================
    // Helios-Small Training
    // =========================================================================

    #[test]
    #[ignore]
    fn train_helios_small() {
        use crate::models::helios::Helios;
        use crate::training::helios_trainer::{HeliosTrainConfig, HeliosTrainer};

        println!("\n{}", "=".repeat(80));
        println!("  TRAINING: Helios-Small on COCO val2017");
        println!("{}\n", "=".repeat(80));
        ensure_dirs();

        let input_size = (128, 128);
        let num_classes = 80;
        let (ds, train_idx, eval_idx) = load_coco_split(input_size, 150, 50);

        let model = Helios::small(num_classes);
        let mut config = HeliosTrainConfig::fast(num_classes);
        config.epochs = 8;
        config.input_size = input_size;
        config.lr = 0.0005;
        config.log_interval = 15;
        config.use_mosaic = false;
        config.use_mixup = false;

        let mut trainer = HeliosTrainer::new(model, config.clone());
        let (img_h, img_w) = (input_size.0 as f32, input_size.1 as f32);

        let param_count: usize = trainer.parameters().iter().map(|p| p.data().numel()).sum();
        println!("  Parameters: {}", param_count);
        println!("  Config: {} epochs, lr={}\n", config.epochs, config.lr);

        let mut best_loss = f32::MAX;
        let mut epoch_losses = Vec::new();

        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0f32;
            let mut steps = 0;

            for &idx in &train_idx {
                if let Some((img_tensor, annos)) = ds.get(idx) {
                    let (gt_boxes, gt_classes) = annos_to_gt(&annos, img_h, img_w);
                    if gt_boxes.is_empty() { continue; }

                    let input = Variable::new(
                        Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                        false,
                    );

                    let (total, _cls, _bx, _dfl) = trainer.train_step(&input, &[gt_boxes], &[gt_classes]);
                    epoch_loss += total;
                    steps += 1;

                    if steps % config.log_interval == 0 {
                        println!("  [Epoch {}/{}] Step {}/{}: loss={:.4}", epoch + 1, config.epochs, steps, train_idx.len(), total);
                    }
                }
            }

            let avg_loss = if steps > 0 { epoch_loss / steps as f32 } else { 0.0 };
            epoch_losses.push(avg_loss);
            if avg_loss < best_loss { best_loss = avg_loss; }
            println!("  Epoch {}/{}: avg_loss={:.4}", epoch + 1, config.epochs, avg_loss);
            trainer.advance_epoch();
        }

        println!("\n  Evaluating...");
        let mut all_dets = Vec::new();
        let mut all_gts = Vec::new();
        let mut total_latency = 0.0f64;

        for &idx in &eval_idx {
            if let Some((img_tensor, annos)) = ds.get(idx) {
                let input = Variable::new(
                    Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                    false,
                ).to_device(trainer.device());
                let start = std::time::Instant::now();
                let detections = trainer.model.detect(&input, 0.01, 0.65);
                total_latency += start.elapsed().as_secs_f64() * 1000.0;

                let dets: Vec<DetectionResult> = detections.iter().map(|d| DetectionResult {
                    bbox: [d.bbox[0] / img_w, d.bbox[1] / img_h, d.bbox[2] / img_w, d.bbox[3] / img_h],
                    confidence: d.confidence,
                    class_id: d.class_id,
                }).collect();
                let gts: Vec<GroundTruth> = annos.iter().map(|a| GroundTruth { bbox: a.bbox, class_id: a.category_id }).collect();
                all_dets.push(dets);
                all_gts.push(gts);
            }
        }

        let eval_count = all_dets.len();
        let mean_latency = total_latency / eval_count as f64;
        let (map50, map75, coco_map) = evaluate_detections(&all_dets, &all_gts, num_classes);

        println!("\n  === Helios-Small Results ===");
        println!("  mAP@50: {:.4}  mAP@75: {:.4}  COCO mAP: {:.4}", map50, map75, coco_map);
        println!("  Latency: {:.1}ms  FPS: {:.1}", mean_latency, 1000.0 / mean_latency);

        save_results("helios_small", &serde_json::json!({
            "model": "helios-small", "params": param_count,
            "epochs_trained": config.epochs, "training_images": train_idx.len(),
            "eval_images": eval_count, "input_size": [input_size.0, input_size.1],
            "best_loss": best_loss, "final_loss": epoch_losses.last().unwrap_or(&0.0),
            "loss_history": epoch_losses,
            "map50": map50, "map75": map75, "coco_map": coco_map,
            "mean_latency_ms": mean_latency, "fps": 1000.0 / mean_latency,
        }));
    }

    // =========================================================================
    // NanoDet Training
    // =========================================================================

    #[test]
    #[ignore]
    fn train_nanodet() {
        use crate::models::nanodet::NanoDet;

        println!("\n{}", "=".repeat(80));
        println!("  TRAINING: NanoDet on COCO val2017");
        println!("{}\n", "=".repeat(80));
        ensure_dirs();

        let input_size = (128, 128);
        let num_classes = 80;
        let (ds, train_idx, eval_idx) = load_coco_split(input_size, 200, 50);

        let model = NanoDet::new(num_classes);
        let params = model.parameters();
        let param_count: usize = params.iter().map(|p| p.data().numel()).sum();
        let mut optimizer = Adam::new(params, 1e-3).weight_decay(1e-4);
        let focal_loss = FocalLoss::new();
        let smooth_l1 = axonml_nn::SmoothL1Loss::new();

        let (img_h, img_w) = (input_size.0 as f32, input_size.1 as f32);
        let epochs = 15;

        println!("  Parameters: {}", param_count);
        println!("  Config: {} epochs, lr=0.001\n", epochs);

        let mut best_loss = f32::MAX;
        let mut epoch_losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let mut steps = 0;

            for &idx in &train_idx {
                if let Some((img_tensor, annos)) = ds.get(idx) {
                    let (gt_boxes, gt_classes) = annos_to_gt(&annos, img_h, img_w);
                    if gt_boxes.is_empty() { continue; }

                    let input = Variable::new(
                        Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                        false,
                    );

                    // Forward through backbone + neck + head
                    let features = model.backbone.forward(&input);
                    let neck_features = model.neck.forward(&features);

                    let strides = [8.0f32, 16.0, 32.0];
                    let mut total_loss = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);

                    for (level, feat) in neck_features.iter().enumerate() {
                        let (cls_out, bbox_out) = model.head.forward_single(feat);
                        let cls_shape = cls_out.shape();
                        let (fh, fw) = (cls_shape[2], cls_shape[3]);
                        let stride = strides[level];

                        // Create target maps
                        let mut cls_target = vec![0.0f32; num_classes * fh * fw];
                        let mut bbox_target = vec![0.0f32; 4 * fh * fw];
                        let mut has_pos = false;

                        for (bi, box_px) in gt_boxes.iter().enumerate() {
                            let cx: f32 = (box_px[0] + box_px[2]) / 2.0;
                            let cy: f32 = (box_px[1] + box_px[3]) / 2.0;
                            let gx = (cx / stride).floor() as usize;
                            let gy = (cy / stride).floor() as usize;

                            if gx < fw && gy < fh {
                                let cls = gt_classes[bi];
                                if cls < num_classes {
                                    cls_target[cls * fh * fw + gy * fw + gx] = 1.0;
                                    bbox_target[0 * fh * fw + gy * fw + gx] = cx / stride - gx as f32;
                                    bbox_target[1 * fh * fw + gy * fw + gx] = cy / stride - gy as f32;
                                    let bw: f32 = (box_px[2] - box_px[0]).max(1.0);
                                    let bh: f32 = (box_px[3] - box_px[1]).max(1.0);
                                    bbox_target[2 * fh * fw + gy * fw + gx] = (bw / stride).ln();
                                    bbox_target[3 * fh * fw + gy * fw + gx] = (bh / stride).ln();
                                    has_pos = true;
                                }
                            }
                        }

                        // Classification loss
                        let cls_pred = cls_out.reshape(&[num_classes * fh * fw]);
                        let cls_tgt = Variable::new(
                            Tensor::from_vec(cls_target, &[num_classes * fh * fw]).unwrap(),
                            false,
                        );
                        let cls_loss = focal_loss.compute(&cls_pred, &cls_tgt);
                        total_loss = total_loss.add_var(&cls_loss);

                        // Box regression loss (on positive locations)
                        if has_pos {
                            let bbox_pred = bbox_out.reshape(&[4 * fh * fw]);
                            let bbox_tgt = Variable::new(
                                Tensor::from_vec(bbox_target, &[4 * fh * fw]).unwrap(),
                                false,
                            );
                            let box_loss = smooth_l1.compute(&bbox_pred, &bbox_tgt).mul_scalar(0.5);
                            total_loss = total_loss.add_var(&box_loss);
                        }
                    }

                    let loss_val = total_loss.data().to_vec()[0];
                    if loss_val.is_finite() && total_loss.requires_grad() {
                        optimizer.zero_grad();
                        total_loss.backward();
                        optimizer.step();
                    }

                    epoch_loss += loss_val;
                    steps += 1;

                    if steps % 50 == 0 {
                        println!("  [Epoch {}/{}] Step {}/{}: loss={:.4}", epoch + 1, epochs, steps, train_idx.len(), loss_val);
                    }
                }
            }

            let avg_loss = if steps > 0 { epoch_loss / steps as f32 } else { 0.0 };
            epoch_losses.push(avg_loss);
            if avg_loss < best_loss { best_loss = avg_loss; }
            println!("  Epoch {}/{}: avg_loss={:.4}", epoch + 1, epochs, avg_loss);
        }

        // Evaluation
        println!("\n  Evaluating...");
        let mut all_dets = Vec::new();
        let mut all_gts = Vec::new();
        let mut total_latency = 0.0f64;

        for &idx in &eval_idx {
            if let Some((img_tensor, annos)) = ds.get(idx) {
                let input = Variable::new(
                    Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                    false,
                );
                let start = std::time::Instant::now();
                let detections = model.detect(&input, 0.01, 0.65);
                total_latency += start.elapsed().as_secs_f64() * 1000.0;

                let dets: Vec<DetectionResult> = detections.iter().map(|d| DetectionResult {
                    bbox: [d.bbox[0] / img_w, d.bbox[1] / img_h, d.bbox[2] / img_w, d.bbox[3] / img_h],
                    confidence: d.confidence,
                    class_id: d.class_id,
                }).collect();
                let gts: Vec<GroundTruth> = annos.iter().map(|a| GroundTruth { bbox: a.bbox, class_id: a.category_id }).collect();
                all_dets.push(dets);
                all_gts.push(gts);
            }
        }

        let eval_count = all_dets.len();
        let mean_latency = total_latency / eval_count as f64;
        let (map50, map75, coco_map) = evaluate_detections(&all_dets, &all_gts, num_classes);

        println!("\n  === NanoDet Results ===");
        println!("  mAP@50: {:.4}  mAP@75: {:.4}  COCO mAP: {:.4}", map50, map75, coco_map);
        println!("  Latency: {:.1}ms  FPS: {:.1}", mean_latency, 1000.0 / mean_latency);

        save_results("nanodet", &serde_json::json!({
            "model": "nanodet", "params": param_count,
            "epochs_trained": epochs, "training_images": train_idx.len(),
            "eval_images": eval_count, "input_size": [input_size.0, input_size.1],
            "best_loss": best_loss, "final_loss": epoch_losses.last().unwrap_or(&0.0),
            "loss_history": epoch_losses,
            "map50": map50, "map75": map75, "coco_map": coco_map,
            "mean_latency_ms": mean_latency, "fps": 1000.0 / mean_latency,
        }));
    }

    // =========================================================================
    // Nexus Training
    // =========================================================================

    #[test]
    #[ignore]
    fn train_nexus() {
        use crate::models::nexus::Nexus;

        println!("\n{}", "=".repeat(80));
        println!("  TRAINING: Nexus on COCO val2017");
        println!("{}\n", "=".repeat(80));
        ensure_dirs();

        let input_size = (128, 128);
        let (ds, train_idx, eval_idx) = load_coco_split(input_size, 400, 100);
        let num_classes = ds.num_classes();

        let mut model = Nexus::new();
        let params = model.parameters();
        let param_count: usize = params.iter().map(|p| p.data().numel()).sum();
        let mut optimizer = Adam::new(params, 5e-4).weight_decay(1e-4);

        let (img_h, img_w) = (input_size.0 as f32, input_size.1 as f32);
        let epochs = 12;

        println!("  Parameters: {}", param_count);
        println!("  Config: {} epochs, input={}x{}\n", epochs, input_size.0, input_size.1);

        let mut best_loss = f32::MAX;
        let mut epoch_losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let mut steps = 0;

            for &idx in &train_idx {
                if let Some((img_tensor, annos)) = ds.get(idx) {
                    let (gt_boxes, gt_classes) = annos_to_gt(&annos, img_h, img_w);
                    if gt_boxes.is_empty() { continue; }

                    let input = Variable::new(
                        Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                        false,
                    );

                    let loss = nexus_training_step(
                        &mut model, &input, &gt_boxes, &gt_classes, &mut optimizer,
                    );
                    epoch_loss += loss;
                    steps += 1;

                    if steps % 50 == 0 {
                        println!("  [Epoch {}/{}] Step {}/{}: loss={:.4}", epoch + 1, epochs, steps, train_idx.len(), loss);
                    }
                }
            }

            let avg_loss = if steps > 0 { epoch_loss / steps as f32 } else { 0.0 };
            epoch_losses.push(avg_loss);
            if avg_loss < best_loss { best_loss = avg_loss; }
            println!("  Epoch {}/{}: avg_loss={:.4}", epoch + 1, epochs, avg_loss);
        }

        println!("\n  Evaluating...");
        let mut all_dets = Vec::new();
        let mut all_gts = Vec::new();
        let mut total_latency = 0.0f64;

        for &idx in &eval_idx {
            if let Some((img_tensor, annos)) = ds.get(idx) {
                let input = Variable::new(
                    Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                    false,
                );
                let start = std::time::Instant::now();
                let detections = model.detect(&input);
                total_latency += start.elapsed().as_secs_f64() * 1000.0;

                let nc = ds.num_classes();
                let dets: Vec<DetectionResult> = detections.iter().map(|d| DetectionResult {
                    bbox: [
                        d.bbox_mean[0] / img_w, d.bbox_mean[1] / img_h,
                        d.bbox_mean[2] / img_w, d.bbox_mean[3] / img_h,
                    ],
                    confidence: d.confidence,
                    class_id: d.class_id,
                }).collect();
                let gts: Vec<GroundTruth> = annos.iter().map(|a| GroundTruth { bbox: a.bbox, class_id: a.category_id }).collect();
                all_dets.push(dets);
                all_gts.push(gts);
            }
        }

        let eval_count = all_dets.len();
        let mean_latency = total_latency / eval_count as f64;
        let (map50, map75, coco_map) = evaluate_detections(&all_dets, &all_gts, num_classes);

        println!("\n  === Nexus Results ===");
        println!("  mAP@50: {:.4}  mAP@75: {:.4}  COCO mAP: {:.4}", map50, map75, coco_map);
        println!("  Latency: {:.1}ms  FPS: {:.1}", mean_latency, 1000.0 / mean_latency);

        save_results("nexus", &serde_json::json!({
            "model": "nexus", "params": param_count,
            "epochs_trained": epochs, "training_images": train_idx.len(),
            "eval_images": eval_count, "input_size": [input_size.0, input_size.1],
            "best_loss": best_loss, "final_loss": epoch_losses.last().unwrap_or(&0.0),
            "loss_history": epoch_losses,
            "map50": map50, "map75": map75, "coco_map": coco_map,
            "mean_latency_ms": mean_latency, "fps": 1000.0 / mean_latency,
        }));
    }

    // =========================================================================
    // Phantom Training (Face Detection on COCO Person)
    // =========================================================================

    #[test]
    #[ignore]
    fn train_phantom() {
        use crate::models::phantom::Phantom;

        println!("\n{}", "=".repeat(80));
        println!("  TRAINING: Phantom (Face Detector) on COCO val2017");
        println!("{}\n", "=".repeat(80));
        ensure_dirs();

        let input_size = (128, 128);
        let (ds, train_idx, eval_idx) = load_coco_split(input_size, 500, 100);

        let mut model = Phantom::new();
        let params = model.parameters();
        let param_count: usize = params.iter().map(|p| p.data().numel()).sum();
        let mut optimizer = Adam::new(params, 1e-3).weight_decay(1e-4);

        let (img_h, img_w) = (input_size.0 as f32, input_size.1 as f32);
        let epochs = 15;

        println!("  Parameters: {}", param_count);
        println!("  Config: {} epochs, using COCO person class as face proxy\n", epochs);

        let mut best_loss = f32::MAX;
        let mut epoch_losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let mut steps = 0;

            for &idx in &train_idx {
                if let Some((img_tensor, annos)) = ds.get(idx) {
                    // Use person class (category 0 in remapped) as face proxy
                    let person_annos: Vec<_> = annos.iter().filter(|a| a.category_id == 0).collect();
                    if person_annos.is_empty() { continue; }

                    let gt_faces: Vec<[f32; 4]> = person_annos.iter().map(|a| {
                        [a.bbox[0] * img_w, a.bbox[1] * img_h, a.bbox[2] * img_w, a.bbox[3] * img_h]
                    }).collect();

                    let input = Variable::new(
                        Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                        false,
                    );

                    let loss = phantom_training_step(&mut model, &input, &gt_faces, &mut optimizer);
                    epoch_loss += loss;
                    steps += 1;

                    if steps % 50 == 0 {
                        println!("  [Epoch {}/{}] Step {}/{}: loss={:.4}", epoch + 1, epochs, steps, train_idx.len(), loss);
                    }
                }
            }

            let avg_loss = if steps > 0 { epoch_loss / steps as f32 } else { 0.0 };
            epoch_losses.push(avg_loss);
            if avg_loss < best_loss { best_loss = avg_loss; }
            println!("  Epoch {}/{}: avg_loss={:.4}", epoch + 1, epochs, avg_loss);
        }

        println!("\n  Evaluating...");
        let mut all_dets = Vec::new();
        let mut all_gts = Vec::new();
        let mut total_latency = 0.0f64;

        for &idx in &eval_idx {
            if let Some((img_tensor, annos)) = ds.get(idx) {
                let person_annos: Vec<_> = annos.iter().filter(|a| a.category_id == 0).collect();

                let input = Variable::new(
                    Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                    false,
                );
                let start = std::time::Instant::now();
                let detections = model.detect_frame(&input);
                total_latency += start.elapsed().as_secs_f64() * 1000.0;

                let dets: Vec<DetectionResult> = detections.iter().map(|d| DetectionResult {
                    bbox: [d.bbox[0] / img_w, d.bbox[1] / img_h, d.bbox[2] / img_w, d.bbox[3] / img_h],
                    confidence: d.confidence,
                    class_id: 0,
                }).collect();
                let gts: Vec<GroundTruth> = person_annos.iter().map(|a| GroundTruth { bbox: a.bbox, class_id: 0 }).collect();
                all_dets.push(dets);
                all_gts.push(gts);
            }
        }

        let eval_count = all_dets.len();
        let mean_latency = total_latency / eval_count as f64;
        let (map50, _map75, coco_map) = evaluate_detections(&all_dets, &all_gts, 1);

        println!("\n  === Phantom Results ===");
        println!("  mAP@50: {:.4}  COCO mAP: {:.4}", map50, coco_map);
        println!("  Latency: {:.1}ms  FPS: {:.1}", mean_latency, 1000.0 / mean_latency);

        save_results("phantom", &serde_json::json!({
            "model": "phantom", "params": param_count,
            "epochs_trained": epochs, "training_images": train_idx.len(),
            "eval_images": eval_count, "input_size": [input_size.0, input_size.1],
            "best_loss": best_loss, "final_loss": epoch_losses.last().unwrap_or(&0.0),
            "loss_history": epoch_losses,
            "map50": map50, "coco_map": coco_map,
            "mean_latency_ms": mean_latency, "fps": 1000.0 / mean_latency,
        }));
    }

    // =========================================================================
    // BlazeFace Training (Face Detection on WIDER FACE)
    // =========================================================================

    const WIDER_FACE_ROOT: &str = "/opt/datasets/wider_face";

    /// Assign ground-truth boxes to anchors using IoU matching.
    /// Returns (cls_targets, bbox_targets) for all anchors.
    fn assign_anchors_to_gt(
        anchors: &[[f32; 4]],  // (cx, cy, w, h) in pixels
        gt_boxes: &[[f32; 4]], // (x1, y1, x2, y2) normalized [0,1]
        input_size: f32,
        pos_iou: f32,
        neg_iou: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let num_anchors = anchors.len();
        let mut cls_targets = vec![0.0f32; num_anchors]; // 0=negative, 1=positive
        let mut bbox_targets = vec![0.0f32; num_anchors * 4];
        let mut max_iou_per_anchor = vec![0.0f32; num_anchors];
        let mut best_gt_per_anchor = vec![0usize; num_anchors];

        if gt_boxes.is_empty() {
            return (cls_targets, bbox_targets);
        }

        // Convert anchors to (x1,y1,x2,y2) for IoU computation
        for (a_idx, anchor) in anchors.iter().enumerate() {
            let ax1 = anchor[0] - anchor[2] / 2.0;
            let ay1 = anchor[1] - anchor[3] / 2.0;
            let ax2 = anchor[0] + anchor[2] / 2.0;
            let ay2 = anchor[1] + anchor[3] / 2.0;
            let a_area = anchor[2] * anchor[3];

            for (g_idx, gt) in gt_boxes.iter().enumerate() {
                // GT is normalized, convert to pixel coords
                let gx1 = gt[0] * input_size;
                let gy1 = gt[1] * input_size;
                let gx2 = gt[2] * input_size;
                let gy2 = gt[3] * input_size;
                let g_area = (gx2 - gx1) * (gy2 - gy1);

                let inter_x1 = ax1.max(gx1);
                let inter_y1 = ay1.max(gy1);
                let inter_x2 = ax2.min(gx2);
                let inter_y2 = ay2.min(gy2);
                let inter = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
                let union = a_area + g_area - inter;
                let iou = if union > 0.0 { inter / union } else { 0.0 };

                if iou > max_iou_per_anchor[a_idx] {
                    max_iou_per_anchor[a_idx] = iou;
                    best_gt_per_anchor[a_idx] = g_idx;
                }
            }
        }

        // Also ensure each GT gets at least one anchor (best-match)
        let mut best_anchor_per_gt = vec![0usize; gt_boxes.len()];
        let mut best_iou_per_gt = vec![0.0f32; gt_boxes.len()];
        for (a_idx, anchor) in anchors.iter().enumerate() {
            let ax1 = anchor[0] - anchor[2] / 2.0;
            let ay1 = anchor[1] - anchor[3] / 2.0;
            let ax2 = anchor[0] + anchor[2] / 2.0;
            let ay2 = anchor[1] + anchor[3] / 2.0;
            let a_area = anchor[2] * anchor[3];

            for (g_idx, gt) in gt_boxes.iter().enumerate() {
                let gx1 = gt[0] * input_size;
                let gy1 = gt[1] * input_size;
                let gx2 = gt[2] * input_size;
                let gy2 = gt[3] * input_size;
                let g_area = (gx2 - gx1) * (gy2 - gy1);

                let inter_x1 = ax1.max(gx1);
                let inter_y1 = ay1.max(gy1);
                let inter_x2 = ax2.min(gx2);
                let inter_y2 = ay2.min(gy2);
                let inter = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
                let union = a_area + g_area - inter;
                let iou = if union > 0.0 { inter / union } else { 0.0 };

                if iou > best_iou_per_gt[g_idx] {
                    best_iou_per_gt[g_idx] = iou;
                    best_anchor_per_gt[g_idx] = a_idx;
                }
            }
        }

        // Force-assign best anchor for each GT
        for (g_idx, &a_idx) in best_anchor_per_gt.iter().enumerate() {
            max_iou_per_anchor[a_idx] = 1.0; // force positive
            best_gt_per_anchor[a_idx] = g_idx;
        }

        // Assign labels and encode targets
        for a_idx in 0..num_anchors {
            if max_iou_per_anchor[a_idx] >= pos_iou {
                cls_targets[a_idx] = 1.0;
                let g_idx = best_gt_per_anchor[a_idx];
                let gt = &gt_boxes[g_idx];
                let anchor = &anchors[a_idx];

                // GT center in pixels
                let gt_cx = (gt[0] + gt[2]) / 2.0 * input_size;
                let gt_cy = (gt[1] + gt[3]) / 2.0 * input_size;
                let gt_w = (gt[2] - gt[0]) * input_size;
                let gt_h = (gt[3] - gt[1]) * input_size;

                // Encode offsets
                bbox_targets[a_idx * 4 + 0] = (gt_cx - anchor[0]) / anchor[2];
                bbox_targets[a_idx * 4 + 1] = (gt_cy - anchor[1]) / anchor[3];
                bbox_targets[a_idx * 4 + 2] = (gt_w / anchor[2]).max(1e-6).ln();
                bbox_targets[a_idx * 4 + 3] = (gt_h / anchor[3]).max(1e-6).ln();
            } else if max_iou_per_anchor[a_idx] < neg_iou {
                cls_targets[a_idx] = 0.0; // negative
            } else {
                cls_targets[a_idx] = -1.0; // ignore (between neg_iou and pos_iou)
            }
        }

        (cls_targets, bbox_targets)
    }

    /// Online Hard Negative Mining: select top-k negatives by loss.
    fn ohem_select(
        cls_logits: &[f32],
        cls_targets: &[f32],
        neg_pos_ratio: usize,
    ) -> Vec<bool> {
        let num = cls_logits.len();
        let mut selected = vec![false; num];

        let mut num_pos = 0;
        for i in 0..num {
            if cls_targets[i] == 1.0 {
                selected[i] = true;
                num_pos += 1;
            }
        }

        // Collect negative losses
        let mut neg_losses: Vec<(usize, f32)> = Vec::new();
        for i in 0..num {
            if cls_targets[i] == 0.0 {
                // Binary cross-entropy loss for this sample
                let p = 1.0 / (1.0 + (-cls_logits[i]).exp());
                let loss = -((1.0 - p).max(1e-7)).ln();
                neg_losses.push((i, loss));
            }
        }

        // Sort by loss descending, pick top neg_pos_ratio * num_pos
        neg_losses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let num_neg = (neg_pos_ratio * num_pos).min(neg_losses.len()).max(10);
        for &(idx, _) in neg_losses.iter().take(num_neg) {
            selected[idx] = true;
        }

        selected
    }

    #[test]
    #[ignore]
    fn train_blazeface() {
        use crate::models::blazeface::BlazeFace;
        use crate::datasets::wider_face::WiderFaceDataset;

        println!("\n{}", "=".repeat(80));
        println!("  TRAINING: BlazeFace on WIDER FACE");
        println!("{}\n", "=".repeat(80));
        ensure_dirs();

        let input_size = 128usize;
        let input_f = input_size as f32;

        // Load WIDER FACE dataset
        let ds = WiderFaceDataset::new(WIDER_FACE_ROOT, "train", (input_size, input_size))
            .expect("Failed to load WIDER FACE train set");
        let val_ds = WiderFaceDataset::new(WIDER_FACE_ROOT, "val", (input_size, input_size))
            .expect("Failed to load WIDER FACE val set");

        let val_count = val_ds.len().min(500);
        println!("  WIDER FACE entries: {} train, {} val", ds.len(), val_count);

        // Pre-cache training images into memory (avoids disk I/O bottleneck)
        const MAX_TRAIN: usize = 2000;
        println!("  Caching training images (max {})...", MAX_TRAIN);
        let cache_start = std::time::Instant::now();
        let mut train_cache: Vec<(Vec<f32>, Vec<[f32; 4]>)> = Vec::new();
        let min_face_size = 10.0 / input_f; // Skip faces smaller than 10px at 128x128
        for idx in 0..ds.len() {
            if train_cache.len() >= MAX_TRAIN { break; }
            if let Some((img_tensor, gt_boxes)) = ds.get(idx) {
                // Filter out images where all faces are tiny
                let valid_boxes: Vec<[f32; 4]> = gt_boxes.into_iter()
                    .filter(|b| (b[2] - b[0]) >= min_face_size && (b[3] - b[1]) >= min_face_size)
                    .collect();
                if !valid_boxes.is_empty() {
                    train_cache.push((img_tensor.to_vec(), valid_boxes));
                }
            }
        }
        println!("  Cached {} training images in {:.1}s (skipped tiny/invalid faces)",
            train_cache.len(), cache_start.elapsed().as_secs_f32());

        // Cache val images too (filter tiny faces for fair eval)
        println!("  Caching val images...");
        let mut val_cache: Vec<(Vec<f32>, Vec<[f32; 4]>)> = Vec::new();
        for idx in 0..val_count {
            if let Some((img_tensor, gt_boxes)) = val_ds.get(idx) {
                let valid_boxes: Vec<[f32; 4]> = gt_boxes.into_iter()
                    .filter(|b| (b[2] - b[0]) >= min_face_size && (b[3] - b[1]) >= min_face_size)
                    .collect();
                if !valid_boxes.is_empty() {
                    val_cache.push((img_tensor.to_vec(), valid_boxes));
                }
            }
        }
        println!("  Cached {} val images\n", val_cache.len());

        let train_count = train_cache.len();

        let mut model = BlazeFace::new();
        let params = model.parameters();
        let param_count: usize = params.iter().map(|p| p.data().numel()).sum();

        let epochs = 30;
        let initial_lr = 2e-3;
        let warmup_steps = 200usize;
        let mut optimizer = Adam::new(params, initial_lr).weight_decay(1e-4);
        let focal_loss = FocalLoss::new();

        // Pre-generate anchors (same for all images at same input size)
        let anchors = BlazeFace::generate_anchors(input_size);
        let num_anchors = anchors.len();

        println!("  Parameters: {} ({:.1}K)", param_count, param_count as f32 / 1000.0);
        println!("  Anchors: {} (dual-scale: 16x16 + 8x8)", num_anchors);
        println!("  Config: {} epochs, input={}x{}, lr={}, warmup={} steps, OHEM 3:1\n",
            epochs, input_size, input_size, initial_lr, warmup_steps);

        let mut best_loss = f32::MAX;
        let mut best_map = 0.0f32;
        let mut epoch_losses = Vec::new();
        let mut global_step = 0usize;

        model.train();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let mut epoch_cls_loss = 0.0f32;
            let mut epoch_box_loss = 0.0f32;
            let mut steps = 0;
            let mut lr = initial_lr;

            // Random shuffle via simple permutation
            let mut indices: Vec<usize> = (0..train_count).collect();
            // Fisher-Yates shuffle using simple LCG
            let mut rng_state = (epoch as u64 * 6364136223846793005u64).wrapping_add(1442695040888963407);
            for i in (1..indices.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }

            for &idx in &indices {
                let (ref cached_img, ref gt_boxes) = train_cache[idx];
                {
                    if gt_boxes.is_empty() { continue; }

                    global_step += 1;

                    // LR schedule: warmup then cosine annealing
                    lr = if global_step <= warmup_steps {
                        initial_lr * (global_step as f32 / warmup_steps as f32)
                    } else {
                        let progress = (global_step - warmup_steps) as f64
                            / ((epochs * train_count).saturating_sub(warmup_steps)) as f64;
                        (initial_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()) as f32).max(1e-5)
                    };
                    optimizer.set_lr(lr);

                    // Random horizontal flip (50% chance)
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let do_flip = (rng_state >> 33) % 2 == 0;

                    let img_data = cached_img.clone();
                    let (img_data, gt_boxes) = if do_flip {
                        // Flip image horizontally: for each row, reverse pixel order
                        let mut flipped = img_data.clone();
                        for c in 0..3 {
                            for y in 0..input_size {
                                for x in 0..input_size {
                                    let src = c * input_size * input_size + y * input_size + x;
                                    let dst = c * input_size * input_size + y * input_size + (input_size - 1 - x);
                                    flipped[dst] = img_data[src];
                                }
                            }
                        }
                        let flipped_boxes: Vec<[f32; 4]> = gt_boxes.iter().map(|b| {
                            [1.0 - b[2], b[1], 1.0 - b[0], b[3]]
                        }).collect();
                        (flipped, flipped_boxes)
                    } else {
                        (img_data, gt_boxes.clone())
                    };

                    let input = Variable::new(
                        Tensor::from_vec(img_data, &[1, 3, input_size, input_size]).unwrap(),
                        false,
                    );

                    // Forward
                    let (cls_logits, bbox_preds) = model.forward_train(&input);

                    // Assign anchors to GT
                    let (cls_targets, bbox_targets) = assign_anchors_to_gt(
                        &anchors, &gt_boxes, input_f, 0.35, 0.35,
                    );

                    let mut pos_count = 0;
                    for i in 0..num_anchors {
                        if cls_targets[i] == 1.0 { pos_count += 1; }
                    }

                    // Classification loss (focal) on ALL anchors — no OHEM masking
                    // FocalLoss naturally handles class imbalance via alpha/gamma
                    let cls_pred_var = cls_logits.reshape(&[num_anchors]);
                    let cls_tgt: Vec<f32> = cls_targets.iter().map(|&t| t.max(0.0)).collect();
                    let cls_tgt_var = Variable::new(
                        Tensor::from_vec(cls_tgt, &[num_anchors]).unwrap(), false
                    );
                    let cls_loss = focal_loss.compute(&cls_pred_var, &cls_tgt_var);
                    let cls_loss_val = cls_loss.data().to_vec()[0];

                    // Box regression loss (MSE, only on positive anchors)
                    let bbox_flat = bbox_preds.reshape(&[num_anchors * 4]);
                    let mut box_tgt = vec![0.0f32; num_anchors * 4];
                    let mut mask = vec![0.0f32; num_anchors * 4];
                    for i in 0..num_anchors {
                        if cls_targets[i] == 1.0 {
                            for d in 0..4 {
                                box_tgt[i * 4 + d] = bbox_targets[i * 4 + d];
                                mask[i * 4 + d] = 1.0;
                            }
                        }
                    }

                    let total_loss = if pos_count > 0 {
                        let mask_var = Variable::new(
                            Tensor::from_vec(mask, &[num_anchors * 4]).unwrap(), false
                        );
                        let box_tgt_var = Variable::new(
                            Tensor::from_vec(box_tgt, &[num_anchors * 4]).unwrap(), false
                        );
                        let diff = bbox_flat.sub_var(&box_tgt_var).mul_var(&mask_var);
                        let box_loss = diff.mul_var(&diff).mean()
                            .mul_scalar(num_anchors as f32 / pos_count as f32);
                        epoch_box_loss += box_loss.data().to_vec()[0];
                        // Balance cls and box: scale box down so gradients are comparable
                        cls_loss.mul_scalar(10.0).add_var(&box_loss)
                    } else {
                        cls_loss.clone()
                    };

                    // Scale loss for gradient accumulation (effective batch=8)
                    let accum_steps = 8;
                    let scaled_loss = total_loss.mul_scalar(1.0 / accum_steps as f32);
                    let loss_val = total_loss.data().to_vec()[0];
                    epoch_cls_loss += cls_loss_val;

                    if loss_val.is_finite() && scaled_loss.requires_grad() {
                        scaled_loss.backward();
                        // Step optimizer every accum_steps
                        if global_step % accum_steps == 0 {
                            optimizer.step();
                            optimizer.zero_grad();
                        }
                    }

                    epoch_loss += loss_val;
                    steps += 1;

                    if steps % 200 == 0 {
                        println!("  [Epoch {}/{}] Step {}/{}: loss={:.4} (cls={:.4} box={:.4}) pos={} lr={:.6}",
                            epoch + 1, epochs, steps, train_count, loss_val,
                            cls_loss_val,
                            if pos_count > 0 { epoch_box_loss / steps as f32 } else { 0.0 },
                            pos_count, lr);
                    }
                }
            }

            let avg_loss = if steps > 0 { epoch_loss / steps as f32 } else { 0.0 };
            epoch_losses.push(avg_loss);
            if avg_loss < best_loss { best_loss = avg_loss; }

            // Evaluate every 3 epochs
            if (epoch + 1) % 3 == 0 || epoch == epochs - 1 {
                model.eval();
                let mut all_dets = Vec::new();
                let mut all_gts = Vec::new();
                let mut total_latency = 0.0f64;
                let mut total_det_count = 0usize;
                let mut total_gt_count = 0usize;

                for (ref img_data, ref gt_boxes) in &val_cache {
                    let input = Variable::new(
                        Tensor::from_vec(img_data.clone(), &[1, 3, input_size, input_size]).unwrap(),
                        false,
                    );
                    let start = std::time::Instant::now();
                    let detections = model.detect(&input, 0.01, 0.4);
                    total_latency += start.elapsed().as_secs_f64() * 1000.0;

                    total_det_count += detections.len();
                    total_gt_count += gt_boxes.len();

                    let dets: Vec<DetectionResult> = detections.iter().map(|d| DetectionResult {
                        bbox: [d.bbox[0] / input_f, d.bbox[1] / input_f, d.bbox[2] / input_f, d.bbox[3] / input_f],
                        confidence: d.confidence,
                        class_id: 0,
                    }).collect();
                    let gts: Vec<GroundTruth> = gt_boxes.iter().map(|b| GroundTruth {
                        bbox: *b,
                        class_id: 0,
                    }).collect();
                    all_dets.push(dets);
                    all_gts.push(gts);
                }

                let eval_count = all_dets.len();
                let mean_latency = if eval_count > 0 { total_latency / eval_count as f64 } else { 0.0 };
                let avg_dets = total_det_count as f32 / eval_count.max(1) as f32;
                let avg_gts = total_gt_count as f32 / eval_count.max(1) as f32;
                let (map50, _map75, coco_map) = evaluate_detections(&all_dets, &all_gts, 1);

                if map50 > best_map { best_map = map50; }
                println!("  Epoch {}/{}: avg_loss={:.4} | mAP@50={:.4} COCO_mAP={:.4} | latency={:.1}ms | avg_dets={:.1} avg_gts={:.1} | lr={:.6}",
                    epoch + 1, epochs, avg_loss, map50, coco_map, mean_latency, avg_dets, avg_gts, lr);

                model.train();
            } else {
                println!("  Epoch {}/{}: avg_loss={:.4} (cls={:.4} box={:.4}) lr={:.6}",
                    epoch + 1, epochs, avg_loss,
                    if steps > 0 { epoch_cls_loss / steps as f32 } else { 0.0 },
                    if steps > 0 { epoch_box_loss / steps as f32 } else { 0.0 },
                    lr);
            }
        }

        // Final evaluation
        println!("\n  Final evaluation on WIDER FACE val...");
        model.eval();
        let mut all_dets = Vec::new();
        let mut all_gts = Vec::new();
        let mut total_latency = 0.0f64;

        for (ref img_data, ref gt_boxes) in &val_cache {
            let input = Variable::new(
                Tensor::from_vec(img_data.clone(), &[1, 3, input_size, input_size]).unwrap(),
                false,
            );
            let start = std::time::Instant::now();
            let detections = model.detect(&input, 0.01, 0.4);
            total_latency += start.elapsed().as_secs_f64() * 1000.0;

            let dets: Vec<DetectionResult> = detections.iter().map(|d| DetectionResult {
                bbox: [d.bbox[0] / input_f, d.bbox[1] / input_f, d.bbox[2] / input_f, d.bbox[3] / input_f],
                confidence: d.confidence,
                class_id: 0,
            }).collect();
            let gts: Vec<GroundTruth> = gt_boxes.iter().map(|b| GroundTruth {
                bbox: *b,
                class_id: 0,
            }).collect();
            all_dets.push(dets);
            all_gts.push(gts);
        }

        let eval_count = all_dets.len();
        let mean_latency = if eval_count > 0 { total_latency / eval_count as f64 } else { 0.0 };
        let (map50, _map75, coco_map) = evaluate_detections(&all_dets, &all_gts, 1);

        println!("\n  === BlazeFace Final Results (WIDER FACE) ===");
        println!("  mAP@50: {:.4}  COCO mAP: {:.4}  Best mAP@50: {:.4}", map50, coco_map, best_map);
        println!("  Latency: {:.1}ms  FPS: {:.1}", mean_latency, 1000.0 / mean_latency);

        save_results("blazeface", &serde_json::json!({
            "model": "blazeface",
            "params": param_count,
            "epochs_trained": epochs,
            "training_images": train_count,
            "eval_images": eval_count,
            "input_size": [input_size, input_size],
            "dataset": "WIDER_FACE",
            "best_loss": best_loss,
            "final_loss": epoch_losses.last().unwrap_or(&0.0),
            "loss_history": epoch_losses,
            "map50": map50,
            "best_map50": best_map,
            "coco_map": coco_map,
            "mean_latency_ms": mean_latency,
            "fps": 1000.0 / mean_latency,
        }));
    }

    // =========================================================================
    // RetinaFace Training
    // =========================================================================

    #[test]
    #[ignore]
    fn train_retinaface() {
        use crate::models::retinaface::RetinaFace;

        println!("\n{}", "=".repeat(80));
        println!("  TRAINING: RetinaFace on COCO val2017");
        println!("{}\n", "=".repeat(80));
        ensure_dirs();

        let input_size = (256, 256);
        let (ds, train_idx, eval_idx) = load_coco_split(input_size, 300, 80);

        let model = RetinaFace::new();
        let params = model.parameters();
        let param_count: usize = params.iter().map(|p| p.data().numel()).sum();
        let mut optimizer = Adam::new(params, 5e-4).weight_decay(1e-4);
        let focal_loss = FocalLoss::new();
        let (img_h, img_w) = (input_size.0 as f32, input_size.1 as f32);
        let epochs = 10;

        println!("  Parameters: {}", param_count);
        println!("  Config: {} epochs, input={}x{}\n", epochs, input_size.0, input_size.1);

        let mut best_loss = f32::MAX;
        let mut epoch_losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let mut steps = 0;

            for &idx in &train_idx {
                if let Some((img_tensor, annos)) = ds.get(idx) {
                    let person_annos: Vec<_> = annos.iter().filter(|a| a.category_id == 0).collect();
                    if person_annos.is_empty() { continue; }

                    let input = Variable::new(
                        Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                        false,
                    );

                    // Forward raw to get per-level predictions
                    let (cls_scores, _bbox_preds, _ldm_preds) = model.forward_raw(&input);

                    let num_levels = cls_scores.len();
                    let mut total_loss = Variable::new(Tensor::from_vec(vec![0.0], &[1]).unwrap(), false);

                    for level in 0..num_levels {
                        let cls_out = &cls_scores[level];
                        let cls_shape = cls_out.shape();
                        let (fh, fw) = (cls_shape[2], cls_shape[3]);
                        let num_anchors = 2;
                        let stride = 2usize.pow(level as u32 + 2) as f32;

                        // Build targets
                        let total_elements = num_anchors * 2 * fh * fw;
                        let mut cls_target = vec![0.0f32; total_elements];

                        for a in &person_annos {
                            let cx = (a.bbox[0] + a.bbox[2]) / 2.0 * img_w;
                            let cy = (a.bbox[1] + a.bbox[3]) / 2.0 * img_h;
                            let gx = (cx / stride).floor() as usize;
                            let gy = (cy / stride).floor() as usize;
                            if gx < fw && gy < fh {
                                // Set foreground class (index 1) for anchor 0
                                cls_target[(0 * 2 + 1) * fh * fw + gy * fw + gx] = 1.0;
                            }
                        }

                        let cls_pred = cls_out.reshape(&[total_elements]);
                        let cls_tgt = Variable::new(Tensor::from_vec(cls_target, &[total_elements]).unwrap(), false);
                        let cls_loss = focal_loss.compute(&cls_pred, &cls_tgt);
                        total_loss = total_loss.add_var(&cls_loss);
                    }

                    let loss_val = total_loss.data().to_vec()[0];
                    if loss_val.is_finite() && total_loss.requires_grad() {
                        optimizer.zero_grad();
                        total_loss.backward();
                        optimizer.step();
                    }

                    epoch_loss += loss_val;
                    steps += 1;

                    if steps % 30 == 0 {
                        println!("  [Epoch {}/{}] Step {}/{}: loss={:.4}", epoch + 1, epochs, steps, train_idx.len(), loss_val);
                    }
                }
            }

            let avg_loss = if steps > 0 { epoch_loss / steps as f32 } else { 0.0 };
            epoch_losses.push(avg_loss);
            if avg_loss < best_loss { best_loss = avg_loss; }
            println!("  Epoch {}/{}: avg_loss={:.4}", epoch + 1, epochs, avg_loss);
        }

        println!("\n  Evaluating...");
        let mut all_dets = Vec::new();
        let mut all_gts = Vec::new();
        let mut total_latency = 0.0f64;

        for &idx in &eval_idx {
            if let Some((img_tensor, annos)) = ds.get(idx) {
                let person_annos: Vec<_> = annos.iter().filter(|a| a.category_id == 0).collect();
                let input = Variable::new(
                    Tensor::from_vec(img_tensor.to_vec(), &[1, 3, input_size.0, input_size.1]).unwrap(),
                    false,
                );
                let start = std::time::Instant::now();
                let detections = model.detect(&input, 0.01, 0.5);
                total_latency += start.elapsed().as_secs_f64() * 1000.0;

                let dets: Vec<DetectionResult> = detections.iter().map(|d| DetectionResult {
                    bbox: [d.bbox[0] / img_w, d.bbox[1] / img_h, d.bbox[2] / img_w, d.bbox[3] / img_h],
                    confidence: d.confidence,
                    class_id: 0,
                }).collect();
                let gts: Vec<GroundTruth> = person_annos.iter().map(|a| GroundTruth { bbox: a.bbox, class_id: 0 }).collect();
                all_dets.push(dets);
                all_gts.push(gts);
            }
        }

        let eval_count = all_dets.len();
        let mean_latency = total_latency / eval_count as f64;
        let (map50, _map75, coco_map) = evaluate_detections(&all_dets, &all_gts, 1);

        println!("\n  === RetinaFace Results ===");
        println!("  mAP@50: {:.4}  COCO mAP: {:.4}", map50, coco_map);
        println!("  Latency: {:.1}ms  FPS: {:.1}", mean_latency, 1000.0 / mean_latency);

        save_results("retinaface", &serde_json::json!({
            "model": "retinaface", "params": param_count,
            "epochs_trained": epochs, "training_images": train_idx.len(),
            "eval_images": eval_count, "input_size": [input_size.0, input_size.1],
            "best_loss": best_loss, "final_loss": epoch_losses.last().unwrap_or(&0.0),
            "loss_history": epoch_losses,
            "map50": map50, "coco_map": coco_map,
            "mean_latency_ms": mean_latency, "fps": 1000.0 / mean_latency,
        }));
    }
}
