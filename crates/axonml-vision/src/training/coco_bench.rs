//! COCO Evaluation Benchmarks — Full Detection Model Evaluation Suite
//!
//! # File
//! `crates/axonml-vision/src/training/coco_bench.rs`
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

#![allow(dead_code)]

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::datasets::CocoAnnotation;
use crate::ops::Detection;
use crate::training::metrics::{DetectionResult, GroundTruth, compute_coco_map, compute_map};

// =============================================================================
// Conversion Helpers
// =============================================================================

/// Convert a `Detection` (pixel coords) to `DetectionResult` (normalized [0,1]).
fn detection_to_result(det: &Detection, img_w: f32, img_h: f32) -> DetectionResult {
    DetectionResult {
        bbox: [
            (det.bbox[0] / img_w).clamp(0.0, 1.0),
            (det.bbox[1] / img_h).clamp(0.0, 1.0),
            (det.bbox[2] / img_w).clamp(0.0, 1.0),
            (det.bbox[3] / img_h).clamp(0.0, 1.0),
        ],
        confidence: det.confidence,
        class_id: det.class_id,
    }
}

/// Convert a `CocoAnnotation` (already normalized [0,1]) to `GroundTruth`.
fn annotation_to_gt(anno: &CocoAnnotation) -> GroundTruth {
    GroundTruth {
        bbox: anno.bbox,
        class_id: anno.category_id,
    }
}

// =============================================================================
// Evaluation Runner (public API)
// =============================================================================

/// Evaluate Helios on a COCO dataset, returning (mAP@50, COCO_mAP@[0.5:0.95]).
pub fn evaluate_helios_coco(
    model: &crate::models::helios::Helios,
    dataset: &crate::datasets::CocoDataset,
    input_size: (usize, usize),
    score_threshold: f32,
    nms_threshold: f32,
    max_images: usize,
) -> (f32, f32) {
    let num_classes = dataset.num_classes();
    let n = if max_images > 0 && max_images < dataset.len() {
        max_images
    } else {
        dataset.len()
    };

    let (img_h, img_w) = (input_size.0 as f32, input_size.1 as f32);

    let mut all_detections: Vec<Vec<DetectionResult>> = Vec::with_capacity(n);
    let mut all_ground_truths: Vec<Vec<GroundTruth>> = Vec::with_capacity(n);

    for i in 0..n {
        if let Some((img_tensor, annotations)) = dataset.get(i) {
            let input = Variable::new(img_tensor, false);
            let dets = model.detect(&input, score_threshold, nms_threshold);

            let det_results: Vec<DetectionResult> = dets
                .iter()
                .map(|d| detection_to_result(d, img_w, img_h))
                .collect();

            let gts: Vec<GroundTruth> = annotations.iter().map(annotation_to_gt).collect();

            all_detections.push(det_results);
            all_ground_truths.push(gts);
        }
    }

    if all_detections.is_empty() {
        return (0.0, 0.0);
    }

    let map50 = compute_map(&all_detections, &all_ground_truths, num_classes, 0.5);
    let coco_map = compute_coco_map(&all_detections, &all_ground_truths, num_classes);

    (map50, coco_map)
}

// =============================================================================
// Internal Benchmark Infrastructure
// =============================================================================

/// Comprehensive evaluation result with per-IoU breakdown.
struct FullEvalResult {
    name: String,
    resolution: String,
    params: usize,
    size_mb: f64,
    n_images: usize,
    total_dets: usize,
    total_gts: usize,
    avg_dets_per_image: f64,
    avg_confidence: f64,
    avg_box_area: f64,
    map50: f32,
    map75: f32,
    coco_map: f32,
    warmup_ms: f64,
    latency_ms: f64,
    fps: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
}

impl FullEvalResult {
    fn print(&self) {
        println!("  {:<20} {}", "Model:", self.name);
        println!("  {:<20} {}", "Resolution:", self.resolution);
        println!(
            "  {:<20} {} ({:.2} MB f32)",
            "Parameters:", self.params, self.size_mb
        );
        println!("  {:<20} {}", "Images:", self.n_images);
        println!(
            "  {:<20} {} (avg {:.1}/img)",
            "Detections:", self.total_dets, self.avg_dets_per_image
        );
        println!("  {:<20} {}", "Ground Truths:", self.total_gts);
        println!("  {:<20} {:.4}", "Avg Confidence:", self.avg_confidence);
        println!("  {:<20} {:.4}", "Avg Box Area:", self.avg_box_area);
        println!("  {:<20} {:.4}", "mAP@50:", self.map50);
        println!("  {:<20} {:.4}", "mAP@75:", self.map75);
        println!("  {:<20} {:.4}", "COCO mAP@[.5:.95]:", self.coco_map);
        println!("  {:<20} {:.1}ms", "Warmup latency:", self.warmup_ms);
        println!(
            "  {:<20} {:.1}ms (mean)",
            "Inference latency:", self.latency_ms
        );
        println!("  {:<20} {:.1}ms", "P50 latency:", self.p50_latency_ms);
        println!("  {:<20} {:.1}ms", "P95 latency:", self.p95_latency_ms);
        println!("  {:<20} {:.1}", "Throughput (FPS):", self.fps);
    }
}

/// Generate diverse synthetic ground truth for benchmark evaluation.
///
/// Produces varied box sizes, positions, and aspect ratios to stress-test
/// detection pipelines beyond trivial 2-box-per-image patterns.
fn diverse_gt(image_idx: usize, num_classes: usize) -> Vec<GroundTruth> {
    // Deterministic pseudo-random from image index
    let seed = image_idx as f32;
    let hash = |v: f32| -> f32 { ((v * 127.1 + 311.7).sin() * 43_758.547).fract().abs() };

    let n_boxes = 2 + (hash(seed) * 4.0) as usize; // 2-5 boxes per image
    let mut gts = Vec::with_capacity(n_boxes);

    for b in 0..n_boxes {
        let h1 = hash(seed + b as f32 * 7.3);
        let h2 = hash(seed + b as f32 * 13.1 + 1.0);
        let h3 = hash(seed + b as f32 * 23.7 + 2.0);
        let h4 = hash(seed + b as f32 * 37.9 + 3.0);

        // Varied box sizes: small (0.05-0.15), medium (0.15-0.35), large (0.35-0.6)
        let size_class = (h1 * 3.0) as usize;
        let (min_s, max_s) = match size_class {
            0 => (0.05, 0.15), // small objects
            1 => (0.15, 0.35), // medium objects
            _ => (0.35, 0.60), // large objects
        };
        let w = min_s + h2 * (max_s - min_s);
        let h = min_s + h3 * (max_s - min_s);

        // Random position (ensure box stays within [0,1])
        let x1 = h4 * (1.0 - w);
        let y1 = hash(seed + b as f32 * 51.3 + 4.0) * (1.0 - h);

        let class_id = ((image_idx + b) * 7 + b * 3) % num_classes;

        gts.push(GroundTruth {
            bbox: [x1, y1, x1 + w, y1 + h],
            class_id,
        });
    }

    gts
}

/// Create a synthetic input tensor with varied pixel patterns.
fn make_input(image_idx: usize, h: usize, w: usize) -> Variable {
    let size = 3 * h * w;
    let mut data = Vec::with_capacity(size);
    let base = 0.2 + ((image_idx as f32 * 0.37).sin().abs() * 0.6);

    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                // Gradient + noise pattern per channel
                let gx = x as f32 / w as f32;
                let gy = y as f32 / h as f32;
                let v = base + (c as f32 * 0.1) + (gx * 0.2) - (gy * 0.1);
                data.push(v.clamp(0.0, 1.0));
            }
        }
    }

    Variable::new(Tensor::from_vec(data, &[1, 3, h, w]).unwrap(), false)
}

/// Run a full evaluation with warmup, multi-image inference, and statistics.
fn run_full_eval<F>(
    name: &str,
    resolution: &str,
    param_count: usize,
    input_size: (usize, usize),
    num_classes: usize,
    n_images: usize,
    warmup_images: usize,
    detect_fn: &mut F,
) -> FullEvalResult
where
    F: FnMut(Variable, f32, f32) -> Vec<DetectionResult>,
{
    let (ih, iw) = input_size;
    let img_w = iw as f32;
    let img_h = ih as f32;

    // Warmup phase (not timed for throughput, but measured separately)
    let warmup_start = std::time::Instant::now();
    for i in 0..warmup_images {
        let input = make_input(1000 + i, ih, iw);
        let _ = detect_fn(input, img_w, img_h);
    }
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0 / warmup_images.max(1) as f64;

    // Timed evaluation phase
    let mut all_detections: Vec<Vec<DetectionResult>> = Vec::with_capacity(n_images);
    let mut all_ground_truths: Vec<Vec<GroundTruth>> = Vec::with_capacity(n_images);
    let mut per_image_times: Vec<f64> = Vec::with_capacity(n_images);
    let mut all_confidences: Vec<f32> = Vec::new();
    let mut all_box_areas: Vec<f32> = Vec::new();

    for i in 0..n_images {
        let input = make_input(i, ih, iw);
        let gts = diverse_gt(i, num_classes);

        let t0 = std::time::Instant::now();
        let det_results = detect_fn(input, img_w, img_h);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        per_image_times.push(dt);

        // Collect statistics
        for d in &det_results {
            all_confidences.push(d.confidence);
            let bw = (d.bbox[2] - d.bbox[0]).abs();
            let bh = (d.bbox[3] - d.bbox[1]).abs();
            all_box_areas.push(bw * bh);
        }

        all_detections.push(det_results);
        all_ground_truths.push(gts);
    }

    // Compute metrics
    let map50 = compute_map(&all_detections, &all_ground_truths, num_classes, 0.5);
    let map75 = compute_map(&all_detections, &all_ground_truths, num_classes, 0.75);
    let coco_map = compute_coco_map(&all_detections, &all_ground_truths, num_classes);

    // Latency statistics
    per_image_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_ms = per_image_times.iter().sum::<f64>() / n_images as f64;
    let p50_ms = per_image_times[n_images / 2];
    let p95_idx = (n_images as f64 * 0.95) as usize;
    let p95_ms = per_image_times[p95_idx.min(n_images - 1)];

    // Detection statistics
    let total_dets: usize = all_detections.iter().map(|d| d.len()).sum();
    let total_gts: usize = all_ground_truths.iter().map(|g| g.len()).sum();
    let avg_conf = if all_confidences.is_empty() {
        0.0
    } else {
        all_confidences.iter().sum::<f32>() as f64 / all_confidences.len() as f64
    };
    let avg_area = if all_box_areas.is_empty() {
        0.0
    } else {
        all_box_areas.iter().sum::<f32>() as f64 / all_box_areas.len() as f64
    };

    FullEvalResult {
        name: name.to_string(),
        resolution: resolution.to_string(),
        params: param_count,
        size_mb: param_count as f64 * 4.0 / 1_048_576.0,
        n_images,
        total_dets,
        total_gts,
        avg_dets_per_image: total_dets as f64 / n_images as f64,
        avg_confidence: avg_conf,
        avg_box_area: avg_area,
        map50,
        map75,
        coco_map,
        warmup_ms,
        latency_ms: mean_ms,
        fps: 1000.0 / mean_ms,
        p50_latency_ms: p50_ms,
        p95_latency_ms: p95_ms,
    }
}

fn print_summary_table(results: &[FullEvalResult]) {
    println!(
        "\n  {:<18} {:>6} {:>7} {:>6} {:>8} {:>8} {:>8} {:>9} {:>9} {:>6}",
        "Model",
        "Res",
        "Params",
        "MB",
        "mAP@50",
        "mAP@75",
        "COCO mAP",
        "Mean(ms)",
        "P95(ms)",
        "FPS"
    );
    println!("  {}", "-".repeat(104));
    for r in results {
        println!(
            "  {:<18} {:>6} {:>7} {:>5.1} {:>8.4} {:>8.4} {:>8.4} {:>8.1}ms {:>8.1}ms {:>6.1}",
            r.name,
            r.resolution,
            format_params(r.params),
            r.size_mb,
            r.map50,
            r.map75,
            r.coco_map,
            r.latency_ms,
            r.p95_latency_ms,
            r.fps
        );
    }
}

fn format_params(p: usize) -> String {
    if p >= 1_000_000 {
        format!("{:.1}M", p as f64 / 1_000_000.0)
    } else if p >= 1_000 {
        format!("{:.0}K", p as f64 / 1_000.0)
    } else {
        format!("{p}")
    }
}

fn print_detection_stats(results: &[FullEvalResult]) {
    println!(
        "\n  {:<18} {:>6} {:>8} {:>10} {:>10} {:>10}",
        "Model", "Res", "Dets", "Avg/Img", "Avg Conf", "Avg Area"
    );
    println!("  {}", "-".repeat(72));
    for r in results {
        println!(
            "  {:<18} {:>6} {:>8} {:>10.1} {:>10.4} {:>10.4}",
            r.name,
            r.resolution,
            r.total_dets,
            r.avg_dets_per_image,
            r.avg_confidence,
            r.avg_box_area
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_nn::Module;
    use std::time::Instant;

    fn count_params(params: &[axonml_nn::Parameter]) -> usize {
        params
            .iter()
            .map(|p| p.variable().data().to_vec().len())
            .sum()
    }

    // =========================================================================
    // Helios Multi-Resolution Benchmark
    // =========================================================================

    #[test]
    fn coco_bench_helios_multiresolution() {
        let num_classes = 20;

        println!("\n{}", "=".repeat(80));
        println!("  HELIOS MULTI-RESOLUTION COCO BENCHMARK");
        println!("  Classes: {num_classes} | Images: 50/res | Warmup: 3");
        println!("{}", "=".repeat(80));

        let mut results = Vec::new();

        // Helios-Nano at 128, 320
        {
            use crate::models::helios::Helios;
            let model = Helios::nano(num_classes);
            let pc = count_params(&model.parameters());

            for &(h, w, label) in &[(128, 128, "128"), (320, 320, "320")] {
                println!("\n--- Helios-Nano @ {label}x{label} ---");
                let r = run_full_eval(
                    "Helios-Nano",
                    label,
                    pc,
                    (h, w),
                    num_classes,
                    50,
                    3,
                    &mut |input, iw, ih| {
                        model
                            .detect(&input, 0.001, 0.45)
                            .iter()
                            .map(|d| detection_to_result(d, iw, ih))
                            .collect()
                    },
                );
                r.print();
                results.push(r);
            }
        }

        // Helios-Small at 128, 320
        {
            use crate::models::helios::Helios;
            let model = Helios::small(num_classes);
            let pc = count_params(&model.parameters());

            for &(h, w, label) in &[(128, 128, "128"), (320, 320, "320")] {
                println!("\n--- Helios-Small @ {label}x{label} ---");
                let r = run_full_eval(
                    "Helios-Small",
                    label,
                    pc,
                    (h, w),
                    num_classes,
                    30,
                    2,
                    &mut |input, iw, ih| {
                        model
                            .detect(&input, 0.001, 0.45)
                            .iter()
                            .map(|d| detection_to_result(d, iw, ih))
                            .collect()
                    },
                );
                r.print();
                results.push(r);
            }
        }

        println!("\n{}", "=".repeat(80));
        println!("  HELIOS SUMMARY");
        print_summary_table(&results);
        print_detection_stats(&results);
        println!("{}\n", "=".repeat(80));

        for r in &results {
            assert!(
                r.map50.is_finite(),
                "{} {}: mAP@50 not finite",
                r.name,
                r.resolution
            );
            assert!(
                r.coco_map.is_finite(),
                "{} {}: COCO mAP not finite",
                r.name,
                r.resolution
            );
        }
    }

    // =========================================================================
    // NanoDet Multi-Resolution Benchmark
    // =========================================================================

    #[test]
    fn coco_bench_nanodet_multiresolution() {
        let num_classes = 20;

        println!("\n{}", "=".repeat(80));
        println!("  NANODET MULTI-RESOLUTION COCO BENCHMARK");
        println!("  Classes: {num_classes} | Images: 50/res | Warmup: 3");
        println!("{}", "=".repeat(80));

        let mut results = Vec::new();

        use crate::models::nanodet::NanoDet;
        let model = NanoDet::new(num_classes);
        let pc = count_params(&model.parameters());

        for &(h, w, label) in &[(128, 128, "128"), (320, 320, "320")] {
            println!("\n--- NanoDet @ {label}x{label} ---");
            let r = run_full_eval(
                "NanoDet",
                label,
                pc,
                (h, w),
                num_classes,
                50,
                3,
                &mut |input, iw, ih| {
                    model
                        .detect(&input, 0.001, 0.45)
                        .iter()
                        .map(|d| detection_to_result(d, iw, ih))
                        .collect()
                },
            );
            r.print();
            results.push(r);
        }

        println!("\n{}", "=".repeat(80));
        println!("  NANODET SUMMARY");
        print_summary_table(&results);
        print_detection_stats(&results);
        println!("{}\n", "=".repeat(80));

        for r in &results {
            assert!(r.map50.is_finite());
        }
    }

    // =========================================================================
    // Nexus Multi-Resolution Benchmark
    // =========================================================================

    #[test]
    fn coco_bench_nexus_multiresolution() {
        let num_classes = 20;

        println!("\n{}", "=".repeat(80));
        println!("  NEXUS MULTI-RESOLUTION COCO BENCHMARK");
        println!("  Classes: {num_classes} | Images: 50/res | Warmup: 3");
        println!("{}", "=".repeat(80));

        let mut results = Vec::new();

        use crate::models::nexus::Nexus;
        let mut model = Nexus::new();
        model.eval();
        let pc = count_params(&model.parameters());

        for &(h, w, label) in &[(64, 64, "64"), (128, 128, "128")] {
            model.reset();
            println!("\n--- Nexus @ {label}x{label} ---");
            let r = run_full_eval(
                "Nexus",
                label,
                pc,
                (h, w),
                num_classes,
                50,
                3,
                &mut |input, iw, ih| {
                    let dets = model.detect(&input);
                    dets.iter()
                        .map(|d| DetectionResult {
                            bbox: [
                                (d.bbox_mean[0] / iw).clamp(0.0, 1.0),
                                (d.bbox_mean[1] / ih).clamp(0.0, 1.0),
                                (d.bbox_mean[2] / iw).clamp(0.0, 1.0),
                                (d.bbox_mean[3] / ih).clamp(0.0, 1.0),
                            ],
                            confidence: d.confidence,
                            class_id: d.class_id,
                        })
                        .collect()
                },
            );
            r.print();
            results.push(r);
        }

        println!("\n{}", "=".repeat(80));
        println!("  NEXUS SUMMARY");
        print_summary_table(&results);
        print_detection_stats(&results);
        println!("{}\n", "=".repeat(80));

        for r in &results {
            assert!(r.map50.is_finite());
        }
    }

    // =========================================================================
    // Face Detectors Benchmark (Phantom, BlazeFace, RetinaFace)
    // =========================================================================

    #[test]
    fn coco_bench_face_detectors() {
        println!("\n{}", "=".repeat(80));
        println!("  FACE DETECTOR COCO BENCHMARK");
        println!("  Task: Face detection (1 class) | Images: 50/model | Warmup: 3");
        println!("{}", "=".repeat(80));

        let mut results = Vec::new();
        let num_classes = 1;

        let _face_gt = |image_idx: usize, _nc: usize| -> Vec<GroundTruth> {
            // Diverse face-like ground truth
            let h = |v: f32| -> f32 { ((v * 127.1 + 311.7).sin() * 43_758.547).fract().abs() };
            let n = 1 + (h(image_idx as f32) * 3.0) as usize; // 1-3 faces
            (0..n)
                .map(|b| {
                    let s = 0.08 + h(image_idx as f32 + b as f32 * 11.3) * 0.35;
                    let x = h(image_idx as f32 + b as f32 * 23.7) * (1.0 - s);
                    let y = h(image_idx as f32 + b as f32 * 37.1) * (1.0 - s);
                    GroundTruth {
                        bbox: [x, y, x + s, y + s * 1.2],
                        class_id: 0,
                    }
                })
                .collect()
        };

        // Phantom
        {
            use crate::models::phantom::Phantom;
            let mut model = Phantom::new();
            model.eval();
            let pc = count_params(&model.parameters());

            for &(h, w, label) in &[(64, 64, "64"), (128, 128, "128")] {
                model.reset();
                println!("\n--- Phantom @ {label}x{label} ---");
                let r = run_full_eval(
                    "Phantom",
                    label,
                    pc,
                    (h, w),
                    num_classes,
                    50,
                    3,
                    &mut |input, iw, ih| {
                        let dets = model.detect_frame(&input);
                        dets.iter()
                            .map(|d| DetectionResult {
                                bbox: [
                                    (d.bbox[0] / iw).clamp(0.0, 1.0),
                                    (d.bbox[1] / ih).clamp(0.0, 1.0),
                                    (d.bbox[2] / iw).clamp(0.0, 1.0),
                                    (d.bbox[3] / ih).clamp(0.0, 1.0),
                                ],
                                confidence: d.confidence,
                                class_id: 0,
                            })
                            .collect()
                    },
                );
                r.print();
                results.push(r);
            }
        }

        // BlazeFace
        {
            use crate::models::blazeface::BlazeFace;
            let model = BlazeFace::new();
            let pc = count_params(&model.parameters());

            for &(h, w, label) in &[(128, 128, "128"), (256, 256, "256")] {
                println!("\n--- BlazeFace @ {label}x{label} ---");
                let r = run_full_eval(
                    "BlazeFace",
                    label,
                    pc,
                    (h, w),
                    num_classes,
                    50,
                    3,
                    &mut |input, iw, ih| {
                        let dets = model.detect(&input, 0.001, 0.45);
                        dets.iter()
                            .map(|d| DetectionResult {
                                bbox: [
                                    (d.bbox[0] / iw).clamp(0.0, 1.0),
                                    (d.bbox[1] / ih).clamp(0.0, 1.0),
                                    (d.bbox[2] / iw).clamp(0.0, 1.0),
                                    (d.bbox[3] / ih).clamp(0.0, 1.0),
                                ],
                                confidence: d.confidence,
                                class_id: 0,
                            })
                            .collect()
                    },
                );
                r.print();
                results.push(r);
            }
        }

        // RetinaFace
        {
            use crate::models::retinaface::RetinaFace;
            let model = RetinaFace::new();
            let pc = count_params(&model.parameters());

            for &(h, w, label) in &[(128, 128, "128"), (320, 320, "320")] {
                println!("\n--- RetinaFace @ {label}x{label} ---");
                let r = run_full_eval(
                    "RetinaFace",
                    label,
                    pc,
                    (h, w),
                    num_classes,
                    50,
                    3,
                    &mut |input, iw, ih| {
                        let dets = model.detect(&input, 0.001, 0.45);
                        dets.iter()
                            .map(|d| DetectionResult {
                                bbox: [
                                    (d.bbox[0] / iw).clamp(0.0, 1.0),
                                    (d.bbox[1] / ih).clamp(0.0, 1.0),
                                    (d.bbox[2] / iw).clamp(0.0, 1.0),
                                    (d.bbox[3] / ih).clamp(0.0, 1.0),
                                ],
                                confidence: d.confidence,
                                class_id: 0,
                            })
                            .collect()
                    },
                );
                r.print();
                results.push(r);
            }
        }

        println!("\n{}", "=".repeat(80));
        println!("  FACE DETECTOR SUMMARY");
        print_summary_table(&results);
        print_detection_stats(&results);
        println!("{}\n", "=".repeat(80));

        for r in &results {
            assert!(r.map50.is_finite());
        }
    }

    // =========================================================================
    // Cross-Model Comparison (all models, single resolution)
    // =========================================================================

    #[test]
    fn coco_bench_all_models_comparison() {
        let num_classes = 20;
        let n_images = 50;

        println!("\n{}", "=".repeat(80));
        println!("  ALL-MODEL COMPARISON BENCHMARK");
        println!("  Classes: {num_classes} | Images: {n_images} | Conf: 0.001 | NMS: 0.45");
        println!("{}", "=".repeat(80));

        let mut results = Vec::new();

        // Helios-Nano @ 128
        {
            use crate::models::helios::Helios;
            let model = Helios::nano(num_classes);
            let pc = count_params(&model.parameters());
            println!("\n--- Helios-Nano ---");
            let r = run_full_eval(
                "Helios-Nano",
                "128",
                pc,
                (128, 128),
                num_classes,
                n_images,
                3,
                &mut |input, iw, ih| {
                    model
                        .detect(&input, 0.001, 0.45)
                        .iter()
                        .map(|d| detection_to_result(d, iw, ih))
                        .collect()
                },
            );
            r.print();
            results.push(r);
        }

        // NanoDet @ 128
        {
            use crate::models::nanodet::NanoDet;
            let model = NanoDet::new(num_classes);
            let pc = count_params(&model.parameters());
            println!("\n--- NanoDet ---");
            let r = run_full_eval(
                "NanoDet",
                "128",
                pc,
                (128, 128),
                num_classes,
                n_images,
                3,
                &mut |input, iw, ih| {
                    model
                        .detect(&input, 0.001, 0.45)
                        .iter()
                        .map(|d| detection_to_result(d, iw, ih))
                        .collect()
                },
            );
            r.print();
            results.push(r);
        }

        // Nexus @ 64
        {
            use crate::models::nexus::Nexus;
            let mut model = Nexus::new();
            model.eval();
            let pc = count_params(&model.parameters());
            println!("\n--- Nexus ---");
            let r = run_full_eval(
                "Nexus",
                "64",
                pc,
                (64, 64),
                num_classes,
                n_images,
                3,
                &mut |input, iw, ih| {
                    let dets = model.detect(&input);
                    dets.iter()
                        .map(|d| DetectionResult {
                            bbox: [
                                (d.bbox_mean[0] / iw).clamp(0.0, 1.0),
                                (d.bbox_mean[1] / ih).clamp(0.0, 1.0),
                                (d.bbox_mean[2] / iw).clamp(0.0, 1.0),
                                (d.bbox_mean[3] / ih).clamp(0.0, 1.0),
                            ],
                            confidence: d.confidence,
                            class_id: d.class_id,
                        })
                        .collect()
                },
            );
            r.print();
            results.push(r);
        }

        // Phantom @ 64 (1 class)
        {
            use crate::models::phantom::Phantom;
            let mut model = Phantom::new();
            model.eval();
            let pc = count_params(&model.parameters());
            println!("\n--- Phantom ---");
            let r = run_full_eval(
                "Phantom",
                "64",
                pc,
                (64, 64),
                1,
                n_images,
                3,
                &mut |input, iw, ih| {
                    model
                        .detect_frame(&input)
                        .iter()
                        .map(|d| DetectionResult {
                            bbox: [
                                (d.bbox[0] / iw).clamp(0.0, 1.0),
                                (d.bbox[1] / ih).clamp(0.0, 1.0),
                                (d.bbox[2] / iw).clamp(0.0, 1.0),
                                (d.bbox[3] / ih).clamp(0.0, 1.0),
                            ],
                            confidence: d.confidence,
                            class_id: 0,
                        })
                        .collect()
                },
            );
            r.print();
            results.push(r);
        }

        // BlazeFace @ 128 (1 class)
        {
            use crate::models::blazeface::BlazeFace;
            let model = BlazeFace::new();
            let pc = count_params(&model.parameters());
            println!("\n--- BlazeFace ---");
            let r = run_full_eval(
                "BlazeFace",
                "128",
                pc,
                (128, 128),
                1,
                n_images,
                3,
                &mut |input, iw, ih| {
                    model
                        .detect(&input, 0.001, 0.45)
                        .iter()
                        .map(|d| DetectionResult {
                            bbox: [
                                (d.bbox[0] / iw).clamp(0.0, 1.0),
                                (d.bbox[1] / ih).clamp(0.0, 1.0),
                                (d.bbox[2] / iw).clamp(0.0, 1.0),
                                (d.bbox[3] / ih).clamp(0.0, 1.0),
                            ],
                            confidence: d.confidence,
                            class_id: 0,
                        })
                        .collect()
                },
            );
            r.print();
            results.push(r);
        }

        // RetinaFace @ 128 (1 class)
        {
            use crate::models::retinaface::RetinaFace;
            let model = RetinaFace::new();
            let pc = count_params(&model.parameters());
            println!("\n--- RetinaFace ---");
            let r = run_full_eval(
                "RetinaFace",
                "128",
                pc,
                (128, 128),
                1,
                n_images,
                3,
                &mut |input, iw, ih| {
                    model
                        .detect(&input, 0.001, 0.45)
                        .iter()
                        .map(|d| DetectionResult {
                            bbox: [
                                (d.bbox[0] / iw).clamp(0.0, 1.0),
                                (d.bbox[1] / ih).clamp(0.0, 1.0),
                                (d.bbox[2] / iw).clamp(0.0, 1.0),
                                (d.bbox[3] / ih).clamp(0.0, 1.0),
                            ],
                            confidence: d.confidence,
                            class_id: 0,
                        })
                        .collect()
                },
            );
            r.print();
            results.push(r);
        }

        println!("\n{}", "=".repeat(80));
        println!("  ALL-MODEL COMPARISON");
        print_summary_table(&results);
        println!();
        print_detection_stats(&results);
        println!("{}\n", "=".repeat(80));

        for r in &results {
            assert!(r.map50.is_finite(), "{}: mAP@50 not finite", r.name);
            assert!(r.coco_map.is_finite(), "{}: COCO mAP not finite", r.name);
            assert!(r.map50 >= 0.0 && r.map50 <= 1.0);
            assert!(r.coco_map >= 0.0 && r.coco_map <= 1.0);
        }
    }

    // =========================================================================
    // mAP Pipeline Validation
    // =========================================================================

    #[test]
    fn coco_bench_map_validation() {
        println!("\n{}", "=".repeat(80));
        println!("  mAP PIPELINE VALIDATION");
        println!("{}", "=".repeat(80));

        // Test 1: Perfect detections
        {
            let num_classes = 3;
            let all_dets = vec![
                vec![
                    DetectionResult {
                        bbox: [0.1, 0.1, 0.4, 0.4],
                        confidence: 0.95,
                        class_id: 0,
                    },
                    DetectionResult {
                        bbox: [0.5, 0.5, 0.9, 0.9],
                        confidence: 0.90,
                        class_id: 1,
                    },
                ],
                vec![DetectionResult {
                    bbox: [0.2, 0.2, 0.6, 0.6],
                    confidence: 0.85,
                    class_id: 2,
                }],
            ];
            let all_gts = vec![
                vec![
                    GroundTruth {
                        bbox: [0.1, 0.1, 0.4, 0.4],
                        class_id: 0,
                    },
                    GroundTruth {
                        bbox: [0.5, 0.5, 0.9, 0.9],
                        class_id: 1,
                    },
                ],
                vec![GroundTruth {
                    bbox: [0.2, 0.2, 0.6, 0.6],
                    class_id: 2,
                }],
            ];

            let map50 = compute_map(&all_dets, &all_gts, num_classes, 0.5);
            let map75 = compute_map(&all_dets, &all_gts, num_classes, 0.75);
            let coco = compute_coco_map(&all_dets, &all_gts, num_classes);
            println!("\n  Perfect detections:");
            println!("    mAP@50:  {map50:.4} (expect 1.0)");
            println!("    mAP@75:  {map75:.4} (expect 1.0)");
            println!("    COCO:    {coco:.4} (expect 1.0)");
            assert!((map50 - 1.0).abs() < 1e-4);
            assert!((map75 - 1.0).abs() < 1e-4);
            assert!((coco - 1.0).abs() < 1e-4);
        }

        // Test 2: Mixed quality
        {
            let num_classes = 3;
            let all_dets = vec![vec![
                DetectionResult {
                    bbox: [0.0, 0.0, 0.5, 0.5],
                    confidence: 0.9,
                    class_id: 0,
                },
                DetectionResult {
                    bbox: [0.5, 0.5, 1.0, 1.0],
                    confidence: 0.8,
                    class_id: 1,
                },
            ]];
            let all_gts = vec![vec![
                GroundTruth {
                    bbox: [0.0, 0.0, 0.5, 0.5],
                    class_id: 0,
                },
                GroundTruth {
                    bbox: [0.5, 0.5, 1.0, 1.0],
                    class_id: 1,
                },
                GroundTruth {
                    bbox: [0.2, 0.2, 0.3, 0.3],
                    class_id: 1,
                },
                GroundTruth {
                    bbox: [0.7, 0.1, 0.9, 0.3],
                    class_id: 2,
                },
            ]];

            let map50 = compute_map(&all_dets, &all_gts, num_classes, 0.5);
            println!("\n  Mixed quality (partial recall):");
            println!("    mAP@50:  {map50:.4} (expect ~0.5)");
            assert!(map50 > 0.3 && map50 < 0.7);
        }

        // Test 3: All false positives
        {
            let all_dets = vec![vec![
                DetectionResult {
                    bbox: [0.0, 0.0, 0.1, 0.1],
                    confidence: 0.9,
                    class_id: 0,
                },
                DetectionResult {
                    bbox: [0.9, 0.9, 1.0, 1.0],
                    confidence: 0.8,
                    class_id: 0,
                },
            ]];
            let all_gts = vec![vec![GroundTruth {
                bbox: [0.4, 0.4, 0.6, 0.6],
                class_id: 0,
            }]];

            let map50 = compute_map(&all_dets, &all_gts, 1, 0.5);
            println!("\n  All false positives:");
            println!("    mAP@50:  {map50:.4} (expect ~0.0)");
            assert!(map50 < 0.15);
        }

        // Test 4: Many classes, sparse detections
        {
            let nc = 80;
            let mut all_dets = Vec::new();
            let mut all_gts = Vec::new();
            for i in 0..20 {
                let cls = i % nc;
                all_dets.push(vec![DetectionResult {
                    bbox: [0.1, 0.1, 0.5, 0.5],
                    confidence: 0.7,
                    class_id: cls,
                }]);
                all_gts.push(vec![GroundTruth {
                    bbox: [0.1, 0.1, 0.5, 0.5],
                    class_id: cls,
                }]);
            }
            let map50 = compute_map(&all_dets, &all_gts, nc, 0.5);
            let coco = compute_coco_map(&all_dets, &all_gts, nc);
            println!("\n  80 classes, 20 images, 1 det/img (perfect per-image):");
            println!("    mAP@50:  {map50:.4} (expect 1.0)");
            println!("    COCO:    {coco:.4} (expect 1.0)");
            assert!((map50 - 1.0).abs() < 1e-4);
        }

        println!("\n  All validation checks PASSED");
        println!("{}\n", "=".repeat(80));
    }

    // =========================================================================
    // Real COCO Evaluation (requires $COCO_ROOT)
    // =========================================================================

    #[test]
    fn coco_bench_real_helios() {
        let coco_root = match std::env::var("COCO_ROOT") {
            Ok(p) => p,
            Err(_) => {
                println!("\n--- Skipping real COCO benchmark (set COCO_ROOT to enable) ---");
                return;
            }
        };

        let image_dir = format!("{coco_root}/val2017");
        let anno_json = format!("{coco_root}/annotations/instances_val2017.json");

        println!("\n{}", "=".repeat(80));
        println!("  REAL COCO val2017 BENCHMARK");
        println!("  COCO_ROOT: {coco_root}");
        println!("{}", "=".repeat(80));

        for &(input_h, input_w, variant, max_default) in &[
            (320, 320, "nano", 200usize),
            (320, 320, "small", 100),
            (640, 640, "nano", 50),
        ] {
            let input_size = (input_h, input_w);
            let dataset =
                match crate::datasets::CocoDataset::new(&image_dir, &anno_json, input_size) {
                    Ok(ds) => ds,
                    Err(e) => {
                        println!("  Failed to load: {e}");
                        return;
                    }
                };

            let num_classes = dataset.num_classes();
            let max_images = std::env::var("COCO_MAX_IMAGES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(max_default);

            let model = match variant {
                "nano" => crate::models::helios::Helios::nano(num_classes),
                "small" => crate::models::helios::Helios::small(num_classes),
                _ => unreachable!(),
            };

            println!("\n--- Helios-{variant} @ {input_h}x{input_w} ({max_images} images) ---");
            let start = Instant::now();
            let (map50, coco_map) =
                evaluate_helios_coco(&model, &dataset, input_size, 0.001, 0.45, max_images);
            let elapsed = start.elapsed();
            let per_image = elapsed.as_secs_f64() * 1000.0 / max_images as f64;
            let _map75 = {
                // Quick separate mAP@75 computation would need refactoring evaluate_helios_coco
                // For now report the two main metrics
                0.0f32
            };

            println!("  mAP@50:       {map50:.4}");
            println!("  COCO mAP:     {coco_map:.4}");
            println!(
                "  latency:      {per_image:.1}ms/img ({:.1} FPS)",
                1000.0 / per_image
            );
            println!("  total time:   {:.1}s", elapsed.as_secs_f64());

            assert!(map50.is_finite());
            assert!(coco_map.is_finite());
        }

        println!("{}\n", "=".repeat(80));
    }
}
