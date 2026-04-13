//! Benchmark Mnemosyne — Face Verification Evaluation
//!
//! Loads the trained model and evaluates face verification accuracy
//! on LFW pairs: same-identity vs different-identity classification.
//!
//! Reports: accuracy, ROC-AUC, EER, FAR/FRR at multiple thresholds.
//!
//! ```bash
//! cargo run --example bench_mnemosyne --release -p axonml-vision
//! cargo run --example bench_mnemosyne --release -p axonml-vision -- \
//!   --model /opt/AxonML/checkpoints/mnemosyne/best_model.axonml \
//!   --pairs 3000
//! ```

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_serialize::load_state_dict;
use axonml_tensor::Tensor;

use axonml_vision::models::biometric::MnemosyneIdentity;

// =============================================================================
// Dataset Loading (same as training)
// =============================================================================

struct IdentityData {
    faces: Vec<Vec<f32>>,
}

fn load_identities(data_dir: &Path) -> Vec<IdentityData> {
    let mut identities = Vec::new();
    let mut files: Vec<_> = fs::read_dir(data_dir)
        .expect("Failed to read data dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .map(|f| f.to_string_lossy().starts_with("identity_"))
                .unwrap_or(false)
        })
        .collect();
    files.sort_by_key(|e| e.file_name());

    for entry in &files {
        let path = entry.path();
        let mut file = fs::File::open(&path).unwrap();
        let mut header = [0u8; 16];
        file.read_exact(&mut header).unwrap();

        let num = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let c = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let h = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let w = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;

        let face_size = c * h * w;
        let mut byte_buf = vec![0u8; num * face_size * 4];
        file.read_exact(&mut byte_buf).unwrap();

        let all_data: Vec<f32> = byte_buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let mut faces = Vec::with_capacity(num);
        for i in 0..num {
            faces.push(all_data[i * face_size..(i + 1) * face_size].to_vec());
        }
        identities.push(IdentityData { faces });
    }
    identities
}

// =============================================================================
// Crystallize a face sequence → embedding
// =============================================================================

fn crystallize_to_embedding(model: &MnemosyneIdentity, faces: &[&Vec<f32>]) -> Vec<f32> {
    let mut hidden: Option<Variable> = None;

    for face_data in faces {
        let face = Variable::new(
            Tensor::from_vec((*face_data).clone(), &[1, 3, 64, 64]).unwrap(),
            false,
        );
        let (h, _, _, _) = model.crystallize_step(&face, hidden.as_ref());
        hidden = Some(h);
    }

    model.extract_identity(&hidden.unwrap())
}

/// Cosine similarity between two L2-normalized embeddings.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// =============================================================================
// Pair Generation
// =============================================================================

fn lcg_range(state: &mut u64, max: usize) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as usize) % max
}

struct VerificationPair {
    score: f32,
    is_same: bool,
}

fn generate_pairs(
    model: &MnemosyneIdentity,
    identities: &[IdentityData],
    num_pairs: usize,
    seq_len: usize,
) -> Vec<VerificationPair> {
    let mut pairs = Vec::with_capacity(num_pairs * 2);
    let mut rng = 12345u64;

    let usable: Vec<usize> = identities
        .iter()
        .enumerate()
        .filter(|(_, id)| id.faces.len() >= seq_len * 2)
        .map(|(i, _)| i)
        .collect();

    println!(
        "  Generating {} same + {} different pairs (seq_len={})...",
        num_pairs, num_pairs, seq_len
    );
    println!("  {} identities with {}+ faces", usable.len(), seq_len * 2);

    let start = Instant::now();

    // Same-identity pairs
    for i in 0..num_pairs {
        let id_idx = usable[lcg_range(&mut rng, usable.len())];
        let id = &identities[id_idx];

        // Two different subsequences from the same identity
        let seq_a: Vec<&Vec<f32>> = (0..seq_len)
            .map(|_| &id.faces[lcg_range(&mut rng, id.faces.len())])
            .collect();
        let seq_b: Vec<&Vec<f32>> = (0..seq_len)
            .map(|_| &id.faces[lcg_range(&mut rng, id.faces.len())])
            .collect();

        let emb_a = crystallize_to_embedding(model, &seq_a);
        let emb_b = crystallize_to_embedding(model, &seq_b);

        pairs.push(VerificationPair {
            score: cosine_sim(&emb_a, &emb_b),
            is_same: true,
        });

        if (i + 1) % 500 == 0 {
            let elapsed = start.elapsed().as_secs_f32();
            let rate = (i + 1) as f32 / elapsed;
            println!("    Same pairs: {}/{} ({:.0}/s)", i + 1, num_pairs, rate);
        }
    }

    // Different-identity pairs
    for i in 0..num_pairs {
        let id_a = usable[lcg_range(&mut rng, usable.len())];
        let mut id_b = usable[lcg_range(&mut rng, usable.len())];
        while id_b == id_a {
            id_b = usable[lcg_range(&mut rng, usable.len())];
        }

        let seq_a: Vec<&Vec<f32>> = (0..seq_len)
            .map(|_| &identities[id_a].faces[lcg_range(&mut rng, identities[id_a].faces.len())])
            .collect();
        let seq_b: Vec<&Vec<f32>> = (0..seq_len)
            .map(|_| &identities[id_b].faces[lcg_range(&mut rng, identities[id_b].faces.len())])
            .collect();

        let emb_a = crystallize_to_embedding(model, &seq_a);
        let emb_b = crystallize_to_embedding(model, &seq_b);

        pairs.push(VerificationPair {
            score: cosine_sim(&emb_a, &emb_b),
            is_same: false,
        });

        if (i + 1) % 500 == 0 {
            let elapsed = start.elapsed().as_secs_f32();
            let total_done = num_pairs + i + 1;
            let rate = total_done as f32 / elapsed;
            println!("    Diff pairs: {}/{} ({:.0}/s)", i + 1, num_pairs, rate);
        }
    }

    let elapsed = start.elapsed();
    println!(
        "  Generated {} pairs in {:.1}s",
        pairs.len(),
        elapsed.as_secs_f32()
    );

    pairs
}

// =============================================================================
// Metrics
// =============================================================================

fn compute_metrics(pairs: &[VerificationPair]) {
    let same: Vec<f32> = pairs
        .iter()
        .filter(|p| p.is_same)
        .map(|p| p.score)
        .collect();
    let diff: Vec<f32> = pairs
        .iter()
        .filter(|p| !p.is_same)
        .map(|p| p.score)
        .collect();

    let same_mean: f32 = same.iter().sum::<f32>() / same.len() as f32;
    let diff_mean: f32 = diff.iter().sum::<f32>() / diff.len() as f32;
    let same_min = same.iter().cloned().fold(f32::MAX, f32::min);
    let same_max = same.iter().cloned().fold(f32::MIN, f32::max);
    let diff_min = diff.iter().cloned().fold(f32::MAX, f32::min);
    let diff_max = diff.iter().cloned().fold(f32::MIN, f32::max);

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(" Mnemosyne Face Verification Benchmark");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  Score Distribution:");
    println!(
        "    Same-identity:  mean={:.4}, min={:.4}, max={:.4} (n={})",
        same_mean,
        same_min,
        same_max,
        same.len()
    );
    println!(
        "    Diff-identity:  mean={:.4}, min={:.4}, max={:.4} (n={})",
        diff_mean,
        diff_min,
        diff_max,
        diff.len()
    );
    println!(
        "    Separation:     {:.4} (same_mean - diff_mean)",
        same_mean - diff_mean
    );
    println!();

    // ROC-AUC
    let auc = compute_auc(&same, &diff);
    println!("  ROC-AUC: {:.4}", auc);

    // Accuracy at various thresholds
    println!();
    println!("  Threshold Analysis:");
    println!(
        "  {:>10} {:>8} {:>8} {:>8} {:>8}",
        "Threshold", "Acc", "FAR", "FRR", "F1"
    );
    println!("  {}", "-".repeat(50));

    let thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut best_acc = 0.0f32;
    let mut best_thresh = 0.0f32;

    for &thresh in &thresholds {
        let tp = same.iter().filter(|&&s| s >= thresh).count();
        let fn_ = same.iter().filter(|&&s| s < thresh).count();
        let fp = diff.iter().filter(|&&s| s >= thresh).count();
        let tn = diff.iter().filter(|&&s| s < thresh).count();

        let acc = (tp + tn) as f32 / (tp + tn + fp + fn_) as f32;
        let far = fp as f32 / (fp + tn).max(1) as f32; // false accept rate
        let frr = fn_ as f32 / (fn_ + tp).max(1) as f32; // false reject rate
        let precision = tp as f32 / (tp + fp).max(1) as f32;
        let recall = tp as f32 / (tp + fn_).max(1) as f32;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        if acc > best_acc {
            best_acc = acc;
            best_thresh = thresh;
        }

        println!(
            "  {:>10.2} {:>7.1}% {:>7.1}% {:>7.1}% {:>8.4}",
            thresh,
            acc * 100.0,
            far * 100.0,
            frr * 100.0,
            f1
        );
    }

    // EER (Equal Error Rate) — find threshold where FAR ≈ FRR
    let eer = compute_eer(&same, &diff);

    println!();
    println!("  Summary:");
    println!(
        "    Best accuracy: {:.1}% at threshold {:.2}",
        best_acc * 100.0,
        best_thresh
    );
    println!("    ROC-AUC:       {:.4}", auc);
    println!("    EER:           {:.2}%", eer * 100.0);
    println!();

    // Verdict
    if auc > 0.95 {
        println!("  Verdict: EXCELLENT — model reliably distinguishes identities");
    } else if auc > 0.85 {
        println!("  Verdict: GOOD — model shows meaningful identity separation");
    } else if auc > 0.70 {
        println!("  Verdict: FAIR — model partially learned identity features");
    } else if auc > 0.55 {
        println!("  Verdict: POOR — barely above random chance");
    } else {
        println!("  Verdict: FAILED — not learning identity discrimination");
    }
    println!("═══════════════════════════════════════════════════════════");
}

fn compute_auc(same_scores: &[f32], diff_scores: &[f32]) -> f32 {
    // Wilcoxon-Mann-Whitney AUC
    let mut auc = 0.0f64;
    for &s in same_scores {
        for &d in diff_scores {
            if s > d {
                auc += 1.0;
            } else if (s - d).abs() < 1e-8 {
                auc += 0.5;
            }
        }
    }
    (auc / (same_scores.len() as f64 * diff_scores.len() as f64)) as f32
}

fn compute_eer(same_scores: &[f32], diff_scores: &[f32]) -> f32 {
    let mut best_eer = 1.0f32;
    for i in 0..100 {
        let thresh = i as f32 / 100.0;
        let far =
            diff_scores.iter().filter(|&&s| s >= thresh).count() as f32 / diff_scores.len() as f32;
        let frr =
            same_scores.iter().filter(|&&s| s < thresh).count() as f32 / same_scores.len() as f32;
        let diff = (far - frr).abs();
        if diff < (best_eer - 0.0).abs() {
            best_eer = (far + frr) / 2.0;
        }
    }
    best_eer
}

// =============================================================================
// Model Loading
// =============================================================================

fn load_model(model_path: &Path) -> MnemosyneIdentity {
    let model = MnemosyneIdentity::new();

    if model_path.exists() {
        // Load state dict (from checkpoint or standalone)
        let state_dict = if let Ok(checkpoint) = axonml_serialize::load_checkpoint(model_path) {
            println!("  Checkpoint epoch: {}", checkpoint.epoch());
            Some(checkpoint.model_state)
        } else {
            load_state_dict(model_path).ok()
        };

        if let Some(state_dict) = state_dict {
            // Collect all saved tensors sorted by shape for deterministic matching
            let saved_tensors: Vec<_> = state_dict
                .entries()
                .filter_map(|(_, entry)| entry.data.to_tensor().ok())
                .collect();

            // Match model parameters to saved tensors by shape, in order
            let params = model.parameters();
            let mut loaded = 0;
            let mut used = vec![false; saved_tensors.len()];

            for param in &params {
                let param_data = param.data();
                let param_shape = param_data.shape();
                // Find first unused saved tensor with matching shape
                for (i, saved) in saved_tensors.iter().enumerate() {
                    if !used[i] && saved.shape() == param_shape {
                        param.update_data(saved.clone());
                        used[i] = true;
                        loaded += 1;
                        break;
                    }
                }
            }
            println!(
                "  Loaded {}/{} parameters from {}",
                loaded,
                params.len(),
                model_path.display()
            );
        }
    } else {
        println!(
            "  WARNING: No model found at {} — using random weights!",
            model_path.display()
        );
    }

    model
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let data_dir = args
        .iter()
        .position(|a| a == "--data-dir")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("/opt/datasets/lfw/processed"));

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| {
            PathBuf::from("/opt/AxonML/checkpoints/mnemosyne/checkpoint_best.axonml")
        });

    let num_pairs: usize = args
        .iter()
        .position(|a| a == "--pairs")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(1000);

    let seq_len: usize = args
        .iter()
        .position(|a| a == "--seq-len")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(5);

    println!("Loading model...");
    let model = load_model(&model_path);

    println!("Loading identities...");
    let identities = load_identities(&data_dir);
    let total: usize = identities.iter().map(|id| id.faces.len()).sum();
    println!("  {} identities, {} faces", identities.len(), total);

    let pairs = generate_pairs(&model, &identities, num_pairs, seq_len);
    compute_metrics(&pairs);
}
