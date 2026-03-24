//! Train Mnemosyne — Face Identity via Temporal Crystallization
//!
//! Trains on LFW (Labeled Faces in the Wild) dataset.
//! Uses triplet loss with convergence regularization.
//!
//! ```bash
//! cargo run --example train_mnemosyne --release -p axonml-vision
//! cargo run --example train_mnemosyne --release -p axonml-vision --features cuda
//! ```

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_core::Device;
use axonml_nn::{Module, Parameter};
use axonml_optim::{AdamW, Optimizer};
use axonml_serialize::{
    load_checkpoint, save_checkpoint, save_model, Checkpoint, StateDict, TrainingState,
};
use axonml_tensor::Tensor;

use axonml_vision::models::biometric::{CrystallizationLoss, MnemosyneIdentity};

// =============================================================================
// Dataset
// =============================================================================

struct IdentityData {
    faces: Vec<Vec<f32>>, // Each face: 3*64*64 = 12288 floats
}

fn load_lfw_identities(data_dir: &Path) -> Vec<IdentityData> {
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
// Batched Triplet Mining
// =============================================================================

const FACE_SIZE: usize = 3 * 64 * 64;

/// Mine a batch of B triplets and return batched face data for each seq step.
/// Returns: for each seq step, (anchor_batch, pos_batch, neg_batch) as flat Vec<f32>
/// Each batch is [B, 3, 64, 64].
fn mine_batch(
    identities: &[IdentityData],
    rng: &mut u64,
    batch_size: usize,
    seq_len: usize,
) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut steps = Vec::with_capacity(seq_len);

    // Pre-select triplet identities for the whole batch
    let mut triplet_ids: Vec<(usize, usize)> = Vec::with_capacity(batch_size); // (anchor_id, neg_id)
    let valid: Vec<usize> = identities
        .iter()
        .enumerate()
        .filter(|(_, id)| id.faces.len() >= 2)
        .map(|(i, _)| i)
        .collect();

    for _ in 0..batch_size {
        let a_idx = valid[lcg_range(rng, valid.len())];
        let mut n_idx = lcg_range(rng, identities.len());
        while n_idx == a_idx {
            n_idx = lcg_range(rng, identities.len());
        }
        triplet_ids.push((a_idx, n_idx));
    }

    // For each sequence step, sample one face per triplet member and batch them
    for _ in 0..seq_len {
        let mut anchor_data = Vec::with_capacity(batch_size * FACE_SIZE);
        let mut pos_data = Vec::with_capacity(batch_size * FACE_SIZE);
        let mut neg_data = Vec::with_capacity(batch_size * FACE_SIZE);

        for &(a_id, n_id) in &triplet_ids {
            let anchor_id = &identities[a_id];
            let neg_id = &identities[n_id];

            let a_face = lcg_range(rng, anchor_id.faces.len());
            let mut p_face = lcg_range(rng, anchor_id.faces.len());
            if anchor_id.faces.len() > 1 {
                while p_face == a_face {
                    p_face = lcg_range(rng, anchor_id.faces.len());
                }
            }
            let n_face = lcg_range(rng, neg_id.faces.len());

            anchor_data.extend_from_slice(&anchor_id.faces[a_face]);
            pos_data.extend_from_slice(&anchor_id.faces[p_face]);
            neg_data.extend_from_slice(&neg_id.faces[n_face]);
        }

        steps.push((anchor_data, pos_data, neg_data));
    }

    steps
}

fn lcg_range(state: &mut u64, max: usize) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as usize) % max
}

// =============================================================================
// Batched Crystallization
// =============================================================================

/// Crystallize a batch of sequences through Mnemosyne.
/// `steps` has seq_len entries, each is [B, 3, 64, 64] as flat f32.
/// Returns (final_hidden [B, hidden_dim], mean_velocity [B, 1]).
fn crystallize_batched(
    model: &MnemosyneIdentity,
    steps: &[Vec<f32>],
    batch_size: usize,
    device: &Device,
) -> (Variable, Variable) {
    let mut hidden: Option<Variable> = None;
    let mut velocities = Vec::new();

    for step_data in steps {
        let mut t = Tensor::from_vec(step_data.clone(), &[batch_size, 3, 64, 64]).unwrap();
        if device.is_gpu() {
            t = t.to_device(device.clone()).unwrap();
        }
        let face = Variable::new(t, false);
        let (h, velocity, _logvar, _quality) = model.crystallize_step(&face, hidden.as_ref());
        velocities.push(velocity);
        hidden = Some(h);
    }

    let final_hidden = hidden.unwrap();

    // Mean velocity across sequence steps
    let mean_vel = if velocities.len() == 1 {
        velocities.into_iter().next().unwrap()
    } else {
        let mut sum = velocities[0].clone();
        for v in &velocities[1..] {
            sum = sum.add_var(v);
        }
        sum.mul_scalar(1.0 / steps.len() as f32)
    };

    (final_hidden, mean_vel)
}

// =============================================================================
// L2 Normalize (graph-tracked)
// =============================================================================

fn l2_normalize_var(x: &Variable) -> Variable {
    let sq = x.mul_var(x);
    let sum_sq = sq.sum();
    let norm = sum_sq.add_scalar(1e-8).sqrt();
    x.div_var(&norm)
}

fn cosine_lr(base_lr: f32, warmup: usize, total: usize, epoch: usize) -> f32 {
    if epoch < warmup {
        base_lr * (epoch + 1) as f32 / warmup as f32
    } else {
        let progress = (epoch - warmup) as f32 / (total - warmup).max(1) as f32;
        let min_lr = base_lr * 0.01;
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

// =============================================================================
// Config
// =============================================================================

struct TrainConfig {
    data_dir: PathBuf,
    output_dir: PathBuf,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    seq_len: usize,
    batches_per_epoch: usize,
    save_every: usize,
    warmup_epochs: usize,
    /// Resume mode: "none", "latest", "best", or a file path
    resume: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("/opt/datasets/lfw/processed"),
            output_dir: PathBuf::from("/opt/AxonML/checkpoints/mnemosyne"),
            epochs: 50,
            lr: 1e-3,
            batch_size: 32,
            seq_len: 5,
            batches_per_epoch: 100,
            save_every: 10,
            warmup_epochs: 3,
            resume: "latest".to_string(),
        }
    }
}

impl TrainConfig {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut config = Self::default();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--data-dir" => { i += 1; config.data_dir = PathBuf::from(&args[i]); }
                "--output-dir" => { i += 1; config.output_dir = PathBuf::from(&args[i]); }
                "--epochs" => { i += 1; config.epochs = args[i].parse().unwrap(); }
                "--lr" => { i += 1; config.lr = args[i].parse().unwrap(); }
                "--batch-size" | "--bs" => { i += 1; config.batch_size = args[i].parse().unwrap(); }
                "--seq-len" => { i += 1; config.seq_len = args[i].parse().unwrap(); }
                "--batches" => { i += 1; config.batches_per_epoch = args[i].parse().unwrap(); }
                "--save-every" => { i += 1; config.save_every = args[i].parse().unwrap(); }
                "--resume" => { i += 1; config.resume = args[i].clone(); }
                "--fresh" => { config.resume = "none".to_string(); }
                _ => {}
            }
            i += 1;
        }
        config
    }
}

// =============================================================================
// Checkpoint Resume
// =============================================================================

fn find_checkpoint(output_dir: &Path, mode: &str) -> Option<PathBuf> {
    match mode {
        "none" => None,
        "latest" => {
            let p = output_dir.join("checkpoint_latest.axonml");
            if p.exists() { Some(p) } else {
                // Try best as fallback
                let b = output_dir.join("checkpoint_best.axonml");
                if b.exists() { Some(b) } else {
                    // Try any epoch checkpoint
                    let best_model = output_dir.join("best_model.axonml");
                    if best_model.exists() { Some(best_model) } else { None }
                }
            }
        }
        "best" => {
            let p = output_dir.join("checkpoint_best.axonml");
            if p.exists() { Some(p) } else {
                let b = output_dir.join("best_model.axonml");
                if b.exists() { Some(b) } else { None }
            }
        }
        path => {
            let p = PathBuf::from(path);
            if p.exists() { Some(p) } else { None }
        }
    }
}

fn load_model_weights(model: &MnemosyneIdentity, path: &Path) -> (usize, TrainingState) {
    // Try loading as full checkpoint first
    if let Ok(checkpoint) = load_checkpoint(path) {
        let state_dict = &checkpoint.model_state;
        let model_params = model.named_parameters();
        let mut loaded = 0;
        for (name, param) in &model_params {
            if let Some(entry) = state_dict.get(name) {
                if let Ok(tensor) = entry.data.to_tensor() {
                    if tensor.shape() == param.data().shape() {
                        param.update_data(tensor);
                        loaded += 1;
                    }
                }
            }
        }
        println!("  Loaded checkpoint: {} (epoch {}, {}/{} params)",
            path.display(), checkpoint.epoch(), loaded, model_params.len());
        return (checkpoint.epoch(), checkpoint.training_state.clone());
    }

    // Try loading as state dict (model weights only)
    if let Ok(state_dict) = axonml_serialize::load_state_dict(path) {
        let model_params = model.named_parameters();
        let mut loaded = 0;
        for (name, param) in &model_params {
            if let Some(entry) = state_dict.get(name) {
                if let Ok(tensor) = entry.data.to_tensor() {
                    if tensor.shape() == param.data().shape() {
                        param.update_data(tensor);
                        loaded += 1;
                    }
                }
            }
        }
        println!("  Loaded model weights: {} ({}/{} params)",
            path.display(), loaded, model_params.len());
        return (0, TrainingState::new());
    }

    println!("  WARNING: Failed to load checkpoint from {}", path.display());
    (0, TrainingState::new())
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let config = TrainConfig::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" Mnemosyne — Face Identity via Temporal Crystallization");
    println!(" Training on LFW (Labeled Faces in the Wild)");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    println!("Loading LFW identities from {}...", config.data_dir.display());
    let identities = load_lfw_identities(&config.data_dir);
    let total_faces: usize = identities.iter().map(|id| id.faces.len()).sum();
    let usable: usize = identities.iter().filter(|id| id.faces.len() >= 2).count();
    println!("  {} identities, {} faces ({} usable)", identities.len(), total_faces, usable);

    // Detect GPU
    let device = if cfg!(feature = "cuda") {
        match Device::cuda(0) {
            d @ Device::Cuda(_) => {
                println!("  Using GPU: {:?}", d);
                d
            }
            _ => {
                println!("  GPU not available, using CPU");
                Device::Cpu
            }
        }
    } else {
        println!("  Using CPU (build with --features cuda for GPU)");
        Device::Cpu
    };

    println!("\nCreating Mnemosyne model...");
    let model = MnemosyneIdentity::new();
    let param_count = model.parameters().iter().map(|p| p.numel()).sum::<usize>();
    println!("  Parameters: {}", param_count);

    // Move model to GPU
    if device.is_gpu() {
        for param in model.parameters() {
            param.to_device(device.clone());
        }
        println!("  Model moved to GPU");
    }

    let monitor = axonml::monitor::TrainingMonitor::new("Mnemosyne — Face Identity", param_count)
        .total_epochs(config.epochs)
        .batch_size(config.batch_size * config.batches_per_epoch)
        .launch();

    let mut optimizer = AdamW::new(model.parameters(), config.lr);
    let loss_fn = CrystallizationLoss::default();
    fs::create_dir_all(&config.output_dir).ok();

    // Resume from checkpoint
    let mut training_state = TrainingState::new();
    let mut best_loss = f32::MAX;
    let mut start_epoch = 0usize;

    if config.resume != "none" {
        if let Some(ckpt_path) = find_checkpoint(&config.output_dir, &config.resume) {
            let (epoch, state) = load_model_weights(&model, &ckpt_path);
            start_epoch = epoch;
            if let Some(bl) = state.best_metric {
                best_loss = bl;
            }
            training_state = state;
            // Re-move params to GPU after loading (weights were loaded on CPU)
            if device.is_gpu() {
                for param in model.parameters() {
                    param.to_device(device.clone());
                }
            }
        } else {
            println!("  No checkpoint found for resume='{}', starting fresh.", config.resume);
        }
    }

    let mut rng_state = 42u64 + start_epoch as u64 * 1000;

    let triplets_per_epoch = config.batch_size * config.batches_per_epoch;
    println!(
        "\nTraining: epochs {}-{}, batch={}, seq_len={}, {}/epoch ({} batches)",
        start_epoch + 1, config.epochs, config.batch_size, config.seq_len,
        triplets_per_epoch, config.batches_per_epoch
    );
    println!("  LR: {}, Warmup: {} epochs, Best so far: {:.4}", config.lr, config.warmup_epochs, best_loss);
    println!();

    let training_start = Instant::now();

    for epoch in start_epoch..config.epochs {
        let epoch_start = Instant::now();
        let lr = cosine_lr(config.lr, config.warmup_epochs, config.epochs, epoch);
        optimizer.set_lr(lr);

        let mut epoch_loss = 0.0f32;
        let mut epoch_velocity = 0.0f32;
        let mut num_batches = 0;

        for _ in 0..config.batches_per_epoch {
            // Mine a batch of triplets
            let steps = mine_batch(&identities, &mut rng_state, config.batch_size, config.seq_len);

            // Split steps into anchor/pos/neg
            let anchor_steps: Vec<Vec<f32>> = steps.iter().map(|(a, _, _)| a.clone()).collect();
            let pos_steps: Vec<Vec<f32>> = steps.iter().map(|(_, p, _)| p.clone()).collect();
            let neg_steps: Vec<Vec<f32>> = steps.iter().map(|(_, _, n)| n.clone()).collect();

            // Batched crystallization
            let (anchor_h, anchor_vel) = crystallize_batched(&model, &anchor_steps, config.batch_size, &device);
            let (pos_h, _) = crystallize_batched(&model, &pos_steps, config.batch_size, &device);
            let (neg_h, _) = crystallize_batched(&model, &neg_steps, config.batch_size, &device);

            // L2 normalize
            let anchor_emb = l2_normalize_var(&anchor_h);
            let pos_emb = l2_normalize_var(&pos_h);
            let neg_emb = l2_normalize_var(&neg_h);

            // Loss
            let loss = loss_fn.compute_var(&anchor_emb, &pos_emb, &neg_emb, &anchor_vel);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            let loss_val = loss.data().to_vec()[0];
            let vel_val = anchor_vel.data().to_vec()[0];

            epoch_loss += loss_val;
            epoch_velocity += vel_val;
            num_batches += 1;
            training_state.next_step();
            training_state.record_loss(loss_val);
        }

        let avg_loss = epoch_loss / num_batches as f32;
        let avg_velocity = epoch_velocity / num_batches as f32;
        let epoch_time = epoch_start.elapsed();

        println!(
            "Epoch {:3}/{} | loss: {:.4} | vel: {:.4} | lr: {:.6} | {:.1}s",
            epoch + 1, config.epochs, avg_loss, avg_velocity, lr, epoch_time.as_secs_f32()
        );

        monitor.log_epoch(epoch + 1, avg_loss, None, vec![("velocity", avg_velocity), ("lr", lr)]);

        if avg_loss < best_loss {
            best_loss = avg_loss;
            training_state.update_best("loss", avg_loss, false);
            // Save best model weights
            let model_path = config.output_dir.join("best_model.axonml");
            save_model(&model, &model_path).ok();
            // Save best full checkpoint
            let best_ckpt = Checkpoint::builder()
                .model_state(StateDict::from_module(&model))
                .training_state(training_state.clone())
                .epoch(epoch + 1)
                .build();
            save_checkpoint(&best_ckpt, config.output_dir.join("checkpoint_best.axonml")).ok();
            println!("  ★ New best: {:.4}", avg_loss);
        }

        // Always save latest checkpoint (for resume)
        let latest_ckpt = Checkpoint::builder()
            .model_state(StateDict::from_module(&model))
            .training_state(training_state.clone())
            .epoch(epoch + 1)
            .build();
        save_checkpoint(&latest_ckpt, config.output_dir.join("checkpoint_latest.axonml")).ok();

        // Periodic numbered checkpoint
        if (epoch + 1) % config.save_every == 0 {
            let ckpt_path = config.output_dir.join(format!("checkpoint_epoch_{:04}.axonml", epoch + 1));
            save_checkpoint(&latest_ckpt, &ckpt_path).ok();
        }

        training_state.next_epoch();
    }

    monitor.set_status("complete");
    let total_time = training_start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(" Done — {:.1}s ({:.1} min) | Best: {:.4}", total_time.as_secs_f32(), total_time.as_secs_f32() / 60.0, best_loss);
    println!("═══════════════════════════════════════════════════════════");
}
