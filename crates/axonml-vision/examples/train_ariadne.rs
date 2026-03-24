//! Train Ariadne — Fingerprint Identity via Ridge Event Fields
//!
//! Trains on FVC2000 DB4_B fingerprint dataset.
//! Uses ContrastiveLoss (margin-based with orientation regularization).
//!
//! ```bash
//! cargo run --example train_ariadne --release -p axonml-vision --features cuda
//! ```

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_core::Device;
use axonml_optim::{AdamW, Optimizer};
use axonml_serialize::{
    Checkpoint, StateDict, TrainingState, load_checkpoint, save_checkpoint, save_model,
};
use axonml_tensor::Tensor;

use axonml_vision::models::biometric::{AriadneFingerprint, ContrastiveLoss};

// =============================================================================
// Dataset
// =============================================================================

const IMG_SIZE: usize = 128 * 128; // 1 channel

struct IdentityData {
    images: Vec<Vec<f32>>,
}

fn load_fingerprint_identities(data_dir: &Path) -> Vec<IdentityData> {
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

        let img_size = c * h * w;
        let mut byte_buf = vec![0u8; num * img_size * 4];
        file.read_exact(&mut byte_buf).unwrap();

        let all_data: Vec<f32> = byte_buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let mut images = Vec::with_capacity(num);
        for i in 0..num {
            images.push(all_data[i * img_size..(i + 1) * img_size].to_vec());
        }
        identities.push(IdentityData { images });
    }
    identities
}

// =============================================================================
// Batched Pair Mining
// =============================================================================

/// Mine a batch of pairs. Returns (batch_a, batch_b, is_same per pair).
fn mine_pair_batch(
    identities: &[IdentityData],
    rng: &mut u64,
    batch_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
    let mut data_a = Vec::with_capacity(batch_size * IMG_SIZE);
    let mut data_b = Vec::with_capacity(batch_size * IMG_SIZE);
    let mut labels = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let is_same = i % 2 == 0;
        labels.push(is_same);

        if is_same {
            let idx = lcg_range(rng, identities.len());
            let id = &identities[idx];
            let a = lcg_range(rng, id.images.len());
            let mut b = lcg_range(rng, id.images.len());
            if id.images.len() > 1 {
                while b == a {
                    b = lcg_range(rng, id.images.len());
                }
            }
            data_a.extend_from_slice(&id.images[a]);
            data_b.extend_from_slice(&id.images[b]);
        } else {
            let idx_a = lcg_range(rng, identities.len());
            let mut idx_b = lcg_range(rng, identities.len());
            while idx_b == idx_a {
                idx_b = lcg_range(rng, identities.len());
            }
            let a = lcg_range(rng, identities[idx_a].images.len());
            let b = lcg_range(rng, identities[idx_b].images.len());
            data_a.extend_from_slice(&identities[idx_a].images[a]);
            data_b.extend_from_slice(&identities[idx_b].images[b]);
        }
    }

    (data_a, data_b, labels)
}

fn lcg_range(state: &mut u64, max: usize) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as usize) % max
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
    batches_per_epoch: usize,
    save_every: usize,
    warmup_epochs: usize,
    resume: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("/opt/datasets/fingerprint/processed"),
            output_dir: PathBuf::from("/opt/AxonML/checkpoints/ariadne"),
            epochs: 50,
            lr: 1e-3,
            batch_size: 16,
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
                "--data-dir" => {
                    i += 1;
                    config.data_dir = PathBuf::from(&args[i]);
                }
                "--output-dir" => {
                    i += 1;
                    config.output_dir = PathBuf::from(&args[i]);
                }
                "--epochs" => {
                    i += 1;
                    config.epochs = args[i].parse().unwrap();
                }
                "--lr" => {
                    i += 1;
                    config.lr = args[i].parse().unwrap();
                }
                "--bs" | "--batch-size" => {
                    i += 1;
                    config.batch_size = args[i].parse().unwrap();
                }
                "--batches" => {
                    i += 1;
                    config.batches_per_epoch = args[i].parse().unwrap();
                }
                "--resume" => {
                    i += 1;
                    config.resume = args[i].clone();
                }
                "--fresh" => {
                    config.resume = "none".to_string();
                }
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
            if p.exists() {
                Some(p)
            } else {
                let b = output_dir.join("checkpoint_best.axonml");
                if b.exists() {
                    Some(b)
                } else {
                    let m = output_dir.join("best_model.axonml");
                    if m.exists() { Some(m) } else { None }
                }
            }
        }
        "best" => {
            let p = output_dir.join("checkpoint_best.axonml");
            if p.exists() {
                Some(p)
            } else {
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

fn load_model_weights(model: &AriadneFingerprint, path: &Path) -> (usize, TrainingState) {
    // Ariadne uses positional parameter loading (no named_parameters impl)
    if let Ok(checkpoint) = load_checkpoint(path) {
        let params = model.parameters();
        let state_entries: Vec<_> = checkpoint.model_state.entries().collect();
        let mut loaded = 0usize;
        for ((_name, entry), param) in state_entries.iter().zip(params.iter()) {
            if let Ok(tensor) = entry.data.to_tensor() {
                if tensor.shape() == param.data().shape() {
                    param.update_data(tensor);
                    loaded += 1;
                }
            }
        }
        println!(
            "  Loaded checkpoint: {} (epoch {}, {}/{} params)",
            path.display(),
            checkpoint.epoch(),
            loaded,
            params.len()
        );
        return (checkpoint.epoch(), checkpoint.training_state.clone());
    }

    if let Ok(state_dict) = axonml_serialize::load_state_dict(path) {
        let params = model.parameters();
        let state_entries: Vec<_> = state_dict.entries().collect();
        let mut loaded = 0usize;
        for ((_name, entry), param) in state_entries.iter().zip(params.iter()) {
            if let Ok(tensor) = entry.data.to_tensor() {
                if tensor.shape() == param.data().shape() {
                    param.update_data(tensor);
                    loaded += 1;
                }
            }
        }
        println!(
            "  Loaded model weights: {} ({}/{} params)",
            path.display(),
            loaded,
            params.len()
        );
        return (0, TrainingState::new());
    }

    println!("  WARNING: Failed to load from {}", path.display());
    (0, TrainingState::new())
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let config = TrainConfig::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" Ariadne — Fingerprint Identity via Ridge Event Fields");
    println!(" Training on FVC2000 DB4_B");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Detect GPU
    #[cfg(feature = "cuda")]
    let device = match Device::cuda(0) {
        d @ Device::Cuda(_) => {
            println!("  Using GPU: {:?}", d);
            d
        }
        _ => {
            println!("  GPU not available, using CPU");
            Device::Cpu
        }
    };
    #[cfg(not(feature = "cuda"))]
    let device = {
        println!("  Using CPU (build with --features cuda for GPU)");
        Device::Cpu
    };

    println!(
        "Loading fingerprint identities from {}...",
        config.data_dir.display()
    );
    let identities = load_fingerprint_identities(&config.data_dir);
    let total: usize = identities.iter().map(|id| id.images.len()).sum();
    println!(
        "  {} identities, {} fingerprint images",
        identities.len(),
        total
    );

    println!("\nCreating Ariadne model...");
    let model = AriadneFingerprint::new();
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("  Parameters: {}", param_count);

    // Move model to GPU
    if device.is_gpu() {
        for param in model.parameters() {
            param.to_device(device.clone());
        }
        println!("  Model moved to GPU");
    }

    let pairs_per_epoch = config.batch_size * config.batches_per_epoch;
    let monitor =
        axonml::monitor::TrainingMonitor::new("Ariadne — Fingerprint Identity", param_count)
            .total_epochs(config.epochs)
            .batch_size(pairs_per_epoch)
            .launch();

    let mut optimizer = AdamW::new(model.parameters(), config.lr);
    let loss_fn = ContrastiveLoss::default();

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
            if device.is_gpu() {
                for param in model.parameters() {
                    param.to_device(device.clone());
                }
            }
        } else {
            println!("  No checkpoint found, starting fresh.");
        }
    }

    let mut rng_state = 13u64 + start_epoch as u64 * 1000;

    println!(
        "\nTraining: epochs {}-{}, batch={}, {}/epoch ({} batches)",
        start_epoch + 1,
        config.epochs,
        config.batch_size,
        pairs_per_epoch,
        config.batches_per_epoch
    );
    println!("  Best so far: {:.4}", best_loss);
    println!();

    let training_start = Instant::now();

    for epoch in start_epoch..config.epochs {
        let epoch_start = Instant::now();
        let lr = cosine_lr(config.lr, config.warmup_epochs, config.epochs, epoch);
        optimizer.set_lr(lr);

        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0;

        for _ in 0..config.batches_per_epoch {
            let (data_a, data_b, labels) =
                mine_pair_batch(&identities, &mut rng_state, config.batch_size);

            // Create tensors and move to GPU
            let mut t_a = Tensor::from_vec(data_a, &[config.batch_size, 1, 128, 128]).unwrap();
            let mut t_b = Tensor::from_vec(data_b, &[config.batch_size, 1, 128, 128]).unwrap();
            if device.is_gpu() {
                t_a = t_a.to_device(device.clone()).unwrap();
                t_b = t_b.to_device(device.clone()).unwrap();
            }

            let var_a = Variable::new(t_a, false);
            let var_b = Variable::new(t_b, false);

            // Batched forward
            let (emb_a, _lv_a) = model.forward_full(&var_a);
            let (emb_b, _lv_b) = model.forward_full(&var_b);

            // Contrastive loss — average over same/different pairs in batch
            // For now, use first pair's label (alternating same/different)
            // TODO: per-sample contrastive loss with mixed labels
            let is_same = labels[0];
            let loss = loss_fn.compute_var(&emb_a, &emb_b, is_same);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            let loss_val = loss.data().to_vec()[0];
            epoch_loss += loss_val;
            num_batches += 1;
            training_state.next_step();
            training_state.record_loss(loss_val);
        }

        let avg_loss = epoch_loss / num_batches as f32;
        let epoch_time = epoch_start.elapsed();

        println!(
            "Epoch {:3}/{} | loss: {:.4} | lr: {:.6} | {:.1}s",
            epoch + 1,
            config.epochs,
            avg_loss,
            lr,
            epoch_time.as_secs_f32()
        );

        monitor.log_epoch(epoch + 1, avg_loss, None, vec![("lr", lr)]);

        if avg_loss < best_loss {
            best_loss = avg_loss;
            training_state.update_best("loss", avg_loss, false);
            save_model(&model, config.output_dir.join("best_model.axonml")).ok();
            let best_ckpt = Checkpoint::builder()
                .model_state(StateDict::from_module(&model))
                .training_state(training_state.clone())
                .epoch(epoch + 1)
                .build();
            save_checkpoint(&best_ckpt, config.output_dir.join("checkpoint_best.axonml")).ok();
            println!("  ★ New best: {:.4}", avg_loss);
        }

        let latest_ckpt = Checkpoint::builder()
            .model_state(StateDict::from_module(&model))
            .training_state(training_state.clone())
            .epoch(epoch + 1)
            .build();
        save_checkpoint(
            &latest_ckpt,
            config.output_dir.join("checkpoint_latest.axonml"),
        )
        .ok();

        if (epoch + 1) % config.save_every == 0 {
            save_checkpoint(
                &latest_ckpt,
                config
                    .output_dir
                    .join(format!("checkpoint_epoch_{:04}.axonml", epoch + 1)),
            )
            .ok();
        }

        training_state.next_epoch();
    }

    monitor.set_status("complete");

    let total_time = training_start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(
        " Done — {:.1}s ({:.1} min) | Best: {:.4}",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / 60.0,
        best_loss
    );
    println!("═══════════════════════════════════════════════════════════");
}
