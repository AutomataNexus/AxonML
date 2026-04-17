//! Train Argus — Iris Identity via Radial Phase Encoding
//!
//! End-to-end training binary for the AxonML [`ArgusIris`] iris-identity
//! model on the pre-computed CASIA-Iris-Syn polar cache. Polar unwrap is an
//! expensive Cartesian→polar transform, so we cache it once at
//! `/opt/datasets/iris/polar_cache/` as `[1, 32, 256]` strips and feed them
//! straight into `ArgusIris::encode_polar` — roughly 5× faster than
//! `forward_full`. The trainer asserts the cache shape
//! (channels=1, height=32 radial bins, width=256 angular bins) at startup.
//!
//! ## What this file contains
//! - `Config` struct + `Config::from_args` CLI parser and `print_help`,
//!   covering data dir, output dir, triplet batch size, batches per epoch,
//!   learning-rate + warmup, checkpoint cadence, seed, and resume mode.
//! - `cosine_lr` — linear warmup then cosine decay to 1% of `base_lr`.
//! - `main` — asserts cache shape, builds the [`ArgusIris`] model, resumes
//!   from a checkpoint, launches the browser [`axonml::TrainingMonitor`],
//!   wires up [`AdamW`] + [`ArgusLoss`], and runs the triplet training loop
//!   (mining with [`mine_triplet_batch`], encoding anchors/positives/
//!   negatives via `encode_polar`, using [`l2_normalize_var`] on the
//!   anchor/positive pair as rotation-consistency codes, and computing
//!   `ArgusLoss::compute_var`). Saves `best_model.axonml`,
//!   `checkpoint_best.axonml`, `checkpoint_latest.axonml`, and periodic
//!   `checkpoint_epoch_NNNN.axonml` files.
//!
//! Usage:
//!   cargo run --release --bin train_argus -p biometric-training --features cuda
//!   cargo run --release --bin train_argus -p biometric-training --features cuda -- \
//!       --epochs 30 --bs 32 --batches 150 --resume latest
//!
//! # File
//! `biometric-training/src/bin/train_argus.rs`
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

use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_optim::{AdamW, Optimizer};
use axonml_serialize::{save_checkpoint, save_model, Checkpoint, StateDict, TrainingState};
use axonml_tensor::Tensor;
use axonml_vision::models::biometric::{ArgusIris, ArgusLoss};

use biometric_training::{
    find_checkpoint, format_count, l2_normalize_var, load_model_from_checkpoint, mine_triplet_batch,
    IdentityDataset, ResumeMode,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_DATA_DIR: &str = "/opt/datasets/iris/polar_cache";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/biometric-training/checkpoints/argus";
const DEFAULT_EPOCHS: usize = 30;
const DEFAULT_LR: f32 = 1e-3;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_BATCHES_PER_EPOCH: usize = 150;
const DEFAULT_WARMUP_EPOCHS: usize = 3;
const DEFAULT_LOG_EVERY: usize = 15;
const DEFAULT_SAVE_EVERY: usize = 5;
const DEFAULT_SEED: u64 = 7;

// =============================================================================
// CLI / config
// =============================================================================

struct Config {
    data_dir: PathBuf,
    output_dir: PathBuf,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    batches_per_epoch: usize,
    warmup_epochs: usize,
    log_every: usize,
    save_every: usize,
    seed: u64,
    resume: ResumeMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from(DEFAULT_DATA_DIR),
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            epochs: DEFAULT_EPOCHS,
            lr: DEFAULT_LR,
            batch_size: DEFAULT_BATCH_SIZE,
            batches_per_epoch: DEFAULT_BATCHES_PER_EPOCH,
            warmup_epochs: DEFAULT_WARMUP_EPOCHS,
            log_every: DEFAULT_LOG_EVERY,
            save_every: DEFAULT_SAVE_EVERY,
            seed: DEFAULT_SEED,
            resume: ResumeMode::Latest,
        }
    }
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut cfg = Self::default();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--data-dir" => { i += 1; cfg.data_dir = PathBuf::from(&args[i]); }
                "--out" | "--output-dir" => { i += 1; cfg.output_dir = PathBuf::from(&args[i]); }
                "--epochs" => { i += 1; cfg.epochs = args[i].parse().unwrap(); }
                "--lr" => { i += 1; cfg.lr = args[i].parse().unwrap(); }
                "--bs" | "--batch-size" => { i += 1; cfg.batch_size = args[i].parse().unwrap(); }
                "--batches" => { i += 1; cfg.batches_per_epoch = args[i].parse().unwrap(); }
                "--warmup" => { i += 1; cfg.warmup_epochs = args[i].parse().unwrap(); }
                "--log-every" => { i += 1; cfg.log_every = args[i].parse().unwrap(); }
                "--save-every" => { i += 1; cfg.save_every = args[i].parse().unwrap(); }
                "--seed" => { i += 1; cfg.seed = args[i].parse().unwrap(); }
                "--resume" => { i += 1; cfg.resume = ResumeMode::from_str(&args[i]); }
                "--fresh" => { cfg.resume = ResumeMode::None; }
                "--help" | "-h" => { print_help(); std::process::exit(0); }
                other => {
                    eprintln!("Unknown argument: {other}");
                    print_help();
                    std::process::exit(1);
                }
            }
            i += 1;
        }
        cfg
    }
}

fn print_help() {
    println!(r#"Train Argus iris-identity model on CASIA-Iris-Syn polar cache.

Usage: train_argus [OPTIONS]

Options:
  --data-dir PATH       Polar-cache dir (default: /opt/datasets/iris/polar_cache)
  --out PATH            Checkpoint dir (default: .../checkpoints/argus)
  --epochs N            Number of epochs (default: 30)
  --lr FLOAT            Base learning rate (default: 1e-3)
  --bs N                Triplets per batch (default: 32)
  --batches N           Batches per epoch (default: 150)
  --warmup N            Linear warmup epochs (default: 3)
  --log-every N         Log every N batches (default: 15)
  --save-every N        Save periodic checkpoint every N epochs (default: 5)
  --seed N              RNG seed (default: 7)
  --resume MODE         Resume: none|latest|best|<path> (default: latest)
  --fresh               Equivalent to --resume none
  --help, -h            Show help"#);
}

// =============================================================================
// LR Schedule
// =============================================================================

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
// Main Entry Point
// =============================================================================

fn main() {
    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" Argus — Iris Identity via Radial Phase Encoding");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // ---- Load polar cache ----
    println!("Loading polar cache from {}...", cfg.data_dir.display());
    let dataset = IdentityDataset::load(&cfg.data_dir);
    println!(
        "  {} identities, {} iris samples ({}x{}x{} per strip)",
        dataset.num_identities(),
        format_count(dataset.total_samples()),
        dataset.channels,
        dataset.height,
        dataset.width,
    );
    assert_eq!(dataset.channels, 1, "Argus polar cache expects 1 channel");
    assert_eq!(dataset.height, 32, "Argus expects 32 radial bins");
    assert_eq!(dataset.width, 256, "Argus expects 256 angular bins");
    println!();

    // ---- Model ----
    println!("Creating Argus model...");
    let model = ArgusIris::new();
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("  Parameters: {}", format_count(param_count));
    println!("  Embedding : 128-dim");
    println!();

    // ---- Resume ----
    std::fs::create_dir_all(&cfg.output_dir).expect("Failed to create output dir");
    let mut training_state = TrainingState::new();
    let mut start_epoch = 0usize;
    if let Some(ckpt_path) = find_checkpoint(&cfg.output_dir, &cfg.resume) {
        match load_model_from_checkpoint(&model, &ckpt_path) {
            Ok((epoch, state)) => {
                start_epoch = epoch;
                training_state = state;
                println!("Resuming from epoch {}", start_epoch);
            }
            Err(e) => eprintln!("Resume failed: {e} — starting fresh"),
        }
    } else {
        println!("Starting fresh training run");
    }

    // ---- Monitor ----
    let monitor =
        axonml::TrainingMonitor::new("Argus (CASIA-Iris-Syn)", param_count)
            .total_epochs(cfg.epochs)
            .batch_size(cfg.batch_size)
            .launch();
    println!("Monitor: http://127.0.0.1:{}", monitor.port());
    println!();

    // ---- Optimizer + loss ----
    let mut optimizer = AdamW::new(model.parameters(), cfg.lr);
    let loss_fn = ArgusLoss::default();

    let triplets_per_epoch = cfg.batch_size * cfg.batches_per_epoch;
    println!("Training:");
    println!("  batch size    : {}", cfg.batch_size);
    println!("  batches/epoch : {}", cfg.batches_per_epoch);
    println!("  triplets/epoch: {}", format_count(triplets_per_epoch));
    println!(
        "  epochs        : {} (starting at {})",
        cfg.epochs,
        start_epoch + 1
    );
    println!(
        "  lr            : {} (cosine + {}-epoch warmup)",
        cfg.lr, cfg.warmup_epochs
    );
    println!();
    println!("{:>6} {:>8} {:>10} {:>10}", "Epoch", "Batch", "Loss", "Time");
    println!("{}", "-".repeat(40));

    let mut best_loss = training_state.best_metric.unwrap_or(f32::MAX);
    let mut rng = cfg.seed;
    let global_start = Instant::now();

    for epoch in (start_epoch + 1)..=cfg.epochs {
        let epoch_start = Instant::now();
        let lr = cosine_lr(cfg.lr, cfg.warmup_epochs, cfg.epochs, epoch - 1);
        optimizer.set_lr(lr);

        let mut epoch_loss_sum = 0.0f32;
        let mut epoch_count = 0usize;
        let mut running_loss = 0.0f32;
        let mut running_count = 0usize;

        for batch_idx in 1..=cfg.batches_per_epoch {
            // Mine triplet batch of polar strips
            let (anchor_data, pos_data, neg_data) =
                mine_triplet_batch(&dataset, cfg.batch_size, &mut rng);

            let anchor_var = Variable::new(
                Tensor::from_vec(anchor_data, &[cfg.batch_size, 1, 32, 256]).unwrap(),
                false,
            );
            let pos_var = Variable::new(
                Tensor::from_vec(pos_data, &[cfg.batch_size, 1, 32, 256]).unwrap(),
                false,
            );
            let neg_var = Variable::new(
                Tensor::from_vec(neg_data, &[cfg.batch_size, 1, 32, 256]).unwrap(),
                false,
            );

            // Encode polar strips directly (skips the expensive Cartesian unwrap)
            let (anchor_emb, _alv) = model.encode_polar(&anchor_var);
            let (pos_emb, _plv) = model.encode_polar(&pos_var);
            let (neg_emb, _nlv) = model.encode_polar(&neg_var);

            // Phase consistency: original (anchor) and "rotated" (positive —
            // a different capture of the same eye simulates rotation invariance)
            let code_orig = l2_normalize_var(&anchor_emb);
            let code_rot = l2_normalize_var(&pos_emb);

            let loss = loss_fn.compute_var(
                &anchor_emb,
                &pos_emb,
                &neg_emb,
                &code_orig,
                &code_rot,
            );

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            let loss_val = loss.data().to_vec()[0];
            epoch_loss_sum += loss_val;
            epoch_count += 1;
            running_loss += loss_val;
            running_count += 1;
            training_state.next_step();
            training_state.record_loss(loss_val);

            if batch_idx % cfg.log_every == 0 {
                let avg = running_loss / running_count as f32;
                let elapsed = global_start.elapsed().as_secs_f32();
                println!(
                    "{:>6} {:>8} {:>10.4} {:>9.1}s",
                    format!("{}/{}", epoch, cfg.epochs),
                    batch_idx,
                    avg,
                    elapsed,
                );
                running_loss = 0.0;
                running_count = 0;
            }
        }

        // ---- End-of-epoch ----
        let epoch_avg = epoch_loss_sum / epoch_count.max(1) as f32;
        let epoch_time = epoch_start.elapsed();

        monitor.log_epoch(epoch, epoch_avg, None, vec![("lr", lr)]);

        if epoch_avg < best_loss {
            best_loss = epoch_avg;
            training_state.update_best("loss", epoch_avg, false);
            let best_path = cfg.output_dir.join("best_model.axonml");
            if let Err(e) = save_model(&model, &best_path) {
                eprintln!("  Error saving best model: {e}");
            } else {
                println!("  ★ new best loss {:.4} → {}", epoch_avg, best_path.display());
            }
            let best_ckpt = cfg.output_dir.join("checkpoint_best.axonml");
            let cp = Checkpoint::builder()
                .model_state(StateDict::from_module(&model))
                .training_state(training_state.clone())
                .epoch(epoch)
                .build();
            save_checkpoint(&cp, &best_ckpt).ok();
        }

        let latest_ckpt = cfg.output_dir.join("checkpoint_latest.axonml");
        let cp = Checkpoint::builder()
            .model_state(StateDict::from_module(&model))
            .training_state(training_state.clone())
            .epoch(epoch)
            .build();
        save_checkpoint(&cp, &latest_ckpt).ok();

        if epoch % cfg.save_every == 0 {
            let epoch_ckpt = cfg
                .output_dir
                .join(format!("checkpoint_epoch_{epoch:04}.axonml"));
            let cp = Checkpoint::builder()
                .model_state(StateDict::from_module(&model))
                .training_state(training_state.clone())
                .epoch(epoch)
                .build();
            save_checkpoint(&cp, &epoch_ckpt).ok();
        }

        println!(
            "  epoch {} done in {:.1}s | loss {:.4}",
            epoch,
            epoch_time.as_secs_f32(),
            epoch_avg,
        );
        training_state.next_epoch();
    }

    monitor.set_status("complete");
    let total_time = global_start.elapsed();

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(" Training Complete");
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  time      : {:.1}s ({:.1} min)",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / 60.0,
    );
    println!("  best loss : {:.4}", best_loss);
    println!("  output    : {}", cfg.output_dir.display());
}
