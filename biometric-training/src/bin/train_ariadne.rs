//! Train Ariadne — Fingerprint Identity via Ridge Event Fields
//!
//! Trains on the FVC2000 DB4_B preprocessed dataset with ContrastiveLoss
//! (margin-based: same-identity → minimize distance, different → push
//! beyond margin). Input is 128x128 grayscale fingerprint images; the
//! model's 8 learnable Gabor orientation filters extract ridge event
//! fields that feed depthwise separable convolution blocks.
//!
//! Note: FVC2000 DB4_B only has 10 identities × 80 samples. That's small —
//! useful for smoke-testing the architecture and pipeline but not enough
//! for a competitive model. See the Aegis paper's limitations section.
//!
//! Usage:
//!   cargo run --release --bin train_ariadne -p biometric-training --features cuda
//!   cargo run --release --bin train_ariadne -p biometric-training --features cuda -- \
//!       --epochs 30 --bs 16 --batches 150 --resume latest

use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_optim::{AdamW, Optimizer};
use axonml_serialize::{save_checkpoint, save_model, Checkpoint, StateDict, TrainingState};
use axonml_tensor::Tensor;
use axonml_vision::models::biometric::{AriadneFingerprint, ContrastiveLoss};

use biometric_training::{
    find_checkpoint, format_count, load_model_from_checkpoint, mine_pair_batch, IdentityDataset,
    ResumeMode,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_DATA_DIR: &str = "/opt/datasets/fingerprint/processed";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/biometric-training/checkpoints/ariadne";
const DEFAULT_EPOCHS: usize = 30;
const DEFAULT_LR: f32 = 1e-3;
const DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_BATCHES_PER_EPOCH: usize = 150;
const DEFAULT_WARMUP_EPOCHS: usize = 3;
const DEFAULT_LOG_EVERY: usize = 15;
const DEFAULT_SAVE_EVERY: usize = 5;
const DEFAULT_SEED: u64 = 13;

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
    println!(r#"Train Ariadne fingerprint-identity model on FVC2000 DB4_B.

Usage: train_ariadne [OPTIONS]

Options:
  --data-dir PATH       Dataset dir (default: /opt/datasets/fingerprint/processed)
  --out PATH            Checkpoint dir (default: .../checkpoints/ariadne)
  --epochs N            Number of epochs (default: 30)
  --lr FLOAT            Base learning rate (default: 1e-3)
  --bs N                Pairs per batch (default: 16)
  --batches N           Batches per epoch (default: 150)
  --warmup N            Linear warmup epochs (default: 3)
  --log-every N         Log every N batches (default: 15)
  --save-every N        Save periodic checkpoint every N epochs (default: 5)
  --seed N              RNG seed (default: 13)
  --resume MODE         Resume: none|latest|best|<path> (default: latest)
  --fresh               Equivalent to --resume none
  --help, -h            Show help"#);
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

fn main() {
    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" Ariadne — Fingerprint Identity via Ridge Event Fields");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // ---- Load dataset ----
    println!("Loading fingerprints from {}...", cfg.data_dir.display());
    let dataset = IdentityDataset::load(&cfg.data_dir);
    println!(
        "  {} identities, {} fingerprint images ({}x{}x{})",
        dataset.num_identities(),
        format_count(dataset.total_samples()),
        dataset.channels,
        dataset.height,
        dataset.width,
    );
    assert_eq!(dataset.channels, 1, "Ariadne expects 1-channel grayscale fingerprints");
    assert_eq!(dataset.height, 128, "Ariadne expects 128x128 input");
    assert_eq!(dataset.width, 128, "Ariadne expects 128x128 input");
    println!();

    // ---- Model ----
    println!("Creating Ariadne model...");
    let model = AriadneFingerprint::new();
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("  Parameters: {}", format_count(param_count));
    println!("  Embedding : 128-dim (8 Gabor orientations)");
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
        axonml::TrainingMonitor::new("Ariadne (FVC2000)", param_count)
            .total_epochs(cfg.epochs)
            .batch_size(cfg.batch_size)
            .launch();
    println!("Monitor: http://127.0.0.1:{}", monitor.port());
    println!();

    // ---- Optimizer + loss ----
    let mut optimizer = AdamW::new(model.parameters(), cfg.lr);
    let loss_fn = ContrastiveLoss::default();

    let pairs_per_epoch = cfg.batch_size * cfg.batches_per_epoch;
    println!("Training:");
    println!("  batch size    : {}", cfg.batch_size);
    println!("  batches/epoch : {}", cfg.batches_per_epoch);
    println!("  pairs/epoch   : {}", format_count(pairs_per_epoch));
    println!("  epochs        : {} (starting at {})", cfg.epochs, start_epoch + 1);
    println!("  lr            : {} (cosine + {}-epoch warmup)", cfg.lr, cfg.warmup_epochs);
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
            // Mine a pair batch: half same-identity, half different
            let (a_data, b_data, labels) = mine_pair_batch(&dataset, cfg.batch_size, &mut rng);

            let var_a = Variable::new(
                Tensor::from_vec(a_data, &[cfg.batch_size, 1, 128, 128]).unwrap(),
                false,
            );
            let var_b = Variable::new(
                Tensor::from_vec(b_data, &[cfg.batch_size, 1, 128, 128]).unwrap(),
                false,
            );

            // Batched forward
            let (emb_a, _alv) = model.forward_full(&var_a);
            let (emb_b, _blv) = model.forward_full(&var_b);

            // ContrastiveLoss operates per-pair; accumulate across the batch
            let mut loss_sum: Option<Variable> = None;
            for (i, &is_same) in labels.iter().enumerate() {
                // Narrow to the i-th row
                let row_a = emb_a.narrow(0, i, 1);
                let row_b = emb_b.narrow(0, i, 1);
                let pair_loss = loss_fn.compute_var(&row_a, &row_b, is_same);
                loss_sum = Some(match loss_sum {
                    None => pair_loss,
                    Some(acc) => acc.add_var(&pair_loss),
                });
            }
            let loss = loss_sum.unwrap().mul_scalar(1.0 / labels.len() as f32);

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
