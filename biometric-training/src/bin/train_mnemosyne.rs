//! Train Mnemosyne — Face Identity via Temporal Crystallization
//!
//! Trains the Aegis face model on LFW (Labeled Faces in the Wild). The
//! training objective is the `CrystallizationLoss`: triplet margin in cosine
//! space plus a convergence-velocity regularization that pushes the GRU
//! hidden state to stabilize after repeated face observations.
//!
//! Features:
//! - GPU acceleration via `--features cuda`
//! - Live browser training monitor (axonml::TrainingMonitor)
//! - Best/latest/epoch checkpoints with shape-based resume
//! - Batched crystallization (all triplets through one Conv2d pass per step)
//! - CLI flags for every hyperparameter
//!
//! Usage:
//!   cargo run --release --bin train_mnemosyne -p biometric-training --features cuda
//!   cargo run --release --bin train_mnemosyne -p biometric-training --features cuda -- \
//!       --epochs 30 --bs 32 --seq-len 5 --batches 100 --resume latest

use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_optim::{AdamW, Optimizer};
use axonml_serialize::{save_checkpoint, save_model, Checkpoint, StateDict, TrainingState};
use axonml_tensor::Tensor;
use axonml_vision::models::biometric::{CrystallizationLoss, MnemosyneIdentity};

use biometric_training::{
    find_checkpoint, format_count, l2_normalize_var, load_model_from_checkpoint,
    mine_identity_sequence_batches, IdentityDataset, ResumeMode,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_DATA_DIR: &str = "/opt/datasets/lfw/processed";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/biometric-training/checkpoints/mnemosyne";
const DEFAULT_EPOCHS: usize = 30;
const DEFAULT_LR: f32 = 1e-3;
const DEFAULT_WEIGHT_DECAY: f32 = 1e-4;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_SEQ_LEN: usize = 5;
const DEFAULT_BATCHES_PER_EPOCH: usize = 100;
const DEFAULT_WARMUP_EPOCHS: usize = 3;
const DEFAULT_LOG_EVERY: usize = 10;
const DEFAULT_SAVE_EVERY: usize = 5;
const DEFAULT_SEED: u64 = 42;

// =============================================================================
// CLI / config
// =============================================================================

struct Config {
    data_dir: PathBuf,
    output_dir: PathBuf,
    epochs: usize,
    lr: f32,
    weight_decay: f32,
    batch_size: usize,
    seq_len: usize,
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
            weight_decay: DEFAULT_WEIGHT_DECAY,
            batch_size: DEFAULT_BATCH_SIZE,
            seq_len: DEFAULT_SEQ_LEN,
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
                "--wd" | "--weight-decay" => { i += 1; cfg.weight_decay = args[i].parse().unwrap(); }
                "--bs" | "--batch-size" => { i += 1; cfg.batch_size = args[i].parse().unwrap(); }
                "--seq-len" => { i += 1; cfg.seq_len = args[i].parse().unwrap(); }
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
    println!(r#"Train Mnemosyne face-identity model on LFW.

Usage: train_mnemosyne [OPTIONS]

Options:
  --data-dir PATH       LFW preprocessed directory (default: /opt/datasets/lfw/processed)
  --out PATH            Checkpoint directory (default: .../checkpoints/mnemosyne)
  --epochs N            Number of epochs (default: 30)
  --lr FLOAT            Base learning rate (default: 1e-3)
  --wd FLOAT            AdamW weight decay (default: 1e-4)
  --bs N                Batch size — triplets per batch (default: 32)
  --seq-len N           Faces per crystallization sequence (default: 5)
  --batches N           Batches per epoch (default: 100)
  --warmup N            Linear warmup epochs (default: 3)
  --log-every N         Log every N batches (default: 10)
  --save-every N        Save periodic checkpoint every N epochs (default: 5)
  --seed N              RNG seed (default: 42)
  --resume MODE         Resume: none|latest|best|<path> (default: latest)
  --fresh               Equivalent to --resume none
  --help, -h            Show help"#);
}

// =============================================================================
// LR schedule
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
// Batched crystallization
// =============================================================================

/// Run one batched crystallization sequence through Mnemosyne.
/// `steps` contains `seq_len` entries, each a flat `[B*3*64*64]` Vec<f32>.
/// Returns `(final_hidden [B, hidden_dim], mean_velocity [B, 1])`.
fn crystallize_batched(
    model: &MnemosyneIdentity,
    steps: &[Vec<f32>],
    batch_size: usize,
) -> (Variable, Variable) {
    let mut hidden: Option<Variable> = None;
    let mut velocities = Vec::with_capacity(steps.len());

    for step_data in steps {
        let face_tensor =
            Tensor::from_vec(step_data.clone(), &[batch_size, 3, 64, 64]).unwrap();
        let face = Variable::new(face_tensor, false);
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
// Main
// =============================================================================

fn main() {
    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" Mnemosyne — Face Identity via Temporal Crystallization");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // ---- Load dataset ----
    println!("Loading LFW from {}...", cfg.data_dir.display());
    let dataset = IdentityDataset::load(&cfg.data_dir);
    let total_faces = dataset.total_samples();
    let usable = dataset.count_with_at_least(2);
    println!(
        "  {} identities, {} faces ({} usable for triplets)",
        dataset.num_identities(),
        format_count(total_faces),
        usable,
    );
    assert_eq!(
        dataset.channels, 3,
        "Mnemosyne expects 3-channel face images"
    );
    assert_eq!(dataset.height, 64, "Mnemosyne expects 64x64 input");
    assert_eq!(dataset.width, 64, "Mnemosyne expects 64x64 input");
    println!();

    // ---- Build model ----
    println!("Creating Mnemosyne model...");
    let model = MnemosyneIdentity::new();
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("  Parameters: {}", format_count(param_count));
    println!();

    // ---- Resume from checkpoint ----
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

    // ---- Launch browser training monitor ----
    let monitor = axonml::TrainingMonitor::new("Mnemosyne (LFW)", param_count)
        .total_epochs(cfg.epochs)
        .batch_size(cfg.batch_size)
        .launch();
    println!("Monitor: http://127.0.0.1:{}", monitor.port());
    println!();

    // ---- Optimizer + loss ----
    let mut optimizer = AdamW::new(model.parameters(), cfg.lr);
    let loss_fn = CrystallizationLoss::default();

    let triplets_per_epoch = cfg.batch_size * cfg.batches_per_epoch;
    println!("Training:");
    println!("  batch size    : {}", cfg.batch_size);
    println!("  seq_len       : {}", cfg.seq_len);
    println!("  batches/epoch : {}", cfg.batches_per_epoch);
    println!("  triplets/epoch: {}", format_count(triplets_per_epoch));
    println!("  epochs        : {} (starting at {})", cfg.epochs, start_epoch + 1);
    println!("  lr            : {} (cosine + {}-epoch warmup)", cfg.lr, cfg.warmup_epochs);
    println!("  weight decay  : {}", cfg.weight_decay);
    println!();
    println!(
        "{:>6} {:>8} {:>10} {:>10} {:>10}",
        "Epoch", "Batch", "Loss", "Vel", "Time"
    );
    println!("{}", "-".repeat(50));

    let mut best_loss = training_state.best_metric.unwrap_or(f32::MAX);
    let mut rng = cfg.seed;
    let global_start = Instant::now();

    for epoch in (start_epoch + 1)..=cfg.epochs {
        let epoch_start = Instant::now();
        let lr = cosine_lr(cfg.lr, cfg.warmup_epochs, cfg.epochs, epoch - 1);
        optimizer.set_lr(lr);

        let mut epoch_loss_sum = 0.0f32;
        let mut epoch_vel_sum = 0.0f32;
        let mut epoch_count = 0usize;
        let mut running_loss = 0.0f32;
        let mut running_vel = 0.0f32;
        let mut running_count = 0usize;

        for batch_idx in 1..=cfg.batches_per_epoch {
            // Mine one batched triplet sequence
            let steps = mine_identity_sequence_batches(
                &dataset,
                cfg.batch_size,
                cfg.seq_len,
                &mut rng,
            );

            // Split into per-step anchor/positive/negative buffers
            let anchor_steps: Vec<Vec<f32>> =
                steps.iter().map(|(a, _, _)| a.clone()).collect();
            let pos_steps: Vec<Vec<f32>> =
                steps.iter().map(|(_, p, _)| p.clone()).collect();
            let neg_steps: Vec<Vec<f32>> =
                steps.iter().map(|(_, _, n)| n.clone()).collect();

            // Crystallize each triplet branch
            let (anchor_h, anchor_vel) =
                crystallize_batched(&model, &anchor_steps, cfg.batch_size);
            let (pos_h, _) = crystallize_batched(&model, &pos_steps, cfg.batch_size);
            let (neg_h, _) = crystallize_batched(&model, &neg_steps, cfg.batch_size);

            // L2 normalize embeddings
            let anchor_emb = l2_normalize_var(&anchor_h);
            let pos_emb = l2_normalize_var(&pos_h);
            let neg_emb = l2_normalize_var(&neg_h);

            // Crystallization loss = triplet + convergence regularization
            let loss =
                loss_fn.compute_var(&anchor_emb, &pos_emb, &neg_emb, &anchor_vel);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            let loss_val = loss.data().to_vec()[0];
            let vel_val = anchor_vel.data().to_vec()[0];

            epoch_loss_sum += loss_val;
            epoch_vel_sum += vel_val;
            epoch_count += 1;
            running_loss += loss_val;
            running_vel += vel_val;
            running_count += 1;
            training_state.next_step();
            training_state.record_loss(loss_val);

            if batch_idx % cfg.log_every == 0 {
                let avg_loss = running_loss / running_count as f32;
                let avg_vel = running_vel / running_count as f32;
                let elapsed = global_start.elapsed().as_secs_f32();
                println!(
                    "{:>6} {:>8} {:>10.4} {:>10.4} {:>9.1}s",
                    format!("{}/{}", epoch, cfg.epochs),
                    batch_idx,
                    avg_loss,
                    avg_vel,
                    elapsed,
                );
                running_loss = 0.0;
                running_vel = 0.0;
                running_count = 0;
            }
        }

        // ---- End-of-epoch bookkeeping ----
        let epoch_avg = epoch_loss_sum / epoch_count.max(1) as f32;
        let avg_vel = epoch_vel_sum / epoch_count.max(1) as f32;
        let epoch_time = epoch_start.elapsed();

        monitor.log_epoch(
            epoch,
            epoch_avg,
            None,
            vec![("velocity", avg_vel), ("lr", lr)],
        );

        // Save best
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

        // Always save latest
        let latest_ckpt = cfg.output_dir.join("checkpoint_latest.axonml");
        let cp = Checkpoint::builder()
            .model_state(StateDict::from_module(&model))
            .training_state(training_state.clone())
            .epoch(epoch)
            .build();
        if let Err(e) = save_checkpoint(&cp, &latest_ckpt) {
            eprintln!("  Error saving latest checkpoint: {e}");
        }

        // Periodic epoch checkpoint
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
            "  epoch {} done in {:.1}s | loss {:.4} | vel {:.4}",
            epoch,
            epoch_time.as_secs_f32(),
            epoch_avg,
            avg_vel,
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
