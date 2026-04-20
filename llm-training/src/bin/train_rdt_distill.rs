//! train_rdt_distill — distill an RDT student from an Oracle teacher GGUF.
//!
//! Combines two prior trainers:
//! * `train_rdt.rs` — RDT student with K-sampling per batch (U{k_min,k_max}).
//! * `train_draft_distill.rs` — frozen-teacher KL distillation with NoGradGuard.
//!
//! Loss per step (RDT_DESIGN §8a):
//!
//! ```text
//!   L = α · CE(student_logits@K, next_token_labels)
//!     + (1 − α) · KL(student_logits@K, teacher_logits, T)
//! ```
//!
//! α = 0.1, T = 3 by default — the student gets 90% of its signal from
//! matching the teacher's soft distribution at each position and 10% from
//! hard next-token CE. Teacher is frozen under `NoGradGuard`; only the
//! student's graph carries gradients.
//!
//! The student's vocab_size is force-matched to the teacher's so the KL
//! head has shape-aligned logit distributions — passing `--teacher-gguf
//! /opt/AxonML/models/oracle-distill/oracle-r1-distill-q4km.gguf` makes
//! the student adopt the R1-Distill BPE vocab (≈152064 tokens). The
//! dataset passed via `--tokens-bin` must have been tokenized with the
//! same tokenizer, or KL is meaningless.
//!
//! # File
//! `llm-training/src/bin/train_rdt_distill.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_autograd::no_grad::NoGradGuard;
use axonml_core::Device;
use axonml_llm::{RDTConfig, RDTForCausalLM, load_qwen3_from_gguf};
use axonml_nn::{KLDivLoss, Module, Reduction};
use axonml_optim::{AdamW, Optimizer};
use axonml_serialize::TrainingState;
use axonml_tensor::Tensor;

use llm_training::{
    LoopAction, ResumeMode, TextDataset, TrainingLifecycle, find_checkpoint, format_count,
    lcg_range, load_model_from_checkpoint, shifted_cross_entropy,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/llm-training/checkpoints/rdt_distill";
const DEFAULT_ARCH: &str = "small";
const DEFAULT_SEQ_LEN: usize = 512;
const DEFAULT_BATCH_SIZE: usize = 2;
const DEFAULT_EPOCHS: usize = 3;
const DEFAULT_STEPS_PER_EPOCH: usize = 500;
const DEFAULT_LR: f32 = 3e-4;
const DEFAULT_WARMUP_STEPS: usize = 100;
const DEFAULT_TEMPERATURE: f32 = 3.0;
const DEFAULT_CE_WEIGHT: f32 = 0.1;
const DEFAULT_WEIGHT_DECAY: f32 = 0.1;
const DEFAULT_GRAD_CLIP: f32 = 1.0;
const DEFAULT_CHECKPOINT_EVERY: u64 = 50;
const DEFAULT_KEEP_LAST_K: usize = 3;
const DEFAULT_LOG_EVERY: usize = 10;

// =============================================================================
// CLI
// =============================================================================

struct Config {
    tokens_bin: PathBuf,
    teacher_gguf: PathBuf,
    output_dir: PathBuf,
    arch: String,
    seq_len: usize,
    batch_size: usize,
    epochs: usize,
    steps_per_epoch: usize,
    lr: f32,
    warmup_steps: usize,
    temperature: f32,
    ce_weight: f32,
    weight_decay: f32,
    grad_clip: f32,
    k_min: Option<usize>,
    k_max: Option<usize>,
    checkpoint_every_steps: u64,
    keep_last_k: usize,
    log_every: usize,
    seed: u64,
    resume: ResumeMode,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--help" || a == "-h") {
            print_help();
            std::process::exit(0);
        }

        let mut cfg = Self {
            tokens_bin: PathBuf::from("/opt/datasets/oracle-lora/corpus.tokens.bin"),
            teacher_gguf: PathBuf::from(
                "/opt/AxonML/models/oracle-distill/oracle-r1-distill-q4km.gguf",
            ),
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            arch: DEFAULT_ARCH.to_string(),
            seq_len: DEFAULT_SEQ_LEN,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
            steps_per_epoch: DEFAULT_STEPS_PER_EPOCH,
            lr: DEFAULT_LR,
            warmup_steps: DEFAULT_WARMUP_STEPS,
            temperature: DEFAULT_TEMPERATURE,
            ce_weight: DEFAULT_CE_WEIGHT,
            weight_decay: DEFAULT_WEIGHT_DECAY,
            grad_clip: DEFAULT_GRAD_CLIP,
            k_min: None,
            k_max: None,
            checkpoint_every_steps: DEFAULT_CHECKPOINT_EVERY,
            keep_last_k: DEFAULT_KEEP_LAST_K,
            log_every: DEFAULT_LOG_EVERY,
            seed: 42,
            resume: ResumeMode::None,
        };

        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            let next = |i: usize| -> String {
                args.get(i + 1).cloned().unwrap_or_else(|| {
                    eprintln!("Missing value for {a}");
                    std::process::exit(1);
                })
            };
            match a.as_str() {
                "--tokens-bin" => {
                    cfg.tokens_bin = PathBuf::from(next(i));
                    i += 2;
                }
                "--teacher-gguf" => {
                    cfg.teacher_gguf = PathBuf::from(next(i));
                    i += 2;
                }
                "--output-dir" => {
                    cfg.output_dir = PathBuf::from(next(i));
                    i += 2;
                }
                "--arch" => {
                    cfg.arch = next(i);
                    i += 2;
                }
                "--seq-len" => {
                    cfg.seq_len = next(i).parse().unwrap();
                    i += 2;
                }
                "--bs" | "--batch-size" => {
                    cfg.batch_size = next(i).parse().unwrap();
                    i += 2;
                }
                "--epochs" => {
                    cfg.epochs = next(i).parse().unwrap();
                    i += 2;
                }
                "--steps" => {
                    cfg.steps_per_epoch = next(i).parse().unwrap();
                    i += 2;
                }
                "--lr" => {
                    cfg.lr = next(i).parse().unwrap();
                    i += 2;
                }
                "--warmup" => {
                    cfg.warmup_steps = next(i).parse().unwrap();
                    i += 2;
                }
                "--temperature" | "-T" => {
                    cfg.temperature = next(i).parse().unwrap();
                    i += 2;
                }
                "--alpha" | "--ce-weight" => {
                    cfg.ce_weight = next(i).parse().unwrap();
                    i += 2;
                }
                "--weight-decay" => {
                    cfg.weight_decay = next(i).parse().unwrap();
                    i += 2;
                }
                "--grad-clip" => {
                    cfg.grad_clip = next(i).parse().unwrap();
                    i += 2;
                }
                "--k-min" => {
                    cfg.k_min = Some(next(i).parse().unwrap());
                    i += 2;
                }
                "--k-max" => {
                    cfg.k_max = Some(next(i).parse().unwrap());
                    i += 2;
                }
                "--checkpoint-every-steps" => {
                    cfg.checkpoint_every_steps = next(i).parse().unwrap();
                    i += 2;
                }
                "--keep-last-k" => {
                    cfg.keep_last_k = next(i).parse().unwrap();
                    i += 2;
                }
                "--log-every" => {
                    cfg.log_every = next(i).parse().unwrap();
                    i += 2;
                }
                "--seed" => {
                    cfg.seed = next(i).parse().unwrap();
                    i += 2;
                }
                "--resume" => {
                    cfg.resume = match next(i).as_str() {
                        "latest" => ResumeMode::Latest,
                        "best" => ResumeMode::Best,
                        other => ResumeMode::Path(PathBuf::from(other)),
                    };
                    i += 2;
                }
                other => {
                    eprintln!("Unknown flag: {other}");
                    print_help();
                    std::process::exit(1);
                }
            }
        }

        cfg
    }

    fn rdt_config(&self, vocab_size: usize) -> RDTConfig {
        let mut c = match self.arch.as_str() {
            "tiny" => RDTConfig::rdt_tiny(),
            "small" => RDTConfig::rdt_small(),
            "mid" => RDTConfig::rdt_mid(),
            other => {
                eprintln!("Unknown --arch '{other}'. Valid: tiny | small | mid");
                std::process::exit(1);
            }
        };
        if let Some(k) = self.k_min {
            c.k_min = k;
        }
        if let Some(k) = self.k_max {
            c.k_max = k;
        }
        assert!(c.k_min <= c.k_max, "k_min must be <= k_max");
        c.base.vocab_size = vocab_size;
        c.base.max_position_embeddings = self.seq_len;
        c
    }
}

fn print_help() {
    eprintln!("train_rdt_distill — Oracle → RDT knowledge distillation");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  train_rdt_distill [OPTIONS]");
    eprintln!();
    eprintln!("DATA (both required — defaults point to Oracle corpus + teacher):");
    eprintln!("  --tokens-bin PATH     Pre-tokenized corpus (flat u32 LE)");
    eprintln!("  --teacher-gguf PATH   Oracle teacher GGUF (qwen2 or qwen3 arch)");
    eprintln!();
    eprintln!("MODEL:");
    eprintln!("  --arch NAME           RDT preset: tiny | small | mid (default: {DEFAULT_ARCH})");
    eprintln!("  --k-min N             Override preset k_min");
    eprintln!("  --k-max N             Override preset k_max");
    eprintln!();
    eprintln!("TRAINING:");
    eprintln!("  --seq-len N           Sequence length (default: {DEFAULT_SEQ_LEN})");
    eprintln!("  --bs N                Batch size (default: {DEFAULT_BATCH_SIZE})");
    eprintln!("  --epochs N            Number of epochs (default: {DEFAULT_EPOCHS})");
    eprintln!("  --steps N             Steps per epoch (default: {DEFAULT_STEPS_PER_EPOCH})");
    eprintln!();
    eprintln!("OPTIMIZATION:");
    eprintln!("  --lr F                Peak learning rate (default: {DEFAULT_LR})");
    eprintln!("  --warmup N            Linear warmup steps (default: {DEFAULT_WARMUP_STEPS})");
    eprintln!("  --weight-decay F      AdamW weight decay (default: {DEFAULT_WEIGHT_DECAY})");
    eprintln!("  --grad-clip F         Gradient clip norm (default: {DEFAULT_GRAD_CLIP})");
    eprintln!();
    eprintln!("DISTILLATION:");
    eprintln!(
        "  -T, --temperature F   Softmax temperature for KL (default: {DEFAULT_TEMPERATURE})"
    );
    eprintln!("  --alpha F             CE weight (KL weight = 1-α) (default: {DEFAULT_CE_WEIGHT})");
    eprintln!();
    eprintln!("CHECKPOINTING:");
    eprintln!("  --output-dir PATH            (default: {DEFAULT_OUTPUT_DIR})");
    eprintln!("  --checkpoint-every-steps N   (default: {DEFAULT_CHECKPOINT_EVERY})");
    eprintln!("  --keep-last-k N              (default: {DEFAULT_KEEP_LAST_K})");
    eprintln!("  --resume latest|best|PATH    Resume from checkpoint");
    eprintln!();
    eprintln!("LOGGING:");
    eprintln!("  --log-every N         Log every N steps (default: {DEFAULT_LOG_EVERY})");
    eprintln!("  --seed N              RNG seed (default: 42)");
    eprintln!();
    eprintln!("See /opt/AxonML/docs/RDT_DESIGN.md §8a for the full design.");
}

// =============================================================================
// Device
// =============================================================================

#[cfg(feature = "cuda")]
fn pick_device() -> Device {
    if axonml_core::backends::cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

#[cfg(not(feature = "cuda"))]
fn pick_device() -> Device {
    Device::Cpu
}

fn device_name(device: &Device) -> String {
    if device.is_gpu() {
        format!("GPU ({device:?})")
    } else {
        "CPU".to_string()
    }
}

// =============================================================================
// Distillation loss
// =============================================================================

fn shifted_kl_divergence(
    student_logits: &Variable,
    teacher_logits: &Variable,
    temperature: f32,
) -> Variable {
    let shape = student_logits.data().shape().to_vec();
    assert_eq!(
        shape,
        teacher_logits.data().shape(),
        "student and teacher logits must have the same shape",
    );
    let batch_size = shape[0];
    let seq_len = shape[1];
    let vocab_size = shape[2];

    if seq_len <= 1 {
        let zero = Tensor::from_vec(vec![0.0f32], &[1]).unwrap();
        return Variable::new(zero, false);
    }

    let n = batch_size * (seq_len - 1);
    let s = student_logits
        .narrow(1, 0, seq_len - 1)
        .reshape(&[n, vocab_size]);
    let t = teacher_logits
        .narrow(1, 0, seq_len - 1)
        .reshape(&[n, vocab_size]);

    KLDivLoss::with_reduction(temperature, Reduction::Mean).compute(&s, &t)
}

// =============================================================================
// LR schedule (linear warmup → cosine decay, logged only for now)
// =============================================================================

fn lr_for_step(step: usize, peak: f32, warmup: usize, total: usize) -> f32 {
    if step < warmup {
        peak * (step as f32 / warmup.max(1) as f32)
    } else if total <= warmup {
        peak
    } else {
        let progress = (step - warmup) as f32 / (total - warmup) as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress.min(1.0)).cos());
        peak * (0.1 + 0.9 * cosine)
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════════════");
    println!(" Oracle → RDT Distillation (RDT_DESIGN §8a)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let device = pick_device();
    println!("Device: {}", device_name(&device));

    // ---- Load pre-tokenized corpus ----
    let dataset = TextDataset::from_tokens_bin(&cfg.tokens_bin, cfg.seq_len).unwrap_or_else(|e| {
        eprintln!(
            "failed to load --tokens-bin {}: {e}",
            cfg.tokens_bin.display()
        );
        std::process::exit(1);
    });
    println!(
        "Corpus: {} ({} tokens, {} windows of len {})",
        cfg.tokens_bin.display(),
        format_count(dataset.tokens().len()),
        format_count(dataset.len()),
        cfg.seq_len,
    );

    // ---- Load teacher (frozen, GGUF) ----
    println!();
    println!("Teacher: loading {}", cfg.teacher_gguf.display());
    let t0 = Instant::now();
    let (teacher, teacher_cfg) = load_qwen3_from_gguf(&cfg.teacher_gguf).unwrap_or_else(|e| {
        eprintln!("Failed to load teacher GGUF: {e}");
        std::process::exit(1);
    });
    println!(
        "         ✓ loaded in {:.1}s — vocab={}, hidden={}, layers={}, heads={}x{}, head_dim={}",
        t0.elapsed().as_secs_f32(),
        teacher_cfg.vocab_size,
        teacher_cfg.hidden_size,
        teacher_cfg.num_hidden_layers,
        teacher_cfg.num_attention_heads,
        teacher_cfg.num_key_value_heads,
        teacher_cfg.head_dim,
    );

    // ---- Build RDT student with matched vocab ----
    let rdt_cfg = cfg.rdt_config(teacher_cfg.vocab_size);
    let mut student = RDTForCausalLM::new(&rdt_cfg);
    let param_count: usize = student.parameters().iter().map(|p| p.data().numel()).sum();

    println!();
    println!(
        "Student: RDT (preset={}, vocab={}, hidden={}, prelude/core/coda={}/{}/{}, K∈[{},{}])",
        cfg.arch,
        rdt_cfg.base.vocab_size,
        rdt_cfg.base.hidden_size,
        rdt_cfg.n_prelude,
        rdt_cfg.n_core,
        rdt_cfg.n_coda,
        rdt_cfg.k_min,
        rdt_cfg.k_max,
    );
    println!(
        "  params    : {} ({:.2}M)",
        format_count(param_count),
        param_count as f64 / 1e6
    );

    // ---- Resume from checkpoint if any ----
    std::fs::create_dir_all(&cfg.output_dir).expect("create output_dir");
    let mut training_state = TrainingState::new();
    let mut start_epoch = 0usize;
    if let Some(ckpt_path) = find_checkpoint(&cfg.output_dir, &cfg.resume) {
        match load_model_from_checkpoint(&student, &ckpt_path) {
            Ok((epoch, state)) => {
                start_epoch = epoch;
                training_state = state;
                println!("Resuming from epoch {start_epoch}");
            }
            Err(e) => eprintln!("Resume failed: {e} — starting fresh"),
        }
    } else {
        println!("Starting fresh training run");
    }

    // ---- Move params to device ----
    if device.is_gpu() {
        for p in student.parameters() {
            p.to_device(device.clone());
        }
        for p in teacher.parameters() {
            p.to_device(device.clone());
        }
    }

    // ---- Training lifecycle ----
    let lifecycle = TrainingLifecycle::builder()
        .model_name(&format!("rdt-{}-oracle-distill", cfg.arch))
        .output_dir(&cfg.output_dir)
        .param_count(param_count)
        .total_epochs(cfg.epochs)
        .batch_size(cfg.batch_size)
        .checkpoint_every_steps(cfg.checkpoint_every_steps)
        .keep_last_k(cfg.keep_last_k)
        .start();
    println!();

    // ---- Optimizer ----
    let mut optimizer = AdamW::new(student.parameters(), cfg.lr).weight_decay(cfg.weight_decay);

    let total_steps = cfg.epochs * cfg.steps_per_epoch;
    println!("Training:");
    println!("  batch         : {}", cfg.batch_size);
    println!(
        "  epochs        : {} (starting at {})",
        cfg.epochs,
        start_epoch + 1
    );
    println!("  steps/ep      : {}", cfg.steps_per_epoch);
    println!("  total steps   : {total_steps}");
    println!("  peak lr       : {}", cfg.lr);
    println!("  warmup steps  : {}", cfg.warmup_steps);
    println!("  weight_decay  : {}", cfg.weight_decay);
    println!("  grad_clip     : {}", cfg.grad_clip);
    println!();
    println!("Distillation:");
    println!("  T (temp)      : {}", cfg.temperature);
    println!(
        "  α (ce_weight) : {}  (1-α = {} for KL)",
        cfg.ce_weight,
        1.0 - cfg.ce_weight
    );
    println!();
    println!(
        "{:>6} {:>8} {:>4} {:>9} {:>9} {:>9} {:>10} {:>9}",
        "Epoch", "Step", "K", "CE", "KL", "Total", "LR", "Time"
    );
    println!("{}", "-".repeat(72));

    let mut best_loss = training_state.best_metric.unwrap_or(f32::MAX);
    let global_start = Instant::now();
    let mut global_step = training_state.global_step;

    // Two RNG streams — batch + K advance independently for reproducibility.
    let mut rng_batch: u64 = cfg.seed;
    let mut rng_k: u64 = cfg.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let k_range_len = rdt_cfg.k_max - rdt_cfg.k_min + 1;

    let mut stopped_early = false;
    'outer: for epoch in (start_epoch + 1)..=cfg.epochs {
        lifecycle.set_epoch(epoch);
        student.train();
        let epoch_start = Instant::now();
        let mut running_total = 0.0f32;
        let mut running_ce = 0.0f32;
        let mut running_kl = 0.0f32;
        let mut running_count = 0usize;
        let mut epoch_loss_sum = 0.0f32;
        let mut epoch_count = 0usize;

        for step in 1..=cfg.steps_per_epoch {
            match lifecycle.poll() {
                LoopAction::Stop => {
                    lifecycle.save_final(&student, &training_state, epoch);
                    stopped_early = true;
                    break 'outer;
                }
                LoopAction::CheckpointNow => {
                    lifecycle.save_step(&student, &training_state, epoch);
                }
                LoopAction::Continue => {}
            }

            // Sample K for this batch — the defining RDT training trick.
            // Fixed-K training degrades badly when K varies at inference.
            let k = rdt_cfg.k_min + lcg_range(&mut rng_k, k_range_len);

            // Sample batch of token IDs (shared across labels since we
            // predict next-token and shift inside the loss).
            let batch_data = dataset.sample_batch(cfg.batch_size, &mut rng_batch);
            let input_ids =
                Tensor::<u32>::from_vec(batch_data.clone(), &[cfg.batch_size, cfg.seq_len])
                    .unwrap();
            let labels =
                Tensor::<u32>::from_vec(batch_data, &[cfg.batch_size, cfg.seq_len]).unwrap();

            // Teacher forward under NoGradGuard — frozen, no grad, no graph.
            let teacher_logits = {
                let _guard = NoGradGuard::new();
                teacher.forward_ids(&input_ids)
            };

            // Student forward at sampled K.
            optimizer.zero_grad();
            let student_logits = student.forward_ids(&input_ids, k);

            // Combined loss: α·CE + (1-α)·KL(T²).
            let ce_loss = shifted_cross_entropy(&student_logits, &labels);
            let kl_loss = shifted_kl_divergence(&student_logits, &teacher_logits, cfg.temperature);

            let ce_val = ce_loss.data().to_vec()[0];
            let kl_val = kl_loss.data().to_vec()[0];

            let loss = ce_loss
                .mul_scalar(cfg.ce_weight)
                .add(&kl_loss.mul_scalar(1.0 - cfg.ce_weight));
            let total_val = loss.data().to_vec()[0];

            let current_lr = lr_for_step(global_step, cfg.lr, cfg.warmup_steps, total_steps);
            // AdamW lacks an in-place LR setter in this tree; peak LR is
            // what's actually applied. Schedule is logged for visibility
            // and will be wired once the optimizer grows a setter.
            let _ = current_lr;

            loss.backward();
            optimizer.step();

            running_total += total_val;
            running_ce += ce_val;
            running_kl += kl_val;
            running_count += 1;
            epoch_loss_sum += total_val;
            epoch_count += 1;
            global_step += 1;
            training_state.next_step();
            training_state.record_loss(total_val);
            lifecycle.tick(global_step as u64, total_val);

            if lifecycle.should_step_checkpoint(global_step as u64) {
                lifecycle.save_step(&student, &training_state, epoch);
            }

            if step % cfg.log_every == 0 {
                let avg_total = running_total / running_count as f32;
                let avg_ce = running_ce / running_count as f32;
                let avg_kl = running_kl / running_count as f32;
                let elapsed = global_start.elapsed().as_secs_f32();
                println!(
                    "{:>6} {:>8} {:>4} {:>9.4} {:>9.4} {:>9.4} {:>10.2e} {:>8.1}s",
                    format!("{}/{}", epoch, cfg.epochs),
                    global_step,
                    k,
                    avg_ce,
                    avg_kl,
                    avg_total,
                    current_lr,
                    elapsed,
                );
                running_total = 0.0;
                running_ce = 0.0;
                running_kl = 0.0;
                running_count = 0;
            }
        }

        let epoch_avg = epoch_loss_sum / epoch_count.max(1) as f32;
        let epoch_time = epoch_start.elapsed();

        lifecycle.log_epoch(epoch, epoch_avg, None, vec![]);

        let prev_best = best_loss;
        if lifecycle.save_if_best(&student, &training_state, epoch, epoch_avg, prev_best) {
            best_loss = epoch_avg;
            training_state.update_best("total_loss", epoch_avg, false);
            println!("  ★ new best total-loss {:.4}", epoch_avg);
        }

        lifecycle.save_epoch(&student, &training_state, epoch);
        println!(
            "  epoch {} done in {:.1}s | total {:.4}",
            epoch,
            epoch_time.as_secs_f32(),
            epoch_avg,
        );
        training_state.next_epoch();
    }

    if stopped_early {
        lifecycle.set_status("stopped");
    }
    lifecycle.finish();
    let total_time = global_start.elapsed();

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!(" Oracle → RDT Distillation Complete");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  time      : {:.1}s ({:.1} min)",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / 60.0,
    );
    println!("  best loss : {:.4}", best_loss);
    println!("  output    : {}", cfg.output_dir.display());
}
