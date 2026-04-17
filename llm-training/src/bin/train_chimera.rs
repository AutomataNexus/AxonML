//! Train Chimera (MoE + Differential Attention) — AxonML Shakespeare Trainer
//!
//! End-to-end training binary for the AxonML [`ChimeraModel`] on a text
//! corpus. Chimera combines:
//! - Sparse Mixture-of-Experts MLP (top-k routing per token)
//! - Differential attention with a learned `lambda` parameter
//! - RMSNorm + residual transformer blocks
//!
//! Like Hydra, `ChimeraModel` exposes its own
//! `forward_with_loss(input_ids, labels)` method (chimera.rs:355) that
//! performs the shift-then-CE step and adds the MoE load-balance auxiliary
//! term (weighted by `load_balance_weight`), so this binary uses it directly
//! instead of the shared `shifted_cross_entropy` helper.
//!
//! ## What this file contains
//! - `Config` struct + `Config::from_args` CLI parser, `print_help` — full
//!   set of MoE (experts / top-k / load-balance weight) + differential-
//!   attention (`lambda_init`) + standard transformer hyperparameters.
//! - `generate` — greedy auto-regressive sampler that feeds the tail of the
//!   running id buffer back into `ChimeraModel::forward_ids`, pulls the
//!   last-step logits, and picks `argmax` for in-flight text previews.
//! - `pick_device` / `device_name` — CUDA-feature-gated device selection.
//! - `main` — validates head-divisibility + `top_k <= experts`, loads the
//!   corpus, builds a `CharTokenizer` + [`TextDataset`], constructs the
//!   [`ChimeraConfig`] / [`ChimeraModel`], resumes from a checkpoint,
//!   moves params to GPU, wires the `TrainingLifecycle`, and runs the
//!   Adam-optimized training loop with periodic greedy sampling and final
//!   generation of a 400-char ROMEO soliloquy.
//!
//! Usage:
//!   cargo run --release --bin train_chimera -p llm-training --features cuda
//!   cargo run --release --bin train_chimera -p llm-training --features cuda -- \
//!       --epochs 10 --bs 16 --seq-len 128 --experts 4 --top-k 2 --resume latest
//!
//! # File
//! `llm-training/src/bin/train_chimera.rs`
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

use axonml_core::Device;
use axonml_llm::{ChimeraConfig, ChimeraModel};
use axonml_nn::Module;
use axonml_optim::{Adam, Optimizer};
use axonml_serialize::TrainingState;
use axonml_tensor::Tensor;

use llm_training::{
    find_checkpoint, format_count, load_model_from_checkpoint, read_corpus, CharTokenizer,
    LoopAction, ResumeMode, TextDataset, TrainingLifecycle,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_CORPUS: &str = "/opt/datasets/text/shakespeare.txt";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/llm-training/checkpoints/chimera";
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_D_MODEL: usize = 192;
const DEFAULT_INTERMEDIATE: usize = 512;
const DEFAULT_NUM_LAYERS: usize = 4;
const DEFAULT_NUM_HEADS: usize = 6;
const DEFAULT_NUM_EXPERTS: usize = 4;
const DEFAULT_TOP_K: usize = 2;
const DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_EPOCHS: usize = 5;
const DEFAULT_LR: f32 = 3e-4;
const DEFAULT_STEPS_PER_EPOCH: usize = 500;
const DEFAULT_LOG_EVERY: usize = 50;
const DEFAULT_GENERATE_EVERY: usize = 100;
const DEFAULT_SEED: u64 = 1337;
const DEFAULT_RMS_EPS: f32 = 1e-5;
const DEFAULT_LAMBDA_INIT: f32 = 0.05;
const DEFAULT_LOAD_BAL_WEIGHT: f32 = 0.01;
const DEFAULT_CHECKPOINT_EVERY_STEPS: u64 = 0;
const DEFAULT_KEEP_LAST_K: usize = 5;

// =============================================================================
// Config / CLI
// =============================================================================

struct Config {
    corpus: PathBuf,
    output_dir: PathBuf,
    seq_len: usize,
    d_model: usize,
    intermediate: usize,
    num_layers: usize,
    num_heads: usize,
    num_experts: usize,
    top_k: usize,
    lambda_init: f32,
    load_balance_weight: f32,
    batch_size: usize,
    epochs: usize,
    lr: f32,
    steps_per_epoch: usize,
    log_every: usize,
    generate_every: usize,
    seed: u64,
    resume: ResumeMode,
    checkpoint_every_steps: u64,
    keep_last_k: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            corpus: PathBuf::from(DEFAULT_CORPUS),
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            seq_len: DEFAULT_SEQ_LEN,
            d_model: DEFAULT_D_MODEL,
            intermediate: DEFAULT_INTERMEDIATE,
            num_layers: DEFAULT_NUM_LAYERS,
            num_heads: DEFAULT_NUM_HEADS,
            num_experts: DEFAULT_NUM_EXPERTS,
            top_k: DEFAULT_TOP_K,
            lambda_init: DEFAULT_LAMBDA_INIT,
            load_balance_weight: DEFAULT_LOAD_BAL_WEIGHT,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
            lr: DEFAULT_LR,
            steps_per_epoch: DEFAULT_STEPS_PER_EPOCH,
            log_every: DEFAULT_LOG_EVERY,
            generate_every: DEFAULT_GENERATE_EVERY,
            seed: DEFAULT_SEED,
            resume: ResumeMode::Latest,
            checkpoint_every_steps: DEFAULT_CHECKPOINT_EVERY_STEPS,
            keep_last_k: DEFAULT_KEEP_LAST_K,
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
                "--corpus" => { i += 1; cfg.corpus = PathBuf::from(&args[i]); }
                "--out" => { i += 1; cfg.output_dir = PathBuf::from(&args[i]); }
                "--seq-len" => { i += 1; cfg.seq_len = args[i].parse().unwrap(); }
                "--d-model" => { i += 1; cfg.d_model = args[i].parse().unwrap(); }
                "--intermediate" => { i += 1; cfg.intermediate = args[i].parse().unwrap(); }
                "--layers" => { i += 1; cfg.num_layers = args[i].parse().unwrap(); }
                "--heads" => { i += 1; cfg.num_heads = args[i].parse().unwrap(); }
                "--experts" => { i += 1; cfg.num_experts = args[i].parse().unwrap(); }
                "--top-k" => { i += 1; cfg.top_k = args[i].parse().unwrap(); }
                "--lambda-init" => { i += 1; cfg.lambda_init = args[i].parse().unwrap(); }
                "--load-balance" => { i += 1; cfg.load_balance_weight = args[i].parse().unwrap(); }
                "--bs" | "--batch-size" => { i += 1; cfg.batch_size = args[i].parse().unwrap(); }
                "--epochs" => { i += 1; cfg.epochs = args[i].parse().unwrap(); }
                "--lr" => { i += 1; cfg.lr = args[i].parse().unwrap(); }
                "--steps" => { i += 1; cfg.steps_per_epoch = args[i].parse().unwrap(); }
                "--log-every" => { i += 1; cfg.log_every = args[i].parse().unwrap(); }
                "--generate-every" => { i += 1; cfg.generate_every = args[i].parse().unwrap(); }
                "--seed" => { i += 1; cfg.seed = args[i].parse().unwrap(); }
                "--resume" => { i += 1; cfg.resume = ResumeMode::from_str(&args[i]); }
                "--fresh" => { cfg.resume = ResumeMode::None; }
                "--checkpoint-every-steps" => { i += 1; cfg.checkpoint_every_steps = args[i].parse().unwrap(); }
                "--keep-last-k" => { i += 1; cfg.keep_last_k = args[i].parse().unwrap(); }
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
    println!(r#"Train Chimera (MoE + Differential Attention) on a text corpus.

Usage: train_chimera [OPTIONS]

Options:
  --corpus PATH       Text corpus (default: /opt/datasets/text/shakespeare.txt)
  --out PATH          Checkpoint directory (default: .../checkpoints/chimera)
  --seq-len N         Context window length (default: 128)
  --d-model N         Hidden dimension (default: 192)
  --intermediate N    MoE expert intermediate size (default: 512)
  --layers N          Transformer blocks (default: 4)
  --heads N           Differential-attention heads (default: 6)
  --experts N         Experts per MoE layer (default: 4)
  --top-k N           Experts activated per token (default: 2)
  --lambda-init FLOAT Initial lambda for differential attention (default: 0.05)
  --load-balance FLOAT Weight on MoE load-balance aux loss (default: 0.01)
  --bs N              Batch size (default: 16)
  --epochs N          Epochs (default: 5)
  --lr FLOAT          Learning rate (default: 3e-4)
  --steps N           Training steps per epoch (default: 500)
  --log-every N       Log every N steps (default: 50)
  --generate-every N  Generate sample every N steps (default: 100)
  --seed N            RNG seed (default: 1337)
  --resume MODE       Resume: none|latest|best|<path> (default: latest)
  --fresh             Equivalent to --resume none
  --checkpoint-every-steps N   Rotating step-level checkpoint every N steps (0 = off)
  --keep-last-k N     Keep last N step checkpoints on disk (default: 5)
  --help, -h          Show help"#);
}

// =============================================================================
// Greedy text generation
// =============================================================================

fn generate(
    model: &ChimeraModel,
    tokenizer: &CharTokenizer,
    prompt: &str,
    n_chars: usize,
    max_seq_len: usize,
    _device: &Device,
) -> String {
    let mut ids = tokenizer.encode(prompt);
    if ids.is_empty() {
        ids.push(0);
    }
    let mut generated = String::from(prompt);

    for _ in 0..n_chars {
        let ctx_start = ids.len().saturating_sub(max_seq_len);
        let ctx = &ids[ctx_start..];
        let ctx_len = ctx.len();

        let input = Tensor::<u32>::from_vec(ctx.to_vec(), &[1, ctx_len]).unwrap();

        let logits = model.forward_ids(&input);
        let logits_data = logits.data();
        let logits_vec = logits_data.to_vec();
        let vocab_size = tokenizer.vocab_size();

        let last_offset = (ctx_len - 1) * vocab_size;
        let last_logits = &logits_vec[last_offset..last_offset + vocab_size];

        let next_id = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        ids.push(next_id);
        generated.push_str(&tokenizer.decode(&[next_id]));
    }
    generated
}

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" Chimera Training — AxonML on Shakespeare");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    if cfg.d_model % cfg.num_heads != 0 {
        eprintln!(
            "Invalid head config: --d-model ({}) must be divisible by --heads ({})",
            cfg.d_model, cfg.num_heads
        );
        std::process::exit(1);
    }
    if cfg.top_k > cfg.num_experts {
        eprintln!(
            "Invalid MoE config: --top-k ({}) must be <= --experts ({})",
            cfg.top_k, cfg.num_experts
        );
        std::process::exit(1);
    }

    // ---- Device detection ----
    let device = pick_device();
    println!("Device: {}", device_name(&device));

    // ---- Load corpus ----
    let corpus_text = read_corpus(&cfg.corpus);
    println!("Corpus: {} ({} chars)", cfg.corpus.display(), format_count(corpus_text.len()));

    // ---- Tokenizer + dataset ----
    let tokenizer = CharTokenizer::from_corpus(&corpus_text);
    let vocab_size = tokenizer.vocab_size();
    println!("Vocab:  {} chars", vocab_size);

    let dataset = TextDataset::from_string(&corpus_text, &tokenizer, cfg.seq_len);
    println!("Windows: {}", format_count(dataset.len()));
    println!();

    // ---- Build model ----
    let model_config = ChimeraConfig {
        vocab_size,
        d_model: cfg.d_model,
        num_layers: cfg.num_layers,
        num_heads: cfg.num_heads,
        num_experts: cfg.num_experts,
        top_k: cfg.top_k,
        intermediate_size: cfg.intermediate,
        max_seq_len: cfg.seq_len,
        rms_norm_eps: DEFAULT_RMS_EPS,
        lambda_init: cfg.lambda_init,
        load_balance_weight: cfg.load_balance_weight,
    };
    let mut model = ChimeraModel::new(&model_config);
    let param_count: usize = model.parameters().iter().map(|p| p.data().numel()).sum();
    let total_params = model_config.estimate_total_params();
    let active_params = model_config.estimate_active_params();

    println!("Model:  Chimera (sparse MoE + Differential Attention)");
    println!("  d_model         : {}", cfg.d_model);
    println!("  intermediate    : {}", cfg.intermediate);
    println!("  layers          : {}", cfg.num_layers);
    println!("  heads           : {}", cfg.num_heads);
    println!("  experts         : {} (top-{} active per token)", cfg.num_experts, cfg.top_k);
    println!("  lambda_init     : {}", cfg.lambda_init);
    println!("  load_balance    : {}", cfg.load_balance_weight);
    println!("  seq_len         : {}", cfg.seq_len);
    println!("  params          : {} actual", format_count(param_count));
    println!("  params (est)    : {} total, {} active per token", format_count(total_params), format_count(active_params));
    println!();

    // ---- Resume from checkpoint if available ----
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

    // ---- Move model params to device ----
    if device.is_gpu() {
        for p in model.parameters() {
            p.to_device(device.clone());
        }
    }

    // ---- Training lifecycle (monitor + signals + control socket) ----
    let lifecycle = TrainingLifecycle::builder()
        .model_name("Chimera (Shakespeare)")
        .output_dir(&cfg.output_dir)
        .param_count(param_count)
        .total_epochs(cfg.epochs)
        .batch_size(cfg.batch_size)
        .checkpoint_every_steps(cfg.checkpoint_every_steps)
        .keep_last_k(cfg.keep_last_k)
        .start();
    println!();

    // ---- Optimizer ----
    let mut optimizer = Adam::new(model.parameters(), cfg.lr);

    println!("Training:");
    println!("  batch     : {}", cfg.batch_size);
    println!("  epochs    : {} (starting at {})", cfg.epochs, start_epoch + 1);
    println!("  steps/ep  : {}", cfg.steps_per_epoch);
    println!("  lr        : {}", cfg.lr);
    println!();
    println!("{:>6} {:>8} {:>10} {:>10} {:>10}", "Epoch", "Step", "Loss", "PPL", "Time");
    println!("{}", "-".repeat(50));

    let mut best_loss = training_state.best_metric.unwrap_or(f32::MAX);
    let mut rng = cfg.seed;
    let global_start = Instant::now();
    let mut global_step = training_state.global_step;

    let mut stopped_early = false;
    'outer: for epoch in (start_epoch + 1)..=cfg.epochs {
        lifecycle.set_epoch(epoch);
        model.train();
        let epoch_start = Instant::now();
        let mut running_loss = 0.0f32;
        let mut running_count = 0usize;
        let mut epoch_loss_sum = 0.0f32;
        let mut epoch_count = 0usize;

        for step in 1..=cfg.steps_per_epoch {
            match lifecycle.poll() {
                LoopAction::Stop => {
                    lifecycle.save_final(&model, &training_state, epoch);
                    stopped_early = true;
                    break 'outer;
                }
                LoopAction::CheckpointNow => {
                    lifecycle.save_step(&model, &training_state, epoch);
                }
                LoopAction::Continue => {}
            }

            let batch_data = dataset.sample_batch(cfg.batch_size, &mut rng);
            let input_ids = Tensor::<u32>::from_vec(
                batch_data.clone(),
                &[cfg.batch_size, cfg.seq_len],
            ).unwrap();
            let labels = Tensor::<u32>::from_vec(
                batch_data,
                &[cfg.batch_size, cfg.seq_len],
            ).unwrap();

            // ChimeraModel's forward_with_loss returns (logits, CE + load_balance_weight * LB).
            optimizer.zero_grad();
            let (_logits, loss) = model.forward_with_loss(&input_ids, &labels);
            let loss_val = loss.data().to_vec()[0];

            loss.backward();
            optimizer.step();

            running_loss += loss_val;
            running_count += 1;
            epoch_loss_sum += loss_val;
            epoch_count += 1;
            global_step += 1;
            training_state.next_step();
            training_state.record_loss(loss_val);
            lifecycle.tick(global_step as u64, loss_val);

            if lifecycle.should_step_checkpoint(global_step as u64) {
                lifecycle.save_step(&model, &training_state, epoch);
            }

            if step % cfg.log_every == 0 {
                let avg = running_loss / running_count as f32;
                let ppl = avg.exp().min(99999.0);
                let elapsed = global_start.elapsed().as_secs_f32();
                println!(
                    "{:>6} {:>8} {:>10.4} {:>10.2} {:>9.1}s",
                    format!("{}/{}", epoch, cfg.epochs),
                    global_step,
                    avg,
                    ppl,
                    elapsed,
                );
                running_loss = 0.0;
                running_count = 0;
            }

            if step % cfg.generate_every == 0 {
                model.eval();
                let sample = generate(&model, &tokenizer, "ROMEO:\n", 160, cfg.seq_len, &device);
                let preview = sample.replace('\n', " ").chars().take(160).collect::<String>();
                println!("    sample: {preview}");
                model.train();
            }
        }

        let epoch_avg = epoch_loss_sum / epoch_count.max(1) as f32;
        let epoch_ppl = epoch_avg.exp().min(99999.0);
        let epoch_time = epoch_start.elapsed();

        lifecycle.log_epoch(epoch, epoch_avg, None, vec![("perplexity", epoch_ppl)]);

        let prev_best = best_loss;
        if lifecycle.save_if_best(&model, &training_state, epoch, epoch_avg, prev_best) {
            best_loss = epoch_avg;
            training_state.update_best("loss", epoch_avg, false);
            println!("  ★ new best loss {:.4}", epoch_avg);
        }

        lifecycle.save_epoch(&model, &training_state, epoch);

        println!(
            "  epoch {} done in {:.1}s | loss {:.4} | ppl {:.2}",
            epoch,
            epoch_time.as_secs_f32(),
            epoch_avg,
            epoch_ppl,
        );
        training_state.next_epoch();
    }

    if stopped_early {
        lifecycle.set_status("stopped");
    }
    lifecycle.finish();
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
    println!();

    model.eval();
    println!("=== Final generation ===");
    let final_sample = generate(&model, &tokenizer, "ROMEO:\n", 400, cfg.seq_len, &device);
    println!("{final_sample}");
}

// =============================================================================
// Device helpers
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
