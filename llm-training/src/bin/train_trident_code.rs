//! Train Trident-Coder (1.58-bit Ternary SLM) — AxonML From-Scratch Trainer
//!
//! End-to-end training binary for the AxonML [`TridentModel`] — a 1.58-bit
//! ternary small language model — on pre-tokenized code tokens, with full
//! Phase-0 lifecycle controls (pause / resume / stop / checkpoint /
//! monitor) and a linear-warmup → cosine-decay learning-rate schedule.
//!
//! # Configs
//! Driven by the `ModelVariant` enum (`Smoke` / `OneB` / `ThreeB`):
//! - `smoke`  : ~30M-param tiny 1B-shaped model for local CPU sanity
//!   checks; defaults to `seq_len=64`, `batch_size=8`, `steps=1000`,
//!   no step-level checkpoints.
//! - `1b`     : 24L × 2048d × GQA 4:1, ReLU²-gated FFN, SubLN, RoPE
//!   θ=500k; defaults to `seq_len=4096`, `batch_size=4`, `steps=100_000`,
//!   rotating checkpoint every 1000 steps.
//! - `3b`     : stubbed for later scaling — defaults carried over from
//!   `1b` but don't train yet.
//!
//! # Tokenizer
//! Loads the 32k-vocab byte-level BPE at
//! `/opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json` via the
//! `tokenizers` crate (`Tokenizer::from_file`).
//!
//! # Dataset
//! Expects pre-tokenized token IDs on disk as a flat u32 little-endian
//! file (`.bin`). For smoke runs without a pre-tokenized corpus,
//! `build_or_load_smoke_dataset` tokenizes
//! `/opt/datasets/text/shakespeare.txt` into
//! `/tmp/shakespeare.trident-bpe.bin` and caches it. `load_token_bin`
//! reads the `.bin` file into an in-memory `Vec<u32>` after validating
//! length-mod-4, and `sample_batch` draws random sliding-window batches
//! using [`lcg_range`].
//!
//! The companion Python pre-tokenizer
//! `/opt/AxonML/llm-training/tools/pretokenize_stack_v2.py` emits u32-LE
//! shards from The Stack v2 for the real run.
//!
//! # What this file contains
//! - `ModelVariant` + `default_*` helpers + `Config` + `Config::from_args`
//!   with a two-pass argv parser (first pass reads `--config` so later
//!   defaults can depend on it) and `print_help`.
//! - `load_tokenizer`, `build_or_load_smoke_dataset`, `load_token_bin`,
//!   `sample_batch` — tokenizer + dataset I/O.
//! - `cosine_lr` — linear-warmup then cosine-decay to `lr * min_lr_ratio`.
//! - `pick_device` / `device_name` — CUDA-feature-gated device detection.
//! - `main` — sets up tokenizer, dataset, `TridentConfig::smoke /
//!   trident_1b / trident_3b`, resumes from a checkpoint, migrates params
//!   to GPU, wires the `TrainingLifecycle`, and runs the LR-scheduled
//!   Adam-optimized training loop using the model's built-in
//!   `forward_with_loss` for causal-LM loss, tracking best loss and
//!   flushing final + step checkpoints on exit.
//!
//! # Usage
//! ```bash
//! # Local CPU smoke test
//! cargo run --release --bin train_trident_code -- \
//!     --config smoke --steps 100 --seq-len 64 --batch-size 8 --lr 3e-4
//!
//! # Full 1B run on Colab / H100
//! cargo run --release --bin train_trident_code --features cuda -- \
//!     --config 1b --dataset /mnt/stack-v2.bin --steps 100000 \
//!     --seq-len 4096 --batch-size 4 --lr 3e-4 \
//!     --checkpoint-every-steps 1000 --keep-last-k 10
//!
//! # From another terminal:
//! cargo run --release --bin train_ctl -- status
//! cargo run --release --bin train_ctl -- pause
//! cargo run --release --bin train_ctl -- resume
//! cargo run --release --bin train_ctl -- stop
//! ```
//!
//! # File
//! `llm-training/src/bin/train_trident_code.rs`
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

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use axonml_core::Device;
use axonml_llm::{TridentConfig, TridentModel};
use axonml_nn::Module;
use axonml_optim::{Adam, Optimizer};
use axonml_serialize::TrainingState;
use axonml_tensor::Tensor;
use tokenizers::Tokenizer;

use llm_training::{
    LoopAction, ResumeMode, TrainingLifecycle, find_checkpoint, format_count, lcg_range,
    load_model_from_checkpoint,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_TOKENIZER: &str = "/opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json";
const DEFAULT_OUTPUT_DIR_SMOKE: &str = "/opt/AxonML/llm-training/checkpoints/trident-smoke";
const DEFAULT_OUTPUT_DIR_LAPTOP: &str = "/opt/AxonML/llm-training/checkpoints/trident-laptop";
const DEFAULT_OUTPUT_DIR_300M: &str = "/opt/AxonML/llm-training/checkpoints/trident-300m";
const DEFAULT_OUTPUT_DIR_500M: &str = "/opt/AxonML/llm-training/checkpoints/trident-500m";
const DEFAULT_OUTPUT_DIR_1B: &str = "/opt/AxonML/llm-training/checkpoints/trident-1b";
const DEFAULT_OUTPUT_DIR_3B: &str = "/opt/AxonML/llm-training/checkpoints/trident-3b";
const DEFAULT_SMOKE_CORPUS: &str = "/opt/datasets/text/shakespeare.txt";
const DEFAULT_SMOKE_DATASET_CACHE: &str = "/tmp/shakespeare.trident-bpe.bin";

const DEFAULT_SEED: u64 = 1337;
const DEFAULT_LOG_EVERY: usize = 10;

// =============================================================================
// Config / CLI
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ModelVariant {
    Smoke,
    Laptop,
    ThreeHundredM,
    FiveHundredM,
    OneB,
    ThreeB,
}

impl ModelVariant {
    fn from_str(s: &str) -> Self {
        match s {
            "smoke" => Self::Smoke,
            "laptop" | "Laptop" | "LAPTOP" => Self::Laptop,
            "300m" | "300M" | "small" => Self::ThreeHundredM,
            "500m" | "500M" | "med" | "medium" => Self::FiveHundredM,
            "1b" | "1B" => Self::OneB,
            "3b" | "3B" => Self::ThreeB,
            other => {
                eprintln!(
                    "unknown --config {other:?}; expected smoke | laptop | 1b | 3b"
                );
                std::process::exit(1);
            }
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Laptop => "laptop",
            Self::ThreeHundredM => "300m",
            Self::FiveHundredM => "500m",
            Self::OneB => "1b",
            Self::ThreeB => "3b",
        }
    }
}

struct Config {
    variant: ModelVariant,
    tokenizer_path: PathBuf,
    dataset_path: Option<PathBuf>,
    output_dir: PathBuf,
    seq_len: usize,
    batch_size: usize,
    lr: f32,
    steps: usize,
    warmup_steps: usize,
    min_lr_ratio: f32,
    log_every: usize,
    seed: u64,
    resume: ResumeMode,
    checkpoint_every_steps: u64,
    keep_last_k: usize,
    /// When set via `--ticker`, lifecycle spawns the
    /// `nexus-training-ticker` desktop widget after the control socket and
    /// browser monitor are up. Browser monitor is unaffected.
    ticker: bool,
}

fn default_seq_len(variant: ModelVariant) -> usize {
    match variant {
        ModelVariant::Smoke => 64,
        ModelVariant::Laptop => 256,
        ModelVariant::ThreeHundredM => 2048,
        ModelVariant::FiveHundredM => 2048,
        // 1B/3B: AxonML autograd retains every intermediate, so peak
        // activation memory grows quadratically with seq via the full
        // [bs, heads, seq, seq] attention scores tensor. At 1B-bs1:
        //   seq=2048 → 256 MB/layer × 24 = 6.1 GB just for scores
        //   seq=1024 → 64 MB/layer × 24 = 1.5 GB
        // Combined with the TernaryLinear saved_input CPU-staging
        // optimization (saves another ~2.3 GB), seq=1024 leaves ~50 GB
        // free on A100 80 GB. Override via TRIDENT_SEQ in go.sh.
        ModelVariant::OneB | ModelVariant::ThreeB => 1024,
    }
}

fn default_batch_size(variant: ModelVariant) -> usize {
    match variant {
        ModelVariant::Smoke => 8,
        ModelVariant::Laptop => 2,
        ModelVariant::ThreeHundredM => 1,
        ModelVariant::FiveHundredM => 1,
        ModelVariant::OneB | ModelVariant::ThreeB => 1,
    }
}

fn default_steps(variant: ModelVariant) -> usize {
    match variant {
        ModelVariant::Smoke => 1000,
        ModelVariant::Laptop => 50_000,
        ModelVariant::ThreeHundredM => 60_000,
        ModelVariant::FiveHundredM => 80_000,
        ModelVariant::OneB => 100_000,
        ModelVariant::ThreeB => 100_000,
    }
}

fn default_checkpoint_every(variant: ModelVariant) -> u64 {
    match variant {
        ModelVariant::Smoke => 0,
        ModelVariant::Laptop => 500,
        ModelVariant::ThreeHundredM => 1000,
        ModelVariant::FiveHundredM => 1000,
        ModelVariant::OneB | ModelVariant::ThreeB => 1000,
    }
}

fn default_output_dir(variant: ModelVariant) -> PathBuf {
    PathBuf::from(match variant {
        ModelVariant::Smoke => DEFAULT_OUTPUT_DIR_SMOKE,
        ModelVariant::Laptop => DEFAULT_OUTPUT_DIR_LAPTOP,
        ModelVariant::ThreeHundredM => DEFAULT_OUTPUT_DIR_300M,
        ModelVariant::FiveHundredM => DEFAULT_OUTPUT_DIR_500M,
        ModelVariant::OneB => DEFAULT_OUTPUT_DIR_1B,
        ModelVariant::ThreeB => DEFAULT_OUTPUT_DIR_3B,
    })
}

impl Config {
    fn from_args() -> Self {
        // First pass: find --config so we can derive variant-aware defaults.
        let args: Vec<String> = std::env::args().collect();
        let mut variant = ModelVariant::Smoke;
        let mut i = 1;
        while i < args.len() {
            if args[i] == "--config" && i + 1 < args.len() {
                variant = ModelVariant::from_str(&args[i + 1]);
                break;
            }
            if args[i] == "--help" || args[i] == "-h" {
                print_help();
                std::process::exit(0);
            }
            i += 1;
        }

        let mut cfg = Self {
            variant,
            tokenizer_path: PathBuf::from(DEFAULT_TOKENIZER),
            dataset_path: None,
            output_dir: default_output_dir(variant),
            seq_len: default_seq_len(variant),
            batch_size: default_batch_size(variant),
            lr: 3e-4,
            steps: default_steps(variant),
            warmup_steps: 100,
            min_lr_ratio: 0.1,
            log_every: DEFAULT_LOG_EVERY,
            seed: DEFAULT_SEED,
            resume: ResumeMode::Latest,
            checkpoint_every_steps: default_checkpoint_every(variant),
            keep_last_k: 5,
            ticker: false,
        };

        // Second pass: apply all args.
        i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--config" => {
                    i += 1;
                }
                "--tokenizer" => {
                    i += 1;
                    cfg.tokenizer_path = PathBuf::from(&args[i]);
                }
                "--dataset" => {
                    i += 1;
                    cfg.dataset_path = Some(PathBuf::from(&args[i]));
                }
                "--out" | "--output-dir" => {
                    i += 1;
                    cfg.output_dir = PathBuf::from(&args[i]);
                }
                "--seq-len" => {
                    i += 1;
                    cfg.seq_len = args[i].parse().unwrap();
                }
                "--batch-size" | "--bs" => {
                    i += 1;
                    cfg.batch_size = args[i].parse().unwrap();
                }
                "--lr" => {
                    i += 1;
                    cfg.lr = args[i].parse().unwrap();
                }
                "--steps" => {
                    i += 1;
                    cfg.steps = args[i].parse().unwrap();
                }
                "--warmup-steps" => {
                    i += 1;
                    cfg.warmup_steps = args[i].parse().unwrap();
                }
                "--min-lr-ratio" => {
                    i += 1;
                    cfg.min_lr_ratio = args[i].parse().unwrap();
                }
                "--log-every" => {
                    i += 1;
                    cfg.log_every = args[i].parse().unwrap();
                }
                "--seed" => {
                    i += 1;
                    cfg.seed = args[i].parse().unwrap();
                }
                "--resume" => {
                    i += 1;
                    cfg.resume = ResumeMode::from_str(&args[i]);
                }
                "--fresh" => {
                    cfg.resume = ResumeMode::None;
                }
                "--checkpoint-every-steps" => {
                    i += 1;
                    cfg.checkpoint_every_steps = args[i].parse().unwrap();
                }
                "--keep-last-k" => {
                    i += 1;
                    cfg.keep_last_k = args[i].parse().unwrap();
                }
                "--ticker" => {
                    cfg.ticker = true;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
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
    println!(
        r#"Train Trident-Coder (1.58-bit ternary SLM).

Usage: train_trident_code [OPTIONS]

Options:
  --config MODE        Model size: smoke | laptop | 500m | 1b | 3b (default: smoke)
                         smoke   : ~30M, CPU-friendly toy
                         laptop  : ~37M, fits 12 GB consumer GPU at bs=2 seq=256
                         500m    : ~525M, fits A100 80 GB at bs=1 seq=2048
                                   (recommended in-house training target)
                         1b      : 1.19B, OOMs on A100 80 GB without
                                   gradient checkpointing — use 500m unless
                                   you've added it
                         3b      : 3B stub for later scaling
  --tokenizer PATH     Tokenizer JSON (default: /opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json)
  --dataset PATH       Pre-tokenized corpus (.bin of u32 LE token IDs).
                       If omitted, smoke mode tokenizes shakespeare.txt
                       on the fly and caches at /tmp/shakespeare.trident-bpe.bin.
  --out PATH           Checkpoint directory (default: .../checkpoints/trident-{{variant}})
  --seq-len N          Context window (default: 64 smoke, 256 laptop, 1024 1b/3b)
  --batch-size N       Micro-batch size (default: 8 smoke, 2 laptop, 1 1b/3b)
                       NB: 1b/3b defaults fit A100 80 GB with ~50 GB headroom.
                       seq scales attention-scores memory quadratically
                       ([bs,heads,seq,seq] per layer × 24 layers), so doubling
                       seq quadruples score memory — bench before bumping.
  --lr FLOAT           Peak learning rate (default: 3e-4)
  --steps N            Total opt steps (default: 1000 smoke, 100000 1b/3b)
  --warmup-steps N     Linear warmup steps (default: 100)
  --min-lr-ratio FLOAT Cosine floor, min_lr = lr * ratio (default: 0.1)
  --log-every N        Log every N steps (default: 10)
  --seed N             RNG seed (default: 1337)
  --resume MODE        Resume: none|latest|best|<path> (default: latest)
  --fresh              Equivalent to --resume none
  --checkpoint-every-steps N   Rotating step-level checkpoint every N steps
                               (default: 0 smoke, 1000 1b/3b)
  --keep-last-k N      Keep last N step checkpoints on disk (default: 5)
  --ticker             Also spawn the nexus-training-ticker desktop widget
                       (compact live loss + pause/stop/checkpoint controls).
                       Browser monitor stays on either way.
  --help, -h           Show help"#
    );
}

// =============================================================================
// Tokenizer + dataset
// =============================================================================

/// Load the HuggingFace tokenizer file. Returns (tokenizer, vocab_size).
fn load_tokenizer(path: &Path) -> (Tokenizer, usize) {
    let tok = Tokenizer::from_file(path).unwrap_or_else(|e| {
        eprintln!("Failed to load tokenizer from {}: {e}", path.display());
        std::process::exit(1);
    });
    let vocab_size = tok.get_vocab_size(true);
    (tok, vocab_size)
}

/// Pre-tokenize a text file into u32-LE bytes on disk, caching for reuse.
///
/// Used only by the smoke path. For real runs, dataset is pre-built by
/// `tools/pretokenize_stack_v2.py` (not invoked here — Colab has better I/O).
fn build_or_load_smoke_dataset(tokenizer: &Tokenizer) -> PathBuf {
    let cache = PathBuf::from(DEFAULT_SMOKE_DATASET_CACHE);
    if cache.exists() {
        eprintln!("[dataset] reusing cached tokens at {}", cache.display());
        return cache;
    }
    let corpus_path = PathBuf::from(DEFAULT_SMOKE_CORPUS);
    let corpus = fs::read_to_string(&corpus_path).unwrap_or_else(|e| {
        eprintln!(
            "Failed to read smoke corpus from {}: {e}.\n\
             Either place a text file there, or pass an explicit --dataset.",
            corpus_path.display()
        );
        std::process::exit(1);
    });
    eprintln!(
        "[dataset] tokenizing {} ({} bytes) with trident-coder-bpe...",
        corpus_path.display(),
        corpus.len()
    );
    let enc = tokenizer.encode(corpus, false).unwrap_or_else(|e| {
        eprintln!("tokenize failed: {e}");
        std::process::exit(1);
    });
    let ids: Vec<u32> = enc.get_ids().to_vec();
    // Write as little-endian u32.
    let mut out = File::create(&cache).unwrap();
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    out.write_all(&bytes).unwrap();
    eprintln!(
        "[dataset] cached {} tokens at {}",
        ids.len(),
        cache.display()
    );
    cache
}

/// Memory-map-friendly read of a u32-LE token file. For the sizes we care
/// about (sub-100GB) a plain in-memory Vec is fine; mmap can be added later.
fn load_token_bin(path: &Path) -> Vec<u32> {
    let mut file = File::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to open dataset {}: {e}", path.display());
        std::process::exit(1);
    });
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    if buf.len() % 4 != 0 {
        eprintln!(
            "Dataset {} length {} is not a multiple of 4 bytes (u32 LE)",
            path.display(),
            buf.len()
        );
        std::process::exit(1);
    }
    let mut tokens = Vec::with_capacity(buf.len() / 4);
    for chunk in buf.chunks_exact(4) {
        tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    tokens
}

/// Sample a random sliding-window batch into a flat `Vec<u32>` of length
/// `batch_size * seq_len`.
fn sample_batch(tokens: &[u32], batch_size: usize, seq_len: usize, rng: &mut u64) -> Vec<u32> {
    let max_start = tokens.len().saturating_sub(seq_len + 1).max(1);
    let mut out = Vec::with_capacity(batch_size * seq_len);
    for _ in 0..batch_size {
        let start = lcg_range(rng, max_start);
        out.extend_from_slice(&tokens[start..start + seq_len]);
    }
    out
}

// =============================================================================
// LR schedule
// =============================================================================

/// Cosine schedule with linear warmup. Returns LR at `step` given peak `lr`,
/// `warmup_steps` linear-ramp, then cosine decay to `lr * min_lr_ratio` over
/// the remaining `total_steps - warmup_steps`.
fn cosine_lr(
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    lr: f32,
    min_lr_ratio: f32,
) -> f32 {
    if step < warmup_steps {
        return lr * (step as f32 + 1.0) / (warmup_steps as f32).max(1.0);
    }
    let progress = (step - warmup_steps) as f32 / ((total_steps - warmup_steps).max(1) as f32);
    let progress = progress.clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    let min_lr = lr * min_lr_ratio;
    min_lr + (lr - min_lr) * cosine
}

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" Trident-Coder Training — AxonML ternary SLM");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Variant : {}", cfg.variant.label());

    // ---- Device ----
    let device = pick_device();
    println!("Device  : {}", device_name(&device));

    // ---- Tokenizer ----
    let (tokenizer, vocab_size) = load_tokenizer(&cfg.tokenizer_path);
    println!(
        "Tokenizer: {} (vocab {})",
        cfg.tokenizer_path.display(),
        vocab_size
    );

    // ---- Dataset ----
    let dataset_path = match &cfg.dataset_path {
        Some(p) => p.clone(),
        None if cfg.variant == ModelVariant::Smoke => build_or_load_smoke_dataset(&tokenizer),
        None => {
            eprintln!(
                "--dataset is required for --config laptop/500m/1b/3b. Pre-tokenize with tools/pretokenize_stack_v2.py first."
            );
            std::process::exit(1);
        }
    };
    let tokens = load_token_bin(&dataset_path);
    println!(
        "Dataset : {} ({} tokens)",
        dataset_path.display(),
        format_count(tokens.len())
    );

    if tokens.len() < cfg.seq_len + 1 {
        eprintln!(
            "Dataset has {} tokens but need at least seq_len+1={}. Shrink --seq-len or get more data.",
            tokens.len(),
            cfg.seq_len + 1
        );
        std::process::exit(1);
    }

    // ---- Model ----
    let model_config = match cfg.variant {
        ModelVariant::Smoke => TridentConfig::smoke(vocab_size),
        ModelVariant::Laptop => TridentConfig::trident_laptop(vocab_size),
        ModelVariant::ThreeHundredM => TridentConfig::trident_300m(vocab_size),
        ModelVariant::FiveHundredM => TridentConfig::trident_500m(vocab_size),
        ModelVariant::OneB => TridentConfig::trident_1b(vocab_size),
        ModelVariant::ThreeB => TridentConfig::trident_3b(vocab_size),
    };
    let model = TridentModel::new(&model_config);
    let param_count: usize = model.parameters().iter().map(|p| p.data().numel()).sum();

    println!("Model   : Trident (1.58-bit ternary)");
    println!("  d_model       : {}", model_config.d_model);
    println!("  layers        : {}", model_config.num_layers);
    println!("  heads         : {}", model_config.num_heads);
    println!("  kv_heads      : {}", model_config.num_kv_heads);
    println!("  intermediate  : {}", model_config.intermediate_size);
    println!(
        "  seq_len       : {} (max {})",
        cfg.seq_len, model_config.max_seq_len
    );
    println!(
        "  rope          : {}  θ={}",
        model_config.use_rope, model_config.rope_theta
    );
    println!(
        "  ffn           : {}",
        if model_config.use_squared_relu {
            "ReLU²-gated"
        } else {
            "SiLU"
        }
    );
    println!("  sub_ln        : {}", model_config.use_sub_ln);
    println!("  params        : {}", format_count(param_count));
    println!();

    // ---- Resume ----
    std::fs::create_dir_all(&cfg.output_dir).expect("create output dir");
    let mut training_state = TrainingState::new();
    let mut start_epoch = 0usize;
    if let Some(ckpt_path) = find_checkpoint(&cfg.output_dir, &cfg.resume) {
        match load_model_from_checkpoint(&model, &ckpt_path) {
            Ok((epoch, state)) => {
                start_epoch = epoch;
                training_state = state;
                println!(
                    "Resuming from checkpoint (epoch {}, global_step {})",
                    start_epoch, training_state.global_step
                );
            }
            Err(e) => eprintln!("Resume failed: {e} — starting fresh"),
        }
    } else {
        println!("Starting fresh training run");
    }

    // ---- Move to GPU ----
    if device.is_gpu() {
        for p in model.parameters() {
            p.to_device(device.clone());
        }
    }

    // ---- Training lifecycle ----
    let lifecycle = TrainingLifecycle::builder()
        .model_name(&format!("Trident-Coder-{}", cfg.variant.label()))
        .output_dir(&cfg.output_dir)
        .param_count(param_count)
        .total_epochs(1)
        .batch_size(cfg.batch_size)
        .checkpoint_every_steps(cfg.checkpoint_every_steps)
        .keep_last_k(cfg.keep_last_k)
        .ticker(cfg.ticker)
        .start();
    println!();

    // ---- Optimizer ----
    let mut optimizer = Adam::new(model.parameters(), cfg.lr);

    println!("Training:");
    println!("  steps         : {}", cfg.steps);
    println!("  batch_size    : {}", cfg.batch_size);
    println!("  peak lr       : {}", cfg.lr);
    println!(
        "  warmup / min  : {} steps / {}× peak",
        cfg.warmup_steps, cfg.min_lr_ratio
    );
    println!("  log_every     : {}", cfg.log_every);
    println!();
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "Step", "Loss", "PPL", "LR", "Time"
    );
    println!("{}", "-".repeat(52));

    let mut best_loss = training_state.best_metric.unwrap_or(f32::MAX);
    let mut rng = cfg.seed.wrapping_add(training_state.global_step as u64);
    let global_start = Instant::now();
    let mut global_step = training_state.global_step;
    let mut running_loss = 0.0f32;
    let mut running_count = 0usize;

    lifecycle.set_epoch(1);

    let mut stopped_early = false;
    for local_step in 1..=cfg.steps {
        match lifecycle.poll() {
            LoopAction::Stop => {
                lifecycle.save_final(&model, &training_state, start_epoch + 1);
                stopped_early = true;
                break;
            }
            LoopAction::CheckpointNow => {
                lifecycle.save_step(&model, &training_state, start_epoch + 1);
            }
            LoopAction::Continue => {}
        }

        // LR schedule
        let current_lr = cosine_lr(
            global_step,
            cfg.steps + training_state.global_step, // progress across total planned steps
            cfg.warmup_steps,
            cfg.lr,
            cfg.min_lr_ratio,
        );
        optimizer.set_lr(current_lr);

        // Sample batch — labels == input_ids so causal LM shift inside the
        // model's forward_with_loss picks up next-token prediction.
        let batch = sample_batch(&tokens, cfg.batch_size, cfg.seq_len, &mut rng);
        let input_ids = Tensor::<u32>::from_vec(batch.clone(), &[cfg.batch_size, cfg.seq_len])
            .expect("input_ids shape");
        let labels =
            Tensor::<u32>::from_vec(batch, &[cfg.batch_size, cfg.seq_len]).expect("labels shape");

        // Forward + loss — uses CrossEntropyLoss built-in (graph-tracked).
        optimizer.zero_grad();
        let (_logits, loss) = model.forward_with_loss(&input_ids, &labels);
        let loss_val = loss.data().to_vec()[0];

        loss.backward();
        optimizer.step();

        global_step += 1;
        training_state.next_step();
        training_state.record_loss(loss_val);
        lifecycle.tick(global_step as u64, loss_val);
        running_loss += loss_val;
        running_count += 1;

        if lifecycle.should_step_checkpoint(global_step as u64) {
            lifecycle.save_step(&model, &training_state, start_epoch + 1);
        }

        // Track best
        if loss_val < best_loss {
            best_loss = loss_val;
            training_state.update_best("loss", loss_val, false);
        }

        if local_step % cfg.log_every == 0 || local_step == cfg.steps {
            let avg = running_loss / running_count.max(1) as f32;
            let ppl = avg.exp().min(1e9);
            let elapsed = global_start.elapsed().as_secs_f32();
            println!(
                "{:>8} {:>10.4} {:>10.2} {:>10.5} {:>9.1}s",
                global_step, avg, ppl, current_lr, elapsed,
            );
            running_loss = 0.0;
            running_count = 0;
        }
    }

    // Final checkpoint flush.
    if !stopped_early {
        lifecycle.save_epoch(&model, &training_state, start_epoch + 1);
        lifecycle.save_final(&model, &training_state, start_epoch + 1);
    } else {
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
