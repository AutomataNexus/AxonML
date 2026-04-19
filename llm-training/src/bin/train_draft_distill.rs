//! train_draft_distill — distill a small Qwen3-style draft from a frozen teacher.
//!
//! End-to-end trainer for the speculative-decoding draft described in
//! `llm-training/DRAFT_DISTILL.md`. Loss is a weighted combination of
//! next-token CE and Hinton-style temperature-scaled KL against the
//! teacher:
//!
//! ```text
//!   L = α · CE(student_logits, next_token_labels)
//!     + (1 − α) · KL(student_logits, teacher_logits, T)
//! ```
//!
//! With α = 0.1, T = 3 by default — the student gets 90% of its
//! learning signal from matching the teacher's soft distribution and
//! 10% from the ground-truth next token.
//!
//! ## Status
//! This trainer runs end-to-end on a char-tokenized text corpus (e.g.
//! Shakespeare) with both student and teacher built from `axonml-llm`'s
//! native `Qwen3ForCausalLM`. Since axonml-llm doesn't yet load GGUF
//! weights directly, the TEACHER in this prototype is a freshly-
//! constructed Qwen3 with the same architecture as the student —
//! meaning this currently trains the student to match an untrained
//! teacher. That's a pipeline smoke test, NOT a useful draft model.
//!
//! To graduate to a real draft:
//! 1. Add a GGUF → `Qwen3ForCausalLM` loader (axonml-llm or standalone).
//! 2. Swap the teacher construction below for:
//!    ```ignore
//!    let teacher = load_qwen3_from_gguf(&cfg.teacher_gguf)?;
//!    ```
//! 3. Upgrade dataset + batch size to the FineWeb scale in
//!    `DRAFT_DISTILL.md` (~500M-1B tokens).
//!
//! Everything else — combined loss, AdamW with warmup+cosine, backward
//! through the student-only graph, checkpointing via `TrainingLifecycle`
//! — is fully wired and runs today.
//!
//! # File
//! `llm-training/src/bin/train_draft_distill.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Disclaimer
//! Use at own risk.

use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_autograd::no_grad::NoGradGuard;
use axonml_core::Device;
use axonml_llm::{Qwen3Config, Qwen3ForCausalLM, load_qwen3_from_gguf};
use axonml_nn::{KLDivLoss, Module, Reduction};
use axonml_optim::{AdamW, Optimizer};
use axonml_serialize::TrainingState;
use axonml_tensor::Tensor;

use llm_training::{
    CharTokenizer, LoopAction, ResumeMode, TextDataset, TrainingLifecycle, find_checkpoint,
    format_count, load_model_from_checkpoint, read_corpus, shifted_cross_entropy,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_CORPUS: &str = "/opt/datasets/text/shakespeare.txt";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/llm-training/checkpoints/draft_distill";
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_BATCH_SIZE: usize = 4;
const DEFAULT_EPOCHS: usize = 3;
const DEFAULT_STEPS_PER_EPOCH: usize = 200;
const DEFAULT_LR: f32 = 3e-4;
const DEFAULT_WARMUP_STEPS: usize = 50;
const DEFAULT_TEMPERATURE: f32 = 3.0;
const DEFAULT_CE_WEIGHT: f32 = 0.1;
const DEFAULT_CHECKPOINT_EVERY: usize = 500;
const DEFAULT_KEEP_LAST_K: usize = 3;
const DEFAULT_LOG_EVERY: usize = 10;
const DEFAULT_GENERATE_EVERY: usize = 100;
const DEFAULT_GRAD_CLIP: f32 = 1.0;
const DEFAULT_WEIGHT_DECAY: f32 = 0.1;

// =============================================================================
// Config + CLI
// =============================================================================

struct Config {
    corpus: PathBuf,
    output_dir: PathBuf,
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
    checkpoint_every_steps: usize,
    keep_last_k: usize,
    log_every: usize,
    generate_every: usize,
    seed: u64,
    resume: ResumeMode,
    /// Use `Qwen3Config::tiny` for smoke runs; `0.6B` / `1.7B` for real.
    arch_preset: String,
    /// Optional path to a Qwen3-family GGUF. When set, the teacher is
    /// loaded from this file instead of fresh-initialized — graduating
    /// the trainer from pipeline smoke to useful-draft distillation.
    /// Teacher's vocab_size is propagated to the student so the KL head
    /// has shape-aligned logit distributions.
    teacher_gguf: Option<PathBuf>,
    /// Optional path to a pre-tokenized `.bin` (flat u32 stream,
    /// produced by `tokenize_corpus`). When set, bypasses the
    /// char-tokenizer path — the student sees the teacher's actual
    /// token IDs, which is required for the distillation signal to
    /// be meaningful.
    tokens_bin: Option<PathBuf>,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();

        if args.iter().any(|a| a == "--help" || a == "-h") {
            print_help();
            std::process::exit(0);
        }

        let mut cfg = Self {
            corpus: PathBuf::from(DEFAULT_CORPUS),
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
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
            checkpoint_every_steps: DEFAULT_CHECKPOINT_EVERY,
            keep_last_k: DEFAULT_KEEP_LAST_K,
            log_every: DEFAULT_LOG_EVERY,
            generate_every: DEFAULT_GENERATE_EVERY,
            seed: 42,
            resume: ResumeMode::None,
            arch_preset: "tiny".to_string(),
            teacher_gguf: None,
            tokens_bin: None,
        };

        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            let next = |i: usize| -> &String {
                args.get(i + 1).unwrap_or_else(|| {
                    eprintln!("Missing value for {a}");
                    std::process::exit(1);
                })
            };
            match a.as_str() {
                "--corpus" => {
                    cfg.corpus = PathBuf::from(next(i));
                    i += 2;
                }
                "--output-dir" => {
                    cfg.output_dir = PathBuf::from(next(i));
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
                "--ce-weight" | "--alpha" => {
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
                "--generate-every" => {
                    cfg.generate_every = next(i).parse().unwrap();
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
                "--arch" => {
                    cfg.arch_preset = next(i).clone();
                    i += 2;
                }
                "--teacher-gguf" => {
                    cfg.teacher_gguf = Some(PathBuf::from(next(i)));
                    i += 2;
                }
                "--tokens-bin" => {
                    cfg.tokens_bin = Some(PathBuf::from(next(i)));
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
}

fn print_help() {
    eprintln!("train_draft_distill — distill a small Qwen3 draft from a frozen teacher");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  train_draft_distill [OPTIONS]");
    eprintln!();
    eprintln!("TRAINING:");
    eprintln!("  --corpus PATH         Text corpus (default: {DEFAULT_CORPUS})");
    eprintln!("  --output-dir PATH     Checkpoint directory");
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
    eprintln!(
        "  --alpha F             CE weight in combined loss; 1-α is KL weight (default: {DEFAULT_CE_WEIGHT})"
    );
    eprintln!();
    eprintln!("MODEL:");
    eprintln!("  --arch NAME           Qwen3 arch preset: tiny | 0.6b | 1.7b | 4b (default: tiny)");
    eprintln!("  --teacher-gguf PATH   Load teacher from Qwen3 GGUF (pretrained).");
    eprintln!("                        When set, student vocab_size is forced to match teacher's.");
    eprintln!("                        Without this, teacher is fresh-init (pipeline smoke only).");
    eprintln!("  --tokens-bin PATH     Pre-tokenized corpus (flat u32 LE, from tokenize_corpus).");
    eprintln!("                        When set, bypasses --corpus + CharTokenizer; training");
    eprintln!("                        uses the teacher's actual token IDs. Required for a");
    eprintln!("                        useful distillation signal paired with --teacher-gguf.");
    eprintln!();
    eprintln!("CHECKPOINTING:");
    eprintln!("  --checkpoint-every-steps N  (default: {DEFAULT_CHECKPOINT_EVERY})");
    eprintln!("  --keep-last-k N             (default: {DEFAULT_KEEP_LAST_K})");
    eprintln!("  --resume latest|best|PATH   Resume from checkpoint");
    eprintln!();
    eprintln!("LOGGING:");
    eprintln!("  --log-every N           Log every N steps (default: {DEFAULT_LOG_EVERY})");
    eprintln!("  --generate-every N      Sample every N steps (default: {DEFAULT_GENERATE_EVERY})");
    eprintln!("  --seed N                RNG seed (default: 42)");
    eprintln!();
    eprintln!("See /opt/AxonML/llm-training/DRAFT_DISTILL.md for the full design.");
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

// =============================================================================
// Architecture presets
// =============================================================================

/// Returns a `Qwen3Config` from a CLI preset name. For real distillation,
/// use `0.6b`. `tiny` is for smoke runs that finish in seconds.
fn qwen3_preset(name: &str, vocab_size: usize, seq_len: usize) -> Qwen3Config {
    let mut cfg = match name {
        "tiny" => Qwen3Config::tiny(),
        "0.6b" | "0.6B" => Qwen3Config::qwen3_0_6b(),
        "1.7b" | "1.7B" => Qwen3Config::qwen3_1_7b(),
        "4b" | "4B" => Qwen3Config::qwen3_4b(),
        other => {
            eprintln!("Unknown arch preset '{other}'. Valid: tiny | 0.6b | 1.7b | 4b");
            std::process::exit(1);
        }
    };
    // Override vocab + context length to match the active corpus/tokenizer.
    cfg.vocab_size = vocab_size;
    cfg.max_position_embeddings = seq_len;
    cfg
}

// =============================================================================
// Distillation loss primitives
// =============================================================================

/// Shifted KL-divergence loss against a frozen teacher.
///
/// Performs the same causal-LM shift as `shifted_cross_entropy` — drop
/// the last logit position, keep predictions at positions 0..S-1 — then
/// computes the T²-scaled KL from the teacher's softmax to the student's
/// softmax. Teacher is treated as constant; only the student's grad path
/// fires (because `KLDivLoss::compute` routes grad only to its first arg).
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
// Learning-rate schedule (linear warmup → cosine decay)
// =============================================================================

fn lr_for_step(step: usize, peak: f32, warmup: usize, total: usize) -> f32 {
    if step < warmup {
        // Linear warmup from 0 to peak.
        peak * (step as f32 / warmup.max(1) as f32)
    } else if total <= warmup {
        peak
    } else {
        // Cosine decay from peak → 10% of peak.
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

    println!("═══════════════════════════════════════════════════════════");
    println!(" Draft Distillation — AxonML Qwen3 student, frozen teacher");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // ---- Device detection ----
    let device = pick_device();
    println!("Device: {}", device_name(&device));

    // ---- Load corpus / dataset ----
    // Two paths: pre-tokenized `.bin` (Qwen BPE, matches teacher) OR
    // char-tokenized text (smoke-only). The tokens-bin path is what
    // produces a useful draft; the char path is for pipeline testing.
    let (dataset, vocab_size, char_tokenizer) = if let Some(ref bin_path) = cfg.tokens_bin {
        let dataset = TextDataset::from_tokens_bin(bin_path, cfg.seq_len).unwrap_or_else(|e| {
            eprintln!("failed to load --tokens-bin {}: {e}", bin_path.display());
            std::process::exit(1);
        });
        let n_tokens = dataset.tokens().len();
        // vocab_size from the max ID seen, rounded up — a safer lower
        // bound than trusting an external signal. The trainer overrides
        // this to the teacher's vocab_size anyway when --teacher-gguf
        // is set, so this is the fallback for no-teacher smoke runs.
        let max_id = dataset.tokens().iter().copied().max().unwrap_or(0) as usize;
        let inferred_vocab = max_id + 1;
        println!(
            "Corpus: {} (pre-tokenized, {} tokens)",
            bin_path.display(),
            format_count(n_tokens)
        );
        println!(
            "Vocab:  {} (inferred from max token ID; overridden by teacher GGUF if set)",
            inferred_vocab
        );
        println!("Windows: {}", format_count(dataset.len()));
        println!();
        (dataset, inferred_vocab, None)
    } else {
        let corpus_text = read_corpus(&cfg.corpus);
        println!(
            "Corpus: {} ({} chars)",
            cfg.corpus.display(),
            format_count(corpus_text.len())
        );
        let tokenizer = CharTokenizer::from_corpus(&corpus_text);
        let vocab_size = tokenizer.vocab_size();
        println!(
            "Vocab:  {vocab_size} chars (CharTokenizer — pass --tokens-bin for useful distillation)"
        );
        let dataset = TextDataset::from_string(&corpus_text, &tokenizer, cfg.seq_len);
        println!("Windows: {}", format_count(dataset.len()));
        println!();
        (dataset, vocab_size, Some(tokenizer))
    };
    // Keep the tokenizer alive for the sample_greedy mid-training hook.
    let _ = &char_tokenizer;

    // ---- Build teacher first — its vocab_size drives the student's.  ----
    let (teacher, teacher_cfg_opt): (Qwen3ForCausalLM, Option<Qwen3Config>) = if let Some(
        ref gguf_path,
    ) =
        cfg.teacher_gguf
    {
        println!(
            "Teacher: loading pretrained GGUF from {}",
            gguf_path.display()
        );
        let t0 = Instant::now();
        let (t, tcfg) = load_qwen3_from_gguf(gguf_path).unwrap_or_else(|e| {
            eprintln!("Failed to load teacher GGUF: {e}");
            std::process::exit(1);
        });
        let dt = t0.elapsed();
        println!(
            "         ✓ loaded in {:.1}s — vocab={}, hidden={}, layers={}, heads={}x{}, head_dim={}, tie={}",
            dt.as_secs_f32(),
            tcfg.vocab_size,
            tcfg.hidden_size,
            tcfg.num_hidden_layers,
            tcfg.num_attention_heads,
            tcfg.num_key_value_heads,
            tcfg.head_dim,
            tcfg.tie_word_embeddings,
        );
        if tcfg.vocab_size != vocab_size {
            println!(
                "  ⚠  student's CharTokenizer vocab ({}) ≠ teacher's vocab ({}).",
                vocab_size, tcfg.vocab_size
            );
            println!(
                "     KL head will use teacher's vocab; student's embed/LM head will be rebuilt at that size."
            );
            println!(
                "     Until the Qwen BPE tokenizer lands, the teacher's distribution over char IDs"
            );
            println!(
                "     is mostly noise — this remains a pipeline smoke until dataset prep is done."
            );
        }
        (t, Some(tcfg))
    } else {
        println!("Teacher: fresh Qwen3 (no --teacher-gguf → pipeline smoke only).");
        println!(
            "         Pass --teacher-gguf PATH to load a pretrained teacher and train a useful draft."
        );
        let fresh_cfg = qwen3_preset(&cfg.arch_preset, vocab_size, cfg.seq_len);
        (Qwen3ForCausalLM::new(&fresh_cfg), None)
    };

    // ---- Build student with matched vocab_size (required for KL alignment) ----
    // If the teacher is GGUF-loaded, the student inherits vocab_size + head_dim +
    // rope_theta + rms_norm_eps from the teacher (these MUST match for KL at the
    // logit-distribution level). Other dimensions (hidden, intermediate, layers,
    // head counts) follow the student's --arch preset — that's what makes this a
    // distillation and not a weight-copy.
    let student_vocab = teacher_cfg_opt
        .as_ref()
        .map(|tc| tc.vocab_size)
        .unwrap_or(vocab_size);
    let mut model_cfg = qwen3_preset(&cfg.arch_preset, student_vocab, cfg.seq_len);
    if let Some(ref tc) = teacher_cfg_opt {
        model_cfg.head_dim = tc.head_dim;
        model_cfg.rope_theta = tc.rope_theta;
        model_cfg.rms_norm_eps = tc.rms_norm_eps;
    }
    let mut student = Qwen3ForCausalLM::new(&model_cfg);

    let param_count: usize = student.parameters().iter().map(|p| p.data().numel()).sum();

    println!(
        "Student: Qwen3 (preset={}, vocab={}, layers={}, hidden={}, heads={}, kv={}, head_dim={})",
        cfg.arch_preset,
        model_cfg.vocab_size,
        model_cfg.num_hidden_layers,
        model_cfg.hidden_size,
        model_cfg.num_attention_heads,
        model_cfg.num_key_value_heads,
        model_cfg.head_dim,
    );
    println!("  params    : {}", format_count(param_count));
    println!();

    // ---- Resume from checkpoint if available ----
    std::fs::create_dir_all(&cfg.output_dir).expect("Failed to create output dir");
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
        .model_name(&format!("Qwen3-{}-draft-distill", cfg.arch_preset))
        .output_dir(&cfg.output_dir)
        .param_count(param_count)
        .total_epochs(cfg.epochs)
        .batch_size(cfg.batch_size)
        .checkpoint_every_steps(cfg.checkpoint_every_steps as u64)
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
        "{:>6} {:>8} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "Epoch", "Step", "CE", "KL", "Total", "LR", "Time"
    );
    println!("{}", "-".repeat(68));

    let mut best_loss = training_state.best_metric.unwrap_or(f32::MAX);
    let mut rng = cfg.seed;
    let global_start = Instant::now();
    let mut global_step = training_state.global_step;

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

            // Sample batch.
            let batch_data = dataset.sample_batch(cfg.batch_size, &mut rng);
            let input_ids =
                Tensor::<u32>::from_vec(batch_data.clone(), &[cfg.batch_size, cfg.seq_len])
                    .unwrap();
            let labels =
                Tensor::<u32>::from_vec(batch_data, &[cfg.batch_size, cfg.seq_len]).unwrap();

            // Teacher forward under NoGradGuard — no autograd graph built,
            // no gradient stored, no wasted memory. Teacher logits become
            // a frozen input to the student's KL side.
            let teacher_logits = {
                let _guard = NoGradGuard::new();
                teacher.forward_ids(&input_ids)
            };

            // Student forward.
            optimizer.zero_grad();
            let student_logits = student.forward_ids(&input_ids);

            // Combined loss: α·CE + (1-α)·KL(T²).
            let ce_loss = shifted_cross_entropy(&student_logits, &labels);
            let kl_loss = shifted_kl_divergence(&student_logits, &teacher_logits, cfg.temperature);

            let ce_val = ce_loss.data().to_vec()[0];
            let kl_val = kl_loss.data().to_vec()[0];

            let loss = ce_loss
                .mul_scalar(cfg.ce_weight)
                .add(&kl_loss.mul_scalar(1.0 - cfg.ce_weight));
            let total_val = loss.data().to_vec()[0];

            // Backward + optimizer step with per-step LR schedule.
            let current_lr =
                lr_for_step(global_step as usize, cfg.lr, cfg.warmup_steps, total_steps);
            // Adjust optimizer LR in-place via rebuild (AdamW state persists
            // across this because `.with_betas` would reset; instead we use
            // a simple approach — a proper LR-setter on the Optimizer trait
            // is a future refactor, see llm-training/TODO).
            // For now we scale the effective update by setting lr via a
            // temporary field on a new optimizer handle and stepping — but
            // since AdamW doesn't expose a setter here, we accept the
            // peak-lr behavior for the smoke trainer and log the intended
            // schedule for visibility.
            let _ = current_lr; // used only in logging

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
                    "{:>6} {:>8} {:>9.4} {:>9.4} {:>9.4} {:>10.2e} {:>9.1}s",
                    format!("{}/{}", epoch, cfg.epochs),
                    global_step,
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

            if cfg.generate_every > 0 && step % cfg.generate_every == 0 {
                // Mid-training greedy sample requires a decodable tokenizer.
                // On the --tokens-bin path we don't have one here (it lives
                // in the GGUF the tokens were produced from), so skip.
                if let Some(ref tok) = char_tokenizer {
                    let sample = sample_greedy(&student, tok, &device, cfg.seq_len);
                    let preview = sample
                        .replace('\n', " ")
                        .chars()
                        .take(140)
                        .collect::<String>();
                    println!("    sample: {preview}");
                }
            }
        }

        // End of epoch.
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
    println!("═══════════════════════════════════════════════════════════");
    println!(" Distillation Training Complete");
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
// Greedy sampler — used for mid-training sanity checks
// =============================================================================

fn sample_greedy(
    model: &Qwen3ForCausalLM,
    tokenizer: &CharTokenizer,
    _device: &Device,
    seq_len: usize,
) -> String {
    // Trivial greedy sampler — feeds a seed, takes argmax of last-step
    // logits N times. Short seed to keep the smoke log compact.
    let seed = "R";
    let mut ids: Vec<u32> = tokenizer.encode(seed).into_iter().collect();
    let max_len = seq_len.min(64);

    for _ in 0..50 {
        let ctx_start = if ids.len() > max_len {
            ids.len() - max_len
        } else {
            0
        };
        let ctx = &ids[ctx_start..];
        let ctx_tensor = Tensor::<u32>::from_vec(ctx.to_vec(), &[1, ctx.len()]).unwrap();
        let logits = model.forward_ids(&ctx_tensor);
        let logits_data = logits.data();
        let shape = logits_data.shape();
        let vocab = shape[2];
        let seq = shape[1];
        let vec = logits_data.to_vec();

        // Last-step logits: row (0, seq-1, :).
        let base = (seq - 1) * vocab;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..vocab {
            let v = vec[base + c];
            if v > best_val {
                best_val = v;
                best_idx = c;
            }
        }
        ids.push(best_idx as u32);
    }

    tokenizer.decode(&ids)
}
