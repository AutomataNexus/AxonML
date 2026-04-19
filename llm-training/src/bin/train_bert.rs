//! Train BERT (Masked Language Modeling) — AxonML Shakespeare Trainer
//!
//! End-to-end training binary for the AxonML [`BertForMaskedLM`] on a text
//! corpus. BERT is the only **encoder** in AxonML's LLM suite: it uses
//! bidirectional self-attention instead of causal masking and is trained with
//! a **masked language modeling** objective — randomly replace 15% of input
//! tokens with a `[MASK]` token and predict the originals from the unmasked
//! context. This is fundamentally different from GPT-2 / LLaMA / Mistral /
//! Phi, which are causal decoders trained with next-token prediction.
//!
//! ## What this file contains
//! - `Config` struct + `Config::from_args` CLI parser and `print_help`.
//! - `apply_mlm_mask` — Devlin-et-al. 2018 masking: 80% `[MASK]`, 10% random
//!   token, 10% unchanged, with `IGNORE_INDEX` used for positions that should
//!   not contribute to the loss.
//! - `mlm_loss` — gathers only the masked positions, computes per-row
//!   [`CrossEntropyLoss`] and averages them into a scalar `Variable`.
//! - `main` — loads the corpus, builds a `CharTokenizer` + one extra
//!   `[MASK]` token id, constructs a [`BertConfig`] / [`BertForMaskedLM`],
//!   resumes from a checkpoint if available, wires up the
//!   `TrainingLifecycle`, and runs the Adam-optimized MLM training loop
//!   with per-epoch best-model tracking and perplexity reporting.
//!
//! ## MLM details (Devlin et al., 2018)
//! Of the 15% chosen positions:
//!   - 80% are replaced with `[MASK]`
//!   - 10% are replaced with a random token
//!   - 10% are left unchanged
//! Loss is only computed at the chosen positions (the "label mask").
//!
//! Usage:
//!   cargo run --release --bin train_bert -p llm-training --features cuda
//!   cargo run --release --bin train_bert -p llm-training --features cuda -- \
//!       --epochs 5 --bs 16 --seq-len 128 --mlm-prob 0.15
//!
//! # File
//! `llm-training/src/bin/train_bert.rs`
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
use axonml_llm::{BertConfig, BertForMaskedLM};
use axonml_nn::Module;
use axonml_nn::loss::CrossEntropyLoss;
use axonml_optim::{Adam, Optimizer};
use axonml_serialize::TrainingState;
use axonml_tensor::Tensor;

use llm_training::{
    CharTokenizer, LoopAction, ResumeMode, TextDataset, TrainingLifecycle, find_checkpoint,
    format_count, lcg_range, load_model_from_checkpoint, read_corpus,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_CORPUS: &str = "/opt/datasets/text/shakespeare.txt";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/llm-training/checkpoints/bert";
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_HIDDEN: usize = 192;
const DEFAULT_INTERMEDIATE: usize = 512;
const DEFAULT_NUM_LAYERS: usize = 4;
const DEFAULT_NUM_HEADS: usize = 6;
const DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_EPOCHS: usize = 3;
const DEFAULT_LR: f32 = 3e-4;
const DEFAULT_STEPS_PER_EPOCH: usize = 150;
const DEFAULT_LOG_EVERY: usize = 10;
const DEFAULT_MLM_PROB: f32 = 0.15;
const DEFAULT_SEED: u64 = 1337;
const DEFAULT_CHECKPOINT_EVERY_STEPS: u64 = 0;
const DEFAULT_KEEP_LAST_K: usize = 5;

// =============================================================================
// Config / CLI
// =============================================================================

struct Config {
    corpus: PathBuf,
    output_dir: PathBuf,
    seq_len: usize,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    num_heads: usize,
    batch_size: usize,
    epochs: usize,
    lr: f32,
    steps_per_epoch: usize,
    log_every: usize,
    mlm_prob: f32,
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
            hidden: DEFAULT_HIDDEN,
            intermediate: DEFAULT_INTERMEDIATE,
            num_layers: DEFAULT_NUM_LAYERS,
            num_heads: DEFAULT_NUM_HEADS,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
            lr: DEFAULT_LR,
            steps_per_epoch: DEFAULT_STEPS_PER_EPOCH,
            log_every: DEFAULT_LOG_EVERY,
            mlm_prob: DEFAULT_MLM_PROB,
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
                "--corpus" => {
                    i += 1;
                    cfg.corpus = PathBuf::from(&args[i]);
                }
                "--out" => {
                    i += 1;
                    cfg.output_dir = PathBuf::from(&args[i]);
                }
                "--seq-len" => {
                    i += 1;
                    cfg.seq_len = args[i].parse().unwrap();
                }
                "--hidden" | "--d-model" => {
                    i += 1;
                    cfg.hidden = args[i].parse().unwrap();
                }
                "--intermediate" | "--ffn" => {
                    i += 1;
                    cfg.intermediate = args[i].parse().unwrap();
                }
                "--layers" => {
                    i += 1;
                    cfg.num_layers = args[i].parse().unwrap();
                }
                "--heads" => {
                    i += 1;
                    cfg.num_heads = args[i].parse().unwrap();
                }
                "--bs" | "--batch-size" => {
                    i += 1;
                    cfg.batch_size = args[i].parse().unwrap();
                }
                "--epochs" => {
                    i += 1;
                    cfg.epochs = args[i].parse().unwrap();
                }
                "--lr" => {
                    i += 1;
                    cfg.lr = args[i].parse().unwrap();
                }
                "--steps" => {
                    i += 1;
                    cfg.steps_per_epoch = args[i].parse().unwrap();
                }
                "--log-every" => {
                    i += 1;
                    cfg.log_every = args[i].parse().unwrap();
                }
                "--mlm-prob" => {
                    i += 1;
                    cfg.mlm_prob = args[i].parse().unwrap();
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
        r#"Train BERT (Masked LM) on a text corpus.

Usage: train_bert [OPTIONS]

Options:
  --corpus PATH       Text corpus (default: /opt/datasets/text/shakespeare.txt)
  --out PATH          Checkpoint directory (default: .../checkpoints/bert)
  --seq-len N         Context window length (default: 128)
  --hidden N          Hidden size (default: 192)
  --intermediate N    FFN intermediate dim (default: 512)
  --layers N          Transformer blocks (default: 4)
  --heads N           Attention heads (default: 6)
  --bs N              Batch size (default: 16)
  --epochs N          Epochs (default: 3)
  --lr FLOAT          Learning rate (default: 3e-4)
  --steps N           Training steps per epoch (default: 150)
  --log-every N       Log every N steps (default: 10)
  --mlm-prob FLOAT    Fraction of tokens to mask (default: 0.15)
  --seed N            RNG seed (default: 1337)
  --resume MODE       Resume: none|latest|best|<path> (default: latest)
  --fresh             Equivalent to --resume none
  --checkpoint-every-steps N   Rotating step-level checkpoint every N steps (0 = off)
  --keep-last-k N     Keep last N step checkpoints on disk (default: 5)
  --help, -h          Show help"#
    );
}

// =============================================================================
// Masked LM batch preparation
// =============================================================================

/// Apply the MLM masking strategy (Devlin et al., 2018) to a flat token buffer.
///
/// Returns:
///   - `masked_input`: flat [B*S] Vec<u32> with masked positions replaced
///   - `labels`:       flat [B*S] Vec<u32> — original token at masked positions,
///                     `IGNORE_INDEX` elsewhere
///
/// For each position with probability `mlm_prob`:
///   - 80% → replaced with `mask_token_id`
///   - 10% → replaced with a random token in [1, vocab_size)
///   - 10% → unchanged
/// Positions that aren't chosen get the label `IGNORE_INDEX` and contribute
/// zero to the loss.
fn apply_mlm_mask(
    tokens: &[u32],
    mlm_prob: f32,
    mask_token_id: u32,
    vocab_size: u32,
    rng: &mut u64,
) -> (Vec<u32>, Vec<u32>) {
    let n = tokens.len();
    let mut masked = tokens.to_vec();
    let mut labels = vec![IGNORE_INDEX; n];
    let prob_thresh = (mlm_prob * 1_000_000.0) as usize;
    let eighty_pct = 800_000usize;
    let ninety_pct = 900_000usize;

    for i in 0..n {
        // 1M-step LCG sample → uniform in [0, 1M)
        let r = lcg_range(rng, 1_000_000);
        if r < prob_thresh {
            // This position is chosen for masking
            labels[i] = tokens[i];
            let r2 = lcg_range(rng, 1_000_000);
            if r2 < eighty_pct {
                // 80%: replace with [MASK]
                masked[i] = mask_token_id;
            } else if r2 < ninety_pct {
                // 10%: replace with a random token (avoid token 0 = pad/unk)
                masked[i] = (lcg_range(rng, (vocab_size - 1) as usize) + 1) as u32;
            }
            // else 10%: leave unchanged (masked[i] = tokens[i])
        }
    }

    (masked, labels)
}

/// Sentinel for positions that should not contribute to the MLM loss.
/// We achieve this by converting the flat label tensor into a two-vector
/// form: only indices where `labels[i] != IGNORE_INDEX` are extracted and
/// the loss is computed on that subset.
const IGNORE_INDEX: u32 = u32::MAX;

/// Compute the MLM cross-entropy loss over the masked positions only.
///
/// `logits`: [B, S, V] Variable from `BertForMaskedLM::forward_mlm`
/// `labels`: flat [B*S] Vec<u32> with `IGNORE_INDEX` for unmasked positions
///
/// Returns a scalar `Variable` — the mean cross-entropy over masked positions.
fn mlm_loss(logits: &Variable, labels: &[u32], vocab_size: usize) -> Variable {
    let logits_data = logits.data();
    let shape = logits_data.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let v = shape[2];
    debug_assert_eq!(v, vocab_size);
    debug_assert_eq!(labels.len(), batch_size * seq_len);

    // Find all positions that are actually masked.
    let mut active_indices: Vec<usize> = Vec::with_capacity(labels.len() / 6);
    let mut active_labels: Vec<f32> = Vec::with_capacity(labels.len() / 6);
    for (i, &l) in labels.iter().enumerate() {
        if l != IGNORE_INDEX {
            active_indices.push(i);
            active_labels.push(if (l as usize) < vocab_size {
                l as f32
            } else {
                0.0
            });
        }
    }

    if active_indices.is_empty() {
        // No positions masked this batch — return zero loss with a valid graph
        // by computing a dummy loss that's effectively zero.
        let zero = Tensor::from_vec(vec![0.0f32], &[1]).unwrap();
        return Variable::new(zero, false);
    }

    // Flatten logits to [B*S, V], then gather active rows via index tensor.
    let n_total = batch_size * seq_len;
    let logits_flat = logits.reshape(&[n_total, v]);

    // Build a selector tensor that we use to index into logits_flat.
    // We gather-and-reshape via a CPU-side index loop since AxonML's gather
    // on a variable is not directly exposed — instead we build the active
    // logits Variable via a narrow+cat chain (matches the approach Trident's
    // compare.rs uses for indexed cross-entropy).
    //
    // For simplicity and correctness, we compute each active row's cross-
    // entropy individually and average. This is O(num_masked) autograd ops
    // per batch, which is fine for the small char-level BERT training runs
    // we're targeting here (~16 masked positions × 15% = ~150 per batch).
    let ce = CrossEntropyLoss::new();
    let mut loss_sum: Option<Variable> = None;
    for (&idx, &lbl) in active_indices.iter().zip(active_labels.iter()) {
        let row = logits_flat.narrow(0, idx, 1); // [1, V]
        let target = Tensor::from_vec(vec![lbl], &[1]).unwrap();
        let target_var = Variable::new(target, false);
        let step_loss = ce.compute(&row, &target_var);
        loss_sum = Some(match loss_sum {
            None => step_loss,
            Some(acc) => acc.add_var(&step_loss),
        });
    }
    let loss = loss_sum.unwrap();
    loss.mul_scalar(1.0 / active_indices.len() as f32)
}

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" BERT (Masked LM) Training — AxonML on Shakespeare");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // ---- Load corpus ----
    let corpus_text = read_corpus(&cfg.corpus);
    println!(
        "Corpus: {} ({} chars)",
        cfg.corpus.display(),
        format_count(corpus_text.len())
    );

    // ---- Tokenizer + dataset ----
    // BERT needs a [MASK] token that isn't any natural character. We use the
    // CharTokenizer's reserved token 0 pattern: vocab includes '\0' at id 0,
    // and we reserve id `vocab_size` (one past the last real char) for [MASK].
    let base_tokenizer = CharTokenizer::from_corpus(&corpus_text);
    let base_vocab = base_tokenizer.vocab_size();
    let mask_token_id = base_vocab as u32; // MASK lives just past the char vocab
    let vocab_size = base_vocab + 1; // +1 for [MASK]
    println!(
        "Vocab:  {} chars + 1 [MASK] = {} tokens (mask_token_id = {})",
        base_vocab, vocab_size, mask_token_id
    );

    let dataset = TextDataset::from_string(&corpus_text, &base_tokenizer, cfg.seq_len);
    println!("Windows: {}", format_count(dataset.len()));
    println!();

    // ---- Build model ----
    let model_config = BertConfig {
        vocab_size,
        hidden_size: cfg.hidden,
        num_hidden_layers: cfg.num_layers,
        num_attention_heads: cfg.num_heads,
        intermediate_size: cfg.intermediate,
        hidden_act: "gelu".to_string(),
        hidden_dropout_prob: 0.1,
        attention_probs_dropout_prob: 0.1,
        max_position_embeddings: cfg.seq_len,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
    };
    let mut model = BertForMaskedLM::new(&model_config);
    let param_count: usize = model.parameters().iter().map(|p| p.data().numel()).sum();

    println!("Model:  BERT (encoder + masked LM head)");
    println!("  hidden   : {}", cfg.hidden);
    println!("  layers   : {}", cfg.num_layers);
    println!("  heads    : {}", cfg.num_heads);
    println!("  ffn      : {}", cfg.intermediate);
    println!("  seq_len  : {}", cfg.seq_len);
    println!("  mlm_prob : {}", cfg.mlm_prob);
    println!("  params   : {}", format_count(param_count));
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

    // ---- Training lifecycle (monitor + signals + control socket) ----
    let lifecycle = TrainingLifecycle::builder()
        .model_name("BERT (Shakespeare)")
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
    println!(
        "  epochs    : {} (starting at {})",
        cfg.epochs,
        start_epoch + 1
    );
    println!("  steps/ep  : {}", cfg.steps_per_epoch);
    println!("  lr        : {}", cfg.lr);
    println!();
    println!(
        "{:>6} {:>8} {:>10} {:>10} {:>10}",
        "Epoch", "Step", "Loss", "PPL", "Time"
    );
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

            // Sample raw token windows
            let clean_tokens = dataset.sample_batch(cfg.batch_size, &mut rng);

            // Apply MLM masking to the batch
            let (masked_tokens, labels) = apply_mlm_mask(
                &clean_tokens,
                cfg.mlm_prob,
                mask_token_id,
                vocab_size as u32,
                &mut rng,
            );

            // u32 token tensor stays on CPU — embedding layer handles CPU→GPU
            let input_ids =
                Tensor::<u32>::from_vec(masked_tokens, &[cfg.batch_size, cfg.seq_len]).unwrap();

            // Forward: masked input → logits [B, S, V]
            optimizer.zero_grad();
            let logits = model.forward_mlm(&input_ids);

            // Compute cross-entropy only at masked positions
            let loss = mlm_loss(&logits, &labels, vocab_size);
            let loss_val = loss.data().to_vec()[0];

            // Backward + step
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
        }

        // ---- End of epoch ----
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
}
