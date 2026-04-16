//! Train Phi on Shakespeare
//!
//! End-to-end training of the AxonML `PhiForCausalLM` on real text, with:
//! - Rotary position embeddings (defaults to full RoPE — see note below)
//! - Grouped-query attention (GQA)
//! - LayerNorm (Phi uses LayerNorm, not RMSNorm)
//! - Optional bias in Linear layers (Phi-1 / Phi-2 style)
//! - GPU acceleration (`--features cuda`)
//!
//! ## Partial RoPE is currently broken in the framework
//!
//! Phi's signature feature is "partial RoPE" (only the first `rotary_dim`
//! channels of each head get rotated). Running with `--partial-rotary 0.5`
//! (or any value < 1.0) panics inside `axonml-llm/src/llama.rs:426` with
//! `ShapeMismatch { expected: [B*H*S*head_dim], actual: [B,H,S,rotary_dim] }`.
//!
//! Root cause: `PhiAttention::apply_partial_rotary` calls `q.narrow(3, 0, rotary_dim)`
//! to slice the head dimension. That returns a non-contiguous view with the
//! smaller logical shape, but `RotaryEmbedding::apply` then calls
//! `x.data().to_vec()` which returns the *underlying* unsliced buffer, and
//! finally `Tensor::from_vec(output, shape)` rejects the length mismatch.
//! The fix would be for phi.rs to call `.contiguous()` on the narrowed view
//! before passing it to `rotary_emb.apply`, but that requires editing the
//! locked framework crate.
//!
//! **Workaround:** default `--partial-rotary 1.0` so we take the
//! `self.rotary_emb.apply(&q, &k, ...)` branch at phi.rs:239 (which runs on
//! the contiguous full-head tensor). This matches Phi-3-Mini's config
//! exactly, which legitimately uses `partial_rotary_factor: 1.0`.
//! - Live browser training monitor
//! - Periodic best-model + full-checkpoint saving
//! - Resume from latest / best / specific path
//! - In-flight greedy text sampling
//!
//! Usage:
//!   cargo run --release --bin train_phi -p llm-training --features cuda
//!   cargo run --release --bin train_phi -p llm-training --features cuda -- \
//!       --epochs 10 --bs 16 --seq-len 128 --partial-rotary 0.5 --resume latest
//!
//! Like LLaMA and Mistral, `PhiForCausalLM` does not expose a
//! `forward_with_loss` method, so the shifted cross-entropy is computed via
//! the shared `llm_training::shifted_cross_entropy` helper.

use std::path::PathBuf;
use std::time::Instant;

use axonml_core::Device;
use axonml_llm::{PhiConfig, PhiForCausalLM};
use axonml_nn::Module;
use axonml_optim::{Adam, Optimizer};
use axonml_serialize::TrainingState;
use axonml_tensor::Tensor;

use llm_training::{
    find_checkpoint, format_count, load_model_from_checkpoint, read_corpus, shifted_cross_entropy,
    CharTokenizer, LoopAction, ResumeMode, TextDataset, TrainingLifecycle,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_CORPUS: &str = "/opt/datasets/text/shakespeare.txt";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/llm-training/checkpoints/phi";
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_D_MODEL: usize = 192;
const DEFAULT_INTERMEDIATE: usize = 768; // 4 * d_model for GELU MLP
const DEFAULT_NUM_LAYERS: usize = 4;
const DEFAULT_NUM_HEADS: usize = 6;
const DEFAULT_NUM_KV_HEADS: usize = 2; // GQA 3:1
const DEFAULT_PARTIAL_ROTARY: f32 = 1.0; // full RoPE — Phi-3-Mini style; partial RoPE < 1.0 hits a framework bug (see module docs)
const DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_EPOCHS: usize = 5;
const DEFAULT_LR: f32 = 3e-4;
const DEFAULT_STEPS_PER_EPOCH: usize = 500;
const DEFAULT_LOG_EVERY: usize = 50;
const DEFAULT_GENERATE_EVERY: usize = 100;
const DEFAULT_SEED: u64 = 1337;
const DEFAULT_ROPE_THETA: f32 = 10000.0;
const DEFAULT_LN_EPS: f32 = 1e-5;
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
    num_kv_heads: usize,
    partial_rotary: f32,
    use_bias: bool,
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
            num_kv_heads: DEFAULT_NUM_KV_HEADS,
            partial_rotary: DEFAULT_PARTIAL_ROTARY,
            use_bias: true,
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
                "--kv-heads" => { i += 1; cfg.num_kv_heads = args[i].parse().unwrap(); }
                "--partial-rotary" => { i += 1; cfg.partial_rotary = args[i].parse().unwrap(); }
                "--no-bias" => { cfg.use_bias = false; }
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
    println!(r#"Train Phi on a text corpus.

Usage: train_phi [OPTIONS]

Options:
  --corpus PATH         Text corpus (default: /opt/datasets/text/shakespeare.txt)
  --out PATH            Checkpoint directory (default: .../checkpoints/phi)
  --seq-len N           Context window length (default: 128)
  --d-model N           Hidden dimension (default: 192)
  --intermediate N      MLP intermediate size (default: 768)
  --layers N            Transformer blocks (default: 4)
  --heads N             Attention heads (default: 6)
  --kv-heads N          KV heads for GQA; must divide --heads (default: 2)
  --partial-rotary F    RoPE fraction of head dim, 0.0..1.0 (default: 0.5)
  --no-bias             Disable bias in Linear layers (Phi-3 style)
  --bs N                Batch size (default: 16)
  --epochs N            Epochs (default: 5)
  --lr FLOAT            Learning rate (default: 3e-4)
  --steps N             Training steps per epoch (default: 500)
  --log-every N         Log every N steps (default: 50)
  --generate-every N    Generate sample every N steps (default: 100)
  --seed N              RNG seed (default: 1337)
  --resume MODE         Resume: none|latest|best|<path> (default: latest)
  --fresh               Equivalent to --resume none
  --checkpoint-every-steps N   Rotating step-level checkpoint every N steps (0 = off)
  --keep-last-k N       Keep last N step checkpoints on disk (default: 5)
  --help, -h            Show help"#);
}

// =============================================================================
// Greedy text generation
// =============================================================================

fn generate(
    model: &PhiForCausalLM,
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

        // u32 inputs stay on CPU; the embedding gather handles device crossing.
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
// Main
// =============================================================================

fn main() {
    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" Phi Training — AxonML on Shakespeare");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // ---- Sanity check GQA / head divisibility ----
    if cfg.num_heads % cfg.num_kv_heads != 0 {
        eprintln!(
            "Invalid GQA config: --heads ({}) must be divisible by --kv-heads ({})",
            cfg.num_heads, cfg.num_kv_heads
        );
        std::process::exit(1);
    }
    if cfg.d_model % cfg.num_heads != 0 {
        eprintln!(
            "Invalid head config: --d-model ({}) must be divisible by --heads ({})",
            cfg.d_model, cfg.num_heads
        );
        std::process::exit(1);
    }
    if !(0.0..=1.0).contains(&cfg.partial_rotary) {
        eprintln!(
            "Invalid --partial-rotary ({}): must be in [0.0, 1.0]",
            cfg.partial_rotary
        );
        std::process::exit(1);
    }
    if cfg.partial_rotary < 1.0 {
        eprintln!();
        eprintln!("⚠  --partial-rotary {} < 1.0 will hit a framework bug at", cfg.partial_rotary);
        eprintln!("   axonml-llm/src/llama.rs:426 (ShapeMismatch — narrow() produces a");
        eprintln!("   non-contiguous view that RoPE apply does not handle).");
        eprintln!("   See the module-level docs in this file for the full root cause.");
        eprintln!("   Proceeding anyway because you asked for it — use 1.0 if it panics.");
        eprintln!();
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
    let model_config = PhiConfig {
        vocab_size,
        hidden_size: cfg.d_model,
        intermediate_size: cfg.intermediate,
        num_hidden_layers: cfg.num_layers,
        num_attention_heads: cfg.num_heads,
        num_key_value_heads: cfg.num_kv_heads,
        max_position_embeddings: cfg.seq_len,
        layer_norm_eps: DEFAULT_LN_EPS,
        rope_theta: DEFAULT_ROPE_THETA,
        partial_rotary_factor: cfg.partial_rotary,
        attention_dropout: 0.0,
        hidden_dropout: 0.0,
        use_bias: cfg.use_bias,
    };
    let mut model = PhiForCausalLM::new(&model_config);
    let param_count: usize = model.parameters().iter().map(|p| p.data().numel()).sum();
    let head_dim = cfg.d_model / cfg.num_heads;
    let rotary_dim = (head_dim as f32 * cfg.partial_rotary) as usize;

    println!("Model:  Phi (partial RoPE + GQA + LayerNorm + GELU)");
    println!("  d_model         : {}", cfg.d_model);
    println!("  intermediate    : {}", cfg.intermediate);
    println!("  layers          : {}", cfg.num_layers);
    println!("  heads           : {}", cfg.num_heads);
    println!("  kv_heads        : {} (GQA ratio {}:1)", cfg.num_kv_heads, cfg.num_heads / cfg.num_kv_heads);
    println!("  head_dim        : {} (rotary_dim {} = {:.0}% partial RoPE)", head_dim, rotary_dim, cfg.partial_rotary * 100.0);
    println!("  use_bias        : {}", cfg.use_bias);
    println!("  seq_len         : {}", cfg.seq_len);
    println!("  params          : {}", format_count(param_count));
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
        .model_name("Phi (Shakespeare)")
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
            // Poll lifecycle: handle pause (blocks), stop, ad-hoc checkpoint.
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

            // u32 tensors stay on CPU — the embedding gather handles the
            // CPU-indices → GPU-weights crossing internally.
            let batch_data = dataset.sample_batch(cfg.batch_size, &mut rng);
            let input_ids = Tensor::<u32>::from_vec(
                batch_data.clone(),
                &[cfg.batch_size, cfg.seq_len],
            ).unwrap();
            let labels = Tensor::<u32>::from_vec(
                batch_data,
                &[cfg.batch_size, cfg.seq_len],
            ).unwrap();

            // Forward + shifted CE loss (Phi has no forward_with_loss).
            optimizer.zero_grad();
            let logits = model.forward_ids(&input_ids);
            let loss = shifted_cross_entropy(&logits, &labels);
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

            // Step-level rotating checkpoint (configurable via --checkpoint-every-steps).
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

        // ---- End of epoch ----
        let epoch_avg = epoch_loss_sum / epoch_count.max(1) as f32;
        let epoch_ppl = epoch_avg.exp().min(99999.0);
        let epoch_time = epoch_start.elapsed();

        lifecycle.log_epoch(epoch, epoch_avg, None, vec![("perplexity", epoch_ppl)]);

        // Save best (writes both best_model.axonml and checkpoint_best.axonml).
        let prev_best = best_loss;
        if lifecycle.save_if_best(&model, &training_state, epoch, epoch_avg, prev_best) {
            best_loss = epoch_avg;
            training_state.update_best("loss", epoch_avg, false);
            println!("  ★ new best loss {:.4}", epoch_avg);
        }

        // Save latest + numbered epoch checkpoints.
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

    // Final sample
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
