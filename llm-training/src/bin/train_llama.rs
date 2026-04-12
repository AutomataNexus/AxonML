//! Train LLaMA on Shakespeare
//!
//! End-to-end training of the AxonML `LLaMAForCausalLM` on real text, with:
//! - RoPE rotary positional embeddings
//! - Grouped-query attention (GQA)
//! - SwiGLU MLP with RMSNorm
//! - GPU acceleration (`--features cuda`)
//! - Live browser training monitor
//! - Periodic best-model + full-checkpoint saving
//! - Resume from latest / best / specific path
//! - In-flight text sampling to watch the model learn
//!
//! Usage:
//!   cargo run --release --bin train_llama -p llm-training --features cuda
//!   cargo run --release --bin train_llama -p llm-training --features cuda -- \
//!       --epochs 10 --bs 16 --seq-len 128 --resume latest
//!
//! Unlike GPT-2, the LLaMA crate does not expose a `forward_with_loss` method,
//! so this binary computes the shifted cross-entropy loss locally via
//! `axonml_nn::loss::CrossEntropyLoss` (same pattern GPT-2 uses internally).

use std::path::PathBuf;
use std::time::Instant;

use axonml_core::Device;
use axonml_llm::{LLaMAConfig, LLaMAForCausalLM};
use axonml_nn::loss::CrossEntropyLoss;
use axonml_nn::Module;
use axonml_optim::{Adam, Optimizer};
use axonml_serialize::{save_checkpoint, save_model, Checkpoint, StateDict, TrainingState};
use axonml_tensor::Tensor;
use axonml_autograd::Variable;

use llm_training::{
    find_checkpoint, format_count, load_model_from_checkpoint, read_corpus, CharTokenizer,
    ResumeMode, TextDataset,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_CORPUS: &str = "/opt/datasets/text/shakespeare.txt";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/llm-training/checkpoints/llama";
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_D_MODEL: usize = 192;
const DEFAULT_INTERMEDIATE: usize = 512; // ~8/3 * d_model for SwiGLU
const DEFAULT_NUM_LAYERS: usize = 4;
const DEFAULT_NUM_HEADS: usize = 6;
const DEFAULT_NUM_KV_HEADS: usize = 2; // GQA with 3:1 query:kv ratio
const DEFAULT_BATCH_SIZE: usize = 16;
const DEFAULT_EPOCHS: usize = 5;
const DEFAULT_LR: f32 = 3e-4;
const DEFAULT_STEPS_PER_EPOCH: usize = 500;
const DEFAULT_LOG_EVERY: usize = 50;
const DEFAULT_GENERATE_EVERY: usize = 100;
const DEFAULT_SEED: u64 = 1337;
const DEFAULT_ROPE_THETA: f32 = 10000.0;
const DEFAULT_RMS_EPS: f32 = 1e-5;

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
    batch_size: usize,
    epochs: usize,
    lr: f32,
    steps_per_epoch: usize,
    log_every: usize,
    generate_every: usize,
    seed: u64,
    resume: ResumeMode,
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
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
            lr: DEFAULT_LR,
            steps_per_epoch: DEFAULT_STEPS_PER_EPOCH,
            log_every: DEFAULT_LOG_EVERY,
            generate_every: DEFAULT_GENERATE_EVERY,
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
                "--corpus" => { i += 1; cfg.corpus = PathBuf::from(&args[i]); }
                "--out" => { i += 1; cfg.output_dir = PathBuf::from(&args[i]); }
                "--seq-len" => { i += 1; cfg.seq_len = args[i].parse().unwrap(); }
                "--d-model" => { i += 1; cfg.d_model = args[i].parse().unwrap(); }
                "--intermediate" => { i += 1; cfg.intermediate = args[i].parse().unwrap(); }
                "--layers" => { i += 1; cfg.num_layers = args[i].parse().unwrap(); }
                "--heads" => { i += 1; cfg.num_heads = args[i].parse().unwrap(); }
                "--kv-heads" => { i += 1; cfg.num_kv_heads = args[i].parse().unwrap(); }
                "--bs" | "--batch-size" => { i += 1; cfg.batch_size = args[i].parse().unwrap(); }
                "--epochs" => { i += 1; cfg.epochs = args[i].parse().unwrap(); }
                "--lr" => { i += 1; cfg.lr = args[i].parse().unwrap(); }
                "--steps" => { i += 1; cfg.steps_per_epoch = args[i].parse().unwrap(); }
                "--log-every" => { i += 1; cfg.log_every = args[i].parse().unwrap(); }
                "--generate-every" => { i += 1; cfg.generate_every = args[i].parse().unwrap(); }
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
    println!(r#"Train LLaMA on a text corpus.

Usage: train_llama [OPTIONS]

Options:
  --corpus PATH        Text corpus (default: /opt/datasets/text/shakespeare.txt)
  --out PATH           Checkpoint directory (default: .../checkpoints/llama)
  --seq-len N          Context window length (default: 128)
  --d-model N          Hidden dimension (default: 192)
  --intermediate N     SwiGLU intermediate size (default: 512)
  --layers N           Transformer blocks (default: 4)
  --heads N            Attention heads (default: 6)
  --kv-heads N         KV heads for GQA; must divide --heads (default: 2)
  --bs N               Batch size (default: 16)
  --epochs N           Epochs (default: 5)
  --lr FLOAT           Learning rate (default: 3e-4)
  --steps N            Training steps per epoch (default: 500)
  --log-every N        Log every N steps (default: 50)
  --generate-every N   Generate sample every N steps (default: 100)
  --seed N             RNG seed (default: 1337)
  --resume MODE        Resume: none|latest|best|<path> (default: latest)
  --fresh              Equivalent to --resume none
  --help, -h           Show help"#);
}

// =============================================================================
// Shifted cross-entropy loss for causal LM
// =============================================================================
//
// LLaMAForCausalLM exposes `forward_ids` but no `forward_with_loss`, so we
// replicate the shift-then-CE pattern GPT-2 uses internally (see
// axonml-llm/src/gpt2.rs `forward_with_loss`). The loss is computed on CPU or
// GPU depending on where the logits live — `CrossEntropyLoss::compute` has a
// fused GPU fast path and a CPU fallback.

fn shifted_cross_entropy(
    logits: &Variable,
    labels: &Tensor<u32>,
) -> Variable {
    let logits_data = logits.data();
    let shape = logits_data.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let vocab_size = shape[2];

    if seq_len <= 1 {
        // Degenerate case — return zero loss on the same device as the logits.
        let zero = Tensor::from_vec(vec![0.0f32], &[1]).unwrap();
        return Variable::new(zero, false);
    }

    // Shift logits: drop the last position → predict positions 1..S from 0..S-1
    let shift_logits = logits.narrow(1, 0, seq_len - 1);
    let n = batch_size * (seq_len - 1);
    let logits_flat = shift_logits.reshape(&[n, vocab_size]);

    // Shift labels: drop position 0, keep positions 1..S, flatten to [N].
    let labels_vec = labels.to_vec();
    let mut shift_labels = Vec::with_capacity(n);
    for b in 0..batch_size {
        for s in 1..seq_len {
            // Clamp any out-of-range labels to 0 defensively (matches gpt2.rs).
            let l = labels_vec[b * seq_len + s] as usize;
            shift_labels.push(if l < vocab_size { l as f32 } else { 0.0 });
        }
    }
    let mut target_tensor = Tensor::from_vec(shift_labels, &[n]).unwrap();
    // Move targets to the logits' device so the GPU fast path triggers.
    let logits_device = logits_data.device();
    if logits_device.is_gpu() {
        target_tensor = target_tensor.to_device(logits_device).unwrap();
    }
    let target_var = Variable::new(target_tensor, false);

    CrossEntropyLoss::new().compute(&logits_flat, &target_var)
}

// =============================================================================
// Greedy text generation
// =============================================================================

fn generate(
    model: &LLaMAForCausalLM,
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
    println!(" LLaMA Training — AxonML on Shakespeare");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // ---- Sanity check GQA ratio ----
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
    let model_config = LLaMAConfig {
        vocab_size,
        hidden_size: cfg.d_model,
        intermediate_size: cfg.intermediate,
        num_hidden_layers: cfg.num_layers,
        num_attention_heads: cfg.num_heads,
        num_key_value_heads: cfg.num_kv_heads,
        max_position_embeddings: cfg.seq_len,
        rms_norm_eps: DEFAULT_RMS_EPS,
        rope_theta: DEFAULT_ROPE_THETA,
        attention_dropout: 0.0,
        hidden_dropout: 0.0,
    };
    let mut model = LLaMAForCausalLM::new(&model_config);
    let param_count: usize = model.parameters().iter().map(|p| p.data().numel()).sum();

    println!("Model:  LLaMA (RoPE + GQA + SwiGLU + RMSNorm)");
    println!("  d_model       : {}", cfg.d_model);
    println!("  intermediate  : {}", cfg.intermediate);
    println!("  layers        : {}", cfg.num_layers);
    println!("  heads         : {}", cfg.num_heads);
    println!("  kv_heads      : {} (GQA ratio {}:1)", cfg.num_kv_heads, cfg.num_heads / cfg.num_kv_heads);
    println!("  seq_len       : {}", cfg.seq_len);
    println!("  params        : {}", format_count(param_count));
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

    // ---- Launch training monitor ----
    let monitor = axonml::TrainingMonitor::new("LLaMA (Shakespeare)", param_count)
        .total_epochs(cfg.epochs)
        .batch_size(cfg.batch_size)
        .launch();
    println!("Monitor: http://127.0.0.1:{}", monitor.port());
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

    for epoch in (start_epoch + 1)..=cfg.epochs {
        model.train();
        let epoch_start = Instant::now();
        let mut running_loss = 0.0f32;
        let mut running_count = 0usize;
        let mut epoch_loss_sum = 0.0f32;
        let mut epoch_count = 0usize;

        for step in 1..=cfg.steps_per_epoch {
            // Sample batch
            let batch_data = dataset.sample_batch(cfg.batch_size, &mut rng);

            // u32 tensors stay on CPU — `Embedding.lookup` handles the CPU-indices →
            // GPU-weights gather internally via `embedding_gather_cuda`. Moving a
            // `Tensor<u32>` to GPU panics with "GPU tensors are only supported for f32".
            let input_ids = Tensor::<u32>::from_vec(
                batch_data.clone(),
                &[cfg.batch_size, cfg.seq_len],
            ).unwrap();
            let labels = Tensor::<u32>::from_vec(
                batch_data,
                &[cfg.batch_size, cfg.seq_len],
            ).unwrap();

            // Forward + loss (locally computed since LLaMA has no forward_with_loss)
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

        monitor.log_epoch(
            epoch,
            epoch_avg,
            None,
            vec![("perplexity", epoch_ppl)],
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
            // Also save full best checkpoint so we can resume from it
            let best_ckpt = cfg.output_dir.join("checkpoint_best.axonml");
            let cp = Checkpoint::builder()
                .model_state(StateDict::from_module(&model))
                .training_state(training_state.clone())
                .epoch(epoch)
                .build();
            save_checkpoint(&cp, &best_ckpt).ok();
        }

        // Always save latest (so --resume latest always works)
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
        let epoch_ckpt = cfg.output_dir.join(format!("checkpoint_epoch_{epoch:04}.axonml"));
        let cp = Checkpoint::builder()
            .model_state(StateDict::from_module(&model))
            .training_state(training_state.clone())
            .epoch(epoch)
            .build();
        save_checkpoint(&cp, &epoch_ckpt).ok();

        println!(
            "  epoch {} done in {:.1}s | loss {:.4} | ppl {:.2}",
            epoch,
            epoch_time.as_secs_f32(),
            epoch_avg,
            epoch_ppl,
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
