//! train_rdt — train a Recurrent-Depth Transformer on a tokenized corpus.
//!
//! End-to-end trainer for the axonml-llm `RDTForCausalLM` module. The
//! RDT-specific twist: **K (number of core iterations) is sampled
//! uniformly from `[k_min, k_max]` per batch** so the trained model
//! generalizes across test-time iteration counts. Fixed-K training
//! degrades badly when K varies at inference.
//!
//! Loss: standard next-token cross-entropy via `shifted_cross_entropy`.
//! Full unroll of the K core iterations through autograd in v1 — memory
//! scales linearly with K, so at K_max=16 + seq=1024 + rdt-small you'll
//! want bs=2 and gradient accumulation (planned). Smoke runs (tiny arch,
//! K_max=8, seq=128) fit on any GPU.
//!
//! ## Usage
//!
//! ```bash
//! # Smoke (char-tokenized Shakespeare, rdt-tiny):
//! train_rdt --arch tiny --corpus /opt/datasets/text/shakespeare.txt \
//!   --seq-len 128 --bs 4 --steps 200
//!
//! # Real run (Oracle corpus, pre-tokenized via tokenize_corpus):
//! train_rdt --arch small --tokens-bin /opt/datasets/oracle.qwen.bin \
//!   --seq-len 512 --bs 2 --steps 500 --k-min 4 --k-max 12
//! ```
//!
//! Design doc: `/opt/AxonML/docs/RDT_DESIGN.md`.
//!
//! # File
//! `llm-training/src/bin/train_rdt.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use std::path::PathBuf;
use std::time::Instant;

use axonml_core::Device;
use axonml_llm::{RDTConfig, RDTForCausalLM};
use axonml_optim::{AdamW, Optimizer};
use axonml_serialize::TrainingState;
use axonml_tensor::Tensor;

use llm_training::{
    CharTokenizer, LoopAction, TextDataset, TrainingLifecycle, format_count, lcg_range,
    read_corpus, shifted_cross_entropy,
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULT_CORPUS: &str = "/opt/datasets/text/shakespeare.txt";
const DEFAULT_OUTPUT_DIR: &str = "/opt/AxonML/llm-training/checkpoints/rdt";
const DEFAULT_ARCH: &str = "tiny";
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_BATCH_SIZE: usize = 4;
const DEFAULT_EPOCHS: usize = 3;
const DEFAULT_STEPS_PER_EPOCH: usize = 200;
const DEFAULT_LR: f32 = 3e-4;
const DEFAULT_WEIGHT_DECAY: f32 = 0.1;
const DEFAULT_CHECKPOINT_EVERY: u64 = 500;
const DEFAULT_KEEP_LAST_K: usize = 3;

// =============================================================================
// CLI
// =============================================================================

struct TrainConfig {
    corpus: PathBuf,
    tokens_bin: Option<PathBuf>,
    output_dir: PathBuf,
    arch: String,
    seq_len: usize,
    batch_size: usize,
    epochs: usize,
    steps_per_epoch: usize,
    lr: f32,
    weight_decay: f32,
    k_min: Option<usize>,
    k_max: Option<usize>,
    checkpoint_every_steps: u64,
    keep_last_k: usize,
    seed: u64,
}

impl TrainConfig {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--help" || a == "-h") {
            print_help();
            std::process::exit(0);
        }
        let mut cfg = Self {
            corpus: PathBuf::from(DEFAULT_CORPUS),
            tokens_bin: None,
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            arch: DEFAULT_ARCH.to_string(),
            seq_len: DEFAULT_SEQ_LEN,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs: DEFAULT_EPOCHS,
            steps_per_epoch: DEFAULT_STEPS_PER_EPOCH,
            lr: DEFAULT_LR,
            weight_decay: DEFAULT_WEIGHT_DECAY,
            k_min: None,
            k_max: None,
            checkpoint_every_steps: DEFAULT_CHECKPOINT_EVERY,
            keep_last_k: DEFAULT_KEEP_LAST_K,
            seed: 42,
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
                "--corpus" => { cfg.corpus = PathBuf::from(next(i)); i += 2; }
                "--tokens-bin" => { cfg.tokens_bin = Some(PathBuf::from(next(i))); i += 2; }
                "--output-dir" => { cfg.output_dir = PathBuf::from(next(i)); i += 2; }
                "--arch" => { cfg.arch = next(i); i += 2; }
                "--seq-len" => { cfg.seq_len = next(i).parse().unwrap(); i += 2; }
                "--bs" | "--batch-size" => { cfg.batch_size = next(i).parse().unwrap(); i += 2; }
                "--epochs" => { cfg.epochs = next(i).parse().unwrap(); i += 2; }
                "--steps" => { cfg.steps_per_epoch = next(i).parse().unwrap(); i += 2; }
                "--lr" => { cfg.lr = next(i).parse().unwrap(); i += 2; }
                "--weight-decay" => { cfg.weight_decay = next(i).parse().unwrap(); i += 2; }
                "--k-min" => { cfg.k_min = Some(next(i).parse().unwrap()); i += 2; }
                "--k-max" => { cfg.k_max = Some(next(i).parse().unwrap()); i += 2; }
                "--checkpoint-every-steps" => {
                    cfg.checkpoint_every_steps = next(i).parse().unwrap();
                    i += 2;
                }
                "--keep-last-k" => { cfg.keep_last_k = next(i).parse().unwrap(); i += 2; }
                "--seed" => { cfg.seed = next(i).parse().unwrap(); i += 2; }
                _ => {
                    eprintln!("Unknown arg: {a}");
                    print_help();
                    std::process::exit(1);
                }
            }
        }
        cfg
    }

    fn rdt_config(&self) -> RDTConfig {
        let mut c = match self.arch.as_str() {
            "tiny" => RDTConfig::rdt_tiny(),
            "small" => RDTConfig::rdt_small(),
            "mid" => RDTConfig::rdt_mid(),
            other => {
                eprintln!("Unknown --arch '{other}'. Use: tiny | small | mid");
                std::process::exit(1);
            }
        };
        if let Some(k) = self.k_min { c.k_min = k; }
        if let Some(k) = self.k_max { c.k_max = k; }
        assert!(c.k_min <= c.k_max, "k_min must be <= k_max");
        c
    }
}

fn print_help() {
    println!(
        "train_rdt — Recurrent-Depth Transformer trainer\n\n\
         ARCH:          --arch tiny|small|mid\n\
                        --k-min N   --k-max N   (override preset sampling range)\n\
         DATA:          --corpus PATH  |  --tokens-bin PATH\n\
         TRAIN:         --seq-len N --bs N --epochs N --steps N --lr F\n\
                        --weight-decay F --seed N\n\
         CHECKPOINT:    --output-dir PATH --checkpoint-every-steps N --keep-last-k N\n"
    );
}

// =============================================================================
// Device
// =============================================================================

#[cfg(feature = "cuda")]
fn pick_device() -> Device {
    if axonml_core::backends::cuda::is_available() { Device::Cuda(0) } else { Device::Cpu }
}

#[cfg(not(feature = "cuda"))]
fn pick_device() -> Device { Device::Cpu }

// =============================================================================
// Main
// =============================================================================

fn main() {
    let cfg = TrainConfig::from_args();
    std::fs::create_dir_all(&cfg.output_dir).expect("create output_dir");

    let device = pick_device();
    println!("device: {device:?}");

    // Load dataset. Two paths: pre-tokenized bin (Qwen BPE) OR
    // char-tokenized text corpus (smoke only).
    let (dataset, vocab_size) = if let Some(ref bin) = cfg.tokens_bin {
        let ds = TextDataset::from_tokens_bin(bin, cfg.seq_len)
            .unwrap_or_else(|e| panic!("load tokens bin {}: {e}", bin.display()));
        let max_id = ds.tokens().iter().copied().max().unwrap_or(0) as usize;
        (ds, max_id + 1)
    } else {
        let corpus = read_corpus(&cfg.corpus);
        let tok = CharTokenizer::from_corpus(&corpus);
        let vs = tok.vocab_size();
        let ds = TextDataset::from_string(&corpus, &tok, cfg.seq_len);
        (ds, vs)
    };

    // Build model. Override preset vocab_size to match the data.
    let mut rdt_cfg = cfg.rdt_config();
    rdt_cfg.base.vocab_size = vocab_size;

    println!(
        "=== RDT trainer ===\n\
         arch:             {}\n\
         prelude/core/coda: {}/{}/{}   (K sampled ∈ [{}, {}])\n\
         hidden:           {}\n\
         vocab:            {}\n\
         seq_len:          {}\n\
         batch_size:       {}\n\
         epochs × steps:   {} × {}  (total = {})\n\
         lr:               {}   weight_decay: {}\n\
         corpus tokens:    {}\n",
        cfg.arch, rdt_cfg.n_prelude, rdt_cfg.n_core, rdt_cfg.n_coda,
        rdt_cfg.k_min, rdt_cfg.k_max,
        rdt_cfg.base.hidden_size, rdt_cfg.base.vocab_size,
        cfg.seq_len, cfg.batch_size,
        cfg.epochs, cfg.steps_per_epoch, cfg.epochs * cfg.steps_per_epoch,
        cfg.lr, cfg.weight_decay,
        format_count(dataset.tokens().len()),
    );

    let model = RDTForCausalLM::new(&rdt_cfg);
    let param_count: usize = model.parameters().iter().map(|p| p.data().shape().iter().product::<usize>()).sum();
    println!("trainable params: {} ({:.2}M)", format_count(param_count), param_count as f64 / 1e6);

    // Move to device.
    for p in model.parameters() {
        p.to_device(device.clone());
    }

    // Optimizer — AdamW with builder-style weight_decay chain.
    let mut optim = AdamW::new(model.parameters(), cfg.lr).weight_decay(cfg.weight_decay);

    // Lifecycle — monitor + pause/resume/stop + checkpoint rotation.
    let lifecycle = TrainingLifecycle::builder()
        .model_name(&format!("rdt-{}", cfg.arch))
        .output_dir(&cfg.output_dir)
        .param_count(param_count)
        .total_epochs(cfg.epochs)
        .batch_size(cfg.batch_size)
        .checkpoint_every_steps(cfg.checkpoint_every_steps)
        .keep_last_k(cfg.keep_last_k)
        .start();

    let mut training_state = TrainingState::new();

    // RNG streams — batch + K sampling advance independently for reproducibility.
    let mut rng_batch: u64 = cfg.seed;
    let mut rng_k: u64 = cfg.seed.wrapping_add(0x9E3779B97F4A7C15);
    let k_range_len = rdt_cfg.k_max - rdt_cfg.k_min + 1;

    let t_total = Instant::now();
    let mut global_step: usize = 0;

    'outer: for epoch in 1..=cfg.epochs {
        lifecycle.set_epoch(epoch);
        let t_epoch = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_count = 0usize;

        for step in 1..=cfg.steps_per_epoch {
            match lifecycle.poll() {
                LoopAction::Stop => {
                    lifecycle.save_final(&model, &training_state, epoch);
                    println!("stop requested at epoch {epoch} step {step}");
                    break 'outer;
                }
                LoopAction::CheckpointNow => {
                    lifecycle.save_step(&model, &training_state, epoch);
                }
                LoopAction::Continue => {}
            }

            // Sample K for this batch — the defining RDT training trick.
            let k = rdt_cfg.k_min + lcg_range(&mut rng_k, k_range_len);

            // Sample batch.
            let flat = dataset.sample_batch(cfg.batch_size, &mut rng_batch);
            let input_ids =
                Tensor::from_vec(flat, &[cfg.batch_size, cfg.seq_len]).unwrap();
            let labels = input_ids.clone();

            // Forward at sampled K.
            let logits = model.forward_ids(&input_ids, k);
            let loss = shifted_cross_entropy(&logits, &labels);
            let loss_val = loss.data().to_vec()[0];

            optim.zero_grad();
            loss.backward();
            optim.step();

            epoch_loss += loss_val;
            epoch_count += 1;
            global_step += 1;

            lifecycle.tick(global_step as u64, loss_val);

            if step % 10 == 0 {
                println!(
                    "  ep={epoch} step={step}/{}  k={k}  loss={:.4}  avg={:.4}",
                    cfg.steps_per_epoch,
                    loss_val,
                    epoch_loss / epoch_count as f32,
                );
            }
        }

        let avg = epoch_loss / epoch_count.max(1) as f32;
        training_state.global_step = global_step;
        println!(
            "epoch {epoch} done: avg_loss={avg:.4} in {:.1}s",
            t_epoch.elapsed().as_secs_f32()
        );
        lifecycle.save_epoch(&model, &training_state, epoch);
    }

    lifecycle.save_final(&model, &training_state, cfg.epochs);
    println!(
        "training complete: {} steps in {:.1}s",
        global_step,
        t_total.elapsed().as_secs_f32()
    );
}
