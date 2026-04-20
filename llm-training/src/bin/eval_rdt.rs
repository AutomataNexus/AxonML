//! eval_rdt — validation perplexity + K-scaling curve for a trained RDT.
//!
//! Loads a checkpoint produced by `train_rdt` and measures mean cross-
//! entropy over held-out val tokens at several K values (test-time core
//! iterations). Produces the data for the "test-time compute scaling
//! curve" cell in the design-doc evaluation matrix.
//!
//! This runs the RDT through `axonml-llm::RDTForCausalLM::forward_ids`
//! directly — no nexus-serve integration required. When nexus-serve's
//! forward_one_rdt lands (task #60), a parallel eval can validate
//! numeric parity between the Rust-training-path forward and the
//! Rust-serving-path forward.
//!
//! ## Usage
//!
//! ```bash
//! eval_rdt \
//!   --arch tiny \
//!   --checkpoint /opt/AxonML/llm-training/checkpoints/rdt-oracle-tiny/checkpoint_final.axonml \
//!   --tokens-bin /opt/datasets/oracle-lora/corpus.tokens.bin \
//!   --seq-len 256 --val-windows 100 \
//!   --k 1,2,4,8,16
//! ```
//!
//! Output table:
//!
//! ```text
//!   K        CE (nats)   ppl      seq/s   tok/s
//!   1        …          …        …       …
//!   2        …          …        …       …
//!   4        …          …        …       …
//!   …
//! ```
//!
//! # File
//! `llm-training/src/bin/eval_rdt.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
use axonml_core::Device;
use axonml_llm::{RDTConfig, RDTForCausalLM};
use axonml_tensor::Tensor;

use llm_training::{
    ResumeMode, TextDataset, find_checkpoint, format_count,
    load_model_from_checkpoint, shifted_cross_entropy,
};

// =============================================================================
// CLI
// =============================================================================

struct Args {
    arch: String,
    checkpoint: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    tokens_bin: PathBuf,
    seq_len: usize,
    val_windows: usize,
    k_values: Vec<usize>,
    seed: u64,
}

impl Args {
    fn from_env() -> Self {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        if argv.iter().any(|a| a == "--help" || a == "-h") {
            print_help();
            std::process::exit(0);
        }

        let mut a = Self {
            arch: "tiny".into(),
            checkpoint: None,
            checkpoint_dir: None,
            tokens_bin: PathBuf::from("/opt/datasets/oracle-lora/corpus.tokens.bin"),
            seq_len: 256,
            val_windows: 50,
            k_values: vec![1, 2, 4, 8, 16],
            seed: 17, // distinct from training seed so val windows differ
        };
        let mut i = 0;
        while i < argv.len() {
            let arg = &argv[i];
            let next = |i: usize| -> String {
                argv.get(i + 1).cloned().unwrap_or_else(|| {
                    eprintln!("missing value for {arg}");
                    std::process::exit(1);
                })
            };
            match arg.as_str() {
                "--arch" => { a.arch = next(i); i += 2; }
                "--checkpoint" => { a.checkpoint = Some(PathBuf::from(next(i))); i += 2; }
                "--checkpoint-dir" => { a.checkpoint_dir = Some(PathBuf::from(next(i))); i += 2; }
                "--tokens-bin" => { a.tokens_bin = PathBuf::from(next(i)); i += 2; }
                "--seq-len" => { a.seq_len = next(i).parse().unwrap(); i += 2; }
                "--val-windows" | "-n" => { a.val_windows = next(i).parse().unwrap(); i += 2; }
                "--k" => {
                    a.k_values = next(i)
                        .split(',')
                        .map(|s| s.trim().parse().expect("bad --k"))
                        .collect();
                    i += 2;
                }
                "--seed" => { a.seed = next(i).parse().unwrap(); i += 2; }
                _ => {
                    eprintln!("unknown arg: {arg}");
                    print_help();
                    std::process::exit(1);
                }
            }
        }
        if a.checkpoint.is_none() && a.checkpoint_dir.is_none() {
            eprintln!("--checkpoint or --checkpoint-dir required");
            std::process::exit(1);
        }
        a
    }
}

fn print_help() {
    println!(
        "eval_rdt — K-scaling perplexity curve for a trained RDT checkpoint\n\n\
         REQUIRED:\n  \
           --checkpoint PATH     exact .axonml file\n  \
           --checkpoint-dir PATH checkpoint directory (picks latest)\n\n\
         OPTIONS:\n  \
           --arch tiny|small|mid                  preset (default: tiny)\n  \
           --tokens-bin PATH                       val tokens (default: Oracle corpus)\n  \
           --seq-len N                             context window (default: 256)\n  \
           --val-windows N                         number of eval windows (default: 50)\n  \
           --k N,M,...                             K values to sweep (default: 1,2,4,8,16)\n  \
           --seed N                                RNG seed (default: 17)\n"
    );
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args = Args::from_env();

    // Resolve checkpoint path.
    let ckpt = if let Some(ref path) = args.checkpoint {
        path.clone()
    } else {
        let dir = args.checkpoint_dir.as_ref().unwrap();
        find_checkpoint(dir, &ResumeMode::Latest)
            .unwrap_or_else(|| {
                eprintln!("no checkpoint in {}", dir.display());
                std::process::exit(1);
            })
    };
    println!("checkpoint: {}", ckpt.display());

    // Build model.
    let rdt_cfg = match args.arch.as_str() {
        "tiny" => RDTConfig::rdt_tiny(),
        "small" => RDTConfig::rdt_small(),
        "mid" => RDTConfig::rdt_mid(),
        other => { eprintln!("unknown --arch '{other}'"); std::process::exit(1); }
    };

    // Dataset — vocab inferred from max token ID, same override as train_rdt.
    let dataset = TextDataset::from_tokens_bin(&args.tokens_bin, args.seq_len)
        .unwrap_or_else(|e| { eprintln!("load tokens bin: {e}"); std::process::exit(1); });
    let max_id = dataset.tokens().iter().copied().max().unwrap_or(0) as usize;
    let vocab_size = max_id + 1;

    let mut rdt_cfg = rdt_cfg;
    rdt_cfg.base.vocab_size = vocab_size;

    println!(
        "arch={}  hidden={}  vocab={}  prelude/core/coda = {}/{}/{}",
        args.arch, rdt_cfg.base.hidden_size, rdt_cfg.base.vocab_size,
        rdt_cfg.n_prelude, rdt_cfg.n_core, rdt_cfg.n_coda
    );

    let mut model = RDTForCausalLM::new(&rdt_cfg);
    let n_params: usize =
        model.parameters().iter().map(|p| p.data().shape().iter().product::<usize>()).sum();
    println!(
        "params: {} ({:.2}M)   val windows: {}   seq_len: {}",
        format_count(n_params), n_params as f64 / 1e6,
        args.val_windows, args.seq_len
    );

    // Load checkpoint into model.
    println!("loading weights from {} ...", ckpt.display());
    let _ = load_model_from_checkpoint(&mut model, &ckpt);

    // Move to CUDA if available.
    #[cfg(feature = "cuda")]
    if axonml_core::backends::cuda::is_available() {
        for p in model.parameters() {
            p.to_device(Device::Cuda(0));
        }
        println!("device: Cuda(0)");
    } else {
        println!("device: Cpu");
    }
    #[cfg(not(feature = "cuda"))]
    println!("device: Cpu");

    // Pre-sample val batches deterministically so each K uses the same
    // sequence of windows. This makes the CE comparison across K
    // well-defined (no RNG variance).
    let mut rng = args.seed;
    let val_batches: Vec<Tensor<u32>> = (0..args.val_windows)
        .map(|_| {
            let flat = dataset.sample_batch(1, &mut rng);
            Tensor::from_vec(flat, &[1, args.seq_len]).unwrap()
        })
        .collect();

    println!();
    println!("{:<6}{:>12}{:>12}{:>10}{:>10}", "K", "CE (nats)", "ppl", "seq/s", "tok/s");
    println!("{}", "-".repeat(50));

    for &k in &args.k_values {
        let t0 = Instant::now();
        let mut ce_total = 0.0f32;
        let mut n_eval = 0usize;
        for ids in &val_batches {
            let logits = model.forward_ids(ids, k);
            let loss = shifted_cross_entropy(&logits, ids);
            let ce = loss.data().to_vec()[0];
            ce_total += ce;
            n_eval += 1;
        }
        let ce = ce_total / n_eval.max(1) as f32;
        let ppl = ce.exp();
        let elapsed = t0.elapsed().as_secs_f32();
        let seq_per_s = n_eval as f32 / elapsed.max(1e-6);
        let tok_per_s = seq_per_s * args.seq_len as f32;
        println!(
            "{:<6}{:>12.4}{:>12.2}{:>10.2}{:>10.1}",
            k, ce, ppl, seq_per_s, tok_per_s
        );
    }

    println!();
    println!("eval complete. Interpretation:");
    println!("- CE should decrease monotonically with K (more compute → better predictions)");
    println!("- Plateau past some K = diminishing returns; the model has 'converged' at that depth");
    println!("- If CE INCREASES past some K = overthinking (v1 has no ACT; v3's ACT halting fixes this)");
}
