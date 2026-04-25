//! Export a trained Trident-Coder `.axonml` checkpoint to BitNet b1.58 GGUF.
//!
//! Reads a model state dict produced by `train_trident_code`, rebuilds a
//! `TridentModel` with the matching variant config, loads the checkpoint
//! shapes by name (with shape-fallback), and writes a GGUF that
//! `nexus-serve` can load via its existing I2_S dispatch (`bitnet-b1.58`
//! architecture, ggml dtype 36 for ternary projections, F32 for
//! norms, F16 for embeddings + LM head).
//!
//! # Usage
//! ```bash
//! cargo run --release --bin export_trident_gguf -- \
//!     --config 1b \
//!     --checkpoint .../checkpoints/trident-1b/checkpoint_final.axonml \
//!     --out .../models/trident/trident-coder-1b-i2s.gguf \
//!     --name trident-coder-1b
//!     [--tokenizer .../bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf]
//! ```
//!
//! # File
//! `llm-training/src/bin/export_trident_gguf.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty
//! of any kind, express or implied.

use std::path::PathBuf;

use axonml_llm::{TridentConfig, TridentModel, export_trident_to_gguf};
use llm_training::load_model_from_checkpoint;

#[derive(Debug)]
enum Variant {
    Smoke,
    Laptop,
    OneB,
    ThreeB,
}

impl Variant {
    fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "smoke" => Ok(Self::Smoke),
            "laptop" | "trident_laptop" => Ok(Self::Laptop),
            "1b" | "trident_1b" => Ok(Self::OneB),
            "3b" | "trident_3b" => Ok(Self::ThreeB),
            other => Err(format!(
                "Unknown --config '{other}'; expected smoke|laptop|1b|3b"
            )),
        }
    }

    fn build_config(&self, vocab_size: usize) -> TridentConfig {
        match self {
            Self::Smoke => TridentConfig::smoke(vocab_size),
            Self::Laptop => TridentConfig::trident_laptop(vocab_size),
            Self::OneB => TridentConfig::trident_1b(vocab_size),
            Self::ThreeB => TridentConfig::trident_3b(vocab_size),
        }
    }
}

struct Args {
    config: Variant,
    checkpoint: PathBuf,
    out: PathBuf,
    name: String,
    tokenizer: Option<PathBuf>,
    vocab_size: usize,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut config: Option<Variant> = None;
    let mut checkpoint: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut vocab_size: usize = 32_000; // matches trident-coder-bpe default

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--config" => {
                config = Some(Variant::parse(&raw[i + 1]).expect("invalid --config"));
                i += 2;
            }
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(&raw[i + 1]));
                i += 2;
            }
            "--out" => {
                out = Some(PathBuf::from(&raw[i + 1]));
                i += 2;
            }
            "--name" => {
                name = Some(raw[i + 1].clone());
                i += 2;
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(&raw[i + 1]));
                i += 2;
            }
            "--vocab-size" => {
                vocab_size = raw[i + 1].parse().expect("invalid --vocab-size");
                i += 2;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                std::process::exit(1);
            }
        }
    }

    Args {
        config: config.unwrap_or_else(|| {
            eprintln!("--config is required");
            std::process::exit(1);
        }),
        checkpoint: checkpoint.unwrap_or_else(|| {
            eprintln!("--checkpoint is required");
            std::process::exit(1);
        }),
        out: out.unwrap_or_else(|| {
            eprintln!("--out is required");
            std::process::exit(1);
        }),
        name: name.unwrap_or_else(|| "trident-coder".to_string()),
        tokenizer,
        vocab_size,
    }
}

fn print_help() {
    eprintln!(
        "Export a Trident-Coder .axonml checkpoint to BitNet b1.58 GGUF.\n\n\
         Usage: export_trident_gguf [OPTIONS]\n\n\
         Options:\n  \
           --config MODE        smoke | 1b | 3b\n  \
           --checkpoint PATH    Input .axonml checkpoint\n  \
           --out PATH           Output .gguf path\n  \
           --name NAME          Friendly name (default: trident-coder)\n  \
           --tokenizer PATH     Optional GGUF whose tokenizer.ggml.* keys are\n                       \
                copied verbatim into the output\n  \
           --vocab-size N       Vocab size used to build the config (default 32000)\n  \
           --help, -h           Show this message\n"
    );
}

fn main() {
    let args = parse_args();

    println!("=== Trident-Coder → GGUF export ===");
    println!("  config       : {:?}", args.config);
    println!("  vocab_size   : {}", args.vocab_size);
    println!("  checkpoint   : {}", args.checkpoint.display());
    println!("  output       : {}", args.out.display());
    println!("  name         : {}", args.name);
    if let Some(t) = &args.tokenizer {
        println!("  tokenizer    : {}", t.display());
    }

    let cfg = args.config.build_config(args.vocab_size);
    let model = TridentModel::new(&cfg);
    println!("  built model  : {} params", count_params(&model));

    println!("Loading checkpoint…");
    let (epoch, state) = load_model_from_checkpoint(&model, &args.checkpoint)
        .expect("Failed to load checkpoint");
    println!(
        "  resume epoch={epoch}, step={}, best_metric={:?}",
        state.global_step, state.best_metric
    );

    println!("Writing GGUF…");
    export_trident_to_gguf(&model, &args.out, &args.name, args.tokenizer.as_deref())
        .expect("export_trident_to_gguf failed");

    let bytes = std::fs::metadata(&args.out)
        .map(|m| m.len())
        .unwrap_or(0);
    println!(
        "Done. Wrote {} bytes ({:.2} MB) to {}",
        bytes,
        bytes as f64 / 1024.0 / 1024.0,
        args.out.display()
    );
}

fn count_params<M: axonml_nn::Module>(model: &M) -> usize {
    model.parameters().iter().map(|p| p.numel()).sum()
}
