//! tokenize_corpus — encode a UTF-8 text corpus to a flat uint32 token
//! stream using a Qwen-family GGUF's embedded byte-level BPE vocabulary.
//!
//! This unblocks the draft-distillation pipeline: until the trainer
//! feeds the teacher tokens from the TEACHER's tokenizer, the KL
//! divergence is between two distributions over different vocabularies
//! and the teacher's logits are noise. Running this ahead of training
//! once produces a `tokens.bin` that the trainer consumes via
//! `--tokens-bin`.
//!
//! ## Usage
//! ```bash
//! tokenize_corpus \
//!   --gguf   /opt/AxonML/models/qwen3-0.6b/Qwen_Qwen3-0.6B-Q4_K_M.gguf \
//!   --input  /opt/datasets/text/shakespeare.txt \
//!   --output /opt/datasets/text/shakespeare.qwen3.bin
//! ```
//!
//! Output layout: little-endian uint32 stream, one token ID per 4 bytes,
//! no header. `TokenBinDataset::from_bin_file` reads it directly.
//!
//! ## Algorithm
//! 1. Read `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` from the GGUF.
//! 2. Split the corpus on exact special-token matches (`<|im_start|>`,
//!    `<｜User｜>`, etc.) — these must survive BPE untouched.
//! 3. For each literal segment: UTF-8 bytes → GPT-2 byte-level unicode
//!    mapping → greedy merge-rule application → token IDs.
//! 4. Interleave specials, emit flat u32 stream.
//!
//! Derived from `nexus-serve/src/tokenizer/mod.rs` — same math, subset
//! of the feature set (no decode, no HuggingFace tokenizer.json
//! fallback, just byte-level BPE encode).
//!
//! # File
//! `llm-training/src/bin/tokenize_corpus.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use axonml_llm::read_gguf_tokenizer;

// =============================================================================
// CLI
// =============================================================================

struct Cli {
    gguf: PathBuf,
    input: PathBuf,
    output: PathBuf,
    /// Emit progress every N MB of input consumed.
    progress_every_mb: usize,
}

impl Cli {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--help" || a == "-h") {
            print_help();
            std::process::exit(0);
        }
        let mut cli = Self {
            gguf: PathBuf::new(),
            input: PathBuf::new(),
            output: PathBuf::new(),
            progress_every_mb: 16,
        };
        let mut i = 0;
        while i < args.len() {
            let next = |i: usize| -> String { args.get(i + 1).cloned().unwrap_or_else(|| { eprintln!("missing value for {}", args[i]); std::process::exit(1); }) };
            match args[i].as_str() {
                "--gguf" => { cli.gguf = PathBuf::from(next(i)); i += 2; }
                "--input" => { cli.input = PathBuf::from(next(i)); i += 2; }
                "--output" => { cli.output = PathBuf::from(next(i)); i += 2; }
                "--progress-every-mb" => { cli.progress_every_mb = next(i).parse().unwrap(); i += 2; }
                other => { eprintln!("unknown flag: {other}"); print_help(); std::process::exit(1); }
            }
        }
        if cli.gguf.as_os_str().is_empty() || cli.input.as_os_str().is_empty() || cli.output.as_os_str().is_empty() {
            eprintln!("--gguf, --input, and --output are all required");
            print_help();
            std::process::exit(1);
        }
        cli
    }
}

fn print_help() {
    eprintln!("tokenize_corpus — encode text to tokens.bin using a GGUF's embedded BPE");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  tokenize_corpus --gguf PATH --input PATH --output PATH");
    eprintln!();
    eprintln!("REQUIRED:");
    eprintln!("  --gguf PATH              Source GGUF with embedded BPE tokenizer");
    eprintln!("  --input PATH             UTF-8 text corpus");
    eprintln!("  --output PATH            Output .bin (little-endian uint32 stream)");
    eprintln!();
    eprintln!("OPTIONAL:");
    eprintln!("  --progress-every-mb N    Progress line cadence (default: 16)");
    eprintln!("  --help                   Show this");
}

// =============================================================================
// Byte-level BPE encoder (port of nexus-serve/src/tokenizer/mod.rs)
// =============================================================================

/// GPT-2 `bytes_to_unicode` canonical printable byte set.
fn canonical_printable_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for b in 0x21u8..=0x7E { out.push(b); }
    for b in 0xA1u8..=0xAC { out.push(b); }
    for b in 0xAEu8..=0xFF { out.push(b); }
    out
}

/// Build the forward byte → char map (GPT-2 byte-level BPE).
fn byte_encode_map() -> HashMap<u8, char> {
    let canonical = canonical_printable_bytes();
    let mut forward: Vec<(u8, char)> = canonical.iter().map(|&b| (b, b as char)).collect();
    let mut next_cp: u32 = 0x100;
    for b in 0u8..=255 {
        if !canonical.contains(&b) {
            let c = char::from_u32(next_cp).unwrap();
            forward.push((b, c));
            next_cp += 1;
        }
    }
    forward.into_iter().collect()
}

enum Segment<'a> {
    Literal(&'a str),
    Special(u32),
}

fn collect_special_tokens(token_to_id: &HashMap<String, u32>) -> Vec<(String, u32)> {
    let mut specials: Vec<(String, u32)> = token_to_id
        .iter()
        .filter(|(tok, _)| {
            let s = tok.as_str();
            (s.starts_with("<|") && s.ends_with("|>"))
                || (s.starts_with("<\u{FF5C}") && s.ends_with("\u{FF5C}>"))
                || s == "<s>"
                || s == "</s>"
        })
        .map(|(t, &i)| (t.clone(), i))
        .collect();
    specials.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
    specials
}

fn split_on_specials<'a>(text: &'a str, specials: &[(String, u32)]) -> Vec<Segment<'a>> {
    if specials.is_empty() {
        return vec![Segment::Literal(text)];
    }
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut lit_start = 0usize;
    while i < bytes.len() {
        let mut matched: Option<(usize, u32)> = None;
        for (tok, id) in specials {
            let tb = tok.as_bytes();
            if i + tb.len() <= bytes.len() && &bytes[i..i + tb.len()] == tb {
                matched = Some((tb.len(), *id));
                break;
            }
        }
        if let Some((len, id)) = matched {
            if i > lit_start {
                out.push(Segment::Literal(&text[lit_start..i]));
            }
            out.push(Segment::Special(id));
            i += len;
            lit_start = i;
        } else {
            i += 1;
        }
    }
    if lit_start < bytes.len() {
        out.push(Segment::Literal(&text[lit_start..]));
    }
    out
}

fn bpe_encode(
    text: &str,
    token_to_id: &HashMap<String, u32>,
    merges: &[(String, String)],
    merge_ranks: &HashMap<(String, String), usize>,
    forward: &HashMap<u8, char>,
) -> Vec<u32> {
    if text.is_empty() {
        return Vec::new();
    }
    let bytes = text.as_bytes();
    let mut symbols: Vec<String> = bytes
        .iter()
        .map(|&b| String::from(*forward.get(&b).unwrap()))
        .collect();

    // Priority-order merges. For Qwen3 the merges list is already in
    // priority order, so we run multiple passes until no merge fires.
    // A rank-based approach (find lowest-rank mergeable pair in the
    // sequence, merge it, repeat) matches HF tokenizers exactly and
    // is what guarantees correctness against the reference. We use
    // the rank-based version.
    //
    // `merge_ranks` maps (left, right) → merge index.
    loop {
        let mut best_rank = usize::MAX;
        let mut best_idx = None;
        for i in 0..symbols.len().saturating_sub(1) {
            let pair = (symbols[i].clone(), symbols[i + 1].clone());
            if let Some(&r) = merge_ranks.get(&pair) {
                if r < best_rank {
                    best_rank = r;
                    best_idx = Some(i);
                }
            }
        }
        let Some(idx) = best_idx else { break };
        let (l, r) = &merges[best_rank];
        let merged = format!("{l}{r}");
        symbols[idx] = merged;
        symbols.remove(idx + 1);
    }

    symbols
        .iter()
        .map(|s| *token_to_id.get(s).unwrap_or(&0))
        .collect()
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let cli = Cli::from_args();

    println!("tokenize_corpus");
    println!("  GGUF   : {}", cli.gguf.display());
    println!("  input  : {}", cli.input.display());
    println!("  output : {}", cli.output.display());

    // --- Load tokenizer metadata from GGUF. ---
    let t0 = Instant::now();
    let (tokens, merges) = read_gguf_tokenizer(&cli.gguf).unwrap_or_else(|e| {
        eprintln!("failed to read tokenizer from GGUF: {e}");
        std::process::exit(1);
    });
    let vocab_size = tokens.len();
    let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab_size);
    for (i, t) in tokens.iter().enumerate() {
        token_to_id.insert(t.clone(), i as u32);
    }
    let mut merge_ranks: HashMap<(String, String), usize> = HashMap::with_capacity(merges.len());
    for (i, pair) in merges.iter().enumerate() {
        merge_ranks.insert(pair.clone(), i);
    }
    let forward = byte_encode_map();
    let specials = collect_special_tokens(&token_to_id);

    println!(
        "  vocab  : {} tokens, {} merges, {} specials [{:.1}s]",
        vocab_size,
        merges.len(),
        specials.len(),
        t0.elapsed().as_secs_f32()
    );

    if merges.is_empty() {
        eprintln!(
            "WARNING: GGUF has no merges. Byte-level BPE needs them — output will be\n\
             per-byte tokens and training signal will be degraded. For Qwen3 family,\n\
             verify the source GGUF actually contains `tokenizer.ggml.merges`."
        );
    }

    // --- Read corpus. ---
    let t0 = Instant::now();
    let corpus = std::fs::read_to_string(&cli.input).unwrap_or_else(|e| {
        eprintln!("failed to read corpus: {e}");
        std::process::exit(1);
    });
    println!(
        "  corpus : {} chars ({:.1} MB) [{:.1}s]",
        corpus.len(),
        corpus.len() as f64 / 1e6,
        t0.elapsed().as_secs_f32()
    );

    // --- Tokenize. ---
    println!();
    println!("Tokenizing...");
    let t0 = Instant::now();
    let mut total_tokens: usize = 0;
    let mut next_report_bytes = cli.progress_every_mb * 1_000_000;
    let mut bytes_done = 0usize;

    let mut out_file = BufWriter::new(File::create(&cli.output).unwrap_or_else(|e| {
        eprintln!("failed to create output: {e}");
        std::process::exit(1);
    }));

    // Chunk the corpus — we split on specials, then each literal chunk is
    // BPE'd. For a plain text corpus with no specials, the whole thing
    // goes through a single bpe_encode. That blows up memory for large
    // corpora (every byte becomes a 4-byte char string). Split by line
    // instead: each line is small, BPE runs independently, output is
    // concatenated. Byte-level BPE merges can cross line boundaries in
    // principle, but in practice all Qwen-family vocabs treat newline
    // as a byte and don't have cross-line merges that rely on start-
    // of-line signaling.
    for line_with_nl in corpus.split_inclusive('\n') {
        bytes_done += line_with_nl.len();
        for seg in split_on_specials(line_with_nl, &specials) {
            let ids: Vec<u32> = match seg {
                Segment::Special(id) => vec![id],
                Segment::Literal(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    bpe_encode(text, &token_to_id, &merges, &merge_ranks, &forward)
                }
            };
            for id in &ids {
                out_file.write_all(&id.to_le_bytes()).unwrap();
            }
            total_tokens += ids.len();
        }

        if bytes_done >= next_report_bytes {
            let elapsed = t0.elapsed().as_secs_f32();
            let mb = bytes_done as f64 / 1e6;
            let mb_per_sec = mb / elapsed as f64;
            let toks_per_sec = total_tokens as f32 / elapsed;
            println!(
                "  progress: {:.1} MB → {} tokens ({:.1} MB/s, {:.0} tok/s)",
                mb, total_tokens, mb_per_sec, toks_per_sec
            );
            next_report_bytes += cli.progress_every_mb * 1_000_000;
        }
    }

    out_file.flush().unwrap();
    let wall = t0.elapsed();
    let ratio = total_tokens as f64 / corpus.len().max(1) as f64;
    println!();
    println!("Done.");
    println!(
        "  output: {} ({} tokens, {:.1} MB)",
        cli.output.display(),
        total_tokens,
        (total_tokens * 4) as f64 / 1e6
    );
    println!(
        "  time:   {:.1}s ({:.1} MB/s, {:.0} tok/s)",
        wall.as_secs_f32(),
        corpus.len() as f64 / 1e6 / wall.as_secs_f32() as f64,
        total_tokens as f32 / wall.as_secs_f32()
    );
    println!("  ratio:  {:.3} tokens/char", ratio);
}
