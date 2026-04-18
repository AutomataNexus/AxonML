//! tokenize_test — quick CLI to encode a test string via nexus-serve's
//! GGUF-backed tokenizer and print the token IDs + round-trip decode.
//!
//! Usage:
//!   cargo run --release --bin tokenize_test -- <MODEL.gguf> "<text>"

use std::path::PathBuf;

use nexus_serve::model::gguf::GgufFile;
use nexus_serve::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: tokenize_test <MODEL.gguf> \"<text>\"");
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let text = args[2..].join(" ");

    let gguf = GgufFile::open(&gguf_path).expect("open gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("build tokenizer");

    println!("Tokenizer: {}", tok.variant());
    println!("Vocab size: {}", tok.vocab_size());
    println!();
    println!("Input: {:?}", text);

    let ids = tok.encode(&text);
    println!("Token IDs ({}): {:?}", ids.len(), ids);

    // Show each ID's string.
    for id in &ids {
        let s = tok.decode(&[*id]);
        print!("[{:>5}={:?}] ", id, s);
    }
    println!();

    let decoded = tok.decode(&ids);
    println!("Round-trip decode: {:?}", decoded);
}
