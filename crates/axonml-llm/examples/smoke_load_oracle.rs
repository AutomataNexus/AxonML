//! Smoke test: load a qwen2-arch GGUF (R1-Distill-Qwen teachers) through
//! `load_qwen3_from_gguf` after the arch-guard widening (L91). Reports the
//! parsed config — a successful run means teacher-distill loads are
//! unblocked end-to-end.
//!
//! Run:
//!   cargo run --release -p axonml-llm --example smoke_load_oracle \
//!     -- /opt/AxonML/models/oracle-distill/oracle-r1-distill-q4km.gguf

use std::env;
use std::path::Path;

use axonml_llm::load_qwen3_from_gguf;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "/opt/AxonML/models/oracle-distill/oracle-r1-distill-q4km.gguf".into());
    println!("loading: {path}");

    match load_qwen3_from_gguf(Path::new(&path)) {
        Ok((_model, cfg)) => {
            println!("PASS — loaded successfully");
            println!("  hidden_size        = {}", cfg.hidden_size);
            println!("  intermediate_size  = {}", cfg.intermediate_size);
            println!("  num_hidden_layers  = {}", cfg.num_hidden_layers);
            println!("  num_attention_heads = {}", cfg.num_attention_heads);
            println!("  num_key_value_heads = {}", cfg.num_key_value_heads);
            println!("  head_dim           = {}", cfg.head_dim);
            println!("  vocab_size         = {}", cfg.vocab_size);
            println!("  max_position       = {}", cfg.max_position_embeddings);
            println!("  rms_norm_eps       = {}", cfg.rms_norm_eps);
            println!("  rope_theta         = {}", cfg.rope_theta);
        }
        Err(e) => {
            eprintln!("FAIL — load error: {e}");
            std::process::exit(1);
        }
    }
}
