//! One-off metadata inspection for the Oracle (Gemma 4) GGUF blob.
//!
//! Run with:
//!   cargo test --release --test inspect_gemma -- --nocapture --ignored
//!
//! The `--ignored` flag is required because the test depends on a local file
//! that may not exist on all machines (ollama must have pulled Gemma 4).

use nexus_serve::model::gguf::GgufFile;
use nexus_serve::model::inference::InferenceConfig;

const ORACLE_PATH: &str = "/usr/share/ollama/.ollama/models/blobs/\
    sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a";

#[test]
#[ignore]
fn dump_oracle_metadata_and_tensor_names() {
    let gguf = GgufFile::open(std::path::Path::new(ORACLE_PATH))
        .expect("failed to open Oracle GGUF");

    println!("\n=== GGUF SUMMARY ===");
    gguf.summary();

    println!("\n=== ALL METADATA KEYS (sorted) ===");
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = gguf.metadata.get(k).unwrap();
        // Print value but truncate huge arrays
        let val_str = format!("{:?}", v);
        let display = if val_str.len() > 200 {
            format!("{}... ({} chars)", &val_str[..200], val_str.len())
        } else {
            val_str
        };
        println!("  {} = {}", k, display);
    }

    println!("\n=== TENSOR NAMES (unique prefixes per layer 0) ===");
    // Filter to layer-0 tensors so we see the per-layer weight set
    for t in gguf.tensors.iter() {
        let name = &t.name;
        if name.contains("blk.0.") || !name.contains("blk.") {
            println!(
                "  {}  shape={:?}  dtype={:?}",
                name, t.dims, t.dtype
            );
        }
    }

    println!("\n=== TENSOR COUNT ===");
    println!("  total tensors: {}", gguf.tensors.len());
    let n_layers = (0..200)
        .filter(|i| {
            gguf.tensors.iter().any(|t| t.name.starts_with(&format!("blk.{}.", i)))
        })
        .count();
    println!("  inferred block/layer count: {}", n_layers);

    // Verify InferenceConfig parses Oracle correctly.
    println!("\n=== InferenceConfig round-trip ===");
    let cfg = InferenceConfig::from_gguf(&gguf);
    println!("  architecture    = {}", cfg.architecture);
    println!("  hidden_size     = {}", cfg.hidden_size);
    println!("  num_layers      = {}", cfg.num_layers);
    println!("  num_heads       = {}", cfg.num_heads);
    println!("  num_kv_heads    = {}", cfg.num_kv_heads);
    println!("  head_dim        = {} (Gemma overrides from attention.key_length)", cfg.head_dim);
    println!("  intermediate    = {}", cfg.intermediate_size);
    println!("  max_seq_len     = {}", cfg.max_seq_len);
    println!("  rms_norm_eps    = {}", cfg.rms_norm_eps);
    println!("  rope_theta      = {} (full-attention base)", cfg.rope_theta);
    println!("  vocab_size      = {}", cfg.vocab_size);

    match &cfg.gemma {
        None => println!("  gemma           = (none — not a Gemma model)"),
        Some(g) => {
            println!("  gemma.sliding_window        = {}", g.sliding_window);
            println!("  gemma.head_dim_swa          = {}", g.head_dim_swa);
            println!("  gemma.rope_theta_swa        = {}", g.rope_theta_swa);
            println!("  gemma.rope_dim              = {}", g.rope_dim);
            println!("  gemma.rope_dim_swa          = {}", g.rope_dim_swa);
            println!("  gemma.final_logit_softcap   = {:?}", g.final_logit_softcap);
            println!("  gemma.per_layer_input_width = {}", g.per_layer_input_width);
            println!("  gemma.qk_norm_dim           = {}", g.qk_norm_dim);
            let swa_count = g.sliding_window_pattern.iter().filter(|&&b| b).count();
            let full_count = g.sliding_window_pattern.iter().filter(|&&b| !b).count();
            println!("  gemma.sliding_window_pattern = {} SWA / {} full (total {})",
                swa_count, full_count, g.sliding_window_pattern.len());
            println!("  pattern = {:?}", g.sliding_window_pattern);
        }
    }

    // Sanity assertions — these should hold for Gemma 4 Oracle.
    assert_eq!(cfg.architecture, "gemma4");
    assert_eq!(cfg.num_layers, 42);
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.head_dim, 512, "expected key_length=512 from GGUF");
    let g = cfg.gemma.as_ref().expect("gemma config missing");
    assert_eq!(g.sliding_window_pattern.len(), 42, "pattern length != num_layers");
    assert_eq!(g.sliding_window, 512);
    assert_eq!(g.head_dim_swa, 256);
    assert!((g.rope_theta_swa - 10_000.0).abs() < 1e-3);
    assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-3);
    assert_eq!(g.final_logit_softcap, Some(30.0));
    assert_eq!(g.per_layer_input_width, 256);
}
