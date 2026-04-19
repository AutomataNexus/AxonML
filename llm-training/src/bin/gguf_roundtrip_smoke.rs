//! gguf_roundtrip_smoke — verify Qwen3 GGUF export + load produces a
//! structurally-valid file that our own loader can parse. Not a
//! numerical-fidelity test (F16 round-trips lose ULP-level precision);
//! that's for a later nexus-serve-inference verification.

use std::path::{Path, PathBuf};

use axonml_llm::{
    Qwen3Config, Qwen3ForCausalLM, export_qwen3_to_gguf, load_qwen3_from_gguf,
    read_gguf_metadata_raw_bytes,
};

fn main() {
    let cfg = Qwen3Config::tiny();
    let model = Qwen3ForCausalLM::new(&cfg);
    let params = model.parameters();
    let n_params = params.len();
    let total_elements: usize = params.iter().map(|p| p.data().numel()).sum();
    println!("Student built: {n_params} tensors, {total_elements} elements");

    let out = Path::new("/tmp/qwen3_tiny_export.gguf");
    if out.exists() {
        let _ = std::fs::remove_file(out);
    }

    // Pick a tokenizer-source GGUF if the environment has one handy, so
    // we exercise the metadata-passthrough path in addition to the
    // tokenizer-less (None) path.
    let tokenizer_source: Option<PathBuf> = std::env::var("TOKENIZER_SOURCE_GGUF")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let default =
                PathBuf::from("/opt/AxonML/models/qwen3-1.7b/Qwen_Qwen3-1.7B-Q4_K_M.gguf");
            default.exists().then_some(default)
        });

    println!("Exporting to {}...", out.display());
    if let Some(src) = &tokenizer_source {
        println!("  tokenizer-source: {}", src.display());
    } else {
        println!("  tokenizer-source: None (no passthrough)");
    }
    export_qwen3_to_gguf(
        &model,
        &cfg,
        out,
        "qwen3-tiny-test",
        tokenizer_source.as_deref(),
    )
    .expect("export failed");
    let size = std::fs::metadata(out).unwrap().len();
    println!("Export complete: {} bytes", size);

    println!("Reloading via load_qwen3_from_gguf...");
    let (reloaded, reloaded_cfg) = load_qwen3_from_gguf(out).expect("reload failed");
    println!(
        "Reloaded config: vocab={} hidden={} layers={} heads={}x{} head_dim={} tie={}",
        reloaded_cfg.vocab_size,
        reloaded_cfg.hidden_size,
        reloaded_cfg.num_hidden_layers,
        reloaded_cfg.num_attention_heads,
        reloaded_cfg.num_key_value_heads,
        reloaded_cfg.head_dim,
        reloaded_cfg.tie_word_embeddings,
    );
    let reloaded_params = reloaded.parameters();
    println!(
        "Reloaded tensor count: {} (vs {} expected)",
        reloaded_params.len(),
        n_params
    );

    // Spot-check: compare first-param magnitudes pre vs post roundtrip.
    for i in [0usize, 3, 10] {
        if i < params.len() && i < reloaded_params.len() {
            let a = params[i].data().to_vec();
            let b = reloaded_params[i].data().to_vec();
            let take = a.len().min(b.len()).min(100);
            let diff: f32 = a
                .iter()
                .zip(b.iter())
                .take(take)
                .map(|(x, y)| (x - y).abs())
                .sum::<f32>()
                / take as f32;
            println!(
                "  param[{i}]: len_orig={} len_reload={} mean_abs_diff(first {take})={:.6}",
                a.len(),
                b.len(),
                diff
            );
        }
    }

    // Verify tokenizer passthrough if we had a source: the exported
    // file should now contain `tokenizer.ggml.tokens` (and friends) as
    // readable metadata entries.
    if let Some(src) = &tokenizer_source {
        let keys = [
            "tokenizer.ggml.model",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
        ];
        let src_raw = read_gguf_metadata_raw_bytes(src, &keys).expect("read src");
        let out_raw = read_gguf_metadata_raw_bytes(out, &keys).expect("read out");
        println!("Tokenizer passthrough check ({}):", src.display());
        for k in keys {
            let src_present = src_raw.contains_key(k);
            let out_present = out_raw.contains_key(k);
            let bytes_match = match (src_raw.get(k), out_raw.get(k)) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            println!(
                "  {k}: src={} out={} bytes_match={}",
                src_present, out_present, bytes_match
            );
            if src_present && !out_present {
                panic!("tokenizer passthrough dropped key {k}");
            }
            if src_present && out_present && !bytes_match {
                panic!("tokenizer passthrough corrupted bytes for key {k}");
            }
        }
    }

    println!("OK");
}
