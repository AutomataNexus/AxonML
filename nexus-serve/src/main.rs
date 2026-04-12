//! nexus-serve — Pure-Rust LLM inference server
//!
//! Usage:
//!   nexus-serve --model /path/to/model.gguf
//!   nexus-serve --model /path/to/model.gguf --port 11435
//!   nexus-serve --model /path/to/model.gguf --model /path/to/another.gguf
//!
//! Query via the OpenAI-compatible API:
//!   curl http://localhost:11435/v1/chat/completions \
//!     -H "Content-Type: application/json" \
//!     -d '{"model":"mymodel","messages":[{"role":"user","content":"Hello"}]}'

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tracing_subscriber;

#[allow(unused_imports)]
use axonml_core::Device;

use nexus_serve::api::routes::{self, AppState};
use nexus_serve::model::gguf::{GgufFile, GgufValue};
use nexus_serve::model::inference::{InferenceEngine, MappedGguf};
use nexus_serve::model::registry::{ModelInfo, ModelRegistry};
use nexus_serve::tokenizer::Tokenizer;

// =============================================================================
// CLI
// =============================================================================

struct Config {
    model_paths: Vec<PathBuf>,
    /// (alias, model_path) — the path must also appear in model_paths
    aliases: Vec<(String, PathBuf)>,
    port: u16,
    host: String,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut cfg = Config {
            model_paths: Vec::new(),
            aliases: Vec::new(),
            port: 11435,  // ollama is 11434, we're +1
            host: "0.0.0.0".to_string(),
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--model" | "-m" => {
                    i += 1;
                    cfg.model_paths.push(PathBuf::from(&args[i]));
                }
                "--alias" | "-a" => {
                    // --alias NAME PATH   — registers NAME as alias, also loads the model
                    i += 1;
                    let name = args[i].clone();
                    i += 1;
                    let path = PathBuf::from(&args[i]);
                    cfg.aliases.push((name, path.clone()));
                    cfg.model_paths.push(path);
                }
                "--port" | "-p" => {
                    i += 1;
                    cfg.port = args[i].parse().expect("Invalid port");
                }
                "--host" => {
                    i += 1;
                    cfg.host = args[i].clone();
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    // If it looks like a file path, treat it as a model
                    if other.ends_with(".gguf") || other.ends_with(".axonml") || other.ends_with(".safetensors") {
                        cfg.model_paths.push(PathBuf::from(other));
                    } else {
                        eprintln!("Unknown argument: {other}");
                        print_help();
                        std::process::exit(1);
                    }
                }
            }
            i += 1;
        }

        cfg
    }
}

fn print_help() {
    println!(r#"nexus-serve — Pure-Rust LLM inference server

USAGE:
    nexus-serve [OPTIONS] [MODEL_FILES...]

OPTIONS:
    --model, -m PATH       Load a model (GGUF, SafeTensors, or .axonml)
    --alias, -a NAME PATH  Load a model AND register a friendly alias.
                           Requests for NAME will route to the model at PATH.
                           (e.g., --alias sage /path/to/qwen.gguf)
    --port, -p PORT        Listen port (default: 11435)
    --host HOST            Listen host (default: 0.0.0.0)
    --help, -h             Show this help

EXAMPLES:
    nexus-serve --model qwen2.5-coder-1.5b.gguf
    nexus-serve --alias sage /path/to/qwen.gguf --alias oracle /path/to/gemma.gguf
    nexus-serve --model model1.gguf --model model2.gguf --port 11435

API ENDPOINTS (OpenAI-compatible):
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models
    GET  /health
"#);
}

// =============================================================================
// Main
// =============================================================================

/// Check if a file starts with GGUF magic bytes (0x46475547).
fn is_gguf(path: &std::path::Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).is_ok() && magic == [0x47, 0x47, 0x55, 0x46]  // "GGUF"
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" nexus-serve — Pure-Rust LLM Inference Server");
    println!(" Powered by AxonML");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let registry = ModelRegistry::new();
    let mut engines: std::collections::HashMap<String, Arc<InferenceEngine>> = std::collections::HashMap::new();
    let mut tokenizer_map: std::collections::HashMap<String, Arc<Tokenizer>> = std::collections::HashMap::new();
    // Track path → canonical model_name so we can wire up aliases after loading.
    let mut path_to_name: std::collections::HashMap<PathBuf, String> = std::collections::HashMap::new();

    // Load models — deduplicate paths (an --alias and --model can reference the same path)
    let mut unique_paths: Vec<PathBuf> = Vec::new();
    for p in &cfg.model_paths {
        if !unique_paths.contains(p) {
            unique_paths.push(p.clone());
        }
    }

    if unique_paths.is_empty() {
        println!("No models specified. Use --model PATH to load a model.");
        println!("Server will start with empty registry (models can be added at runtime).");
        println!();
    }

    for path in &unique_paths {
        println!("Loading: {}", path.display());

        // Try GGUF first (check magic bytes, not just extension — ollama
        // blobs have no extension but are valid GGUF files).
        if is_gguf(path) {
            match GgufFile::open(path) {
                Ok(gguf) => {
                    gguf.summary();

                    let model_name = gguf
                        .model_name()
                        .unwrap_or_else(|| {
                            path.file_stem()
                                .unwrap_or_default()
                                .to_str()
                                .unwrap_or("unknown")
                        })
                        .to_string();

                    let arch = gguf.architecture().unwrap_or("unknown").to_string();

                    // Extract key metadata
                    let ctx_len = gguf
                        .get_meta(&format!("{}.context_length", arch))
                        .and_then(|v| v.as_u32())
                        .unwrap_or(2048) as usize;

                    let vocab_size = gguf
                        .get_meta(&format!("{}.vocab_size", arch))
                        .or_else(|| gguf.get_meta("tokenizer.ggml.tokens"))
                        .and_then(|v| match v {
                            GgufValue::Array(a) => Some(a.len() as u32),
                            _ => v.as_u32(),
                        })
                        .unwrap_or(32000) as usize;

                    let n_params = gguf.tensors.iter().map(|t| t.n_elements()).sum::<u64>();

                    let quant = gguf
                        .tensors
                        .first()
                        .map(|t| format!("{:?}", t.dtype))
                        .unwrap_or_else(|| "unknown".to_string());

                    registry
                        .register(ModelInfo {
                            id: model_name.clone(),
                            path: path.clone(),
                            architecture: arch,
                            parameters: n_params,
                            quantization: quant,
                            context_length: ctx_len,
                            vocab_size,
                        })
                        .await;

                    path_to_name.insert(path.clone(), model_name.clone());
                    println!("  Registered as: {}", model_name);

                    // Load inference engine (dequantize weights → f32)
                    match MappedGguf::open(path, &gguf) {
                        Ok(mapped) => {
                            match InferenceEngine::from_gguf(&gguf, &mapped) {
                                Ok(mut engine) => {
                                    // Move weights to GPU if CUDA is available
                                    #[cfg(feature = "cuda")]
                                    {
                                        if axonml_core::backends::cuda::is_available() {
                                            engine.to_device(Device::Cuda(0));
                                            println!("  Inference engine: READY (GPU)");
                                        } else {
                                            println!("  Inference engine: READY (CPU — no CUDA device found)");
                                        }
                                    }
                                    #[cfg(not(feature = "cuda"))]
                                    {
                                        println!("  Inference engine: READY (CPU)");
                                    }
                                    engines.insert(model_name.clone(), Arc::new(engine));
                                }
                                Err(e) => eprintln!("  Inference engine failed: {}", e),
                            }
                        }
                        Err(e) => eprintln!("  Memory map failed: {}", e),
                    }

                    // Load tokenizer — priority: tokenizer.json > GGUF-embedded > char-level
                    // Search for tokenizer.json in multiple locations
                    let tok_search_paths = [
                        path.with_extension("tokenizer.json"),
                        path.parent().unwrap_or(path).join("tokenizer.json"),
                        // nexus-serve/tokenizers/ directory (manually placed)
                        std::env::current_exe()
                            .unwrap_or_default()
                            .parent()
                            .and_then(|p| p.parent())
                            .map(|p| p.join("tokenizers"))
                            .unwrap_or_default()
                            .join(format!("{}.tokenizer.json",
                                model_name.to_lowercase().replace(' ', "-"))),
                        PathBuf::from("/opt/AxonML/nexus-serve/tokenizers")
                            .join(format!("{}.tokenizer.json",
                                model_name.to_lowercase().replace(' ', "-"))),
                    ];

                    let tok = tok_search_paths.iter().find_map(|p| {
                        if p.exists() {
                            match Tokenizer::from_file(p) {
                                Ok(tok) => {
                                    println!("  Tokenizer: {} ({} tokens) from {}",
                                        tok.variant(), tok.vocab_size(), p.display());
                                    Some(tok)
                                }
                                Err(e) => {
                                    eprintln!("  Tokenizer error at {}: {}", p.display(), e);
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    });

                    let tok = tok.or_else(|| {
                        Tokenizer::from_gguf(&gguf).map(|t| {
                            println!("  Tokenizer: {} ({} tokens)", t.variant(), t.vocab_size());
                            t
                        })
                    });

                    let tok = tok.unwrap_or_else(|| {
                        let corpus = (32u8..=126).map(|b| b as char).collect::<String>();
                        let t = Tokenizer::char_level(&corpus);
                        println!("  Tokenizer: {} ({} tokens) — WARNING: real vocab not found", t.variant(), t.vocab_size());
                        t
                    });

                    tokenizer_map.insert(model_name.clone(), Arc::new(tok));

                    println!();
                }
                Err(e) => {
                    eprintln!("  Failed to load {}: {}", path.display(), e);
                }
            }
        } else {
            // TODO: SafeTensors and .axonml loading
            println!("  TODO: non-GGUF format loading not yet implemented");
        }
    }

    // Register aliases after all models are loaded
    if !cfg.aliases.is_empty() {
        println!("Registering aliases:");
        for (alias_name, alias_path) in &cfg.aliases {
            if let Some(canonical) = path_to_name.get(alias_path) {
                registry.register_alias(alias_name, canonical).await;
                println!("  {alias_name} → {canonical}");
            } else {
                eprintln!("  WARNING: alias '{alias_name}' path not loaded: {}", alias_path.display());
            }
        }
        println!();
    }

    // Build router
    let state = Arc::new(AppState {
        registry,
        engines: tokio::sync::RwLock::new(engines),
        tokenizers: tokio::sync::RwLock::new(tokenizer_map),
    });

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/completions", post(routes::completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port)
        .parse()
        .expect("Invalid listen address");

    println!("Listening on http://{}", addr);
    println!();
    println!("Endpoints:");
    println!("  Chat:    POST http://{}/v1/chat/completions", addr);
    println!("  Compl:   POST http://{}/v1/completions", addr);
    println!("  Models:  GET  http://{}/v1/models", addr);
    println!("  Health:  GET  http://{}/health", addr);
    println!();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
