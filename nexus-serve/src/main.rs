//! nexus-serve — Server Binary Entry Point
//!
//! Command-line launcher for the nexus-serve LLM inference daemon. Parses CLI
//! flags (`--model`, `--alias`, `--port`, `--host`, `--quantized`, `--threads`,
//! `--config`), merges them with a TOML config file at
//! `~/.config/nexus-serve/config.toml` (or `--config PATH`), echoes the
//! resolved settings with per-key source tags, loads every GGUF model given
//! (detected by magic bytes, not extension — ollama blobs have no extension),
//! registers friendly aliases, wires up `Tokenizer`s (priority:
//! `tokenizer.json` → GGUF-embedded → char-level fallback), optionally uploads
//! weights to GPU when `cuda` feature is enabled, then serves the axum
//! `Router` with `/health`, `/v1/models`, `/v1/chat/completions`,
//! `/v1/completions`, and `/v1/messages` (Anthropic Messages API) endpoints.
//!
//! Contains the [`Config`] struct, [`Source`] enum for per-field origin
//! tracking, [`print_help`], [`is_gguf`], and the `#[tokio::main]` entry.
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
//!
//! # File
//! `nexus-serve/src/main.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

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
// CLI Configuration
// =============================================================================

// -----------------------------------------------------------------------------
// Source Tracking
// -----------------------------------------------------------------------------

/// Which input source produced a given config value.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Source {
    Default,
    Config,
    Cli,
}

impl Source {
    fn label(self) -> &'static str {
        match self {
            Source::Default => "default",
            Source::Config => "config",
            Source::Cli => "cli",
        }
    }
}

// -----------------------------------------------------------------------------
// Config Struct
// -----------------------------------------------------------------------------

struct Config {
    model_paths: Vec<PathBuf>,
    /// (alias, model_path) — the path must also appear in model_paths
    aliases: Vec<(String, PathBuf)>,
    port: u16,
    host: String,
    /// If true, keep weights in their compact GGUF form and dequantize per-matmul.
    /// Saves ~5x RAM (e.g., 27B Gemma: 50GB → 10GB) at the cost of inference speed.
    quantized_weights: bool,
    /// Enable TurboQuant Q8 KV cache: store KV as int8 with per-(token,head)
    /// f32 scales. ~3.9× compression, correctness-equivalent at decode.
    kv_quant_q8: bool,
    /// TurboQuant key bits (0=disabled, 3=Q3, 4=Q4). Random rotation + aggressive quant.
    kv_turbo_keys: u8,
    /// TurboQuant value bits (0=disabled, 3=Q3).
    kv_turbo_values: u8,
    /// Lock model weights in RAM (mlock). Prevents the kernel from paging
    /// expert weights to disk during long-running inference sessions.
    mlock: bool,
    /// Preload entire GGUF into RAM instead of mmap. Eliminates page faults
    /// during inference at the cost of higher startup time and RSS.
    no_mmap: bool,
    /// Number of transformer layers to place on GPU. Remaining layers stay on CPU.
    /// `None` = all layers on GPU (if CUDA available) or all on CPU.
    n_gpu_layers: Option<usize>,
    /// For MoE models: number of layers whose experts are pinned to CPU while
    /// attention stays on GPU. Reduces VRAM usage at the cost of PCIe transfers.
    /// `None` = all experts follow their layer's device.
    n_cpu_moe: Option<usize>,
    /// Number of CPU threads for matmul/dequant. `None` = rayon default (all cores).
    threads: Option<usize>,
    /// Explicit config file path from --config. If `None`, falls back to
    /// `~/.config/nexus-serve/config.toml`.
    config_path: Option<PathBuf>,
    /// Optional `[hardware]` metadata from the config file. Used for validation warnings.
    hw_cores: Option<usize>,
    hw_cpu: Option<String>,
    hw_ram_gb: Option<usize>,
    /// Hailo-10H HEF path. When set, nexus-serve uses the Hailo NPU backend
    /// instead of CPU/CUDA GGUF inference. Requires `--features hailo10h`.
    #[cfg(feature = "hailo_genai")]
    hailo_hef: Option<PathBuf>,
    #[cfg(feature = "hailo10h")]
    hailo_custom_hef: Option<PathBuf>,
    #[cfg(feature = "nexusrt")]
    nexusrt_hef: Option<PathBuf>,
    /// Per-key source tracking so startup can echo where each value came from.
    src_port: Source,
    src_host: Source,
    src_threads: Source,
    src_quantized: Source,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut cfg = Config {
            model_paths: Vec::new(),
            aliases: Vec::new(),
            port: 11435,  // ollama is 11434, we're +1
            host: "0.0.0.0".to_string(),
            quantized_weights: false,
            kv_quant_q8: false,
            kv_turbo_keys: 0,
            kv_turbo_values: 0,
            mlock: false,
            no_mmap: false,
            n_gpu_layers: None,
            n_cpu_moe: None,
            threads: None,
            config_path: None,
            hw_cores: None,
            hw_cpu: None,
            hw_ram_gb: None,
            #[cfg(feature = "hailo_genai")]
            hailo_hef: None,
            #[cfg(feature = "hailo10h")]
            hailo_custom_hef: None,
            #[cfg(feature = "nexusrt")]
            nexusrt_hef: None,
            src_port: Source::Default,
            src_host: Source::Default,
            src_threads: Source::Default,
            src_quantized: Source::Default,
        };

        // Pre-pass for --config PATH so the config file is resolved *before*
        // we decide whether CLI values should override it.
        {
            let mut i = 1;
            while i < args.len() {
                if args[i] == "--config" || args[i] == "-c" {
                    i += 1;
                    if i < args.len() {
                        cfg.config_path = Some(PathBuf::from(&args[i]));
                    }
                }
                i += 1;
            }
        }

        // Load config file (~/.config/nexus-serve/config.toml or --config PATH)
        cfg.merge_config_file();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--config" | "-c" => {
                    // Already handled in the pre-pass above; skip its value.
                    i += 1;
                }
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
                    cfg.src_port = Source::Cli;
                }
                "--host" => {
                    i += 1;
                    cfg.host = args[i].clone();
                    cfg.src_host = Source::Cli;
                }
                "--quantized" | "-q" => {
                    cfg.quantized_weights = true;
                    cfg.src_quantized = Source::Cli;
                }
                "--kv-quant" => {
                    i += 1;
                    match args[i].as_str() {
                        "q8" => cfg.kv_quant_q8 = true,
                        "turbo" | "turbo4" => {
                            cfg.kv_quant_q8 = false;
                            cfg.kv_turbo_keys = 4;
                            cfg.kv_turbo_values = 3;
                        }
                        "turbo3" => {
                            cfg.kv_quant_q8 = false;
                            cfg.kv_turbo_keys = 3;
                            cfg.kv_turbo_values = 3;
                        }
                        "none" | "f32" => cfg.kv_quant_q8 = false,
                        other => {
                            eprintln!(
                                "Unknown --kv-quant value: {other} (expected q8, turbo, turbo3, or none)"
                            );
                            std::process::exit(1);
                        }
                    }
                }
                "--threads" | "-t" => {
                    i += 1;
                    cfg.threads = Some(args[i].parse().expect("Invalid --threads"));
                    cfg.src_threads = Source::Cli;
                }
                "--mlock" => {
                    cfg.mlock = true;
                }
                "--no-mmap" => {
                    cfg.no_mmap = true;
                }
                "--n-gpu-layers" | "-ngl" => {
                    i += 1;
                    cfg.n_gpu_layers = Some(args[i].parse().expect("Invalid --n-gpu-layers"));
                }
                "--n-cpu-moe" => {
                    i += 1;
                    cfg.n_cpu_moe = Some(args[i].parse().expect("Invalid --n-cpu-moe"));
                }
                #[cfg(feature = "hailo_genai")]
                "--hailo" => {
                    i += 1;
                    cfg.hailo_hef = Some(PathBuf::from(&args[i]));
                }
                #[cfg(feature = "hailo10h")]
                "--hailo-custom" => {
                    i += 1;
                    cfg.hailo_custom_hef = Some(PathBuf::from(&args[i]));
                }
                #[cfg(feature = "nexusrt")]
                "--nexusrt" => {
                    i += 1;
                    cfg.nexusrt_hef = Some(PathBuf::from(&args[i]));
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

    /// Merge values from `~/.config/nexus-serve/config.toml` if it exists.
    /// CLI flags take precedence over config file values.
    ///
    /// Example config:
    /// ```toml
    /// threads = 24
    /// port = 11435
    /// quantized = true
    ///
    /// [hardware]
    /// # Documentation only — used for sanity-check warnings.
    /// cpu  = "Intel Core Ultra 9 275HX"
    /// cores = 24
    /// ram_gb = 64
    /// ```
    fn merge_config_file(&mut self) {
        // Resolution order:
        //   1. --config PATH (explicit, errors loudly if missing)
        //   2. ~/.config/nexus-serve/config.toml (silently skipped if absent)
        let (path, from_cli) = if let Some(p) = &self.config_path {
            (p.clone(), true)
        } else {
            let Some(p) = std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".config/nexus-serve/config.toml"))
                .filter(|p| p.exists())
            else {
                return;
            };
            (p, false)
        };

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                if from_cli {
                    eprintln!("error: failed to read --config {}: {}", path.display(), e);
                    std::process::exit(1);
                }
                return;
            }
        };

        println!("Loading config: {}", path.display());

        // Very small hand-rolled TOML parser for just the keys we care about.
        // Tracks the current section so we can route `[hardware]` keys separately.
        let mut section = String::new();
        for raw in content.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                section = rest.trim().to_string();
                continue;
            }
            // Strip trailing "# comment" if any.
            let line = line.split('#').next().unwrap_or(line).trim();
            let Some((key, val)) = line.split_once('=') else { continue };
            let key = key.trim();
            let val = val.trim().trim_matches(&['"', '\''][..]);

            match (section.as_str(), key) {
                ("", "threads") => {
                    if let Ok(n) = val.parse::<usize>() {
                        self.threads = Some(n);
                        self.src_threads = Source::Config;
                    }
                }
                ("", "port") => {
                    if let Ok(p) = val.parse::<u16>() {
                        self.port = p;
                        self.src_port = Source::Config;
                    }
                }
                ("", "quantized") => {
                    if val == "true" {
                        self.quantized_weights = true;
                        self.src_quantized = Source::Config;
                    } else if val == "false" {
                        self.quantized_weights = false;
                        self.src_quantized = Source::Config;
                    }
                }
                ("", "host") => {
                    self.host = val.to_string();
                    self.src_host = Source::Config;
                }
                ("hardware", "cpu") => self.hw_cpu = Some(val.to_string()),
                ("hardware", "cores") => self.hw_cores = val.parse::<usize>().ok(),
                ("hardware", "ram_gb") => self.hw_ram_gb = val.parse::<usize>().ok(),
                _ => {}
            }
        }
    }

    /// Print the resolved configuration (with source tags) and warn on anything
    /// suspicious — threads > detected cores, hardware.cores mismatch, etc.
    fn echo_and_validate(&self) {
        let detected_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0);

        println!("Resolved config:");
        println!(
            "  host      = {:<20}  [{}]",
            self.host,
            self.src_host.label()
        );
        println!(
            "  port      = {:<20}  [{}]",
            self.port,
            self.src_port.label()
        );
        match self.threads {
            Some(n) => println!(
                "  threads   = {:<20}  [{}]",
                n,
                self.src_threads.label()
            ),
            None => println!(
                "  threads   = {:<20}  [{}]",
                format!("auto ({})", detected_cores),
                self.src_threads.label()
            ),
        }
        println!(
            "  quantized = {:<20}  [{}]",
            self.quantized_weights,
            self.src_quantized.label()
        );
        if let Some(cpu) = &self.hw_cpu {
            println!("  hardware  = {}", cpu);
        }
        println!();

        // Warn if requested thread count exceeds detected cores.
        if let Some(requested) = self.threads {
            if detected_cores > 0 && requested > detected_cores {
                eprintln!(
                    "warning: threads={} exceeds detected CPU parallelism ({}). \
                     Expect worse performance from oversubscription.",
                    requested, detected_cores
                );
            }
        }

        // Warn if the config's [hardware].cores doesn't match the machine.
        if let Some(declared) = self.hw_cores {
            if detected_cores > 0 && declared != detected_cores {
                eprintln!(
                    "warning: config [hardware] cores = {} but available_parallelism() = {}. \
                     Config may be stale.",
                    declared, detected_cores
                );
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Help Text
// -----------------------------------------------------------------------------

fn print_help() {
    println!(r#"nexus-serve — Pure-Rust LLM inference server

USAGE:
    nexus-serve [OPTIONS] [MODEL_FILES...]

OPTIONS:
    --model, -m PATH       Load a model (GGUF, SafeTensors, or .axonml)
    --alias, -a NAME PATH  Load a model AND register a friendly alias.
                           Requests for NAME will route to the model at PATH.
                           (e.g., --alias sage /path/to/qwen.gguf)
    --quantized, -q        Keep weights in compact GGUF form, dequantize per-matmul.
                           Saves ~5x RAM at the cost of inference speed.
                           Required to fit large models (e.g., 27B Gemma) in 16GB RAM.
    --kv-quant KIND        TurboQuant KV cache. KIND=q8 stores KV as int8 with
                           per-(token,head) f32 scales — ~3.9x memory compression
                           (e.g. 450MB → 115MB for DeepSeek-7B at 4k context), same
                           attention algorithm, dequant inline in the kernel.
                           KIND=none (default) keeps KV as f32.
    --threads, -t N        CPU thread count for matmul/dequantization.
                           Default: all physical cores (24 on Intel Core Ultra 9 275HX).
    --port, -p PORT        Listen port (default: 11435)
    --host HOST            Listen host (default: 0.0.0.0)
    --config, -c PATH      Load settings from a TOML file. Overrides the default
                           ~/.config/nexus-serve/config.toml location. CLI flags
                           still take precedence over config file values.
    --help, -h             Show this help

CONFIG FILE
    Default path: ~/.config/nexus-serve/config.toml
    Override:     --config PATH
    Example:      nexus-serve/config.example.toml (in this repo)

    Supported keys: threads, port, quantized, host, and a [hardware] section
    (cpu, cores, ram_gb) used for validation warnings.

    Precedence: CLI flags > config file > built-in defaults.

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
// GGUF Detection
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

// =============================================================================
// Main Entry Point
// =============================================================================

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cfg = Config::from_args();

    println!("═══════════════════════════════════════════════════════════");
    println!(" nexus-serve — Pure-Rust LLM Inference Server");
    println!(" Powered by AxonML");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // -------------------------------------------------------------------------
    // Config Echo + Thread Pool Setup
    // -------------------------------------------------------------------------

    // Echo resolved config with source tags and run sanity validation.
    cfg.echo_and_validate();

    // Configure the global rayon thread pool for parallel dequantization + CPU matmul.
    // If the user set --threads N (or threads=N in config), use N. Otherwise rayon
    // defaults to num_cpus::get() which on a 24-core machine is 24.
    if let Some(n) = cfg.threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(n).build_global() {
            eprintln!("Warning: failed to set rayon thread count to {n}: {e}");
        } else {
            println!("Rayon thread pool: {} threads", n);
        }
    }
    // Also hint to downstream BLAS (matrixmultiply uses this).
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        if let Some(n) = cfg.threads {
            // SAFETY: single-threaded at startup.
            unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
        }
    }

    // -------------------------------------------------------------------------
    // Registry + Engine Maps
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Hailo-10H NPU backend (when --hailo <hef> is passed)
    // -------------------------------------------------------------------------

    #[cfg(feature = "hailo_genai")]
    let hailo_engine: Option<Arc<nexus_serve::model::hailo10h::Hailo10hEngine>> = {
        if let Some(ref hef_path) = cfg.hailo_hef {
            println!("Loading Hailo-10H LLM from: {}", hef_path.display());
            match nexus_serve::model::hailo10h::Hailo10hEngine::load(hef_path) {
                Ok(engine) => {
                    println!("  ✓ Hailo-10H engine loaded (max context: {} tokens)", engine.max_context());
                    if let Some(tmpl) = engine.prompt_template() {
                        println!("  prompt template: {}", &tmpl[..tmpl.len().min(80)]);
                    }
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    eprintln!("  ✗ Failed to load Hailo-10H engine: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            None
        }
    };

    // -------------------------------------------------------------------------
    // Hailo-10H custom HEF backend (when --hailo-custom <hef> is passed)
    // -------------------------------------------------------------------------

    #[cfg(feature = "hailo10h")]
    let hailo_custom_engine: Option<Arc<nexus_serve::model::hailo_custom::HailoCustomEngine>> = {
        if let Some(ref hef_path) = cfg.hailo_custom_hef {
            println!("Loading Hailo custom HEF from: {}", hef_path.display());
            match nexus_serve::model::hailo_custom::HailoCustomEngine::load(hef_path) {
                Ok(engine) => {
                    println!("  ✓ Hailo custom engine loaded: {}", engine.hef_path());
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    eprintln!("  ✗ Failed to load Hailo custom engine: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            None
        }
    };

    // -------------------------------------------------------------------------
    // NexusRT direct-ioctl backend (when --nexusrt <hef> is passed)
    // -------------------------------------------------------------------------

    #[cfg(feature = "nexusrt")]
    let nexusrt_engine: Option<Arc<nexus_serve::model::nexusrt_engine::NexusRtEngine>> = {
        if let Some(ref hef_path) = cfg.nexusrt_hef {
            println!("Loading HEF via NexusRT (zero libhailort): {}", hef_path.display());
            match nexus_serve::model::nexusrt_engine::NexusRtEngine::load(hef_path) {
                Ok(engine) => {
                    let arch = if engine.is_h10() { "Hailo-10H" } else { "Hailo-8" };
                    println!("  NexusRT engine ready: {} on {}", engine.hef_path(), arch);
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    eprintln!("  Failed to load NexusRT engine: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            None
        }
    };

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

    // -------------------------------------------------------------------------
    // Model Loading Loop
    // -------------------------------------------------------------------------

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

                    // Load inference engine (dequantize weights → f32, or keep quantized)
                    match MappedGguf::open_with_opts(path, &gguf, cfg.no_mmap, cfg.mlock) {
                        Ok(mapped) => {
                            match InferenceEngine::from_gguf_with_mode(&gguf, &mapped, cfg.quantized_weights) {
                                #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
                                Ok(mut engine) => {
                                    // Move weights to GPU if CUDA is available
                                    #[cfg(feature = "cuda")]
                                    {
                                        if axonml_core::backends::cuda::is_available() {
                                            engine.to_device(Device::Cuda(0));
                                            engine.set_kv_quant_q8(cfg.kv_quant_q8);
                                            let kv_tag = if cfg.kv_quant_q8 { " [KV=Q8]" } else { "" };
                                            println!("  Inference engine: READY (GPU){kv_tag}");
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

    // -------------------------------------------------------------------------
    // Alias Registration
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Router + Listener
    // -------------------------------------------------------------------------

    // Build router
    let state = Arc::new(AppState {
        registry,
        engines: tokio::sync::RwLock::new(engines),
        tokenizers: tokio::sync::RwLock::new(tokenizer_map),
        #[cfg(feature = "hailo_genai")]
        hailo_engine,
        #[cfg(feature = "hailo10h")]
        hailo_custom_engine,
        #[cfg(feature = "nexusrt")]
        nexusrt_engine,
    });

    let mut app = Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/messages", post(nexus_serve::api::messages::messages));

    #[cfg(feature = "hailo10h")]
    {
        app = app.route("/v1/hailo/infer", post(routes::hailo_infer));
    }

    let app = app
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
