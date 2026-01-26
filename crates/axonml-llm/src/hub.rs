//! LLM Model Hub - Pretrained Language Model Weights
//!
//! Download, cache, and load pretrained weights for language models.
//!
//! # Supported Models
//! - BERT (base, large, tiny)
//! - GPT-2 (small, medium, large, xl)
//! - RoBERTa
//! - DistilBERT
//! - ALBERT
//!
//! # Example
//! ```rust,ignore
//! use axonml_llm::hub::{llm_registry, download_weights, PretrainedLLM};
//!
//! // List available models
//! let registry = llm_registry();
//! for (name, model) in &registry {
//!     println!("{}: {} params", name, model.num_parameters);
//! }
//!
//! // Download a model
//! let path = download_weights("bert-base-uncased", false).unwrap();
//! ```
//!
//! @version 0.1.0

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

// =============================================================================
// Error Type
// =============================================================================

/// LLM Hub errors.
#[derive(Debug)]
pub enum HubError {
    /// Network error during download.
    NetworkError(String),
    /// IO error.
    IoError(std::io::Error),
    /// Model not found.
    ModelNotFound(String),
    /// Invalid format.
    InvalidFormat(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::NetworkError(e) => write!(f, "Network error: {}", e),
            HubError::IoError(e) => write!(f, "IO error: {}", e),
            HubError::ModelNotFound(name) => write!(f, "Model not found: {}", name),
            HubError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for HubError {}

impl From<std::io::Error> for HubError {
    fn from(e: std::io::Error) -> Self {
        HubError::IoError(e)
    }
}

/// Result type for hub operations.
pub type HubResult<T> = Result<T, HubError>;

// =============================================================================
// Pretrained Model Registry
// =============================================================================

/// Information about a pretrained LLM.
#[derive(Debug, Clone)]
pub struct PretrainedLLM {
    /// Model name (e.g., "bert-base-uncased").
    pub name: String,
    /// URL to download weights.
    pub url: String,
    /// Expected SHA256 checksum (optional).
    pub checksum: Option<String>,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden size / embedding dimension.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of parameters (approximate).
    pub num_parameters: u64,
    /// Model architecture type.
    pub architecture: String,
    /// Training dataset.
    pub dataset: String,
}

/// Get the cache directory for LLM weights.
pub fn cache_dir() -> PathBuf {
    let base = dirs::cache_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("axonml").join("hub").join("llm")
}

/// Get registry of available pretrained LLM models.
pub fn llm_registry() -> HashMap<String, PretrainedLLM> {
    let mut registry = HashMap::new();

    // =========================================================================
    // BERT Family
    // =========================================================================

    registry.insert(
        "bert-tiny".to_string(),
        PretrainedLLM {
            name: "bert-tiny".to_string(),
            url: "https://huggingface.co/axonml-ml/bert-tiny/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 17_000_000,
            vocab_size: 30522,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 2,
            max_seq_len: 512,
            num_parameters: 4_400_000,
            architecture: "BERT".to_string(),
            dataset: "Wikipedia + BookCorpus".to_string(),
        },
    );

    registry.insert(
        "bert-mini".to_string(),
        PretrainedLLM {
            name: "bert-mini".to_string(),
            url: "https://huggingface.co/axonml-ml/bert-mini/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 45_000_000,
            vocab_size: 30522,
            hidden_size: 256,
            num_layers: 4,
            num_heads: 4,
            max_seq_len: 512,
            num_parameters: 11_200_000,
            architecture: "BERT".to_string(),
            dataset: "Wikipedia + BookCorpus".to_string(),
        },
    );

    registry.insert(
        "bert-base-uncased".to_string(),
        PretrainedLLM {
            name: "bert-base-uncased".to_string(),
            url: "https://huggingface.co/axonml-ml/bert-base-uncased/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 440_000_000,
            vocab_size: 30522,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_seq_len: 512,
            num_parameters: 110_000_000,
            architecture: "BERT".to_string(),
            dataset: "Wikipedia + BookCorpus".to_string(),
        },
    );

    registry.insert(
        "bert-base-cased".to_string(),
        PretrainedLLM {
            name: "bert-base-cased".to_string(),
            url: "https://huggingface.co/axonml-ml/bert-base-cased/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 440_000_000,
            vocab_size: 28996,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_seq_len: 512,
            num_parameters: 109_000_000,
            architecture: "BERT".to_string(),
            dataset: "Wikipedia + BookCorpus".to_string(),
        },
    );

    registry.insert(
        "bert-large-uncased".to_string(),
        PretrainedLLM {
            name: "bert-large-uncased".to_string(),
            url: "https://huggingface.co/axonml-ml/bert-large-uncased/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 1_340_000_000,
            vocab_size: 30522,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            max_seq_len: 512,
            num_parameters: 336_000_000,
            architecture: "BERT".to_string(),
            dataset: "Wikipedia + BookCorpus".to_string(),
        },
    );

    // =========================================================================
    // GPT-2 Family
    // =========================================================================

    registry.insert(
        "gpt2".to_string(),
        PretrainedLLM {
            name: "gpt2".to_string(),
            url: "https://huggingface.co/axonml-ml/gpt2/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 548_000_000,
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_seq_len: 1024,
            num_parameters: 124_000_000,
            architecture: "GPT-2".to_string(),
            dataset: "WebText".to_string(),
        },
    );

    registry.insert(
        "gpt2-medium".to_string(),
        PretrainedLLM {
            name: "gpt2-medium".to_string(),
            url: "https://huggingface.co/axonml-ml/gpt2-medium/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 1_420_000_000,
            vocab_size: 50257,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            max_seq_len: 1024,
            num_parameters: 355_000_000,
            architecture: "GPT-2".to_string(),
            dataset: "WebText".to_string(),
        },
    );

    registry.insert(
        "gpt2-large".to_string(),
        PretrainedLLM {
            name: "gpt2-large".to_string(),
            url: "https://huggingface.co/axonml-ml/gpt2-large/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 3_100_000_000,
            vocab_size: 50257,
            hidden_size: 1280,
            num_layers: 36,
            num_heads: 20,
            max_seq_len: 1024,
            num_parameters: 774_000_000,
            architecture: "GPT-2".to_string(),
            dataset: "WebText".to_string(),
        },
    );

    registry.insert(
        "gpt2-xl".to_string(),
        PretrainedLLM {
            name: "gpt2-xl".to_string(),
            url: "https://huggingface.co/axonml-ml/gpt2-xl/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 6_200_000_000,
            vocab_size: 50257,
            hidden_size: 1600,
            num_layers: 48,
            num_heads: 25,
            max_seq_len: 1024,
            num_parameters: 1_558_000_000,
            architecture: "GPT-2".to_string(),
            dataset: "WebText".to_string(),
        },
    );

    // =========================================================================
    // DistilBERT
    // =========================================================================

    registry.insert(
        "distilbert-base-uncased".to_string(),
        PretrainedLLM {
            name: "distilbert-base-uncased".to_string(),
            url: "https://huggingface.co/axonml-ml/distilbert-base-uncased/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 265_000_000,
            vocab_size: 30522,
            hidden_size: 768,
            num_layers: 6,
            num_heads: 12,
            max_seq_len: 512,
            num_parameters: 66_000_000,
            architecture: "DistilBERT".to_string(),
            dataset: "Wikipedia + BookCorpus".to_string(),
        },
    );

    // =========================================================================
    // RoBERTa
    // =========================================================================

    registry.insert(
        "roberta-base".to_string(),
        PretrainedLLM {
            name: "roberta-base".to_string(),
            url: "https://huggingface.co/axonml-ml/roberta-base/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 500_000_000,
            vocab_size: 50265,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_seq_len: 512,
            num_parameters: 125_000_000,
            architecture: "RoBERTa".to_string(),
            dataset: "OpenWebText + Others".to_string(),
        },
    );

    registry.insert(
        "roberta-large".to_string(),
        PretrainedLLM {
            name: "roberta-large".to_string(),
            url: "https://huggingface.co/axonml-ml/roberta-large/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 1_420_000_000,
            vocab_size: 50265,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            max_seq_len: 512,
            num_parameters: 355_000_000,
            architecture: "RoBERTa".to_string(),
            dataset: "OpenWebText + Others".to_string(),
        },
    );

    // =========================================================================
    // ALBERT
    // =========================================================================

    registry.insert(
        "albert-base-v2".to_string(),
        PretrainedLLM {
            name: "albert-base-v2".to_string(),
            url: "https://huggingface.co/axonml-ml/albert-base-v2/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 47_000_000,
            vocab_size: 30000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_seq_len: 512,
            num_parameters: 12_000_000,
            architecture: "ALBERT".to_string(),
            dataset: "Wikipedia + BookCorpus".to_string(),
        },
    );

    registry.insert(
        "albert-large-v2".to_string(),
        PretrainedLLM {
            name: "albert-large-v2".to_string(),
            url: "https://huggingface.co/axonml-ml/albert-large-v2/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 70_000_000,
            vocab_size: 30000,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            max_seq_len: 512,
            num_parameters: 18_000_000,
            architecture: "ALBERT".to_string(),
            dataset: "Wikipedia + BookCorpus".to_string(),
        },
    );

    // =========================================================================
    // LLaMA Family
    // =========================================================================

    registry.insert(
        "llama-2-7b".to_string(),
        PretrainedLLM {
            name: "llama-2-7b".to_string(),
            url: "https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 13_500_000_000,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            max_seq_len: 4096,
            num_parameters: 6_738_000_000,
            architecture: "LLaMA".to_string(),
            dataset: "Web crawl + curated data".to_string(),
        },
    );

    registry.insert(
        "llama-2-13b".to_string(),
        PretrainedLLM {
            name: "llama-2-13b".to_string(),
            url: "https://huggingface.co/meta-llama/Llama-2-13b-hf/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 26_000_000_000,
            vocab_size: 32000,
            hidden_size: 5120,
            num_layers: 40,
            num_heads: 40,
            max_seq_len: 4096,
            num_parameters: 13_016_000_000,
            architecture: "LLaMA".to_string(),
            dataset: "Web crawl + curated data".to_string(),
        },
    );

    registry.insert(
        "tinyllama-1.1b".to_string(),
        PretrainedLLM {
            name: "tinyllama-1.1b".to_string(),
            url: "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 2_200_000_000,
            vocab_size: 32000,
            hidden_size: 2048,
            num_layers: 22,
            num_heads: 32,
            max_seq_len: 2048,
            num_parameters: 1_100_000_000,
            architecture: "LLaMA".to_string(),
            dataset: "SlimPajama + StarCoder".to_string(),
        },
    );

    // =========================================================================
    // Mistral Family
    // =========================================================================

    registry.insert(
        "mistral-7b".to_string(),
        PretrainedLLM {
            name: "mistral-7b".to_string(),
            url: "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 14_500_000_000,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            max_seq_len: 8192,
            num_parameters: 7_241_000_000,
            architecture: "Mistral".to_string(),
            dataset: "Web data".to_string(),
        },
    );

    registry.insert(
        "mistral-7b-instruct".to_string(),
        PretrainedLLM {
            name: "mistral-7b-instruct".to_string(),
            url: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 14_500_000_000,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            max_seq_len: 32768,
            num_parameters: 7_241_000_000,
            architecture: "Mistral".to_string(),
            dataset: "Web data + instruction tuning".to_string(),
        },
    );

    // =========================================================================
    // Phi Family (Microsoft)
    // =========================================================================

    registry.insert(
        "phi-2".to_string(),
        PretrainedLLM {
            name: "phi-2".to_string(),
            url: "https://huggingface.co/microsoft/phi-2/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 5_600_000_000,
            vocab_size: 51200,
            hidden_size: 2560,
            num_layers: 32,
            num_heads: 32,
            max_seq_len: 2048,
            num_parameters: 2_780_000_000,
            architecture: "Phi".to_string(),
            dataset: "Synthetic + Web data".to_string(),
        },
    );

    registry.insert(
        "phi-1.5".to_string(),
        PretrainedLLM {
            name: "phi-1.5".to_string(),
            url: "https://huggingface.co/microsoft/phi-1_5/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 2_800_000_000,
            vocab_size: 51200,
            hidden_size: 2048,
            num_layers: 24,
            num_heads: 32,
            max_seq_len: 2048,
            num_parameters: 1_300_000_000,
            architecture: "Phi".to_string(),
            dataset: "Synthetic textbooks".to_string(),
        },
    );

    // =========================================================================
    // Qwen Family (Alibaba)
    // =========================================================================

    registry.insert(
        "qwen-1.8b".to_string(),
        PretrainedLLM {
            name: "qwen-1.8b".to_string(),
            url: "https://huggingface.co/Qwen/Qwen-1_8B/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 3_800_000_000,
            vocab_size: 151936,
            hidden_size: 2048,
            num_layers: 24,
            num_heads: 16,
            max_seq_len: 8192,
            num_parameters: 1_800_000_000,
            architecture: "Qwen".to_string(),
            dataset: "Web data + curated".to_string(),
        },
    );

    registry.insert(
        "qwen-7b".to_string(),
        PretrainedLLM {
            name: "qwen-7b".to_string(),
            url: "https://huggingface.co/Qwen/Qwen-7B/resolve/main/model.safetensors".to_string(),
            checksum: None,
            size_bytes: 15_000_000_000,
            vocab_size: 151936,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            max_seq_len: 8192,
            num_parameters: 7_720_000_000,
            architecture: "Qwen".to_string(),
            dataset: "Web data + curated".to_string(),
        },
    );

    registry
}

// =============================================================================
// Weight Management
// =============================================================================

/// Check if pretrained weights are cached.
pub fn is_cached(model_name: &str) -> bool {
    let path = cache_dir().join(format!("{}.safetensors", model_name));
    path.exists()
}

/// Get cached weight path.
pub fn cached_path(model_name: &str) -> PathBuf {
    cache_dir().join(format!("{}.safetensors", model_name))
}

/// Download pretrained weights if not cached.
///
/// # Arguments
/// * `model_name` - Name of the model (e.g., "bert-base-uncased")
/// * `force` - Force re-download even if cached
///
/// # Returns
/// Path to the downloaded weights file
pub fn download_weights(model_name: &str, force: bool) -> HubResult<PathBuf> {
    let registry = llm_registry();
    let model_info = registry
        .get(model_name)
        .ok_or_else(|| HubError::ModelNotFound(model_name.to_string()))?;

    let cache_path = cached_path(model_name);

    // Return cached path if exists and not forcing
    if cache_path.exists() && !force {
        return Ok(cache_path);
    }

    // Ensure cache directory exists
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Download weights
    println!(
        "Downloading {} ({:.1} MB, {} parameters)...",
        model_name,
        model_info.size_bytes as f64 / 1_000_000.0,
        format_params(model_info.num_parameters)
    );

    let response = reqwest::blocking::get(&model_info.url)
        .map_err(|e| HubError::NetworkError(e.to_string()))?;

    if !response.status().is_success() {
        return Err(HubError::NetworkError(format!(
            "HTTP {}: {}",
            response.status(),
            model_info.url
        )));
    }

    let bytes = response
        .bytes()
        .map_err(|e| HubError::NetworkError(e.to_string()))?;

    let mut file = File::create(&cache_path)?;
    file.write_all(&bytes)?;

    println!("Downloaded to {:?}", cache_path);

    Ok(cache_path)
}

/// Format parameter count for display.
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else {
        format!("{}", params)
    }
}

/// List available pretrained LLM models.
pub fn list_models() -> Vec<PretrainedLLM> {
    llm_registry().into_values().collect()
}

/// Get info for a specific model.
pub fn model_info(name: &str) -> Option<PretrainedLLM> {
    llm_registry().get(name).cloned()
}

/// Get models filtered by architecture.
pub fn models_by_architecture(arch: &str) -> Vec<PretrainedLLM> {
    llm_registry()
        .into_values()
        .filter(|m| m.architecture.eq_ignore_ascii_case(arch))
        .collect()
}

/// Get models that fit within a parameter budget.
pub fn models_by_max_params(max_params: u64) -> Vec<PretrainedLLM> {
    let mut models: Vec<_> = llm_registry()
        .into_values()
        .filter(|m| m.num_parameters <= max_params)
        .collect();
    models.sort_by_key(|m| m.num_parameters);
    models
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_registry() {
        let registry = llm_registry();
        assert!(registry.contains_key("bert-base-uncased"));
        assert!(registry.contains_key("gpt2"));
        assert!(registry.contains_key("distilbert-base-uncased"));
    }

    #[test]
    fn test_cache_dir() {
        let dir = cache_dir();
        assert!(dir.to_string_lossy().contains("axonml"));
        assert!(dir.to_string_lossy().contains("llm"));
    }

    #[test]
    fn test_list_models() {
        let models = list_models();
        assert!(!models.is_empty());
        assert!(models.len() >= 10);
    }

    #[test]
    fn test_model_info() {
        let info = model_info("bert-base-uncased");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.hidden_size, 768);
        assert_eq!(info.num_layers, 12);
        assert_eq!(info.vocab_size, 30522);
    }

    #[test]
    fn test_gpt2_info() {
        let info = model_info("gpt2");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.hidden_size, 768);
        assert_eq!(info.num_layers, 12);
        assert_eq!(info.vocab_size, 50257);
        assert_eq!(info.max_seq_len, 1024);
    }

    #[test]
    fn test_models_by_architecture() {
        let bert_models = models_by_architecture("BERT");
        assert!(!bert_models.is_empty());
        for model in &bert_models {
            assert_eq!(model.architecture, "BERT");
        }

        let gpt2_models = models_by_architecture("GPT-2");
        assert!(!gpt2_models.is_empty());
        for model in &gpt2_models {
            assert_eq!(model.architecture, "GPT-2");
        }
    }

    #[test]
    fn test_models_by_max_params() {
        let small_models = models_by_max_params(100_000_000);
        assert!(!small_models.is_empty());
        for model in &small_models {
            assert!(model.num_parameters <= 100_000_000);
        }
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(1_500_000_000), "1.5B");
        assert_eq!(format_params(110_000_000), "110.0M");
        assert_eq!(format_params(4_400_000), "4.4M");
        assert_eq!(format_params(1_500), "1.5K");
    }

    #[test]
    fn test_cached_path() {
        let path = cached_path("bert-base-uncased");
        assert!(path.to_string_lossy().contains("bert-base-uncased"));
        assert!(path.to_string_lossy().ends_with(".safetensors"));
    }

    #[test]
    fn test_model_urls() {
        let registry = llm_registry();
        for (name, model) in &registry {
            assert!(!model.url.is_empty(), "Model {} has empty URL", name);
            assert!(
                model.url.starts_with("https://"),
                "Model {} URL should be HTTPS",
                name
            );
            assert!(model.size_bytes > 0, "Model {} has zero size", name);
            assert!(
                model.num_parameters > 0,
                "Model {} has zero parameters",
                name
            );
        }
    }
}
