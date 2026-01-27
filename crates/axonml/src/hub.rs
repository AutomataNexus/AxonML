//! Unified Model Hub - Central Registry for All Pretrained Models
//!
//! Provides a unified interface for discovering and loading pretrained models
//! across all domains (vision, text, audio).
//!
//! # Example
//! ```rust,ignore
//! use axonml::hub::{list_all_models, search_models, ModelCategory};
//!
//! // List all available models
//! let models = list_all_models();
//!
//! // Search for specific models
//! let bert_models = search_models("bert");
//!
//! // Filter by category
//! let vision_models = models_by_category(ModelCategory::Vision);
//! ```
//!
//! @version 0.1.0

// HashMap is used by feature-gated functions
#[allow(unused_imports)]
use std::collections::HashMap;

// =============================================================================
// Model Categories
// =============================================================================

/// Category of pretrained model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    /// Vision models (ResNet, VGG, ViT, etc.)
    Vision,
    /// Language models (BERT, GPT-2, LLaMA, etc.)
    Language,
    /// Audio models (Wav2Vec, Whisper, etc.)
    Audio,
    /// Multimodal models (CLIP, etc.)
    Multimodal,
}

impl std::fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::Vision => write!(f, "Vision"),
            ModelCategory::Language => write!(f, "Language"),
            ModelCategory::Audio => write!(f, "Audio"),
            ModelCategory::Multimodal => write!(f, "Multimodal"),
        }
    }
}

// =============================================================================
// Unified Model Info
// =============================================================================

/// Unified model information across all categories.
#[derive(Debug, Clone)]
pub struct UnifiedModelInfo {
    /// Model name
    pub name: String,
    /// Category
    pub category: ModelCategory,
    /// Architecture (e.g., "ResNet", "BERT", "LLaMA")
    pub architecture: String,
    /// Number of parameters
    pub num_parameters: u64,
    /// File size in bytes
    pub size_bytes: u64,
    /// Download URL
    pub url: String,
    /// Training dataset
    pub dataset: String,
    /// Description
    pub description: String,
    /// Tags for search
    pub tags: Vec<String>,
}

impl UnifiedModelInfo {
    /// Returns size in megabytes.
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / 1_000_000.0
    }

    /// Returns formatted parameter count.
    pub fn params_str(&self) -> String {
        if self.num_parameters >= 1_000_000_000 {
            format!("{:.1}B", self.num_parameters as f64 / 1_000_000_000.0)
        } else if self.num_parameters >= 1_000_000 {
            format!("{:.1}M", self.num_parameters as f64 / 1_000_000.0)
        } else if self.num_parameters >= 1_000 {
            format!("{:.1}K", self.num_parameters as f64 / 1_000.0)
        } else {
            format!("{}", self.num_parameters)
        }
    }
}

// =============================================================================
// Registry Functions
// =============================================================================

/// Get all available models across all categories.
#[cfg(all(feature = "vision", feature = "llm"))]
pub fn list_all_models() -> Vec<UnifiedModelInfo> {
    let mut models = Vec::new();

    // Add vision models
    #[cfg(feature = "vision")]
    {
        for (name, info) in axonml_vision::hub::model_registry() {
            models.push(UnifiedModelInfo {
                name: name.clone(),
                category: ModelCategory::Vision,
                architecture: extract_architecture(&name),
                num_parameters: estimate_params_from_size(info.size_bytes),
                size_bytes: info.size_bytes,
                url: info.url.clone(),
                dataset: info.dataset.clone(),
                description: format!(
                    "{} trained on {} (Top-1: {:.1}%)",
                    name, info.dataset, info.accuracy
                ),
                tags: generate_vision_tags(&name, &info),
            });
        }
    }

    // Add LLM models
    #[cfg(feature = "llm")]
    {
        for (name, info) in axonml_llm::hub::llm_registry() {
            models.push(UnifiedModelInfo {
                name: name.clone(),
                category: ModelCategory::Language,
                architecture: info.architecture.clone(),
                num_parameters: info.num_parameters,
                size_bytes: info.size_bytes,
                url: info.url.clone(),
                dataset: info.dataset.clone(),
                description: format!(
                    "{} ({} params, {} layers)",
                    name,
                    format_params(info.num_parameters),
                    info.num_layers
                ),
                tags: generate_llm_tags(&name, &info),
            });
        }
    }

    models
}

/// Search models by name or tag.
#[cfg(all(feature = "vision", feature = "llm"))]
pub fn search_models(query: &str) -> Vec<UnifiedModelInfo> {
    let query_lower = query.to_lowercase();
    list_all_models()
        .into_iter()
        .filter(|m| {
            m.name.to_lowercase().contains(&query_lower)
                || m.architecture.to_lowercase().contains(&query_lower)
                || m.tags
                    .iter()
                    .any(|t| t.to_lowercase().contains(&query_lower))
        })
        .collect()
}

/// Get models by category.
#[cfg(all(feature = "vision", feature = "llm"))]
pub fn models_by_category(category: ModelCategory) -> Vec<UnifiedModelInfo> {
    list_all_models()
        .into_iter()
        .filter(|m| m.category == category)
        .collect()
}

/// Get models within a size budget (in MB).
#[cfg(all(feature = "vision", feature = "llm"))]
pub fn models_by_max_size_mb(max_mb: f64) -> Vec<UnifiedModelInfo> {
    let max_bytes = (max_mb * 1_000_000.0) as u64;
    let mut models: Vec<_> = list_all_models()
        .into_iter()
        .filter(|m| m.size_bytes <= max_bytes)
        .collect();
    models.sort_by_key(|m| m.size_bytes);
    models
}

/// Get models within a parameter budget.
#[cfg(all(feature = "vision", feature = "llm"))]
pub fn models_by_max_params(max_params: u64) -> Vec<UnifiedModelInfo> {
    let mut models: Vec<_> = list_all_models()
        .into_iter()
        .filter(|m| m.num_parameters <= max_params)
        .collect();
    models.sort_by_key(|m| m.num_parameters);
    models
}

/// Get recommended models for a task.
#[cfg(all(feature = "vision", feature = "llm"))]
pub fn recommended_models(task: &str) -> Vec<UnifiedModelInfo> {
    let task_lower = task.to_lowercase();

    if task_lower.contains("image")
        || task_lower.contains("vision")
        || task_lower.contains("classify")
    {
        // Image classification - recommend efficient models first
        let mut models = models_by_category(ModelCategory::Vision);
        models.sort_by(|a, b| {
            // Prefer models with good accuracy/size ratio
            let ratio_a = a.size_bytes as f64;
            let ratio_b = b.size_bytes as f64;
            ratio_a
                .partial_cmp(&ratio_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        models.truncate(5);
        models
    } else if task_lower.contains("text")
        || task_lower.contains("nlp")
        || task_lower.contains("language")
    {
        // NLP tasks - recommend smaller models first
        let mut models = models_by_category(ModelCategory::Language);
        models.sort_by_key(|m| m.num_parameters);
        models.truncate(5);
        models
    } else if task_lower.contains("chat")
        || task_lower.contains("instruct")
        || task_lower.contains("generate")
    {
        // Text generation - recommend instruction-tuned models
        search_models("instruct")
    } else {
        // Default - return smallest models from each category
        let mut result = Vec::new();
        for category in [ModelCategory::Vision, ModelCategory::Language] {
            let mut cat_models = models_by_category(category);
            cat_models.sort_by_key(|m| m.size_bytes);
            if let Some(m) = cat_models.into_iter().next() {
                result.push(m);
            }
        }
        result
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn extract_architecture(name: &str) -> String {
    if name.starts_with("resnet") {
        "ResNet".to_string()
    } else if name.starts_with("vgg") {
        "VGG".to_string()
    } else if name.starts_with("mobilenet") {
        "MobileNet".to_string()
    } else if name.starts_with("efficientnet") {
        "EfficientNet".to_string()
    } else if name.starts_with("densenet") {
        "DenseNet".to_string()
    } else if name.starts_with("vit") {
        "ViT".to_string()
    } else if name.starts_with("swin") {
        "Swin".to_string()
    } else if name.starts_with("convnext") {
        "ConvNeXt".to_string()
    } else {
        "Unknown".to_string()
    }
}

fn estimate_params_from_size(size_bytes: u64) -> u64 {
    // Rough estimate: 4 bytes per float32 parameter
    size_bytes / 4
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else {
        format!("{:.1}K", params as f64 / 1_000.0)
    }
}

#[cfg(feature = "vision")]
fn generate_vision_tags(name: &str, _info: &axonml_vision::hub::PretrainedModel) -> Vec<String> {
    let mut tags = vec![
        "vision".to_string(),
        "image".to_string(),
        "classification".to_string(),
    ];

    if name.contains("mobile") {
        tags.push("mobile".to_string());
        tags.push("efficient".to_string());
    }
    if name.contains("efficient") {
        tags.push("efficient".to_string());
    }
    if name.contains("vit") || name.contains("swin") {
        tags.push("transformer".to_string());
    }

    tags
}

#[cfg(feature = "llm")]
fn generate_llm_tags(name: &str, info: &axonml_llm::hub::PretrainedLLM) -> Vec<String> {
    let mut tags = vec![
        "language".to_string(),
        "nlp".to_string(),
        "text".to_string(),
    ];

    tags.push(info.architecture.to_lowercase());

    if name.contains("instruct") || name.contains("chat") {
        tags.push("instruct".to_string());
        tags.push("chat".to_string());
    }
    if info.num_parameters < 1_000_000_000 {
        tags.push("small".to_string());
    } else if info.num_parameters < 10_000_000_000 {
        tags.push("medium".to_string());
    } else {
        tags.push("large".to_string());
    }

    tags
}

// =============================================================================
// Model Benchmark Utilities
// =============================================================================

/// Results from model benchmarking.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Model name
    pub model_name: String,
    /// Average inference time in milliseconds
    pub avg_latency_ms: f64,
    /// 95th percentile latency
    pub p95_latency_ms: f64,
    /// Throughput (samples/second)
    pub throughput: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Number of iterations run
    pub iterations: usize,
}

impl BenchmarkResult {
    /// Create a new benchmark result.
    pub fn new(model_name: &str, latencies_ms: &[f64], peak_memory_bytes: u64) -> Self {
        let iterations = latencies_ms.len();
        let avg_latency_ms = if iterations > 0 {
            latencies_ms.iter().sum::<f64>() / iterations as f64
        } else {
            0.0
        };

        // Calculate p95
        let mut sorted = latencies_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx = (iterations as f64 * 0.95) as usize;
        let p95_latency_ms = sorted
            .get(p95_idx.min(iterations.saturating_sub(1)))
            .copied()
            .unwrap_or(0.0);

        let throughput = if avg_latency_ms > 0.0 {
            1000.0 / avg_latency_ms
        } else {
            0.0
        };

        Self {
            model_name: model_name.to_string(),
            avg_latency_ms,
            p95_latency_ms,
            throughput,
            peak_memory_bytes,
            iterations,
        }
    }

    /// Print a formatted summary.
    pub fn print_summary(&self) {
        println!("Benchmark: {}", self.model_name);
        println!("  Iterations: {}", self.iterations);
        println!("  Avg latency: {:.2} ms", self.avg_latency_ms);
        println!("  P95 latency: {:.2} ms", self.p95_latency_ms);
        println!("  Throughput: {:.1} samples/sec", self.throughput);
        println!(
            "  Peak memory: {:.1} MB",
            self.peak_memory_bytes as f64 / 1_000_000.0
        );
    }
}

/// Compare multiple benchmark results.
pub fn compare_benchmarks(results: &[BenchmarkResult]) {
    if results.is_empty() {
        println!("No benchmark results to compare.");
        return;
    }

    println!(
        "\n{:<25} {:>12} {:>12} {:>14} {:>12}",
        "Model", "Avg (ms)", "P95 (ms)", "Throughput", "Memory (MB)"
    );
    println!("{}", "-".repeat(80));

    for result in results {
        println!(
            "{:<25} {:>12.2} {:>12.2} {:>12.1}/s {:>12.1}",
            result.model_name,
            result.avg_latency_ms,
            result.p95_latency_ms,
            result.throughput,
            result.peak_memory_bytes as f64 / 1_000_000.0
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_category_display() {
        assert_eq!(format!("{}", ModelCategory::Vision), "Vision");
        assert_eq!(format!("{}", ModelCategory::Language), "Language");
    }

    #[test]
    fn test_unified_model_info_size() {
        let info = UnifiedModelInfo {
            name: "test".to_string(),
            category: ModelCategory::Vision,
            architecture: "Test".to_string(),
            num_parameters: 1_500_000_000,
            size_bytes: 6_000_000_000,
            url: "https://example.com".to_string(),
            dataset: "Test".to_string(),
            description: "Test model".to_string(),
            tags: vec!["test".to_string()],
        };

        assert!((info.size_mb() - 6000.0).abs() < 0.1);
        assert_eq!(info.params_str(), "1.5B");
    }

    #[test]
    fn test_benchmark_result() {
        let latencies = vec![10.0, 12.0, 11.0, 15.0, 10.5];
        let result = BenchmarkResult::new("test_model", &latencies, 100_000_000);

        assert_eq!(result.iterations, 5);
        assert!(result.avg_latency_ms > 0.0);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_extract_architecture() {
        assert_eq!(extract_architecture("resnet50"), "ResNet");
        assert_eq!(extract_architecture("vgg16"), "VGG");
        assert_eq!(extract_architecture("mobilenet_v2"), "MobileNet");
        assert_eq!(extract_architecture("vit_b_16"), "ViT");
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(1_500_000_000), "1.5B");
        assert_eq!(format_params(110_000_000), "110.0M");
        assert_eq!(format_params(50_000), "50.0K");
    }

    #[cfg(all(feature = "vision", feature = "llm"))]
    #[test]
    fn test_list_all_models() {
        let models = list_all_models();
        assert!(!models.is_empty());
    }

    #[cfg(all(feature = "vision", feature = "llm"))]
    #[test]
    fn test_search_models() {
        let results = search_models("resnet");
        for model in &results {
            assert!(
                model.name.to_lowercase().contains("resnet")
                    || model.architecture.to_lowercase().contains("resnet")
            );
        }
    }
}
