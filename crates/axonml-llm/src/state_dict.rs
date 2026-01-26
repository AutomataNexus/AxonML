//! State Dictionary Loading
//!
//! Provides functionality to load pretrained weights into models.

use std::collections::HashMap;
use axonml_tensor::Tensor;

use crate::error::{LLMError, LLMResult};

// =============================================================================
// State Dict Trait
// =============================================================================

/// Trait for models that can load state dictionaries.
pub trait LoadStateDict {
    /// Load weights from a state dictionary.
    ///
    /// # Arguments
    /// * `state_dict` - Map of parameter names to tensors
    /// * `strict` - If true, error on missing/unexpected keys
    ///
    /// # Returns
    /// List of missing and unexpected keys
    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> LLMResult<LoadResult>;

    /// Get all parameter names in this model.
    fn state_dict_keys(&self) -> Vec<String>;
}

/// Result of loading a state dict.
#[derive(Debug, Default)]
pub struct LoadResult {
    /// Keys in state_dict but not in model
    pub unexpected_keys: Vec<String>,
    /// Keys in model but not in state_dict
    pub missing_keys: Vec<String>,
    /// Number of parameters loaded
    pub loaded_count: usize,
}

impl LoadResult {
    /// Check if load was successful (no missing keys in strict mode).
    pub fn is_success(&self, strict: bool) -> bool {
        !strict || self.missing_keys.is_empty()
    }

    /// Print a summary of the load result.
    pub fn print_summary(&self) {
        println!("Loaded {} parameters", self.loaded_count);
        if !self.missing_keys.is_empty() {
            println!("Missing keys ({}):", self.missing_keys.len());
            for key in &self.missing_keys {
                println!("  - {}", key);
            }
        }
        if !self.unexpected_keys.is_empty() {
            println!("Unexpected keys ({}):", self.unexpected_keys.len());
            for key in &self.unexpected_keys {
                println!("  - {}", key);
            }
        }
    }
}

// =============================================================================
// Weight Name Mapping
// =============================================================================

/// Maps HuggingFace weight names to AxonML parameter names.
pub fn map_hf_to_axonml(hf_name: &str, arch: &str) -> String {
    // Remove common prefixes
    let name = hf_name
        .strip_prefix("model.")
        .or_else(|| hf_name.strip_prefix("transformer."))
        .unwrap_or(hf_name);

    match arch {
        "llama" | "mistral" => map_llama_weights(name),
        "phi" => map_phi_weights(name),
        _ => name.to_string(),
    }
}

fn map_llama_weights(name: &str) -> String {
    // HuggingFace LLaMA format -> AxonML format
    // Most names are already compatible, just need minor adjustments
    name.replace("self_attn.", "attention.")
        .replace("input_layernorm", "input_norm")
        .replace("post_attention_layernorm", "post_attn_norm")
}

fn map_phi_weights(name: &str) -> String {
    // Phi uses different naming conventions
    name.replace("self_attn.", "attention.")
        .replace("fc1", "up_proj")
        .replace("fc2", "down_proj")
}

/// Maps AxonML parameter names back to HuggingFace format.
pub fn map_axonml_to_hf(axonml_name: &str, arch: &str) -> String {
    match arch {
        "llama" | "mistral" => {
            let name = axonml_name
                .replace("attention.", "self_attn.")
                .replace("input_norm", "input_layernorm")
                .replace("post_attn_norm", "post_attention_layernorm");
            format!("model.{}", name)
        }
        "phi" => {
            let name = axonml_name
                .replace("attention.", "self_attn.")
                .replace("up_proj", "fc1")
                .replace("down_proj", "fc2");
            format!("model.{}", name)
        }
        _ => axonml_name.to_string(),
    }
}

// =============================================================================
// Load Helpers
// =============================================================================

/// Load state dict with automatic key mapping.
pub fn load_with_mapping<M: LoadStateDict>(
    model: &mut M,
    weights: &HashMap<String, Tensor<f32>>,
    arch: &str,
    strict: bool,
) -> LLMResult<LoadResult> {
    // Map HF names to AxonML names
    let mapped: HashMap<String, Tensor<f32>> = weights
        .iter()
        .map(|(k, v)| (map_hf_to_axonml(k, arch), v.clone()))
        .collect();

    model.load_state_dict(&mapped, strict)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_mapping() {
        assert_eq!(
            map_hf_to_axonml("model.layers.0.self_attn.q_proj.weight", "llama"),
            "layers.0.attention.q_proj.weight"
        );
    }

    #[test]
    fn test_load_result() {
        let mut result = LoadResult::default();
        result.loaded_count = 10;
        assert!(result.is_success(true));

        result.missing_keys.push("test".to_string());
        assert!(!result.is_success(true));
        assert!(result.is_success(false));
    }
}
