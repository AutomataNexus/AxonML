//! HuggingFace Model Loader
//!
//! Downloads and loads pretrained weights from HuggingFace Hub into AxonML models.
//!
//! # Example
//! ```rust,ignore
//! use axonml_llm::hf_loader::HFLoader;
//! use axonml_llm::{LLaMA, LLaMAConfig};
//!
//! // Load weights from HuggingFace
//! let loader = HFLoader::new("meta-llama/Llama-2-7b-hf")?;
//! let mut model = LLaMA::new(&LLaMAConfig::llama2_7b());
//! loader.load_into_llama(&mut model)?;
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use axonml_tensor::Tensor;
use indicatif::{ProgressBar, ProgressStyle};

use crate::error::{LLMError, LLMResult};

// =============================================================================
// HuggingFace Hub API
// =============================================================================

const HF_API_BASE: &str = "https://huggingface.co";

/// HuggingFace model loader.
pub struct HFLoader {
    /// Model ID (e.g., "meta-llama/Llama-2-7b-hf")
    model_id: String,
    /// Local cache directory
    cache_dir: PathBuf,
    /// Loaded tensors (name -> data)
    tensors: HashMap<String, TensorInfo>,
    /// Model config from HuggingFace
    config: Option<serde_json::Value>,
}

/// Information about a loaded tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Tensor data (f32)
    pub data: Vec<f32>,
    /// Original dtype from file
    pub dtype: String,
}

impl HFLoader {
    /// Create a new loader for a HuggingFace model.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
    pub fn new(model_id: &str) -> LLMResult<Self> {
        let cache_dir = Self::get_cache_dir(model_id);
        fs::create_dir_all(&cache_dir).map_err(|e| LLMError::IoError(e.to_string()))?;

        Ok(Self {
            model_id: model_id.to_string(),
            cache_dir,
            tensors: HashMap::new(),
            config: None,
        })
    }

    /// Load from a local directory instead of HuggingFace.
    pub fn from_local(path: &str) -> LLMResult<Self> {
        let cache_dir = PathBuf::from(path);
        if !cache_dir.exists() {
            return Err(LLMError::ModelNotFound(path.to_string()));
        }

        Ok(Self {
            model_id: path.to_string(),
            cache_dir,
            tensors: HashMap::new(),
            config: None,
        })
    }

    /// Get the cache directory for a model.
    fn get_cache_dir(model_id: &str) -> PathBuf {
        let base = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("axonml")
            .join("hub");

        // Sanitize model_id for filesystem
        let safe_id = model_id.replace('/', "--");
        base.join(safe_id)
    }

    /// Download a file from HuggingFace Hub.
    pub fn download_file(&self, filename: &str) -> LLMResult<PathBuf> {
        let local_path = self.cache_dir.join(filename);

        // Return if already cached
        if local_path.exists() {
            println!("Using cached: {}", local_path.display());
            return Ok(local_path);
        }

        let url = format!(
            "{}/{}/resolve/main/{}",
            HF_API_BASE, self.model_id, filename
        );

        println!("Downloading: {}", url);

        // Create progress bar
        let pb = ProgressBar::new(0);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(format!("Downloading {}", filename));

        // Download with reqwest blocking
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(&url)
            .send()
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(LLMError::NetworkError(format!(
                "Failed to download {}: HTTP {}",
                filename,
                response.status()
            )));
        }

        // Get content length for progress
        if let Some(len) = response.content_length() {
            pb.set_length(len);
        }

        // Read and write in chunks
        let bytes = response.bytes().map_err(|e| LLMError::NetworkError(e.to_string()))?;
        pb.set_position(bytes.len() as u64);

        let mut file = File::create(&local_path).map_err(|e| LLMError::IoError(e.to_string()))?;
        file.write_all(&bytes).map_err(|e| LLMError::IoError(e.to_string()))?;

        pb.finish_with_message(format!("Downloaded {}", filename));

        Ok(local_path)
    }

    /// Load the model config (config.json).
    pub fn load_config(&mut self) -> LLMResult<serde_json::Value> {
        if let Some(ref config) = self.config {
            return Ok(config.clone());
        }

        let path = self.download_file("config.json")?;
        let content = fs::read_to_string(&path).map_err(|e| LLMError::IoError(e.to_string()))?;
        let config: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| LLMError::ParseError(e.to_string()))?;

        self.config = Some(config.clone());
        Ok(config)
    }

    /// Load tensors from safetensors file(s).
    pub fn load_tensors(&mut self) -> LLMResult<()> {
        // Try single file first
        let single_file = self.cache_dir.join("model.safetensors");
        if single_file.exists() || self.download_file("model.safetensors").is_ok() {
            return self.load_safetensors_file("model.safetensors");
        }

        // Try sharded files (model-00001-of-00002.safetensors, etc.)
        let index_path = self.download_file("model.safetensors.index.json")?;
        let index_content =
            fs::read_to_string(&index_path).map_err(|e| LLMError::IoError(e.to_string()))?;
        let index: serde_json::Value =
            serde_json::from_str(&index_content).map_err(|e| LLMError::ParseError(e.to_string()))?;

        // Get list of shard files
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| LLMError::ParseError("Invalid index file".to_string()))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();

        // Download and load each shard
        for shard in &shard_files {
            self.download_file(shard)?;
            self.load_safetensors_file(shard)?;
        }

        Ok(())
    }

    /// Load a single safetensors file.
    fn load_safetensors_file(&mut self, filename: &str) -> LLMResult<()> {
        let path = self.cache_dir.join(filename);
        let data = fs::read(&path).map_err(|e| LLMError::IoError(e.to_string()))?;

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| LLMError::ParseError(e.to_string()))?;

        for (name, tensor) in tensors.tensors() {
            let shape: Vec<usize> = tensor.shape().to_vec();
            let dtype = format!("{:?}", tensor.dtype());

            // Convert to f32
            let data = self.convert_tensor_to_f32(&tensor)?;

            self.tensors.insert(
                name.to_string(),
                TensorInfo { shape, data, dtype },
            );
        }

        println!("Loaded {} tensors from {}", tensors.len(), filename);
        Ok(())
    }

    /// Convert tensor data to f32.
    fn convert_tensor_to_f32(&self, tensor: &safetensors::tensor::TensorView) -> LLMResult<Vec<f32>> {
        use safetensors::Dtype;

        let data = tensor.data();

        match tensor.dtype() {
            Dtype::F32 => {
                // Already f32
                Ok(data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            Dtype::F16 => {
                // Convert f16 to f32
                Ok(data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect())
            }
            Dtype::BF16 => {
                // Convert bf16 to f32
                Ok(data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half::bf16::from_bits(bits).to_f32()
                    })
                    .collect())
            }
            dtype => Err(LLMError::UnsupportedFormat(format!(
                "Unsupported tensor dtype: {:?}",
                dtype
            ))),
        }
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get a tensor as an AxonML Tensor.
    pub fn get_as_tensor(&self, name: &str) -> LLMResult<Tensor<f32>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            LLMError::WeightNotFound(name.to_string())
        })?;

        Tensor::from_vec(info.data.clone(), &info.shape)
            .map_err(|e| LLMError::TensorError(e.to_string()))
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Print tensor info for debugging.
    pub fn print_tensor_info(&self) {
        println!("\nLoaded tensors:");
        let mut names: Vec<_> = self.tensors.keys().collect();
        names.sort();
        for name in names {
            let info = &self.tensors[name];
            println!("  {} {:?} ({})", name, info.shape, info.dtype);
        }
    }

    /// Get the cache directory path.
    pub fn cache_dir(&self) -> &std::path::Path {
        &self.cache_dir
    }

    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Download a file if it exists (doesn't error on 404).
    pub fn download_file_if_exists(&self, filename: &str) -> LLMResult<bool> {
        match self.download_file(filename) {
            Ok(_) => Ok(true),
            Err(LLMError::NetworkError(msg)) if msg.contains("404") || msg.contains("HTTP 4") => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Get all loaded tensors.
    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }
}

// =============================================================================
// Weight Mapping for Different Architectures
// =============================================================================

/// Maps HuggingFace weight names to AxonML model parameter names.
pub trait WeightMapper {
    /// Map a HuggingFace weight name to AxonML parameter name.
    fn map_name(&self, hf_name: &str) -> Option<String>;

    /// Get all expected weight names for this architecture.
    fn expected_weights(&self) -> Vec<String>;
}

/// Weight mapper for LLaMA models.
pub struct LLaMAWeightMapper {
    num_layers: usize,
}

impl LLaMAWeightMapper {
    /// Create a new LLaMA weight mapper.
    pub fn new(num_layers: usize) -> Self {
        Self { num_layers }
    }
}

impl WeightMapper for LLaMAWeightMapper {
    fn map_name(&self, hf_name: &str) -> Option<String> {
        // HuggingFace LLaMA naming:
        // model.embed_tokens.weight
        // model.layers.0.self_attn.q_proj.weight
        // model.layers.0.self_attn.k_proj.weight
        // model.layers.0.self_attn.v_proj.weight
        // model.layers.0.self_attn.o_proj.weight
        // model.layers.0.mlp.gate_proj.weight
        // model.layers.0.mlp.up_proj.weight
        // model.layers.0.mlp.down_proj.weight
        // model.layers.0.input_layernorm.weight
        // model.layers.0.post_attention_layernorm.weight
        // model.norm.weight
        // lm_head.weight

        let name = hf_name.strip_prefix("model.").unwrap_or(hf_name);

        // Direct mappings
        Some(name.to_string())
    }

    fn expected_weights(&self) -> Vec<String> {
        let mut weights = vec![
            "embed_tokens.weight".to_string(),
            "norm.weight".to_string(),
        ];

        for i in 0..self.num_layers {
            weights.extend([
                format!("layers.{}.self_attn.q_proj.weight", i),
                format!("layers.{}.self_attn.k_proj.weight", i),
                format!("layers.{}.self_attn.v_proj.weight", i),
                format!("layers.{}.self_attn.o_proj.weight", i),
                format!("layers.{}.mlp.gate_proj.weight", i),
                format!("layers.{}.mlp.up_proj.weight", i),
                format!("layers.{}.mlp.down_proj.weight", i),
                format!("layers.{}.input_layernorm.weight", i),
                format!("layers.{}.post_attention_layernorm.weight", i),
            ]);
        }

        weights
    }
}

/// Weight mapper for Mistral models.
pub struct MistralWeightMapper {
    num_layers: usize,
}

impl MistralWeightMapper {
    /// Create a new Mistral weight mapper.
    pub fn new(num_layers: usize) -> Self {
        Self { num_layers }
    }
}

impl WeightMapper for MistralWeightMapper {
    fn map_name(&self, hf_name: &str) -> Option<String> {
        // Mistral uses same naming as LLaMA
        let name = hf_name.strip_prefix("model.").unwrap_or(hf_name);
        Some(name.to_string())
    }

    fn expected_weights(&self) -> Vec<String> {
        // Same as LLaMA
        LLaMAWeightMapper::new(self.num_layers).expected_weights()
    }
}

/// Weight mapper for Phi models.
pub struct PhiWeightMapper {
    num_layers: usize,
}

impl PhiWeightMapper {
    /// Create a new Phi weight mapper.
    pub fn new(num_layers: usize) -> Self {
        Self { num_layers }
    }
}

impl WeightMapper for PhiWeightMapper {
    fn map_name(&self, hf_name: &str) -> Option<String> {
        // Phi naming varies by version
        // Phi-2: model.embed_tokens.weight, model.layers.N.*, model.final_layernorm.*
        // Some use: transformer.embd.wte.weight, transformer.h.N.*

        let name = hf_name
            .strip_prefix("model.")
            .or_else(|| hf_name.strip_prefix("transformer."))
            .unwrap_or(hf_name);

        Some(name.to_string())
    }

    fn expected_weights(&self) -> Vec<String> {
        let mut weights = vec![
            "embed_tokens.weight".to_string(),
            "final_layernorm.weight".to_string(),
        ];

        for i in 0..self.num_layers {
            weights.extend([
                format!("layers.{}.self_attn.q_proj.weight", i),
                format!("layers.{}.self_attn.k_proj.weight", i),
                format!("layers.{}.self_attn.v_proj.weight", i),
                format!("layers.{}.self_attn.dense.weight", i),
                format!("layers.{}.mlp.fc1.weight", i),
                format!("layers.{}.mlp.fc2.weight", i),
                format!("layers.{}.input_layernorm.weight", i),
            ]);
        }

        weights
    }
}

// =============================================================================
// Model Loading Functions
// =============================================================================

/// Load LLaMA weights from HuggingFace.
pub fn load_llama_from_hf(model_id: &str) -> LLMResult<(crate::LLaMAConfig, HashMap<String, Tensor<f32>>)> {
    let mut loader = HFLoader::new(model_id)?;

    // Load config
    let config_json = loader.load_config()?;
    let config = parse_llama_config_from_json(&config_json)?;

    // Load tensors
    loader.load_tensors()?;

    // Map weights
    let mapper = LLaMAWeightMapper::new(config.num_hidden_layers);
    let mut weights = HashMap::new();

    for (hf_name, tensor_info) in &loader.tensors {
        if let Some(mapped_name) = mapper.map_name(hf_name) {
            let tensor = Tensor::from_vec(tensor_info.data.clone(), &tensor_info.shape)
                .map_err(|e| LLMError::TensorError(e.to_string()))?;
            weights.insert(mapped_name, tensor);
        }
    }

    Ok((config, weights))
}

/// Parse LLaMA config from HuggingFace config.json.
pub fn parse_llama_config_from_json(json: &serde_json::Value) -> LLMResult<crate::LLaMAConfig> {
    Ok(crate::LLaMAConfig {
        vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as usize,
        hidden_size: json["hidden_size"].as_u64().unwrap_or(4096) as usize,
        intermediate_size: json["intermediate_size"].as_u64().unwrap_or(11008) as usize,
        num_hidden_layers: json["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_key_value_heads: json["num_key_value_heads"]
            .as_u64()
            .unwrap_or(json["num_attention_heads"].as_u64().unwrap_or(32)) as usize,
        max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(4096) as usize,
        rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
        rope_theta: json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        attention_dropout: 0.0,
        hidden_dropout: 0.0,
    })
}

/// Load Mistral weights from HuggingFace.
pub fn load_mistral_from_hf(model_id: &str) -> LLMResult<(crate::MistralConfig, HashMap<String, Tensor<f32>>)> {
    let mut loader = HFLoader::new(model_id)?;

    let config_json = loader.load_config()?;
    let config = parse_mistral_config(&config_json)?;

    loader.load_tensors()?;

    let mapper = MistralWeightMapper::new(config.num_hidden_layers);
    let mut weights = HashMap::new();

    for (hf_name, tensor_info) in &loader.tensors {
        if let Some(mapped_name) = mapper.map_name(hf_name) {
            let tensor = Tensor::from_vec(tensor_info.data.clone(), &tensor_info.shape)
                .map_err(|e| LLMError::TensorError(e.to_string()))?;
            weights.insert(mapped_name, tensor);
        }
    }

    Ok((config, weights))
}

/// Parse Mistral config from HuggingFace config.json.
fn parse_mistral_config(json: &serde_json::Value) -> LLMResult<crate::MistralConfig> {
    Ok(crate::MistralConfig {
        vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as usize,
        hidden_size: json["hidden_size"].as_u64().unwrap_or(4096) as usize,
        intermediate_size: json["intermediate_size"].as_u64().unwrap_or(14336) as usize,
        num_hidden_layers: json["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
        num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(32) as usize,
        num_key_value_heads: json["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(32768) as usize,
        sliding_window: json["sliding_window"].as_u64().unwrap_or(4096) as usize,
        rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
        rope_theta: json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        attention_dropout: 0.0,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir() {
        let dir = HFLoader::get_cache_dir("meta-llama/Llama-2-7b-hf");
        assert!(dir.to_string_lossy().contains("meta-llama--Llama-2-7b-hf"));
    }

    #[test]
    fn test_llama_weight_mapper() {
        let mapper = LLaMAWeightMapper::new(2);

        assert_eq!(
            mapper.map_name("model.embed_tokens.weight"),
            Some("embed_tokens.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers.0.self_attn.q_proj.weight".to_string())
        );
    }

    #[test]
    fn test_expected_weights() {
        let mapper = LLaMAWeightMapper::new(2);
        let weights = mapper.expected_weights();

        assert!(weights.contains(&"embed_tokens.weight".to_string()));
        assert!(weights.contains(&"layers.0.self_attn.q_proj.weight".to_string()));
        assert!(weights.contains(&"layers.1.mlp.down_proj.weight".to_string()));
    }
}
