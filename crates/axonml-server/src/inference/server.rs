//! Inference server for AxonML
//!
//! Handles model loading and serving HTTP endpoints.

use axonml_autograd::Variable;
use axonml_nn::{Linear, Module, ReLU, Sequential, Sigmoid, Softmax, Tanh};
use axonml_serialize::{load_state_dict, StateDict};
use axonml_tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Loaded model with its architecture and weights
pub struct LoadedModel {
    pub state_dict: StateDict,
    pub architecture: ModelArchitecture,
}

/// Model architecture description
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    pub input_size: usize,
    pub output_size: usize,
    pub layers: Vec<LayerInfo>,
}

/// Layer information for reconstruction
#[derive(Debug, Clone)]
pub enum LayerInfo {
    Linear { in_features: usize, out_features: usize },
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
}

/// Model instance for inference
#[derive(Debug)]
pub struct ModelInstance {
    pub model_id: String,
    pub version_id: String,
    pub version: u32,
    pub file_path: String,
    pub loaded: bool,
}

/// Full model entry with loaded weights
pub struct ModelEntry {
    pub instance: ModelInstance,
    pub model: Option<LoadedModel>,
}

/// Inference server configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub port: u16,
    pub batch_size: u32,
    pub timeout_ms: u64,
    pub max_queue_size: u32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            port: 8100,
            batch_size: 1,
            timeout_ms: 30000,
            max_queue_size: 100,
        }
    }
}

/// Inference server for serving models
pub struct InferenceServer {
    models: Arc<RwLock<HashMap<String, ModelEntry>>>,
    config: InferenceConfig,
}

impl InferenceServer {
    /// Create a new inference server
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Load a model for serving
    pub async fn load_model(
        &self,
        endpoint_id: &str,
        model_id: &str,
        version_id: &str,
        version: u32,
        file_path: &str,
    ) -> Result<(), String> {
        // Check if file exists
        if !Path::new(file_path).exists() {
            return Err(format!(
                "Model file not found: {}",
                file_path
            ));
        }

        // Load the state dict from file
        let loaded_model = Self::load_model_from_file(file_path)?;

        let instance = ModelInstance {
            model_id: model_id.to_string(),
            version_id: version_id.to_string(),
            version,
            file_path: file_path.to_string(),
            loaded: true,
        };

        let entry = ModelEntry {
            instance,
            model: Some(loaded_model),
        };

        let mut models = self.models.write().await;
        models.insert(endpoint_id.to_string(), entry);

        tracing::info!(
            endpoint_id = endpoint_id,
            model_id = model_id,
            version = version,
            file_path = file_path,
            "Model loaded for inference"
        );

        Ok(())
    }

    /// Load model from file and detect architecture
    fn load_model_from_file(file_path: &str) -> Result<LoadedModel, String> {
        let state_dict = load_state_dict(file_path)
            .map_err(|e| format!("Failed to load state dict: {}", e))?;

        // Detect architecture from state dict parameter names and shapes
        let architecture = Self::detect_architecture(&state_dict)?;

        Ok(LoadedModel {
            state_dict,
            architecture,
        })
    }

    /// Detect model architecture from state dict
    fn detect_architecture(state_dict: &StateDict) -> Result<ModelArchitecture, String> {
        let mut layers = Vec::new();
        let mut layer_indices: HashMap<usize, (Option<usize>, Option<usize>)> = HashMap::new();
        let mut activation_hints: HashMap<usize, String> = HashMap::new();

        // Parse layer information from parameter names
        // Expected format: "0.weight", "0.bias", "1.weight", etc. for Sequential
        // Or "layer_0.weight", "linear_0.weight", etc.
        for key in state_dict.keys() {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() >= 2 {
                // Detect activation hints from parameter names
                let key_lower = key.to_lowercase();
                if key_lower.contains("sigmoid") {
                    if let Some(idx) = Self::extract_layer_index(parts[0]) {
                        activation_hints.insert(idx, "sigmoid".to_string());
                    }
                } else if key_lower.contains("tanh") {
                    if let Some(idx) = Self::extract_layer_index(parts[0]) {
                        activation_hints.insert(idx, "tanh".to_string());
                    }
                } else if key_lower.contains("softmax") {
                    if let Some(idx) = Self::extract_layer_index(parts[0]) {
                        activation_hints.insert(idx, "softmax".to_string());
                    }
                }

                // Try to parse layer index
                if let Some(idx) = Self::extract_layer_index(parts[0]) {
                    if let Some(entry) = state_dict.get(key) {
                        let shape = &entry.data.shape;
                        let (in_size, out_size) = layer_indices.entry(idx).or_insert((None, None));

                        if parts[1] == "weight" && shape.len() == 2 {
                            // Weight shape is [out_features, in_features]
                            *out_size = Some(shape[0]);
                            *in_size = Some(shape[1]);
                        }
                    }
                }
            }
        }

        // Build layer list from detected information
        let mut sorted_indices: Vec<usize> = layer_indices.keys().copied().collect();
        sorted_indices.sort();

        let mut input_size = 0;
        let mut output_size = 0;

        for (i, idx) in sorted_indices.iter().enumerate() {
            if let Some((in_feat, out_feat)) = layer_indices.get(idx) {
                if let (Some(in_f), Some(out_f)) = (in_feat, out_feat) {
                    if i == 0 {
                        input_size = *in_f;
                    }
                    output_size = *out_f;

                    layers.push(LayerInfo::Linear {
                        in_features: *in_f,
                        out_features: *out_f,
                    });

                    // Determine activation function for this layer
                    let is_last = i == sorted_indices.len() - 1;

                    if let Some(activation) = activation_hints.get(idx) {
                        match activation.as_str() {
                            "sigmoid" => layers.push(LayerInfo::Sigmoid),
                            "tanh" => layers.push(LayerInfo::Tanh),
                            "softmax" => layers.push(LayerInfo::Softmax),
                            _ => {
                                if !is_last {
                                    layers.push(LayerInfo::ReLU);
                                }
                            }
                        }
                    } else if is_last {
                        // Last layer often uses softmax for classification
                        // Don't add activation - let user specify or infer from output size
                    } else {
                        // Default to ReLU for hidden layers
                        layers.push(LayerInfo::ReLU);
                    }
                }
            }
        }

        if layers.is_empty() {
            return Err("Could not detect model architecture from state dict".to_string());
        }

        Ok(ModelArchitecture {
            input_size,
            output_size,
            layers,
        })
    }

    /// Extract layer index from a parameter name prefix
    fn extract_layer_index(prefix: &str) -> Option<usize> {
        let index_str = prefix
            .trim_start_matches("layer_")
            .trim_start_matches("linear_")
            .trim_start_matches("fc_")
            .trim_start_matches("activation_")
            .trim_start_matches("act_");

        index_str.parse::<usize>().ok()
    }

    /// Build a Sequential model from architecture and load weights
    fn build_model(loaded: &LoadedModel) -> Result<Sequential, String> {
        let mut seq = Sequential::new();
        let mut linear_idx = 0;

        for layer_info in loaded.architecture.layers.iter() {
            match layer_info {
                LayerInfo::Linear { in_features, out_features } => {
                    // Load weights from state dict
                    let weight_key = format!("{}.weight", linear_idx);
                    let bias_key = format!("{}.bias", linear_idx);

                    let weight_tensor = loaded.state_dict.get(&weight_key)
                        .ok_or_else(|| format!("Missing weight for layer {}", linear_idx))?
                        .data.to_tensor()
                        .map_err(|e| format!("Failed to load weight tensor: {}", e))?;

                    // Validate weight dimensions match expected
                    let weight_shape = weight_tensor.shape();
                    if weight_shape.len() != 2 || weight_shape[0] != *out_features || weight_shape[1] != *in_features {
                        return Err(format!(
                            "Weight shape mismatch for layer {}: expected [{}, {}], got {:?}",
                            linear_idx, out_features, in_features, weight_shape
                        ));
                    }

                    let bias_tensor = loaded.state_dict.get(&bias_key)
                        .map(|e| e.data.to_tensor())
                        .transpose()
                        .map_err(|e| format!("Failed to load bias tensor: {}", e))?;

                    // Validate bias dimensions if present
                    if let Some(ref bias) = bias_tensor {
                        let bias_shape = bias.shape();
                        if bias_shape.len() != 1 || bias_shape[0] != *out_features {
                            return Err(format!(
                                "Bias shape mismatch for layer {}: expected [{}], got {:?}",
                                linear_idx, out_features, bias_shape
                            ));
                        }
                    }

                    let linear = Linear::from_weights(weight_tensor, bias_tensor);
                    seq = seq.add(linear);
                    linear_idx += 1;
                }
                LayerInfo::ReLU => {
                    seq = seq.add(ReLU);
                }
                LayerInfo::Sigmoid => {
                    seq = seq.add(Sigmoid);
                }
                LayerInfo::Tanh => {
                    seq = seq.add(Tanh);
                }
                LayerInfo::Softmax => {
                    seq = seq.add(Softmax::new(-1));
                }
            }
        }

        Ok(seq)
    }

    /// Unload a model
    pub async fn unload_model(&self, endpoint_id: &str) -> Result<(), String> {
        let mut models = self.models.write().await;
        if models.remove(endpoint_id).is_some() {
            tracing::info!(endpoint_id = endpoint_id, "Model unloaded");
            Ok(())
        } else {
            Err(format!("Model not found for endpoint {}", endpoint_id))
        }
    }

    /// Run inference on a model
    pub async fn predict(
        &self,
        endpoint_id: &str,
        inputs: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let models = self.models.read().await;
        let entry = models
            .get(endpoint_id)
            .ok_or_else(|| format!("Model not loaded for endpoint {}", endpoint_id))?;

        // If we have a loaded model, run real inference
        if let Some(ref loaded) = entry.model {
            return Self::run_inference(loaded, inputs);
        }

        // Model not loaded - return error instead of fake predictions
        Err(format!(
            "Model weights not loaded for endpoint '{}'. Please verify the model file exists and is valid.",
            endpoint_id
        ))
    }

    /// Run real inference using loaded model
    fn run_inference(loaded: &LoadedModel, inputs: serde_json::Value) -> Result<serde_json::Value, String> {
        // Parse input data
        let input_array = Self::parse_input(&inputs)?;
        let batch_size = input_array.len();
        let input_size = if batch_size > 0 { input_array[0].len() } else { 0 };

        // Validate input size matches model
        if input_size != loaded.architecture.input_size {
            return Err(format!(
                "Input size mismatch: expected {}, got {}",
                loaded.architecture.input_size, input_size
            ));
        }

        // Flatten input for tensor creation
        let flat_input: Vec<f32> = input_array.into_iter().flatten().collect();

        // Create input tensor
        let input_tensor = Tensor::from_vec(flat_input, &[batch_size, input_size])
            .map_err(|e| format!("Failed to create input tensor: {}", e))?;

        // Create Variable for forward pass (no gradients needed for inference)
        let input_var = Variable::new(input_tensor, false);

        // Build and run model
        let model = Self::build_model(loaded)?;
        let output_var = model.forward(&input_var);

        // Extract output data
        let output_shape = output_var.shape();
        let output_data = output_var.data().to_vec();

        // Reshape output into batch format
        let output_size = loaded.architecture.output_size;
        let predictions: Vec<Vec<f32>> = output_data
            .chunks(output_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(serde_json::json!({
            "predictions": predictions,
            "output_shape": output_shape,
            "model_loaded": true
        }))
    }

    /// Parse input JSON into array of f32 vectors
    fn parse_input(inputs: &serde_json::Value) -> Result<Vec<Vec<f32>>, String> {
        // Support multiple input formats:
        // 1. {"inputs": [[1.0, 2.0], [3.0, 4.0]]} - batch of vectors
        // 2. {"inputs": [1.0, 2.0, 3.0]} - single vector
        // 3. [[1.0, 2.0], [3.0, 4.0]] - direct array

        let data = inputs.get("inputs").unwrap_or(inputs);

        if let Some(arr) = data.as_array() {
            if arr.is_empty() {
                return Ok(vec![]);
            }

            // Check if first element is an array (batch) or number (single)
            if arr[0].is_array() {
                // Batch input
                arr.iter()
                    .map(|row| {
                        row.as_array()
                            .ok_or_else(|| "Invalid input format".to_string())?
                            .iter()
                            .map(|v| {
                                v.as_f64()
                                    .map(|f| f as f32)
                                    .ok_or_else(|| "Non-numeric value in input".to_string())
                            })
                            .collect::<Result<Vec<f32>, String>>()
                    })
                    .collect()
            } else {
                // Single input - wrap in batch
                let single: Vec<f32> = arr
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| f as f32)
                            .ok_or_else(|| "Non-numeric value in input".to_string())
                    })
                    .collect::<Result<Vec<f32>, String>>()?;
                Ok(vec![single])
            }
        } else if let Some(n) = data.as_f64() {
            // Single scalar input
            Ok(vec![vec![n as f32]])
        } else {
            Err("Invalid input format: expected array or number".to_string())
        }
    }

    /// Check if a model is loaded
    pub async fn is_loaded(&self, endpoint_id: &str) -> bool {
        let models = self.models.read().await;
        models.get(endpoint_id).map(|e| e.instance.loaded).unwrap_or(false)
    }

    /// Check if a model has weights loaded
    pub async fn has_weights(&self, endpoint_id: &str) -> bool {
        let models = self.models.read().await;
        models.get(endpoint_id).map(|e| e.model.is_some()).unwrap_or(false)
    }

    /// Get loaded models count
    pub async fn loaded_count(&self) -> usize {
        let models = self.models.read().await;
        models.len()
    }

    /// Get server port
    pub fn port(&self) -> u16 {
        self.config.port
    }

    /// Get batch size configuration
    pub fn batch_size(&self) -> u32 {
        self.config.batch_size
    }

    /// Get timeout in milliseconds
    pub fn timeout_ms(&self) -> u64 {
        self.config.timeout_ms
    }

    /// Get max queue size
    pub fn max_queue_size(&self) -> u32 {
        self.config.max_queue_size
    }

    /// Get full config
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Get model info for an endpoint
    pub async fn get_model_info(&self, endpoint_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(endpoint_id).map(|entry| {
            ModelInfo {
                model_id: entry.instance.model_id.clone(),
                version_id: entry.instance.version_id.clone(),
                version: entry.instance.version,
                file_path: entry.instance.file_path.clone(),
                loaded: entry.instance.loaded,
                has_weights: entry.model.is_some(),
                architecture: entry.model.as_ref().map(|m| m.architecture.clone()),
            }
        })
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_id: String,
    pub version_id: String,
    pub version: u32,
    pub file_path: String,
    pub loaded: bool,
    pub has_weights: bool,
    pub architecture: Option<ModelArchitecture>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_unload_model() {
        use axonml_serialize::{save_state_dict, StateDict, TensorData, Format};
        use axonml_tensor::Tensor;

        // Create a temporary model file
        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_model.axonml");

        // Create a simple state dict and save it
        let mut state_dict = StateDict::new();
        let weight = Tensor::from_vec(vec![0.1f32; 10 * 5], &[5, 10]).unwrap();
        let bias = Tensor::from_vec(vec![0.0f32; 5], &[5]).unwrap();
        state_dict.insert("0.weight".to_string(), TensorData::from_tensor(&weight));
        state_dict.insert("0.bias".to_string(), TensorData::from_tensor(&bias));
        save_state_dict(&state_dict, &model_path, Format::Axonml).unwrap();

        let server = InferenceServer::new(InferenceConfig::default());

        server
            .load_model("ep-1", "model-1", "ver-1", 1, model_path.to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(server.loaded_count().await, 1);
        assert!(server.is_loaded("ep-1").await);
        assert!(server.has_weights("ep-1").await);

        server.unload_model("ep-1").await.unwrap();

        assert_eq!(server.loaded_count().await, 0);
        assert!(!server.is_loaded("ep-1").await);

        // Clean up
        let _ = std::fs::remove_file(&model_path);
    }

    #[test]
    fn test_parse_input_batch() {
        let input = serde_json::json!({
            "inputs": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        });

        let parsed = InferenceServer::parse_input(&input).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(parsed[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_parse_input_single() {
        let input = serde_json::json!({
            "inputs": [1.0, 2.0, 3.0]
        });

        let parsed = InferenceServer::parse_input(&input).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_input_direct_array() {
        let input = serde_json::json!([[1.0, 2.0], [3.0, 4.0]]);

        let parsed = InferenceServer::parse_input(&input).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], vec![1.0, 2.0]);
    }
}
