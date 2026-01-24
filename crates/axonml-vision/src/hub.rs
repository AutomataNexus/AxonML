//! Model Hub - Pretrained Weights Management
//!
//! Download, cache, and load pretrained model weights.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

use axonml_tensor::Tensor;

// =============================================================================
// Error Type
// =============================================================================

/// Hub errors.
#[derive(Debug)]
pub enum HubError {
    /// Network error during download.
    NetworkError(String),
    /// IO error.
    IoError(std::io::Error),
    /// Model not found.
    ModelNotFound(String),
    /// Invalid weight format.
    InvalidFormat(String),
    /// Checksum mismatch between expected and actual hash.
    ChecksumMismatch {
        /// Expected checksum value.
        expected: String,
        /// Actual computed checksum.
        actual: String,
    },
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::NetworkError(e) => write!(f, "Network error: {}", e),
            HubError::IoError(e) => write!(f, "IO error: {}", e),
            HubError::ModelNotFound(name) => write!(f, "Model not found: {}", name),
            HubError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            HubError::ChecksumMismatch { expected, actual } => {
                write!(f, "Checksum mismatch: expected {}, got {}", expected, actual)
            }
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

/// Information about a pretrained model.
#[derive(Debug, Clone)]
pub struct PretrainedModel {
    /// Model name (e.g., "resnet18").
    pub name: String,
    /// URL to download weights.
    pub url: String,
    /// Expected SHA256 checksum (optional).
    pub checksum: Option<String>,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Number of classes the model was trained on.
    pub num_classes: usize,
    /// Input image size (height, width).
    pub input_size: (usize, usize),
    /// Dataset trained on.
    pub dataset: String,
    /// Top-1 accuracy on validation set.
    pub accuracy: f32,
}

/// Get the cache directory for pretrained weights.
pub fn cache_dir() -> PathBuf {
    let base = dirs::cache_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("axonml").join("hub").join("weights")
}

/// Get registry of available pretrained models.
pub fn model_registry() -> HashMap<String, PretrainedModel> {
    let mut registry = HashMap::new();

    // ResNet models (ImageNet pretrained)
    registry.insert(
        "resnet18".to_string(),
        PretrainedModel {
            name: "resnet18".to_string(),
            url: "https://huggingface.co/axonml-ml/resnet18-imagenet/resolve/main/resnet18.safetensors".to_string(),
            checksum: None,
            size_bytes: 44_700_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 69.76,
        },
    );

    registry.insert(
        "resnet34".to_string(),
        PretrainedModel {
            name: "resnet34".to_string(),
            url: "https://huggingface.co/axonml-ml/resnet34-imagenet/resolve/main/resnet34.safetensors".to_string(),
            checksum: None,
            size_bytes: 83_300_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 73.31,
        },
    );

    registry.insert(
        "resnet50".to_string(),
        PretrainedModel {
            name: "resnet50".to_string(),
            url: "https://huggingface.co/axonml-ml/resnet50-imagenet/resolve/main/resnet50.safetensors".to_string(),
            checksum: None,
            size_bytes: 97_800_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 76.13,
        },
    );

    // VGG models (ImageNet pretrained)
    registry.insert(
        "vgg16".to_string(),
        PretrainedModel {
            name: "vgg16".to_string(),
            url: "https://huggingface.co/axonml-ml/vgg16-imagenet/resolve/main/vgg16.safetensors".to_string(),
            checksum: None,
            size_bytes: 528_000_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 71.59,
        },
    );

    registry.insert(
        "vgg19".to_string(),
        PretrainedModel {
            name: "vgg19".to_string(),
            url: "https://huggingface.co/axonml-ml/vgg19-imagenet/resolve/main/vgg19.safetensors".to_string(),
            checksum: None,
            size_bytes: 548_000_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 72.38,
        },
    );

    registry.insert(
        "vgg16_bn".to_string(),
        PretrainedModel {
            name: "vgg16_bn".to_string(),
            url: "https://huggingface.co/axonml-ml/vgg16bn-imagenet/resolve/main/vgg16_bn.safetensors".to_string(),
            checksum: None,
            size_bytes: 528_000_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 73.36,
        },
    );

    registry
}

// =============================================================================
// Weight Loading
// =============================================================================

/// State dictionary - named tensor mapping.
pub type StateDict = HashMap<String, Tensor<f32>>;

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
/// * `model_name` - Name of the model (e.g., "resnet18")
/// * `force` - Force re-download even if cached
///
/// # Returns
/// Path to the downloaded weights file
pub fn download_weights(model_name: &str, force: bool) -> HubResult<PathBuf> {
    let registry = model_registry();
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

    // Download weights from pretrained model hub
    println!("Downloading {} weights ({:.1} MB)...", model_name, model_info.size_bytes as f64 / 1_000_000.0);

    let response = reqwest::blocking::get(&model_info.url)
        .map_err(|e| HubError::NetworkError(e.to_string()))?;

    if !response.status().is_success() {
        return Err(HubError::NetworkError(format!(
            "HTTP {}: {}",
            response.status(),
            model_info.url
        )));
    }

    let bytes = response.bytes()
        .map_err(|e| HubError::NetworkError(e.to_string()))?;

    let mut file = File::create(&cache_path)?;
    file.write_all(&bytes)?;

    println!("Downloaded to {:?}", cache_path);

    Ok(cache_path)
}

/// Save state dict to file (simple binary format).
///
/// # Arguments
/// * `state` - The state dictionary to save
/// * `path` - Path where the file will be saved
///
/// # Example
/// ```ignore
/// use axonml_vision::hub::{save_state_dict, StateDict};
/// let mut state = StateDict::new();
/// // ... populate state dict ...
/// save_state_dict(&state, &PathBuf::from("model.bin")).unwrap();
/// ```
pub fn save_state_dict(state: &StateDict, path: &PathBuf) -> HubResult<()> {
    use std::io::BufWriter;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write number of tensors
    let num_tensors = state.len() as u32;
    writer.write_all(&num_tensors.to_le_bytes())?;

    for (name, tensor) in state {
        // Write name length and name
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u32;
        writer.write_all(&name_len.to_le_bytes())?;
        writer.write_all(name_bytes)?;

        // Write shape
        let shape = tensor.shape();
        let ndim = shape.len() as u32;
        writer.write_all(&ndim.to_le_bytes())?;
        for &dim in shape {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }

        // Write data
        let data = tensor.to_vec();
        for val in data {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Load state dict from file.
pub fn load_state_dict(path: &PathBuf) -> HubResult<StateDict> {
    use std::io::BufReader;

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read number of tensors
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let num_tensors = u32::from_le_bytes(buf4);

    let mut state = HashMap::new();

    for _ in 0..num_tensors {
        // Read name
        reader.read_exact(&mut buf4)?;
        let name_len = u32::from_le_bytes(buf4) as usize;
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        // Read shape
        reader.read_exact(&mut buf4)?;
        let ndim = u32::from_le_bytes(buf4) as usize;
        let mut shape = Vec::with_capacity(ndim);
        let mut buf8 = [0u8; 8];
        for _ in 0..ndim {
            reader.read_exact(&mut buf8)?;
            shape.push(u64::from_le_bytes(buf8) as usize);
        }

        // Read data
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            reader.read_exact(&mut buf4)?;
            data.push(f32::from_le_bytes(buf4));
        }

        let tensor = Tensor::from_vec(data, &shape)
            .map_err(|e| HubError::InvalidFormat(format!("{:?}", e)))?;
        state.insert(name, tensor);
    }

    Ok(state)
}

/// List available pretrained models.
pub fn list_models() -> Vec<PretrainedModel> {
    model_registry().into_values().collect()
}

/// Get info for a specific model.
pub fn model_info(name: &str) -> Option<PretrainedModel> {
    model_registry().get(name).cloned()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry() {
        let registry = model_registry();
        assert!(registry.contains_key("resnet18"));
        assert!(registry.contains_key("vgg16"));
    }

    #[test]
    fn test_cache_dir() {
        let dir = cache_dir();
        assert!(dir.to_string_lossy().contains("axonml"));
    }

    #[test]
    fn test_list_models() {
        let models = list_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = model_info("resnet18");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.num_classes, 1000);
        assert_eq!(info.input_size, (224, 224));
    }

    #[test]
    fn test_model_urls() {
        let registry = model_registry();
        for (name, model) in &registry {
            assert!(!model.url.is_empty(), "Model {} has empty URL", name);
            assert!(model.url.starts_with("https://"), "Model {} URL should be HTTPS", name);
            assert!(model.size_bytes > 0, "Model {} has zero size", name);
        }
    }

    #[test]
    fn test_cached_path() {
        let path = cached_path("resnet18");
        assert!(path.to_string_lossy().contains("resnet18"));
        assert!(path.to_string_lossy().ends_with(".safetensors"));
    }

    #[test]
    fn test_save_load_state_dict() {
        // Create a simple state dict for testing
        let mut state = StateDict::new();
        state.insert("layer.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        state.insert("layer.bias".to_string(), Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap());

        let temp_path = std::env::temp_dir().join("test_weights.bin");
        save_state_dict(&state, &temp_path).unwrap();

        let loaded = load_state_dict(&temp_path).unwrap();
        assert_eq!(state.len(), loaded.len());

        // Verify tensor shapes
        let weight = loaded.get("layer.weight").unwrap();
        assert_eq!(weight.shape(), &[2, 2]);

        let bias = loaded.get("layer.bias").unwrap();
        assert_eq!(bias.shape(), &[2]);

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}
