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

    // Larger ResNet variants
    registry.insert(
        "resnet101".to_string(),
        PretrainedModel {
            name: "resnet101".to_string(),
            url: "https://huggingface.co/axonml-ml/resnet101-imagenet/resolve/main/resnet101.safetensors".to_string(),
            checksum: None,
            size_bytes: 170_500_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 77.37,
        },
    );

    registry.insert(
        "resnet152".to_string(),
        PretrainedModel {
            name: "resnet152".to_string(),
            url: "https://huggingface.co/axonml-ml/resnet152-imagenet/resolve/main/resnet152.safetensors".to_string(),
            checksum: None,
            size_bytes: 230_400_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 78.31,
        },
    );

    // Mobile-optimized models
    registry.insert(
        "mobilenet_v2".to_string(),
        PretrainedModel {
            name: "mobilenet_v2".to_string(),
            url: "https://huggingface.co/axonml-ml/mobilenetv2-imagenet/resolve/main/mobilenet_v2.safetensors".to_string(),
            checksum: None,
            size_bytes: 13_600_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 71.88,
        },
    );

    registry.insert(
        "mobilenet_v3_small".to_string(),
        PretrainedModel {
            name: "mobilenet_v3_small".to_string(),
            url: "https://huggingface.co/axonml-ml/mobilenetv3-small-imagenet/resolve/main/mobilenet_v3_small.safetensors".to_string(),
            checksum: None,
            size_bytes: 9_800_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 67.67,
        },
    );

    registry.insert(
        "mobilenet_v3_large".to_string(),
        PretrainedModel {
            name: "mobilenet_v3_large".to_string(),
            url: "https://huggingface.co/axonml-ml/mobilenetv3-large-imagenet/resolve/main/mobilenet_v3_large.safetensors".to_string(),
            checksum: None,
            size_bytes: 21_100_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 74.04,
        },
    );

    // EfficientNet family
    registry.insert(
        "efficientnet_b0".to_string(),
        PretrainedModel {
            name: "efficientnet_b0".to_string(),
            url: "https://huggingface.co/axonml-ml/efficientnet-b0-imagenet/resolve/main/efficientnet_b0.safetensors".to_string(),
            checksum: None,
            size_bytes: 20_300_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 77.10,
        },
    );

    registry.insert(
        "efficientnet_b1".to_string(),
        PretrainedModel {
            name: "efficientnet_b1".to_string(),
            url: "https://huggingface.co/axonml-ml/efficientnet-b1-imagenet/resolve/main/efficientnet_b1.safetensors".to_string(),
            checksum: None,
            size_bytes: 30_100_000,
            num_classes: 1000,
            input_size: (240, 240),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 78.80,
        },
    );

    registry.insert(
        "efficientnet_b2".to_string(),
        PretrainedModel {
            name: "efficientnet_b2".to_string(),
            url: "https://huggingface.co/axonml-ml/efficientnet-b2-imagenet/resolve/main/efficientnet_b2.safetensors".to_string(),
            checksum: None,
            size_bytes: 35_200_000,
            num_classes: 1000,
            input_size: (260, 260),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 79.80,
        },
    );

    // DenseNet family
    registry.insert(
        "densenet121".to_string(),
        PretrainedModel {
            name: "densenet121".to_string(),
            url: "https://huggingface.co/axonml-ml/densenet121-imagenet/resolve/main/densenet121.safetensors".to_string(),
            checksum: None,
            size_bytes: 30_800_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 74.43,
        },
    );

    registry.insert(
        "densenet169".to_string(),
        PretrainedModel {
            name: "densenet169".to_string(),
            url: "https://huggingface.co/axonml-ml/densenet169-imagenet/resolve/main/densenet169.safetensors".to_string(),
            checksum: None,
            size_bytes: 54_700_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 75.60,
        },
    );

    // Vision Transformer (ViT)
    registry.insert(
        "vit_b_16".to_string(),
        PretrainedModel {
            name: "vit_b_16".to_string(),
            url: "https://huggingface.co/axonml-ml/vit-b16-imagenet/resolve/main/vit_b_16.safetensors".to_string(),
            checksum: None,
            size_bytes: 330_200_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 81.07,
        },
    );

    registry.insert(
        "vit_b_32".to_string(),
        PretrainedModel {
            name: "vit_b_32".to_string(),
            url: "https://huggingface.co/axonml-ml/vit-b32-imagenet/resolve/main/vit_b_32.safetensors".to_string(),
            checksum: None,
            size_bytes: 337_500_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 75.91,
        },
    );

    // Swin Transformer
    registry.insert(
        "swin_t".to_string(),
        PretrainedModel {
            name: "swin_t".to_string(),
            url: "https://huggingface.co/axonml-ml/swin-tiny-imagenet/resolve/main/swin_t.safetensors".to_string(),
            checksum: None,
            size_bytes: 110_700_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 81.30,
        },
    );

    registry.insert(
        "swin_s".to_string(),
        PretrainedModel {
            name: "swin_s".to_string(),
            url: "https://huggingface.co/axonml-ml/swin-small-imagenet/resolve/main/swin_s.safetensors".to_string(),
            checksum: None,
            size_bytes: 193_500_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 83.20,
        },
    );

    // ConvNeXt
    registry.insert(
        "convnext_tiny".to_string(),
        PretrainedModel {
            name: "convnext_tiny".to_string(),
            url: "https://huggingface.co/axonml-ml/convnext-tiny-imagenet/resolve/main/convnext_tiny.safetensors".to_string(),
            checksum: None,
            size_bytes: 109_100_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 82.10,
        },
    );

    registry.insert(
        "convnext_small".to_string(),
        PretrainedModel {
            name: "convnext_small".to_string(),
            url: "https://huggingface.co/axonml-ml/convnext-small-imagenet/resolve/main/convnext_small.safetensors".to_string(),
            checksum: None,
            size_bytes: 195_600_000,
            num_classes: 1000,
            input_size: (224, 224),
            dataset: "ImageNet-1K".to_string(),
            accuracy: 83.10,
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
