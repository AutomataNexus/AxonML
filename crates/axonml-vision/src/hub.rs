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
    /// Checksum mismatch.
    ChecksumMismatch { expected: String, actual: String },
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

    // Download weights
    println!("Downloading {} weights ({:.1} MB)...", model_name, model_info.size_bytes as f64 / 1_000_000.0);

    // Use reqwest for download (blocking)
    #[cfg(feature = "download")]
    {
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
    }

    #[cfg(not(feature = "download"))]
    {
        // Fallback: create placeholder with synthetic weights
        println!("Note: Download feature not enabled. Creating synthetic weights.");
        create_synthetic_weights(model_name, &cache_path)?;
    }

    Ok(cache_path)
}

/// Create synthetic weights for testing (when download is unavailable).
fn create_synthetic_weights(model_name: &str, path: &PathBuf) -> HubResult<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let state_dict = match model_name {
        "resnet18" => create_resnet18_state_dict(&mut rng),
        "resnet34" => create_resnet34_state_dict(&mut rng),
        "vgg16" | "vgg16_bn" => create_vgg16_state_dict(&mut rng),
        "vgg19" => create_vgg19_state_dict(&mut rng),
        _ => return Err(HubError::ModelNotFound(model_name.to_string())),
    };

    save_state_dict(&state_dict, path)?;
    Ok(())
}

/// Create synthetic ResNet18 state dict.
fn create_resnet18_state_dict<R: rand::Rng>(rng: &mut R) -> StateDict {
    let mut state = HashMap::new();

    // Conv1: 3 -> 64, kernel 7x7
    state.insert("conv1.weight".to_string(), random_tensor(rng, &[64, 3, 7, 7]));
    state.insert("conv1.bias".to_string(), random_tensor(rng, &[64]));

    // BN1
    state.insert("bn1.weight".to_string(), ones_tensor(&[64]));
    state.insert("bn1.bias".to_string(), zeros_tensor(&[64]));
    state.insert("bn1.running_mean".to_string(), zeros_tensor(&[64]));
    state.insert("bn1.running_var".to_string(), ones_tensor(&[64]));

    // Layer 1: 2 blocks, 64 channels
    for i in 0..2 {
        add_basic_block_weights(&mut state, rng, &format!("layer1.{}", i), 64, 64);
    }

    // Layer 2: 2 blocks, 128 channels (first has downsample)
    add_basic_block_weights(&mut state, rng, "layer2.0", 64, 128);
    add_downsample_weights(&mut state, rng, "layer2.0.downsample", 64, 128);
    add_basic_block_weights(&mut state, rng, "layer2.1", 128, 128);

    // Layer 3: 2 blocks, 256 channels
    add_basic_block_weights(&mut state, rng, "layer3.0", 128, 256);
    add_downsample_weights(&mut state, rng, "layer3.0.downsample", 128, 256);
    add_basic_block_weights(&mut state, rng, "layer3.1", 256, 256);

    // Layer 4: 2 blocks, 512 channels
    add_basic_block_weights(&mut state, rng, "layer4.0", 256, 512);
    add_downsample_weights(&mut state, rng, "layer4.0.downsample", 256, 512);
    add_basic_block_weights(&mut state, rng, "layer4.1", 512, 512);

    // FC: 512 -> 1000
    state.insert("fc.weight".to_string(), random_tensor(rng, &[1000, 512]));
    state.insert("fc.bias".to_string(), random_tensor(rng, &[1000]));

    state
}

/// Create synthetic ResNet34 state dict.
fn create_resnet34_state_dict<R: rand::Rng>(rng: &mut R) -> StateDict {
    let mut state = HashMap::new();

    // Conv1
    state.insert("conv1.weight".to_string(), random_tensor(rng, &[64, 3, 7, 7]));
    state.insert("conv1.bias".to_string(), random_tensor(rng, &[64]));

    // BN1
    state.insert("bn1.weight".to_string(), ones_tensor(&[64]));
    state.insert("bn1.bias".to_string(), zeros_tensor(&[64]));
    state.insert("bn1.running_mean".to_string(), zeros_tensor(&[64]));
    state.insert("bn1.running_var".to_string(), ones_tensor(&[64]));

    // Layer 1: 3 blocks
    for i in 0..3 {
        add_basic_block_weights(&mut state, rng, &format!("layer1.{}", i), 64, 64);
    }

    // Layer 2: 4 blocks
    add_basic_block_weights(&mut state, rng, "layer2.0", 64, 128);
    add_downsample_weights(&mut state, rng, "layer2.0.downsample", 64, 128);
    for i in 1..4 {
        add_basic_block_weights(&mut state, rng, &format!("layer2.{}", i), 128, 128);
    }

    // Layer 3: 6 blocks
    add_basic_block_weights(&mut state, rng, "layer3.0", 128, 256);
    add_downsample_weights(&mut state, rng, "layer3.0.downsample", 128, 256);
    for i in 1..6 {
        add_basic_block_weights(&mut state, rng, &format!("layer3.{}", i), 256, 256);
    }

    // Layer 4: 3 blocks
    add_basic_block_weights(&mut state, rng, "layer4.0", 256, 512);
    add_downsample_weights(&mut state, rng, "layer4.0.downsample", 256, 512);
    for i in 1..3 {
        add_basic_block_weights(&mut state, rng, &format!("layer4.{}", i), 512, 512);
    }

    // FC
    state.insert("fc.weight".to_string(), random_tensor(rng, &[1000, 512]));
    state.insert("fc.bias".to_string(), random_tensor(rng, &[1000]));

    state
}

/// Create synthetic VGG16 state dict.
fn create_vgg16_state_dict<R: rand::Rng>(rng: &mut R) -> StateDict {
    let mut state = HashMap::new();

    // Features
    let configs = [
        (3, 64), (64, 64),       // Block 1
        (64, 128), (128, 128),   // Block 2
        (128, 256), (256, 256), (256, 256), // Block 3
        (256, 512), (512, 512), (512, 512), // Block 4
        (512, 512), (512, 512), (512, 512), // Block 5
    ];

    for (i, (in_c, out_c)) in configs.iter().enumerate() {
        let prefix = format!("features.{}", i * 2); // Skip ReLU layers
        state.insert(format!("{}.weight", prefix), random_tensor(rng, &[*out_c, *in_c, 3, 3]));
        state.insert(format!("{}.bias", prefix), random_tensor(rng, &[*out_c]));
    }

    // Classifier
    state.insert("classifier.0.weight".to_string(), random_tensor(rng, &[4096, 512 * 7 * 7]));
    state.insert("classifier.0.bias".to_string(), random_tensor(rng, &[4096]));
    state.insert("classifier.3.weight".to_string(), random_tensor(rng, &[4096, 4096]));
    state.insert("classifier.3.bias".to_string(), random_tensor(rng, &[4096]));
    state.insert("classifier.6.weight".to_string(), random_tensor(rng, &[1000, 4096]));
    state.insert("classifier.6.bias".to_string(), random_tensor(rng, &[1000]));

    state
}

/// Create synthetic VGG19 state dict.
fn create_vgg19_state_dict<R: rand::Rng>(rng: &mut R) -> StateDict {
    let mut state = HashMap::new();

    // Features (VGG19 has more conv layers)
    let configs = [
        (3, 64), (64, 64),
        (64, 128), (128, 128),
        (128, 256), (256, 256), (256, 256), (256, 256),
        (256, 512), (512, 512), (512, 512), (512, 512),
        (512, 512), (512, 512), (512, 512), (512, 512),
    ];

    for (i, (in_c, out_c)) in configs.iter().enumerate() {
        let prefix = format!("features.{}", i * 2);
        state.insert(format!("{}.weight", prefix), random_tensor(rng, &[*out_c, *in_c, 3, 3]));
        state.insert(format!("{}.bias", prefix), random_tensor(rng, &[*out_c]));
    }

    // Classifier
    state.insert("classifier.0.weight".to_string(), random_tensor(rng, &[4096, 512 * 7 * 7]));
    state.insert("classifier.0.bias".to_string(), random_tensor(rng, &[4096]));
    state.insert("classifier.3.weight".to_string(), random_tensor(rng, &[4096, 4096]));
    state.insert("classifier.3.bias".to_string(), random_tensor(rng, &[4096]));
    state.insert("classifier.6.weight".to_string(), random_tensor(rng, &[1000, 4096]));
    state.insert("classifier.6.bias".to_string(), random_tensor(rng, &[1000]));

    state
}

fn add_basic_block_weights<R: rand::Rng>(
    state: &mut StateDict,
    rng: &mut R,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
) {
    // Conv1
    state.insert(
        format!("{}.conv1.weight", prefix),
        random_tensor(rng, &[out_channels, in_channels, 3, 3]),
    );
    state.insert(
        format!("{}.conv1.bias", prefix),
        random_tensor(rng, &[out_channels]),
    );

    // BN1
    state.insert(format!("{}.bn1.weight", prefix), ones_tensor(&[out_channels]));
    state.insert(format!("{}.bn1.bias", prefix), zeros_tensor(&[out_channels]));
    state.insert(format!("{}.bn1.running_mean", prefix), zeros_tensor(&[out_channels]));
    state.insert(format!("{}.bn1.running_var", prefix), ones_tensor(&[out_channels]));

    // Conv2
    state.insert(
        format!("{}.conv2.weight", prefix),
        random_tensor(rng, &[out_channels, out_channels, 3, 3]),
    );
    state.insert(
        format!("{}.conv2.bias", prefix),
        random_tensor(rng, &[out_channels]),
    );

    // BN2
    state.insert(format!("{}.bn2.weight", prefix), ones_tensor(&[out_channels]));
    state.insert(format!("{}.bn2.bias", prefix), zeros_tensor(&[out_channels]));
    state.insert(format!("{}.bn2.running_mean", prefix), zeros_tensor(&[out_channels]));
    state.insert(format!("{}.bn2.running_var", prefix), ones_tensor(&[out_channels]));
}

fn add_downsample_weights<R: rand::Rng>(
    state: &mut StateDict,
    rng: &mut R,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
) {
    state.insert(
        format!("{}.0.weight", prefix),
        random_tensor(rng, &[out_channels, in_channels, 1, 1]),
    );
    state.insert(format!("{}.1.weight", prefix), ones_tensor(&[out_channels]));
    state.insert(format!("{}.1.bias", prefix), zeros_tensor(&[out_channels]));
    state.insert(format!("{}.1.running_mean", prefix), zeros_tensor(&[out_channels]));
    state.insert(format!("{}.1.running_var", prefix), ones_tensor(&[out_channels]));
}

fn random_tensor<R: rand::Rng>(rng: &mut R, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let stddev = (2.0 / shape[0] as f32).sqrt();
    let data: Vec<f32> = (0..numel)
        .map(|_| rng.gen::<f32>() * stddev * 2.0 - stddev)
        .collect();
    Tensor::from_vec(data, shape).unwrap()
}

fn zeros_tensor(shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    Tensor::from_vec(vec![0.0; numel], shape).unwrap()
}

fn ones_tensor(shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    Tensor::from_vec(vec![1.0; numel], shape).unwrap()
}

/// Save state dict to file (simple binary format).
fn save_state_dict(state: &StateDict, path: &PathBuf) -> HubResult<()> {
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
    fn test_synthetic_weights() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let state = create_resnet18_state_dict(&mut rng);
        assert!(state.contains_key("conv1.weight"));
        assert!(state.contains_key("fc.weight"));

        let conv1 = state.get("conv1.weight").unwrap();
        assert_eq!(conv1.shape(), &[64, 3, 7, 7]);
    }

    #[test]
    fn test_save_load_state_dict() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let state = create_resnet18_state_dict(&mut rng);

        let temp_path = std::env::temp_dir().join("test_weights.bin");
        save_state_dict(&state, &temp_path).unwrap();

        let loaded = load_state_dict(&temp_path).unwrap();
        assert_eq!(state.len(), loaded.len());

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}
