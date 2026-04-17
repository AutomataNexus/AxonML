//! Model Bundle — `.axonml` container with architecture + hyperparameters + flat weights
//!
//! A `ModelBundle` is the portable on-disk format that `tools/model_converter/convert.py`
//! consumes to reconstruct a trained model in PyTorch and emit ONNX. Unlike the
//! generic `StateDict` (a dict of named tensors), a bundle carries three things
//! the ONNX converter needs to rebuild the architecture:
//!
//! 1. `architecture` — an enum tag (`sentinel`, `lstm_autoencoder`, `gru_predictor`,
//!    `rnn`, `phantom`, `conv1d`, `conv2d`, `res_net`, `vgg`, `bert`, `gpt2`, `vi_t`,
//!    `nexus`) matching `Architecture` in the AxonML model zoo.
//! 2. `hyperparameters` — `hidden_dim`, `num_layers`, `sequence_length`, etc.
//!    These, together with `input_features`, completely determine layer shapes.
//! 3. `weights` — flat `Vec<f32>` in the exact layer-by-layer order the Python
//!    converter expects (see `tools/model_converter/convert.py::build_*`).
//!
//! # Binary Layout
//!
//! ```text
//!   0  1  2  3  4  5  6  7  8  9 10  ...
//! | A  X  O  N  M  L | V | H_LEN (u32 LE) | HEADER JSON | W_LEN (u32 LE) | WEIGHTS JSON |
//! ```
//!
//! - `V = 1` is the current format version.
//! - HEADER is a small metadata blob (architecture name, input_features, param count,
//!   quantization flag). It's separately decodable without reading the weights blob —
//!   useful for model registries that only want to display metadata.
//! - WEIGHTS holds the full `ModelBundle` payload (architecture + hyperparameters +
//!   flat weights + optional input-normalization stats).
//!
//! # File
//! `crates/axonml-serialize/src/bundle.rs`
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

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

// =============================================================================
// On-disk Format Constants
// =============================================================================

/// Magic bytes at the start of every `.axonml` bundle file.
pub const AXONML_MAGIC: &[u8; 6] = b"AXONML";

/// Current bundle format version.
pub const AXONML_BUNDLE_VERSION: u8 = 1;

// =============================================================================
// Error
// =============================================================================

/// Errors produced by bundle save/load.
#[derive(Debug)]
pub enum BundleError {
    /// I/O failure.
    Io(io::Error),
    /// JSON (de)serialization failure.
    Serde(serde_json::Error),
    /// The file does not start with `AXONML` magic bytes.
    BadMagic,
    /// The file version byte is not `AXONML_BUNDLE_VERSION`.
    BadVersion(u8),
    /// The file ended before the header or weights section was complete.
    Truncated(&'static str),
    /// A field required for round-tripping is missing or malformed.
    Invalid(String),
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "bundle I/O error: {e}"),
            Self::Serde(e) => write!(f, "bundle JSON error: {e}"),
            Self::BadMagic => write!(f, "invalid .axonml file: bad magic bytes"),
            Self::BadVersion(v) => write!(
                f,
                "unsupported .axonml bundle version: {v} (expected {AXONML_BUNDLE_VERSION})"
            ),
            Self::Truncated(what) => write!(f, "invalid .axonml file: truncated {what}"),
            Self::Invalid(msg) => write!(f, "invalid .axonml bundle: {msg}"),
        }
    }
}

impl std::error::Error for BundleError {}

impl From<io::Error> for BundleError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for BundleError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e)
    }
}

/// Convenience alias for bundle operation results.
pub type BundleResult<T> = Result<T, BundleError>;

// =============================================================================
// Header (first decodable metadata block)
// =============================================================================

/// Lightweight header decoded before the weights blob. Kept in sync with
/// the Python converter's `parse_axonml` header JSON (must round-trip verbatim).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleHeader {
    /// Architecture tag (`sentinel`, `lstm_autoencoder`, `res_net`, `bert`, etc.).
    pub architecture: String,
    /// Number of input features (for MLPs) or input channels (for CNNs).
    pub input_features: usize,
    /// Total parameter count (flat length of the weights vector).
    pub num_parameters: usize,
    /// Whether weights are post-quantization (INT8 stored as f32 dequantized).
    pub quantized: bool,
    /// Quantization bit width when `quantized = true`.
    pub quant_bits: Option<u8>,
}

// =============================================================================
// Bundle (full payload)
// =============================================================================

/// Full bundle payload carrying everything the ONNX converter needs.
///
/// The `hyperparameters` map uses flexible JSON to accommodate per-architecture
/// configs (e.g. LSTM wants `hidden_dim` + `num_layers` + `sequence_length`,
/// ViT wants `patch_size` + `num_heads`, etc). Keys are snake_case to match
/// the Python converter's expectations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundle {
    /// Architecture tag (same as `BundleHeader::architecture`).
    pub architecture: String,
    /// Number of input features.
    pub input_features: usize,
    /// Free-form hyperparameter map (hidden_dim, num_layers, sequence_length, ...).
    #[serde(default)]
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// Flat weight vector in layer-by-layer order expected by the converter.
    pub weights: Vec<f32>,
    /// Per-feature input-normalization means (empty if no normalization was applied).
    #[serde(default)]
    pub norm_means: Vec<f32>,
    /// Per-feature input-normalization std-devs.
    #[serde(default)]
    pub norm_stds: Vec<f32>,
    /// Anomaly/decision threshold for binary models (optional).
    #[serde(default)]
    pub anomaly_threshold: Option<f32>,
}

impl ModelBundle {
    /// Construct a new bundle with no normalization stats and no anomaly threshold.
    pub fn new(architecture: &str, input_features: usize, weights: Vec<f32>) -> Self {
        Self {
            architecture: architecture.to_string(),
            input_features,
            hyperparameters: HashMap::new(),
            weights,
            norm_means: Vec::new(),
            norm_stds: Vec::new(),
            anomaly_threshold: None,
        }
    }

    /// Set a single hyperparameter (overwrites any existing value).
    pub fn with_hyperparam(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.hyperparameters.insert(key.to_string(), value.into());
        self
    }

    /// Attach per-feature normalization statistics.
    pub fn with_normalization(mut self, means: Vec<f32>, stds: Vec<f32>) -> Self {
        self.norm_means = means;
        self.norm_stds = stds;
        self
    }

    /// Attach an anomaly/decision threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.anomaly_threshold = Some(threshold);
        self
    }

    /// Derive the lightweight `BundleHeader` from this bundle.
    pub fn header(&self) -> BundleHeader {
        BundleHeader {
            architecture: self.architecture.clone(),
            input_features: self.input_features,
            num_parameters: self.weights.len(),
            quantized: false,
            quant_bits: None,
        }
    }
}

// =============================================================================
// Save
// =============================================================================

/// Write a bundle to disk in the `.axonml` container format.
///
/// Creates parent directories if they don't exist. Ensures the output path has
/// a `.axonml` extension (renames if missing). Returns the final path written.
pub fn save_bundle<P: AsRef<Path>>(
    bundle: &ModelBundle,
    path: P,
) -> BundleResult<std::path::PathBuf> {
    let raw = path.as_ref();
    if let Some(parent) = raw.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let final_path = if raw.extension().is_none_or(|e| e != "axonml") {
        raw.with_extension("axonml")
    } else {
        raw.to_path_buf()
    };

    let header_json = serde_json::to_vec(&bundle.header())?;
    let weights_json = serde_json::to_vec(bundle)?;

    let mut file = fs::File::create(&final_path)?;
    file.write_all(AXONML_MAGIC)?;
    file.write_all(&[AXONML_BUNDLE_VERSION])?;
    file.write_all(&(header_json.len() as u32).to_le_bytes())?;
    file.write_all(&header_json)?;
    file.write_all(&(weights_json.len() as u32).to_le_bytes())?;
    file.write_all(&weights_json)?;

    Ok(final_path)
}

// =============================================================================
// Load
// =============================================================================

/// Read a bundle from disk, returning both the header (decoded eagerly) and the
/// full payload.
///
/// Errors if the file is too short, the magic bytes are wrong, the version is
/// unsupported, or either JSON blob fails to deserialize.
pub fn load_bundle<P: AsRef<Path>>(path: P) -> BundleResult<(BundleHeader, ModelBundle)> {
    let data = fs::read(path)?;
    load_bundle_from_bytes(&data)
}

/// In-memory variant of [`load_bundle`] — useful for HTTP handlers or tests.
pub fn load_bundle_from_bytes(data: &[u8]) -> BundleResult<(BundleHeader, ModelBundle)> {
    if data.len() < 11 {
        return Err(BundleError::Truncated("file (too short for header)"));
    }
    if &data[0..6] != AXONML_MAGIC {
        return Err(BundleError::BadMagic);
    }
    let version = data[6];
    if version != AXONML_BUNDLE_VERSION {
        return Err(BundleError::BadVersion(version));
    }

    let header_len = u32::from_le_bytes([data[7], data[8], data[9], data[10]]) as usize;
    let header_start = 11usize;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or(BundleError::Invalid("header length overflow".into()))?;
    if data.len() < header_end + 4 {
        return Err(BundleError::Truncated("header"));
    }
    let header: BundleHeader = serde_json::from_slice(&data[header_start..header_end])?;

    let weights_len_bytes = &data[header_end..header_end + 4];
    let weights_len = u32::from_le_bytes([
        weights_len_bytes[0],
        weights_len_bytes[1],
        weights_len_bytes[2],
        weights_len_bytes[3],
    ]) as usize;
    let weights_start = header_end + 4;
    let weights_end = weights_start
        .checked_add(weights_len)
        .ok_or(BundleError::Invalid("weights length overflow".into()))?;
    if data.len() < weights_end {
        return Err(BundleError::Truncated("weights"));
    }
    let bundle: ModelBundle = serde_json::from_slice(&data[weights_start..weights_end])?;

    Ok((header, bundle))
}

/// Decode only the header of a `.axonml` file without reading the weights blob.
///
/// Useful for model registries / UIs that list metadata without paying for the
/// full weight deserialization cost on every request.
pub fn load_header<P: AsRef<Path>>(path: P) -> BundleResult<BundleHeader> {
    let data = fs::read(path)?;
    if data.len() < 11 {
        return Err(BundleError::Truncated("file (too short for header)"));
    }
    if &data[0..6] != AXONML_MAGIC {
        return Err(BundleError::BadMagic);
    }
    let version = data[6];
    if version != AXONML_BUNDLE_VERSION {
        return Err(BundleError::BadVersion(version));
    }
    let header_len = u32::from_le_bytes([data[7], data[8], data[9], data[10]]) as usize;
    let header_end = 11usize
        .checked_add(header_len)
        .ok_or(BundleError::Invalid("header length overflow".into()))?;
    if data.len() < header_end {
        return Err(BundleError::Truncated("header"));
    }
    Ok(serde_json::from_slice(&data[11..header_end])?)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn round_trip_sentinel_bundle() {
        let weights: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let bundle = ModelBundle::new("sentinel", 11, weights.clone())
            .with_hyperparam("hidden_dim", 128)
            .with_hyperparam("num_layers", 2)
            .with_normalization(vec![0.0; 11], vec![1.0; 11])
            .with_threshold(0.5);

        let tmp = NamedTempFile::new().unwrap();
        let final_path = save_bundle(&bundle, tmp.path()).unwrap();

        let (header, loaded) = load_bundle(&final_path).unwrap();
        assert_eq!(header.architecture, "sentinel");
        assert_eq!(header.input_features, 11);
        assert_eq!(header.num_parameters, weights.len());
        assert!(!header.quantized);

        assert_eq!(loaded.architecture, "sentinel");
        assert_eq!(loaded.weights, weights);
        assert_eq!(
            loaded
                .hyperparameters
                .get("hidden_dim")
                .and_then(|v| v.as_i64()),
            Some(128)
        );
        assert_eq!(loaded.anomaly_threshold, Some(0.5));
    }

    #[test]
    fn header_only_decode_skips_weights() {
        let weights: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
        let bundle = ModelBundle::new("bert", 128, weights);

        let tmp = NamedTempFile::new().unwrap();
        let path = save_bundle(&bundle, tmp.path()).unwrap();

        let header = load_header(&path).unwrap();
        assert_eq!(header.architecture, "bert");
        assert_eq!(header.num_parameters, 10_000);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = vec![b'X', b'X', b'X', b'X', b'X', b'X'];
        bytes.extend_from_slice(&[1]);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = load_bundle_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, BundleError::BadMagic));
    }

    #[test]
    fn rejects_bad_version() {
        let mut bytes = AXONML_MAGIC.to_vec();
        bytes.push(99);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = load_bundle_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, BundleError::BadVersion(99)));
    }

    #[test]
    fn rejects_truncated_header() {
        let mut bytes = AXONML_MAGIC.to_vec();
        bytes.push(AXONML_BUNDLE_VERSION);
        bytes.extend_from_slice(&500u32.to_le_bytes()); // claims 500-byte header
        // but only 4 more bytes of "header" provided:
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        let err = load_bundle_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, BundleError::Truncated(_)));
    }

    #[test]
    fn save_adds_axonml_extension_when_missing() {
        let bundle = ModelBundle::new("phantom", 20, vec![1.0, 2.0, 3.0]);
        let dir = tempfile::tempdir().unwrap();
        let no_ext = dir.path().join("my_model");
        let final_path = save_bundle(&bundle, &no_ext).unwrap();
        assert_eq!(final_path.extension().unwrap(), "axonml");
        assert!(final_path.exists());
    }
}
