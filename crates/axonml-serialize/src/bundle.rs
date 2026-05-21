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
//! April 27, 2026 — added optional `graph` field to `ModelBundle` carrying full
//! compute-graph topology (nodes, inputs/outputs, initializers as named tensors).
//! Backwards compatible: legacy bundles without `graph` still load via
//! `#[serde(default)]`. The graph field is what NexusFoundry's AxonML frontend
//! reads to compile to HEF without needing a Python rebuilder.
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
// Graph payload (optional — for NexusFoundry direct-compile path)
// =============================================================================

/// A named tensor with explicit shape + dtype. Used for graph initializers
/// (weights, biases, batchnorm params) and for declaring graph I/O shapes.
///
/// `dtype` is currently always `"f32"`; reserved for future quantized variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedTensor {
    /// Tensor shape (row-major / NCHW for 4D conv weights).
    pub shape: Vec<i64>,
    /// Element dtype as a short string (currently always `"f32"`).
    #[serde(default = "default_dtype_f32")]
    pub dtype: String,
    /// Flat row-major data buffer.
    pub data: Vec<f32>,
}

fn default_dtype_f32() -> String {
    "f32".to_string()
}

/// A graph-level input or output declaration. Shape may contain -1 for dynamic
/// dimensions (commonly the batch dim).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphIo {
    /// Tensor name as referenced by graph nodes.
    pub name: String,
    /// Shape with `-1` denoting a dynamic (batch) dimension.
    pub shape: Vec<i64>,
    /// Element dtype as a short string (currently always `"f32"`).
    #[serde(default = "default_dtype_f32")]
    pub dtype: String,
}

/// A single compute node. `op` matches a NexusFoundry `IrOp` variant name
/// (e.g. `"Conv2d"`, `"BatchNorm"`, `"Relu"`, `"MaxPool"`, `"GlobalAvgPool"`,
/// `"Gemm"`). `attrs` is op-specific JSON; consumers parse based on `op`.
///
/// `inputs`/`outputs` are tensor names — both activations (declared in
/// `BundleGraph::inputs` or produced by an earlier node) and initializers
/// (declared in `BundleGraph::initializers`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node name.
    pub name: String,
    /// Op kind, matching a NexusFoundry `IrOp` variant name verbatim.
    pub op: String,
    /// Op-specific attribute bag (kernel_shape / strides / padding / etc).
    #[serde(default)]
    pub attrs: serde_json::Value,
    /// Input tensor names (activations + initializers, in op-defined order).
    pub inputs: Vec<String>,
    /// Output tensor names (one per produced activation).
    pub outputs: Vec<String>,
}

/// Full compute graph topology: I/O declarations + topologically-ordered
/// compute nodes + named weight tensors.
///
/// This is what NexusFoundry's AxonML frontend consumes to build a populated
/// FoundryIR. Without `graph`, the AxonML file is weights-only and the parser
/// can only return raw tensors (which the rest of the compile pipeline drops
/// because there are zero compute nodes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleGraph {
    /// Graph-level input declarations.
    pub inputs: Vec<GraphIo>,
    /// Graph-level output declarations.
    pub outputs: Vec<GraphIo>,
    /// Compute nodes in topological order.
    pub nodes: Vec<GraphNode>,
    /// Named initializer tensors (weights, biases, BN params).
    pub initializers: HashMap<String, NamedTensor>,
}

impl BundleGraph {
    /// Create an empty graph (no I/O, no nodes, no initializers).
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            initializers: HashMap::new(),
        }
    }

    /// Add a graph-level input tensor (fp32).
    pub fn add_input(&mut self, name: &str, shape: Vec<i64>) {
        self.inputs.push(GraphIo {
            name: name.into(),
            shape,
            dtype: "f32".into(),
        });
    }

    /// Add a graph-level output tensor (fp32).
    pub fn add_output(&mut self, name: &str, shape: Vec<i64>) {
        self.outputs.push(GraphIo {
            name: name.into(),
            shape,
            dtype: "f32".into(),
        });
    }

    /// Append a compute node to the graph (caller is responsible for topological order).
    pub fn add_node(
        &mut self,
        name: &str,
        op: &str,
        attrs: serde_json::Value,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
    ) {
        self.nodes.push(GraphNode {
            name: name.into(),
            op: op.into(),
            attrs,
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
        });
    }

    /// Insert a named initializer (weight / bias / BN param). Asserts shape matches data length.
    pub fn add_initializer(&mut self, name: &str, shape: Vec<i64>, data: Vec<f32>) {
        let expected: usize = shape.iter().map(|&d| d as usize).product();
        debug_assert_eq!(
            expected,
            data.len(),
            "initializer {name}: shape product {expected} != data length {}",
            data.len()
        );
        self.initializers.insert(
            name.into(),
            NamedTensor {
                shape,
                dtype: "f32".into(),
                data,
            },
        );
    }
}

impl Default for BundleGraph {
    fn default() -> Self {
        Self::new()
    }
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
    /// Optional full compute-graph topology + named initializer tensors.
    ///
    /// When present, `weights` may be empty — the canonical weight payload
    /// lives under `graph.initializers` (named tensors with shape + data).
    /// When absent, the bundle is in legacy "flat-weights + Python rebuilder"
    /// mode and the architecture must be reconstructed by name from the
    /// `architecture` tag (see `tools/model_converter/convert.py`).
    #[serde(default)]
    pub graph: Option<BundleGraph>,
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
            graph: None,
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

    /// Attach a full compute graph (replacing any previously attached graph).
    pub fn with_graph(mut self, graph: BundleGraph) -> Self {
        self.graph = Some(graph);
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

    #[test]
    fn round_trip_bundle_with_graph() {
        // Conv2d -> BatchNorm -> Relu -> GlobalAvgPool -> Gemm — the same skeleton
        // the NexusFoundry e2e_pipeline test uses for its synthetic IR.
        let mut graph = BundleGraph::new();

        graph.add_input("input", vec![-1, 3, 32, 32]);
        graph.add_output("logits", vec![-1, 10]);

        graph.add_initializer(
            "conv.weight",
            vec![16, 3, 3, 3],
            (0..16 * 3 * 3 * 3).map(|i| i as f32 * 0.01).collect(),
        );
        graph.add_initializer("conv.bias", vec![16], vec![0.0; 16]);
        graph.add_initializer("bn.weight", vec![16], vec![1.0; 16]);
        graph.add_initializer("bn.bias", vec![16], vec![0.0; 16]);
        graph.add_initializer("bn.running_mean", vec![16], vec![0.0; 16]);
        graph.add_initializer("bn.running_var", vec![16], vec![1.0; 16]);
        graph.add_initializer(
            "fc.weight",
            vec![10, 16],
            (0..160).map(|i| i as f32 * 0.001).collect(),
        );
        graph.add_initializer("fc.bias", vec![10], vec![0.0; 10]);

        graph.add_node(
            "conv1",
            "Conv2d",
            serde_json::json!({
                "kernel_shape": [3, 3],
                "strides": [1, 1],
                "padding": [1, 1, 1, 1],
                "group": 1,
            }),
            vec!["input", "conv.weight", "conv.bias"],
            vec!["conv_out"],
        );
        graph.add_node(
            "bn1",
            "BatchNorm",
            serde_json::json!({"epsilon": 1e-5, "momentum": 0.9}),
            vec![
                "conv_out",
                "bn.weight",
                "bn.bias",
                "bn.running_mean",
                "bn.running_var",
            ],
            vec!["bn_out"],
        );
        graph.add_node(
            "relu1",
            "Relu",
            serde_json::Value::Null,
            vec!["bn_out"],
            vec!["relu_out"],
        );
        graph.add_node(
            "gap1",
            "GlobalAvgPool",
            serde_json::Value::Null,
            vec!["relu_out"],
            vec!["pooled"],
        );
        graph.add_node(
            "fc1",
            "Gemm",
            serde_json::json!({"alpha": 1.0, "beta": 1.0, "trans_a": false, "trans_b": true}),
            vec!["pooled", "fc.weight", "fc.bias"],
            vec!["logits"],
        );

        let bundle = ModelBundle::new("conv2d", 3, Vec::new()) // weights vec empty when graph is present
            .with_hyperparam("input_h", 32)
            .with_hyperparam("input_w", 32)
            .with_hyperparam("num_classes", 10)
            .with_graph(graph);

        let tmp = NamedTempFile::new().unwrap();
        let final_path = save_bundle(&bundle, tmp.path()).unwrap();

        let (header, loaded) = load_bundle(&final_path).unwrap();
        assert_eq!(header.architecture, "conv2d");
        assert!(loaded.graph.is_some(), "graph should round-trip");

        let g = loaded.graph.as_ref().unwrap();
        assert_eq!(g.inputs.len(), 1);
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.nodes.len(), 5);
        assert_eq!(g.initializers.len(), 8);

        // Spot-check ops appear in the right order
        assert_eq!(g.nodes[0].op, "Conv2d");
        assert_eq!(g.nodes[1].op, "BatchNorm");
        assert_eq!(g.nodes[2].op, "Relu");
        assert_eq!(g.nodes[3].op, "GlobalAvgPool");
        assert_eq!(g.nodes[4].op, "Gemm");

        // Initializer data round-trips exactly
        let conv_w = g.initializers.get("conv.weight").unwrap();
        assert_eq!(conv_w.shape, vec![16, 3, 3, 3]);
        assert_eq!(conv_w.data.len(), 16 * 3 * 3 * 3);
        assert!((conv_w.data[5] - 0.05).abs() < 1e-6);
    }

    #[test]
    fn legacy_bundle_without_graph_loads_with_graph_none() {
        // A pre-2026-04-27 bundle (no graph field). Must still load.
        let bundle = ModelBundle::new("sentinel", 11, vec![1.0, 2.0, 3.0]);
        let tmp = NamedTempFile::new().unwrap();
        save_bundle(&bundle, tmp.path()).unwrap();

        let (_, loaded) = load_bundle(tmp.path().with_extension("axonml")).unwrap();
        assert!(loaded.graph.is_none());
        assert_eq!(loaded.weights, vec![1.0, 2.0, 3.0]);
    }
}
