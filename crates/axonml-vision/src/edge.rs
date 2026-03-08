//! Edge Deployment Pipeline
//!
//! Utilities for preparing vision models for edge deployment on resource-
//! constrained devices (ARM, Raspberry Pi, mobile). Provides model profiling,
//! quantization-ready export, and deployment configuration.
//!
//! # Targets
//!
//! - `armv7-unknown-linux-musleabihf` (Raspberry Pi, static musl)
//! - `aarch64-unknown-linux-gnu` (ARM64 Linux)
//! - `wasm32-unknown-unknown` (WebAssembly)

use axonml_nn::Module;

// =============================================================================
// Model Profile
// =============================================================================

/// Profile of a model's resource requirements.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Model name
    pub name: String,
    /// Total number of trainable parameters
    pub num_parameters: usize,
    /// Estimated model size in bytes (f32 weights)
    pub size_bytes_f32: usize,
    /// Estimated model size with INT8 quantization
    pub size_bytes_int8: usize,
    /// Estimated FLOPs for a single forward pass
    pub estimated_flops: usize,
    /// Recommended deployment target
    pub target: DeployTarget,
}

/// Deployment target recommendation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeployTarget {
    /// Suitable for edge: <5M params, <20MB
    Edge,
    /// Suitable for server: any size
    Server,
    /// Suitable for both with quantization
    Both,
}

impl std::fmt::Display for DeployTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeployTarget::Edge => write!(f, "Edge (ARM/Mobile)"),
            DeployTarget::Server => write!(f, "Server (GPU)"),
            DeployTarget::Both => write!(f, "Edge + Server"),
        }
    }
}

/// Profile a model's resource requirements.
pub fn profile_model(name: &str, model: &dyn Module) -> ModelProfile {
    let params = model.parameters();
    let num_parameters: usize = params
        .iter()
        .map(|p| p.variable().data().to_vec().len())
        .sum();

    let size_f32 = num_parameters * 4; // f32 = 4 bytes
    let size_int8 = num_parameters; // INT8 = 1 byte

    let target = if num_parameters < 5_000_000 {
        DeployTarget::Edge
    } else if num_parameters < 50_000_000 {
        DeployTarget::Both
    } else {
        DeployTarget::Server
    };

    ModelProfile {
        name: name.to_string(),
        num_parameters,
        size_bytes_f32: size_f32,
        size_bytes_int8: size_int8,
        estimated_flops: 0, // Would require tracing to compute accurately
        target,
    }
}

// =============================================================================
// Edge Model Registry
// =============================================================================

/// Information about an edge-optimized model.
#[derive(Debug, Clone)]
pub struct EdgeModelInfo {
    /// Model name
    pub name: &'static str,
    /// Brief description
    pub description: &'static str,
    /// Approximate parameter count
    pub approx_params: &'static str,
    /// Recommended input size
    pub input_size: &'static str,
    /// Primary use case
    pub task: &'static str,
}

/// List all edge-suitable vision models.
pub fn edge_models() -> Vec<EdgeModelInfo> {
    vec![
        EdgeModelInfo {
            name: "BlazeFace",
            description: "Ultra-fast face detection with depthwise separable convolutions",
            approx_params: "~100K",
            input_size: "128x128",
            task: "Face Detection",
        },
        EdgeModelInfo {
            name: "NanoDet",
            description: "Lightweight anchor-free object detection (ShuffleNet + Ghost PAN)",
            approx_params: "<1M",
            input_size: "320x320",
            task: "Object Detection",
        },
        EdgeModelInfo {
            name: "FastDepth",
            description: "Real-time monocular depth estimation with depthwise encoder",
            approx_params: "<4M",
            input_size: "224x224",
            task: "Depth Estimation",
        },
        EdgeModelInfo {
            name: "StudentTeacher",
            description: "Lightweight anomaly detection via teacher-student disagreement",
            approx_params: "~2M",
            input_size: "224x224",
            task: "Anomaly Detection",
        },
        EdgeModelInfo {
            name: "Aegis3D (LOD 0-4)",
            description: "3D reconstruction with coarse LOD for edge inference",
            approx_params: "Variable",
            input_size: "N/A",
            task: "3D Reconstruction",
        },
        EdgeModelInfo {
            name: "Mnemosyne",
            description: "Face identity via temporal crystallization (GRU attractor convergence)",
            approx_params: "~115K",
            input_size: "64x64",
            task: "Biometric Identity (Face)",
        },
        EdgeModelInfo {
            name: "Ariadne",
            description: "Fingerprint identity via Gabor ridge event fields",
            approx_params: "~65K",
            input_size: "128x128",
            task: "Biometric Identity (Fingerprint)",
        },
        EdgeModelInfo {
            name: "Echo",
            description: "Voice identity via predictive speaker residuals",
            approx_params: "~68K",
            input_size: "40xT mel",
            task: "Biometric Identity (Voice)",
        },
        EdgeModelInfo {
            name: "Argus",
            description: "Iris identity via polar-native radial phase encoding",
            approx_params: "~65K",
            input_size: "32x256 polar",
            task: "Biometric Identity (Iris)",
        },
        EdgeModelInfo {
            name: "Themis",
            description: "Multimodal biometric fusion via uncertainty-aware belief propagation",
            approx_params: "~49K",
            input_size: "N/A (fusion)",
            task: "Biometric Fusion",
        },
        EdgeModelInfo {
            name: "AegisIdentity",
            description: "Unified biometric system (all modalities, <400K params total)",
            approx_params: "~362K",
            input_size: "Multi-modal",
            task: "Biometric Identity (Full)",
        },
    ]
}

/// List all server-grade vision models.
pub fn server_models() -> Vec<EdgeModelInfo> {
    vec![
        EdgeModelInfo {
            name: "RetinaFace",
            description: "High-accuracy face detection with 5-point landmarks (ResNet + FPN)",
            approx_params: "~27M",
            input_size: "640x640",
            task: "Face Detection",
        },
        EdgeModelInfo {
            name: "DETR",
            description: "End-to-end object detection with Transformers (no NMS needed)",
            approx_params: "~41M",
            input_size: "800x800",
            task: "Object Detection",
        },
        EdgeModelInfo {
            name: "DPT",
            description: "Dense Prediction Transformer for high-quality monocular depth",
            approx_params: "~25M",
            input_size: "384x384",
            task: "Depth Estimation",
        },
        EdgeModelInfo {
            name: "PatchCore",
            description: "Feature-based anomaly detection (frozen backbone + kNN memory bank)",
            approx_params: "~11M",
            input_size: "224x224",
            task: "Anomaly Detection",
        },
        EdgeModelInfo {
            name: "VQAModel",
            description: "Visual Question Answering (ViT + Transformer + CrossAttention)",
            approx_params: "~15M",
            input_size: "224x224 + text",
            task: "Visual QA",
        },
        EdgeModelInfo {
            name: "Aegis3D (LOD 0-8)",
            description: "Full-resolution 3D reconstruction with adaptive octree SDF",
            approx_params: "Variable",
            input_size: "N/A",
            task: "3D Reconstruction",
        },
    ]
}

// =============================================================================
// Deployment Configuration
// =============================================================================

/// Configuration for edge deployment.
#[derive(Debug, Clone)]
pub struct EdgeDeployConfig {
    /// Target architecture
    pub target_arch: String,
    /// Whether to use INT8 quantization
    pub quantize_int8: bool,
    /// Maximum model size in bytes
    pub max_size_bytes: usize,
    /// Maximum inference latency target in milliseconds
    pub target_latency_ms: f32,
    /// Batch size for inference
    pub batch_size: usize,
}

impl EdgeDeployConfig {
    /// Configuration for Raspberry Pi deployment.
    pub fn raspberry_pi() -> Self {
        Self {
            target_arch: "armv7-unknown-linux-musleabihf".to_string(),
            quantize_int8: true,
            max_size_bytes: 20_000_000, // 20MB
            target_latency_ms: 100.0,
            batch_size: 1,
        }
    }

    /// Configuration for ARM64 Linux.
    pub fn arm64() -> Self {
        Self {
            target_arch: "aarch64-unknown-linux-gnu".to_string(),
            quantize_int8: true,
            max_size_bytes: 50_000_000,
            target_latency_ms: 50.0,
            batch_size: 1,
        }
    }

    /// Configuration for WASM deployment.
    pub fn wasm() -> Self {
        Self {
            target_arch: "wasm32-unknown-unknown".to_string(),
            quantize_int8: true,
            max_size_bytes: 10_000_000, // 10MB
            target_latency_ms: 200.0,
            batch_size: 1,
        }
    }

    /// Check if a model fits within this deployment config.
    pub fn model_fits(&self, profile: &ModelProfile) -> bool {
        let size = if self.quantize_int8 {
            profile.size_bytes_int8
        } else {
            profile.size_bytes_f32
        };
        size <= self.max_size_bytes
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_models_list() {
        let models = edge_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.name == "BlazeFace"));
        assert!(models.iter().any(|m| m.name == "NanoDet"));
        assert!(models.iter().any(|m| m.name == "FastDepth"));
    }

    #[test]
    fn test_server_models_list() {
        let models = server_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.name == "RetinaFace"));
        assert!(models.iter().any(|m| m.name == "DETR"));
    }

    #[test]
    fn test_deploy_config_rpi() {
        let config = EdgeDeployConfig::raspberry_pi();
        assert_eq!(config.target_arch, "armv7-unknown-linux-musleabihf");
        assert!(config.quantize_int8);

        // BlazeFace should fit on RPi
        let profile = ModelProfile {
            name: "blazeface".to_string(),
            num_parameters: 100_000,
            size_bytes_f32: 400_000,
            size_bytes_int8: 100_000,
            estimated_flops: 0,
            target: DeployTarget::Edge,
        };
        assert!(config.model_fits(&profile));
    }

    #[test]
    fn test_deploy_config_size_check() {
        let config = EdgeDeployConfig::wasm();

        // Large model should NOT fit
        let large = ModelProfile {
            name: "detr".to_string(),
            num_parameters: 41_000_000,
            size_bytes_f32: 164_000_000,
            size_bytes_int8: 41_000_000,
            estimated_flops: 0,
            target: DeployTarget::Server,
        };
        assert!(!config.model_fits(&large));
    }
}
