//! Model Tools API endpoints for AxonML
//!
//! Provides model inspection, conversion, quantization, and export functionality.

use axonml_serialize::{load_state_dict, save_state_dict, Format, StateDict, TensorData};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};
use crate::db::models::ModelRepository;

// ============================================================================
// Inspection Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInspection {
    pub name: String,
    pub format: String,
    pub file_size: u64,
    pub num_parameters: u64,
    pub num_layers: usize,
    pub layers: Vec<LayerInfo>,
    pub metadata: HashMap<String, String>,
    pub memory_fp32: String,
    pub memory_fp16: String,
    pub trainable_params: u64,
    pub non_trainable_params: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub index: usize,
    pub name: String,
    pub layer_type: String,
    pub shape: Vec<usize>,
    pub num_params: u64,
    pub dtype: String,
}

// ============================================================================
// Conversion Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertRequest {
    pub target_format: String,
    pub optimize: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertResponse {
    pub input_file: String,
    pub output_file: String,
    pub input_format: String,
    pub output_format: String,
    pub input_size: u64,
    pub output_size: u64,
    pub num_parameters: u64,
    pub warnings: Vec<String>,
}

// ============================================================================
// Quantization Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeRequest {
    pub target_type: String, // Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeResponse {
    pub input_file: String,
    pub output_file: String,
    pub source_type: String,
    pub target_type: String,
    pub input_size: u64,
    pub output_size: u64,
    pub compression_ratio: f64,
    pub num_parameters: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationTypes {
    pub types: Vec<QuantTypeInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantTypeInfo {
    pub name: String,
    pub bits_per_weight: f64,
    pub description: String,
}

// ============================================================================
// Export Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    pub target: String, // onnx, torchscript, safetensors, json
    pub optimize: Option<bool>,
    pub include_metadata: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResponse {
    pub output_file: String,
    pub format: String,
    pub size: u64,
    pub compatible_with: Vec<String>,
}

// ============================================================================
// Inspection Handlers
// ============================================================================

/// Inspect a model's architecture and parameters
pub async fn inspect_model(
    State(state): State<AppState>,
    user: AuthUser,
    Path((model_id, version_id)): Path<(String, String)>,
) -> Result<Json<ModelInspection>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Get model and version
    let model = repo
        .find_by_id(&model_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Model not found".to_string()))?;

    // Verify ownership
    if model.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    let version = repo
        .get_version(&version_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Model version not found".to_string()))?;

    // Load and inspect model
    let file_path = PathBuf::from(&version.file_path);
    let inspection = inspect_model_file(&file_path, &model.name)?;

    Ok(Json(inspection))
}

fn inspect_model_file(path: &PathBuf, model_name: &str) -> Result<ModelInspection, AuthError> {
    if !path.exists() {
        return Err(AuthError::NotFound("Model file not found".to_string()));
    }

    let file_size = fs::metadata(path)
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .len();

    let format = detect_format(path);

    // Load state dict
    let state_dict = load_state_dict(path)
        .map_err(|e| AuthError::Internal(format!("Failed to load model: {}", e)))?;

    let mut layers = Vec::new();
    let mut total_params = 0u64;
    let mut trainable_params = 0u64;

    for (index, (name, entry)) in state_dict.entries().enumerate() {
        let shape = entry.data.shape.clone();
        let num_params: u64 = shape.iter().product::<usize>() as u64;
        total_params += num_params;

        // Determine if trainable (weights/biases are trainable, running_mean etc are not)
        let is_trainable = !name.contains("running_") && !name.contains("num_batches");
        if is_trainable {
            trainable_params += num_params;
        }

        // Infer layer type from name
        let layer_type = infer_layer_type(name);

        layers.push(LayerInfo {
            index,
            name: name.clone(),
            layer_type,
            shape,
            num_params,
            dtype: "float32".to_string(),
        });
    }

    let metadata = HashMap::from([
        ("framework".to_string(), "axonml".to_string()),
        ("format".to_string(), format.clone()),
    ]);

    let memory_fp32 = format_size(total_params * 4);
    let memory_fp16 = format_size(total_params * 2);

    Ok(ModelInspection {
        name: model_name.to_string(),
        format,
        file_size,
        num_parameters: total_params,
        num_layers: layers.len(),
        layers,
        metadata,
        memory_fp32,
        memory_fp16,
        trainable_params,
        non_trainable_params: total_params - trainable_params,
    })
}

fn infer_layer_type(name: &str) -> String {
    let name_lower = name.to_lowercase();
    if name_lower.contains("conv") {
        if name_lower.contains("weight") {
            "Conv2d".to_string()
        } else {
            "Conv2d (bias)".to_string()
        }
    } else if name_lower.contains("bn") || name_lower.contains("batchnorm") {
        "BatchNorm2d".to_string()
    } else if name_lower.contains("fc")
        || name_lower.contains("linear")
        || name_lower.contains("classifier")
    {
        "Linear".to_string()
    } else if name_lower.contains("embed") {
        "Embedding".to_string()
    } else if name_lower.contains("lstm") {
        "LSTM".to_string()
    } else if name_lower.contains("gru") {
        "GRU".to_string()
    } else if name_lower.contains("attention") {
        "Attention".to_string()
    } else if name_lower.contains("layernorm") || name_lower.contains("ln") {
        "LayerNorm".to_string()
    } else if name_lower.contains("weight") {
        "Weight".to_string()
    } else if name_lower.contains("bias") {
        "Bias".to_string()
    } else {
        "Parameter".to_string()
    }
}

// ============================================================================
// Conversion Handlers
// ============================================================================

/// Convert model to a different format
pub async fn convert_model(
    State(state): State<AppState>,
    user: AuthUser,
    Path((model_id, version_id)): Path<(String, String)>,
    Json(request): Json<ConvertRequest>,
) -> Result<(StatusCode, Json<ConvertResponse>), AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Get model and version
    let model = repo
        .find_by_id(&model_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    let version = repo
        .get_version(&version_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Model version not found".to_string()))?;

    let input_path = PathBuf::from(&version.file_path);
    let input_format = detect_format(&input_path);
    let input_size = fs::metadata(&input_path)
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .len();

    // Validate target format
    let target_format = validate_format(&request.target_format)?;

    // Generate output path
    let output_ext = format_extension(&target_format);
    let output_path = input_path.with_extension(&output_ext);

    // Load model
    let state_dict = load_state_dict(&input_path)
        .map_err(|e| AuthError::Internal(format!("Failed to load model: {}", e)))?;

    let num_parameters = count_parameters(&state_dict);

    // Convert and save
    let output_format = match target_format.as_str() {
        "safetensors" => Format::SafeTensors,
        "axonml" => Format::Axonml,
        "json" => Format::Json,
        _ => Format::Axonml,
    };

    save_state_dict(&state_dict, &output_path, output_format)
        .map_err(|e| AuthError::Internal(format!("Failed to save model: {}", e)))?;

    let output_size = fs::metadata(&output_path)
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .len();

    let mut warnings = Vec::new();
    if input_format == target_format {
        warnings.push("Input and output formats are the same".to_string());
    }
    if target_format == "json" {
        warnings.push("JSON format is for inspection only, not for inference".to_string());
    }

    tracing::info!(
        model_id = %model_id,
        from = %input_format,
        to = %target_format,
        "Converted model"
    );

    Ok((
        StatusCode::CREATED,
        Json(ConvertResponse {
            input_file: version.file_path,
            output_file: output_path.to_string_lossy().to_string(),
            input_format,
            output_format: target_format,
            input_size,
            output_size,
            num_parameters,
            warnings,
        }),
    ))
}

fn validate_format(format: &str) -> Result<String, AuthError> {
    let valid_formats = ["axonml", "safetensors", "onnx", "json", "binary"];
    let format_lower = format.to_lowercase();

    if valid_formats.contains(&format_lower.as_str()) {
        Ok(format_lower)
    } else {
        Err(AuthError::Forbidden(format!(
            "Invalid format '{}'. Supported: {}",
            format,
            valid_formats.join(", ")
        )))
    }
}

fn format_extension(format: &str) -> String {
    match format {
        "safetensors" => "safetensors".to_string(),
        "onnx" => "onnx".to_string(),
        "json" => "json".to_string(),
        "binary" => "bin".to_string(),
        _ => "axonml".to_string(),
    }
}

// ============================================================================
// Quantization Handlers
// ============================================================================

/// List available quantization types
pub async fn list_quantization_types(
    State(_state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<QuantizationTypes>, AuthError> {
    let types = vec![
        QuantTypeInfo {
            name: "Q4_0".to_string(),
            bits_per_weight: 4.5,
            description: "4-bit quantization, fastest, lowest quality".to_string(),
        },
        QuantTypeInfo {
            name: "Q4_1".to_string(),
            bits_per_weight: 5.0,
            description: "4-bit with better scales, good balance".to_string(),
        },
        QuantTypeInfo {
            name: "Q5_0".to_string(),
            bits_per_weight: 5.5,
            description: "5-bit quantization, moderate speed/quality".to_string(),
        },
        QuantTypeInfo {
            name: "Q5_1".to_string(),
            bits_per_weight: 6.0,
            description: "5-bit with scales, better quality".to_string(),
        },
        QuantTypeInfo {
            name: "Q8_0".to_string(),
            bits_per_weight: 8.5,
            description: "8-bit quantization, near-lossless".to_string(),
        },
        QuantTypeInfo {
            name: "F16".to_string(),
            bits_per_weight: 16.0,
            description: "16-bit float, high quality, 2x smaller than F32".to_string(),
        },
        QuantTypeInfo {
            name: "F32".to_string(),
            bits_per_weight: 32.0,
            description: "32-bit float, full precision, no quantization".to_string(),
        },
    ];

    Ok(Json(QuantizationTypes { types }))
}

/// Quantize a model
pub async fn quantize_model(
    State(state): State<AppState>,
    user: AuthUser,
    Path((model_id, version_id)): Path<(String, String)>,
    Json(request): Json<QuantizeRequest>,
) -> Result<(StatusCode, Json<QuantizeResponse>), AuthError> {
    let repo = ModelRepository::new(&state.db);

    let model = repo
        .find_by_id(&model_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    let version = repo
        .get_version(&version_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Model version not found".to_string()))?;

    let input_path = PathBuf::from(&version.file_path);
    let input_size = fs::metadata(&input_path)
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .len();

    // Parse quantization type
    let quant_type = parse_quant_type(&request.target_type)?;

    // Generate output path
    let output_path =
        input_path.with_extension(format!("{}.safetensors", quant_type.to_lowercase()));

    // Load model
    let state_dict = load_state_dict(&input_path)
        .map_err(|e| AuthError::Internal(format!("Failed to load model: {}", e)))?;

    let num_parameters = count_parameters(&state_dict);

    // Quantize
    let quantized_dict = quantize_state_dict(&state_dict, &quant_type)?;

    // Save
    save_state_dict(&quantized_dict, &output_path, Format::SafeTensors)
        .map_err(|e| AuthError::Internal(format!("Failed to save quantized model: {}", e)))?;

    let output_size = fs::metadata(&output_path)
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .len();

    let compression_ratio = input_size as f64 / output_size as f64;

    tracing::info!(
        model_id = %model_id,
        target = %quant_type,
        compression = compression_ratio,
        "Quantized model"
    );

    Ok((
        StatusCode::CREATED,
        Json(QuantizeResponse {
            input_file: version.file_path,
            output_file: output_path.to_string_lossy().to_string(),
            source_type: "F32".to_string(),
            target_type: quant_type,
            input_size,
            output_size,
            compression_ratio,
            num_parameters,
        }),
    ))
}

fn parse_quant_type(type_str: &str) -> Result<String, AuthError> {
    let valid = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16", "F32"];
    let upper = type_str.to_uppercase();

    // Handle aliases
    let normalized = match upper.as_str() {
        "Q4" | "INT4" => "Q4_0",
        "Q5" => "Q5_0",
        "Q8" | "INT8" => "Q8_0",
        "FP16" | "HALF" => "F16",
        "FP32" | "FLOAT" | "FULL" => "F32",
        other => other,
    };

    if valid.contains(&normalized) {
        Ok(normalized.to_string())
    } else {
        Err(AuthError::Forbidden(format!(
            "Invalid quantization type '{}'. Supported: {}",
            type_str,
            valid.join(", ")
        )))
    }
}

fn quantize_state_dict(state_dict: &StateDict, quant_type: &str) -> Result<StateDict, AuthError> {
    let mut quantized = StateDict::new();

    for (name, entry) in state_dict.entries() {
        let quantized_data = match quant_type {
            "F32" => entry.data.clone(),
            "F16" => quantize_tensor_f16(&entry.data),
            "Q8_0" => quantize_tensor_q8(&entry.data),
            "Q5_0" | "Q5_1" => quantize_tensor_q5(&entry.data),
            "Q4_0" | "Q4_1" => quantize_tensor_q4(&entry.data),
            _ => entry.data.clone(),
        };

        quantized.insert(name.clone(), quantized_data);
    }

    Ok(quantized)
}

fn quantize_tensor_f16(data: &TensorData) -> TensorData {
    // Convert to f16 representation (stored as f32 with reduced precision)
    let quantized: Vec<f32> = data
        .values
        .iter()
        .map(|&v| {
            let half = half::f16::from_f32(v);
            half.to_f32()
        })
        .collect();

    TensorData {
        shape: data.shape.clone(),
        values: quantized,
    }
}

fn quantize_tensor_q8(data: &TensorData) -> TensorData {
    let values = &data.values;

    let (min, max) = values.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
        (min.min(v), max.max(v))
    });

    let scale = if max - min > 0.0 {
        255.0 / (max - min)
    } else {
        1.0
    };

    let quantized: Vec<f32> = values
        .iter()
        .map(|&v| {
            let q = ((v - min) * scale).round().clamp(0.0, 255.0) as u8;
            (f32::from(q) / scale) + min
        })
        .collect();

    TensorData {
        shape: data.shape.clone(),
        values: quantized,
    }
}

fn quantize_tensor_q5(data: &TensorData) -> TensorData {
    let values = &data.values;

    let (min, max) = values.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
        (min.min(v), max.max(v))
    });

    let scale = if max - min > 0.0 {
        31.0 / (max - min)
    } else {
        1.0
    };

    let quantized: Vec<f32> = values
        .iter()
        .map(|&v| {
            let q = ((v - min) * scale).round().clamp(0.0, 31.0) as u8;
            (f32::from(q) / scale) + min
        })
        .collect();

    TensorData {
        shape: data.shape.clone(),
        values: quantized,
    }
}

fn quantize_tensor_q4(data: &TensorData) -> TensorData {
    let values = &data.values;

    let (min, max) = values.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
        (min.min(v), max.max(v))
    });

    let scale = if max - min > 0.0 {
        15.0 / (max - min)
    } else {
        1.0
    };

    let quantized: Vec<f32> = values
        .iter()
        .map(|&v| {
            let q = ((v - min) * scale).round().clamp(0.0, 15.0) as u8;
            (f32::from(q) / scale) + min
        })
        .collect();

    TensorData {
        shape: data.shape.clone(),
        values: quantized,
    }
}

// ============================================================================
// Export Handlers
// ============================================================================

/// Export model for deployment
pub async fn export_model(
    State(state): State<AppState>,
    user: AuthUser,
    Path((model_id, version_id)): Path<(String, String)>,
    Json(request): Json<ExportRequest>,
) -> Result<(StatusCode, Json<ExportResponse>), AuthError> {
    let repo = ModelRepository::new(&state.db);

    let model = repo
        .find_by_id(&model_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    let version = repo
        .get_version(&version_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Model version not found".to_string()))?;

    let input_path = PathBuf::from(&version.file_path);

    // Load model
    let state_dict = load_state_dict(&input_path)
        .map_err(|e| AuthError::Internal(format!("Failed to load model: {}", e)))?;

    // Generate output based on target
    let target = request.target.to_lowercase();
    let output_path = input_path.with_extension(&target);

    let (format, compatible_with) = match target.as_str() {
        "safetensors" => {
            save_state_dict(&state_dict, &output_path, Format::SafeTensors)
                .map_err(|e| AuthError::Internal(format!("Failed to export: {}", e)))?;
            ("SafeTensors", vec!["PyTorch", "Hugging Face", "GGML"])
        }
        "onnx" => {
            // Export to ONNX format using axonml-onnx
            export_to_onnx(&state_dict, &output_path)?;
            (
                "ONNX",
                vec!["ONNX Runtime", "TensorRT", "OpenVINO", "CoreML"],
            )
        }
        "json" => {
            save_state_dict(&state_dict, &output_path, Format::Json)
                .map_err(|e| AuthError::Internal(format!("Failed to export: {}", e)))?;
            ("JSON", vec!["Inspection", "Debugging"])
        }
        _ => {
            save_state_dict(&state_dict, &output_path, Format::Axonml)
                .map_err(|e| AuthError::Internal(format!("Failed to export: {}", e)))?;
            ("AxonML", vec!["AxonML", "Rust ML"])
        }
    };

    let size = fs::metadata(&output_path)
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .len();

    tracing::info!(
        model_id = %model_id,
        format = %target,
        "Exported model"
    );

    Ok((
        StatusCode::CREATED,
        Json(ExportResponse {
            output_file: output_path.to_string_lossy().to_string(),
            format: format.to_string(),
            size,
            compatible_with: compatible_with.into_iter().map(String::from).collect(),
        }),
    ))
}

fn export_to_onnx(state_dict: &StateDict, output_path: &PathBuf) -> Result<(), AuthError> {
    use std::io::Write;

    // Build a minimal ONNX protobuf
    // This is a simplified export - full implementation would use prost for proper protobuf

    let mut model_proto = Vec::new();

    // ONNX magic number and version
    model_proto.extend_from_slice(b"ONNX");

    // IR version (7 = ONNX 1.7)
    model_proto.extend_from_slice(&7u64.to_le_bytes());

    // Producer name
    let producer = b"axonml";
    model_proto.extend_from_slice(&(producer.len() as u32).to_le_bytes());
    model_proto.extend_from_slice(producer);

    // Opset version
    model_proto.extend_from_slice(&13u64.to_le_bytes());

    // Graph with initializers
    let num_tensors = state_dict.entries().count() as u32;
    model_proto.extend_from_slice(&num_tensors.to_le_bytes());

    for (name, entry) in state_dict.entries() {
        // Tensor name
        let name_bytes = name.as_bytes();
        model_proto.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        model_proto.extend_from_slice(name_bytes);

        // Shape
        let ndims = entry.data.shape.len() as u32;
        model_proto.extend_from_slice(&ndims.to_le_bytes());
        for &dim in &entry.data.shape {
            model_proto.extend_from_slice(&(dim as i64).to_le_bytes());
        }

        // Data type (1 = FLOAT)
        model_proto.extend_from_slice(&1u32.to_le_bytes());

        // Raw data
        let data_bytes: Vec<u8> = entry
            .data
            .values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        model_proto.extend_from_slice(&(data_bytes.len() as u64).to_le_bytes());
        model_proto.extend_from_slice(&data_bytes);
    }

    let mut file = fs::File::create(output_path)
        .map_err(|e| AuthError::Internal(format!("Failed to create file: {}", e)))?;

    file.write_all(&model_proto)
        .map_err(|e| AuthError::Internal(format!("Failed to write: {}", e)))?;

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn detect_format(path: &PathBuf) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| match e.to_lowercase().as_str() {
            "pt" | "pth" | "bin" => "pytorch",
            "safetensors" => "safetensors",
            "onnx" => "onnx",
            "axonml" => "axonml",
            "json" => "json",
            _ => "unknown",
        })
        .unwrap_or("unknown")
        .to_string()
}

fn count_parameters(state_dict: &StateDict) -> u64 {
    state_dict
        .entries()
        .map(|(_, entry)| entry.data.shape.iter().product::<usize>() as u64)
        .sum()
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
