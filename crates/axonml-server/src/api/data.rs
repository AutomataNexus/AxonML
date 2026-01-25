//! Data API endpoints for AxonML
//!
//! Provides dataset analysis, validation, configuration generation, and preview.

use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};
use crate::db::datasets::Dataset;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetType {
    Image,
    Tabular,
    Text,
    Audio,
    Mixed,
    Unknown,
}

impl Default for DatasetType {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    pub total_size_bytes: u64,
    pub num_files: usize,
    pub file_types: HashMap<String, usize>,
    pub missing_values: usize,
    pub duplicate_samples: usize,
    pub mean_values: Option<Vec<f64>>,
    pub std_values: Option<Vec<f64>>,
    pub min_values: Option<Vec<f64>>,
    pub max_values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecommendations {
    pub architecture: String,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub optimizer: String,
    pub loss_function: String,
    pub transforms: Vec<String>,
    pub augmentations: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAnalysis {
    pub name: String,
    pub path: String,
    pub data_type: DatasetType,
    pub task_type: String,
    pub num_samples: usize,
    pub num_classes: Option<usize>,
    pub class_distribution: Option<HashMap<String, usize>>,
    pub input_shape: Option<Vec<usize>>,
    pub feature_names: Option<Vec<String>>,
    pub statistics: DataStatistics,
    pub recommendations: TrainingRecommendations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreviewResponse {
    pub dataset_id: String,
    pub data_type: DatasetType,
    pub samples: Vec<SamplePreview>,
    pub total_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplePreview {
    pub index: usize,
    pub filename: Option<String>,
    pub label: Option<String>,
    pub preview: String,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub warnings: Vec<String>,
    pub class_distribution: Option<HashMap<String, usize>>,
    pub missing_files: Vec<String>,
    pub corrupted_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: String,
    pub message: String,
    pub file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfigRequest {
    pub format: Option<String>, // toml, json
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfigResponse {
    pub config: String,
    pub format: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalyzeQuery {
    pub data_type: Option<String>,
    pub max_samples: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreviewQuery {
    pub num_samples: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ValidateQuery {
    pub num_classes: Option<usize>,
    pub check_balance: Option<bool>,
}

// ============================================================================
// Handlers
// ============================================================================

/// Analyze a dataset
pub async fn analyze_dataset(
    State(state): State<AppState>,
    user: AuthUser,
    Path(dataset_id): Path<String>,
    Query(query): Query<AnalyzeQuery>,
) -> Result<Json<DatasetAnalysis>, AuthError> {
    // Get dataset from database
    let dataset = state.db.doc_get("axonml_datasets", &dataset_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Dataset not found".to_string()))?;

    let dataset: Dataset = serde_json::from_value(dataset)
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    if dataset.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    let path = PathBuf::from(&dataset.file_path);
    let data_type = query.data_type
        .map(|t| parse_data_type(&t))
        .unwrap_or_else(|| detect_data_type(&path));
    let max_samples = query.max_samples.unwrap_or(1000);

    let analysis = analyze_dataset_path(&path, &dataset.name, data_type, max_samples)?;

    Ok(Json(analysis))
}

/// Preview dataset samples
pub async fn preview_dataset(
    State(state): State<AppState>,
    user: AuthUser,
    Path(dataset_id): Path<String>,
    Query(query): Query<PreviewQuery>,
) -> Result<Json<DataPreviewResponse>, AuthError> {
    let dataset = state.db.doc_get("axonml_datasets", &dataset_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Dataset not found".to_string()))?;

    let dataset: Dataset = serde_json::from_value(dataset)
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    if dataset.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    let path = PathBuf::from(&dataset.file_path);
    let num_samples = query.num_samples.unwrap_or(5);
    let data_type = detect_data_type(&path);

    let samples = preview_samples(&path, &data_type, num_samples)?;

    Ok(Json(DataPreviewResponse {
        dataset_id,
        data_type,
        samples,
        total_samples: dataset.num_samples.unwrap_or(0) as usize,
    }))
}

/// Validate dataset structure
pub async fn validate_dataset(
    State(state): State<AppState>,
    user: AuthUser,
    Path(dataset_id): Path<String>,
    Query(query): Query<ValidateQuery>,
) -> Result<Json<ValidationResult>, AuthError> {
    let dataset = state.db.doc_get("axonml_datasets", &dataset_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Dataset not found".to_string()))?;

    let dataset: Dataset = serde_json::from_value(dataset)
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    if dataset.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    let path = PathBuf::from(&dataset.file_path);
    let result = validate_dataset_path(&path, &query)?;

    Ok(Json(result))
}

/// Generate training configuration
pub async fn generate_config(
    State(state): State<AppState>,
    user: AuthUser,
    Path(dataset_id): Path<String>,
    Json(request): Json<DataConfigRequest>,
) -> Result<Json<DataConfigResponse>, AuthError> {
    let dataset = state.db.doc_get("axonml_datasets", &dataset_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or_else(|| AuthError::NotFound("Dataset not found".to_string()))?;

    let dataset: Dataset = serde_json::from_value(dataset)
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    if dataset.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    let path = PathBuf::from(&dataset.file_path);
    let data_type = detect_data_type(&path);
    let analysis = analyze_dataset_path(&path, &dataset.name, data_type, 1000)?;

    let format = request.format.unwrap_or_else(|| "toml".to_string());
    let config = generate_config_string(&analysis, &format)?;

    Ok(Json(DataConfigResponse { config, format }))
}

// ============================================================================
// Analysis Functions
// ============================================================================

fn parse_data_type(s: &str) -> DatasetType {
    match s.to_lowercase().as_str() {
        "image" => DatasetType::Image,
        "tabular" => DatasetType::Tabular,
        "text" => DatasetType::Text,
        "audio" => DatasetType::Audio,
        "mixed" => DatasetType::Mixed,
        _ => DatasetType::Unknown,
    }
}

fn detect_data_type(path: &PathBuf) -> DatasetType {
    let mut file_types: HashMap<String, usize> = HashMap::new();

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                *file_types.entry(ext).or_insert(0) += 1;
            }
        }
    }

    // Check for image extensions
    let image_exts = ["jpg", "jpeg", "png", "bmp", "gif", "webp"];
    let tabular_exts = ["csv", "tsv", "parquet", "json", "jsonl"];
    let text_exts = ["txt", "md", "xml"];
    let audio_exts = ["wav", "mp3", "flac", "ogg"];

    let image_count: usize = image_exts.iter()
        .filter_map(|e| file_types.get(*e))
        .sum();
    let tabular_count: usize = tabular_exts.iter()
        .filter_map(|e| file_types.get(*e))
        .sum();
    let text_count: usize = text_exts.iter()
        .filter_map(|e| file_types.get(*e))
        .sum();
    let audio_count: usize = audio_exts.iter()
        .filter_map(|e| file_types.get(*e))
        .sum();

    let max_count = [image_count, tabular_count, text_count, audio_count]
        .into_iter()
        .max()
        .unwrap_or(0);

    if max_count == 0 {
        return DatasetType::Unknown;
    }

    if image_count == max_count {
        DatasetType::Image
    } else if tabular_count == max_count {
        DatasetType::Tabular
    } else if text_count == max_count {
        DatasetType::Text
    } else if audio_count == max_count {
        DatasetType::Audio
    } else {
        DatasetType::Mixed
    }
}

fn analyze_dataset_path(
    path: &PathBuf,
    name: &str,
    data_type: DatasetType,
    max_samples: usize,
) -> Result<DatasetAnalysis, AuthError> {
    let mut file_types: HashMap<String, usize> = HashMap::new();
    let mut total_size: u64 = 0;
    let mut num_files: usize = 0;
    let mut class_distribution: HashMap<String, usize> = HashMap::new();

    // Walk directory
    fn walk_dir(
        path: &PathBuf,
        file_types: &mut HashMap<String, usize>,
        total_size: &mut u64,
        num_files: &mut usize,
        class_distribution: &mut HashMap<String, usize>,
        depth: usize,
    ) {
        if depth > 3 {
            return;
        }

        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    let class_name = entry_path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();

                    if !class_name.starts_with('.') {
                        let count = fs::read_dir(&entry_path)
                            .map(|e| e.count())
                            .unwrap_or(0);
                        if count > 0 {
                            class_distribution.insert(class_name, count);
                        }
                    }

                    walk_dir(&entry_path, file_types, total_size, num_files, class_distribution, depth + 1);
                } else {
                    *num_files += 1;
                    if let Ok(meta) = entry.metadata() {
                        *total_size += meta.len();
                    }
                    if let Some(ext) = entry_path.extension() {
                        let ext = ext.to_string_lossy().to_lowercase();
                        *file_types.entry(ext).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    walk_dir(path, &mut file_types, &mut total_size, &mut num_files, &mut class_distribution, 0);

    let num_samples = num_files.min(max_samples);
    let num_classes = if class_distribution.is_empty() {
        None
    } else {
        Some(class_distribution.len())
    };

    // Infer task type
    let task_type = infer_task_type(&data_type, num_classes);

    // Generate recommendations
    let recommendations = generate_recommendations(&data_type, num_samples, num_classes);

    // Detect input shape based on data type
    let input_shape = match data_type {
        DatasetType::Image => Some(vec![3, 224, 224]),
        DatasetType::Tabular => None, // Would need to read CSV headers
        DatasetType::Audio => Some(vec![1, 16000]),
        _ => None,
    };

    let statistics = DataStatistics {
        total_size_bytes: total_size,
        num_files,
        file_types,
        missing_values: 0,
        duplicate_samples: 0,
        mean_values: None,
        std_values: None,
        min_values: None,
        max_values: None,
    };

    Ok(DatasetAnalysis {
        name: name.to_string(),
        path: path.to_string_lossy().to_string(),
        data_type,
        task_type,
        num_samples,
        num_classes,
        class_distribution: if class_distribution.is_empty() { None } else { Some(class_distribution) },
        input_shape,
        feature_names: None,
        statistics,
        recommendations,
    })
}

fn infer_task_type(data_type: &DatasetType, num_classes: Option<usize>) -> String {
    match (data_type, num_classes) {
        (DatasetType::Image, Some(_)) => "classification".to_string(),
        (DatasetType::Image, None) => "classification".to_string(),
        (DatasetType::Tabular, Some(n)) if n <= 20 => "classification".to_string(),
        (DatasetType::Tabular, _) => "regression".to_string(),
        (DatasetType::Text, Some(_)) => "classification".to_string(),
        (DatasetType::Text, None) => "language_modeling".to_string(),
        (DatasetType::Audio, Some(_)) => "classification".to_string(),
        (DatasetType::Audio, None) => "transcription".to_string(),
        _ => "unknown".to_string(),
    }
}

fn generate_recommendations(
    data_type: &DatasetType,
    num_samples: usize,
    num_classes: Option<usize>,
) -> TrainingRecommendations {
    let architecture = match data_type {
        DatasetType::Image => "CNN".to_string(),
        DatasetType::Tabular => "MLP".to_string(),
        DatasetType::Text => "Transformer".to_string(),
        DatasetType::Audio => "CNN".to_string(),
        _ => "MLP".to_string(),
    };

    let batch_size = if num_samples < 1000 {
        16
    } else if num_samples < 10000 {
        32
    } else {
        64
    };

    let learning_rate = match data_type {
        DatasetType::Text => 0.0001,
        _ => 0.001,
    };

    let epochs = if num_samples < 1000 {
        50
    } else if num_samples < 10000 {
        20
    } else {
        10
    };

    let loss_function = match num_classes {
        Some(2) => "binary_cross_entropy".to_string(),
        Some(_) => "cross_entropy".to_string(),
        None => "mse".to_string(),
    };

    let transforms = match data_type {
        DatasetType::Image => vec![
            "resize(224, 224)".to_string(),
            "to_tensor".to_string(),
            "normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])".to_string(),
        ],
        DatasetType::Tabular => vec!["normalize".to_string()],
        DatasetType::Audio => vec![
            "resample(16000)".to_string(),
            "mel_spectrogram".to_string(),
            "normalize".to_string(),
        ],
        _ => vec![],
    };

    let augmentations = match data_type {
        DatasetType::Image => vec![
            "random_horizontal_flip".to_string(),
            "random_rotation(10)".to_string(),
            "color_jitter(0.1, 0.1, 0.1)".to_string(),
        ],
        DatasetType::Audio => vec![
            "add_noise(0.01)".to_string(),
            "time_shift".to_string(),
        ],
        _ => vec![],
    };

    let notes = vec![
        format!("Recommended for {} samples", num_samples),
        format!("Architecture chosen based on {} data type", format!("{:?}", data_type).to_lowercase()),
    ];

    TrainingRecommendations {
        architecture,
        batch_size,
        learning_rate,
        epochs,
        optimizer: "adam".to_string(),
        loss_function,
        transforms,
        augmentations,
        notes,
    }
}

fn preview_samples(
    path: &PathBuf,
    data_type: &DatasetType,
    num_samples: usize,
) -> Result<Vec<SamplePreview>, AuthError> {
    let mut samples = Vec::new();

    match data_type {
        DatasetType::Tabular => {
            // Try to find CSV files
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();
                    if entry_path.extension().map(|e| e == "csv").unwrap_or(false) {
                        if let Ok(content) = fs::read_to_string(&entry_path) {
                            for (i, line) in content.lines().take(num_samples + 1).enumerate() {
                                samples.push(SamplePreview {
                                    index: i,
                                    filename: Some(entry_path.file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_default()),
                                    label: if i == 0 { Some("header".to_string()) } else { None },
                                    preview: truncate(line, 200),
                                    size_bytes: None,
                                });
                            }
                        }
                        break;
                    }
                }
            }
        }
        DatasetType::Image | DatasetType::Audio => {
            if let Ok(entries) = fs::read_dir(path) {
                for (i, entry) in entries.flatten().take(num_samples).enumerate() {
                    let entry_path = entry.path();
                    let meta = entry.metadata().ok();

                    samples.push(SamplePreview {
                        index: i,
                        filename: Some(entry_path.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default()),
                        label: entry_path.parent()
                            .and_then(|p| p.file_name())
                            .map(|n| n.to_string_lossy().to_string()),
                        preview: format!("File: {}", entry_path.display()),
                        size_bytes: meta.map(|m| m.len()),
                    });
                }
            }
        }
        DatasetType::Text => {
            if let Ok(entries) = fs::read_dir(path) {
                for (i, entry) in entries.flatten().take(num_samples).enumerate() {
                    let entry_path = entry.path();
                    let preview = fs::read_to_string(&entry_path)
                        .map(|c| truncate(&c, 200))
                        .unwrap_or_else(|_| "Unable to read".to_string());

                    samples.push(SamplePreview {
                        index: i,
                        filename: Some(entry_path.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default()),
                        label: None,
                        preview,
                        size_bytes: entry.metadata().ok().map(|m| m.len()),
                    });
                }
            }
        }
        _ => {}
    }

    Ok(samples)
}

fn validate_dataset_path(
    path: &PathBuf,
    query: &ValidateQuery,
) -> Result<ValidationResult, AuthError> {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let mut class_distribution: HashMap<String, usize> = HashMap::new();
    let missing_files = Vec::new();
    let corrupted_files = Vec::new();

    if !path.exists() {
        issues.push(ValidationIssue {
            severity: "error".to_string(),
            message: "Dataset path does not exist".to_string(),
            file: Some(path.to_string_lossy().to_string()),
        });
        return Ok(ValidationResult {
            is_valid: false,
            issues,
            warnings,
            class_distribution: None,
            missing_files,
            corrupted_files,
        });
    }

    // Walk directory and collect class distribution
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                let class_name = entry_path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();

                if !class_name.starts_with('.') {
                    let count = fs::read_dir(&entry_path)
                        .map(|e| e.count())
                        .unwrap_or(0);
                    class_distribution.insert(class_name, count);
                }
            }
        }
    }

    // Check expected number of classes
    if let Some(expected) = query.num_classes {
        if class_distribution.len() != expected {
            issues.push(ValidationIssue {
                severity: "error".to_string(),
                message: format!(
                    "Expected {} classes, found {}",
                    expected,
                    class_distribution.len()
                ),
                file: None,
            });
        }
    }

    // Check class balance
    if query.check_balance.unwrap_or(true) && !class_distribution.is_empty() {
        let counts: Vec<usize> = class_distribution.values().copied().collect();
        let max = counts.iter().max().unwrap_or(&0);
        let min = counts.iter().min().unwrap_or(&0);

        if *max > 0 && (*min as f64 / *max as f64) < 0.5 {
            warnings.push(format!(
                "Class imbalance detected: smallest class has {} samples, largest has {}",
                min, max
            ));
        }
    }

    let is_valid = issues.iter().all(|i| i.severity != "error");

    Ok(ValidationResult {
        is_valid,
        issues,
        warnings,
        class_distribution: if class_distribution.is_empty() { None } else { Some(class_distribution) },
        missing_files,
        corrupted_files,
    })
}

fn generate_config_string(
    analysis: &DatasetAnalysis,
    format: &str,
) -> Result<String, AuthError> {
    let config = serde_json::json!({
        "dataset": {
            "name": analysis.name,
            "path": analysis.path,
            "type": format!("{:?}", analysis.data_type).to_lowercase(),
            "task": analysis.task_type,
            "num_samples": analysis.num_samples,
            "num_classes": analysis.num_classes,
            "input_shape": analysis.input_shape,
        },
        "training": {
            "architecture": analysis.recommendations.architecture,
            "batch_size": analysis.recommendations.batch_size,
            "learning_rate": analysis.recommendations.learning_rate,
            "epochs": analysis.recommendations.epochs,
            "optimizer": analysis.recommendations.optimizer,
            "loss_function": analysis.recommendations.loss_function,
        },
        "transforms": analysis.recommendations.transforms,
        "augmentations": analysis.recommendations.augmentations,
    });

    match format {
        "json" => serde_json::to_string_pretty(&config)
            .map_err(|e| AuthError::Internal(e.to_string())),
        _ => {
            // Convert to TOML
            let toml_value: toml::Value = serde_json::from_value(config)
                .map_err(|e| AuthError::Internal(e.to_string()))?;
            toml::to_string_pretty(&toml_value)
                .map_err(|e| AuthError::Internal(e.to_string()))
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
