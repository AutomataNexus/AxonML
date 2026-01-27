//! Datasets API endpoints for AxonML
//!
//! Handles dataset upload, listing, and management.

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};
use crate::db::datasets::{DatasetRepository, DatasetType, NewDataset};
use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Serialize)]
pub struct DatasetResponse {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub dataset_type: String,
    pub file_path: String,
    pub file_size: u64,
    pub num_samples: Option<u64>,
    pub num_features: Option<u64>,
    pub num_classes: Option<u64>,
    pub created_at: String,
    pub updated_at: String,
}

// ============================================================================
// Handlers
// ============================================================================

/// List all datasets for the current user
pub async fn list_datasets(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<Vec<DatasetResponse>>, AuthError> {
    let repo = DatasetRepository::new(&state.db);

    let datasets = repo
        .find_by_user(&user.id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let responses: Vec<DatasetResponse> = datasets
        .into_iter()
        .map(|d| DatasetResponse {
            id: d.id,
            user_id: d.user_id,
            name: d.name,
            description: d.description,
            dataset_type: format!("{:?}", d.dataset_type).to_lowercase(),
            file_path: d.file_path,
            file_size: d.file_size,
            num_samples: d.num_samples,
            num_features: d.num_features,
            num_classes: d.num_classes,
            created_at: d.created_at.to_rfc3339(),
            updated_at: d.updated_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(responses))
}

/// Get a specific dataset
pub async fn get_dataset(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<DatasetResponse>, AuthError> {
    let repo = DatasetRepository::new(&state.db);

    let dataset = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Dataset not found".to_string()))?;

    // Check ownership
    if dataset.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    Ok(Json(DatasetResponse {
        id: dataset.id,
        user_id: dataset.user_id,
        name: dataset.name,
        description: dataset.description,
        dataset_type: format!("{:?}", dataset.dataset_type).to_lowercase(),
        file_path: dataset.file_path,
        file_size: dataset.file_size,
        num_samples: dataset.num_samples,
        num_features: dataset.num_features,
        num_classes: dataset.num_classes,
        created_at: dataset.created_at.to_rfc3339(),
        updated_at: dataset.updated_at.to_rfc3339(),
    }))
}

/// Upload a new dataset
pub async fn upload_dataset(
    State(state): State<AppState>,
    user: AuthUser,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<DatasetResponse>), AuthError> {
    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut dataset_type_str: Option<String> = None;
    let mut file_data: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;

    // Parse multipart form
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
    {
        let field_name = field.name().unwrap_or_default().to_string();

        match field_name.as_str() {
            "name" => {
                name = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| AuthError::Internal(e.to_string()))?,
                );
            }
            "description" => {
                description = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| AuthError::Internal(e.to_string()))?,
                );
            }
            "dataset_type" => {
                dataset_type_str = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| AuthError::Internal(e.to_string()))?,
                );
            }
            "file" => {
                file_name = field.file_name().map(|s| s.to_string());
                file_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| AuthError::Internal(e.to_string()))?
                        .to_vec(),
                );
            }
            _ => {}
        }
    }

    let name = name.ok_or(AuthError::Internal("Missing name field".to_string()))?;
    let file_data = file_data.ok_or(AuthError::Internal("Missing file".to_string()))?;
    let file_size = file_data.len() as u64;

    // Determine dataset type
    let dataset_type = match dataset_type_str.as_deref() {
        Some("image") => DatasetType::Image,
        Some("text") => DatasetType::Text,
        Some("audio") => DatasetType::Audio,
        Some("custom") => DatasetType::Custom,
        _ => DatasetType::Tabular,
    };

    // Create datasets directory
    let datasets_dir = state.config.data_dir().join("datasets").join(&user.id);
    fs::create_dir_all(&datasets_dir)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Generate unique filename
    let file_ext = file_name
        .as_ref()
        .and_then(|n| {
            PathBuf::from(n)
                .extension()
                .map(|e| e.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| "csv".to_string());

    let unique_name = format!(
        "{}_{}.{}",
        uuid::Uuid::new_v4(),
        name.replace(' ', "_"),
        file_ext
    );
    let file_path = datasets_dir.join(&unique_name);

    // Write file
    let mut file = fs::File::create(&file_path)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;
    file.write_all(&file_data)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Try to detect dataset properties (basic CSV parsing for tabular)
    let (num_samples, num_features) = if file_ext == "csv" {
        detect_csv_properties(&file_data)
    } else {
        (None, None)
    };

    // Create dataset record
    let repo = DatasetRepository::new(&state.db);
    let dataset = repo
        .create(NewDataset {
            user_id: user.id,
            name: name.clone(),
            description,
            dataset_type,
            file_path: file_path.to_string_lossy().to_string(),
            file_size,
            num_samples,
            num_features,
            num_classes: None,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    tracing::info!(dataset_id = %dataset.id, name = %name, "Dataset uploaded");

    Ok((
        StatusCode::CREATED,
        Json(DatasetResponse {
            id: dataset.id,
            user_id: dataset.user_id,
            name: dataset.name,
            description: dataset.description,
            dataset_type: format!("{:?}", dataset.dataset_type).to_lowercase(),
            file_path: dataset.file_path,
            file_size: dataset.file_size,
            num_samples: dataset.num_samples,
            num_features: dataset.num_features,
            num_classes: dataset.num_classes,
            created_at: dataset.created_at.to_rfc3339(),
            updated_at: dataset.updated_at.to_rfc3339(),
        }),
    ))
}

/// Delete a dataset
pub async fn delete_dataset(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<StatusCode, AuthError> {
    let repo = DatasetRepository::new(&state.db);

    let dataset = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Dataset not found".to_string()))?;

    // Check ownership
    if dataset.user_id != user.id {
        return Err(AuthError::Forbidden("Access denied".to_string()));
    }

    // Delete file
    if let Err(e) = fs::remove_file(&dataset.file_path).await {
        tracing::warn!(error = %e, "Failed to delete dataset file");
    }

    // Delete record
    repo.delete(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Detect basic CSV properties
fn detect_csv_properties(data: &[u8]) -> (Option<u64>, Option<u64>) {
    let content = String::from_utf8_lossy(data);
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return (None, None);
    }

    // Count rows (excluding header)
    let num_samples = if lines.len() > 1 {
        Some((lines.len() - 1) as u64)
    } else {
        None
    };

    // Count columns from header
    let num_features = lines.first().map(|header| header.split(',').count() as u64);

    (num_samples, num_features)
}
