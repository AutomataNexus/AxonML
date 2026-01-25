//! Models API endpoints for AxonML
//!
//! Handles model registry operations including CRUD and file upload.

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};
use crate::db::models::{
    Endpoint, Model, ModelRepository, ModelVersion, NewEndpoint, NewModel,
    NewModelVersion,
};
use axum::{
    body::Bytes,
    extract::{Multipart, Path, Query, State},
    http::{header, StatusCode},
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ListModelsQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    100
}

#[derive(Debug, Deserialize)]
pub struct CreateModelRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub model_type: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateModelRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelResponse {
    pub id: String,
    pub user_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub model_type: String,
    pub version_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_version: Option<u32>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct ModelVersionResponse {
    pub id: String,
    pub model_id: String,
    pub version: u32,
    pub file_path: String,
    pub file_size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_run_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct DeployRequest {
    pub name: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_replicas")]
    pub replicas: u32,
    #[serde(default)]
    pub config: Option<serde_json::Value>,
}

fn default_port() -> u16 {
    8080
}

fn default_replicas() -> u32 {
    1
}

#[derive(Debug, Serialize)]
pub struct EndpointResponse {
    pub id: String,
    pub model_version_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    pub name: String,
    pub status: String,
    pub port: u16,
    pub replicas: u32,
    pub config: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn model_to_response(model: Model, version_count: u32, latest_version: Option<u32>) -> ModelResponse {
    ModelResponse {
        id: model.id,
        user_id: model.user_id,
        name: model.name,
        description: model.description,
        model_type: model.model_type,
        version_count,
        latest_version,
        created_at: model.created_at.to_rfc3339(),
        updated_at: model.updated_at.to_rfc3339(),
    }
}

fn version_to_response(version: ModelVersion) -> ModelVersionResponse {
    ModelVersionResponse {
        id: version.id,
        model_id: version.model_id,
        version: version.version,
        file_path: version.file_path,
        file_size: version.file_size,
        metrics: version.metrics,
        training_run_id: version.training_run_id,
        created_at: version.created_at.to_rfc3339(),
    }
}

fn endpoint_to_response(endpoint: Endpoint) -> EndpointResponse {
    // Provide default config if none exists
    let config = endpoint.config.unwrap_or_else(|| serde_json::json!({
        "batch_size": 1,
        "timeout_ms": 30000,
        "max_concurrent": 10
    }));

    EndpointResponse {
        id: endpoint.id,
        model_version_id: endpoint.model_version_id,
        model_name: None, // Would need to join with model table to get this
        version: None,    // Would need to join with version table to get this
        name: endpoint.name,
        status: format!("{:?}", endpoint.status).to_lowercase(),
        port: endpoint.port,
        replicas: endpoint.replicas,
        config,
        error_message: endpoint.error_message,
        created_at: endpoint.created_at.to_rfc3339(),
        updated_at: endpoint.updated_at.to_rfc3339(),
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// List all models
pub async fn list_models(
    State(state): State<AppState>,
    user: AuthUser,
    Query(query): Query<ListModelsQuery>,
) -> Result<Json<Vec<ModelResponse>>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    let models = if user.role == "admin" {
        repo.list_all(Some(query.limit), Some(query.offset)).await
    } else {
        repo.list_by_user(&user.id, Some(query.limit), Some(query.offset))
            .await
    }
    .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Fetch version info for each model
    let mut response = Vec::with_capacity(models.len());
    for model in models {
        let versions = repo.list_versions(&model.id).await.unwrap_or_default();
        let version_count = versions.len() as u32;
        let latest_version = versions.iter().map(|v| v.version).max();
        response.push(model_to_response(model, version_count, latest_version));
    }

    Ok(Json(response))
}

/// Create a new model
pub async fn create_model(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<CreateModelRequest>,
) -> Result<(StatusCode, Json<ModelResponse>), AuthError> {
    let repo = ModelRepository::new(&state.db);

    let model = repo
        .create(NewModel {
            user_id: user.id,
            name: req.name,
            description: req.description,
            model_type: req.model_type,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Create model directory
    let model_dir = state.config.models_dir().join(&model.id);
    std::fs::create_dir_all(&model_dir).ok();

    // New model has no versions yet
    Ok((StatusCode::CREATED, Json(model_to_response(model, 0, None))))
}

/// Get a model by ID
pub async fn get_model(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<ModelResponse>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    // Check ownership
    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Get version info
    let versions = repo.list_versions(&id).await.unwrap_or_default();
    let version_count = versions.len() as u32;
    let latest_version = versions.iter().map(|v| v.version).max();

    Ok(Json(model_to_response(model, version_count, latest_version)))
}

/// Update a model
pub async fn update_model(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
    Json(req): Json<UpdateModelRequest>,
) -> Result<Json<ModelResponse>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check ownership
    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let model = repo
        .update(&id, req.name, req.description)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Get version info
    let versions = repo.list_versions(&id).await.unwrap_or_default();
    let version_count = versions.len() as u32;
    let latest_version = versions.iter().map(|v| v.version).max();

    Ok(Json(model_to_response(model, version_count, latest_version)))
}

/// Delete a model
pub async fn delete_model(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<StatusCode, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check ownership
    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Delete model directory
    let model_dir = state.config.models_dir().join(&id);
    std::fs::remove_dir_all(&model_dir).ok();

    repo.delete(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// List versions for a model
pub async fn list_versions(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<Vec<ModelVersionResponse>>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check ownership
    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let versions = repo
        .list_versions(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let response: Vec<ModelVersionResponse> = versions.into_iter().map(version_to_response).collect();

    Ok(Json(response))
}

/// Upload a new model version
pub async fn upload_version(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<ModelVersionResponse>), AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check ownership
    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Parse multipart
    let mut file_data: Option<Bytes> = None;
    let mut file_name: Option<String> = None;
    let mut metrics: Option<serde_json::Value> = None;
    let mut training_run_id: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
    {
        let name = field.name().unwrap_or_default().to_string();

        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(|s| s.to_string());
                file_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| AuthError::Internal(e.to_string()))?,
                );
            }
            "metrics" => {
                let text = field
                    .text()
                    .await
                    .map_err(|e| AuthError::Internal(e.to_string()))?;
                metrics = serde_json::from_str(&text).ok();
            }
            "training_run_id" => {
                training_run_id = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| AuthError::Internal(e.to_string()))?,
                );
            }
            _ => {}
        }
    }

    let file_data = file_data.ok_or(AuthError::Internal("No file uploaded".to_string()))?;
    let file_size = file_data.len() as u64;

    // Determine file extension
    let extension = file_name
        .as_ref()
        .and_then(|n| n.rsplit('.').next())
        .unwrap_or("bin");

    // Create version to get the version number
    let version = repo
        .create_version(NewModelVersion {
            model_id: id.clone(),
            file_path: String::new(), // Will update after saving file
            file_size,
            metrics,
            training_run_id,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Save file
    let version_dir = state
        .config
        .models_dir()
        .join(&id)
        .join(format!("v{}", version.version));
    std::fs::create_dir_all(&version_dir)
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let file_path = version_dir.join(format!("model.{}", extension));
    let mut file = tokio::fs::File::create(&file_path)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    file.write_all(&file_data)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Update version with file path
    // Note: In production, we'd update the version record with the file path
    // For now, we return the version with the intended path
    let mut response = version_to_response(version);
    response.file_path = file_path.to_string_lossy().to_string();

    Ok((StatusCode::CREATED, Json(response)))
}

/// Get a model version
pub async fn get_version(
    State(state): State<AppState>,
    user: AuthUser,
    Path((id, version)): Path<(String, u32)>,
) -> Result<Json<ModelVersionResponse>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check ownership
    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let ver = repo
        .get_version_by_number(&id, version)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::Internal("Version not found".to_string()))?;

    Ok(Json(version_to_response(ver)))
}

/// Delete a model version
pub async fn delete_version(
    State(state): State<AppState>,
    user: AuthUser,
    Path((id, version)): Path<(String, u32)>,
) -> Result<StatusCode, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check ownership
    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let ver = repo
        .get_version_by_number(&id, version)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::Internal("Version not found".to_string()))?;

    // Delete version directory
    let version_dir = state.config.models_dir().join(&id).join(format!("v{}", version));
    std::fs::remove_dir_all(&version_dir).ok();

    repo.delete_version(&ver.id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Download a model version
pub async fn download_version(
    State(state): State<AppState>,
    user: AuthUser,
    Path((id, version)): Path<(String, u32)>,
) -> Result<impl IntoResponse, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check ownership
    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Verify version exists
    let _ver = repo
        .get_version_by_number(&id, version)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::Internal("Version not found".to_string()))?;

    // Find the model file
    let version_dir = state.config.models_dir().join(&id).join(format!("v{}", version));
    let mut file_path: Option<PathBuf> = None;

    if version_dir.exists() {
        for entry in std::fs::read_dir(&version_dir)
            .map_err(|e| AuthError::Internal(e.to_string()))?
        {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_file() {
                    file_path = Some(path);
                    break;
                }
            }
        }
    }

    let file_path = file_path.ok_or(AuthError::Internal("Model file not found".to_string()))?;
    let file_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.bin")
        .to_string();

    let file_data = tokio::fs::read(&file_path)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let content_disposition = format!("attachment; filename=\"{}\"", file_name);

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/octet-stream".to_string()),
            (header::CONTENT_DISPOSITION, content_disposition),
        ],
        file_data,
    ))
}

/// Deploy a model version to inference endpoint
pub async fn deploy_version(
    State(state): State<AppState>,
    user: AuthUser,
    Path((id, version)): Path<(String, u32)>,
    Json(req): Json<DeployRequest>,
) -> Result<(StatusCode, Json<EndpointResponse>), AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check ownership
    let model = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    if model.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let ver = repo
        .get_version_by_number(&id, version)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::Internal("Version not found".to_string()))?;

    // Create endpoint
    let endpoint = repo
        .create_endpoint(NewEndpoint {
            model_version_id: ver.id,
            name: req.name,
            port: req.port,
            replicas: req.replicas,
            config: req.config,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(endpoint_to_response(endpoint))))
}
