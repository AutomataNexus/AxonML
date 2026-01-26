//! Kaggle API endpoints for AxonML
//!
//! Provides Kaggle dataset search, download, and credential management.

use axum::{
    extract::{Query, State},
    Json,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleCredentials {
    pub username: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleStatusResponse {
    pub configured: bool,
    pub username: Option<String>,
    pub config_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleDataset {
    pub ref_name: String,
    pub title: String,
    pub size: String,
    pub download_count: u64,
    pub vote_count: u64,
    pub last_updated: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleSearchResponse {
    pub datasets: Vec<KaggleDataset>,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleDownloadRequest {
    pub dataset_ref: String,
    pub output_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleDownloadResponse {
    pub dataset_ref: String,
    pub path: String,
    pub size_bytes: u64,
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleLocalDataset {
    pub filename: String,
    pub size_mb: f64,
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub limit: Option<usize>,
    pub page: Option<usize>,
}

// ============================================================================
// Handlers
// ============================================================================

/// Save Kaggle credentials
pub async fn save_credentials(
    State(_state): State<AppState>,
    user: AuthUser,
    Json(credentials): Json<KaggleCredentials>,
) -> Result<Json<KaggleStatusResponse>, AuthError> {
    // Validate non-empty credentials
    if credentials.username.trim().is_empty() || credentials.key.trim().is_empty() {
        return Err(AuthError::Forbidden("Username and API key are required".to_string()));
    }

    // Validate credentials by making a test API call
    let client = Client::new();
    let response = client
        .get("https://www.kaggle.com/api/v1/datasets/list")
        .basic_auth(&credentials.username, Some(&credentials.key))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| AuthError::Internal(format!("Failed to validate credentials: {}", e)))?;

    if !response.status().is_success() {
        return Err(AuthError::Forbidden("Invalid Kaggle credentials".to_string()));
    }

    // Save credentials to user's config directory
    let config_dir = get_kaggle_config_dir(&user.id);
    fs::create_dir_all(&config_dir)
        .map_err(|e| AuthError::Internal(format!("Failed to create config directory: {}", e)))?;

    let config_path = config_dir.join("kaggle.json");
    let credentials_json = serde_json::to_string_pretty(&credentials)
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    fs::write(&config_path, credentials_json)
        .map_err(|e| AuthError::Internal(format!("Failed to save credentials: {}", e)))?;

    // Set file permissions (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = fs::Permissions::from_mode(0o600);
        let _ = fs::set_permissions(&config_path, permissions);
    }

    Ok(Json(KaggleStatusResponse {
        configured: true,
        username: Some(credentials.username),
        config_path: config_path.to_string_lossy().to_string(),
    }))
}

/// Get Kaggle configuration status
pub async fn get_status(
    State(_state): State<AppState>,
    user: AuthUser,
) -> Result<Json<KaggleStatusResponse>, AuthError> {
    let config_path = get_kaggle_config_dir(&user.id).join("kaggle.json");

    if config_path.exists() {
        let content = fs::read_to_string(&config_path)
            .map_err(|e| AuthError::Internal(e.to_string()))?;
        let credentials: KaggleCredentials = serde_json::from_str(&content)
            .map_err(|e| AuthError::Internal(e.to_string()))?;

        Ok(Json(KaggleStatusResponse {
            configured: true,
            username: Some(credentials.username),
            config_path: config_path.to_string_lossy().to_string(),
        }))
    } else {
        Ok(Json(KaggleStatusResponse {
            configured: false,
            username: None,
            config_path: config_path.to_string_lossy().to_string(),
        }))
    }
}

/// Search Kaggle datasets
pub async fn search_datasets(
    State(_state): State<AppState>,
    user: AuthUser,
    Query(query): Query<SearchQuery>,
) -> Result<Json<KaggleSearchResponse>, AuthError> {
    let credentials = get_credentials(&user.id)?;

    let client = Client::new();
    let limit = query.limit.unwrap_or(10);
    let page = query.page.unwrap_or(1);

    let url = format!(
        "https://www.kaggle.com/api/v1/datasets/list?search={}&page={}&pageSize={}",
        urlencoding::encode(&query.query),
        page,
        limit
    );

    let response = client
        .get(&url)
        .basic_auth(&credentials.username, Some(&credentials.key))
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| AuthError::Internal(format!("Kaggle API error: {}", e)))?;

    if !response.status().is_success() {
        return Err(AuthError::Internal("Kaggle API request failed".to_string()));
    }

    let kaggle_response: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| AuthError::Internal(format!("Failed to parse Kaggle response: {}", e)))?;

    let datasets: Vec<KaggleDataset> = kaggle_response
        .into_iter()
        .map(|d| KaggleDataset {
            ref_name: d["ref"].as_str().unwrap_or("").to_string(),
            title: d["title"].as_str().unwrap_or("").to_string(),
            size: format_size(d["totalBytes"].as_u64().unwrap_or(0)),
            download_count: d["downloadCount"].as_u64().unwrap_or(0),
            vote_count: d["voteCount"].as_u64().unwrap_or(0),
            last_updated: d["lastUpdated"].as_str().unwrap_or("").to_string(),
            description: d["subtitle"].as_str().map(|s| s.to_string()),
        })
        .collect();

    let total = datasets.len();

    Ok(Json(KaggleSearchResponse { datasets, total }))
}

/// SECURITY: Validate dataset reference to prevent path traversal
fn validate_dataset_ref(dataset_ref: &str) -> Result<(), AuthError> {
    // Check for path traversal patterns
    if dataset_ref.contains("..") || dataset_ref.contains("./") || dataset_ref.starts_with('/') {
        return Err(AuthError::InvalidInput("Invalid dataset reference: path traversal detected".to_string()));
    }
    // Only allow alphanumeric, hyphens, underscores, and single forward slash
    let valid = dataset_ref.chars().all(|c| {
        c.is_alphanumeric() || c == '-' || c == '_' || c == '/'
    });
    if !valid {
        return Err(AuthError::InvalidInput("Invalid dataset reference: contains invalid characters".to_string()));
    }
    // Must contain exactly one slash (owner/dataset format)
    if dataset_ref.matches('/').count() != 1 {
        return Err(AuthError::InvalidInput("Invalid dataset reference: must be in format 'owner/dataset'".to_string()));
    }
    Ok(())
}

/// SECURITY: Validate that a path is within the allowed base directory
fn validate_path_within_base(path: &PathBuf, base: &PathBuf) -> Result<(), AuthError> {
    // Canonicalize both paths to resolve any .. or symlinks
    let canonical_base = base.canonicalize()
        .map_err(|_| AuthError::Internal("Failed to resolve base path".to_string()))?;

    // Create the directory first so we can canonicalize it
    fs::create_dir_all(path)
        .map_err(|e| AuthError::Internal(format!("Failed to create directory: {}", e)))?;

    let canonical_path = path.canonicalize()
        .map_err(|_| AuthError::Internal("Failed to resolve output path".to_string()))?;

    if !canonical_path.starts_with(&canonical_base) {
        tracing::warn!(
            path = %canonical_path.display(),
            base = %canonical_base.display(),
            "Path traversal attempt detected"
        );
        return Err(AuthError::InvalidInput("Invalid output directory: path traversal detected".to_string()));
    }
    Ok(())
}

/// Download a Kaggle dataset
pub async fn download_dataset(
    State(_state): State<AppState>,
    user: AuthUser,
    Json(request): Json<KaggleDownloadRequest>,
) -> Result<Json<KaggleDownloadResponse>, AuthError> {
    // SECURITY: Validate dataset reference
    validate_dataset_ref(&request.dataset_ref)?;

    let credentials = get_credentials(&user.id)?;

    // Determine base directory (user's data directory)
    let base_dir = get_data_dir(&user.id);

    // Determine output directory
    let output_dir = if let Some(ref custom_dir) = request.output_dir {
        // SECURITY: Custom directory must be within user's data directory
        let requested = PathBuf::from(custom_dir);
        if requested.is_absolute() {
            return Err(AuthError::InvalidInput("Absolute paths not allowed for output directory".to_string()));
        }
        base_dir.join(requested)
    } else {
        base_dir.clone()
    };

    // SECURITY: Validate the output path is within allowed base
    fs::create_dir_all(&base_dir)
        .map_err(|e| AuthError::Internal(format!("Failed to create base directory: {}", e)))?;
    validate_path_within_base(&output_dir, &base_dir)?;

    // Download dataset
    let client = Client::new();
    let url = format!(
        "https://www.kaggle.com/api/v1/datasets/download/{}",
        request.dataset_ref
    );

    let response = client
        .get(&url)
        .basic_auth(&credentials.username, Some(&credentials.key))
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
        .map_err(|e| AuthError::Internal(format!("Download failed: {}", e)))?;

    if !response.status().is_success() {
        return Err(AuthError::Internal(format!(
            "Download failed with status: {}",
            response.status()
        )));
    }

    // Save to file - SECURITY: filename is derived from validated dataset_ref
    let filename = request.dataset_ref.replace('/', "_") + ".zip";
    let output_path = output_dir.join(&filename);

    let bytes = response.bytes().await
        .map_err(|e| AuthError::Internal(format!("Failed to read response: {}", e)))?;

    let size_bytes = bytes.len() as u64;

    fs::write(&output_path, &bytes)
        .map_err(|e| AuthError::Internal(format!("Failed to save file: {}", e)))?;

    Ok(Json(KaggleDownloadResponse {
        dataset_ref: request.dataset_ref,
        path: output_path.to_string_lossy().to_string(),
        size_bytes,
        files: vec![filename],
    }))
}

/// List downloaded Kaggle datasets
pub async fn list_downloaded(
    State(_state): State<AppState>,
    user: AuthUser,
) -> Result<Json<Vec<KaggleLocalDataset>>, AuthError> {
    let data_dir = get_data_dir(&user.id);

    let mut datasets = Vec::new();

    if data_dir.exists() {
        if let Ok(entries) = fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "zip").unwrap_or(false) {
                    if let Ok(meta) = entry.metadata() {
                        let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
                        datasets.push(KaggleLocalDataset {
                            filename: entry.file_name().to_string_lossy().to_string(),
                            size_mb,
                            path: path.to_string_lossy().to_string(),
                        });
                    }
                }
            }
        }
    }

    Ok(Json(datasets))
}

/// Delete Kaggle credentials
pub async fn delete_credentials(
    State(_state): State<AppState>,
    user: AuthUser,
) -> Result<Json<KaggleStatusResponse>, AuthError> {
    let config_path = get_kaggle_config_dir(&user.id).join("kaggle.json");

    if config_path.exists() {
        fs::remove_file(&config_path)
            .map_err(|e| AuthError::Internal(format!("Failed to remove credentials: {}", e)))?;
    }

    Ok(Json(KaggleStatusResponse {
        configured: false,
        username: None,
        config_path: config_path.to_string_lossy().to_string(),
    }))
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_kaggle_config_dir(user_id: &str) -> PathBuf {
    let base = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("axonml")
        .join("users")
        .join(user_id)
        .join(".kaggle");

    base
}

fn get_data_dir(user_id: &str) -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("axonml")
        .join("users")
        .join(user_id)
        .join("data")
}

fn get_credentials(user_id: &str) -> Result<KaggleCredentials, AuthError> {
    let config_path = get_kaggle_config_dir(user_id).join("kaggle.json");

    if !config_path.exists() {
        return Err(AuthError::Forbidden(
            "Kaggle credentials not configured. Please configure your Kaggle API credentials first.".to_string()
        ));
    }

    let content = fs::read_to_string(&config_path)
        .map_err(|e| AuthError::Internal(format!("Failed to read credentials: {}", e)))?;

    serde_json::from_str(&content)
        .map_err(|e| AuthError::Internal(format!("Invalid credentials file: {}", e)))
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
