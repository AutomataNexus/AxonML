//! API Client for AxonML Server Sync
//!
//! Enables CLI to sync with the webapp through the axonml-server API.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Default server URL
pub const DEFAULT_SERVER_URL: &str = "http://localhost:3021";

/// API client errors
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("Server returned error: {0}")]
    Server(String),
    #[error("Authentication required")]
    AuthRequired,
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Not connected to server")]
    NotConnected,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Login request
#[derive(Debug, Serialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

/// Login response
#[derive(Debug, Deserialize)]
pub struct LoginResponse {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub expires_in: Option<i64>,
    pub requires_mfa: bool,
    pub user: Option<User>,
}

/// User info
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct User {
    pub id: String,
    pub email: String,
    pub name: String,
    pub role: String,
}

/// Training run
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub name: String,
    pub model_name: String,
    pub status: String,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub metrics: serde_json::Value,
    pub created_at: String,
}

/// Model info
#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub version: String,
    pub status: String,
    pub file_size: Option<u64>,
    pub created_at: String,
}

/// Dataset info
#[derive(Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub dataset_type: String,
    pub size: u64,
    pub num_samples: Option<u64>,
    pub created_at: String,
}

/// Stored credentials
#[derive(Debug, Serialize, Deserialize)]
struct StoredCredentials {
    server_url: String,
    access_token: String,
    refresh_token: Option<String>,
    user: User,
}

/// API Client for syncing with AxonML server
pub struct ApiClient {
    client: reqwest::Client,
    server_url: String,
    access_token: Option<String>,
    user: Option<User>,
}

impl ApiClient {
    /// Create a new API client
    pub fn new(server_url: Option<&str>) -> Self {
        let url = server_url
            .unwrap_or(DEFAULT_SERVER_URL)
            .trim_end_matches('/');
        Self {
            client: reqwest::Client::new(),
            server_url: url.to_string(),
            access_token: None,
            user: None,
        }
    }

    /// Load saved credentials from disk
    pub fn load_credentials() -> Result<Self, ApiError> {
        let creds_path = Self::credentials_path()?;
        if !creds_path.exists() {
            return Err(ApiError::AuthRequired);
        }

        let content = std::fs::read_to_string(&creds_path)?;
        let creds: StoredCredentials = serde_json::from_str(&content)?;

        Ok(Self {
            client: reqwest::Client::new(),
            server_url: creds.server_url,
            access_token: Some(creds.access_token),
            user: Some(creds.user),
        })
    }

    /// Save credentials to disk
    fn save_credentials(&self) -> Result<(), ApiError> {
        if let (Some(token), Some(user)) = (&self.access_token, &self.user) {
            let creds = StoredCredentials {
                server_url: self.server_url.clone(),
                access_token: token.clone(),
                refresh_token: None,
                user: user.clone(),
            };

            let creds_path = Self::credentials_path()?;
            if let Some(parent) = creds_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&creds_path, serde_json::to_string_pretty(&creds)?)?;
        }
        Ok(())
    }

    /// Clear saved credentials
    pub fn logout() -> Result<(), ApiError> {
        let creds_path = Self::credentials_path()?;
        if creds_path.exists() {
            std::fs::remove_file(&creds_path)?;
        }
        Ok(())
    }

    /// Get credentials file path
    fn credentials_path() -> Result<PathBuf, ApiError> {
        let config_dir = dirs::config_dir().ok_or_else(|| {
            ApiError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not find config directory",
            ))
        })?;
        Ok(config_dir.join("axonml").join("credentials.json"))
    }

    /// Check if connected and authenticated
    pub fn is_authenticated(&self) -> bool {
        self.access_token.is_some()
    }

    /// Get current user
    pub fn current_user(&self) -> Option<&User> {
        self.user.as_ref()
    }

    /// Check if server is available
    pub async fn is_server_available(&self) -> bool {
        let url = format!("{}/health", self.server_url);
        self.client.get(&url).send().await.is_ok()
    }

    /// Login to server
    pub async fn login(&mut self, username: &str, password: &str) -> Result<User, ApiError> {
        let url = format!("{}/api/auth/login", self.server_url);
        let req = LoginRequest {
            email: username.to_string(),
            password: password.to_string(),
        };

        let response = self.client.post(&url).json(&req).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 401 {
                return Err(ApiError::InvalidCredentials);
            }
            return Err(ApiError::Server(format!("{}: {}", status, body)));
        }

        let login_resp: LoginResponse = response.json().await?;

        if login_resp.requires_mfa {
            return Err(ApiError::Server(
                "MFA required - please login via webapp".to_string(),
            ));
        }

        let token = login_resp
            .access_token
            .ok_or_else(|| ApiError::Server("No access token received".to_string()))?;
        let user = login_resp
            .user
            .ok_or_else(|| ApiError::Server("No user info received".to_string()))?;

        self.access_token = Some(token);
        self.user = Some(user.clone());
        self.save_credentials()?;

        Ok(user)
    }

    /// Get authorization header
    fn auth_header(&self) -> Result<String, ApiError> {
        self.access_token
            .as_ref()
            .map(|t| format!("Bearer {}", t))
            .ok_or(ApiError::AuthRequired)
    }

    // =========================================================================
    // Training Runs
    // =========================================================================

    /// List training runs
    pub async fn list_training_runs(&self) -> Result<Vec<TrainingRun>, ApiError> {
        let url = format!("{}/api/training/runs", self.server_url);
        let response = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header()?)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ApiError::Server(response.text().await.unwrap_or_default()));
        }

        Ok(response.json().await?)
    }

    /// Create a training run
    pub async fn create_training_run(
        &self,
        name: &str,
        model_name: &str,
        config: serde_json::Value,
    ) -> Result<TrainingRun, ApiError> {
        let url = format!("{}/api/training/runs", self.server_url);
        let body = serde_json::json!({
            "name": name,
            "model_name": model_name,
            "config": config,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", self.auth_header()?)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ApiError::Server(response.text().await.unwrap_or_default()));
        }

        Ok(response.json().await?)
    }

    /// Update training run metrics
    pub async fn update_training_metrics(
        &self,
        run_id: &str,
        epoch: u32,
        metrics: serde_json::Value,
    ) -> Result<(), ApiError> {
        let url = format!("{}/api/training/runs/{}/metrics", self.server_url, run_id);
        let body = serde_json::json!({
            "epoch": epoch,
            "metrics": metrics,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", self.auth_header()?)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ApiError::Server(response.text().await.unwrap_or_default()));
        }

        Ok(())
    }

    // =========================================================================
    // Models
    // =========================================================================

    /// List models
    pub async fn list_models(&self) -> Result<Vec<Model>, ApiError> {
        let url = format!("{}/api/models", self.server_url);
        let response = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header()?)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ApiError::Server(response.text().await.unwrap_or_default()));
        }

        Ok(response.json().await?)
    }

    /// Upload a model
    pub async fn upload_model(
        &self,
        name: &str,
        model_path: &std::path::Path,
    ) -> Result<Model, ApiError> {
        let url = format!("{}/api/models/upload", self.server_url);

        let file_content = std::fs::read(model_path)?;
        let file_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model.safetensors");

        let form = reqwest::multipart::Form::new()
            .text("name", name.to_string())
            .part(
                "file",
                reqwest::multipart::Part::bytes(file_content).file_name(file_name.to_string()),
            );

        let response = self
            .client
            .post(&url)
            .header("Authorization", self.auth_header()?)
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ApiError::Server(response.text().await.unwrap_or_default()));
        }

        Ok(response.json().await?)
    }

    // =========================================================================
    // Datasets
    // =========================================================================

    /// List datasets
    pub async fn list_datasets(&self) -> Result<Vec<Dataset>, ApiError> {
        let url = format!("{}/api/datasets", self.server_url);
        let response = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header()?)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ApiError::Server(response.text().await.unwrap_or_default()));
        }

        Ok(response.json().await?)
    }
}

/// Check if we should sync with server (server available + authenticated)
pub async fn should_sync() -> bool {
    if let Ok(client) = ApiClient::load_credentials() {
        client.is_server_available().await
    } else {
        false
    }
}
