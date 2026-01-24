//! Data types for AxonML Dashboard
//!
//! These types mirror the backend API models for serialization/deserialization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// Authentication Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    Admin,
    User,
    Viewer,
}

impl Default for UserRole {
    fn default() -> Self {
        Self::Viewer
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub email: String,
    pub name: String,
    pub role: UserRole,
    pub mfa_enabled: bool,
    pub totp_enabled: bool,
    pub webauthn_enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    pub requires_mfa: bool,
    pub mfa_token: Option<String>,
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub user: Option<User>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub email: String,
    pub name: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaVerifyRequest {
    pub mfa_token: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub access_token: String,
    pub refresh_token: String,
    pub user: User,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TotpSetupResponse {
    pub secret: String,
    pub qr_code: String,
    pub backup_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAuthnRegisterStart {
    pub challenge: String,
    pub rp_id: Option<String>,
    pub rp_name: Option<String>,
    pub user_id: Option<String>,
    pub user_name: Option<String>,
    pub user_display_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAuthnAuthenticateStart {
    pub challenge: String,
    pub rp_id: Option<String>,
    #[serde(default)]
    pub allowed_credentials: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryCodesResponse {
    pub codes: Vec<String>,
}

// ============================================================================
// Training Types
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Stopped,
}

impl RunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Stopped => "stopped",
        }
    }

    pub fn color_class(&self) -> &'static str {
        match self {
            Self::Pending => "status-pending",
            Self::Running => "status-running",
            Self::Completed => "status-success",
            Self::Failed => "status-error",
            Self::Stopped => "status-warning",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: u32,
    pub epochs: u32,
    pub optimizer: String,
    pub loss_function: String,
    #[serde(default)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingMetrics {
    pub epoch: u32,
    pub step: u32,
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub learning_rate: f64,
    pub gpu_utilization: Option<f64>,
    pub memory_mb: Option<f64>,
    pub timestamp: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub model_type: String,
    pub status: RunStatus,
    pub config: TrainingConfig,
    pub latest_metrics: Option<TrainingMetrics>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRunRequest {
    pub name: String,
    pub model_type: String,
    pub config: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsHistory {
    pub run_id: String,
    pub metrics: Vec<TrainingMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunLogs {
    pub run_id: String,
    pub logs: Vec<LogEntry>,
}

// ============================================================================
// Model Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub model_type: String,
    pub version_count: u32,
    pub latest_version: Option<u32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateModelRequest {
    pub name: String,
    pub description: Option<String>,
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateModelRequest {
    pub name: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub id: String,
    pub model_id: String,
    pub version: u32,
    pub file_path: String,
    pub file_size: u64,
    pub metrics: Option<serde_json::Value>,
    pub training_run_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWithVersions {
    pub model: Model,
    pub versions: Vec<ModelVersion>,
}

// ============================================================================
// Inference Types
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EndpointStatus {
    Starting,
    Running,
    Stopped,
    Error,
}

impl EndpointStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Stopped => "stopped",
            Self::Error => "error",
        }
    }

    pub fn color_class(&self) -> &'static str {
        match self {
            Self::Starting => "status-pending",
            Self::Running => "status-success",
            Self::Stopped => "status-warning",
            Self::Error => "status-error",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EndpointConfig {
    pub batch_size: u32,
    pub timeout_ms: u64,
    pub max_concurrent: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEndpoint {
    pub id: String,
    pub model_version_id: String,
    pub model_name: Option<String>,
    pub version: Option<u32>,
    pub name: String,
    pub status: EndpointStatus,
    pub port: u16,
    pub replicas: u32,
    pub config: EndpointConfig,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEndpointRequest {
    pub model_version_id: String,
    pub name: String,
    pub port: Option<u16>,
    pub replicas: Option<u32>,
    pub config: Option<EndpointConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateEndpointRequest {
    pub replicas: Option<u32>,
    pub config: Option<EndpointConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub endpoint_id: String,
    pub requests_total: u64,
    pub requests_success: u64,
    pub requests_error: u64,
    pub latency_p50: f64,
    pub latency_p95: f64,
    pub latency_p99: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetricsHistory {
    pub endpoint_id: String,
    pub metrics: Vec<InferenceMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictRequest {
    pub inputs: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictResponse {
    pub outputs: serde_json::Value,
    pub latency_ms: f64,
}

// ============================================================================
// Admin Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub total_users: u64,
    pub total_runs: u64,
    pub active_runs: u64,
    pub total_models: u64,
    pub total_endpoints: u64,
    pub active_endpoints: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateUserRequest {
    pub email: String,
    pub name: String,
    pub password: String,
    pub role: UserRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateUserRequest {
    pub name: Option<String>,
    pub role: Option<UserRole>,
    pub password: Option<String>,
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSuccess<T> {
    pub data: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total: u64,
    pub page: u32,
    pub per_page: u32,
}

// ============================================================================
// Dashboard Overview Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardOverview {
    pub active_runs: u32,
    pub completed_runs: u32,
    pub failed_runs: u32,
    pub total_models: u32,
    pub active_endpoints: u32,
    pub recent_runs: Vec<TrainingRun>,
    pub recent_metrics: Vec<TrainingMetrics>,
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    #[serde(rename = "metrics")]
    Metrics(TrainingMetrics),
    #[serde(rename = "status")]
    Status { run_id: String, status: RunStatus },
    #[serde(rename = "log")]
    Log(LogEntry),
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "pong")]
    Pong,
}
