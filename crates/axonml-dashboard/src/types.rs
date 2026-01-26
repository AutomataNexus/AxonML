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
    #[serde(default)]
    pub mfa_verified: bool,
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
    #[serde(default)]
    pub mfa_methods: Option<Vec<String>>,
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: Option<i64>,
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
    Draft,
    Pending,
    Running,
    Completed,
    Failed,
    Stopped,
}

impl RunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Stopped => "stopped",
        }
    }

    pub fn color_class(&self) -> &'static str {
        match self {
            Self::Draft => "status-draft",
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
    #[serde(default = "default_steps_per_epoch")]
    pub steps_per_epoch: u32,
    #[serde(default)]
    pub optimizer: Option<String>,
    #[serde(default)]
    pub loss_function: Option<String>,
    #[serde(default)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

fn default_steps_per_epoch() -> u32 {
    100
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
    #[serde(default)]
    pub model_version_id: Option<String>,
    #[serde(default)]
    pub dataset_id: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
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
// Dataset Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
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
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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

// ============================================================================
// System Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub platform: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub axonml_version: String,
    pub rust_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub id: usize,
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub memory_total: u64,
    pub is_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuListResponse {
    pub gpus: Vec<GpuInfo>,
    pub cuda_available: bool,
    pub total_gpu_memory: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub gpu_id: usize,
    pub gpu_name: String,
    pub buffer_copy_1mb_ms: f64,
    pub buffer_copy_16mb_ms: f64,
    pub buffer_copy_64mb_ms: f64,
    pub compute_dispatch_ms: f64,
    pub effective_bandwidth_1mb: String,
    pub effective_bandwidth_16mb: String,
    pub effective_bandwidth_64mb: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResponse {
    pub results: Vec<BenchmarkResult>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeMetrics {
    pub timestamp: String,
    pub cpu_usage_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub memory_percent: f64,
    pub disk_used_bytes: u64,
    pub disk_total_bytes: u64,
    pub disk_percent: f64,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub process_count: usize,
    pub uptime_seconds: u64,
    pub load_avg_1m: f64,
    pub load_avg_5m: f64,
    pub load_avg_15m: f64,
    pub cpu_per_core: Vec<f64>,
    pub gpu_metrics: Vec<GpuMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub id: usize,
    pub name: String,
    pub utilization_percent: f64,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub temperature_c: f64,
    pub power_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsHistory {
    pub timestamps: Vec<String>,
    pub cpu_history: Vec<f64>,
    pub memory_history: Vec<f64>,
    pub disk_io_read: Vec<f64>,
    pub disk_io_write: Vec<f64>,
    pub network_rx: Vec<f64>,
    pub network_tx: Vec<f64>,
    pub gpu_utilization: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationData {
    pub points: Vec<CorrelationPoint>,
    pub x_label: String,
    pub y_label: String,
    pub z_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub label: String,
    pub category: String,
}

// ============================================================================
// Hub Types (Pretrained Models)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainedModel {
    pub name: String,
    pub description: String,
    pub architecture: String,
    pub size_mb: f64,
    pub accuracy: f32,
    pub dataset: String,
    pub input_size: (usize, usize),
    pub num_classes: usize,
    pub num_parameters: u64,
    pub is_cached: bool,
    pub cache_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadResponse {
    pub model_name: String,
    pub path: String,
    pub size_bytes: u64,
    pub downloaded: bool,
    pub was_cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    pub total_models: usize,
    pub total_size_bytes: u64,
    pub cache_directory: String,
    pub models: Vec<CachedModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModel {
    pub name: String,
    pub size_bytes: u64,
    pub path: String,
}

// ============================================================================
// Tools Types (Model Inspection, Conversion, Quantization, Export)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInspection {
    pub name: String,
    pub format: String,
    pub file_size: u64,
    pub num_parameters: u64,
    pub num_layers: usize,
    pub layers: Vec<LayerInfo>,
    pub metadata: std::collections::HashMap<String, String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResponse {
    pub output_file: String,
    pub format: String,
    pub size: u64,
    pub compatible_with: Vec<String>,
}

// ============================================================================
// Training Notebook Types
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CellType {
    Code,
    Markdown,
}

impl Default for CellType {
    fn default() -> Self {
        Self::Code
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CellStatus {
    Idle,
    Running,
    Completed,
    Error,
}

impl Default for CellStatus {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellOutput {
    pub output_type: String,
    pub text: Option<String>,
    pub data: Option<serde_json::Value>,
    pub execution_count: Option<u32>,
    pub error_name: Option<String>,
    pub error_value: Option<String>,
    pub traceback: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookCell {
    pub id: String,
    pub cell_type: CellType,
    pub source: String,
    #[serde(default)]
    pub outputs: Vec<CellOutput>,
    #[serde(default)]
    pub status: CellStatus,
    pub execution_count: Option<u32>,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for NotebookCell {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            cell_type: CellType::Code,
            source: String::new(),
            outputs: vec![],
            status: CellStatus::Idle,
            execution_count: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookCheckpoint {
    pub id: String,
    pub notebook_id: String,
    pub epoch: u32,
    pub step: u32,
    pub metrics: serde_json::Value,
    pub model_state_path: String,
    pub optimizer_state_path: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingNotebook {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub cells: Vec<NotebookCell>,
    #[serde(default)]
    pub metadata: NotebookMetadata,
    #[serde(default)]
    pub checkpoints: Vec<NotebookCheckpoint>,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
    pub status: RunStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotebookMetadata {
    pub kernel: Option<String>,
    pub language: Option<String>,
    pub framework: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNotebookRequest {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub cells: Vec<NotebookCell>,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateNotebookRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub cells: Option<Vec<NotebookCell>>,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteCellRequest {
    pub cell_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteCellResponse {
    pub cell_id: String,
    pub outputs: Vec<CellOutput>,
    pub execution_count: u32,
    pub status: CellStatus,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiAssistRequest {
    /// User's prompt/question
    pub prompt: String,
    /// ID of the currently selected cell (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_cell_id: Option<String>,
    /// Type of cell to generate
    #[serde(default)]
    pub cell_type: CellType,
    /// Whether to include imports
    #[serde(default)]
    pub include_imports: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiAssistResponse {
    pub suggestion: String,
    pub explanation: Option<String>,
    pub confidence: f32,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub tokens_generated: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveCheckpointRequest {
    pub epoch: u32,
    pub step: u32,
    pub metrics: serde_json::Value,
    pub model_state: Vec<u8>,
    pub optimizer_state: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadModelVersionRequest {
    pub checkpoint_id: String,
    pub model_id: String,
    pub metrics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportNotebookRequest {
    pub content: String,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportNotebookResponse {
    pub content: String,
    pub format: String,
    pub filename: String,
}

// ============================================================================
// Data Analysis Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnalysisDatasetType {
    Image,
    Tabular,
    Text,
    Audio,
    Mixed,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub missing_count: usize,
    pub missing_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecommendations {
    pub suggested_model: String,
    pub suggested_batch_size: u32,
    pub suggested_lr: f64,
    pub suggested_epochs: u32,
    pub suggested_optimizer: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAnalysis {
    pub name: String,
    pub path: String,
    pub data_type: String,
    pub task_type: String,
    pub num_samples: usize,
    pub num_classes: Option<usize>,
    pub class_distribution: Option<std::collections::HashMap<String, usize>>,
    pub input_shape: Option<Vec<usize>>,
    pub feature_names: Option<Vec<String>>,
    pub statistics: DataStatistics,
    pub recommendations: TrainingRecommendations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreviewSample {
    pub index: usize,
    pub label: Option<String>,
    pub features: Option<Vec<f64>>,
    pub text: Option<String>,
    pub image_dimensions: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreviewResponse {
    pub name: String,
    pub data_type: String,
    pub samples: Vec<DataPreviewSample>,
    pub total_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub category: String,
    pub severity: String,
    pub message: String,
    pub file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub warnings: Vec<String>,
    pub class_distribution: Option<std::collections::HashMap<String, usize>>,
    pub num_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedTrainingConfig {
    pub name: String,
    pub model_type: String,
    pub config: TrainingConfig,
    pub data_config: DataConfig,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataConfig {
    pub train_split: f64,
    pub val_split: f64,
    pub test_split: f64,
    pub shuffle: bool,
    pub augmentation: bool,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeQuery {
    pub data_type: Option<String>,
    pub max_samples: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewQuery {
    pub num_samples: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateQuery {
    pub num_classes: Option<usize>,
    pub check_balance: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateConfigRequest {
    pub format: Option<String>,
}

// ============================================================================
// Kaggle Types
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

// ============================================================================
// Built-in Dataset Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinDataset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub num_samples: u64,
    pub num_features: u64,
    pub num_classes: u64,
    pub size_mb: f64,
    pub data_type: String,
    pub task_type: String,
    pub source: String,
    pub download_url: Option<String>,
    pub loading_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSource {
    pub id: String,
    pub name: String,
    pub description: String,
    pub dataset_count: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSearchResult {
    pub id: String,
    pub name: String,
    pub source: String,
    pub size: String,
    pub download_count: u64,
    pub description: String,
}
