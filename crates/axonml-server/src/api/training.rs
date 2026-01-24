//! Training API endpoints for AxonML
//!
//! Handles training runs, metrics, and real-time streaming.

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};
use crate::db::runs::{NewTrainingRun, RunConfig, RunRepository, RunStatus};
use crate::training::websocket::MetricsStreamer;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ListRunsQuery {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    100
}

#[derive(Debug, Deserialize)]
pub struct CreateRunRequest {
    pub name: String,
    pub model_type: String,
    pub config: RunConfigRequest,
}

#[derive(Debug, Deserialize)]
pub struct RunConfigRequest {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    #[serde(default)]
    pub optimizer: String,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct RunResponse {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub model_type: String,
    pub status: String,
    pub config: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_metrics: Option<MetricsResponse>,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetricsResponse {
    pub epoch: u32,
    pub step: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_util: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<f64>,
    #[serde(default)]
    pub custom: serde_json::Value,
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
pub struct RecordMetricsRequest {
    pub epoch: u32,
    pub step: u32,
    #[serde(default)]
    pub loss: Option<f64>,
    #[serde(default)]
    pub accuracy: Option<f64>,
    #[serde(default)]
    pub lr: Option<f64>,
    #[serde(default)]
    pub gpu_util: Option<f64>,
    #[serde(default)]
    pub memory_mb: Option<f64>,
    #[serde(default)]
    pub custom: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct AppendLogRequest {
    pub message: String,
    #[serde(default)]
    pub level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct LogsResponse {
    pub logs: Vec<LogEntry>,
}

// ============================================================================
// Handlers
// ============================================================================

/// List training runs
pub async fn list_runs(
    State(state): State<AppState>,
    user: AuthUser,
    Query(query): Query<ListRunsQuery>,
) -> Result<Json<Vec<RunResponse>>, AuthError> {
    let repo = RunRepository::new(&state.db);

    let status = query.status.as_ref().and_then(|s| match s.as_str() {
        "pending" => Some(RunStatus::Pending),
        "running" => Some(RunStatus::Running),
        "completed" => Some(RunStatus::Completed),
        "failed" => Some(RunStatus::Failed),
        "stopped" => Some(RunStatus::Stopped),
        _ => None,
    });

    let runs = if user.role == "admin" {
        repo.list_all(status, Some(query.limit), Some(query.offset))
            .await
    } else {
        repo.list_by_user(&user.id, status, Some(query.limit), Some(query.offset))
            .await
    }
    .map_err(|e| AuthError::Internal(e.to_string()))?;

    let response: Vec<RunResponse> = runs
        .into_iter()
        .map(|r| RunResponse {
            id: r.id,
            user_id: r.user_id,
            name: r.name,
            model_type: r.model_type,
            status: format!("{:?}", r.status).to_lowercase(),
            config: serde_json::to_value(&r.config).unwrap_or_default(),
            latest_metrics: r.latest_metrics.map(|m| MetricsResponse {
                epoch: m.epoch,
                step: m.step,
                loss: m.loss,
                accuracy: m.accuracy,
                lr: m.lr,
                gpu_util: m.gpu_util,
                memory_mb: m.memory_mb,
                custom: m.custom,
                timestamp: m.timestamp.to_rfc3339(),
            }),
            started_at: r.started_at.to_rfc3339(),
            completed_at: r.completed_at.map(|t| t.to_rfc3339()),
            created_at: r.created_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(response))
}

/// Create a new training run
pub async fn create_run(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<CreateRunRequest>,
) -> Result<(StatusCode, Json<RunResponse>), AuthError> {
    let repo = RunRepository::new(&state.db);

    let config = RunConfig {
        epochs: req.config.epochs,
        batch_size: req.config.batch_size,
        learning_rate: req.config.learning_rate,
        optimizer: if req.config.optimizer.is_empty() {
            "adam".to_string()
        } else {
            req.config.optimizer
        },
        extra: serde_json::to_value(&req.config.extra).unwrap_or_default(),
    };

    let run = repo
        .create(NewTrainingRun {
            user_id: user.id,
            name: req.name,
            model_type: req.model_type,
            config,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Create logs directory
    let logs_dir = state.config.runs_dir().join(&run.id);
    std::fs::create_dir_all(&logs_dir).ok();

    // Start tracking the run for real-time metrics broadcasting
    if let Err(e) = state.tracker.start_run(&run.id).await {
        tracing::warn!(run_id = %run.id, error = %e, "Failed to start run tracking");
    }

    Ok((
        StatusCode::CREATED,
        Json(RunResponse {
            id: run.id,
            user_id: run.user_id,
            name: run.name,
            model_type: run.model_type,
            status: format!("{:?}", run.status).to_lowercase(),
            config: serde_json::to_value(&run.config).unwrap_or_default(),
            latest_metrics: None,
            started_at: run.started_at.to_rfc3339(),
            completed_at: None,
            created_at: run.created_at.to_rfc3339(),
        }),
    ))
}

/// Get a training run by ID
pub async fn get_run(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<RunResponse>, AuthError> {
    let repo = RunRepository::new(&state.db);

    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    // Check ownership
    if run.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    Ok(Json(RunResponse {
        id: run.id,
        user_id: run.user_id,
        name: run.name,
        model_type: run.model_type,
        status: format!("{:?}", run.status).to_lowercase(),
        config: serde_json::to_value(&run.config).unwrap_or_default(),
        latest_metrics: run.latest_metrics.map(|m| MetricsResponse {
            epoch: m.epoch,
            step: m.step,
            loss: m.loss,
            accuracy: m.accuracy,
            lr: m.lr,
            gpu_util: m.gpu_util,
            memory_mb: m.memory_mb,
            custom: m.custom,
            timestamp: m.timestamp.to_rfc3339(),
        }),
        started_at: run.started_at.to_rfc3339(),
        completed_at: run.completed_at.map(|t| t.to_rfc3339()),
        created_at: run.created_at.to_rfc3339(),
    }))
}

/// Delete a training run
pub async fn delete_run(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<StatusCode, AuthError> {
    let repo = RunRepository::new(&state.db);

    // Check ownership
    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    if run.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Stop tracking if still active
    if state.tracker.is_tracking(&id).await {
        let _ = state.tracker.stop_run(&id).await;
    }

    // Delete logs directory
    let logs_dir = state.config.runs_dir().join(&id);
    std::fs::remove_dir_all(&logs_dir).ok();

    repo.delete(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Complete a training run (mark as finished)
#[derive(Debug, Deserialize)]
pub struct CompleteRunRequest {
    #[serde(default)]
    pub success: bool,
}

/// Complete a training run
pub async fn complete_run(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
    Json(req): Json<CompleteRunRequest>,
) -> Result<Json<RunResponse>, AuthError> {
    let repo = RunRepository::new(&state.db);

    // Check ownership
    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    if run.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Complete the run via tracker (handles status update and broadcaster cleanup)
    if let Err(e) = state.tracker.complete_run(&id, req.success).await {
        tracing::warn!(run_id = %id, error = %e, "Failed to complete run tracking");
    }

    // Get updated run
    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    Ok(Json(RunResponse {
        id: run.id,
        user_id: run.user_id,
        name: run.name,
        model_type: run.model_type,
        status: format!("{:?}", run.status).to_lowercase(),
        config: serde_json::to_value(&run.config).unwrap_or_default(),
        latest_metrics: run.latest_metrics.map(|m| MetricsResponse {
            epoch: m.epoch,
            step: m.step,
            loss: m.loss,
            accuracy: m.accuracy,
            lr: m.lr,
            gpu_util: m.gpu_util,
            memory_mb: m.memory_mb,
            custom: m.custom,
            timestamp: m.timestamp.to_rfc3339(),
        }),
        started_at: run.started_at.to_rfc3339(),
        completed_at: run.completed_at.map(|t| t.to_rfc3339()),
        created_at: run.created_at.to_rfc3339(),
    }))
}

/// Stop a training run
pub async fn stop_run(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<RunResponse>, AuthError> {
    let repo = RunRepository::new(&state.db);

    // Check ownership
    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    if run.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Stop tracking the run
    if let Err(e) = state.tracker.stop_run(&id).await {
        tracing::warn!(run_id = %id, error = %e, "Failed to stop run tracking");
    }

    let run = repo
        .update_status(&id, RunStatus::Stopped)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(RunResponse {
        id: run.id,
        user_id: run.user_id,
        name: run.name,
        model_type: run.model_type,
        status: format!("{:?}", run.status).to_lowercase(),
        config: serde_json::to_value(&run.config).unwrap_or_default(),
        latest_metrics: run.latest_metrics.map(|m| MetricsResponse {
            epoch: m.epoch,
            step: m.step,
            loss: m.loss,
            accuracy: m.accuracy,
            lr: m.lr,
            gpu_util: m.gpu_util,
            memory_mb: m.memory_mb,
            custom: m.custom,
            timestamp: m.timestamp.to_rfc3339(),
        }),
        started_at: run.started_at.to_rfc3339(),
        completed_at: run.completed_at.map(|t| t.to_rfc3339()),
        created_at: run.created_at.to_rfc3339(),
    }))
}

/// Get metrics history for a run
pub async fn get_metrics(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<Vec<MetricsResponse>>, AuthError> {
    let repo = RunRepository::new(&state.db);

    // Check ownership
    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    if run.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let metrics = repo
        .get_metrics_history(&id, Some(10000))
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let response: Vec<MetricsResponse> = metrics
        .into_iter()
        .map(|m| MetricsResponse {
            epoch: m.epoch,
            step: m.step,
            loss: m.loss,
            accuracy: m.accuracy,
            lr: m.lr,
            gpu_util: m.gpu_util,
            memory_mb: m.memory_mb,
            custom: m.custom,
            timestamp: m.timestamp.to_rfc3339(),
        })
        .collect();

    Ok(Json(response))
}

/// Record metrics for a run
pub async fn record_metrics(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
    Json(req): Json<RecordMetricsRequest>,
) -> Result<StatusCode, AuthError> {
    let repo = RunRepository::new(&state.db);

    // Check ownership
    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    if run.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Use the tracker to record metrics - this handles both storage AND broadcasting
    state.tracker.record_metrics(
        &id,
        req.epoch,
        req.step,
        req.loss,
        req.accuracy,
        req.lr,
        req.gpu_util,
        req.memory_mb,
        req.custom.clone(),
    ).await.map_err(|e| AuthError::Internal(e))?;

    // Update status to running if pending
    if run.status == RunStatus::Pending {
        repo.update_status(&id, RunStatus::Running)
            .await
            .map_err(|e| AuthError::Internal(e.to_string()))?;
    }

    Ok(StatusCode::CREATED)
}

/// Get logs for a run
pub async fn get_logs(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<LogsResponse>, AuthError> {
    let repo = RunRepository::new(&state.db);

    // Check ownership
    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    if run.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Read logs file
    let logs_path = state.config.runs_dir().join(&id).join("logs.jsonl");
    let logs = if logs_path.exists() {
        std::fs::read_to_string(&logs_path)
            .map_err(|e| AuthError::Internal(e.to_string()))?
            .lines()
            .filter_map(|line| serde_json::from_str::<LogEntry>(line).ok())
            .collect()
    } else {
        vec![]
    };

    Ok(Json(LogsResponse { logs }))
}

/// Append log entry
pub async fn append_log(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
    Json(req): Json<AppendLogRequest>,
) -> Result<StatusCode, AuthError> {
    let repo = RunRepository::new(&state.db);

    // Check ownership
    let run = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Run not found".to_string()))?;

    if run.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let entry = LogEntry {
        timestamp: Utc::now().to_rfc3339(),
        level: if req.level.is_empty() {
            "INFO".to_string()
        } else {
            req.level.to_uppercase()
        },
        message: req.message,
    };

    // Append to logs file
    let logs_dir = state.config.runs_dir().join(&id);
    std::fs::create_dir_all(&logs_dir).ok();

    let logs_path = logs_dir.join("logs.jsonl");
    let line = serde_json::to_string(&entry).unwrap_or_default() + "\n";

    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&logs_path)
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    file.write_all(line.as_bytes())
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::CREATED)
}

/// WebSocket handler for streaming metrics
pub async fn stream_metrics(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_metrics_stream(socket, state, id))
}

/// Handle WebSocket connection for metrics streaming
async fn handle_metrics_stream(socket: WebSocket, state: AppState, run_id: String) {
    // Try to subscribe to the tracker's broadcast channel for this run
    if let Some(receiver) = state.tracker.subscribe(&run_id).await {
        // Use MetricsStreamer for efficient streaming
        MetricsStreamer::stream(socket, receiver).await;
    } else {
        // Fall back to polling if run is not being tracked
        handle_metrics_stream_polling(socket, state, run_id).await;
    }
}

/// Fallback polling-based metrics streaming for runs not actively tracked
async fn handle_metrics_stream_polling(socket: WebSocket, state: AppState, run_id: String) {
    let (mut sender, mut receiver) = socket.split();

    // Spawn task to poll for new metrics
    let poll_state = state.clone();
    let poll_id = run_id.clone();

    let poll_handle = tokio::spawn(async move {
        let repo = RunRepository::new(&poll_state.db);
        let mut last_step = 0u32;

        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Check if run still exists and is running
            if let Ok(Some(run)) = repo.find_by_id(&poll_id).await {
                if run.status != RunStatus::Running && run.status != RunStatus::Pending {
                    // Send status update before breaking
                    let status_json = MetricsStreamer::format_status(
                        &format!("{:?}", run.status).to_lowercase(),
                        run.completed_at.as_ref().map(|t| t.to_rfc3339()).as_deref(),
                    );
                    let _ = sender.send(Message::Text(status_json)).await;
                    break;
                }

                // Get new metrics
                if let Some(metrics) = &run.latest_metrics {
                    if metrics.step > last_step {
                        last_step = metrics.step;
                        let json = MetricsStreamer::format_metrics(metrics);
                        if sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                }
            } else {
                break;
            }
        }
    });

    // Handle incoming messages (just for ping/pong or close)
    while let Some(msg) = receiver.next().await {
        if let Ok(msg) = msg {
            if matches!(msg, Message::Close(_)) {
                break;
            }
        } else {
            break;
        }
    }

    // Cleanup
    poll_handle.abort();
}
