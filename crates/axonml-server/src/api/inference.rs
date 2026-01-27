//! Inference API endpoints for AxonML
//!
//! Handles inference endpoint management and predictions.

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};
use crate::db::models::{EndpointStatus, ModelRepository, NewEndpoint};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct CreateEndpointRequest {
    pub model_version_id: String,
    pub name: String,
    pub port: u16,
    #[serde(default = "default_replicas")]
    pub replicas: u32,
    #[serde(default)]
    pub config: Option<serde_json::Value>,
}

fn default_replicas() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
pub struct UpdateEndpointRequest {
    #[serde(default)]
    pub replicas: Option<u32>,
    #[serde(default)]
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct EndpointResponse {
    pub id: String,
    pub model_version_id: String,
    pub name: String,
    pub status: String,
    pub port: u16,
    pub replicas: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct InferenceMetricsResponse {
    pub requests_total: u64,
    pub requests_success: u64,
    pub requests_error: u64,
    pub latency_p50: f64,
    pub latency_p95: f64,
    pub latency_p99: f64,
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub inputs: serde_json::Value,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub outputs: serde_json::Value,
    pub model_version: String,
    pub latency_ms: f64,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn endpoint_to_response(endpoint: crate::db::models::Endpoint) -> EndpointResponse {
    EndpointResponse {
        id: endpoint.id,
        model_version_id: endpoint.model_version_id,
        name: endpoint.name,
        status: format!("{:?}", endpoint.status).to_lowercase(),
        port: endpoint.port,
        replicas: endpoint.replicas,
        config: endpoint.config,
        error_message: endpoint.error_message,
        created_at: endpoint.created_at.to_rfc3339(),
        updated_at: endpoint.updated_at.to_rfc3339(),
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// List all inference endpoints
pub async fn list_endpoints(
    State(state): State<AppState>,
    _user: AuthUser,
) -> Result<Json<Vec<EndpointResponse>>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    let endpoints = repo
        .list_endpoints()
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let response: Vec<EndpointResponse> = endpoints.into_iter().map(endpoint_to_response).collect();

    Ok(Json(response))
}

/// Create a new inference endpoint
pub async fn create_endpoint(
    State(state): State<AppState>,
    _user: AuthUser,
    Json(req): Json<CreateEndpointRequest>,
) -> Result<(StatusCode, Json<EndpointResponse>), AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Verify model version exists
    repo.get_version(&req.model_version_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::Internal("Model version not found".to_string()))?;

    let endpoint = repo
        .create_endpoint(NewEndpoint {
            model_version_id: req.model_version_id,
            name: req.name,
            port: req.port,
            replicas: req.replicas,
            config: req.config,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(endpoint_to_response(endpoint))))
}

/// Get an inference endpoint
pub async fn get_endpoint(
    State(state): State<AppState>,
    _user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<EndpointResponse>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    let endpoint = repo
        .get_endpoint(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Endpoint not found".to_string()))?;

    Ok(Json(endpoint_to_response(endpoint)))
}

/// Update an inference endpoint
pub async fn update_endpoint(
    State(state): State<AppState>,
    _user: AuthUser,
    Path(id): Path<String>,
    Json(req): Json<UpdateEndpointRequest>,
) -> Result<Json<EndpointResponse>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    let endpoint = repo
        .update_endpoint(&id, req.replicas, req.config)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(endpoint_to_response(endpoint)))
}

/// Delete an inference endpoint
pub async fn delete_endpoint(
    State(state): State<AppState>,
    _user: AuthUser,
    Path(id): Path<String>,
) -> Result<StatusCode, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check if endpoint exists
    let endpoint = repo
        .get_endpoint(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Endpoint not found".to_string()))?;

    // Stop endpoint first if running
    if endpoint.status == EndpointStatus::Running {
        // Unload from inference server
        let _ = state.inference.unload_model(&id).await;

        // Remove from pool
        let _ = state.model_pool.remove(&id).await;

        // Remove metrics
        state.inference_metrics.remove(&id).await;

        repo.update_endpoint_status(&id, EndpointStatus::Stopped, None)
            .await
            .ok();
    }

    repo.delete_endpoint(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Start an inference endpoint
pub async fn start_endpoint(
    State(state): State<AppState>,
    _user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<EndpointResponse>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check if endpoint exists first
    let endpoint = repo
        .get_endpoint(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Endpoint not found".to_string()))?;

    // Update status to starting
    repo.update_endpoint_status(&id, EndpointStatus::Starting, None)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Get model version to find file path
    let version = repo
        .get_version(&endpoint.model_version_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::Internal("Model version not found".to_string()))?;

    // Get model info
    let model = repo
        .find_by_id(&version.model_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::Internal("Model not found".to_string()))?;

    // Load the model into the inference server
    if let Err(e) = state
        .inference
        .load_model(
            &id,
            &model.id,
            &version.id,
            version.version,
            &version.file_path,
        )
        .await
    {
        // Update status to error if loading fails
        let endpoint = repo
            .update_endpoint_status(&id, EndpointStatus::Error, Some(e.clone()))
            .await
            .map_err(|e| AuthError::Internal(e.to_string()))?;
        return Ok(Json(endpoint_to_response(endpoint)));
    }

    // Add model to pool for capacity management
    state
        .model_pool
        .add(&id, &model.id, &version.id, endpoint.replicas)
        .await
        .map_err(|e| AuthError::Internal(e))?;

    // Initialize metrics tracking for this endpoint
    state.inference_metrics.init(&id).await;

    // Update status to running
    let endpoint = repo
        .update_endpoint_status(&id, EndpointStatus::Running, None)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(endpoint_to_response(endpoint)))
}

/// Stop an inference endpoint
pub async fn stop_endpoint(
    State(state): State<AppState>,
    _user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<EndpointResponse>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Check if endpoint exists first
    let _ = repo
        .get_endpoint(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Endpoint not found".to_string()))?;

    // Unload the model from the inference server
    let _ = state.inference.unload_model(&id).await;

    // Remove from model pool
    let _ = state.model_pool.remove(&id).await;

    // Remove metrics tracking (preserves historical data in DB)
    state.inference_metrics.remove(&id).await;

    // Update status to stopped
    let endpoint = repo
        .update_endpoint_status(&id, EndpointStatus::Stopped, None)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(endpoint_to_response(endpoint)))
}

/// Get metrics for an inference endpoint
pub async fn get_endpoint_metrics(
    State(state): State<AppState>,
    _user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<Vec<InferenceMetricsResponse>>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Verify endpoint exists
    repo.get_endpoint(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Endpoint not found".to_string()))?;

    // First, get live metrics from InferenceMetrics (if available)
    let mut response = Vec::new();

    if let Some(live_metrics) = state.inference_metrics.get(&id).await {
        response.push(InferenceMetricsResponse {
            requests_total: live_metrics.requests_total,
            requests_success: live_metrics.requests_success,
            requests_error: live_metrics.requests_error,
            latency_p50: live_metrics.p50(),
            latency_p95: live_metrics.p95(),
            latency_p99: live_metrics.p99(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        });
    }

    // Then get historical metrics from database
    let metrics = repo
        .get_inference_metrics(&id, Some(1000))
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let historical: Vec<InferenceMetricsResponse> = metrics
        .into_iter()
        .map(|m| InferenceMetricsResponse {
            requests_total: m
                .get("requests_total")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            requests_success: m
                .get("requests_success")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            requests_error: m
                .get("requests_error")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            latency_p50: m.get("latency_p50").and_then(|v| v.as_f64()).unwrap_or(0.0),
            latency_p95: m.get("latency_p95").and_then(|v| v.as_f64()).unwrap_or(0.0),
            latency_p99: m.get("latency_p99").and_then(|v| v.as_f64()).unwrap_or(0.0),
            timestamp: m
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        })
        .collect();

    response.extend(historical);

    Ok(Json(response))
}

/// Run inference prediction
pub async fn predict(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, AuthError> {
    let repo = ModelRepository::new(&state.db);

    // Find endpoint by name
    let endpoint = repo
        .get_endpoint_by_name(&name)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Endpoint not found".to_string()))?;

    // Check if endpoint is running
    if endpoint.status != EndpointStatus::Running {
        return Err(AuthError::Internal("Endpoint is not running".to_string()));
    }

    // Acquire a slot from the model pool for capacity management
    state
        .model_pool
        .acquire(&endpoint.id)
        .await
        .map_err(|e| AuthError::Internal(e))?;

    // Start timing the request using InferenceMetrics timer
    let timer = state.inference_metrics.time_request(endpoint.id.clone());

    // Get model version
    let version = repo
        .get_version(&endpoint.model_version_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::Internal("Model version not found".to_string()))?;

    // Build input JSON with optional parameters
    let mut input_json = serde_json::json!({
        "inputs": req.inputs
    });
    if let Some(params) = req.parameters {
        input_json["parameters"] = params;
    }

    // Run inference using the inference server
    let result = state.inference.predict(&endpoint.id, input_json).await;

    // Calculate latency from timer
    let latency_ms = timer.elapsed().as_secs_f64() * 1000.0;

    // Release the slot back to the pool
    let _ = state.model_pool.release(&endpoint.id).await;

    // Handle result and record metrics
    match result {
        Ok(outputs) => {
            // Record success metrics
            timer.finish_success().await;

            // Also record to database for historical tracking
            repo.record_inference_metrics(
                &endpoint.id,
                1, // requests_total
                1, // requests_success
                0, // requests_error
                latency_ms,
                latency_ms,
                latency_ms,
            )
            .await
            .ok();

            Ok(Json(PredictResponse {
                outputs,
                model_version: format!("v{}", version.version),
                latency_ms,
            }))
        }
        Err(e) => {
            // Record error metrics
            timer.finish_error().await;

            // Also record to database
            repo.record_inference_metrics(
                &endpoint.id,
                1, // requests_total
                0, // requests_success
                1, // requests_error
                latency_ms,
                latency_ms,
                latency_ms,
            )
            .await
            .ok();

            Err(AuthError::Internal(e))
        }
    }
}
