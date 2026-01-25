//! API routes module for AxonML Server
//!
//! Defines all REST API endpoints.

pub mod auth;
pub mod training;
pub mod models;
pub mod datasets;
pub mod inference;
pub mod metrics;
pub mod system;
pub mod hub;
pub mod tools;
pub mod data;
pub mod kaggle;
pub mod builtin_datasets;

use crate::auth::{JwtAuth, AuthLayer, auth_middleware, require_admin_middleware, require_mfa_middleware, optional_auth_middleware};
use crate::config::Config;
use crate::db::Database;
use crate::inference::server::InferenceServer;
use crate::inference::pool::ModelPool;
use crate::inference::metrics::InferenceMetrics;
use crate::training::tracker::TrainingTracker;
use crate::training::executor::TrainingExecutor;
use axum::{
    extract::State,
    http::StatusCode,
    middleware,
    routing::{delete, get, post, put},
    Json, Router,
};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Database>,
    pub jwt: Arc<JwtAuth>,
    pub config: Arc<Config>,
    pub email: Arc<crate::email::EmailService>,
    pub inference: Arc<InferenceServer>,
    pub tracker: Arc<TrainingTracker>,
    pub executor: Arc<TrainingExecutor>,
    pub model_pool: Arc<ModelPool>,
    pub inference_metrics: Arc<InferenceMetrics>,
    pub metrics_history: Arc<Mutex<system::SystemMetricsHistory>>,
}

/// Create the main API router
pub fn create_router(state: AppState) -> Router {
    // CORS configuration - Only allow requests from trusted origins
    let cors = CorsLayer::new()
        .allow_origin([
            "http://127.0.0.1:8081".parse::<axum::http::HeaderValue>().unwrap(),
            "http://localhost:8081".parse::<axum::http::HeaderValue>().unwrap(),
            "http://127.0.0.1:3021".parse::<axum::http::HeaderValue>().unwrap(),
            "http://localhost:3021".parse::<axum::http::HeaderValue>().unwrap(),
        ])
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::PUT,
            axum::http::Method::DELETE,
            axum::http::Method::PATCH,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([
            axum::http::header::AUTHORIZATION,
            axum::http::header::CONTENT_TYPE,
            axum::http::header::ACCEPT,
        ])
        .allow_credentials(true);

    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/health", get(health_check))
        .route("/api/status/inference", get(inference_status))
        .route("/api/status/cache", get(cache_status))
        .route("/api/status/pool", get(pool_status))
        .route("/api/auth/register", post(auth::register))
        .route("/api/auth/login", post(auth::login))
        .route("/api/auth/verify-email", get(auth::verify_email))
        .route("/api/auth/approve-user", get(auth::approve_user))
        .route("/api/auth/mfa/totp/verify", post(auth::verify_totp))
        .route("/api/auth/mfa/webauthn/authenticate/start", post(auth::webauthn_auth_start))
        .route("/api/auth/mfa/webauthn/authenticate/finish", post(auth::webauthn_auth_finish))
        .route("/api/auth/mfa/recovery", post(auth::use_recovery_code));

    // Protected routes (auth required)
    let protected_routes = Router::new()
        // Auth
        .route("/api/auth/logout", post(auth::logout))
        .route("/api/auth/refresh", post(auth::refresh))
        .route("/api/auth/me", get(auth::me))
        .route("/api/auth/mfa/totp/setup", post(auth::setup_totp))
        .route("/api/auth/mfa/totp/enable", post(auth::enable_totp))
        .route("/api/auth/mfa/webauthn/register/start", post(auth::webauthn_register_start))
        .route("/api/auth/mfa/webauthn/register/finish", post(auth::webauthn_register_finish))
        .route("/api/auth/mfa/recovery/generate", get(auth::generate_recovery_codes))
        .route("/api/auth/mfa/disable", post(auth::disable_mfa))
        // Training
        .route("/api/training/runs", get(training::list_runs))
        .route("/api/training/runs", post(training::create_run))
        .route("/api/training/runs/:id", get(training::get_run))
        .route("/api/training/runs/:id", delete(training::delete_run))
        .route("/api/training/runs/:id/stop", post(training::stop_run))
        .route("/api/training/runs/:id/complete", post(training::complete_run))
        .route("/api/training/runs/:id/metrics", get(training::get_metrics))
        .route("/api/training/runs/:id/metrics", post(training::record_metrics))
        .route("/api/training/runs/:id/logs", get(training::get_logs))
        .route("/api/training/runs/:id/logs", post(training::append_log))
        // Models
        .route("/api/models", get(models::list_models))
        .route("/api/models", post(models::create_model))
        .route("/api/models/:id", get(models::get_model))
        .route("/api/models/:id", put(models::update_model))
        .route("/api/models/:id", delete(models::delete_model))
        .route("/api/models/:id/versions", get(models::list_versions))
        .route("/api/models/:id/versions", post(models::upload_version))
        .route("/api/models/:id/versions/:version", get(models::get_version))
        .route("/api/models/:id/versions/:version", delete(models::delete_version))
        .route("/api/models/:id/versions/:version/download", get(models::download_version))
        .route("/api/models/:id/versions/:version/deploy", post(models::deploy_version))
        // Datasets
        .route("/api/datasets", get(datasets::list_datasets))
        .route("/api/datasets", post(datasets::upload_dataset))
        .route("/api/datasets/:id", get(datasets::get_dataset))
        .route("/api/datasets/:id", delete(datasets::delete_dataset))
        // Inference
        .route("/api/inference/endpoints", get(inference::list_endpoints))
        .route("/api/inference/endpoints", post(inference::create_endpoint))
        .route("/api/inference/endpoints/:id", get(inference::get_endpoint))
        .route("/api/inference/endpoints/:id", put(inference::update_endpoint))
        .route("/api/inference/endpoints/:id", delete(inference::delete_endpoint))
        .route("/api/inference/endpoints/:id/start", post(inference::start_endpoint))
        .route("/api/inference/endpoints/:id/stop", post(inference::stop_endpoint))
        .route("/api/inference/endpoints/:id/metrics", get(inference::get_endpoint_metrics))
        .route("/api/inference/endpoints/:id/info", get(get_inference_info))
        .route("/api/inference/predict/:name", post(inference::predict))
        // Metrics
        .route("/api/metrics", get(metrics::get_metrics))
        // System
        .route("/api/system/info", get(system::get_system_info))
        .route("/api/system/gpus", get(system::list_gpus))
        .route("/api/system/benchmark", post(system::run_benchmark))
        .route("/api/system/metrics", get(system::get_realtime_metrics))
        .route("/api/system/metrics/history", get(system::get_metrics_history))
        .route("/api/system/correlation", get(system::get_correlation_data))
        // Hub (Pretrained Models)
        .route("/api/hub/models", get(hub::list_models))
        .route("/api/hub/models/:name", get(hub::get_model_info))
        .route("/api/hub/models/:name/download", post(hub::download_model))
        .route("/api/hub/cache", get(hub::get_cache_info))
        .route("/api/hub/cache", delete(hub::clear_cache))
        .route("/api/hub/cache/:name", delete(hub::clear_cache))
        // Model Tools
        .route("/api/models/:model_id/versions/:version_id/inspect", get(tools::inspect_model))
        .route("/api/models/:model_id/versions/:version_id/convert", post(tools::convert_model))
        .route("/api/models/:model_id/versions/:version_id/quantize", post(tools::quantize_model))
        .route("/api/models/:model_id/versions/:version_id/export", post(tools::export_model))
        .route("/api/tools/quantization-types", get(tools::list_quantization_types))
        // Data Analysis
        .route("/api/data/:id/analyze", post(data::analyze_dataset))
        .route("/api/data/:id/preview", post(data::preview_dataset))
        .route("/api/data/:id/validate", post(data::validate_dataset))
        .route("/api/data/:id/generate-config", post(data::generate_config))
        // Kaggle Integration
        .route("/api/kaggle/credentials", post(kaggle::save_credentials))
        .route("/api/kaggle/credentials", delete(kaggle::delete_credentials))
        .route("/api/kaggle/status", get(kaggle::get_status))
        .route("/api/kaggle/search", get(kaggle::search_datasets))
        .route("/api/kaggle/download", post(kaggle::download_dataset))
        .route("/api/kaggle/downloaded", get(kaggle::list_downloaded))
        // Built-in Datasets
        .route("/api/builtin-datasets", get(builtin_datasets::list_datasets))
        .route("/api/builtin-datasets/search", get(builtin_datasets::search_datasets))
        .route("/api/builtin-datasets/sources", get(builtin_datasets::list_sources))
        .route("/api/builtin-datasets/:id", get(builtin_datasets::get_dataset_info))
        .route("/api/builtin-datasets/:id/prepare", post(builtin_datasets::prepare_dataset))
        .layer(middleware::from_fn_with_state(
            state.jwt.clone(),
            auth_middleware,
        ));

    // Admin routes (admin role required)
    let admin_routes = Router::new()
        .route("/api/admin/users", get(auth::list_users))
        .route("/api/admin/users", post(auth::create_user))
        .route("/api/admin/users/:id", get(auth::get_user))
        .route("/api/admin/users/:id", put(auth::update_user))
        .route("/api/admin/users/:id", delete(auth::delete_user))
        .route("/api/admin/stats", get(metrics::get_stats))
        .route("/api/admin/query", post(admin_query))
        .route("/api/admin/execute", post(admin_execute))
        .route("/api/admin/metrics/record", post(admin_record_metrics))
        .layer(middleware::from_fn_with_state(
            state.jwt.clone(),
            require_admin_middleware,
        ));

    // Sensitive routes (MFA required when user has MFA enabled)
    let mfa_protected_routes = Router::new()
        .route("/api/inference/endpoints/:id/delete-secure", delete(inference::delete_endpoint))
        .layer(middleware::from_fn_with_state(
            state.jwt.clone(),
            require_mfa_middleware,
        ));

    // Optional auth routes (work with or without authentication)
    let optional_auth_routes = Router::new()
        .route("/api/public/models", get(models::list_models))
        .layer(middleware::from_fn_with_state(
            state.jwt.clone(),
            optional_auth_middleware,
        ));

    // WebSocket routes (handled separately due to upgrade)
    let ws_routes = Router::new()
        .route("/api/training/runs/:id/stream", get(training::stream_metrics));

    // Create auth layer for tower-based auth on specific routes
    // This demonstrates using AuthLayer as a tower Layer trait implementation
    let auth_layer = AuthLayer::new(state.jwt.clone());

    // Log info about the auth layer configuration
    let jwt_secret_len = auth_layer.jwt().secret().len();
    tracing::debug!(jwt_secret_len = jwt_secret_len, "AuthLayer configured");

    // Routes protected via AuthLayer's tower Layer implementation
    let tower_auth_routes = Router::new()
        .route("/api/secure/info", get(secure_info))
        .layer(auth_layer);

    // Combine all routes
    Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .merge(admin_routes)
        .merge(mfa_protected_routes)
        .merge(optional_auth_routes)
        .merge(ws_routes)
        .merge(tower_auth_routes)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Secure info endpoint (protected via AuthLayer's tower Layer implementation)
async fn secure_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    // This endpoint demonstrates the AuthLayer being used via tower Layer trait
    // It returns aggregated system information

    let models_loaded = state.inference.loaded_count().await;
    let pool_stats = state.model_pool.stats().await;
    let pool_entries = state.model_pool.list_entries().await;
    let inference_summary = state.inference_metrics.summary().await;

    // Get JWT configuration info using the jwt field and methods
    let jwt_configured = !state.jwt.secret().is_empty();
    let token_expiry = state.jwt.access_expiry_hours();
    let refresh_expiry = state.jwt.refresh_expiry_days();

    // Get pool idle timeout
    let idle_timeout = state.model_pool.idle_timeout_secs();

    // Get full inference config
    let inference_config = state.inference.config();

    Json(serde_json::json!({
        "system": {
            "jwt_configured": jwt_configured,
            "token_expiry_hours": token_expiry,
            "refresh_expiry_days": refresh_expiry,
            "inference_port": inference_config.port,
            "inference_batch_size": inference_config.batch_size,
            "inference_timeout_ms": inference_config.timeout_ms,
            "inference_max_queue_size": inference_config.max_queue_size,
            "pool_idle_timeout_secs": idle_timeout,
        },
        "inference": {
            "models_loaded": models_loaded,
            "pool_entries": pool_stats.total_entries,
            "pool_load": pool_stats.total_load,
            "pool_capacity": pool_stats.total_capacity,
            "pool_utilization": pool_stats.utilization,
        },
        "pool_details": pool_entries.iter().map(|e| serde_json::json!({
            "endpoint_id": e.endpoint_id,
            "model_id": e.model_id,
            "version_id": e.version_id,
            "replicas": e.replicas,
            "current_load": e.current_load,
            "idle_time_secs": e.idle_time_secs,
        })).collect::<Vec<_>>(),
        "metrics": {
            "endpoints_tracked": inference_summary.endpoints_count,
            "total_requests": inference_summary.total_requests,
            "total_success": inference_summary.total_success,
            "total_errors": inference_summary.total_errors,
            "avg_latency_ms": inference_summary.avg_latency,
        },
    }))
}

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    database: bool,
    inference: bool,
    models_loaded: usize,
    pool_size: usize,
    pool_utilization: f64,
    last_check: String,
}

/// Health check endpoint
async fn health_check(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let db_healthy = state.db.health_check().await;
    let models_loaded = state.inference.loaded_count().await;
    let inference_healthy = true; // Server is running if we're handling this request

    // Get pool stats
    let pool_stats = state.model_pool.stats().await;

    // Store health check timestamp in KV store for monitoring
    let check_time = chrono::Utc::now();
    let _ = state.db.kv_set("health:last_check", serde_json::json!({
        "timestamp": check_time.to_rfc3339(),
        "db_healthy": db_healthy,
        "inference_healthy": inference_healthy,
    })).await;

    let all_healthy = db_healthy;
    let status_code = if all_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status_code, Json(HealthResponse {
        status: if all_healthy { "healthy".to_string() } else { "unhealthy".to_string() },
        database: db_healthy,
        inference: inference_healthy,
        models_loaded,
        pool_size: pool_stats.total_entries,
        pool_utilization: pool_stats.utilization,
        last_check: check_time.to_rfc3339(),
    }))
}

/// Inference server info response
#[derive(Serialize)]
struct InferenceInfoResponse {
    port: u16,
    batch_size: u32,
    timeout_ms: u64,
    max_queue_size: u32,
    models_loaded: usize,
    pool_size: usize,
    pool_utilization: f64,
    total_requests: u64,
    total_errors: u64,
    avg_latency: f64,
}

/// Get inference server status endpoint
async fn inference_status(State(state): State<AppState>) -> Json<InferenceInfoResponse> {
    let models_loaded = state.inference.loaded_count().await;
    let port = state.inference.port();
    let batch_size = state.inference.batch_size();
    let timeout_ms = state.inference.timeout_ms();
    let max_queue_size = state.inference.max_queue_size();

    // Get pool statistics
    let pool_stats = state.model_pool.stats().await;
    let pool_size = state.model_pool.size().await;

    // Get metrics summary
    let metrics_summary = state.inference_metrics.summary().await;

    Json(InferenceInfoResponse {
        port,
        batch_size,
        timeout_ms,
        max_queue_size,
        models_loaded,
        pool_size,
        pool_utilization: pool_stats.utilization,
        total_requests: metrics_summary.total_requests,
        total_errors: metrics_summary.total_errors,
        avg_latency: metrics_summary.avg_latency,
    })
}

/// Get inference endpoint detailed info
async fn get_inference_info(
    State(state): State<AppState>,
    axum::extract::Path(endpoint_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Use the InferenceServer's get_model_info, is_loaded, and has_weights methods
    if let Some(info) = state.inference.get_model_info(&endpoint_id).await {
        // Get additional info from various sources
        let is_loaded = state.inference.is_loaded(&endpoint_id).await;
        let has_weights = state.inference.has_weights(&endpoint_id).await;
        let pool_load = state.model_pool.get_load(&endpoint_id).await;
        let pool_entry = state.model_pool.get_entry(&endpoint_id).await;
        let has_capacity = state.model_pool.has_capacity(&endpoint_id).await;
        let endpoint_metrics = state.inference_metrics.get(&endpoint_id).await;

        // Use all ModelInfo fields and additional server methods
        Ok(Json(serde_json::json!({
            "endpoint_id": endpoint_id,
            "model_id": info.model_id,
            "version_id": info.version_id,
            "version": info.version,
            "file_path": info.file_path,
            "loaded": is_loaded,
            "has_weights": has_weights,
            "model_info_loaded": info.loaded,
            "model_info_has_weights": info.has_weights,
            "pool_load": pool_load,
            "has_capacity": has_capacity,
            "pool_entry": pool_entry.map(|e| serde_json::json!({
                "endpoint_id": e.endpoint_id,
                "model_id": e.model_id,
                "version_id": e.version_id,
                "replicas": e.replicas,
                "current_load": e.current_load,
                "idle_time_secs": e.idle_time_secs,
            })),
            "metrics": endpoint_metrics.map(|m| serde_json::json!({
                "endpoint_id": m.id(),
                "requests_total": m.requests_total,
                "requests_success": m.requests_success,
                "requests_error": m.requests_error,
                "p50_latency": m.p50(),
                "p95_latency": m.p95(),
                "p99_latency": m.p99(),
                "avg_latency": m.avg_latency(),
                "rps": m.rps(),
                "error_rate": m.error_rate(),
                "uptime_secs": m.uptime().as_secs(),
                "latency_histogram": m.latency_histogram().iter().map(|b| serde_json::json!({
                    "le": b.le,
                    "count": b.count,
                })).collect::<Vec<_>>(),
            })),
            "architecture": info.architecture.map(|a| serde_json::json!({
                "input_size": a.input_size,
                "output_size": a.output_size,
                "layer_count": a.layers.len(),
            })),
        })))
    } else {
        Err((StatusCode::NOT_FOUND, "Endpoint not found".to_string()))
    }
}

/// Cache status from KV store
async fn cache_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    // Retrieve the last health check from KV store
    let last_health = state.db.kv_get("health:last_check").await.unwrap_or(None);

    Json(serde_json::json!({
        "kv_store": "connected",
        "last_health_check": last_health,
    }))
}

/// Pool status endpoint - shows model pool state
async fn pool_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    // Get pool statistics using all available methods
    let pool_stats = state.model_pool.stats().await;
    let pool_size = state.model_pool.size().await;

    // Cleanup idle entries as part of status check (maintenance)
    state.model_pool.cleanup_idle().await;

    Json(serde_json::json!({
        "pool_size": pool_size,
        "total_entries": pool_stats.total_entries,
        "total_load": pool_stats.total_load,
        "total_capacity": pool_stats.total_capacity,
        "utilization": pool_stats.utilization,
    }))
}

/// Admin query request
#[derive(serde::Deserialize)]
struct AdminQueryRequest {
    query: String,
    #[serde(default)]
    params: Vec<serde_json::Value>,
}

/// Admin query endpoint - execute raw SQL-like queries (admin only)
async fn admin_query(
    State(state): State<AppState>,
    Json(req): Json<AdminQueryRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Use the database query method
    let result = if req.params.is_empty() {
        state.db.query(&req.query).await
    } else {
        state.db.query_with_params(&req.query, req.params).await
    };

    match result {
        Ok(response) => Ok(Json(serde_json::json!({
            "rows": response.rows,
            "affected_rows": response.affected_rows,
        }))),
        Err(e) => Err((StatusCode::BAD_REQUEST, e.to_string())),
    }
}

/// Admin execute endpoint - execute SQL-like statements (admin only)
async fn admin_execute(
    State(state): State<AppState>,
    Json(req): Json<AdminQueryRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Use the database execute method
    let result = if req.params.is_empty() {
        state.db.execute(&req.query).await
    } else {
        state.db.execute_with_params(&req.query, req.params).await
    };

    match result {
        Ok(affected_rows) => Ok(Json(serde_json::json!({
            "affected_rows": affected_rows,
        }))),
        Err(e) => Err((StatusCode::BAD_REQUEST, e.to_string())),
    }
}

/// Admin record metrics request
#[derive(serde::Deserialize)]
struct AdminRecordMetricsRequest {
    endpoint_id: String,
    latency_ms: f64,
    #[serde(default)]
    success: bool,
}

/// Admin endpoint to manually record inference metrics
async fn admin_record_metrics(
    State(state): State<AppState>,
    Json(req): Json<AdminRecordMetricsRequest>,
) -> Json<serde_json::Value> {
    // Use the InferenceMetrics record_success/record_error methods directly
    if req.success {
        state.inference_metrics.record_success(&req.endpoint_id, req.latency_ms).await;
    } else {
        state.inference_metrics.record_error(&req.endpoint_id, req.latency_ms).await;
    }

    Json(serde_json::json!({
        "recorded": true,
        "endpoint_id": req.endpoint_id,
        "latency_ms": req.latency_ms,
        "success": req.success,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_clone() {
        // This test would require mocking Database and Config
        // Just verify the types are correct for now
    }
}
