//! Metrics API endpoints for AxonML
//!
//! Handles Prometheus metrics export and system statistics.

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};
use crate::db::models::ModelRepository;
use crate::db::runs::RunRepository;
use crate::db::users::UserRepository;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Serialize;

// ============================================================================
// Response Types
// ============================================================================

#[derive(Debug, Serialize)]
pub struct SystemStats {
    pub users_total: u64,
    pub runs_total: u64,
    pub runs_running: u64,
    pub models_total: u64,
    pub endpoints_total: u64,
    pub endpoints_running: u64,
}

// ============================================================================
// Handlers
// ============================================================================

/// Get Prometheus-style metrics
pub async fn get_metrics(
    State(state): State<AppState>,
    _user: AuthUser,
) -> Result<impl IntoResponse, AuthError> {
    let user_repo = UserRepository::new(&state.db);
    let run_repo = RunRepository::new(&state.db);
    let model_repo = ModelRepository::new(&state.db);

    // Collect metrics
    let users_total = user_repo.count().await.unwrap_or(0);
    let runs_running = run_repo.count_running().await.unwrap_or(0);
    let endpoints_running = model_repo.count_running_endpoints().await.unwrap_or(0);

    // Build Prometheus format
    let metrics = format!(
        r#"# HELP axonml_users_total Total number of registered users
# TYPE axonml_users_total gauge
axonml_users_total {}

# HELP axonml_runs_running Number of currently running training runs
# TYPE axonml_runs_running gauge
axonml_runs_running {}

# HELP axonml_endpoints_running Number of currently running inference endpoints
# TYPE axonml_endpoints_running gauge
axonml_endpoints_running {}

# HELP axonml_server_info Server information
# TYPE axonml_server_info gauge
axonml_server_info{{version="0.1.0"}} 1
"#,
        users_total, runs_running, endpoints_running
    );

    Ok((
        StatusCode::OK,
        [("Content-Type", "text/plain; charset=utf-8")],
        metrics,
    ))
}

/// Get system statistics (admin only)
pub async fn get_stats(
    State(state): State<AppState>,
) -> Result<Json<SystemStats>, AuthError> {
    let user_repo = UserRepository::new(&state.db);
    let run_repo = RunRepository::new(&state.db);
    let model_repo = ModelRepository::new(&state.db);

    // Collect stats
    let users_total = user_repo.count().await.unwrap_or(0);

    let runs = run_repo.list_all(None, Some(10000), Some(0)).await.unwrap_or_default();
    let runs_total = runs.len() as u64;
    let runs_running = run_repo.count_running().await.unwrap_or(0);

    let models = model_repo.list_all(Some(10000), Some(0)).await.unwrap_or_default();
    let models_total = models.len() as u64;

    let endpoints = model_repo.list_endpoints().await.unwrap_or_default();
    let endpoints_total = endpoints.len() as u64;
    let endpoints_running = model_repo.count_running_endpoints().await.unwrap_or(0);

    Ok(Json(SystemStats {
        users_total,
        runs_total,
        runs_running,
        models_total,
        endpoints_total,
        endpoints_running,
    }))
}
