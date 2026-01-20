//! API routes module for AxonML Server
//!
//! Defines all REST API endpoints.

pub mod auth;
pub mod training;
pub mod models;
pub mod inference;
pub mod metrics;

use crate::auth::jwt::JwtAuth;
use crate::auth::middleware::{auth_middleware, require_admin_middleware};
use crate::config::Config;
use crate::db::Database;
use axum::{
    middleware,
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Database>,
    pub jwt: Arc<JwtAuth>,
    pub config: Arc<Config>,
}

/// Create the main API router
pub fn create_router(state: AppState) -> Router {
    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/health", get(health_check))
        .route("/api/auth/register", post(auth::register))
        .route("/api/auth/login", post(auth::login))
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
        // Training
        .route("/api/training/runs", get(training::list_runs))
        .route("/api/training/runs", post(training::create_run))
        .route("/api/training/runs/:id", get(training::get_run))
        .route("/api/training/runs/:id", delete(training::delete_run))
        .route("/api/training/runs/:id/stop", post(training::stop_run))
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
        // Inference
        .route("/api/inference/endpoints", get(inference::list_endpoints))
        .route("/api/inference/endpoints", post(inference::create_endpoint))
        .route("/api/inference/endpoints/:id", get(inference::get_endpoint))
        .route("/api/inference/endpoints/:id", put(inference::update_endpoint))
        .route("/api/inference/endpoints/:id", delete(inference::delete_endpoint))
        .route("/api/inference/endpoints/:id/start", post(inference::start_endpoint))
        .route("/api/inference/endpoints/:id/stop", post(inference::stop_endpoint))
        .route("/api/inference/endpoints/:id/metrics", get(inference::get_endpoint_metrics))
        .route("/api/inference/predict/:name", post(inference::predict))
        // Metrics
        .route("/api/metrics", get(metrics::get_metrics))
        .layer(middleware::from_fn_with_state(
            state.jwt.clone(),
            auth_middleware,
        ));

    // Admin routes (admin role required)
    let admin_routes = Router::new()
        .route("/api/admin/users", get(auth::list_users))
        .route("/api/admin/users", post(auth::create_user))
        .route("/api/admin/users/:id", put(auth::update_user))
        .route("/api/admin/users/:id", delete(auth::delete_user))
        .route("/api/admin/stats", get(metrics::get_stats))
        .layer(middleware::from_fn_with_state(
            state.jwt.clone(),
            require_admin_middleware,
        ));

    // WebSocket routes (handled separately due to upgrade)
    let ws_routes = Router::new()
        .route("/api/training/runs/:id/stream", get(training::stream_metrics));

    // Combine all routes
    Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .merge(admin_routes)
        .merge(ws_routes)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
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
