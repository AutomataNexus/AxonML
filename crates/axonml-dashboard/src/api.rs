//! HTTP API Client for AxonML Backend
//!
//! Provides async functions for all backend API endpoints.

use gloo_net::http::{Request, RequestBuilder};
use gloo_storage::Storage;
use serde::{de::DeserializeOwned, Serialize};
use web_sys::FormData;

use crate::types::*;

const API_BASE: &str = "/api";

/// API client error type
#[derive(Debug, Clone)]
pub struct ApiClientError {
    pub status: u16,
    pub message: String,
}

impl std::fmt::Display for ApiClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "API Error {}: {}", self.status, self.message)
    }
}

impl From<gloo_net::Error> for ApiClientError {
    fn from(err: gloo_net::Error) -> Self {
        Self {
            status: 0,
            message: err.to_string(),
        }
    }
}

pub type ApiResult<T> = Result<T, ApiClientError>;

/// Get the stored access token from localStorage
fn get_token() -> Option<String> {
    gloo_storage::LocalStorage::get("access_token").ok()
}

/// Build a request with optional authentication
fn build_request(method: &str, path: &str) -> RequestBuilder {
    let url = format!("{}{}", API_BASE, path);
    let mut builder = match method {
        "GET" => Request::get(&url),
        "POST" => Request::post(&url),
        "PUT" => Request::put(&url),
        "DELETE" => Request::delete(&url),
        "PATCH" => Request::patch(&url),
        _ => Request::get(&url),
    };

    if let Some(token) = get_token() {
        builder = builder.header("Authorization", &format!("Bearer {}", token));
    }

    builder
}

/// Execute a GET/DELETE request and parse JSON response
async fn fetch_json<T: DeserializeOwned>(request: RequestBuilder) -> ApiResult<T> {
    let response = request.send().await?;
    let status = response.status();

    if status >= 200 && status < 300 {
        response.json::<T>().await.map_err(|e| ApiClientError {
            status,
            message: format!("Failed to parse response: {}", e),
        })
    } else {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        Err(ApiClientError {
            status,
            message: error_text,
        })
    }
}

/// Execute a POST/PUT/PATCH request with body and parse JSON response
async fn fetch_json_with_body<T: DeserializeOwned, B: Serialize>(builder: RequestBuilder, body: &B) -> ApiResult<T> {
    let request = builder
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(body).unwrap())
        .map_err(|e| ApiClientError {
            status: 0,
            message: format!("Failed to build request: {}", e),
        })?;

    let response = request.send().await?;
    let status = response.status();

    if status >= 200 && status < 300 {
        response.json::<T>().await.map_err(|e| ApiClientError {
            status,
            message: format!("Failed to parse response: {}", e),
        })
    } else {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        Err(ApiClientError {
            status,
            message: error_text,
        })
    }
}

/// Execute a GET/DELETE request expecting no response body
async fn fetch_empty(request: RequestBuilder) -> ApiResult<()> {
    let response = request.send().await?;
    let status = response.status();

    if status >= 200 && status < 300 {
        Ok(())
    } else {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        Err(ApiClientError {
            status,
            message: error_text,
        })
    }
}

/// Execute a POST/PUT/PATCH request with body expecting no response body
async fn fetch_empty_with_body<B: Serialize>(builder: RequestBuilder, body: &B) -> ApiResult<()> {
    let request = builder
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(body).unwrap())
        .map_err(|e| ApiClientError {
            status: 0,
            message: format!("Failed to build request: {}", e),
        })?;

    let response = request.send().await?;
    let status = response.status();

    if status >= 200 && status < 300 {
        Ok(())
    } else {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        Err(ApiClientError {
            status,
            message: error_text,
        })
    }
}

// ============================================================================
// Authentication API
// ============================================================================

pub mod auth {
    use super::*;

    pub async fn register(request: &RegisterRequest) -> ApiResult<TokenPair> {
        fetch_json_with_body(build_request("POST", "/auth/register"), request).await
    }

    pub async fn login(request: &LoginRequest) -> ApiResult<LoginResponse> {
        fetch_json_with_body(build_request("POST", "/auth/login"), request).await
    }

    pub async fn logout() -> ApiResult<()> {
        fetch_empty(build_request("POST", "/auth/logout")).await
    }

    pub async fn refresh(refresh_token: &str) -> ApiResult<TokenPair> {
        fetch_json_with_body(
            build_request("POST", "/auth/refresh"),
            &serde_json::json!({ "refresh_token": refresh_token }),
        ).await
    }

    pub async fn me() -> ApiResult<User> {
        fetch_json(build_request("GET", "/auth/me")).await
    }

    pub async fn verify_mfa(request: &MfaVerifyRequest) -> ApiResult<TokenPair> {
        fetch_json_with_body(build_request("POST", "/auth/mfa/verify"), request).await
    }

    pub async fn totp_setup() -> ApiResult<TotpSetupResponse> {
        fetch_json_with_body(build_request("POST", "/auth/mfa/totp/setup"), &serde_json::json!({})).await
    }

    pub async fn totp_enable(code: &str) -> ApiResult<()> {
        fetch_empty_with_body(
            build_request("POST", "/auth/mfa/totp/enable"),
            &serde_json::json!({ "code": code }),
        ).await
    }

    pub async fn totp_disable() -> ApiResult<()> {
        fetch_empty(build_request("POST", "/auth/mfa/totp/disable")).await
    }

    pub async fn webauthn_register_start() -> ApiResult<WebAuthnRegisterStart> {
        fetch_json_with_body(build_request("POST", "/auth/mfa/webauthn/register/start"), &serde_json::json!({})).await
    }

    /// WebAuthn registration finish request
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct WebAuthnRegisterFinishRequest {
        pub credential_id: String,
        pub attestation_object: String,
        pub client_data_json: String,
        pub device_name: String,
    }

    pub async fn webauthn_register_finish(request: &WebAuthnRegisterFinishRequest) -> ApiResult<()> {
        fetch_empty_with_body(
            build_request("POST", "/auth/mfa/webauthn/register/finish"),
            request,
        ).await
    }

    pub async fn webauthn_authenticate_start(mfa_token: &str) -> ApiResult<WebAuthnAuthenticateStart> {
        fetch_json_with_body(
            build_request("POST", "/auth/mfa/webauthn/authenticate/start"),
            &serde_json::json!({ "mfa_token": mfa_token }),
        ).await
    }

    /// WebAuthn authentication finish request
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct WebAuthnAuthFinishRequest {
        pub credential_id: String,
        pub authenticator_data: String,
        pub client_data_json: String,
        pub signature: String,
        pub user_handle: Option<String>,
    }

    pub async fn webauthn_authenticate_finish(mfa_token: &str, request: &WebAuthnAuthFinishRequest) -> ApiResult<TokenPair> {
        fetch_json_with_body(
            build_request("POST", "/auth/mfa/webauthn/authenticate/finish"),
            &serde_json::json!({
                "mfa_token": mfa_token,
                "credential_id": request.credential_id,
                "authenticator_data": request.authenticator_data,
                "client_data_json": request.client_data_json,
                "signature": request.signature,
                "user_handle": request.user_handle
            }),
        ).await
    }

    pub async fn get_recovery_codes() -> ApiResult<RecoveryCodesResponse> {
        fetch_json(build_request("GET", "/auth/mfa/recovery")).await
    }

    pub async fn regenerate_recovery_codes() -> ApiResult<RecoveryCodesResponse> {
        fetch_json_with_body(build_request("POST", "/auth/mfa/recovery/regenerate"), &serde_json::json!({})).await
    }

    pub async fn use_recovery_code(mfa_token: &str, code: &str) -> ApiResult<TokenPair> {
        fetch_json_with_body(
            build_request("POST", "/auth/mfa/recovery"),
            &serde_json::json!({
                "mfa_token": mfa_token,
                "code": code
            }),
        ).await
    }

    pub async fn change_password(current: &str, new_password: &str) -> ApiResult<()> {
        fetch_empty_with_body(
            build_request("POST", "/auth/password/change"),
            &serde_json::json!({
                "current_password": current,
                "new_password": new_password
            }),
        ).await
    }
}

// ============================================================================
// Training API
// ============================================================================

pub mod training {
    use super::*;

    pub async fn list_runs(status: Option<RunStatus>, limit: Option<u32>) -> ApiResult<Vec<TrainingRun>> {
        let mut path = "/training/runs".to_string();
        let mut params = Vec::new();
        if let Some(s) = status {
            params.push(format!("status={}", s.as_str()));
        }
        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if !params.is_empty() {
            path.push('?');
            path.push_str(&params.join("&"));
        }
        fetch_json(build_request("GET", &path)).await
    }

    pub async fn get_run(id: &str) -> ApiResult<TrainingRun> {
        fetch_json(build_request("GET", &format!("/training/runs/{}", id))).await
    }

    pub async fn create_run(request: &CreateRunRequest) -> ApiResult<TrainingRun> {
        fetch_json_with_body(build_request("POST", "/training/runs"), request).await
    }

    pub async fn delete_run(id: &str) -> ApiResult<()> {
        fetch_empty(build_request("DELETE", &format!("/training/runs/{}", id))).await
    }

    pub async fn stop_run(id: &str) -> ApiResult<TrainingRun> {
        fetch_json(build_request("POST", &format!("/training/runs/{}/stop", id))).await
    }

    pub async fn get_metrics(id: &str, start: Option<&str>, end: Option<&str>) -> ApiResult<MetricsHistory> {
        let mut path = format!("/training/runs/{}/metrics", id);
        let mut params = Vec::new();
        if let Some(s) = start {
            params.push(format!("start={}", s));
        }
        if let Some(e) = end {
            params.push(format!("end={}", e));
        }
        if !params.is_empty() {
            path.push('?');
            path.push_str(&params.join("&"));
        }
        fetch_json(build_request("GET", &path)).await
    }

    pub async fn get_logs(id: &str, limit: Option<u32>) -> ApiResult<RunLogs> {
        let mut path = format!("/training/runs/{}/logs", id);
        if let Some(l) = limit {
            path.push_str(&format!("?limit={}", l));
        }
        fetch_json(build_request("GET", &path)).await
    }

    /// Get WebSocket URL for real-time training metrics
    pub fn stream_url(id: &str) -> String {
        let window = web_sys::window().expect("no window");
        let location = window.location();
        let protocol = location.protocol().unwrap_or_else(|_| "http:".to_string());
        let host = location.host().unwrap_or_else(|_| "localhost:8080".to_string());

        let ws_protocol = if protocol == "https:" { "wss:" } else { "ws:" };
        format!("{}//{}/api/training/runs/{}/stream", ws_protocol, host, id)
    }
}

// ============================================================================
// Models API
// ============================================================================

pub mod models {
    use super::*;

    pub async fn list() -> ApiResult<Vec<Model>> {
        fetch_json(build_request("GET", "/models")).await
    }

    pub async fn get(id: &str) -> ApiResult<Model> {
        fetch_json(build_request("GET", &format!("/models/{}", id))).await
    }

    pub async fn create(request: &CreateModelRequest) -> ApiResult<Model> {
        fetch_json_with_body(build_request("POST", "/models"), request).await
    }

    pub async fn update(id: &str, request: &UpdateModelRequest) -> ApiResult<Model> {
        fetch_json_with_body(build_request("PUT", &format!("/models/{}", id)), request).await
    }

    pub async fn delete(id: &str) -> ApiResult<()> {
        fetch_empty(build_request("DELETE", &format!("/models/{}", id))).await
    }

    pub async fn list_versions(model_id: &str) -> ApiResult<Vec<ModelVersion>> {
        fetch_json(build_request("GET", &format!("/models/{}/versions", model_id))).await
    }

    pub async fn get_version(model_id: &str, version: u32) -> ApiResult<ModelVersion> {
        fetch_json(build_request("GET", &format!("/models/{}/versions/{}", model_id, version))).await
    }

    pub async fn upload_version(model_id: &str, file: web_sys::File, training_run_id: Option<&str>) -> ApiResult<ModelVersion> {
        let form_data = FormData::new().unwrap();
        form_data.append_with_blob("file", &file).unwrap();
        if let Some(run_id) = training_run_id {
            form_data.append_with_str("training_run_id", run_id).unwrap();
        }

        let url = format!("{}/models/{}/versions", API_BASE, model_id);
        let mut builder = Request::post(&url);

        if let Some(token) = get_token() {
            builder = builder.header("Authorization", &format!("Bearer {}", token));
        }

        let request = builder.body(form_data).map_err(|e| ApiClientError {
            status: 0,
            message: format!("Failed to build request: {}", e),
        })?;

        let response = request.send().await?;
        let status = response.status();

        if status >= 200 && status < 300 {
            response.json::<ModelVersion>().await.map_err(|e| ApiClientError {
                status,
                message: format!("Failed to parse response: {}", e),
            })
        } else {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            Err(ApiClientError {
                status,
                message: error_text,
            })
        }
    }

    pub async fn delete_version(model_id: &str, version: u32) -> ApiResult<()> {
        fetch_empty(build_request("DELETE", &format!("/models/{}/versions/{}", model_id, version))).await
    }

    pub fn download_url(model_id: &str, version: u32) -> String {
        format!("{}/models/{}/versions/{}/download", API_BASE, model_id, version)
    }

    pub async fn deploy_version(model_id: &str, version: u32, endpoint_name: &str) -> ApiResult<InferenceEndpoint> {
        fetch_json_with_body(
            build_request("POST", &format!("/models/{}/versions/{}/deploy", model_id, version)),
            &serde_json::json!({ "name": endpoint_name }),
        ).await
    }
}

// ============================================================================
// Inference API
// ============================================================================

pub mod inference {
    use super::*;

    pub async fn list_endpoints() -> ApiResult<Vec<InferenceEndpoint>> {
        fetch_json(build_request("GET", "/inference/endpoints")).await
    }

    pub async fn get_endpoint(id: &str) -> ApiResult<InferenceEndpoint> {
        fetch_json(build_request("GET", &format!("/inference/endpoints/{}", id))).await
    }

    pub async fn create_endpoint(request: &CreateEndpointRequest) -> ApiResult<InferenceEndpoint> {
        fetch_json_with_body(build_request("POST", "/inference/endpoints"), request).await
    }

    pub async fn update_endpoint(id: &str, request: &UpdateEndpointRequest) -> ApiResult<InferenceEndpoint> {
        fetch_json_with_body(build_request("PUT", &format!("/inference/endpoints/{}", id)), request).await
    }

    pub async fn delete_endpoint(id: &str) -> ApiResult<()> {
        fetch_empty(build_request("DELETE", &format!("/inference/endpoints/{}", id))).await
    }

    pub async fn start_endpoint(id: &str) -> ApiResult<InferenceEndpoint> {
        fetch_json(build_request("POST", &format!("/inference/endpoints/{}/start", id))).await
    }

    pub async fn stop_endpoint(id: &str) -> ApiResult<InferenceEndpoint> {
        fetch_json(build_request("POST", &format!("/inference/endpoints/{}/stop", id))).await
    }

    pub async fn get_metrics(id: &str, start: Option<&str>, end: Option<&str>) -> ApiResult<InferenceMetricsHistory> {
        let mut path = format!("/inference/endpoints/{}/metrics", id);
        let mut params = Vec::new();
        if let Some(s) = start {
            params.push(format!("start={}", s));
        }
        if let Some(e) = end {
            params.push(format!("end={}", e));
        }
        if !params.is_empty() {
            path.push('?');
            path.push_str(&params.join("&"));
        }
        fetch_json(build_request("GET", &path)).await
    }

    pub async fn predict(endpoint_name: &str, inputs: serde_json::Value) -> ApiResult<PredictResponse> {
        fetch_json_with_body(
            build_request("POST", &format!("/inference/predict/{}", endpoint_name)),
            &PredictRequest { inputs },
        ).await
    }
}

// ============================================================================
// Admin API
// ============================================================================

pub mod admin {
    use super::*;

    pub async fn list_users() -> ApiResult<Vec<User>> {
        fetch_json(build_request("GET", "/admin/users")).await
    }

    pub async fn create_user(request: &CreateUserRequest) -> ApiResult<User> {
        fetch_json_with_body(build_request("POST", "/admin/users"), request).await
    }

    pub async fn update_user(id: &str, request: &UpdateUserRequest) -> ApiResult<User> {
        fetch_json_with_body(build_request("PUT", &format!("/admin/users/{}", id)), request).await
    }

    pub async fn delete_user(id: &str) -> ApiResult<()> {
        fetch_empty(build_request("DELETE", &format!("/admin/users/{}", id))).await
    }

    pub async fn get_stats() -> ApiResult<SystemStats> {
        fetch_json(build_request("GET", "/admin/stats")).await
    }
}

// ============================================================================
// Dashboard API
// ============================================================================

pub mod dashboard {
    use super::*;

    pub async fn overview() -> ApiResult<DashboardOverview> {
        fetch_json(build_request("GET", "/dashboard/overview")).await
    }
}
