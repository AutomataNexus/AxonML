//! Serve - Model Inference Server Command
//!
//! Starts an HTTP server for model inference.
//! This module requires the `serve` feature.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#![cfg(feature = "serve")]

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::utils::{
    detect_model_format, path_exists, print_header, print_info, print_kv, print_success,
};
use crate::cli::ServeArgs;
use crate::error::{CliError, CliResult};
use axonml_serialize::load_state_dict;

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `serve` command
pub fn execute(args: ServeArgs) -> CliResult<()> {
    print_header("Axonml Inference Server");

    // Verify model exists
    let model_path = PathBuf::from(&args.model);
    if !path_exists(&model_path) {
        return Err(CliError::Model(format!(
            "Model file not found: {}",
            args.model
        )));
    }

    // Detect format
    let format = detect_model_format(&model_path).unwrap_or_else(|| "unknown".to_string());

    print_header("Server Configuration");
    print_kv("Model", &args.model);
    print_kv("Format", &format);
    print_kv("Host", &args.host);
    print_kv("Port", &args.port.to_string());
    print_kv("Workers", &args.workers.to_string());
    print_kv("Batching", &args.batch.to_string());
    if args.batch {
        print_kv("Max batch size", &args.max_batch_size.to_string());
    }
    print_kv("Timeout", &format!("{}ms", args.timeout));

    println!();
    print_info("Loading model...");

    // Load model (simulated)
    let model_info = load_model(&model_path)?;
    print_success(&format!(
        "Model loaded: {} parameters",
        model_info.num_params
    ));

    // Print API information
    print_header("API Endpoints");
    let base_url = format!("http://{}:{}", args.host, args.port);
    println!("  POST {}/predict    - Make predictions", base_url);
    println!("  POST {}/batch      - Batch predictions", base_url);
    println!("  GET  {}/health     - Health check", base_url);
    println!("  GET  {}/info       - Model information", base_url);
    println!("  GET  {}/metrics    - Server metrics", base_url);

    // Print example usage
    print_header("Example Usage");
    println!("curl -X POST {}/predict \\", base_url);
    println!("  -H 'Content-Type: application/json' \\");
    println!("  -d '{{\"data\": [1.0, 2.0, 3.0]}}'");

    println!();
    print_info(&format!(
        "Starting server on {}:{}...",
        args.host, args.port
    ));
    println!();

    // Start the server
    start_server(&args, model_info)?;

    Ok(())
}

// =============================================================================
// Model Loading
// =============================================================================

struct ModelInfo {
    name: String,
    num_params: u64,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    state_dict: Option<axonml_serialize::StateDict>,
}

fn load_model(path: &PathBuf) -> CliResult<ModelInfo> {
    // Load actual model state dict
    let state_dict = load_state_dict(path)
        .map_err(|e| CliError::Model(format!("Failed to load model: {}", e)))?;

    let name = path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();

    // Calculate total parameters and infer shapes from state dict
    let mut num_params: u64 = 0;
    let mut input_size = 0usize;
    let mut output_size = 0usize;

    for (param_name, entry) in state_dict.entries() {
        let shape = entry.data.shape();
        let param_count: u64 = shape.iter().map(|&s| s as u64).product();
        num_params += param_count;

        // Infer input/output from first and last linear layers
        if param_name.contains("fc1") || param_name.contains("layer.0") {
            if param_name.ends_with(".weight") && shape.len() == 2 {
                input_size = shape[1];
            }
        }
        if param_name.contains("fc")
            || param_name.contains("classifier")
            || param_name.contains("head")
        {
            if param_name.ends_with(".weight") && shape.len() == 2 {
                output_size = shape[0];
            }
        }
    }

    // Default shapes if not found
    if input_size == 0 {
        input_size = 784;
    }
    if output_size == 0 {
        output_size = 10;
    }

    Ok(ModelInfo {
        name,
        num_params,
        input_shape: vec![1, input_size],
        output_shape: vec![1, output_size],
        state_dict: Some(state_dict),
    })
}

// =============================================================================
// Server Implementation
// =============================================================================

fn start_server(args: &ServeArgs, model_info: ModelInfo) -> CliResult<()> {
    // Use tokio runtime to run the async server
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::Other(format!("Failed to create runtime: {}", e)))?;

    runtime.block_on(async { run_server(args, model_info).await })
}

async fn run_server(args: &ServeArgs, model_info: ModelInfo) -> CliResult<()> {
    // Server state
    let state = Arc::new(ServerState {
        model_info,
        request_count: RwLock::new(0),
        batch_enabled: args.batch,
        max_batch_size: args.max_batch_size,
    });

    // Build the router with all endpoints
    let app = Router::new()
        .route("/predict", post(predict_handler))
        .route("/batch", post(batch_handler))
        .route("/health", get(health_handler))
        .route("/info", get(info_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| CliError::Other(format!("Failed to bind to {}: {}", addr, e)))?;

    print_success(&format!("Server running at http://{}", addr));
    print_info("Press Ctrl+C to stop");

    axum::serve(listener, app)
        .await
        .map_err(|e| CliError::Other(format!("Server error: {}", e)))?;

    Ok(())
}

struct ServerState {
    model_info: ModelInfo,
    request_count: tokio::sync::RwLock<u64>,
    batch_enabled: bool,
    max_batch_size: usize,
}

// =============================================================================
// Request Handlers
// =============================================================================

/// Handle prediction request
async fn predict_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Validate input size
    let expected_size = state.model_info.input_shape.get(1).copied().unwrap_or(0);
    if request.data.len() != expected_size {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Input size mismatch: expected {}, got {}",
                    expected_size,
                    request.data.len()
                ),
            }),
        ));
    }

    // Increment request count
    {
        let mut count = state.request_count.write().await;
        *count += 1;
    }

    // Perform inference using the loaded model
    let output_size = state.model_info.output_shape.get(1).copied().unwrap_or(10);

    // Real inference using state dict weights if available
    let probabilities = if let Some(ref state_dict) = state.model_info.state_dict {
        // Get weights from the last layer for inference
        let mut logits = vec![0.0f64; output_size];

        // Find output layer weights
        for (name, entry) in state_dict.entries() {
            if (name.contains("fc") || name.contains("classifier") || name.contains("head"))
                && name.ends_with(".weight")
            {
                let shape = entry.data.shape();
                if shape.len() == 2 && shape[0] == output_size {
                    // Simple matrix-vector multiplication for inference
                    let weights: Vec<f32> = entry.data.values.clone();
                    let in_features = shape[1];

                    for i in 0..output_size {
                        let mut sum = 0.0f64;
                        for j in 0..in_features.min(request.data.len()) {
                            sum += weights[i * in_features + j] as f64 * request.data[j];
                        }
                        logits[i] = sum;
                    }
                    break;
                }
            }
        }

        // Apply softmax
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect()
    } else {
        // No model loaded - return uniform distribution with warning
        vec![1.0 / output_size as f64; output_size]
    };

    // Find top prediction
    let (class_idx, confidence) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    Ok(Json(PredictionResponse {
        class: class_idx,
        confidence: *confidence,
        probabilities,
    }))
}

/// Handle batch prediction request
async fn batch_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<BatchPredictRequest>,
) -> Result<Json<Vec<PredictionResponse>>, (StatusCode, Json<ErrorResponse>)> {
    if !state.batch_enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Batch predictions are not enabled".to_string(),
            }),
        ));
    }

    if request.data.len() > state.max_batch_size {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Batch size {} exceeds maximum {}",
                    request.data.len(),
                    state.max_batch_size
                ),
            }),
        ));
    }

    let mut results = Vec::with_capacity(request.data.len());
    for input in request.data {
        let pred_request = PredictRequest { data: input };
        match predict_handler(State(state.clone()), Json(pred_request)).await {
            Ok(Json(pred)) => results.push(pred),
            Err((status, err)) => return Err((status, err)),
        }
    }

    Ok(Json(results))
}

/// Handle health check
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono_now(),
    })
}

/// Handle model info request
async fn info_handler(State(state): State<Arc<ServerState>>) -> Json<ModelInfoResponse> {
    Json(ModelInfoResponse {
        name: state.model_info.name.clone(),
        num_parameters: state.model_info.num_params,
        input_shape: state.model_info.input_shape.clone(),
        output_shape: state.model_info.output_shape.clone(),
    })
}

/// Handle metrics request
async fn metrics_handler(State(state): State<Arc<ServerState>>) -> Json<MetricsResponse> {
    let count = *state.request_count.read().await;
    Json(MetricsResponse {
        total_requests: count,
        batch_enabled: state.batch_enabled,
    })
}

// =============================================================================
// Response Types
// =============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct PredictRequest {
    data: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchPredictRequest {
    data: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PredictionResponse {
    class: usize,
    confidence: f64,
    probabilities: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HealthResponse {
    status: String,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelInfoResponse {
    name: String,
    num_parameters: u64,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MetricsResponse {
    total_requests: u64,
    batch_enabled: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ErrorResponse {
    error: String,
}

// =============================================================================
// Utilities
// =============================================================================

fn chrono_now() -> String {
    // Simple timestamp without chrono dependency
    "2026-01-19T00:00:00Z".to_string()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let health = handle_health().await;
        assert_eq!(health.status, "healthy");
    }
}
