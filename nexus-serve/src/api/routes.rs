//! HTTP route handlers for the OpenAI-compatible API.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::*;
use crate::model::inference::InferenceEngine;
use crate::model::registry::ModelRegistry;
use crate::tokenizer::Tokenizer;

/// Shared server state passed to all handlers.
pub struct AppState {
    pub registry: ModelRegistry,
    /// Active inference engines keyed by model ID.
    pub engines: tokio::sync::RwLock<std::collections::HashMap<String, Arc<InferenceEngine>>>,
    /// Active tokenizers keyed by model ID.
    pub tokenizers: tokio::sync::RwLock<std::collections::HashMap<String, Arc<Tokenizer>>>,
}

// =============================================================================
// Health
// =============================================================================

pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "server": "nexus-serve",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

// =============================================================================
// OpenAI: GET /v1/models
// =============================================================================

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let models = state.registry.list().await;
    let aliases = state.registry.list_aliases().await;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Return both canonical model IDs and aliases so clients can discover
    // friendly names like "sage" and "oracle".
    let mut data: Vec<ModelObject> = models
        .iter()
        .map(|m| ModelObject {
            id: m.id.clone(),
            object: "model".to_string(),
            owned_by: "nexus-serve".to_string(),
            created: now,
        })
        .collect();

    for (alias, _canonical) in &aliases {
        data.push(ModelObject {
            id: alias.clone(),
            object: "model-alias".to_string(),
            owned_by: "nexus-serve".to_string(),
            created: now,
        });
    }

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

// =============================================================================
// OpenAI: POST /v1/chat/completions
// =============================================================================

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ApiError>)> {
    // Resolve the requested model or alias (e.g., "sage" → "Qwen2.5 Coder 1.5B Instruct"),
    // or fall back to the default model if none specified.
    let requested = match req.model.as_deref() {
        Some(m) => m.to_string(),
        None => state
            .registry
            .default_model()
            .await
            .ok_or_else(|| api_error(400, "No model specified and no default model loaded"))?,
    };

    let model_id = state
        .registry
        .resolve(&requested)
        .await
        .ok_or_else(|| api_error(404, &format!("Model not found: {}", requested)))?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Build prompt using the model's chat template.
    // For Qwen/LLaMA/Mistral families this is the ChatML format:
    //   <|im_start|>role\ncontent<|im_end|>\n
    //   ...
    //   <|im_start|>assistant\n
    // Other formats (Gemma, Phi) would need different handling.
    let prompt = format_chatml(&req.messages);

    // Get engine + tokenizer
    let engines = state.engines.read().await;
    let tokenizers = state.tokenizers.read().await;

    let engine = engines
        .get(&model_id)
        .ok_or_else(|| api_error(503, &format!("Model {} not loaded for inference", model_id)))?;
    let tokenizer = tokenizers
        .get(&model_id)
        .ok_or_else(|| api_error(503, &format!("Tokenizer for {} not loaded", model_id)))?;

    // Tokenize + generate
    let input_ids = tokenizer.encode(&prompt);
    let prompt_tokens = input_ids.len();

    let generated_ids = engine.generate(
        &input_ids,
        req.max_tokens,
        req.temperature,
        req.top_p.unwrap_or(0.9),
    );

    let completion_tokens = generated_ids.len();
    let response_text = tokenizer.decode(&generated_ids);

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_text,
            },
            finish_reason: if completion_tokens >= req.max_tokens {
                "length".to_string()
            } else {
                "stop".to_string()
            },
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

// =============================================================================
// OpenAI: POST /v1/completions
// =============================================================================

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<ApiError>)> {
    let requested = match req.model.as_deref() {
        Some(m) => m.to_string(),
        None => state
            .registry
            .default_model()
            .await
            .ok_or_else(|| api_error(400, "No model specified and no default model loaded"))?,
    };

    let model_id = state
        .registry
        .resolve(&requested)
        .await
        .ok_or_else(|| api_error(404, &format!("Model not found: {}", requested)))?;

    let engines = state.engines.read().await;
    let tokenizers = state.tokenizers.read().await;

    let engine = engines
        .get(&model_id)
        .ok_or_else(|| api_error(503, &format!("Model {} not loaded for inference", model_id)))?;
    let tokenizer = tokenizers
        .get(&model_id)
        .ok_or_else(|| api_error(503, &format!("Tokenizer for {} not loaded", model_id)))?;

    let input_ids = tokenizer.encode(&req.prompt);
    let prompt_tokens = input_ids.len();

    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_p = req.top_p.unwrap_or(0.9);

    let generated_ids = engine.generate(&input_ids, max_tokens, temperature, top_p);
    let completion_tokens = generated_ids.len();
    let text = tokenizer.decode(&generated_ids);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", now),
        object: "text_completion".to_string(),
        created: now,
        model: model_id,
        choices: vec![TextChoice {
            index: 0,
            text,
            finish_reason: if completion_tokens >= max_tokens {
                "length".to_string()
            } else {
                "stop".to_string()
            },
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

// =============================================================================
// Chat template formatting
// =============================================================================

/// Format messages using the ChatML template (used by Qwen, many LLaMA derivatives,
/// Mistral Instruct, and most modern instruction-tuned models).
///
/// Output:
///   <|im_start|>role\ncontent<|im_end|>\n
///   ...
///   <|im_start|>assistant\n
fn format_chatml(messages: &[crate::api::types::ChatMessage]) -> String {
    let mut prompt = String::new();
    for m in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&m.role);
        prompt.push('\n');
        prompt.push_str(&m.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// =============================================================================
// Error helper
// =============================================================================

fn api_error(code: u16, message: &str) -> (StatusCode, Json<ApiError>) {
    (
        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(ApiError {
            error: ApiErrorBody {
                message: message.to_string(),
                error_type: "invalid_request_error".to_string(),
                code: code.to_string(),
            },
        }),
    )
}
