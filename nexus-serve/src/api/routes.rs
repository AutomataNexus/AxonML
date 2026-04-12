//! HTTP route handlers for the OpenAI-compatible API.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::stream::{Stream, StreamExt};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::UnboundedReceiverStream;

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
) -> Result<Response, (StatusCode, Json<ApiError>)> {
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

    // Build prompt using the model's chat template (ChatML for Qwen/LLaMA/Mistral).
    let prompt = format_chatml(&req.messages);

    // Get engine + tokenizer. We clone the Arcs so the background task can own them.
    let engine = {
        let engines = state.engines.read().await;
        engines
            .get(&model_id)
            .cloned()
            .ok_or_else(|| api_error(503, &format!("Model {} not loaded for inference", model_id)))?
    };
    let tokenizer = {
        let tokenizers = state.tokenizers.read().await;
        tokenizers
            .get(&model_id)
            .cloned()
            .ok_or_else(|| api_error(503, &format!("Tokenizer for {} not loaded", model_id)))?
    };

    // Tokenize prompt
    let input_ids = tokenizer.encode(&prompt);
    let prompt_tokens = input_ids.len();

    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_p = req.top_p.unwrap_or(0.9);
    let chat_id = format!("chatcmpl-{}", now);

    if req.stream {
        // SSE streaming path
        let stream = build_chat_stream(
            chat_id,
            model_id,
            now,
            engine,
            tokenizer,
            input_ids,
            max_tokens,
            temperature,
            top_p,
        );
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        // Non-streaming: run to completion and return JSON
        let generated_ids =
            engine.generate(&input_ids, max_tokens, temperature, top_p);
        let completion_tokens = generated_ids.len();
        let response_text = tokenizer.decode(&generated_ids);

        let body = Json(ChatCompletionResponse {
            id: chat_id,
            object: "chat.completion".to_string(),
            created: now,
            model: model_id,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: response_text,
                },
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
        });
        Ok(body.into_response())
    }
}

/// Build an SSE stream that runs generation in a blocking task and yields
/// one chunk per token, following the OpenAI chat.completion.chunk format.
fn build_chat_stream(
    chat_id: String,
    model_id: String,
    created: u64,
    engine: Arc<InferenceEngine>,
    tokenizer: Arc<Tokenizer>,
    input_ids: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();

    // First chunk: announce the assistant role
    let role_chunk = ChatCompletionChunk {
        id: chat_id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_id.clone(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let _ = tx.send(Event::default().data(serde_json::to_string(&role_chunk).unwrap()));

    // Generation runs on a blocking thread (inference is synchronous and CPU/GPU-bound).
    tokio::task::spawn_blocking(move || {
        let mut token_count = 0usize;

        engine.generate_stream(
            &input_ids,
            max_tokens,
            temperature,
            top_p,
            |tok_id| {
                token_count += 1;
                // Decode this single token to its text chunk
                let piece = tokenizer.decode(&[tok_id]);
                let chunk = ChatCompletionChunk {
                    id: chat_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(piece),
                        },
                        finish_reason: None,
                    }],
                };
                let payload = match serde_json::to_string(&chunk) {
                    Ok(s) => s,
                    Err(_) => return false,
                };
                // Send; if the receiver dropped (client disconnected), stop.
                tx.send(Event::default().data(payload)).is_ok()
            },
        );

        // Final chunk with finish_reason
        let finish = if token_count >= max_tokens { "length" } else { "stop" };
        let final_chunk = ChatCompletionChunk {
            id: chat_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChunkDelta::default(),
                finish_reason: Some(finish.to_string()),
            }],
        };
        let _ = tx.send(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
        // OpenAI-spec terminator
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    UnboundedReceiverStream::new(rx).map(Ok)
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
