//! routes — OpenAI-Compatible HTTP Handlers
//!
//! Axum route handlers backing the OpenAI-compatible surface of nexus-serve:
//! `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, and
//! `POST /v1/completions`. Also holds the shared [`AppState`] container
//! (registry + engines + tokenizers) and the chat-template dispatcher
//! ([`format_prompt`], [`format_llama3`], [`format_chatml`], [`format_gemma`])
//! that picks per-architecture turn syntax — Gemma 3/4 need
//! `<start_of_turn>…<end_of_turn>`, BitNet needs LLaMA-3 header ids, and
//! everything else gets ChatML.
//!
//! `chat_completions` resolves a model alias via [`ModelRegistry::resolve`],
//! pulls the `Arc<InferenceEngine>` + `Arc<Tokenizer>` from `AppState`, and
//! either runs `engine.generate(..)` to completion or hands off to
//! [`build_chat_stream`] for SSE streaming (generation runs on a blocking
//! task, tokens stream through an `UnboundedReceiverStream`, terminating with
//! the OpenAI `[DONE]` sentinel). `list_models` emits both canonical model
//! IDs and aliases. `api_error` constructs the standard error response
//! envelope.
//!
//! # File
//! `nexus-serve/src/api/routes.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

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

// =============================================================================
// App State
// =============================================================================

/// Shared server state passed to all handlers.
pub struct AppState {
    pub registry: ModelRegistry,
    /// Active inference engines keyed by model ID.
    pub engines: tokio::sync::RwLock<std::collections::HashMap<String, Arc<InferenceEngine>>>,
    /// Active tokenizers keyed by model ID.
    pub tokenizers: tokio::sync::RwLock<std::collections::HashMap<String, Arc<Tokenizer>>>,
    /// Hailo-10H NPU engine (when `--hailo <hef>` is used). Replaces hailo-ollama.
    /// Only one LLM can be loaded on the NPU at a time.
    #[cfg(feature = "hailo_genai")]
    pub hailo_engine: Option<Arc<crate::model::hailo10h::Hailo10hEngine>>,
    /// Hailo-10H custom HEF engine (when `--hailo-custom <hef>` is used).
    /// For AxonML/NexusFoundry-compiled models using standard HailoRT inference.
    #[cfg(feature = "hailo10h")]
    pub hailo_custom_engine: Option<Arc<crate::model::hailo_custom::HailoCustomEngine>>,
    #[cfg(feature = "nexusrt")]
    pub nexusrt_engine: Option<Arc<crate::model::nexusrt_engine::NexusRtEngine>>,
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
    // Hailo-10H NPU fast path: if a Hailo engine is loaded, route through it
    // instead of the CPU/CUDA GGUF path. The NPU handles tokenization, KV-cache,
    // and sampling entirely on-device.
    #[cfg(feature = "hailo_genai")]
    if let Some(ref hailo) = state.hailo_engine {
        // Try structured write_chat first; the HEF's embedded template handles formatting.
        // If the model has no template or write_chat produces bad output, falls back to
        // generate_raw with manual LLaMA-3 template.
        let messages: Vec<String> = req.messages.iter().map(|m| {
            serde_json::json!({"role": m.role, "content": m.content}).to_string()
        }).collect();
        let temperature = req.temperature;
        let max_tokens = req.max_tokens as u32;
        let top_p = req.top_p.unwrap_or(0.9);
        let hailo = hailo.clone();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

        if req.stream {
            let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
            let model_name = req.model.clone().unwrap_or_else(|| "hailo".to_string());
            tokio::task::spawn_blocking(move || {
                let _ = hailo.generate_chat(&messages, &[], temperature, top_p, 40, max_tokens, |text| {
                    // Filter leaked stop tokens from streaming chunks
                    let text = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").replace("<|im_end|>", "");
                    if text.is_empty() { return; }
                    let chunk = serde_json::json!({
                        "id": format!("chatcmpl-hailo-{now}"),
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": serde_json::Value::Null}]
                    });
                    let _ = tx.blocking_send(format!("data: {}\n\n", chunk));
                });
                let _ = tx.blocking_send("data: [DONE]\n\n".to_string());
            });
            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            let body = axum::body::Body::from_stream(stream.map(Ok::<_, std::convert::Infallible>));
            return Ok(axum::response::Response::builder()
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .body(body)
                .unwrap());
        } else {
            let hailo_c = hailo.clone();
            let result = tokio::task::spawn_blocking(move || {
                hailo_c.generate_chat(&messages, &[], temperature, top_p, 40, max_tokens, |_| {})
            }).await.unwrap();
            let text = result.map_err(|e| api_error(500, &format!("Hailo generate failed: {e}")))?;
            return Ok(Json(serde_json::json!({
                "id": format!("chatcmpl-hailo-{now}"),
                "object": "chat.completion",
                "created": now,
                "model": req.model.unwrap_or_else(|| "hailo".to_string()),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })).into_response());
        }
    }

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

    // Build prompt using the model's chat template. ChatML for Qwen/LLaMA/Mistral;
    // Gemma 3/4 use `<start_of_turn>…<end_of_turn>` turns instead.
    let prompt = format_prompt(engine.architecture(), &req.messages);

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

// -----------------------------------------------------------------------------
// SSE Streaming
// -----------------------------------------------------------------------------

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

/// Dispatch chat-template formatting by model architecture. Gemma 3/4 use a
/// distinct `<start_of_turn>…<end_of_turn>` turn format that ChatML's
/// `<|im_start|>` specials can't substitute for (they tokenize as garbage
/// against the Gemma vocab, which makes the first-token argmax fire EOS).
fn format_prompt(architecture: &str, messages: &[crate::api::types::ChatMessage]) -> String {
    match architecture {
        "gemma" | "gemma2" | "gemma3" | "gemma4" => format_gemma(messages),
        a if a.starts_with("bitnet") => format_llama3(messages),
        _ => format_chatml(messages),
    }
}

/// Format messages using the LLaMA-3 / BitNet chat template.
///
/// Output:
///   <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
///   ...
///   <|start_header_id|>assistant<|end_header_id|>\n\n
///
/// Matches the official LLaMA-3 Instruct template (used by BitNet b1.58-2B-4T).
/// BitNet's vocab omits ChatML's `<|im_start|>`/`<|im_end|>`, so using the
/// wrong template causes the model to echo them as raw UTF-8 bytes in the
/// output.
fn format_llama3(messages: &[crate::api::types::ChatMessage]) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|begin_of_text|>");
    for m in messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(&m.role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(&m.content);
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

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

/// Format messages using the Gemma instruction-tuned chat template.
///
/// Output:
///   <start_of_turn>role\ncontent<end_of_turn>\n
///   ...
///   <start_of_turn>model\n
///
/// Role mapping follows Google's official Gemma Jinja template: `assistant`
/// becomes `model`; `user` and `system` pass through. Content is trimmed
/// (matches the upstream template's `| trim`).
fn format_gemma(messages: &[crate::api::types::ChatMessage]) -> String {
    let mut prompt = String::new();
    for m in messages {
        let role = if m.role == "assistant" { "model" } else { m.role.as_str() };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(m.content.trim());
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
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

// =============================================================================
// POST /v1/hailo/infer — Raw tensor inference on custom Hailo HEF
// =============================================================================

#[cfg(feature = "hailo10h")]
pub async fn hailo_infer(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Result<Response, (StatusCode, Json<ApiError>)> {
    let engine = state.hailo_custom_engine.as_ref().ok_or_else(|| {
        api_error(503, "No custom Hailo HEF loaded. Use --hailo-custom <hef>")
    })?;

    let input_data = body.to_vec();
    let out_size = engine.output_frame_size();
    let mut output_data = vec![0u8; if out_size > 0 { out_size } else { input_data.len() }];

    let engine = engine.clone();
    let result = tokio::task::spawn_blocking(move || {
        engine.infer(&input_data, &mut output_data)?;
        Ok::<Vec<u8>, anyhow::Error>(output_data)
    })
    .await
    .map_err(|e| api_error(500, &format!("Task join error: {e}")))?
    .map_err(|e| api_error(500, &format!("Inference error: {e}")))?;

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/octet-stream")
        .body(axum::body::Body::from(result))
        .unwrap())
}
