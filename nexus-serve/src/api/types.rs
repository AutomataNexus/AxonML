//! types — OpenAI-Compatible Request/Response DTOs
//!
//! Pure-data structs (no logic, just serde derives) shared between the axum
//! handlers in `routes.rs` and any external client. Groups:
//!
//! - Chat Completions: [`ChatCompletionRequest`], [`ChatMessage`],
//!   [`ChatCompletionResponse`], [`ChatChoice`].
//! - SSE streaming chunks: [`ChatCompletionChunk`], [`ChatChunkChoice`],
//!   [`ChunkDelta`] — emitted one-per-token as `data: {json}\n\n` with a
//!   terminal `data: [DONE]\n\n`.
//! - Text completions: [`CompletionRequest`], [`CompletionResponse`],
//!   [`TextChoice`].
//! - Shared: [`Usage`] (token accounting), [`ModelListResponse`] /
//!   [`ModelObject`] for `GET /v1/models`.
//! - Errors: [`ApiError`], [`ApiErrorBody`] with OpenAI's `{message, type,
//!   code}` envelope.
//! - Private defaults: [`default_max_tokens`] = 256, [`default_temperature`] =
//!   0.7 for serde `#[serde(default = "...")]` attributes.
//!
//! # File
//! `nexus-serve/src/api/types.rs`
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

use serde::{Deserialize, Serialize};

// =============================================================================
// OpenAI Chat Completions
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

// =============================================================================
// SSE streaming chunk (OpenAI format)
// =============================================================================

/// One SSE chunk for streaming chat completions.
/// Emitted as `data: {json}\n\n` per token, with a final `data: [DONE]\n\n`.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// =============================================================================
// OpenAI Completions (text)
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct TextChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

// =============================================================================
// Shared
// =============================================================================

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub created: u64,
}

// =============================================================================
// Error
// =============================================================================

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
}

// =============================================================================
// Defaults
// =============================================================================

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}
