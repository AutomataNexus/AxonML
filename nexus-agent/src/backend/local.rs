//! Local Backend — OpenAI-Compatible /v1/chat/completions For nexus-serve
//!
//! Implements `LocalBackend`, an `LlmBackend` that talks to nexus-serve
//! (or any other OpenAI-compatible server) over HTTP. Default endpoint
//! is `http://127.0.0.1:11435`. Supports any model nexus-serve can load
//! via GGUF — Gemma 4, Qwen 3, and the rest of the zoo.
//!
//! Key items:
//! - `LocalBackend` — reqwest-based client with `new()` and `with_url`
//!   builders.
//! - Wire types: `ChatRequest` (sets `tool_choice = "auto"` when tools
//!   are supplied), `ChatResponse`, `Choice`, `ResponseMessage`,
//!   `ResponseToolCall`, `ResponseFunction`, `ModelsResponse`,
//!   `ModelEntry`.
//! - `LlmBackend` impl — POSTs to `/v1/chat/completions`, translates the
//!   OpenAI response `tool_calls` array into our internal `ToolCall`
//!   vector, lists models via `/v1/models`, and pings `/health` for the
//!   health check.
//!
//! This is the legacy path. New agents should use `AnthropicBackend`,
//! whose content-block / `tool_use` shape is the hard rule for
//! nexus-agent ↔ nexus-serve tool calling.
//!
//! # File
//! `nexus-agent/src/backend/local.rs`
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

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{FunctionCall, Message, ToolCall, ToolDefinition};

use super::LlmBackend;

// =============================================================================
// Constants
// =============================================================================

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11435";

// =============================================================================
// Backend Struct And Constructors
// =============================================================================

/// Local inference backend powered by nexus-serve.
pub struct LocalBackend {
    client: reqwest::Client,
    base_url: String,
}

impl LocalBackend {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }
}

impl Default for LocalBackend {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// OpenAI-compatible request/response types
// =============================================================================

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    temperature: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<&'a ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a str>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Deserialize)]
struct ResponseToolCall {
    id: String,
    r#type: String,
    function: ResponseFunction,
}

#[derive(Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

// =============================================================================
// LlmBackend implementation
// =============================================================================

#[async_trait]
impl LlmBackend for LocalBackend {
    async fn chat_completion(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
    ) -> anyhow::Result<Message> {
        let tool_refs: Vec<&ToolDefinition> = tools.iter().collect();

        let request = ChatRequest {
            model,
            messages,
            temperature,
            tools: tool_refs,
            tool_choice: if tools.is_empty() { None } else { Some("auto") },
        };

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("nexus-serve returned {status}: {body}");
        }

        let chat_response: ChatResponse = response.json().await?;
        let choice = chat_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty response from nexus-serve"))?;

        let msg = choice.message;

        // Convert to our Message type
        let tool_calls = msg.tool_calls.map(|tcs| {
            tcs.into_iter()
                .map(|tc| ToolCall {
                    id: tc.id,
                    r#type: tc.r#type,
                    function: FunctionCall {
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    },
                })
                .collect()
        });

        Ok(Message {
            role: "assistant".to_string(),
            content: msg.content.unwrap_or_default(),
            tool_call_id: None,
            tool_calls,
        })
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let response = self
            .client
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await?;

        let models: ModelsResponse = response.json().await?;
        Ok(models.data.into_iter().map(|m| m.id).collect())
    }

    async fn health_check(&self) -> bool {
        self.client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}
