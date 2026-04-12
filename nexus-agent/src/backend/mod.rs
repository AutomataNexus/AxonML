//! LLM backend trait + implementations.
//!
//! The backend talks to nexus-serve (or any OpenAI-compatible endpoint)
//! over HTTP. No external API dependencies — fully self-hosted.

pub mod local;

use async_trait::async_trait;

use crate::{Message, ToolDefinition};

/// Pluggable LLM backend. The default implementation calls nexus-serve's
/// OpenAI-compatible `/v1/chat/completions` endpoint.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Send a chat completion request and return the assistant's response.
    ///
    /// If the LLM wants to call tools, the returned `Message` will have
    /// `tool_calls` populated. If it's a final answer, `content` will be
    /// populated and `tool_calls` will be `None`.
    async fn chat_completion(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
    ) -> anyhow::Result<Message>;

    /// List available models on this backend.
    async fn list_models(&self) -> anyhow::Result<Vec<String>>;

    /// Check if the backend is reachable.
    async fn health_check(&self) -> bool;
}
