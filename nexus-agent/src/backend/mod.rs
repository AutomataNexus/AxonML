//! Backend Module Root — LLM Backend Trait And Implementations
//!
//! Root module for nexus-agent's pluggable LLM backend layer. Defines the
//! `LlmBackend` async trait that the ReAct loop calls, and re-exports the
//! two concrete implementations:
//!
//! - `anthropic` — talks to nexus-serve's `/v1/messages` Anthropic
//!   Messages API endpoint (remote-compatible shape; works against the
//!   real Anthropic API too). Tool calls round-trip as native
//!   `tool_use` / `tool_result` content blocks with
//!   `stop_reason = "tool_use"`.
//! - `local` — talks to nexus-serve's OpenAI-compatible
//!   `/v1/chat/completions` endpoint for local inference.
//!
//! `LlmBackend` contract:
//! - `chat_completion` — send messages + tools, return assistant
//!   `Message`. If the model wants to call tools, `tool_calls` is
//!   populated; for a final answer, `content` is populated.
//! - `list_models` — list available models.
//! - `health_check` — probe reachability.
//!
//! No external API dependencies — fully self-hosted against nexus-serve
//! by default.
//!
//! # File
//! `nexus-agent/src/backend/mod.rs`
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
// Backend Submodules
// =============================================================================

pub mod anthropic;
pub mod local;

// =============================================================================
// Imports
// =============================================================================

use async_trait::async_trait;

use crate::{Message, ToolDefinition};

// =============================================================================
// LlmBackend Trait
// =============================================================================

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
