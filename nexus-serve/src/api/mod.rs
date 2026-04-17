//! api — HTTP Surface Module Root
//!
//! Groups the HTTP-facing submodules for nexus-serve. Re-exports three
//! siblings: [`types`] (serde request/response DTOs shared across OpenAI +
//! custom endpoints), [`routes`] (axum handlers for `/health`, `/v1/models`,
//! `/v1/chat/completions`, `/v1/completions` and the [`routes::AppState`]
//! container shared across handlers), and [`messages`] (Anthropic Messages
//! API `/v1/messages` with SSE streaming and tool_use / tool_result content
//! blocks).
//!
//! Endpoints exposed:
//! - OpenAI-compatible: `POST /v1/chat/completions`, `POST /v1/completions`,
//!   `GET /v1/models`
//! - Anthropic Messages API: `POST /v1/messages` (see `messages.rs`)
//! - Health: `GET /health`
//!
//! # File
//! `nexus-serve/src/api/mod.rs`
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
// Submodules
// =============================================================================

pub mod types;
pub mod routes;
pub mod messages;
