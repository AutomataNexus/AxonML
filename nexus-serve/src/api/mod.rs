//! REST API for nexus-serve.
//!
//! OpenAI-compatible:
//! - POST /v1/chat/completions
//! - POST /v1/completions
//! - GET  /v1/models
//!
//! Anthropic Messages API:
//! - POST /v1/messages   (see `messages.rs`)
//!
//! Other:
//! - GET  /health

pub mod types;
pub mod routes;
pub mod messages;
