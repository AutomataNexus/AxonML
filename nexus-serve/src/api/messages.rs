//! Anthropic Messages API — `POST /v1/messages`
//!
//! Implements the request/response shape of Anthropic's Messages API so
//! nexus-agent (and any Claude-SDK-compatible client) can talk to nexus-serve
//! with one URL swap. Key shape differences from OpenAI's
//! `/v1/chat/completions`:
//!
//! - Response `content` is an **array of content blocks**
//!   (`{"type":"text"}` or `{"type":"tool_use"}`), not a single string.
//! - `stop_reason` uses `"end_turn" | "max_tokens" | "stop_sequence" |
//!   "tool_use"` (not `"stop" | "length"`).
//! - `tools[]` in the request carries `{name, description, input_schema}`.
//! - Assistant turns with tool calls come back as
//!   `[{type:"text",text:"..."}, {type:"tool_use", id:"...", name, input}]`
//!   with `stop_reason: "tool_use"`.
//!
//! # Tool-call delivery (Phase 1)
//!
//! BitNet b1.58-2B wasn't fine-tuned on Anthropic's tool-use tokens, so we
//! use **prompt-template tool calling**: the server injects a system
//! prompt that teaches the model to emit a recognisable tag sequence, then
//! parses the tagged output back into proper `tool_use` content blocks.
//!
//! Format emitted by the model:
//!
//! ```text
//! I'll read the file first.
//! <tool_use>
//! {"name": "read_file", "input": {"path": "/etc/hosts"}}
//! </tool_use>
//! ```
//!
//! Parsing rules:
//! - Anything before the first `<tool_use>` is a `text` block.
//! - Each `<tool_use>...</tool_use>` pair becomes a `tool_use` block; the
//!   inner text is parsed as JSON `{name, input}`.
//! - Generation halts after the first `</tool_use>` so the client can run
//!   the tool and come back with `tool_result`. If the model never emits a
//!   tool call, we fall through to `stop_reason: "end_turn"` like a
//!   normal chat response.
//!
//! # Streaming (SSE)
//!
//! When `stream: true` is set on the request, the handler returns an SSE
//! stream emitting Anthropic-shaped events in order: `message_start`,
//! `content_block_start` (text, index=0), many `content_block_delta`
//! (one per decoded token), `content_block_stop`, `message_delta`
//! (final `stop_reason` + `usage`), `message_stop`. The full tool_use
//! parse still runs server-side AFTER the stream completes; tool_use
//! blocks are not streamed as partial input_json_delta in this first cut.
//! Clients that need the structured tool_use content should collect the
//! deltas, reassemble the text, and rely on the terminal `message_delta`
//! event's `stop_reason` to decide whether to re-fetch non-streamed.
//! (The non-streaming path stays available at `stream: false`.)
//!
//! # Follow-ups
//! - Once we have a BitNet fine-tune with dedicated `<|tool_use|>` /
//!   `<|tool_end|>` special tokens (the Trident-Coder BPE already has
//!   them reserved at IDs 5-7), swap the text-tag parser for a stop-token
//!   parser on the wire.
//! - `input_json_delta` streaming for tool_use blocks — emit a
//!   `content_block_start { tool_use }` when we detect `<tool_use>` in
//!   the stream, then partial-JSON deltas until `</tool_use>`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use super::routes::AppState;

// =============================================================================
// Request types
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    /// Model id / alias. Matches the registry the same way `/v1/chat` does.
    pub model: Option<String>,
    /// Conversation so far. Each message's `content` may be a plain string
    /// OR a list of content blocks; we accept both (serde untagged enum).
    pub messages: Vec<MessagesMessage>,
    /// Optional top-level system prompt. Anthropic puts system outside the
    /// `messages` array.
    #[serde(default)]
    pub system: Option<SystemField>,
    /// Tools the model can call.
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// Tool choice policy. Currently we honour only `auto` (default).
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub stream: bool,
}

/// Anthropic's `system` field accepts either a plain string or a content-
/// block list. We accept both and flatten to a single string.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemField {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl SystemField {
    fn to_text(&self) -> String {
        match self {
            SystemField::Text(s) => s.clone(),
            SystemField::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct MessagesMessage {
    pub role: String, // "user" | "assistant"
    #[serde(default)]
    pub content: MessageContent,
}

/// Content of one message — plain string or list of content blocks.
#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
    #[default]
    #[serde(skip)]
    Empty,
}

/// One content block in a request (user) message.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: serde_json::Value, // string or list of blocks
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Tool definition in the request.
#[derive(Debug, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

// =============================================================================
// Response types
// =============================================================================

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: &'static str, // always "message"
    pub role: &'static str, // always "assistant"
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: MessagesUsage,
}

#[derive(Debug, Serialize)]
pub struct MessagesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

// =============================================================================
// Handler
// =============================================================================

pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessagesRequest>,
) -> Result<Response, Response> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Resolve model (same lookup as chat endpoint).
    let model_id = resolve_model_id(&state, req.model.as_deref()).await.ok_or_else(|| {
        err_response(StatusCode::BAD_REQUEST, "no models loaded", "model_not_found")
    })?;
    let engine = state.engines.read().await.get(&model_id).cloned().ok_or_else(|| {
        err_response(StatusCode::INTERNAL_SERVER_ERROR, "engine not initialized", "engine_missing")
    })?;
    let tokenizer = state.tokenizers.read().await.get(&model_id).cloned().ok_or_else(|| {
        err_response(StatusCode::INTERNAL_SERVER_ERROR, "tokenizer not initialized", "tokenizer_missing")
    })?;

    // Build the prompt: system + rendered history + tool-use instruction.
    let prompt = build_prompt(engine.architecture(), &req);
    let input_ids = tokenizer.encode(&prompt);
    let input_tokens = input_ids.len();
    let msg_id = format!("msg_{now}");

    if req.stream {
        let stream = build_messages_stream(
            msg_id,
            model_id,
            engine,
            tokenizer,
            input_ids,
            input_tokens,
            req.max_tokens,
            req.temperature,
            req.top_p.unwrap_or(0.9),
        );
        return Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response());
    }

    let generated_ids = engine.generate(&input_ids, req.max_tokens, req.temperature, req.top_p.unwrap_or(0.9));
    let output_tokens = generated_ids.len();
    let raw = tokenizer.decode(&generated_ids);

    // Parse the raw text into content blocks, tool_use-aware.
    let (content, stop_reason) = parse_assistant_output(&raw, output_tokens, req.max_tokens);

    let body = Json(MessagesResponse {
        id: msg_id,
        response_type: "message",
        role: "assistant",
        model: model_id,
        content,
        stop_reason,
        stop_sequence: None,
        usage: MessagesUsage { input_tokens, output_tokens },
    });
    Ok(body.into_response())
}

// =============================================================================
// SSE streaming (Anthropic event protocol)
// =============================================================================

/// Build an SSE stream that emits Anthropic `/v1/messages` events as the
/// model decodes. Generation runs on a blocking tokio task to keep the
/// async runtime free; each decoded token produces one `content_block_delta`
/// event with the incremental text piece.
///
/// Event order (matches Anthropic spec):
///   message_start → content_block_start(text, 0)
///     → content_block_delta × N
///   → content_block_stop(0) → message_delta → message_stop
#[allow(clippy::too_many_arguments)]
fn build_messages_stream(
    msg_id: String,
    model_id: String,
    engine: Arc<crate::model::inference::InferenceEngine>,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    input_ids: Vec<u32>,
    input_tokens: usize,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();

    // message_start
    let start_payload = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": msg_id.clone(),
            "type": "message",
            "role": "assistant",
            "model": model_id.clone(),
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": { "input_tokens": input_tokens, "output_tokens": 0 },
        }
    });
    let _ = tx.send(Event::default().event("message_start").data(start_payload.to_string()));

    // content_block_start (text, index=0)
    let block_start_payload = serde_json::json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "text", "text": "" },
    });
    let _ = tx.send(Event::default().event("content_block_start").data(block_start_payload.to_string()));

    tokio::task::spawn_blocking(move || {
        let mut token_count = 0usize;

        engine.generate_stream(
            &input_ids,
            max_tokens,
            temperature,
            top_p,
            |tok_id| {
                token_count += 1;
                let piece = tokenizer.decode(&[tok_id]);
                if piece.is_empty() {
                    // Still keep the connection alive even if a token
                    // decoded to an empty byte chunk (special tokens).
                    return true;
                }
                let delta_payload = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": piece },
                });
                let ev = Event::default()
                    .event("content_block_delta")
                    .data(delta_payload.to_string());
                tx.send(ev).is_ok()
            },
        );

        // content_block_stop(0)
        let block_stop_payload = serde_json::json!({
            "type": "content_block_stop",
            "index": 0,
        });
        let _ = tx.send(
            Event::default()
                .event("content_block_stop")
                .data(block_stop_payload.to_string()),
        );

        // message_delta (final stop_reason + usage)
        //
        // Streaming mode cannot run the full tool_use parse mid-stream
        // without buffering — so we classify terminal reason by whether
        // we hit max_tokens vs a natural EOS here. Clients that need the
        // tool_use structured content should reassemble the text deltas
        // and re-parse, or use the non-streaming endpoint.
        let stop_reason = if token_count >= max_tokens { "max_tokens" } else { "end_turn" };
        let msg_delta_payload = serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": token_count },
        });
        let _ = tx.send(
            Event::default()
                .event("message_delta")
                .data(msg_delta_payload.to_string()),
        );

        // message_stop — final event
        let _ = tx.send(
            Event::default()
                .event("message_stop")
                .data(serde_json::json!({ "type": "message_stop" }).to_string()),
        );
    });

    UnboundedReceiverStream::new(rx).map(Ok)
}

// =============================================================================
// Prompt construction
// =============================================================================

/// The tool-use instruction we inject into the system prompt when the
/// request has tools. Kept simple and format-stable so the regex parser
/// below doesn't need to change.
fn tool_use_system_preamble(tools: &[ToolDefinition]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    // Minimal surface: one format block, one schema-key requirement, one
    // tool listing. Reasoning models (R1/QwQ) narrate the delimiters if
    // the preamble discusses them at length — the parser strips content
    // before `</think>` to handle what leaks through, but reducing
    // narrative footprint here keeps the false-positive rate low for
    // non-reasoning models too.
    let mut s = String::new();
    s.push_str("Tool invocation format (strict):\n\n");
    s.push_str("<tool_use>\n{\"name\": \"<tool_name>\", \"input\": {<args>}}\n</tool_use>\n\n");
    s.push_str("Rules: the JSON body must have both keys `name` (string) and `input` (object). `name` must be one of the tools below. `input` must match that tool's input_schema. Call at most one tool per turn; stop generating after the closing tag. If no tool is needed, answer normally without the tags.\n\n");
    s.push_str("Tools:\n");
    for t in tools {
        s.push_str("- ");
        s.push_str(&t.name);
        if let Some(d) = &t.description {
            s.push_str(": ");
            s.push_str(d);
        }
        if let Some(schema) = &t.input_schema {
            s.push_str("\n  input_schema: ");
            s.push_str(&schema.to_string());
        }
        s.push('\n');
    }
    Some(s)
}

/// Render the request as a chat-template-shaped prompt. For BitNet we use
/// LLaMA-3 headers; other architectures fall through to ChatML.
fn build_prompt(architecture: &str, req: &MessagesRequest) -> String {
    let base_system = req.system.as_ref().map(|s| s.to_text()).unwrap_or_default();
    let tool_system = tool_use_system_preamble(&req.tools);
    let full_system = match (base_system.is_empty(), tool_system) {
        (true, None) => String::new(),
        (true, Some(t)) => t,
        (false, None) => base_system,
        (false, Some(t)) => format!("{base_system}\n\n{t}"),
    };

    let is_llama3 = architecture.starts_with("bitnet")
        || architecture == "llama3"
        || architecture.starts_with("llama-3");
    if is_llama3 {
        render_llama3(&full_system, &req.messages)
    } else {
        render_chatml(&full_system, &req.messages)
    }
}

fn render_llama3(system: &str, messages: &[MessagesMessage]) -> String {
    let mut out = String::new();
    out.push_str("<|begin_of_text|>");
    if !system.is_empty() {
        out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        out.push_str(system);
        out.push_str("<|eot_id|>");
    }
    for m in messages {
        out.push_str("<|start_header_id|>");
        out.push_str(&m.role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(&flatten_message_content(&m.content));
        out.push_str("<|eot_id|>");
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn render_chatml(system: &str, messages: &[MessagesMessage]) -> String {
    let mut out = String::new();
    if !system.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(&m.role);
        out.push('\n');
        out.push_str(&flatten_message_content(&m.content));
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Reduce a user-side content field (plain string or blocks) to a single
/// string to feed to the tokenizer. Tool results are rendered as a stable
/// "TOOL_RESULT <id>: <content>" line so the model can distinguish them
/// from user chat. This is the handshake side of the text-tag scheme.
fn flatten_message_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Empty => String::new(),
        MessageContent::Blocks(blocks) => {
            let mut out = String::new();
            for (i, b) in blocks.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                match b {
                    ContentBlock::Text { text } => out.push_str(text),
                    ContentBlock::ToolUse { id, name, input } => {
                        out.push_str(&format!("<tool_use id=\"{id}\" name=\"{name}\">\n"));
                        out.push_str(&input.to_string());
                        out.push_str("\n</tool_use>");
                    }
                    ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                        let err = is_error.unwrap_or(false);
                        out.push_str(&format!("<tool_result tool_use_id=\"{tool_use_id}\" is_error=\"{err}\">\n"));
                        out.push_str(&content.to_string());
                        out.push_str("\n</tool_result>");
                    }
                }
            }
            out
        }
    }
}

// =============================================================================
// Output parsing: raw text -> content blocks
// =============================================================================

/// Split assistant output into text + tool_use blocks.
///
/// Tool-use format the model should emit:
///
/// ```text
/// <tool_use>
/// {"name": "X", "input": {...}}
/// </tool_use>
/// ```
///
/// - Leading prose before a tool_use becomes a `Text` block.
/// - Each complete `<tool_use>...</tool_use>` pair becomes a `ToolUse`
///   block; any unparseable JSON inside falls back to an error text
///   block so the client can retry.
/// - `stop_reason` is `"tool_use"` when at least one tool block was
///   emitted, `"max_tokens"` when we hit the cap, otherwise `"end_turn"`.
fn parse_assistant_output(raw: &str, output_tokens: usize, max_tokens: usize)
    -> (Vec<ContentBlock>, String)
{
    let mut blocks: Vec<ContentBlock> = Vec::new();
    let mut had_tool_use = false;

    // Reasoning-model guard: R1-Distill / QwQ / o1-style models emit a
    // `<think>...</think>` block containing internal chain-of-thought
    // before their actual answer. They routinely quote the `<tool_use>`
    // delimiters inside that block while explaining the format to
    // themselves, which produced false-positive tool_use parses (e.g.
    // body="` and `" from a sentence like "place it between the ` and `
    // delimiters"). Only content AFTER the final `</think>` counts as
    // the assistant's actual output. For non-reasoning models no
    // `</think>` is present and behavior is unchanged.
    let text = if let Some(idx) = raw.rfind("</think>") {
        &raw[idx + "</think>".len()..]
    } else {
        raw
    };
    let mut cursor = 0usize;
    while let Some(start) = text[cursor..].find("<tool_use>") {
        let abs_start = cursor + start;
        // Emit preceding prose as a text block (trimmed of trailing whitespace).
        let lead = &text[cursor..abs_start];
        let trimmed = lead.trim();
        if !trimmed.is_empty() {
            blocks.push(ContentBlock::Text { text: trimmed.to_string() });
        }
        // Find the closing tag.
        let body_start = abs_start + "<tool_use>".len();
        let Some(end_rel) = text[body_start..].find("</tool_use>") else {
            // Unclosed tool_use — emit the whole remainder as text and bail.
            let remainder = &text[abs_start..];
            blocks.push(ContentBlock::Text { text: remainder.to_string() });
            cursor = text.len();
            break;
        };
        let body_end = body_start + end_rel;
        let body = text[body_start..body_end].trim();

        // Parse the JSON body.
        match serde_json::from_str::<ToolCallJson>(body) {
            Ok(call) => {
                had_tool_use = true;
                let id = format!(
                    "toolu_{:x}",
                    (SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64)
                        ^ (blocks.len() as u64 * 0x9E3779B97F4A7C15),
                );
                blocks.push(ContentBlock::ToolUse {
                    id,
                    name: call.name,
                    input: call.input.unwrap_or(serde_json::Value::Null),
                });
                // Anthropic semantics: stop at the first tool call.
                cursor = body_end + "</tool_use>".len();
                break;
            }
            Err(e) => {
                // Unparseable — keep as text so the user can see what the
                // model produced and why we rejected it.
                blocks.push(ContentBlock::Text {
                    text: format!(
                        "[nexus-serve: failed to parse tool_use body as JSON: {e}; raw body: {body}]"
                    ),
                });
                cursor = body_end + "</tool_use>".len();
            }
        }
    }

    // Emit trailing text only when we did NOT emit a tool_use. Anthropic
    // semantics: the first tool_use terminates the assistant turn, so
    // anything the model generated after it is dropped on the floor.
    if !had_tool_use && cursor < text.len() {
        let tail = text[cursor..].trim();
        if !tail.is_empty() {
            blocks.push(ContentBlock::Text { text: tail.to_string() });
        }
    }

    // If we produced nothing, keep the raw text as a single text block so
    // the client always gets content.
    if blocks.is_empty() {
        blocks.push(ContentBlock::Text { text: raw.trim().to_string() });
    }

    let stop_reason = if had_tool_use {
        "tool_use"
    } else if output_tokens >= max_tokens {
        "max_tokens"
    } else {
        "end_turn"
    }
    .to_string();
    (blocks, stop_reason)
}

#[derive(Debug, Deserialize)]
struct ToolCallJson {
    name: String,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

// =============================================================================
// Helpers
// =============================================================================

async fn resolve_model_id(state: &AppState, requested: Option<&str>) -> Option<String> {
    let engines = state.engines.read().await;
    // Requested name first (direct hit or alias); then first loaded model.
    if let Some(name) = requested {
        if engines.contains_key(name) {
            return Some(name.to_string());
        }
        if let Some(resolved) = state.registry.resolve(name).await {
            if engines.contains_key(&resolved) {
                return Some(resolved);
            }
        }
    }
    engines.keys().next().cloned()
}

fn err_response(status: StatusCode, msg: &str, code: &str) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": { "type": code, "message": msg },
    });
    (status, Json(body)).into_response()
}

fn default_max_tokens() -> usize { 1024 }
fn default_temperature() -> f32 { 1.0 }

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_text_is_end_turn() {
        let (blocks, stop) = parse_assistant_output("Hello, world!", 3, 100);
        assert_eq!(stop, "end_turn");
        assert_eq!(blocks.len(), 1);
        matches!(&blocks[0], ContentBlock::Text { text } if text == "Hello, world!");
    }

    #[test]
    fn parse_hits_max_tokens() {
        let (_blocks, stop) = parse_assistant_output("Hello", 100, 100);
        assert_eq!(stop, "max_tokens");
    }

    #[test]
    fn parse_single_tool_use() {
        let raw = r#"I'll check that.
<tool_use>
{"name": "read_file", "input": {"path": "/etc/hosts"}}
</tool_use>"#;
        let (blocks, stop) = parse_assistant_output(raw, 20, 100);
        assert_eq!(stop, "tool_use");
        assert_eq!(blocks.len(), 2);
        matches!(&blocks[0], ContentBlock::Text { text } if text == "I'll check that.");
        match &blocks[1] {
            ContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "read_file");
                assert_eq!(input.get("path").and_then(|v| v.as_str()), Some("/etc/hosts"));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn parse_tool_use_stops_at_first_call() {
        // A second tool_use after the first should be ignored per Anthropic
        // semantics.
        let raw = r#"<tool_use>{"name":"a","input":{}}</tool_use>
then text
<tool_use>{"name":"b","input":{}}</tool_use>"#;
        let (blocks, stop) = parse_assistant_output(raw, 40, 100);
        assert_eq!(stop, "tool_use");
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse { name, .. } => assert_eq!(name, "a"),
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn parse_reasoning_model_ignores_tool_use_inside_think_block() {
        // R1-Distill style: the model narrates the tool_use format inside
        // its <think> block, which used to trigger false-positive
        // tool_use matches. Anything before the final </think> must be
        // skipped by the parser.
        let raw = r#"<think>
I should call read_file. The format is <tool_use>{"name":"read_file","input":{"path":"/etc/hosts"}}</tool_use>
</think>
<tool_use>
{"name": "read_file", "input": {"path": "/etc/hosts"}}
</tool_use>"#;
        let (blocks, stop) = parse_assistant_output(raw, 40, 100);
        assert_eq!(stop, "tool_use", "post-think tool_use must win");
        // Exactly one ToolUse, no false-positive text blocks from the think body.
        let tool_blocks: Vec<_> = blocks.iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .collect();
        assert_eq!(tool_blocks.len(), 1, "expected 1 tool_use, got {blocks:?}");
        match tool_blocks[0] {
            ContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "read_file");
                assert_eq!(input.get("path").and_then(|v| v.as_str()), Some("/etc/hosts"));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn parse_reasoning_model_no_post_think_tool_call_is_end_turn() {
        // Think block mentions the tags but final answer is plain text.
        let raw = "<think>Maybe I need <tool_use>...</tool_use> — no, I'll just answer.</think>\nThe capital is Paris.";
        let (blocks, stop) = parse_assistant_output(raw, 30, 100);
        assert_eq!(stop, "end_turn");
        assert_eq!(blocks.len(), 1);
        matches!(&blocks[0], ContentBlock::Text { text } if text == "The capital is Paris.");
    }

    #[test]
    fn parse_malformed_tool_use_falls_back_to_text() {
        let raw = "<tool_use>not json</tool_use>";
        let (blocks, stop) = parse_assistant_output(raw, 5, 100);
        // Malformed tool_use → text block with the error + end_turn
        // (since no successful tool call happened).
        assert_eq!(stop, "end_turn");
        assert!(blocks.iter().any(|b| matches!(b, ContentBlock::Text { text } if text.contains("failed to parse"))));
    }
}
