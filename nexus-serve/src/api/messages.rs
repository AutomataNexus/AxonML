//! messages — Anthropic Messages API (`POST /v1/messages`)
//!
//! Implements the request/response shape of Anthropic's Messages API so
//! nexus-agent (and any Claude-SDK-compatible client) can talk to nexus-serve
//! with one URL swap.
//!
//! Types: [`MessagesRequest`], [`MessagesMessage`], [`MessageContent`]
//! (untagged enum: plain string or list of [`ContentBlock`]s), [`SystemField`]
//! (same untagged shape for the system field), [`ToolDefinition`],
//! [`MessagesResponse`], [`MessagesUsage`], and [`ToolCallJson`] (internal
//! deserializer for tool call bodies).
//!
//! Handlers and helpers: [`messages`] (axum handler),
//! [`build_messages_stream`] (SSE streaming via `UnboundedReceiverStream`),
//! [`build_prompt`] / [`render_llama3`] / [`render_chatml`] (prompt shaping),
//! [`tool_use_system_preamble`] (tool instruction injection),
//! [`flatten_message_content`] (content-block → string reducer),
//! [`parse_assistant_output`] (the core text → `ContentBlock` vec + stop
//! reason parser, with a `</think>` guard for reasoning models),
//! [`resolve_model_id`], and [`err_response`].
//!
//! Key shape differences from OpenAI's `/v1/chat/completions`:
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
//!
//! # File
//! `nexus-serve/src/api/messages.rs`
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
    let prompt = build_prompt(engine.architecture(), engine.model_name(), &req);
    let input_ids = tokenizer.encode(&prompt);
    let input_tokens = input_ids.len();
    let msg_id = format!("msg_{now}");

    let stop_strs = compute_stop_strings(&req);

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
            stop_strs,
        );
        return Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response());
    }
    let (generated_ids, stop_hit) = generate_with_stop_strings(
        engine.as_ref(),
        tokenizer.as_ref(),
        &input_ids,
        req.max_tokens,
        req.temperature,
        req.top_p.unwrap_or(0.9),
        &stop_strs,
    );
    let output_tokens = generated_ids.len();
    let raw = tokenizer.decode(&generated_ids);

    // Parse the raw text into content blocks, tool_use-aware.
    let (content, parse_stop_reason) =
        parse_assistant_output(&raw, output_tokens, req.max_tokens);

    // Stop-reason precedence: tool_use > max_tokens > stop_sequence > end_turn.
    // The parser already picked tool_use / max_tokens / end_turn. If we
    // actually tripped a user-supplied stop_sequence (not an auto-stop),
    // promote to "stop_sequence" and report which string matched. Auto-stops
    // like `</tool_use>` remain reported as "tool_use".
    let (stop_reason, stop_sequence) = match (parse_stop_reason.as_str(), &stop_hit) {
        ("tool_use", _) => ("tool_use".to_string(), None),
        ("max_tokens", _) => ("max_tokens".to_string(), None),
        (_, Some(s)) if req.stop_sequences.iter().any(|u| u == s) => {
            ("stop_sequence".to_string(), Some(s.clone()))
        }
        _ => (parse_stop_reason, None),
    };

    let body = Json(MessagesResponse {
        id: msg_id,
        response_type: "message",
        role: "assistant",
        model: model_id,
        content,
        stop_reason,
        stop_sequence,
        usage: MessagesUsage { input_tokens, output_tokens },
    });
    Ok(body.into_response())
}

// =============================================================================
// Stop-string machinery
// =============================================================================

/// Build the full list of stop strings to honour for a given request:
///
/// 1. The user's `stop_sequences` as-is (preserved so we can report exactly
///    which one matched in `stop_sequence`).
/// 2. Auto-stops when the request carries tools. These catch the three
///    post-tool-call leakage patterns R1-Distill-Qwen emits:
///       - `</tool_use>` — preamble-taught close tag. Once the JSON body
///         is delimited the call is complete; further tokens are post-hoc
///         narration ("Now I'll explain what I did...") or, worse,
///         hallucinated tool RESULTS.
///       - ```` ``` ```` — closing fence of the OpenAI-style JSON code
///         block. Same reason.
///       - `</{tool_name}>` for each concrete tool in the request — the
///         Qwen-style named-tag dialect (`<read_file>{...}</read_file>`).
/// 3. `<|im_start|>` and `<|im_end|>` as a belt-and-suspenders string-level
///    catch in case those tokens get decoded into the rolling buffer
///    despite being caught by the token-id stop set (e.g. if the tokenizer
///    ever decodes a containing super-token that happens to end with them).
fn compute_stop_strings(req: &MessagesRequest) -> Vec<String> {
    let mut out = Vec::with_capacity(req.stop_sequences.len() + req.tools.len() + 4);
    out.extend(req.stop_sequences.iter().cloned());
    if !req.tools.is_empty() {
        out.push("</tool_use>".to_string());
        out.push("```\n".to_string());
        for t in &req.tools {
            out.push(format!("</{}>", t.name));
        }
    }
    out.push("<|im_start|>".to_string());
    out.push("<|im_end|>".to_string());
    out
}

/// Trim the rolling window to at most `keep` bytes from the end,
/// backing up to a valid UTF-8 boundary so we don't slice inside a code
/// point. Idempotent if the buffer is already small enough.
fn trim_rolling_window(buf: &mut String, keep: usize) {
    if buf.len() <= keep {
        return;
    }
    let mut cut = buf.len() - keep;
    while cut < buf.len() && !buf.is_char_boundary(cut) {
        cut += 1;
    }
    buf.replace_range(..cut, "");
}

/// Drive `engine.generate_stream` but stop as soon as any of `stop_strings`
/// appears as a suffix of the decoded-so-far text. Returns the token IDs
/// generated so far and, if we tripped a stop string, a copy of the string
/// that matched (for `stop_sequence` reporting).
///
/// This is the non-streaming analog of the per-token callback used in
/// `build_messages_stream`. Keeping both paths going through the same
/// stop-string rules means a tool-using client gets the same behaviour in
/// streaming and non-streaming mode.
fn generate_with_stop_strings(
    engine: &crate::model::inference::InferenceEngine,
    tokenizer: &crate::tokenizer::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    stop_strings: &[String],
) -> (Vec<u32>, Option<String>) {
    let mut ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut rolling = String::new();
    let mut stop_hit: Option<String> = None;
    // Window large enough to contain any reasonable stop string plus a
    // comfortable overlap for multi-byte decoding.
    const MAX_ROLLING_BYTES: usize = 512;
    const KEEP_ROLLING_BYTES: usize = 256;

    engine.generate_stream(input_ids, max_tokens, temperature, top_p, |tok_id| {
        ids.push(tok_id);
        let piece = tokenizer.decode(&[tok_id]);
        rolling.push_str(&piece);
        if rolling.len() > MAX_ROLLING_BYTES {
            trim_rolling_window(&mut rolling, KEEP_ROLLING_BYTES);
        }
        for s in stop_strings {
            if !s.is_empty() && rolling.contains(s.as_str()) {
                stop_hit = Some(s.clone());
                return false;
            }
        }
        true
    });

    (ids, stop_hit)
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
    stop_strings: Vec<String>,
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
        let mut rolling = String::new();
        const MAX_ROLLING_BYTES: usize = 512;
        const KEEP_ROLLING_BYTES: usize = 256;

        engine.generate_stream(
            &input_ids,
            max_tokens,
            temperature,
            top_p,
            |tok_id| {
                token_count += 1;
                let piece = tokenizer.decode(&[tok_id]);
                // Even empty-decoding tokens (special tokens) extend the
                // rolling window conceptually, but there's nothing to
                // contribute to the stop-match buffer. Still keep the
                // stream alive for them.
                if piece.is_empty() {
                    return true;
                }
                rolling.push_str(&piece);
                if rolling.len() > MAX_ROLLING_BYTES {
                    trim_rolling_window(&mut rolling, KEEP_ROLLING_BYTES);
                }
                // Check stop strings against the rolling window BEFORE
                // emitting this token. Matching emits the last piece
                // (which may contain the match tail) then halts — this
                // matches non-streaming behaviour, where the final
                // decoded text still carries the stop string and the
                // tool-use parser strips post-match content.
                let delta_payload = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": piece },
                });
                let ev = Event::default()
                    .event("content_block_delta")
                    .data(delta_payload.to_string());
                if tx.send(ev).is_err() {
                    return false;
                }
                for s in &stop_strings {
                    if !s.is_empty() && rolling.contains(s.as_str()) {
                        return false;
                    }
                }
                true
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
    // anti-hallucination directive, a single one-shot example, and the
    // tool listing. Reasoning models (R1/QwQ) narrate the delimiters if
    // the preamble discusses them at length — the parser strips content
    // before `</think>` to handle what leaks through, but reducing
    // narrative footprint here keeps the false-positive rate low for
    // non-reasoning models too.
    //
    // The anti-hallucination line targets R1-Distill-Qwen's habit of
    // guessing file contents / command output when a task *looks*
    // predictable ("list the files in /etc" → it writes out a plausible
    // /etc listing without ever calling the tool). Phrasing it as a hard
    // rule, immediately before the one-shot, is what got that failure
    // mode to drop out on the code-agent smoke prompts.
    //
    // The one-shot shows a complete three-turn round: assistant emits
    // `<tool_use>`, user returns `tool_result`, assistant gives the final
    // answer. Models distilled from chat data often need the
    // `tool_result` shape demonstrated explicitly before they'll trust
    // it is a real observation rather than another prompt the user is
    // showing them.
    let mut s = String::new();
    s.push_str("Tool invocation format (strict):\n\n");
    s.push_str("<tool_use>\n{\"name\": \"<tool_name>\", \"input\": {<args>}}\n</tool_use>\n\n");
    s.push_str("Rules:\n");
    s.push_str("- The JSON body must have both keys `name` (string) and `input` (object). `name` must be one of the tools below. `input` must match that tool's input_schema.\n");
    s.push_str("- Call at most one tool per turn. Stop generating immediately after the closing `</tool_use>` tag.\n");
    s.push_str("- NEVER invent the contents of a file, the output of a command, or any other tool result. If the answer depends on data a tool can fetch, call the tool — do not guess.\n");
    s.push_str("- When you have enough information to answer directly, answer normally with no tags.\n\n");
    s.push_str("Example (assistant turn that calls a tool, then final answer after the tool_result comes back):\n\n");
    s.push_str("assistant:\nI'll check the file.\n<tool_use>\n{\"name\": \"read_file\", \"input\": {\"path\": \"/etc/hostname\"}}\n</tool_use>\n\n");
    s.push_str("user (tool_result): nexus-dev\n\n");
    s.push_str("assistant:\nThe hostname is `nexus-dev`.\n\n");
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
fn build_prompt(architecture: &str, model_name: &str, req: &MessagesRequest) -> String {
    let base_system = req.system.as_ref().map(|s| s.to_text()).unwrap_or_default();
    let tool_system = tool_use_system_preamble(&req.tools);
    let full_system = match (base_system.is_empty(), tool_system) {
        (true, None) => String::new(),
        (true, Some(t)) => t,
        (false, None) => base_system,
        (false, Some(t)) => format!("{base_system}\n\n{t}"),
    };

    // R1-Distill was fine-tuned with DeepSeek's own chat template, which
    // uses full-width-pipe tokens (`<｜User｜>`, `<｜Assistant｜>`,
    // `<｜begin▁of▁sentence｜>`) on top of the Qwen2 vocab. Rendering it as
    // ChatML produces degenerate output ("Okay00000..." style loops) because
    // the `<|im_start|>` / `<|im_end|>` tokens are either never learned or
    // learned as distractors in the SFT data. `general.architecture` is
    // plain "qwen2" — indistinguishable from base Qwen2-Instruct — so we
    // disambiguate on `general.name`.
    let is_deepseek_r1 = model_name.contains("DeepSeek")
        && (model_name.contains("R1") || model_name.contains("Distill"));
    let is_llama3 = architecture.starts_with("bitnet")
        || architecture == "llama3"
        || architecture.starts_with("llama-3");
    let is_phi3 = architecture == "phi3";
    if is_deepseek_r1 {
        render_deepseek_r1(&full_system, &req.messages)
    } else if is_phi3 {
        render_phi3(&full_system, &req.messages)
    } else if is_llama3 {
        render_llama3(&full_system, &req.messages)
    } else {
        render_chatml(&full_system, &req.messages)
    }
}

/// Render a prompt using Phi-3's chat template: `<|system|>`, `<|user|>`,
/// `<|assistant|>` role markers and `<|end|>` turn terminators, each on
/// its own line. Matches the Jinja template shipped in
/// `Phi-3-mini-4k-instruct/tokenizer_config.json`.
///
/// Phi-3's tokenizer has each of these as a single special token
/// (`<|system|>`=32006, `<|user|>`=32010, `<|assistant|>`=32001,
/// `<|end|>`=32007); the BPE path must keep them literal so the encoder
/// emits the single-token ID rather than splitting them into bytes.
///
/// **Phi-3 REQUIRES a system prompt.** Without one the model produces
/// fluent-but-off-topic garbage ("The Fire \" At colon interpretationalic…")
/// because its SFT pipeline always included a system turn. If the
/// caller didn't supply one, inject the generic "You are a helpful
/// assistant." default so the output distribution matches training.
fn render_phi3(system: &str, messages: &[MessagesMessage]) -> String {
    let mut out = String::new();
    let effective_system: &str = if system.is_empty() {
        "You are a helpful assistant."
    } else {
        system
    };
    out.push_str("<|system|>\n");
    out.push_str(effective_system);
    out.push_str("<|end|>\n");
    for m in messages {
        let role = match m.role.as_str() {
            "user" | "assistant" | "system" => m.role.as_str(),
            _ => "user",
        };
        out.push('<');
        out.push('|');
        out.push_str(role);
        out.push_str("|>\n");
        out.push_str(&flatten_message_content(&m.content));
        out.push_str("<|end|>\n");
    }
    out.push_str("<|assistant|>\n");
    out
}

/// Render a prompt in DeepSeek's R1-family chat template. Mirrors the Jinja
/// template embedded in R1-Distill-Qwen's `tokenizer_config.json`:
///
/// ```text
/// <｜begin▁of▁sentence｜>{system}<｜User｜>{u1}<｜Assistant｜>{a1}<｜end▁of▁sentence｜>
/// <｜User｜>{u2}<｜Assistant｜>
/// ```
///
/// Every literal tag uses full-width pipe `｜` (U+FF5C), NOT ASCII `|` —
/// the tokenizer maps the full-width variant to the single special-token
/// IDs 151643-151646 and maps ASCII-pipe variants to their raw byte
/// sequence, which the model was never trained on.
///
/// The system prompt (if any) sits inline immediately after BOS rather
/// than wrapped in its own tag, matching DeepSeek's Jinja template's
/// `{{ns.system_prompt}}` expansion point.
///
/// Assistant turns in history are terminated with `<｜end▁of▁sentence｜>`;
/// the trailing `<｜Assistant｜>` with no terminator is the generation-
/// prompt marker (equivalent to `add_generation_prompt=true`).
fn render_deepseek_r1(system: &str, messages: &[MessagesMessage]) -> String {
    let mut out = String::new();
    out.push_str("<｜begin▁of▁sentence｜>");
    if !system.is_empty() {
        out.push_str(system);
    }
    for m in messages {
        match m.role.as_str() {
            "user" => {
                out.push_str("<｜User｜>");
                out.push_str(&flatten_message_content(&m.content));
            }
            "assistant" => {
                out.push_str("<｜Assistant｜>");
                out.push_str(&flatten_message_content(&m.content));
                out.push_str("<｜end▁of▁sentence｜>");
            }
            _ => {
                // Fold unexpected roles (system already handled inline
                // above; tool-result style turns are flattened to string
                // by the caller into the user slot) into user-style.
                out.push_str("<｜User｜>");
                out.push_str(&flatten_message_content(&m.content));
            }
        }
    }
    out.push_str("<｜Assistant｜>");
    out
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

    // Find the earliest tool call in the text, trying the three formats
    // the model actually emits despite the preamble specifying only #1:
    //
    //   1. <tool_use>{"name":..,"input":..}</tool_use>   (preamble-taught)
    //   2. ```json\n{"name":..,"input":..}\n```          (OpenAI-style fence)
    //   3. <tool_name>{..json body..}</tool_name>        (Qwen-style tag)
    //
    // DeepSeek-R1-Distill-Qwen-7B was distilled from Qwen2.5 which saw
    // multiple tool-call conventions during instruction tuning; at
    // temperature 0.1 it picks one roughly uniformly. Accepting all
    // three makes tool use reliable without fine-tuning the base model.
    let candidate = find_tool_call(text);

    let mut blocks: Vec<ContentBlock> = Vec::new();
    let had_tool_use = if let Some(c) = candidate {
        // Everything before the tool call is prose.
        let lead = text[..c.lead_end].trim();
        if !lead.is_empty() {
            blocks.push(ContentBlock::Text { text: lead.to_string() });
        }
        let id = format!(
            "toolu_{:x}",
            (SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64)
                ^ (blocks.len() as u64 * 0x9E3779B97F4A7C15),
        );
        blocks.push(ContentBlock::ToolUse { id, name: c.name, input: c.input });
        true
    } else {
        // No tool call — whole text is the assistant's answer.
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            blocks.push(ContentBlock::Text { text: trimmed.to_string() });
        }
        false
    };

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

/// Result of a successful tool-call match in the assistant's raw output.
struct ToolCallMatch {
    /// Byte offset where the tool-call syntax starts (everything before
    /// this is prose).
    lead_end: usize,
    /// Resolved tool name (from the JSON body's `name` field or the XML
    /// tag name, depending on the format that matched).
    name: String,
    /// Tool input payload (from the JSON body's `input` field, or the
    /// whole JSON body when the name came from an XML tag).
    input: serde_json::Value,
}

/// Scan `text` for the earliest tool-call syntax in any of the three
/// supported formats. Returns the match with the smallest starting byte
/// offset so the leading prose is preserved correctly.
fn find_tool_call(text: &str) -> Option<ToolCallMatch> {
    let mut best: Option<ToolCallMatch> = None;

    if let Some(m) = find_tool_use_tag(text) {
        best = Some(m);
    }
    if let Some(m) = find_json_code_fence(text) {
        if best.as_ref().is_none_or(|b| m.lead_end < b.lead_end) {
            best = Some(m);
        }
    }
    if let Some(m) = find_named_tag(text) {
        if best.as_ref().is_none_or(|b| m.lead_end < b.lead_end) {
            best = Some(m);
        }
    }

    best
}

/// Match `<tool_use>{"name":..,"input":..}</tool_use>` — the format the
/// server's preamble teaches.
fn find_tool_use_tag(text: &str) -> Option<ToolCallMatch> {
    let start = text.find("<tool_use>")?;
    let body_start = start + "<tool_use>".len();
    let end_rel = text[body_start..].find("</tool_use>")?;
    let body = text[body_start..body_start + end_rel].trim();
    let call: ToolCallJson = serde_json::from_str(body).ok()?;
    Some(ToolCallMatch {
        lead_end: start,
        name: call.name,
        input: call.input.unwrap_or(serde_json::Value::Null),
    })
}

/// Match a fenced JSON code block containing `{"name":..,"input":..}`. The
/// fence can be ` ```json ... ``` ` or ` ``` ... ``` `. Used by OpenAI-tuned
/// models that emit "I'll call the foo tool" followed by a JSON fence.
fn find_json_code_fence(text: &str) -> Option<ToolCallMatch> {
    let fence_open = text.find("```")?;
    // Skip an optional language tag on the same line (e.g. "```json\n").
    let after = fence_open + 3;
    let newline = text[after..].find('\n')?;
    let body_start = after + newline + 1;
    let close_rel = text[body_start..].find("```")?;
    let body = text[body_start..body_start + close_rel].trim();
    // Must parse as the tool-call object shape; bare JSON values don't count.
    let call: ToolCallJson = serde_json::from_str(body).ok()?;
    Some(ToolCallMatch {
        lead_end: fence_open,
        name: call.name,
        input: call.input.unwrap_or(serde_json::Value::Null),
    })
}

/// Match `<tool_name>{..json..}</tool_name>` — the Qwen-style convention
/// where the XML tag itself carries the tool name and the body is the
/// tool input. Only accepts tag names that match `[a-z][a-z0-9_]*` (so
/// prose tags like `<think>`, `<quote>`, `<p>` don't match unless the
/// body is a JSON object, which they wouldn't be anyway).
fn find_named_tag(text: &str) -> Option<ToolCallMatch> {
    // Skip the `<tool_use>` form (handled elsewhere).
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let Some(rel) = text[i..].find('<') else { return None };
        let start = i + rel;
        let after_lt = start + 1;
        // Find the closing `>` of the opening tag.
        let close_gt_rel = text[after_lt..].find('>')?;
        let tag = &text[after_lt..after_lt + close_gt_rel];
        // Reject anything with whitespace or attributes (not a plain tag).
        if tag.is_empty() || tag.contains(' ') || tag.starts_with('/') {
            i = after_lt + close_gt_rel + 1;
            continue;
        }
        // Snake_case-only to avoid matching prose HTML like <P> or <Quote>.
        if !tag.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            || !tag.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
            || tag == "tool_use"
        {
            i = after_lt + close_gt_rel + 1;
            continue;
        }
        // Locate matching close tag.
        let body_start = after_lt + close_gt_rel + 1;
        let close_tag = format!("</{tag}>");
        let Some(close_rel) = text[body_start..].find(&close_tag) else {
            i = body_start;
            continue;
        };
        let body = text[body_start..body_start + close_rel].trim();
        // Body must be a JSON object; otherwise this is some random tag.
        let input: serde_json::Value = match serde_json::from_str::<serde_json::Value>(body) {
            Ok(v) if v.is_object() => v,
            _ => {
                i = body_start + close_rel + close_tag.len();
                continue;
            }
        };
        return Some(ToolCallMatch {
            lead_end: start,
            name: tag.to_string(),
            input,
        });
    }
    None
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
        // Malformed tool_use body isn't a recognized tool-call format, so
        // the whole output is preserved as a single text block and the
        // turn ends normally (no spurious tool_use stop).
        assert_eq!(stop, "end_turn");
        assert_eq!(blocks.len(), 1);
        matches!(&blocks[0], ContentBlock::Text { text } if text.contains("not json"));
    }

    #[test]
    fn parse_json_code_fence_as_tool_call() {
        // Some models emit the tool call wrapped in a ```json ... ``` fence
        // (OpenAI convention) instead of <tool_use> tags.
        let raw = "I'll read the file.\n\n```json\n{\"name\": \"read_file\", \"input\": {\"path\": \"/etc/hosts\"}}\n```";
        let (blocks, stop) = parse_assistant_output(raw, 30, 100);
        assert_eq!(stop, "tool_use");
        assert!(blocks.iter().any(|b| matches!(
            b,
            ContentBlock::ToolUse { name, input, .. }
                if name == "read_file" && input["path"] == "/etc/hosts"
        )));
    }

    #[test]
    fn parse_named_tag_as_tool_call() {
        // Qwen-style: the XML tag itself names the tool, body is the input.
        let raw = "<search_files>\n{\"query\": \"ModelBundle\", \"path\": \"/opt/AxonML\"}\n</search_files>";
        let (blocks, stop) = parse_assistant_output(raw, 20, 100);
        assert_eq!(stop, "tool_use");
        assert!(blocks.iter().any(|b| matches!(
            b,
            ContentBlock::ToolUse { name, input, .. }
                if name == "search_files" && input["query"] == "ModelBundle"
        )));
    }

    #[test]
    fn parse_named_tag_ignores_non_tool_prose() {
        // <think> blocks are stripped before parsing; prose HTML-ish tags
        // like <p>, <quote> don't contain JSON objects so the named-tag
        // matcher falls through.
        let raw = "<p>Here is the answer.</p>";
        let (blocks, stop) = parse_assistant_output(raw, 5, 100);
        assert_eq!(stop, "end_turn");
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { .. }));
    }

    #[test]
    fn parse_earliest_format_wins_when_multiple_present() {
        // If the model emits two candidate tool-call formats, the earliest
        // by byte offset is taken (so the prose before it is preserved).
        let raw = r#"Preamble.

<tool_use>
{"name": "first", "input": {"x": 1}}
</tool_use>

Then a later code fence:
```json
{"name": "second", "input": {"y": 2}}
```"#;
        let (blocks, stop) = parse_assistant_output(raw, 40, 100);
        assert_eq!(stop, "tool_use");
        // First tool_use "first" wins; "second" is dropped because the
        // assistant turn ends at the first tool call.
        assert!(blocks.iter().any(|b| matches!(
            b,
            ContentBlock::ToolUse { name, .. } if name == "first"
        )));
        assert!(!blocks.iter().any(|b| matches!(
            b,
            ContentBlock::ToolUse { name, .. } if name == "second"
        )));
    }
}
