//! Anthropic Messages API backend — calls nexus-serve's `/v1/messages`
//! (or the real Anthropic API) using the content-block / `tool_use`
//! schema.
//!
//! Internally nexus-agent represents messages in an OpenAI-shaped
//! `Message { role, content, tool_calls, tool_call_id }`. This backend
//! is the translation layer: we convert outbound to Anthropic content
//! blocks, and convert inbound `tool_use` blocks back into our internal
//! `ToolCall` form so the existing ReAct loop doesn't need to change.
//!
//! Hard rule: any nexus-agent ↔ nexus-serve tool call uses this format
//! end-to-end. The OpenAI backend (`backend::local::LocalBackend`) is
//! kept for legacy agents that were written against `/v1/chat/completions`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{FunctionCall, Message, ToolCall, ToolDefinition};

use super::LlmBackend;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11435";

/// Backend that speaks Anthropic's Messages API (`/v1/messages`).
///
/// Default points at nexus-serve on `http://127.0.0.1:11435`; the same
/// wire shape works against the real Anthropic API at
/// `https://api.anthropic.com` if you ever want to swap.
pub struct AnthropicBackend {
    client: reqwest::Client,
    base_url: String,
    /// Optional API key. Not required for nexus-serve, required for the
    /// real Anthropic API (passed as `x-api-key` header).
    api_key: Option<String>,
}

impl AnthropicBackend {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
        }
    }

    pub fn with_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

impl Default for AnthropicBackend {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Wire types (just enough to match Anthropic's Messages API)
// =============================================================================

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: usize,
    temperature: f32,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
}

#[derive(Serialize)]
struct WireTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Serialize)]
struct WireMessage {
    role: String, // "user" | "assistant"
    content: WireContent,
}

/// Either a plain-string content or a list of content blocks.
#[derive(Serialize)]
#[serde(untagged)]
enum WireContent {
    Text(String),
    Blocks(Vec<WireBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum WireBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Deserialize)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<ResponseBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    /// Forward-compatibility: any other content block type is ignored.
    #[serde(other)]
    Other,
}

// =============================================================================
// Internal → wire translation
// =============================================================================

/// Peel the system message off the top of the history and translate the
/// remaining messages into Anthropic's `messages` array. Tool results
/// (role="tool") become a `tool_result` content block inside a `user`
/// message; assistant tool_calls become `tool_use` blocks inside an
/// `assistant` message.
fn to_wire(messages: &[Message]) -> (Option<String>, Vec<WireMessage>) {
    let mut system: Option<String> = None;
    let mut wire: Vec<WireMessage> = Vec::new();

    for m in messages {
        match m.role.as_str() {
            "system" => {
                // Concatenate multiple system turns with a blank line.
                system = Some(match system.take() {
                    Some(prev) => format!("{prev}\n\n{}", m.content),
                    None => m.content.clone(),
                });
            }
            "tool" => {
                // Fold the tool result into a user-role message carrying
                // one tool_result content block. If the previous wire
                // message was also a user tool_result batch, append.
                let block = WireBlock::ToolResult {
                    tool_use_id: m.tool_call_id.clone().unwrap_or_default(),
                    content: m.content.clone(),
                    is_error: None,
                };
                match wire.last_mut() {
                    Some(WireMessage {
                        role,
                        content: WireContent::Blocks(blocks),
                    }) if role == "user" => {
                        blocks.push(block);
                    }
                    _ => {
                        wire.push(WireMessage {
                            role: "user".to_string(),
                            content: WireContent::Blocks(vec![block]),
                        });
                    }
                }
            }
            "assistant" => {
                let mut blocks: Vec<WireBlock> = Vec::new();
                if !m.content.is_empty() {
                    blocks.push(WireBlock::Text { text: m.content.clone() });
                }
                if let Some(tcs) = &m.tool_calls {
                    for tc in tcs {
                        let input: Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(Value::Null);
                        blocks.push(WireBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input,
                        });
                    }
                }
                let content = if blocks.len() == 1 {
                    if let Some(WireBlock::Text { text }) = blocks.first() {
                        WireContent::Text(text.clone())
                    } else {
                        WireContent::Blocks(blocks)
                    }
                } else {
                    WireContent::Blocks(blocks)
                };
                wire.push(WireMessage { role: "assistant".to_string(), content });
            }
            _ /* user / anything else */ => {
                wire.push(WireMessage {
                    role: "user".to_string(),
                    content: WireContent::Text(m.content.clone()),
                });
            }
        }
    }

    (system, wire)
}

fn tools_to_wire(tools: &[ToolDefinition]) -> Vec<WireTool> {
    tools
        .iter()
        .map(|t| WireTool {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            input_schema: t.function.parameters.clone(),
        })
        .collect()
}

fn response_to_message(resp: MessagesResponse) -> Message {
    let mut text_buf = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for block in resp.content {
        match block {
            ResponseBlock::Text { text } => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                text_buf.push_str(&text);
            }
            ResponseBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    r#type: "function".to_string(),
                    function: FunctionCall { name, arguments: input.to_string() },
                });
            }
            ResponseBlock::Other => {}
        }
    }
    Message {
        role: "assistant".to_string(),
        content: text_buf,
        tool_call_id: None,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    }
}

// =============================================================================
// LlmBackend impl
// =============================================================================

#[async_trait]
impl LlmBackend for AnthropicBackend {
    async fn chat_completion(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
    ) -> anyhow::Result<Message> {
        let (system, wire_messages) = to_wire(messages);
        // 256 is plenty for a single ReAct turn — a tool call is ~30 tokens
        // and a final answer is usually <200. Larger budgets burn wall
        // time on slow backends (BitNet-2B at 1-3 tok/s = minutes per turn
        // if we let it run to 1024). Callers can override via a future
        // config field if needed.
        let request = MessagesRequest {
            model,
            max_tokens: 512,
            temperature,
            messages: wire_messages,
            system,
            tools: tools_to_wire(tools),
        };

        let mut req = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .json(&request);
        if let Some(key) = &self.api_key {
            req = req
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01");
        }
        let response = req.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("{}/v1/messages returned {status}: {body}", self.base_url);
        }

        let resp: MessagesResponse = response.json().await?;
        Ok(response_to_message(resp))
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        // Anthropic API has no /v1/models endpoint in Messages API form;
        // nexus-serve does expose /v1/models (OpenAI shape) on the same
        // port, so reuse it.
        let response = self
            .client
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await?;
        #[derive(Deserialize)]
        struct M { id: String }
        #[derive(Deserialize)]
        struct Ms { data: Vec<M> }
        let models: Ms = response.json().await?;
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    #[test]
    fn system_turn_is_extracted() {
        let (system, wire) = to_wire(&[
            msg("system", "you are helpful"),
            msg("user", "hi"),
        ]);
        assert_eq!(system.as_deref(), Some("you are helpful"));
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "user");
    }

    #[test]
    fn tool_role_becomes_user_tool_result_block() {
        let mut assistant = msg("assistant", "I'll read it.");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "toolu_abc".into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: "{\"path\":\"/etc/hosts\"}".into(),
            },
        }]);
        let tool_result = Message {
            role: "tool".into(),
            content: "127.0.0.1 localhost".into(),
            tool_call_id: Some("toolu_abc".into()),
            tool_calls: None,
        };
        let (_sys, wire) = to_wire(&[msg("user", "read /etc/hosts"), assistant, tool_result]);
        // Expected: [user("read /etc/hosts"), assistant(Blocks[text,tool_use]), user(Blocks[tool_result])]
        assert_eq!(wire.len(), 3);
        assert_eq!(wire[2].role, "user");
        match &wire[2].content {
            WireContent::Blocks(bs) => {
                assert_eq!(bs.len(), 1);
                assert!(matches!(&bs[0], WireBlock::ToolResult { tool_use_id, .. } if tool_use_id == "toolu_abc"));
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn response_text_blocks_concatenate() {
        let resp = MessagesResponse {
            content: vec![
                ResponseBlock::Text { text: "Hello".into() },
                ResponseBlock::Text { text: "world".into() },
            ],
            stop_reason: Some("end_turn".into()),
        };
        let m = response_to_message(resp);
        assert_eq!(m.content, "Hello\nworld");
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn response_tool_use_becomes_tool_call() {
        let resp = MessagesResponse {
            content: vec![ResponseBlock::ToolUse {
                id: "toolu_xyz".into(),
                name: "shell".into(),
                input: serde_json::json!({"cmd": "ls"}),
            }],
            stop_reason: Some("tool_use".into()),
        };
        let m = response_to_message(resp);
        let tcs = m.tool_calls.expect("tool_calls set");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "toolu_xyz");
        assert_eq!(tcs[0].function.name, "shell");
        assert!(tcs[0].function.arguments.contains("\"cmd\":\"ls\""));
    }
}
