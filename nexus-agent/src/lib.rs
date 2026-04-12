//! nexus-agent — Autonomous AI agent framework
//!
//! Core types: Agent, Tool, Memory, and the ReAct execution loop.
//! LLM inference is provided by nexus-serve (local, OpenAI-compatible API)
//! running Gemma 4, Qwen 3, or any GGUF model.

pub mod agents;
pub mod backend;
pub mod tools;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Core types
// =============================================================================

/// A single message in the conversation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String, // "system", "user", "assistant", "tool"
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String, // "function"
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Definition of a tool the LLM can invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String, // "function"
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub output: String,
    pub success: bool,
}

// =============================================================================
// Tool trait
// =============================================================================

/// A tool that an agent can invoke. Each tool has a name, a JSON Schema for
/// its parameters, and an async execute method.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must match what the LLM emits in tool_calls).
    fn name(&self) -> &str;

    /// Human-readable description for the LLM's system prompt.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given JSON arguments string.
    async fn execute(&self, arguments: &str) -> anyhow::Result<String>;

    /// Build the OpenAI-format tool definition.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: self.name().to_string(),
                description: self.description().to_string(),
                parameters: self.parameters_schema(),
            },
        }
    }
}

// =============================================================================
// Tool registry
// =============================================================================

/// Registry of available tools, indexed by name.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    pub async fn execute(&self, name: &str, arguments: &str) -> anyhow::Result<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {name}"))?;
        tool.execute(arguments).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Memory trait
// =============================================================================

/// Persistent memory for an agent across invocations.
#[async_trait]
pub trait Memory: Send + Sync {
    /// Store a key-value pair.
    async fn store(&self, key: &str, value: &str) -> anyhow::Result<()>;

    /// Retrieve a value by key.
    async fn recall(&self, key: &str) -> anyhow::Result<Option<String>>;

    /// Search memory by a query string (semantic or keyword).
    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<(String, String)>>;
}

/// Simple file-backed memory (JSON on disk).
pub struct FileMemory {
    path: std::path::PathBuf,
}

impl FileMemory {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }

    fn load(&self) -> HashMap<String, String> {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    fn save(&self, data: &HashMap<String, String>) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, serde_json::to_string_pretty(data)?)?;
        Ok(())
    }
}

#[async_trait]
impl Memory for FileMemory {
    async fn store(&self, key: &str, value: &str) -> anyhow::Result<()> {
        let mut data = self.load();
        data.insert(key.to_string(), value.to_string());
        self.save(&data)
    }

    async fn recall(&self, key: &str) -> anyhow::Result<Option<String>> {
        Ok(self.load().get(key).cloned())
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<(String, String)>> {
        let data = self.load();
        let query_lower = query.to_lowercase();
        let mut results: Vec<(String, String)> = data
            .into_iter()
            .filter(|(k, v)| {
                k.to_lowercase().contains(&query_lower)
                    || v.to_lowercase().contains(&query_lower)
            })
            .collect();
        results.truncate(limit);
        Ok(results)
    }
}

// =============================================================================
// ReAct execution loop
// =============================================================================

/// Configuration for the ReAct loop.
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_iterations: usize,
    pub model: String,
    pub temperature: f32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_iterations: 20,
            model: "qwen3".to_string(),
            temperature: 0.3,
        }
    }
}

/// Run the ReAct loop: LLM thinks → calls tools → observes results → repeats.
///
/// Returns the final text response from the LLM.
pub async fn react_loop(
    backend: &dyn backend::LlmBackend,
    tools: &ToolRegistry,
    config: &AgentConfig,
    user_message: &str,
) -> anyhow::Result<String> {
    let mut messages = vec![
        Message {
            role: "system".to_string(),
            content: config.system_prompt.clone(),
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: "user".to_string(),
            content: user_message.to_string(),
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    let tool_defs = tools.definitions();

    for iteration in 0..config.max_iterations {
        tracing::debug!("ReAct iteration {}/{}", iteration + 1, config.max_iterations);

        // Ask the LLM
        let response = backend
            .chat_completion(&messages, &tool_defs, &config.model, config.temperature)
            .await?;

        // If the LLM returned tool calls, execute them
        if let Some(ref tool_calls) = response.tool_calls {
            // Add the assistant message (with tool_calls) to context
            messages.push(response.clone());

            for tc in tool_calls {
                tracing::info!("Tool call: {} ({})", tc.function.name, tc.id);
                let output = match tools.execute(&tc.function.name, &tc.function.arguments).await {
                    Ok(out) => out,
                    Err(e) => format!("Error: {e}"),
                };
                tracing::debug!("Tool result: {}...", &output[..output.len().min(200)]);

                // Add tool result to context
                messages.push(Message {
                    role: "tool".to_string(),
                    content: output,
                    tool_call_id: Some(tc.id.clone()),
                    tool_calls: None,
                });
            }
        } else {
            // No tool calls — this is the final answer
            return Ok(response.content);
        }
    }

    anyhow::bail!("ReAct loop exceeded max iterations ({})", config.max_iterations)
}
