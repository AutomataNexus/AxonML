//! Code Agent — Local Agentic Coder Powered By nexus-serve
//!
//! Defines the `code` agent configuration: an agentic coder specialized for
//! Andrew Jewell's engineering workspace. Runs through nexus-serve's
//! Anthropic Messages API (`/v1/messages`) so tool calls round-trip as
//! native `tool_use` / `tool_result` content blocks with
//! `stop_reason = "tool_use"`.
//!
//! Exports:
//! - `SYSTEM_PROMPT` — short ROLE-only prompt (tool FORMAT is injected by
//!   the server preamble when `tools[]` is non-empty).
//! - `config()` — returns an `AgentConfig` with
//!   `max_iterations = 12`, `model = "deepseek"`, `temperature = 0.1`,
//!   tuned for DeepSeek-R1-Distill-Qwen-7B at ~1.6 tok/s.
//!
//! The prompt explicitly suppresses R1-Distill's chain-of-thought narration
//! ("think quietly") so a 10-step loop remains usable.
//!
//! # File
//! `nexus-agent/src/agents/code.rs`
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
//!
//! Default invocation (DeepSeek-R1-Distill-Qwen-7B on :11436):
//!   nexus-agent --url http://127.0.0.1:11436 --anthropic code \
//!     "Find the TODO in /opt/AxonML/llm-training and write a one-line plan"
//!
//! ## Why the system prompt is short
//!
//! The `/v1/messages` server-side preamble (in `nexus-serve::api::messages::
//! tool_use_system_preamble`) already teaches the model the `<tool_use>`
//! delimiter format and schema. Duplicating that instruction here doubled
//! the narrative surface and confused reasoning models (R1 / QwQ) into
//! narrating the format back in their `<think>` block. This prompt only
//! defines the agent's ROLE and guardrails; the FORMAT is injected by the
//! server when `tools[]` is non-empty.
//!
//! ## Why "think quietly"
//!
//! R1-Distill's chain-of-thought default eats 200-500 tokens per turn on
//! narration before the actual tool call. At 1.6 tok/s that's 2-5 minutes
//! of overhead per step — a 10-step agent loop becomes unusable. The
//! prompt explicitly suppresses narration so the model gets to the tool
//! call directly.

// =============================================================================
// Imports
// =============================================================================

use crate::AgentConfig;

// =============================================================================
// System Prompt
// =============================================================================

pub const SYSTEM_PROMPT: &str = r#"You are the Code agent for Andrew Jewell's engineering workspace.

Operating rules:
1. Call tools to verify reality before claiming anything. Never guess file contents.
2. Before writing a file, always read_file it first so your write is an informed modification.
3. Keep edits minimal and targeted. Never refactor code you were not asked to touch.
4. When finished, stop and summarize in 1-3 lines: what you changed, what file, why.

Response style:
- Think quietly. Do not narrate your reasoning. Go directly to the tool call.
- One tool call per turn. Stop generating after the closing tag.
- When no tool is needed (the task is done), reply with a short plain-text summary only.

Guardrails:
- Never run destructive commands (rm -rf, git reset --hard, git push --force) without explicit user confirmation in the task.
- Never commit or push — the user handles that.
- Never touch files outside the paths the user pointed you at.
- If you cannot complete the task in the iteration budget, stop and summarize what remains.
"#;

// =============================================================================
// Agent Configuration
// =============================================================================

pub fn config() -> AgentConfig {
    AgentConfig {
        system_prompt: SYSTEM_PROMPT.to_string(),
        // DeepSeek-R1-Distill-Qwen-7B at 1.6 tok/s: ~60s per iteration at
        // 100 output tokens. 12 iterations ≈ 12 min max — fits a human
        // attention span and gives a 7B model room for real multi-step
        // workflows without dragging on.
        max_iterations: 12,
        // Default target: DeepSeek-R1-Distill-Qwen-7B registered on
        // nexus-serve as alias "deepseek" (port 11436). Override on the
        // CLI with `--model qwen3` (etc.) once other aliases are loaded.
        model: "deepseek".to_string(),
        // Low temp keeps edits deterministic. R1-family handles temp=0
        // fine despite the reasoning style.
        temperature: 0.1,
    }
}
