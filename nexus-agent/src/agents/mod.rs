//! Agent Module Root — Pre-Built Agent Configuration Registry
//!
//! Root module for the eight specialized nexus-agent configurations. Each
//! submodule exposes a single `AgentConfig` constructor that fixes a system
//! prompt, preferred model, and tool subset so the ReAct loop can be
//! specialized to a task domain. Submodules:
//!
//! - `ci_fixer` — CI failure triage and patch generation
//! - `code`     — general code-writing / refactor agent
//! - `fieldtech`— HVAC controls field-technician assistant
//! - `knowledge`— retrieval / Q&A over the project knowledge base
//! - `orchestrator` — dispatches work across the other seven agents
//! - `research` — literature and web research
//! - `retrain`  — dataset curation and model retraining workflows
//! - `shield`   — safety / policy guardrail agent
//!
//! All agents share the Anthropic Messages API format (tool_use /
//! tool_result content blocks, `stop_reason = "tool_use"`).
//!
//! # File
//! `nexus-agent/src/agents/mod.rs`
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
// Agent Submodules
// =============================================================================

pub mod ci_fixer;
pub mod code;
pub mod fieldtech;
pub mod knowledge;
pub mod orchestrator;
pub mod research;
pub mod retrain;
pub mod shield;
