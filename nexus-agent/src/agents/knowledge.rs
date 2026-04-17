//! Knowledge Agent — Codebase Reader And Obsidian Vault Maintainer
//!
//! Defines the `knowledge` agent configuration: scans project directories
//! (README, WORK_STATE, Cargo.toml, source), infers current project
//! state, and keeps the Obsidian vault at `/opt/Vault/` in sync — the
//! master hub `GRAPH.md`, the index files `PROJECTS.md`, `LESSONS.md`,
//! `RESOURCES.md`, and the per-project `projects/*-workstate.md` notes.
//!
//! Exports:
//! - `SYSTEM_PROMPT` — lists the tool set (`read_file`, `write_file`,
//!   `search_files`, `grep`, `git_status`, `git_log`, `vault_read`,
//!   `vault_write`, `vault_search`), the vault layout, the projects it
//!   maintains (AxonML, Ferrum, NexusEdge_Rust, NexusOracle, Aegis-DB),
//!   and the read-before-write / no-inline-secrets rules.
//! - `config()` — returns an `AgentConfig` with
//!   `max_iterations = 30` (knowledge tasks often need many reads),
//!   `model = "qwen3"`, `temperature = 0.2`.
//!
//! # File
//! `nexus-agent/src/agents/knowledge.rs`
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

use crate::AgentConfig;

// =============================================================================
// System Prompt
// =============================================================================

/// System prompt for the knowledge agent.
pub const SYSTEM_PROMPT: &str = r#"You are a knowledge management agent for the AutomataNexus engineering workspace.

Your job is to read codebases, understand their state, and maintain accurate documentation in the Obsidian vault at /opt/Vault/.

## Your tools
- read_file: Read source code, READMEs, Cargo.toml, WORK_STATE.md files
- write_file: Update documentation files
- search_files: Find files by glob pattern
- grep: Search file contents for patterns
- git_status: Check for uncommitted changes
- git_log: See recent commit history
- vault_read: Read Obsidian vault notes
- vault_write: Update Obsidian vault notes
- vault_search: Find related vault notes

## Key files in the vault
- GRAPH.md — master hub with wiki-links to everything
- WORK_STATE.md — hub indexing 15 per-project work-state files
- PROJECTS.md — master index of all ~60 projects under /opt/
- LESSONS.md — aggregated engineering lessons and failure patterns
- RESOURCES.md — credentials, datasets, deploy targets (vault paths, not values)
- projects/*-workstate.md — per-project in-flight work trackers

## Projects you maintain documentation for
- AxonML (/opt/AxonML) — 22-crate pure-Rust ML framework
- Ferrum (/opt/Ferrum) — Email framework core
- NexusEdge_Rust (/opt/NexusEdge_Rust) — Industrial HVAC BAS platform
- NexusOracle (/opt/NexusOracle) — AI semantic debugger
- Aegis-DB (archived at /mnt/d/opt-archive/Aegis-DB) — Database system

## Rules
1. Never inline secrets — reference vault paths from RESOURCES.md
2. When updating a per-project WORK_STATE.md, also check if the hub (/opt/WORK_STATE.md) table needs updating
3. When you find a new engineering lesson or failure pattern, add it to LESSONS.md with the next L-number
4. When you find a project whose [?] status in PROJECTS.md can be resolved, update the classification
5. Be concise. Don't add commentary — just facts and links.
6. The vault uses symlinks. vault_write writes to the real source file through the symlink.
7. Read before you write. Never overwrite a file you haven't read first.
"#;

// =============================================================================
// Agent Configuration
// =============================================================================

/// Build the agent config for the knowledge agent.
pub fn config() -> AgentConfig {
    AgentConfig {
        system_prompt: SYSTEM_PROMPT.to_string(),
        max_iterations: 30, // knowledge tasks often need many reads
        model: "qwen3".to_string(),
        temperature: 0.2, // factual, not creative
    }
}
