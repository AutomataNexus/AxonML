//! Tool Module Root — Registry Population
//!
//! Declares the tool submodules (file, git, github, shell, obsidian,
//! tailscale, email, training) and exposes `register_all` which registers
//! all 22 built-in tools into a `ToolRegistry`.
//!
//! The tool set covers: shell execution, filesystem read/write/search/grep,
//! git status/log/diff/commit, Obsidian vault read/write/search,
//! FerumMailSaaS email, Tailscale status/ping, training lifecycle
//! (start/check/list checkpoints), and GitHub CLI wrappers (list PRs,
//! list issues, CI status, view PR).
//!
//! # File
//! `nexus-agent/src/tools/mod.rs`
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
// Submodule Declarations
// =============================================================================

pub mod email;
pub mod file;
pub mod git;
pub mod github;
pub mod obsidian;
pub mod shell;
pub mod tailscale;
pub mod training;

use crate::ToolRegistry;

// =============================================================================
// Registry Population
// =============================================================================

/// Register all built-in tools into a registry.
pub fn register_all(registry: &mut ToolRegistry) {
    // Core
    registry.register(Box::new(shell::ShellTool));
    registry.register(Box::new(file::ReadFileTool));
    registry.register(Box::new(file::WriteFileTool));
    registry.register(Box::new(file::SearchFilesTool));
    registry.register(Box::new(file::GrepTool));
    // Git
    registry.register(Box::new(git::GitStatusTool));
    registry.register(Box::new(git::GitLogTool));
    registry.register(Box::new(git::GitDiffTool));
    registry.register(Box::new(git::GitCommitTool));
    // Obsidian vault
    registry.register(Box::new(obsidian::VaultReadTool));
    registry.register(Box::new(obsidian::VaultWriteTool));
    registry.register(Box::new(obsidian::VaultSearchTool));
    // Email (FerumMailSaaS)
    registry.register(Box::new(email::EmailTool));
    // Tailscale network
    registry.register(Box::new(tailscale::TailscaleStatusTool));
    registry.register(Box::new(tailscale::TailscalePingTool));
    // Training management
    registry.register(Box::new(training::StartTrainingTool));
    registry.register(Box::new(training::CheckTrainingTool));
    registry.register(Box::new(training::ListCheckpointsTool));
    // GitHub (via gh CLI)
    registry.register(Box::new(github::GhListPrsTool));
    registry.register(Box::new(github::GhListIssuesTool));
    registry.register(Box::new(github::GhCiStatusTool));
    registry.register(Box::new(github::GhViewPrTool));
}
