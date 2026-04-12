//! Tool implementations for nexus-agent.

pub mod email;
pub mod file;
pub mod git;
pub mod github;
pub mod obsidian;
pub mod shell;
pub mod tailscale;
pub mod training;

use crate::ToolRegistry;

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
