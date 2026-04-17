//! Obsidian Vault Tools — Read, Write, Search at /opt/Vault/
//!
//! Three tools against the knowledge graph rooted at `/opt/Vault/`:
//! `VaultReadTool` (vault_read) reads a note via the relative path under the
//! vault root, following symlinks so vault-symlinked source files resolve to
//! their real targets; `VaultWriteTool` (vault_write) canonicalizes the path
//! first so writes land on the symlink target rather than the symlink
//! itself (the vault is symlink-based, wrapping source files in
//! `/opt/*/WORK_STATE.md` etc.); `VaultSearchTool` (vault_search) shells out
//! to `rg -l --max-count=1 -i <query> /opt/Vault` and strips the vault
//! prefix from results for readability.
//!
//! The shared `VAULT_ROOT` constant pins the root at `/opt/Vault`.
//!
//! # File
//! `nexus-agent/src/tools/obsidian.rs`
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

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::Tool;

// =============================================================================
// Vault Root
// =============================================================================

const VAULT_ROOT: &str = "/opt/Vault";

// =============================================================================
// VaultReadTool — Read a Vault Note
// =============================================================================

pub struct VaultReadTool;

#[derive(Deserialize)]
struct VaultReadArgs {
    /// Path relative to vault root (e.g. "WORK_STATE.md", "projects/AxonML-workstate.md")
    path: String,
}

#[async_trait]
impl Tool for VaultReadTool {
    fn name(&self) -> &str { "vault_read" }
    fn description(&self) -> &str {
        "Read a note from the Obsidian vault at /opt/Vault/. Use relative paths like 'WORK_STATE.md', 'LESSONS.md', 'projects/AxonML-workstate.md'."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to /opt/Vault/"}
        }, "required": ["path"]})
    }
    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: VaultReadArgs = serde_json::from_str(arguments)?;
        let full_path = format!("{}/{}", VAULT_ROOT, args.path);
        // Follow symlinks (vault is symlink-based)
        let content = tokio::fs::read_to_string(&full_path).await?;
        Ok(content)
    }
}

// =============================================================================
// VaultWriteTool — Write / Update a Vault Note
// =============================================================================

pub struct VaultWriteTool;

#[derive(Deserialize)]
struct VaultWriteArgs {
    /// Path relative to vault root. If this is a symlink, writes to the target.
    path: String,
    content: String,
}

#[async_trait]
impl Tool for VaultWriteTool {
    fn name(&self) -> &str { "vault_write" }
    fn description(&self) -> &str {
        "Write content to a note in the Obsidian vault. If the file is a symlink (most are), the write goes to the real source file. Use for updating WORK_STATE, LESSONS, project notes."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to /opt/Vault/"},
            "content": {"type": "string", "description": "Full file content to write"}
        }, "required": ["path", "content"]})
    }
    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: VaultWriteArgs = serde_json::from_str(arguments)?;
        let full_path = format!("{}/{}", VAULT_ROOT, args.path);
        // Resolve symlink to write to the actual file
        let real_path = tokio::fs::canonicalize(&full_path)
            .await
            .unwrap_or_else(|_| std::path::PathBuf::from(&full_path));
        tokio::fs::write(&real_path, &args.content).await?;
        Ok(format!("Wrote {} bytes to {} (real: {})", args.content.len(), args.path, real_path.display()))
    }
}

// =============================================================================
// VaultSearchTool — Search Vault Notes
// =============================================================================

pub struct VaultSearchTool;

#[derive(Deserialize)]
struct VaultSearchArgs {
    query: String,
}

#[async_trait]
impl Tool for VaultSearchTool {
    fn name(&self) -> &str { "vault_search" }
    fn description(&self) -> &str {
        "Search the Obsidian vault for notes containing a keyword or pattern. Returns matching file paths and preview lines."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {
            "query": {"type": "string", "description": "Search term or regex pattern"}
        }, "required": ["query"]})
    }
    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: VaultSearchArgs = serde_json::from_str(arguments)?;
        let output = tokio::process::Command::new("rg")
            .args(["-l", "--max-count=1", "-i", &args.query, VAULT_ROOT])
            .output()
            .await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.is_empty() {
            Ok(format!("No vault notes matched '{}'", args.query))
        } else {
            // Strip the vault root prefix for cleaner output
            let results: Vec<String> = stdout
                .lines()
                .map(|l| l.strip_prefix(VAULT_ROOT).unwrap_or(l).trim_start_matches('/').to_string())
                .collect();
            Ok(format!("{} notes matched:\n{}", results.len(), results.join("\n")))
        }
    }
}
