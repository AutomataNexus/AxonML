//! Git Tools — Status, Log, Diff, Commit
//!
//! Four tools wrapping the `git` CLI against a caller-supplied `repo` path:
//! `GitStatusTool` runs `git status --short`; `GitLogTool` runs
//! `git log --oneline -<count>` (default 10); `GitDiffTool` runs
//! `git diff --stat` or `git diff --cached --stat` when `staged=true`;
//! `GitCommitTool` optionally stages a file list via `git add` then creates
//! a commit with `-m <message>`.
//!
//! All commands go through the private `git()` helper which sets
//! `current_dir(repo)`, captures stdout/stderr, and `bail!`s with the
//! stderr text if the git process exits non-zero.
//!
//! # File
//! `nexus-agent/src/tools/git.rs`
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
// Shared Git Runner
// =============================================================================

async fn git(args: &[&str], repo: &str) -> anyhow::Result<String> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .await?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        anyhow::bail!("git {} failed: {stderr}", args.join(" "));
    }
    Ok(stdout.to_string())
}

// =============================================================================
// GitStatusTool
// =============================================================================

pub struct GitStatusTool;

#[derive(Deserialize)]
struct RepoArg {
    repo: String,
}

#[async_trait]
impl Tool for GitStatusTool {
    fn name(&self) -> &str { "git_status" }
    fn description(&self) -> &str { "Run `git status --short` in a repository to see uncommitted changes." }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"repo": {"type": "string", "description": "Repository path"}}, "required": ["repo"]})
    }
    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: RepoArg = serde_json::from_str(arguments)?;
        git(&["status", "--short"], &args.repo).await
    }
}

// =============================================================================
// GitLogTool
// =============================================================================

pub struct GitLogTool;

#[derive(Deserialize)]
struct LogArgs {
    repo: String,
    #[serde(default = "default_log_count")]
    count: usize,
}
fn default_log_count() -> usize { 10 }

#[async_trait]
impl Tool for GitLogTool {
    fn name(&self) -> &str { "git_log" }
    fn description(&self) -> &str { "Show recent git commits (oneline format)." }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {
            "repo": {"type": "string", "description": "Repository path"},
            "count": {"type": "integer", "description": "Number of commits (default: 10)"}
        }, "required": ["repo"]})
    }
    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: LogArgs = serde_json::from_str(arguments)?;
        git(&["log", "--oneline", &format!("-{}", args.count)], &args.repo).await
    }
}

// =============================================================================
// GitDiffTool
// =============================================================================

pub struct GitDiffTool;

#[derive(Deserialize)]
struct DiffArgs {
    repo: String,
    #[serde(default)]
    staged: bool,
}

#[async_trait]
impl Tool for GitDiffTool {
    fn name(&self) -> &str { "git_diff" }
    fn description(&self) -> &str { "Show git diff (unstaged changes, or --staged for staged changes)." }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {
            "repo": {"type": "string", "description": "Repository path"},
            "staged": {"type": "boolean", "description": "Show staged changes instead (default: false)"}
        }, "required": ["repo"]})
    }
    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: DiffArgs = serde_json::from_str(arguments)?;
        if args.staged {
            git(&["diff", "--cached", "--stat"], &args.repo).await
        } else {
            git(&["diff", "--stat"], &args.repo).await
        }
    }
}

// =============================================================================
// GitCommitTool
// =============================================================================

pub struct GitCommitTool;

#[derive(Deserialize)]
struct CommitArgs {
    repo: String,
    message: String,
    #[serde(default)]
    files: Vec<String>,
}

#[async_trait]
impl Tool for GitCommitTool {
    fn name(&self) -> &str { "git_commit" }
    fn description(&self) -> &str {
        "Stage specific files and create a git commit. If files is empty, commits whatever is already staged."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {
            "repo": {"type": "string", "description": "Repository path"},
            "message": {"type": "string", "description": "Commit message"},
            "files": {"type": "array", "items": {"type": "string"}, "description": "Files to stage (empty = commit staged)"}
        }, "required": ["repo", "message"]})
    }
    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: CommitArgs = serde_json::from_str(arguments)?;
        if !args.files.is_empty() {
            let file_args: Vec<&str> = std::iter::once("add")
                .chain(args.files.iter().map(|f| f.as_str()))
                .collect();
            git(&file_args, &args.repo).await?;
        }
        git(&["commit", "-m", &args.message], &args.repo).await
    }
}
