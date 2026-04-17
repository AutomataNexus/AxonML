//! GitHub Tools — Issues, PRs, and CI via the `gh` CLI
//!
//! Four tools that shell out to the `gh` GitHub CLI: `GhListPrsTool`
//! (gh_list_prs) runs `gh pr list` with state + limit; `GhListIssuesTool`
//! (gh_list_issues) runs `gh issue list`; `GhCiStatusTool` (gh_ci_status)
//! runs `gh run list` to surface recent Actions workflow runs;
//! `GhViewPrTool` (gh_view_pr) runs `gh pr view <number>` for full PR
//! details including reviews and checks.
//!
//! All four route through the private `gh()` helper which captures
//! stdout/stderr and `bail!`s on non-zero exit.
//!
//! Requires `gh` to be installed and authenticated (`gh auth login`).
//!
//! # File
//! `nexus-agent/src/tools/github.rs`
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
// Shared gh Runner
// =============================================================================

async fn gh(args: &[&str]) -> anyhow::Result<String> {
    let output = tokio::process::Command::new("gh")
        .args(args)
        .output()
        .await?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        anyhow::bail!("gh {} failed: {stderr}", args.join(" "));
    }
    Ok(stdout.to_string())
}

// =============================================================================
// GhListPrsTool — List PRs
// =============================================================================

pub struct GhListPrsTool;

#[derive(Deserialize)]
struct PrListArgs {
    repo: String,
    #[serde(default = "default_pr_state")]
    state: String,
    #[serde(default = "default_pr_limit")]
    limit: usize,
}
fn default_pr_state() -> String { "open".to_string() }
fn default_pr_limit() -> usize { 10 }

#[async_trait]
impl Tool for GhListPrsTool {
    fn name(&self) -> &str { "gh_list_prs" }

    fn description(&self) -> &str {
        "List pull requests for a GitHub repo. Returns PR number, title, author, and status."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "repo": { "type": "string", "description": "GitHub repo (owner/name, e.g. AutomataNexus/AxonML)" },
                "state": { "type": "string", "description": "PR state: open, closed, merged, all (default: open)" },
                "limit": { "type": "integer", "description": "Max PRs to return (default: 10)" }
            },
            "required": ["repo"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: PrListArgs = serde_json::from_str(arguments)?;
        gh(&[
            "pr", "list",
            "--repo", &args.repo,
            "--state", &args.state,
            "--limit", &args.limit.to_string(),
        ]).await
    }
}

// =============================================================================
// GhListIssuesTool — List Issues
// =============================================================================

pub struct GhListIssuesTool;

#[derive(Deserialize)]
struct IssueListArgs {
    repo: String,
    #[serde(default = "default_pr_state")]
    state: String,
    #[serde(default = "default_pr_limit")]
    limit: usize,
}

#[async_trait]
impl Tool for GhListIssuesTool {
    fn name(&self) -> &str { "gh_list_issues" }

    fn description(&self) -> &str {
        "List issues for a GitHub repo."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "repo": { "type": "string", "description": "GitHub repo (owner/name)" },
                "state": { "type": "string", "description": "Issue state: open, closed, all (default: open)" },
                "limit": { "type": "integer", "description": "Max issues (default: 10)" }
            },
            "required": ["repo"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: IssueListArgs = serde_json::from_str(arguments)?;
        gh(&[
            "issue", "list",
            "--repo", &args.repo,
            "--state", &args.state,
            "--limit", &args.limit.to_string(),
        ]).await
    }
}

// =============================================================================
// GhCiStatusTool — CI / Workflow Run Status
// =============================================================================

pub struct GhCiStatusTool;

#[derive(Deserialize)]
struct CiArgs {
    repo: String,
    #[serde(default = "default_ci_limit")]
    limit: usize,
}
fn default_ci_limit() -> usize { 5 }

#[async_trait]
impl Tool for GhCiStatusTool {
    fn name(&self) -> &str { "gh_ci_status" }

    fn description(&self) -> &str {
        "Check CI / GitHub Actions workflow run status for a repo. Returns recent workflow runs with status and conclusion."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "repo": { "type": "string", "description": "GitHub repo (owner/name)" },
                "limit": { "type": "integer", "description": "Max workflow runs (default: 5)" }
            },
            "required": ["repo"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: CiArgs = serde_json::from_str(arguments)?;
        gh(&[
            "run", "list",
            "--repo", &args.repo,
            "--limit", &args.limit.to_string(),
        ]).await
    }
}

// =============================================================================
// GhViewPrTool — View a Specific PR
// =============================================================================

pub struct GhViewPrTool;

#[derive(Deserialize)]
struct ViewPrArgs {
    repo: String,
    number: u64,
}

#[async_trait]
impl Tool for GhViewPrTool {
    fn name(&self) -> &str { "gh_view_pr" }

    fn description(&self) -> &str {
        "View details of a specific pull request including description, review status, and checks."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "repo": { "type": "string", "description": "GitHub repo (owner/name)" },
                "number": { "type": "integer", "description": "PR number" }
            },
            "required": ["repo", "number"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ViewPrArgs = serde_json::from_str(arguments)?;
        gh(&[
            "pr", "view",
            &args.number.to_string(),
            "--repo", &args.repo,
        ]).await
    }
}
