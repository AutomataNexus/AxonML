//! Shell Tool — Bash Command Execution with Timeout
//!
//! Single-tool module providing `ShellTool` (name `shell`). Accepts a
//! `command` string, an optional `working_dir`, and a `timeout_secs` (default
//! 30). Internally spawns `bash -c <command>` via tokio, applies the timeout
//! via `tokio::time::timeout`, and returns a formatted result containing the
//! exit code plus stdout and stderr sections.
//!
//! Used by all agents for cargo builds, tests, system checks, and process
//! management.
//!
//! # File
//! `nexus-agent/src/tools/shell.rs`
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
// ShellTool
// =============================================================================

pub struct ShellTool;

// -----------------------------------------------------------------------------
// Input Types
// -----------------------------------------------------------------------------

#[derive(Deserialize)]
struct ShellArgs {
    command: String,
    #[serde(default)]
    working_dir: Option<String>,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
}

fn default_timeout() -> u64 {
    30
}

// -----------------------------------------------------------------------------
// Tool Implementation
// -----------------------------------------------------------------------------

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its stdout/stderr. Use for cargo builds, tests, system checks, process management."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory (default: current dir)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ShellArgs = serde_json::from_str(arguments)?;

        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-c").arg(&args.command);

        if let Some(ref dir) = args.working_dir {
            cmd.current_dir(dir);
        }

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(args.timeout_secs),
            cmd.output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Command timed out after {}s", args.timeout_secs))?
        .map_err(|e| anyhow::anyhow!("Failed to execute command: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);

        Ok(format!(
            "exit_code: {code}\n--- stdout ---\n{stdout}--- stderr ---\n{stderr}"
        ))
    }
}
