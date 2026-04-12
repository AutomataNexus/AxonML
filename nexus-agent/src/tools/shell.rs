//! Shell command execution tool.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::Tool;

pub struct ShellTool;

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
