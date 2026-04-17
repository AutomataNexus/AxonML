//! Training Tools — Start, Monitor, List Checkpoints
//!
//! Three tools that drive cargo-based training binaries (works with the
//! llm-training crate and any `cargo run --release --features cuda` target):
//!
//! * `StartTrainingTool` (start_training) builds a `cd <working_dir> &&
//!   cargo run --release --bin <binary> --features cuda -- <args> > <log> 2>&1 &`
//!   shell pipeline, captures the backgrounded PID via `echo $!`, and
//!   returns the binary / pid / log tuple. Default log is
//!   `/tmp/nexus-agent-training.log`.
//! * `CheckTrainingTool` (check_training) tails the log file (default 30
//!   lines) via `tail -n` so the agent can follow loss values, epoch
//!   progress, and errors.
//! * `ListCheckpointsTool` (list_checkpoints) shells out to
//!   `ls -lhtr <directory>` to enumerate checkpoint files ordered by
//!   modification time with human-readable sizes.
//!
//! # File
//! `nexus-agent/src/tools/training.rs`
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
// StartTrainingTool — Launch a Training Run
// =============================================================================

pub struct StartTrainingTool;

#[derive(Deserialize)]
struct StartArgs {
    /// Working directory (e.g. /opt/AxonML/llm-training)
    working_dir: String,
    /// Cargo binary name (e.g. train_llama)
    binary: String,
    /// Extra CLI args (e.g. "--epochs 5 --bs 16 --fresh")
    #[serde(default)]
    args: String,
    /// Log file path (output redirected here)
    #[serde(default = "default_log")]
    log_file: String,
}

fn default_log() -> String {
    "/tmp/nexus-agent-training.log".to_string()
}

#[async_trait]
impl Tool for StartTrainingTool {
    fn name(&self) -> &str { "start_training" }

    fn description(&self) -> &str {
        "Start a training run in the background. The run is a cargo binary executed with --release --features cuda. Output is redirected to a log file that you can tail with check_training."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "working_dir": { "type": "string", "description": "Crate directory (e.g. /opt/AxonML/llm-training)" },
                "binary": { "type": "string", "description": "Binary name (e.g. train_llama, train_mistral)" },
                "args": { "type": "string", "description": "Extra CLI arguments (e.g. '--epochs 5 --bs 16 --fresh')" },
                "log_file": { "type": "string", "description": "Log file path (default: /tmp/nexus-agent-training.log)" }
            },
            "required": ["working_dir", "binary"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: StartArgs = serde_json::from_str(arguments)?;

        let cmd = format!(
            "cd {} && cargo run --release --bin {} --features cuda -- {} > {} 2>&1 &\necho $!",
            args.working_dir, args.binary, args.args, args.log_file
        );

        let output = tokio::process::Command::new("bash")
            .args(["-c", &cmd])
            .output()
            .await?;

        let pid = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(format!(
            "Training started: {} (PID: {}, log: {})",
            args.binary, pid, args.log_file
        ))
    }
}

// =============================================================================
// CheckTrainingTool — Tail Training Log
// =============================================================================

pub struct CheckTrainingTool;

#[derive(Deserialize)]
struct CheckArgs {
    /// Log file to tail
    log_file: String,
    /// Number of lines from the end (default: 30)
    #[serde(default = "default_tail")]
    lines: usize,
}

fn default_tail() -> usize { 30 }

#[async_trait]
impl Tool for CheckTrainingTool {
    fn name(&self) -> &str { "check_training" }

    fn description(&self) -> &str {
        "Check the progress of a running training job by tailing its log file. Shows recent loss values, epoch progress, and any errors."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "log_file": { "type": "string", "description": "Log file path" },
                "lines": { "type": "integer", "description": "Number of lines to show (default: 30)" }
            },
            "required": ["log_file"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: CheckArgs = serde_json::from_str(arguments)?;

        let output = tokio::process::Command::new("tail")
            .args(["-n", &args.lines.to_string(), &args.log_file])
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.is_empty() {
            Ok(format!("Log file {} is empty or does not exist", args.log_file))
        } else {
            Ok(format!("Last {} lines of {}:\n{}", args.lines, args.log_file, stdout))
        }
    }
}

// =============================================================================
// ListCheckpointsTool — Enumerate Checkpoint Files
// =============================================================================

pub struct ListCheckpointsTool;

#[derive(Deserialize)]
struct CheckpointArgs {
    /// Checkpoint directory (e.g. /opt/AxonML/llm-training/checkpoints/llama)
    directory: String,
}

#[async_trait]
impl Tool for ListCheckpointsTool {
    fn name(&self) -> &str { "list_checkpoints" }

    fn description(&self) -> &str {
        "List checkpoint files in a directory, sorted by modification time. Shows file sizes and timestamps."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "directory": { "type": "string", "description": "Checkpoint directory path" }
            },
            "required": ["directory"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: CheckpointArgs = serde_json::from_str(arguments)?;

        let output = tokio::process::Command::new("ls")
            .args(["-lhtr", &args.directory])
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.is_empty() {
            Ok(format!("No files in {}", args.directory))
        } else {
            Ok(stdout.to_string())
        }
    }
}
