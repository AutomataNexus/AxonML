//! Training management tools — kick off, monitor, and evaluate training runs.
//!
//! Works with the llm-training crate and any cargo-based training binary.
//! The agent can start a training run in the background, poll its output,
//! and check checkpoint files when done.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::Tool;

// =============================================================================
// Start a training run
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
// Check training progress
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
// List checkpoints
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
