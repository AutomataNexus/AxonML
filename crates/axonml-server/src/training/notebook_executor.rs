//! Notebook Cell Executor for AxonML
//!
//! Executes Rust code cells from training notebooks by compiling and running them.

use crate::db::notebooks::{CellOutput, CellType, NotebookCell};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{error, info};

/// Result of cell execution
#[derive(Debug)]
pub struct ExecutionResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

/// Notebook cell executor
pub struct NotebookExecutor {
    work_dir: PathBuf,
}

impl NotebookExecutor {
    /// Create a new executor with a working directory
    pub fn new(work_dir: PathBuf) -> Self {
        Self { work_dir }
    }

    /// Execute a single cell with context from previous cells
    pub async fn execute_cell(
        &self,
        cell: &NotebookCell,
        previous_cells: &[NotebookCell],
        timeout_ms: u64,
    ) -> ExecutionResult {
        let start = std::time::Instant::now();

        // For markdown cells, just return the source as-is
        if cell.cell_type == CellType::Markdown {
            return ExecutionResult {
                success: true,
                stdout: cell.source.clone(),
                stderr: String::new(),
                duration_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Build the complete Rust source from all code cells
        let source = self.build_source(cell, previous_cells);

        // Create temp directory for this execution
        let exec_id = uuid::Uuid::new_v4().to_string();
        let exec_dir = self.work_dir.join(&exec_id);

        if let Err(e) = tokio::fs::create_dir_all(&exec_dir).await {
            return ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: format!("Failed to create execution directory: {}", e),
                duration_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Write Cargo.toml
        let cargo_toml = self.generate_cargo_toml();
        let cargo_path = exec_dir.join("Cargo.toml");
        if let Err(e) = tokio::fs::write(&cargo_path, cargo_toml).await {
            let _ = tokio::fs::remove_dir_all(&exec_dir).await;
            return ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: format!("Failed to write Cargo.toml: {}", e),
                duration_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Create src directory and write main.rs
        let src_dir = exec_dir.join("src");
        if let Err(e) = tokio::fs::create_dir_all(&src_dir).await {
            let _ = tokio::fs::remove_dir_all(&exec_dir).await;
            return ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: format!("Failed to create src directory: {}", e),
                duration_ms: start.elapsed().as_millis() as u64,
            };
        }

        let main_path = src_dir.join("main.rs");
        if let Err(e) = tokio::fs::write(&main_path, &source).await {
            let _ = tokio::fs::remove_dir_all(&exec_dir).await;
            return ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: format!("Failed to write main.rs: {}", e),
                duration_ms: start.elapsed().as_millis() as u64,
            };
        }

        info!(exec_id = %exec_id, "Compiling and running notebook cell");

        // Run cargo build + run with timeout
        let result = self.run_cargo(&exec_dir, timeout_ms).await;

        // Cleanup
        if let Err(e) = tokio::fs::remove_dir_all(&exec_dir).await {
            error!(exec_id = %exec_id, error = %e, "Failed to cleanup execution directory");
        }

        ExecutionResult {
            success: result.success,
            stdout: result.stdout,
            stderr: result.stderr,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Build complete Rust source from cells
    fn build_source(&self, current_cell: &NotebookCell, previous_cells: &[NotebookCell]) -> String {
        let mut imports = Vec::new();
        let mut code_lines = Vec::new();

        // Extract imports and code from previous cells
        for cell in previous_cells {
            if cell.cell_type != CellType::Code {
                continue;
            }
            self.categorize_source(&cell.source, &mut imports, &mut code_lines);
        }

        // Add current cell
        self.categorize_source(&current_cell.source, &mut imports, &mut code_lines);

        // Build the final source
        let mut source = String::new();

        // Standard prelude
        source.push_str("#![allow(unused_imports, unused_variables, dead_code)]\n\n");

        // Add all imports
        for import in &imports {
            source.push_str(import);
            source.push('\n');
        }
        source.push('\n');

        // Wrap code in main function
        source.push_str("fn main() {\n");
        for line in &code_lines {
            source.push_str("    ");
            source.push_str(line);
            source.push('\n');
        }
        source.push_str("}\n");

        source
    }

    /// Categorize source lines into imports and code
    fn categorize_source(&self, source: &str, imports: &mut Vec<String>, code: &mut Vec<String>) {
        for line in source.lines() {
            let trimmed = line.trim();

            // Skip comments that look like markdown headers
            if trimmed.starts_with("# ") && !trimmed.starts_with("#![") {
                continue;
            }

            // Skip empty lines at this stage
            if trimmed.is_empty() {
                continue;
            }

            // Categorize as import or code
            if trimmed.starts_with("use ") || trimmed.starts_with("extern crate") {
                if !imports.contains(&trimmed.to_string()) {
                    imports.push(trimmed.to_string());
                }
            } else {
                code.push(line.to_string());
            }
        }
    }

    /// Generate Cargo.toml for the temporary project
    fn generate_cargo_toml(&self) -> String {
        r#"[package]
name = "notebook_cell"
version = "0.1.0"
edition = "2021"

[dependencies]
axonml = { path = "/opt/AxonML/crates/axonml" }

[profile.dev]
opt-level = 0
debug = false
"#.to_string()
    }

    /// Run cargo build and execute
    async fn run_cargo(&self, exec_dir: &PathBuf, timeout_ms: u64) -> ExecutionResult {
        let timeout = std::time::Duration::from_millis(timeout_ms);

        // First, build the project
        let build_result = tokio::time::timeout(timeout, async {
            let child = Command::new("cargo")
                .arg("build")
                .arg("--quiet")
                .current_dir(exec_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("Failed to spawn cargo build: {}", e))?;

            let output = child.wait_with_output().await
                .map_err(|e| format!("Failed to wait for cargo build: {}", e))?;

            Ok::<_, String>(output)
        }).await;

        let build_output = match build_result {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return ExecutionResult {
                    success: false,
                    stdout: String::new(),
                    stderr: e,
                    duration_ms: 0,
                };
            }
            Err(_) => {
                return ExecutionResult {
                    success: false,
                    stdout: String::new(),
                    stderr: format!("Compilation timed out after {}ms", timeout_ms),
                    duration_ms: timeout_ms,
                };
            }
        };

        if !build_output.status.success() {
            let stderr = String::from_utf8_lossy(&build_output.stderr).to_string();
            // Clean up error messages to be more readable
            let clean_stderr = self.clean_compiler_output(&stderr);
            return ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: clean_stderr,
                duration_ms: 0,
            };
        }

        // Now run the binary
        let run_result = tokio::time::timeout(timeout, async {
            let child = Command::new("cargo")
                .arg("run")
                .arg("--quiet")
                .current_dir(exec_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("Failed to spawn cargo run: {}", e))?;

            let output = child.wait_with_output().await
                .map_err(|e| format!("Failed to wait for execution: {}", e))?;

            Ok::<_, String>(output)
        }).await;

        match run_result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                ExecutionResult {
                    success: output.status.success(),
                    stdout,
                    stderr,
                    duration_ms: 0,
                }
            }
            Ok(Err(e)) => ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: e,
                duration_ms: 0,
            },
            Err(_) => ExecutionResult {
                success: false,
                stdout: String::new(),
                stderr: format!("Execution timed out after {}ms", timeout_ms),
                duration_ms: timeout_ms,
            },
        }
    }

    /// Clean up compiler output to be more readable in the notebook
    fn clean_compiler_output(&self, output: &str) -> String {
        let mut cleaned = Vec::new();

        for line in output.lines() {
            // Skip lines with temp paths
            if line.contains("/tmp/") && line.contains("notebook_cell") {
                // Replace temp path with just "cell"
                let cleaned_line = line
                    .replace(r"/tmp/", "")
                    .replace("notebook_cell/src/main.rs", "cell");
                cleaned.push(cleaned_line);
            } else if line.starts_with("error") || line.starts_with("warning") || line.contains("-->") {
                cleaned.push(line.to_string());
            } else if line.trim().starts_with('|') || line.trim().starts_with('=') {
                cleaned.push(line.to_string());
            } else if !line.trim().is_empty() {
                cleaned.push(line.to_string());
            }
        }

        cleaned.join("\n")
    }
}

impl Default for NotebookExecutor {
    fn default() -> Self {
        Self::new(PathBuf::from("/tmp/axonml-notebooks"))
    }
}

/// Convert execution result to cell output
pub fn result_to_cell_output(result: ExecutionResult, execution_count: u32) -> CellOutput {
    if result.success {
        CellOutput {
            output_type: "execute_result".to_string(),
            text: if result.stdout.is_empty() {
                Some("(no output)".to_string())
            } else {
                Some(result.stdout)
            },
            data: None,
            execution_count: Some(execution_count),
            error_name: None,
            error_value: None,
            traceback: None,
        }
    } else {
        CellOutput {
            output_type: "error".to_string(),
            text: None,
            data: None,
            execution_count: Some(execution_count),
            error_name: Some("ExecutionError".to_string()),
            error_value: Some(if result.stderr.is_empty() {
                "Unknown error".to_string()
            } else {
                result.stderr.lines().next().unwrap_or("Unknown error").to_string()
            }),
            traceback: Some(result.stderr.lines().map(|s| s.to_string()).collect()),
        }
    }
}
