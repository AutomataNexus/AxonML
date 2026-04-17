//! File Tools — Read, Write, Search, Grep
//!
//! Four filesystem tools for nexus-agent: `ReadFileTool` (read_file) reads a
//! file with optional line offset/limit windowing and numbered output;
//! `WriteFileTool` (write_file) creates or overwrites a file, auto-creating
//! parent directories; `SearchFilesTool` (search_files) matches files by
//! glob pattern under a root directory (default `/opt`, truncates at 100
//! results); `GrepTool` (grep) shells out to ripgrep with `-n --no-heading
//! --max-count=50` and optional case-insensitive matching.
//!
//! Each struct implements the `Tool` trait and provides a JSON Schema for
//! LLM tool calling.
//!
//! # File
//! `nexus-agent/src/tools/file.rs`
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
// ReadFile
// =============================================================================

pub struct ReadFileTool;

#[derive(Deserialize)]
struct ReadArgs {
    path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Optionally read a specific line range with offset + limit."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute file path" },
                "offset": { "type": "integer", "description": "Start line (0-indexed)" },
                "limit": { "type": "integer", "description": "Max lines to read" }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ReadArgs = serde_json::from_str(arguments)?;
        let content = tokio::fs::read_to_string(&args.path).await?;

        let lines: Vec<&str> = content.lines().collect();
        let offset = args.offset.unwrap_or(0);
        let limit = args.limit.unwrap_or(lines.len());
        let end = (offset + limit).min(lines.len());

        if offset >= lines.len() {
            return Ok(format!("(file has {} lines, offset {} is past end)", lines.len(), offset));
        }

        let selected: Vec<String> = lines[offset..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>4}\t{}", offset + i + 1, line))
            .collect();

        Ok(format!(
            "{} ({} lines, showing {}-{})\n{}",
            args.path,
            lines.len(),
            offset + 1,
            end,
            selected.join("\n")
        ))
    }
}

// =============================================================================
// WriteFile
// =============================================================================

pub struct WriteFileTool;

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file (creates or overwrites). Use for creating new files or full rewrites."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute file path" },
                "content": { "type": "string", "description": "File content to write" }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: WriteArgs = serde_json::from_str(arguments)?;
        if let Some(parent) = std::path::Path::new(&args.path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&args.path, &args.content).await?;
        Ok(format!("Wrote {} bytes to {}", args.content.len(), args.path))
    }
}

// =============================================================================
// SearchFiles (glob)
// =============================================================================

pub struct SearchFilesTool;

#[derive(Deserialize)]
struct SearchArgs {
    pattern: String,
    #[serde(default = "default_search_dir")]
    directory: String,
}

fn default_search_dir() -> String {
    "/opt".to_string()
}

#[async_trait]
impl Tool for SearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern (e.g. '**/*.rs', 'src/**/*.toml'). Returns matching paths."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern" },
                "directory": { "type": "string", "description": "Root directory (default: /opt)" }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: SearchArgs = serde_json::from_str(arguments)?;
        let full_pattern = format!("{}/{}", args.directory, args.pattern);

        let mut results = Vec::new();
        for entry in glob::glob(&full_pattern)? {
            if let Ok(path) = entry {
                results.push(path.display().to_string());
                if results.len() >= 100 {
                    results.push("... (truncated at 100 results)".to_string());
                    break;
                }
            }
        }

        if results.is_empty() {
            Ok(format!("No files matched: {full_pattern}"))
        } else {
            Ok(format!("{} matches:\n{}", results.len(), results.join("\n")))
        }
    }
}

// =============================================================================
// Grep (content search)
// =============================================================================

pub struct GrepTool;

#[derive(Deserialize)]
struct GrepArgs {
    pattern: String,
    path: String,
    #[serde(default)]
    case_insensitive: bool,
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents for a regex pattern. Returns matching lines with file paths and line numbers."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regex pattern to search for" },
                "path": { "type": "string", "description": "File or directory to search" },
                "case_insensitive": { "type": "boolean", "description": "Case-insensitive match (default: false)" }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(&self, arguments: &str) -> anyhow::Result<String> {
        let args: GrepArgs = serde_json::from_str(arguments)?;
        let mut cmd = tokio::process::Command::new("rg");
        cmd.arg("--no-heading").arg("-n").arg("--max-count=50");
        if args.case_insensitive {
            cmd.arg("-i");
        }
        cmd.arg(&args.pattern).arg(&args.path);

        let output = cmd.output().await?;
        let stdout = String::from_utf8_lossy(&output.stdout);

        if stdout.is_empty() {
            Ok(format!("No matches for '{}' in {}", args.pattern, args.path))
        } else {
            Ok(stdout.to_string())
        }
    }
}
