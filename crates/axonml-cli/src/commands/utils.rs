//! Utils - Common Utilities for CLI Commands
//!
//! Shared utility functions used across CLI commands.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

// =============================================================================
// Output Formatting
// =============================================================================

/// Print a success message
pub fn print_success(message: &str) {
    println!("{} {}", "✓".green().bold(), message);
}

/// Print an info message
pub fn print_info(message: &str) {
    println!("{} {}", "ℹ".blue().bold(), message);
}

/// Print a warning message
pub fn print_warning(message: &str) {
    println!("{} {}", "⚠".yellow().bold(), message);
}

/// Print a step in a multi-step process
pub fn print_step(step: usize, total: usize, message: &str) {
    println!("{} {}", format!("[{step}/{total}]").cyan().bold(), message);
}

/// Print a header
pub fn print_header(title: &str) {
    println!();
    println!("{}", title.bold().underline());
    println!();
}

/// Print a key-value pair
pub fn print_kv(key: &str, value: &str) {
    println!("  {}: {}", key.dimmed(), value);
}

// =============================================================================
// Progress Bars
// =============================================================================

/// Create a training progress bar
pub fn training_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
    );
    pb
}

/// Create an epoch progress bar
pub fn epoch_progress_bar(epoch: usize, total_epochs: usize, steps: u64) -> ProgressBar {
    let pb = ProgressBar::new(steps);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{spinner:.green}} Epoch {epoch}/{total_epochs} [{{bar:30.cyan/blue}}] {{pos}}/{{len}} ({{eta}})"
            ))
            .unwrap()
            .progress_chars("=>-"),
    );
    pb
}

/// Create a spinner for indeterminate operations
pub fn spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(message.to_string());
    pb
}

// =============================================================================
// File Operations
// =============================================================================

use std::path::Path;

/// Check if a path exists
pub fn path_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

/// Check if a path is a file
pub fn is_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_file()
}

/// Get file extension
pub fn get_extension<P: AsRef<Path>>(path: P) -> Option<String> {
    path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
}

/// Create directory if it doesn't exist
pub fn ensure_dir<P: AsRef<Path>>(path: P) -> std::io::Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

// =============================================================================
// Device Parsing
// =============================================================================

/// Parse device string (e.g., "cpu", "cuda:0", "cuda")
pub fn parse_device(device_str: &str) -> (String, Option<usize>) {
    let device_str = device_str.to_lowercase();

    if device_str == "cpu" {
        return ("cpu".to_string(), None);
    }

    if device_str.starts_with("cuda") {
        if let Some(idx_str) = device_str.strip_prefix("cuda:") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                return ("cuda".to_string(), Some(idx));
            }
        }
        return ("cuda".to_string(), Some(0));
    }

    if device_str.starts_with("vulkan") {
        if let Some(idx_str) = device_str.strip_prefix("vulkan:") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                return ("vulkan".to_string(), Some(idx));
            }
        }
        return ("vulkan".to_string(), Some(0));
    }

    // Default to CPU
    ("cpu".to_string(), None)
}

// =============================================================================
// Format Detection
// =============================================================================

/// Detect model format from file extension
pub fn detect_model_format<P: AsRef<Path>>(path: P) -> Option<String> {
    match get_extension(&path)?.as_str() {
        "axonml" | "fer" => Some("axonml".to_string()),
        "onnx" => Some("onnx".to_string()),
        "pt" | "pth" => Some("pytorch".to_string()),
        "safetensors" => Some("safetensors".to_string()),
        "bin" => Some("binary".to_string()),
        "json" => Some("json".to_string()),
        _ => None,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_device() {
        assert_eq!(parse_device("cpu"), ("cpu".to_string(), None));
        assert_eq!(parse_device("cuda"), ("cuda".to_string(), Some(0)));
        assert_eq!(parse_device("cuda:0"), ("cuda".to_string(), Some(0)));
        assert_eq!(parse_device("cuda:1"), ("cuda".to_string(), Some(1)));
        assert_eq!(parse_device("CUDA:2"), ("cuda".to_string(), Some(2)));
    }

    #[test]
    fn test_detect_model_format() {
        assert_eq!(
            detect_model_format("model.axonml"),
            Some("axonml".to_string())
        );
        assert_eq!(detect_model_format("model.onnx"), Some("onnx".to_string()));
        assert_eq!(detect_model_format("model.pt"), Some("pytorch".to_string()));
        assert_eq!(
            detect_model_format("model.safetensors"),
            Some("safetensors".to_string())
        );
    }
}
