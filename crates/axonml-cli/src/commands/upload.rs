//! Upload - Model Upload Command
//!
//! Uploads model files to Axonml, validates structure, and organizes
//! models for training and inference.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::fs;
use std::path::PathBuf;

use axonml_serialize::{load_state_dict, StateDict};

use super::utils::{
    detect_model_format, ensure_dir, path_exists, print_header, print_info, print_kv,
    print_success, print_warning,
};
use crate::cli::UploadArgs;
use crate::error::{CliError, CliResult};

// =============================================================================
// Model Metadata
// =============================================================================

/// Metadata for an uploaded model
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub description: Option<String>,
    pub format: String,
    pub version: String,
    pub num_parameters: usize,
    pub architecture: Option<String>,
    pub input_shape: Option<Vec<usize>>,
    pub output_shape: Option<Vec<usize>>,
    pub file_size: u64,
    pub checksum: String,
}

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `upload` command
pub fn execute(args: UploadArgs) -> CliResult<()> {
    print_header("Model Upload");

    // Verify source file exists
    let source_path = PathBuf::from(&args.path);
    if !path_exists(&source_path) {
        return Err(CliError::Model(format!(
            "Model file not found: {}",
            args.path
        )));
    }

    // Detect model format
    let format = args
        .format
        .clone()
        .or_else(|| detect_model_format(&source_path))
        .unwrap_or_else(|| "unknown".to_string());

    // Determine model name
    let model_name = args.name.clone().unwrap_or_else(|| {
        source_path
            .file_stem().map_or_else(|| "model".to_string(), |s| s.to_string_lossy().to_string())
    });

    print_kv("Source", &args.path);
    print_kv("Model name", &model_name);
    print_kv("Format", &format);
    print_kv("Version", &args.version);

    // Get file metadata
    let file_metadata = fs::metadata(&source_path)?;
    let file_size = file_metadata.len();
    print_kv("File size", &format_file_size(file_size));

    println!();

    // Ensure output directory exists
    ensure_dir(&args.output)?;

    // Construct destination path
    let dest_filename = format!("{}_{}.{}", model_name, args.version, get_extension(&format));
    let dest_path = PathBuf::from(&args.output).join(&dest_filename);

    // Check if destination already exists
    if path_exists(&dest_path) && !args.overwrite {
        return Err(CliError::Model(format!(
            "Model already exists at {}. Use --overwrite to replace.",
            dest_path.display()
        )));
    }

    // Validate model if requested
    let mut num_parameters = 0;
    if args.validate {
        print_info("Validating model structure...");
        match validate_model(&source_path, &format) {
            Ok(info) => {
                num_parameters = info.num_parameters;
                print_success(&format!("Model validated: {num_parameters} parameters"));
            }
            Err(e) => {
                print_warning(&format!("Validation warning: {e}"));
            }
        }
    }

    // Copy model file to destination
    print_info("Uploading model...");
    fs::copy(&source_path, &dest_path)?;

    // Calculate checksum
    let checksum = calculate_checksum(&dest_path)?;

    // Create metadata file
    let metadata = ModelMetadata {
        name: model_name.clone(),
        description: args.description.clone(),
        format: format.clone(),
        version: args.version.clone(),
        num_parameters,
        architecture: None,
        input_shape: None,
        output_shape: None,
        file_size,
        checksum: checksum.clone(),
    };

    let metadata_path = dest_path.with_extension("meta.json");
    save_metadata(&metadata, &metadata_path)?;

    println!();
    print_success("Model uploaded successfully!");
    print_header("Model Information");
    print_kv("Name", &model_name);
    print_kv("Location", &dest_path.display().to_string());
    print_kv("Parameters", &format_number(num_parameters));
    print_kv("Checksum", &checksum[..16]);

    // Inspect model if requested
    if args.inspect {
        println!();
        print_header("Model Architecture");
        inspect_model(&dest_path, &format)?;
    }

    println!();
    print_info("Use 'axonml train --model' to train with this model");
    print_info("Use 'axonml inspect' for detailed architecture info");

    Ok(())
}

// =============================================================================
// Model Validation
// =============================================================================

struct ValidationInfo {
    num_parameters: usize,
}

fn validate_model(path: &PathBuf, format: &str) -> CliResult<ValidationInfo> {
    match format.to_lowercase().as_str() {
        "axonml" | "safetensors" | "pt" | "pth" => {
            // Try to load as state dict
            let state_dict = load_state_dict(path)
                .map_err(|e| CliError::Model(format!("Failed to load model: {e}")))?;

            let num_parameters = count_parameters(&state_dict);
            Ok(ValidationInfo { num_parameters })
        }
        "onnx" => {
            // ONNX validation would require onnx parsing library
            // For now, just check file header
            let data = fs::read(path)?;
            if data.len() < 8 {
                return Err(CliError::Model("Invalid ONNX file: too small".to_string()));
            }
            // ONNX files start with protobuf structure
            Ok(ValidationInfo { num_parameters: 0 })
        }
        _ => {
            // Unknown format, just check file exists and is readable
            let _ = fs::read(path)?;
            Ok(ValidationInfo { num_parameters: 0 })
        }
    }
}

fn count_parameters(state_dict: &StateDict) -> usize {
    state_dict
        .entries()
        .map(|(_, entry)| entry.data.shape.iter().product::<usize>())
        .sum()
}

// =============================================================================
// Model Inspection
// =============================================================================

fn inspect_model(path: &PathBuf, format: &str) -> CliResult<()> {
    match format.to_lowercase().as_str() {
        "axonml" | "safetensors" | "pt" | "pth" => {
            let state_dict = load_state_dict(path)
                .map_err(|e| CliError::Model(format!("Failed to load model: {e}")))?;

            println!("Layers:");
            for (name, entry) in state_dict.entries() {
                let shape_str = entry
                    .data
                    .shape
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                let params: usize = entry.data.shape.iter().product();
                println!(
                    "  {} [{}] - {} params",
                    name,
                    shape_str,
                    format_number(params)
                );
            }
        }
        _ => {
            println!("  Inspection not available for {format} format");
        }
    }

    Ok(())
}

// =============================================================================
// Metadata Management
// =============================================================================

fn save_metadata(metadata: &ModelMetadata, path: &PathBuf) -> CliResult<()> {
    let json = serde_json::json!({
        "name": metadata.name,
        "description": metadata.description,
        "format": metadata.format,
        "version": metadata.version,
        "num_parameters": metadata.num_parameters,
        "architecture": metadata.architecture,
        "input_shape": metadata.input_shape,
        "output_shape": metadata.output_shape,
        "file_size": metadata.file_size,
        "checksum": metadata.checksum,
        "uploaded_at": chrono::Utc::now().to_rfc3339(),
    });

    let content = serde_json::to_string_pretty(&json)?;
    fs::write(path, content)?;

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn get_extension(format: &str) -> &str {
    match format.to_lowercase().as_str() {
        "axonml" => "axonml",
        "safetensors" => "safetensors",
        "onnx" => "onnx",
        "pt" | "pth" | "pytorch" => "pt",
        _ => "bin",
    }
}

fn format_file_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn calculate_checksum(path: &PathBuf) -> CliResult<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let data = fs::read(path)?;
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash = hasher.finish();

    Ok(format!("{hash:016x}"))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(500), "500 bytes");
        assert_eq!(format_file_size(1024), "1.00 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_file_size(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.50K");
        assert_eq!(format_number(1_500_000), "1.50M");
        assert_eq!(format_number(1_500_000_000), "1.50B");
    }

    #[test]
    fn test_get_extension() {
        assert_eq!(get_extension("axonml"), "axonml");
        assert_eq!(get_extension("safetensors"), "safetensors");
        assert_eq!(get_extension("onnx"), "onnx");
        assert_eq!(get_extension("pytorch"), "pt");
    }
}
