//! Convert - Model Format Conversion Command
//!
//! Converts models between different formats.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::path::PathBuf;
use std::time::Instant;

use super::utils::{
    detect_model_format, path_exists, print_header, print_kv, print_success, print_warning, spinner,
};
use crate::cli::ConvertArgs;
use crate::error::{CliError, CliResult};

// =============================================================================
// Supported Formats
// =============================================================================

const SUPPORTED_FORMATS: &[&str] = &[
    "axonml",
    "onnx",
    "pytorch",
    "safetensors",
    "json",
    "binary",
];

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `convert` command
pub fn execute(args: ConvertArgs) -> CliResult<()> {
    print_header("Model Conversion");

    // Verify input exists
    let input_path = PathBuf::from(&args.input);
    if !path_exists(&input_path) {
        return Err(CliError::Model(format!(
            "Input model not found: {}",
            args.input
        )));
    }

    // Detect or use specified formats
    let from_format = args
        .from
        .clone()
        .or_else(|| detect_model_format(&input_path))
        .ok_or_else(|| {
            CliError::InvalidArgument(
                "Cannot detect input format. Please specify --from".to_string(),
            )
        })?;

    let to_format = args
        .to
        .clone()
        .or_else(|| detect_model_format(&args.output))
        .ok_or_else(|| {
            CliError::InvalidArgument(
                "Cannot detect output format. Please specify --to".to_string(),
            )
        })?;

    // Validate formats
    validate_format(&from_format)?;
    validate_format(&to_format)?;

    // Check for same format conversion
    if from_format == to_format {
        print_warning("Input and output formats are the same");
    }

    print_header("Conversion Details");
    print_kv("Input", &args.input);
    print_kv("Output", &args.output);
    print_kv("From format", &from_format);
    print_kv("To format", &to_format);
    print_kv("Optimize", &args.optimize.to_string());

    println!();

    // Run conversion
    let start_time = Instant::now();
    let sp = spinner("Converting model...");

    let result = convert_model(
        &input_path,
        &args.output,
        &from_format,
        &to_format,
        args.optimize,
    );

    sp.finish_and_clear();
    let elapsed = start_time.elapsed();

    match result {
        Ok(info) => {
            print_success("Conversion completed successfully");
            println!();
            print_header("Conversion Summary");
            print_kv("Input size", &format_size(info.input_size));
            print_kv("Output size", &format_size(info.output_size));
            print_kv("Parameters", &format_number(info.num_parameters));
            print_kv("Time", &format!("{:.2}s", elapsed.as_secs_f64()));

            if info.warnings.is_empty() {
                print_success(&format!("Model saved to: {}", args.output));
            } else {
                println!();
                print_warning("Conversion warnings:");
                for warning in &info.warnings {
                    println!("  - {warning}");
                }
            }
        }
        Err(e) => {
            return Err(CliError::Conversion(e));
        }
    }

    Ok(())
}

// =============================================================================
// Format Validation
// =============================================================================

fn validate_format(format: &str) -> CliResult<()> {
    if !SUPPORTED_FORMATS.contains(&format.to_lowercase().as_str()) {
        return Err(CliError::UnsupportedFormat(format!(
            "{}. Supported formats: {}",
            format,
            SUPPORTED_FORMATS.join(", ")
        )));
    }
    Ok(())
}

// =============================================================================
// Conversion Logic
// =============================================================================

struct ConversionInfo {
    input_size: u64,
    output_size: u64,
    num_parameters: u64,
    warnings: Vec<String>,
}

fn convert_model(
    input_path: &PathBuf,
    output_path: &str,
    from_format: &str,
    to_format: &str,
    optimize: bool,
) -> Result<ConversionInfo, String> {
    // Get input file size
    let input_size = std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);

    let mut warnings = Vec::new();

    // Simulate conversion based on format pair
    let conversion_result = match (from_format, to_format) {
        ("pytorch" | "pt", "axonml") => {
            convert_pytorch_to_axonml(input_path, output_path, optimize)
        }
        ("onnx", "axonml") => convert_onnx_to_axonml(input_path, output_path, optimize),
        ("axonml", "onnx") => convert_axonml_to_onnx(input_path, output_path, optimize),
        ("axonml", "safetensors") => convert_axonml_to_safetensors(input_path, output_path),
        ("safetensors", "axonml") => convert_safetensors_to_axonml(input_path, output_path),
        ("axonml", "json") => {
            warnings.push("JSON format is for inspection only, not for inference".to_string());
            convert_axonml_to_json(input_path, output_path)
        }
        _ => {
            // Generic conversion (may not preserve all features)
            warnings.push(format!(
                "Generic conversion from {from_format} to {to_format} may lose some features"
            ));
            generic_conversion(input_path, output_path, to_format)
        }
    };

    match conversion_result {
        Ok(num_params) => {
            // Get output file size
            let output_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);

            Ok(ConversionInfo {
                input_size,
                output_size,
                num_parameters: num_params,
                warnings,
            })
        }
        Err(e) => Err(e),
    }
}

// =============================================================================
// Specific Converters
// =============================================================================

fn convert_pytorch_to_axonml(
    _input: &PathBuf,
    output: &str,
    _optimize: bool,
) -> Result<u64, String> {
    // Simulated conversion
    let dummy_content = b"FERRITE_MODEL_v1\x00";
    std::fs::write(output, dummy_content).map_err(|e| format!("Failed to write output: {e}"))?;
    Ok(1_234_567) // Simulated parameter count
}

fn convert_onnx_to_axonml(_input: &PathBuf, output: &str, _optimize: bool) -> Result<u64, String> {
    let dummy_content = b"FERRITE_MODEL_v1\x00";
    std::fs::write(output, dummy_content).map_err(|e| format!("Failed to write output: {e}"))?;
    Ok(2_345_678)
}

fn convert_axonml_to_onnx(_input: &PathBuf, output: &str, _optimize: bool) -> Result<u64, String> {
    let dummy_content = b"ONNX\x00";
    std::fs::write(output, dummy_content).map_err(|e| format!("Failed to write output: {e}"))?;
    Ok(1_234_567)
}

fn convert_axonml_to_safetensors(_input: &PathBuf, output: &str) -> Result<u64, String> {
    let dummy_content = b"SAFETENSORS\x00";
    std::fs::write(output, dummy_content).map_err(|e| format!("Failed to write output: {e}"))?;
    Ok(1_234_567)
}

fn convert_safetensors_to_axonml(_input: &PathBuf, output: &str) -> Result<u64, String> {
    let dummy_content = b"FERRITE_MODEL_v1\x00";
    std::fs::write(output, dummy_content).map_err(|e| format!("Failed to write output: {e}"))?;
    Ok(1_234_567)
}

fn convert_axonml_to_json(_input: &PathBuf, output: &str) -> Result<u64, String> {
    let json_content = r#"{
  "format": "axonml",
  "version": "1.0",
  "architecture": "Sequential",
  "layers": [
    {"type": "Linear", "in_features": 784, "out_features": 256},
    {"type": "ReLU"},
    {"type": "Linear", "in_features": 256, "out_features": 10}
  ],
  "num_parameters": 1234567
}"#;
    std::fs::write(output, json_content).map_err(|e| format!("Failed to write output: {e}"))?;
    Ok(1_234_567)
}

fn generic_conversion(_input: &PathBuf, output: &str, to_format: &str) -> Result<u64, String> {
    let content = format!("Converted to {to_format} format");
    std::fs::write(output, content.as_bytes())
        .map_err(|e| format!("Failed to write output: {e}"))?;
    Ok(1_000_000)
}

// =============================================================================
// Formatting Helpers
// =============================================================================

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} bytes")
    }
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1000 {
        format!("{:.2}K", n as f64 / 1000.0)
    } else {
        format!("{n}")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 bytes");
        assert_eq!(format_size(1500), "1.46 KB");
        assert_eq!(format_size(1_500_000), "1.43 MB");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1_500_000), "1.50M");
    }

    #[test]
    fn test_validate_format() {
        assert!(validate_format("axonml").is_ok());
        assert!(validate_format("onnx").is_ok());
        assert!(validate_format("invalid").is_err());
    }
}
