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

use axonml_serialize::{convert_from_pytorch, load_state_dict, save_state_dict, Format, StateDict};

// =============================================================================
// Supported Formats
// =============================================================================

const SUPPORTED_FORMATS: &[&str] = &["axonml", "onnx", "pytorch", "safetensors", "json", "binary"];

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

/// Count total parameters in a state dict
fn count_parameters(state_dict: &StateDict) -> u64 {
    let mut total = 0u64;
    for (_, entry) in state_dict.entries() {
        let count: u64 = entry.data.shape.iter().map(|&s| s as u64).product();
        total += count;
    }
    total
}

fn convert_pytorch_to_axonml(
    input: &PathBuf,
    output: &str,
    _optimize: bool,
) -> Result<u64, String> {
    // Load the PyTorch state dict (assuming it's in a compatible format)
    let state_dict =
        load_state_dict(input).map_err(|e| format!("Failed to load PyTorch model: {}", e))?;

    // Convert key names from PyTorch convention
    let converted = convert_from_pytorch(&state_dict);

    // Save in Axonml format
    save_state_dict(&converted, output, Format::Axonml)
        .map_err(|e| format!("Failed to save Axonml model: {}", e))?;

    Ok(count_parameters(&converted))
}

fn convert_onnx_to_axonml(input: &PathBuf, output: &str, _optimize: bool) -> Result<u64, String> {
    // Use the ONNX parser to load the model
    let onnx_model =
        axonml_onnx::import_onnx(input).map_err(|e| format!("Failed to load ONNX model: {}", e))?;

    // Extract weights from the ONNX model into a state dict
    let state_dict = onnx_model.to_state_dict();

    // Save in Axonml format
    save_state_dict(&state_dict, output, Format::Axonml)
        .map_err(|e| format!("Failed to save Axonml model: {}", e))?;

    Ok(count_parameters(&state_dict))
}

fn convert_axonml_to_onnx(input: &PathBuf, output: &str, _optimize: bool) -> Result<u64, String> {
    // Load the Axonml state dict
    let state_dict =
        load_state_dict(input).map_err(|e| format!("Failed to load Axonml model: {}", e))?;

    let num_params = count_parameters(&state_dict);

    // Create ONNX exporter from state dict
    let model_name = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let mut exporter = axonml_onnx::export::OnnxExporter::new(model_name);

    // Infer network structure from state dict entries
    let mut layers: Vec<(String, Vec<usize>)> = Vec::new();
    let mut input_size = 0usize;
    let mut output_size = 0usize;

    for (name, entry) in state_dict.entries() {
        let shape = entry.data.shape.clone();
        layers.push((name.clone(), shape.clone()));

        // Try to infer input/output from weight shapes
        if name.ends_with(".weight") && shape.len() == 2 {
            if name.contains("fc1") || name.contains("layer.0") || name.contains("encoder") {
                input_size = shape[1];
            }
            if name.contains("fc") || name.contains("classifier") || name.contains("head") {
                output_size = shape[0];
            }
        }

        // Add weights as initializers
        let tensor = axonml_tensor::Tensor::from_vec(entry.data.values.clone(), &shape)
            .map_err(|e| format!("Failed to create tensor: {:?}", e))?;
        exporter.add_initializer(&name, &tensor);
    }

    // Set default sizes if not found
    if input_size == 0 {
        input_size = 784;
    }
    if output_size == 0 {
        output_size = 10;
    }

    // Add input/output
    exporter.add_input(
        "input",
        &[1, input_size as i64],
        axonml_onnx::proto::TensorDataType::Float,
    );
    exporter.add_output(
        "output",
        &[1, output_size as i64],
        axonml_onnx::proto::TensorDataType::Float,
    );

    // Add identity node to connect input to output (minimal graph)
    exporter.add_node(
        "Identity",
        &["input"],
        &["output"],
        std::collections::HashMap::new(),
    );

    // Export to ONNX file
    axonml_onnx::export_onnx(&exporter, output)
        .map_err(|e| format!("Failed to export to ONNX: {}", e))?;

    Ok(num_params)
}

fn convert_axonml_to_safetensors(input: &PathBuf, output: &str) -> Result<u64, String> {
    // Load Axonml state dict
    let state_dict =
        load_state_dict(input).map_err(|e| format!("Failed to load Axonml model: {}", e))?;

    // Save in SafeTensors format
    save_state_dict(&state_dict, output, Format::SafeTensors)
        .map_err(|e| format!("Failed to save SafeTensors: {}", e))?;

    Ok(count_parameters(&state_dict))
}

fn convert_safetensors_to_axonml(input: &PathBuf, output: &str) -> Result<u64, String> {
    // Load SafeTensors format
    let state_dict =
        load_state_dict(input).map_err(|e| format!("Failed to load SafeTensors: {}", e))?;

    // Save in Axonml format
    save_state_dict(&state_dict, output, Format::Axonml)
        .map_err(|e| format!("Failed to save Axonml model: {}", e))?;

    Ok(count_parameters(&state_dict))
}

fn convert_axonml_to_json(input: &PathBuf, output: &str) -> Result<u64, String> {
    // Load Axonml state dict
    let state_dict =
        load_state_dict(input).map_err(|e| format!("Failed to load Axonml model: {}", e))?;

    // Save in JSON format
    save_state_dict(&state_dict, output, Format::Json)
        .map_err(|e| format!("Failed to save JSON: {}", e))?;

    Ok(count_parameters(&state_dict))
}

fn generic_conversion(input: &PathBuf, output: &str, to_format: &str) -> Result<u64, String> {
    // Try to load as any supported format
    let state_dict = load_state_dict(input).map_err(|e| format!("Failed to load model: {}", e))?;

    // Determine output format
    let format = match to_format.to_lowercase().as_str() {
        "axonml" | "binary" => Format::Axonml,
        "json" => Format::Json,
        "safetensors" => Format::SafeTensors,
        _ => return Err(format!("Unsupported output format: {}", to_format)),
    };

    // Save in target format
    save_state_dict(&state_dict, output, format)
        .map_err(|e| format!("Failed to save model: {}", e))?;

    Ok(count_parameters(&state_dict))
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
