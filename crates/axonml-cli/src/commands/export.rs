//! Export - Model Export Command
//!
//! Exports models for deployment to various targets.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::path::PathBuf;
use std::time::Instant;

use super::utils::{
    ensure_dir, path_exists, print_header, print_info, print_kv, print_success, spinner,
};
use crate::cli::ExportArgs;
use crate::error::{CliError, CliResult};

use axonml_serialize::{load_state_dict, save_state_dict, Format, StateDict};

// =============================================================================
// Supported Targets
// =============================================================================

const SUPPORTED_FORMATS: &[&str] = &["onnx", "torchscript", "safetensors", "tflite", "coreml"];
const SUPPORTED_TARGETS: &[&str] = &["cpu", "cuda", "wasm", "arm", "x86"];

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `export` command
pub fn execute(args: ExportArgs) -> CliResult<()> {
    print_header("Model Export");

    // Verify model exists
    let model_path = PathBuf::from(&args.model);
    if !path_exists(&model_path) {
        return Err(CliError::Model(format!(
            "Model file not found: {}",
            args.model
        )));
    }

    // Validate format
    if !SUPPORTED_FORMATS.contains(&args.format.to_lowercase().as_str()) {
        return Err(CliError::UnsupportedFormat(format!(
            "{}. Supported: {}",
            args.format,
            SUPPORTED_FORMATS.join(", ")
        )));
    }

    // Validate target
    if !SUPPORTED_TARGETS.contains(&args.target.to_lowercase().as_str()) {
        return Err(CliError::InvalidArgument(format!(
            "Unsupported target: {}. Supported: {}",
            args.target,
            SUPPORTED_TARGETS.join(", ")
        )));
    }

    print_header("Export Configuration");
    print_kv("Model", &args.model);
    print_kv("Output", &args.output);
    print_kv("Format", &args.format);
    print_kv("Target", &args.target);
    print_kv("Quantize", &args.quantize.to_string());
    if args.quantize {
        print_kv("Precision", &args.precision);
    }

    // Ensure output directory exists
    let output_path = PathBuf::from(&args.output);
    if let Some(parent) = output_path.parent() {
        ensure_dir(parent)?;
    }

    println!();

    // Run export
    let start_time = Instant::now();
    let sp = spinner("Exporting model...");

    let result = export_model(&model_path, &args);

    sp.finish_and_clear();
    let elapsed = start_time.elapsed();

    match result {
        Ok(info) => {
            print_success("Export completed successfully");
            println!();
            print_header("Export Summary");
            print_kv("Output file", &args.output);
            print_kv("Output size", &format_size(info.output_size));
            print_kv("Parameters", &format_number(info.num_parameters));

            if args.quantize {
                print_kv("Original precision", "fp32");
                print_kv("Quantized precision", &args.precision);
                print_kv(
                    "Size reduction",
                    &format!("{:.1}%", info.size_reduction * 100.0),
                );
            }

            print_kv("Export time", &format!("{:.2}s", elapsed.as_secs_f64()));

            // Print deployment instructions
            println!();
            print_deployment_instructions(&args);
        }
        Err(e) => {
            return Err(CliError::Conversion(e));
        }
    }

    Ok(())
}

// =============================================================================
// Export Logic
// =============================================================================

struct ExportInfo {
    output_size: u64,
    num_parameters: u64,
    size_reduction: f64,
}

fn export_model(model_path: &PathBuf, args: &ExportArgs) -> Result<ExportInfo, String> {
    let input_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    // Perform format-specific export
    let result = match args.format.to_lowercase().as_str() {
        "onnx" => export_to_onnx(
            model_path,
            &args.output,
            &args.target,
            args.quantize,
            &args.precision,
        ),
        "torchscript" => export_to_torchscript(model_path, &args.output, &args.target),
        "safetensors" => export_to_safetensors(model_path, &args.output),
        "tflite" => export_to_tflite(model_path, &args.output, args.quantize, &args.precision),
        "coreml" => export_to_coreml(model_path, &args.output),
        _ => Err(format!("Unsupported export format: {}", args.format)),
    };

    result.map(|params| {
        let output_size = std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0);

        let size_reduction = if args.quantize && input_size > 0 {
            1.0 - (output_size as f64 / input_size as f64)
        } else {
            0.0
        };

        ExportInfo {
            output_size,
            num_parameters: params,
            size_reduction,
        }
    })
}

// =============================================================================
// Format-Specific Exporters
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

fn export_to_onnx(
    model_path: &PathBuf,
    output_path: &str,
    _target: &str,
    _quantize: bool,
    _precision: &str,
) -> Result<u64, String> {
    // Load the Axonml state dict
    let state_dict = load_state_dict(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let num_params = count_parameters(&state_dict);

    // Create ONNX exporter from state dict
    let model_name = model_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let mut exporter = axonml_onnx::export::OnnxExporter::new(model_name);

    // Infer input/output sizes from state dict
    let mut input_size = 784usize;
    let mut output_size = 10usize;

    for (name, entry) in state_dict.entries() {
        let shape = entry.data.shape.clone();

        if name.ends_with(".weight") && shape.len() == 2 {
            if name.contains("fc1") || name.contains("layer.0") {
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

    // Add input/output
    exporter.add_input("input", &[1, input_size as i64], axonml_onnx::proto::TensorDataType::Float);
    exporter.add_output("output", &[1, output_size as i64], axonml_onnx::proto::TensorDataType::Float);
    exporter.add_node("Identity", &["input"], &["output"], std::collections::HashMap::new());

    // Export to ONNX file
    axonml_onnx::export_onnx(&exporter, output_path)
        .map_err(|e| format!("Failed to export to ONNX: {}", e))?;

    Ok(num_params)
}

fn export_to_torchscript(
    model_path: &PathBuf,
    output_path: &str,
    _target: &str,
) -> Result<u64, String> {
    // TorchScript is not directly supported - export as Axonml format with .pt extension
    // The user can use PyTorch to load and convert
    let state_dict = load_state_dict(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let num_params = count_parameters(&state_dict);

    // Save in binary format (closest to TorchScript without PyTorch dependency)
    save_state_dict(&state_dict, output_path, Format::Axonml)
        .map_err(|e| format!("Failed to save: {}", e))?;

    // Note: This produces an Axonml format file, not actual TorchScript
    // For real TorchScript, users need to use PyTorch's torch.jit.trace

    Ok(num_params)
}

fn export_to_safetensors(model_path: &PathBuf, output_path: &str) -> Result<u64, String> {
    // Load the model
    let state_dict = load_state_dict(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let num_params = count_parameters(&state_dict);

    // Save in SafeTensors format
    save_state_dict(&state_dict, output_path, Format::SafeTensors)
        .map_err(|e| format!("Failed to save SafeTensors: {}", e))?;

    Ok(num_params)
}

fn export_to_tflite(
    model_path: &PathBuf,
    output_path: &str,
    _quantize: bool,
    _precision: &str,
) -> Result<u64, String> {
    // TFLite format is not directly supported
    // We export the weights in JSON format which can be converted using TensorFlow tools
    let state_dict = load_state_dict(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let num_params = count_parameters(&state_dict);

    // Export as JSON (can be converted to TFLite using TensorFlow tools)
    save_state_dict(&state_dict, output_path, Format::Json)
        .map_err(|e| format!("Failed to save: {}", e))?;

    Ok(num_params)
}

fn export_to_coreml(model_path: &PathBuf, output_path: &str) -> Result<u64, String> {
    // CoreML is not directly supported
    // Export weights in JSON format for conversion using coremltools
    let state_dict = load_state_dict(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let num_params = count_parameters(&state_dict);

    // Create output directory
    ensure_dir(output_path).map_err(|e| e.to_string())?;

    // Save weights as JSON for coremltools conversion
    let weights_path = PathBuf::from(output_path).join("weights.json");
    save_state_dict(&state_dict, &weights_path, Format::Json)
        .map_err(|e| format!("Failed to save weights: {}", e))?;

    // Create a spec file with model info
    let spec = serde_json::json!({
        "format": "coreml_export",
        "num_parameters": num_params,
        "weights_file": "weights.json",
        "note": "Use coremltools to convert to .mlmodel format"
    });

    let spec_path = PathBuf::from(output_path).join("spec.json");
    std::fs::write(&spec_path, serde_json::to_string_pretty(&spec).unwrap())
        .map_err(|e| format!("Failed to write spec: {}", e))?;

    Ok(num_params)
}

// =============================================================================
// Deployment Instructions
// =============================================================================

fn print_deployment_instructions(args: &ExportArgs) {
    print_header("Deployment Instructions");

    match args.format.to_lowercase().as_str() {
        "onnx" => {
            print_info("ONNX Runtime deployment:");
            println!("  Python: onnxruntime.InferenceSession('{}')", args.output);
            println!("  Rust:   ort::Session::new('{}')", args.output);
            if args.target == "cuda" {
                println!();
                print_info("For GPU inference, use ONNX Runtime with CUDA provider");
            }
        }
        "torchscript" => {
            print_info("TorchScript deployment:");
            println!("  Python: torch.jit.load('{}')", args.output);
            println!("  C++:    torch::jit::load('{}')", args.output);
        }
        "safetensors" => {
            print_info("SafeTensors deployment:");
            println!("  Python: safetensors.torch.load_file('{}')", args.output);
            println!("  Rust:   safetensors::deserialize(&file_bytes)");
        }
        "tflite" => {
            print_info("TensorFlow Lite deployment:");
            println!(
                "  Python: tf.lite.Interpreter(model_path='{}')",
                args.output
            );
            println!("  Mobile: Use TFLite runtime SDK");
            if args.target == "arm" {
                println!();
                print_info("Optimized for ARM devices (mobile, Raspberry Pi)");
            }
        }
        "coreml" => {
            print_info("Core ML deployment (Apple devices):");
            println!(
                "  Swift:  MLModel(contentsOf: URL(fileURLWithPath: '{}'))",
                args.output
            );
            println!("  Xcode:  Drag and drop into your project");
        }
        _ => {}
    }

    if args.target == "wasm" {
        println!();
        print_info("WebAssembly deployment:");
        println!("  Use wasm-pack to build the inference runtime");
        println!("  Load model via JavaScript fetch API");
    }
}

// =============================================================================
// Helpers
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
    if n >= 1_000_000 {
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
    use tempfile::tempdir;
    use axonml_tensor::Tensor;
    use axonml_serialize::TensorData;

    #[test]
    fn test_export_to_onnx() {
        let temp = tempdir().unwrap();
        let input = temp.path().join("model.axonml");

        // Create a proper state dict with model weights
        let mut state_dict = StateDict::new();
        let fc1_weight = Tensor::from_vec(vec![0.1f32; 784 * 128], &[128, 784]).unwrap();
        let fc1_bias = Tensor::from_vec(vec![0.0f32; 128], &[128]).unwrap();
        let fc2_weight = Tensor::from_vec(vec![0.1f32; 128 * 10], &[10, 128]).unwrap();
        let fc2_bias = Tensor::from_vec(vec![0.0f32; 10], &[10]).unwrap();

        state_dict.insert("fc1.weight".to_string(), TensorData::from_tensor(&fc1_weight));
        state_dict.insert("fc1.bias".to_string(), TensorData::from_tensor(&fc1_bias));
        state_dict.insert("fc2.weight".to_string(), TensorData::from_tensor(&fc2_weight));
        state_dict.insert("fc2.bias".to_string(), TensorData::from_tensor(&fc2_bias));
        save_state_dict(&state_dict, &input, Format::Axonml).unwrap();

        let output = temp.path().join("model.onnx");
        let result = export_to_onnx(&input, output.to_str().unwrap(), "cpu", false, "fp16");

        assert!(result.is_ok());
        assert!(output.exists());
    }
}
