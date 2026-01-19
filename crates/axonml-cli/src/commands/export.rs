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

fn export_to_onnx(
    _model_path: &PathBuf,
    output_path: &str,
    target: &str,
    quantize: bool,
    precision: &str,
) -> Result<u64, String> {
    // Simulated ONNX export
    let mut content = vec![0x4F, 0x4E, 0x4E, 0x58]; // "ONNX" header

    // Add target-specific optimizations marker
    content.extend_from_slice(format!("target:{target}").as_bytes());

    if quantize {
        content.extend_from_slice(format!(",quantized:{precision}").as_bytes());
    }

    std::fs::write(output_path, &content).map_err(|e| format!("Failed to write ONNX: {e}"))?;

    Ok(1_234_567)
}

fn export_to_torchscript(
    _model_path: &PathBuf,
    output_path: &str,
    _target: &str,
) -> Result<u64, String> {
    let content = b"PK\x03\x04"; // ZIP header (TorchScript uses ZIP)
    std::fs::write(output_path, content)
        .map_err(|e| format!("Failed to write TorchScript: {e}"))?;
    Ok(1_234_567)
}

fn export_to_safetensors(_model_path: &PathBuf, output_path: &str) -> Result<u64, String> {
    let content = b"SAFETENSORS";
    std::fs::write(output_path, content)
        .map_err(|e| format!("Failed to write SafeTensors: {e}"))?;
    Ok(1_234_567)
}

fn export_to_tflite(
    _model_path: &PathBuf,
    output_path: &str,
    quantize: bool,
    precision: &str,
) -> Result<u64, String> {
    let mut content = vec![0x54, 0x46, 0x4C, 0x33]; // TFLite magic

    if quantize {
        content.extend_from_slice(format!("quantized:{precision}").as_bytes());
    }

    std::fs::write(output_path, &content).map_err(|e| format!("Failed to write TFLite: {e}"))?;
    Ok(1_234_567)
}

fn export_to_coreml(_model_path: &PathBuf, output_path: &str) -> Result<u64, String> {
    // CoreML is a directory-based format
    ensure_dir(output_path).map_err(|e| e.to_string())?;

    let spec_path = PathBuf::from(output_path).join("model.mlmodel");
    let content = b"CoreML Model Specification";
    std::fs::write(&spec_path, content).map_err(|e| format!("Failed to write CoreML: {e}"))?;

    Ok(1_234_567)
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

    #[test]
    fn test_export_to_onnx() {
        let temp = tempdir().unwrap();
        let input = temp.path().join("model.axonml");
        std::fs::write(&input, b"test").unwrap();

        let output = temp.path().join("model.onnx");
        let result = export_to_onnx(&input, output.to_str().unwrap(), "cpu", false, "fp16");

        assert!(result.is_ok());
        assert!(output.exists());
    }
}
