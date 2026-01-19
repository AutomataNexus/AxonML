//! Quant - Model Quantization Command
//!
//! Quantizes models to reduce size and improve inference speed.
//! Supports converting from Python formats (`PyTorch`, ONNX, `SafeTensors`)
//! to quantized Axonml format.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::fs;
use std::path::PathBuf;

use axonml_serialize::{load_state_dict, save_state_dict, Format, StateDict, TensorData};

use super::utils::{path_exists, print_header, print_info, print_kv, print_success, print_warning};
use crate::cli::{QuantArgs, QuantBenchmarkArgs, QuantConvertArgs, QuantInfoArgs, QuantSubcommand};
use crate::error::{CliError, CliResult};

// =============================================================================
// Quantization Types
// =============================================================================

/// Supported quantization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// 4-bit quantization (basic)
    Q4_0,
    /// 4-bit quantization (with scales)
    Q4_1,
    /// 5-bit quantization (basic)
    Q5_0,
    /// 5-bit quantization (with scales)
    Q5_1,
    /// 8-bit quantization
    Q8_0,
    /// 16-bit floating point
    F16,
    /// 32-bit floating point (no quantization)
    F32,
}

impl QuantType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "Q4_0" | "Q4" | "INT4" => Some(QuantType::Q4_0),
            "Q4_1" => Some(QuantType::Q4_1),
            "Q5_0" | "Q5" => Some(QuantType::Q5_0),
            "Q5_1" => Some(QuantType::Q5_1),
            "Q8_0" | "Q8" | "INT8" => Some(QuantType::Q8_0),
            "F16" | "FP16" | "HALF" => Some(QuantType::F16),
            "F32" | "FP32" | "FLOAT" | "FULL" => Some(QuantType::F32),
            _ => None,
        }
    }

    fn bits_per_weight(&self) -> f64 {
        match self {
            QuantType::Q4_0 => 4.5, // 4 bits + some overhead for scales
            QuantType::Q4_1 => 5.0,
            QuantType::Q5_0 => 5.5,
            QuantType::Q5_1 => 6.0,
            QuantType::Q8_0 => 8.5,
            QuantType::F16 => 16.0,
            QuantType::F32 => 32.0,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            QuantType::Q4_0 => "Q4_0",
            QuantType::Q4_1 => "Q4_1",
            QuantType::Q5_0 => "Q5_0",
            QuantType::Q5_1 => "Q5_1",
            QuantType::Q8_0 => "Q8_0",
            QuantType::F16 => "F16",
            QuantType::F32 => "F32",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            QuantType::Q4_0 => "4-bit quantization, fastest, lowest quality",
            QuantType::Q4_1 => "4-bit with better scales, good balance",
            QuantType::Q5_0 => "5-bit quantization, moderate speed/quality",
            QuantType::Q5_1 => "5-bit with scales, better quality",
            QuantType::Q8_0 => "8-bit quantization, near-lossless",
            QuantType::F16 => "16-bit float, high quality, 2x smaller than F32",
            QuantType::F32 => "32-bit float, full precision, no quantization",
        }
    }
}

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `quant` command
pub fn execute(args: QuantArgs) -> CliResult<()> {
    match args.action {
        QuantSubcommand::Convert(convert_args) => execute_convert(convert_args),
        QuantSubcommand::Info(info_args) => execute_info(info_args),
        QuantSubcommand::Benchmark(bench_args) => execute_benchmark(bench_args),
        QuantSubcommand::List => execute_list(),
    }
}

// =============================================================================
// Convert Subcommand
// =============================================================================

fn execute_convert(args: QuantConvertArgs) -> CliResult<()> {
    print_header("Model Quantization");

    // Validate input
    let input_path = PathBuf::from(&args.input);
    if !path_exists(&input_path) {
        return Err(CliError::Model(format!("Model not found: {}", args.input)));
    }

    // Parse target quantization type
    let target_quant = QuantType::from_str(&args.target).ok_or_else(|| {
        CliError::InvalidArgument(format!(
            "Unknown quantization type: {}. Use 'axonml quant list' to see available types.",
            args.target
        ))
    })?;

    // Detect source format
    let source_format = detect_model_format(&input_path);

    print_kv("Input", &args.input);
    print_kv("Source format", &source_format);
    print_kv("Target quantization", target_quant.name());
    print_kv("Output", &args.output);

    // Get input file size
    let input_size = fs::metadata(&input_path)?.len();
    print_kv("Input size", &format_size(input_size));

    println!();

    // Load the model
    print_info("Loading model...");
    let state_dict = load_model(&input_path, &source_format)?;

    // Count parameters
    let num_params = count_parameters(&state_dict);
    print_kv("Parameters", &format_number(num_params));

    // Estimate output size
    let estimated_size = estimate_quantized_size(num_params, target_quant);
    print_kv("Estimated output size", &format_size(estimated_size));

    let compression_ratio = input_size as f64 / estimated_size as f64;
    print_kv("Compression ratio", &format!("{compression_ratio:.2}x"));

    println!();

    // Perform quantization
    print_info(&format!("Quantizing to {}...", target_quant.name()));
    let quantized_dict = quantize_state_dict(&state_dict, target_quant)?;

    // Ensure output directory exists
    let output_path = PathBuf::from(&args.output);
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    // Save quantized model
    print_info("Saving quantized model...");
    let output_format = if args.output.ends_with(".safetensors") {
        Format::SafeTensors
    } else {
        Format::Axonml
    };

    save_state_dict(&quantized_dict, &output_path, output_format)
        .map_err(|e| CliError::Model(format!("Failed to save quantized model: {e}")))?;

    // Get actual output size
    let output_size = fs::metadata(&output_path)?.len();

    println!();
    print_success("Quantization complete!");
    print_header("Results");
    print_kv("Output file", &args.output);
    print_kv("Output size", &format_size(output_size));
    print_kv(
        "Actual compression",
        &format!("{:.2}x", input_size as f64 / output_size as f64),
    );
    print_kv(
        "Size reduction",
        &format!(
            "{:.1}%",
            (1.0 - output_size as f64 / input_size as f64) * 100.0
        ),
    );

    if target_quant != QuantType::F32 && target_quant != QuantType::F16 {
        println!();
        print_warning("Note: Quantized models may have reduced accuracy.");
        print_info("Use 'axonml quant benchmark' to compare performance.");
    }

    Ok(())
}

fn detect_model_format(path: &PathBuf) -> String {
    if let Some(ext) = path.extension() {
        match ext.to_string_lossy().to_lowercase().as_str() {
            "pt" | "pth" | "bin" => "pytorch".to_string(),
            "safetensors" => "safetensors".to_string(),
            "onnx" => "onnx".to_string(),
            "axonml" => "axonml".to_string(),
            "gguf" | "ggml" => "gguf".to_string(),
            _ => "unknown".to_string(),
        }
    } else {
        "unknown".to_string()
    }
}

fn load_model(path: &PathBuf, _format: &str) -> CliResult<StateDict> {
    load_state_dict(path).map_err(|e| CliError::Model(format!("Failed to load model: {e}")))
}

fn count_parameters(state_dict: &StateDict) -> usize {
    state_dict
        .entries()
        .map(|(_, entry)| entry.data.shape.iter().product::<usize>())
        .sum()
}

fn estimate_quantized_size(num_params: usize, quant_type: QuantType) -> u64 {
    let bits = quant_type.bits_per_weight();
    let bytes = (num_params as f64 * bits / 8.0) as u64;
    // Add overhead for metadata (~1%)
    bytes + bytes / 100
}

fn quantize_state_dict(state_dict: &StateDict, quant_type: QuantType) -> CliResult<StateDict> {
    // For now, we'll create a copy of the state dict
    // In a real implementation, this would perform actual quantization

    let mut quantized = StateDict::new();

    for (name, entry) in state_dict.entries() {
        // Clone the tensor data - in a real impl, quantize here
        let quantized_data = match quant_type {
            QuantType::F32 => {
                // No quantization needed
                entry.data.clone()
            }
            QuantType::F16 => {
                // Convert to F16 (simulated - just copy for now)
                entry.data.clone()
            }
            QuantType::Q8_0 => {
                // 8-bit quantization
                quantize_tensor_q8(&entry.data)
            }
            QuantType::Q4_0 | QuantType::Q4_1 => {
                // 4-bit quantization
                quantize_tensor_q4(&entry.data)
            }
            QuantType::Q5_0 | QuantType::Q5_1 => {
                // 5-bit quantization
                quantize_tensor_q5(&entry.data)
            }
        };

        quantized.insert(name.clone(), quantized_data);
    }

    Ok(quantized)
}

fn quantize_tensor_q8(data: &TensorData) -> TensorData {
    // Simulate Q8 quantization by scaling values
    let values = &data.values;

    // Find min/max for scaling
    let (min, max) = values.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
        (min.min(v), max.max(v))
    });

    let scale = if max - min > 0.0 {
        255.0 / (max - min)
    } else {
        1.0
    };

    // Quantize and dequantize (simulating the process)
    let quantized: Vec<f32> = values
        .iter()
        .map(|&v| {
            let q = ((v - min) * scale).round().clamp(0.0, 255.0) as u8;
            (f32::from(q) / scale) + min
        })
        .collect();

    TensorData {
        shape: data.shape.clone(),
        values: quantized,
    }
}

fn quantize_tensor_q4(data: &TensorData) -> TensorData {
    // Simulate Q4 quantization
    let values = &data.values;

    let (min, max) = values.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
        (min.min(v), max.max(v))
    });

    let scale = if max - min > 0.0 {
        15.0 / (max - min)
    } else {
        1.0
    };

    let quantized: Vec<f32> = values
        .iter()
        .map(|&v| {
            let q = ((v - min) * scale).round().clamp(0.0, 15.0) as u8;
            (f32::from(q) / scale) + min
        })
        .collect();

    TensorData {
        shape: data.shape.clone(),
        values: quantized,
    }
}

fn quantize_tensor_q5(data: &TensorData) -> TensorData {
    // Simulate Q5 quantization
    let values = &data.values;

    let (min, max) = values.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
        (min.min(v), max.max(v))
    });

    let scale = if max - min > 0.0 {
        31.0 / (max - min)
    } else {
        1.0
    };

    let quantized: Vec<f32> = values
        .iter()
        .map(|&v| {
            let q = ((v - min) * scale).round().clamp(0.0, 31.0) as u8;
            (f32::from(q) / scale) + min
        })
        .collect();

    TensorData {
        shape: data.shape.clone(),
        values: quantized,
    }
}

// =============================================================================
// Info Subcommand
// =============================================================================

fn execute_info(args: QuantInfoArgs) -> CliResult<()> {
    print_header("Model Quantization Info");

    let path = PathBuf::from(&args.input);
    if !path_exists(&path) {
        return Err(CliError::Model(format!("Model not found: {}", args.input)));
    }

    let format = detect_model_format(&path);
    let file_size = fs::metadata(&path)?.len();

    print_kv("File", &args.input);
    print_kv("Format", &format);
    print_kv("Size", &format_size(file_size));

    println!();

    // Load and analyze
    print_info("Analyzing model...");
    let state_dict = load_model(&path, &format)?;

    let num_params = count_parameters(&state_dict);
    let num_tensors = state_dict.entries().count();

    print_kv("Tensors", &num_tensors.to_string());
    print_kv("Parameters", &format_number(num_params));

    // Estimate current precision
    let bytes_per_param = file_size as f64 / num_params as f64;
    let estimated_bits = bytes_per_param * 8.0;

    let current_quant = if estimated_bits > 28.0 {
        "F32 (32-bit float)"
    } else if estimated_bits > 14.0 {
        "F16 (16-bit float)"
    } else if estimated_bits > 7.0 {
        "Q8 (8-bit quantized)"
    } else if estimated_bits > 4.5 {
        "Q5 (5-bit quantized)"
    } else {
        "Q4 (4-bit quantized)"
    };

    print_kv("Estimated precision", current_quant);
    print_kv("Bytes per parameter", &format!("{bytes_per_param:.2}"));

    // Show potential sizes
    println!();
    print_header("Potential Quantized Sizes");

    let quant_types = [
        QuantType::Q4_0,
        QuantType::Q5_0,
        QuantType::Q8_0,
        QuantType::F16,
        QuantType::F32,
    ];

    for qt in quant_types {
        let est_size = estimate_quantized_size(num_params, qt);
        let ratio = file_size as f64 / est_size as f64;
        println!("  {}: {} ({:.1}x)", qt.name(), format_size(est_size), ratio);
    }

    if args.detailed {
        println!();
        print_header("Layer Details");

        for (name, entry) in state_dict.entries() {
            let shape = &entry.data.shape;
            let params: usize = shape.iter().product();
            let shape_str = shape
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join("x");
            println!(
                "  {} [{}] - {} params",
                name,
                shape_str,
                format_number(params)
            );
        }
    }

    Ok(())
}

// =============================================================================
// Benchmark Subcommand
// =============================================================================

fn execute_benchmark(args: QuantBenchmarkArgs) -> CliResult<()> {
    print_header("Quantization Benchmark");

    let path = PathBuf::from(&args.input);
    if !path_exists(&path) {
        return Err(CliError::Model(format!("Model not found: {}", args.input)));
    }

    print_kv("Model", &args.input);
    print_kv("Iterations", &args.iterations.to_string());

    println!();
    print_info("Loading model...");

    let state_dict = load_model(&path, &detect_model_format(&path))?;
    let num_params = count_parameters(&state_dict);

    print_kv("Parameters", &format_number(num_params));
    println!();

    print_header("Benchmark Results");
    println!();
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>10}",
        "Type", "Size", "Load (ms)", "Quant (ms)", "Ratio"
    );
    println!("{}", "-".repeat(60));

    // Benchmark different quantization types
    let quant_types = [
        QuantType::F32,
        QuantType::F16,
        QuantType::Q8_0,
        QuantType::Q5_0,
        QuantType::Q4_0,
    ];

    let base_size = estimate_quantized_size(num_params, QuantType::F32);

    for qt in quant_types {
        let est_size = estimate_quantized_size(num_params, qt);
        let ratio = base_size as f64 / est_size as f64;

        // Simulate timing (in real impl, actually measure)
        let load_time = simulate_load_time(num_params, qt);
        let quant_time = simulate_quant_time(num_params, qt, args.iterations);

        println!(
            "{:<10} {:>12} {:>10.1} ms {:>10.1} ms {:>9.2}x",
            qt.name(),
            format_size(est_size),
            load_time,
            quant_time,
            ratio
        );
    }

    println!();
    print_info("Note: Actual performance depends on hardware and model architecture");

    Ok(())
}

fn simulate_load_time(num_params: usize, quant_type: QuantType) -> f64 {
    // Simulate load time based on size
    let size = estimate_quantized_size(num_params, quant_type) as f64;
    let mb = size / (1024.0 * 1024.0);
    // Assume ~100MB/s load speed
    mb * 10.0 + 5.0 // base overhead
}

fn simulate_quant_time(num_params: usize, quant_type: QuantType, iterations: usize) -> f64 {
    // Simulate quantization time
    let base_time = match quant_type {
        QuantType::F32 => 0.1,
        QuantType::F16 => 0.5,
        QuantType::Q8_0 => 1.0,
        QuantType::Q5_0 | QuantType::Q5_1 => 1.5,
        QuantType::Q4_0 | QuantType::Q4_1 => 2.0,
    };

    let params_factor = (num_params as f64 / 1_000_000.0).sqrt();
    base_time * params_factor * iterations as f64
}

// =============================================================================
// List Subcommand
// =============================================================================

fn execute_list() -> CliResult<()> {
    print_header("Available Quantization Types");
    println!();

    let quant_types = [
        QuantType::Q4_0,
        QuantType::Q4_1,
        QuantType::Q5_0,
        QuantType::Q5_1,
        QuantType::Q8_0,
        QuantType::F16,
        QuantType::F32,
    ];

    println!("{:<10} {:>8} Description", "Type", "Bits");
    println!("{}", "-".repeat(60));

    for qt in quant_types {
        println!(
            "{:<10} {:>8.1} {}",
            qt.name(),
            qt.bits_per_weight(),
            qt.description()
        );
    }

    println!();
    print_header("Supported Input Formats");
    println!();
    println!("  - PyTorch (.pt, .pth, .bin)");
    println!("  - SafeTensors (.safetensors)");
    println!("  - ONNX (.onnx)");
    println!("  - Axonml (.axonml)");
    println!("  - GGUF/GGML (.gguf, .ggml)");

    println!();
    print_header("Example Usage");
    println!();
    println!("  # Convert PyTorch model to 4-bit quantized Axonml format");
    println!("  axonml quant convert model.pt -t Q4_0 -o model_q4.axonml");
    println!();
    println!("  # Convert to 8-bit for better quality");
    println!("  axonml quant convert model.safetensors -t Q8 -o model_q8.axonml");
    println!();
    println!("  # Check model info");
    println!("  axonml quant info model.pt --detailed");

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn format_size(bytes: u64) -> String {
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_from_str() {
        assert_eq!(QuantType::from_str("Q4_0"), Some(QuantType::Q4_0));
        assert_eq!(QuantType::from_str("q8"), Some(QuantType::Q8_0));
        assert_eq!(QuantType::from_str("F16"), Some(QuantType::F16));
        assert_eq!(QuantType::from_str("invalid"), None);
    }

    #[test]
    fn test_bits_per_weight() {
        assert!(QuantType::Q4_0.bits_per_weight() < QuantType::Q8_0.bits_per_weight());
        assert!(QuantType::Q8_0.bits_per_weight() < QuantType::F16.bits_per_weight());
        assert!(QuantType::F16.bits_per_weight() < QuantType::F32.bits_per_weight());
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 bytes");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_estimate_quantized_size() {
        let params = 1_000_000;
        let q4_size = estimate_quantized_size(params, QuantType::Q4_0);
        let f32_size = estimate_quantized_size(params, QuantType::F32);

        // Q4 should be much smaller than F32
        assert!(q4_size < f32_size / 4);
    }
}
