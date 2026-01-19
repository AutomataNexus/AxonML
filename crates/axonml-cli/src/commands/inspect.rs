//! Inspect - Model Inspection Command
//!
//! Displays detailed information about a model's architecture and parameters.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::path::PathBuf;

use super::utils::{detect_model_format, path_exists, print_header, print_info, print_kv};
use crate::cli::InspectArgs;
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `inspect` command
pub fn execute(args: InspectArgs) -> CliResult<()> {
    // Verify model exists
    let model_path = PathBuf::from(&args.model);
    if !path_exists(&model_path) {
        return Err(CliError::Model(format!(
            "Model file not found: {}",
            args.model
        )));
    }

    // Detect format
    let format = detect_model_format(&model_path).unwrap_or_else(|| "unknown".to_string());

    // Load and inspect model
    let info = inspect_model(&model_path, &format)?;

    // Output based on format
    match args.format.to_lowercase().as_str() {
        "json" => output_json(&info, args.detailed)?,
        _ => output_text(&info, &args, &format)?,
    }

    Ok(())
}

// =============================================================================
// Model Information
// =============================================================================

#[derive(Debug)]
struct ModelInfo {
    name: String,
    format: String,
    file_size: u64,
    num_parameters: u64,
    num_layers: usize,
    layers: Vec<LayerInfo>,
    metadata: Vec<(String, String)>,
}

#[derive(Debug)]
struct LayerInfo {
    index: usize,
    name: String,
    layer_type: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    num_params: u64,
    trainable: bool,
}

fn inspect_model(path: &PathBuf, format: &str) -> CliResult<ModelInfo> {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let name = path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();

    // Simulated model inspection
    let layers = vec![
        LayerInfo {
            index: 0,
            name: "conv1".to_string(),
            layer_type: "Conv2d".to_string(),
            input_shape: vec![1, 3, 224, 224],
            output_shape: vec![1, 64, 112, 112],
            num_params: 9472,
            trainable: true,
        },
        LayerInfo {
            index: 1,
            name: "bn1".to_string(),
            layer_type: "BatchNorm2d".to_string(),
            input_shape: vec![1, 64, 112, 112],
            output_shape: vec![1, 64, 112, 112],
            num_params: 128,
            trainable: true,
        },
        LayerInfo {
            index: 2,
            name: "relu".to_string(),
            layer_type: "ReLU".to_string(),
            input_shape: vec![1, 64, 112, 112],
            output_shape: vec![1, 64, 112, 112],
            num_params: 0,
            trainable: false,
        },
        LayerInfo {
            index: 3,
            name: "maxpool".to_string(),
            layer_type: "MaxPool2d".to_string(),
            input_shape: vec![1, 64, 112, 112],
            output_shape: vec![1, 64, 56, 56],
            num_params: 0,
            trainable: false,
        },
        LayerInfo {
            index: 4,
            name: "layer1.0.conv1".to_string(),
            layer_type: "Conv2d".to_string(),
            input_shape: vec![1, 64, 56, 56],
            output_shape: vec![1, 64, 56, 56],
            num_params: 36928,
            trainable: true,
        },
        LayerInfo {
            index: 5,
            name: "fc".to_string(),
            layer_type: "Linear".to_string(),
            input_shape: vec![1, 512],
            output_shape: vec![1, 1000],
            num_params: 513000,
            trainable: true,
        },
    ];

    let num_params: u64 = layers.iter().map(|l| l.num_params).sum();

    let metadata = vec![
        ("version".to_string(), "1.0.0".to_string()),
        ("framework".to_string(), "axonml".to_string()),
        ("created".to_string(), "2026-01-19".to_string()),
        ("dtype".to_string(), "float32".to_string()),
    ];

    Ok(ModelInfo {
        name,
        format: format.to_string(),
        file_size,
        num_parameters: num_params,
        num_layers: layers.len(),
        layers,
        metadata,
    })
}

// =============================================================================
// Text Output
// =============================================================================

fn output_text(info: &ModelInfo, args: &InspectArgs, format: &str) -> CliResult<()> {
    print_header(&format!("Model: {}", info.name));

    print_kv("Format", format);
    print_kv("File size", &format_size(info.file_size));
    print_kv("Parameters", &format_params(info.num_parameters));
    print_kv("Layers", &info.num_layers.to_string());

    // Metadata
    if !info.metadata.is_empty() {
        println!();
        print_info("Metadata:");
        for (key, value) in &info.metadata {
            print_kv(&format!("  {key}"), value);
        }
    }

    // Layer summary
    println!();
    print_header("Architecture");

    if args.detailed {
        // Detailed layer view
        println!(
            "{:<6} {:<25} {:<15} {:<20} {:<20} {:<12}",
            "Index", "Name", "Type", "Input Shape", "Output Shape", "Params"
        );
        println!("{}", "-".repeat(100));

        for layer in &info.layers {
            println!(
                "{:<6} {:<25} {:<15} {:<20} {:<20} {:<12}",
                layer.index,
                truncate(&layer.name, 24),
                &layer.layer_type,
                format_shape(&layer.input_shape),
                format_shape(&layer.output_shape),
                format_params(layer.num_params),
            );
        }
    } else {
        // Compact view
        println!("{:<25} {:<15} {:<12}", "Layer", "Type", "Params");
        println!("{}", "-".repeat(55));

        for layer in &info.layers {
            println!(
                "{:<25} {:<15} {:<12}",
                truncate(&layer.name, 24),
                &layer.layer_type,
                format_params(layer.num_params),
            );
        }
    }

    // Parameter breakdown
    println!();
    print_header("Parameter Summary");

    let trainable: u64 = info
        .layers
        .iter()
        .filter(|l| l.trainable)
        .map(|l| l.num_params)
        .sum();
    let non_trainable = info.num_parameters - trainable;

    print_kv("Total parameters", &format_params(info.num_parameters));
    print_kv("Trainable", &format_params(trainable));
    print_kv("Non-trainable", &format_params(non_trainable));

    // Memory estimate
    let memory_bytes = info.num_parameters * 4; // Assuming float32
    print_kv("Memory (fp32)", &format_size(memory_bytes));
    print_kv("Memory (fp16)", &format_size(memory_bytes / 2));

    // Show sample parameters if requested
    if let Some(n) = args.show_params {
        println!();
        print_header(&format!("Sample Parameters (first {n} per layer)"));
        print_info("Parameter values not shown in simulation mode");
    }

    Ok(())
}

// =============================================================================
// JSON Output
// =============================================================================

fn output_json(info: &ModelInfo, detailed: bool) -> CliResult<()> {
    use serde_json::json;

    let layers_json: Vec<serde_json::Value> = if detailed {
        info.layers
            .iter()
            .map(|l| {
                json!({
                    "index": l.index,
                    "name": l.name,
                    "type": l.layer_type,
                    "input_shape": l.input_shape,
                    "output_shape": l.output_shape,
                    "parameters": l.num_params,
                    "trainable": l.trainable,
                })
            })
            .collect()
    } else {
        info.layers
            .iter()
            .map(|l| {
                json!({
                    "name": l.name,
                    "type": l.layer_type,
                    "parameters": l.num_params,
                })
            })
            .collect()
    };

    let output = json!({
        "name": info.name,
        "format": info.format,
        "file_size": info.file_size,
        "num_parameters": info.num_parameters,
        "num_layers": info.num_layers,
        "layers": layers_json,
        "metadata": info.metadata.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<std::collections::HashMap<_, _>>(),
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
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
        format!("{bytes} B")
    }
}

fn format_params(n: u64) -> String {
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

fn format_shape(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(std::string::ToString::to_string).collect();
    format!("[{}]", parts.join(", "))
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1500), "1.50K");
        assert_eq!(format_params(1_500_000), "1.50M");
    }

    #[test]
    fn test_format_shape() {
        assert_eq!(format_shape(&[1, 3, 224, 224]), "[1, 3, 224, 224]");
        assert_eq!(format_shape(&[512]), "[512]");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a very long string", 10), "this is...");
    }
}
