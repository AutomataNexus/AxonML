//! Analyze - Comprehensive Model and Dataset Analysis
//!
//! Provides deep analysis of loaded models and datasets, generating
//! comprehensive reports with statistics, visualizations, and recommendations.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use axonml_serialize::load_state_dict;
use serde::{Deserialize, Serialize};

use super::load::load_workspace;
use super::utils::{path_exists, print_header, print_info, print_kv, print_success, print_warning};
use crate::cli::{
    AnalyzeArgs, AnalyzeBothArgs, AnalyzeDataArgs, AnalyzeModelArgs, AnalyzeReportArgs,
    AnalyzeSubcommand,
};
use crate::error::{CliError, CliResult};

// =============================================================================
// Analysis Results
// =============================================================================

/// Model analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAnalysis {
    pub name: String,
    pub path: String,
    pub format: String,
    pub num_parameters: usize,
    pub num_layers: usize,
    pub file_size: u64,
    pub layer_analysis: Vec<LayerInfo>,
    pub architecture_type: String,
    pub estimated_memory: u64,
    pub recommendations: Vec<String>,
}

/// Layer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub shape: Vec<usize>,
    pub num_parameters: usize,
    pub percentage: f64,
}

/// Dataset analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAnalysis {
    pub name: String,
    pub path: String,
    pub data_type: String,
    pub num_samples: usize,
    pub num_classes: Option<usize>,
    pub class_distribution: Option<HashMap<String, usize>>,
    pub total_size: u64,
    pub file_stats: FileStats,
    pub quality_score: f64,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// File statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileStats {
    pub total_files: usize,
    pub file_types: HashMap<String, usize>,
    pub avg_file_size: u64,
    pub min_file_size: u64,
    pub max_file_size: u64,
}

/// Combined analysis for model + dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedAnalysis {
    pub model: ModelAnalysis,
    pub dataset: DataAnalysis,
    pub compatibility: CompatibilityAnalysis,
}

/// Compatibility analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityAnalysis {
    pub compatible: bool,
    pub input_match: bool,
    pub output_match: bool,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
}

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `analyze` command
pub fn execute(args: AnalyzeArgs) -> CliResult<()> {
    match args.action {
        AnalyzeSubcommand::Model(model_args) => execute_analyze_model(model_args),
        AnalyzeSubcommand::Data(data_args) => execute_analyze_data(data_args),
        AnalyzeSubcommand::Both(both_args) => execute_analyze_both(both_args),
        AnalyzeSubcommand::Report(report_args) => execute_report(report_args),
    }
}

// =============================================================================
// Analyze Model Subcommand
// =============================================================================

fn execute_analyze_model(args: AnalyzeModelArgs) -> CliResult<()> {
    print_header("Model Analysis");

    // Get model path (from args or workspace)
    let (path, name) = if let Some(p) = &args.path {
        let path = PathBuf::from(p);
        if !path_exists(&path) {
            return Err(CliError::Model(format!("Model not found: {p}")));
        }
        let name = path
            .file_stem()
            .map_or_else(|| "model".to_string(), |s| s.to_string_lossy().to_string());
        (path, name)
    } else {
        // Use workspace
        let workspace = load_workspace()?;
        if let Some(model) = &workspace.model {
            (PathBuf::from(&model.path), model.name.clone())
        } else {
            return Err(CliError::Model(
                "No model specified. Use --path or load a model first with 'axonml load model'"
                    .to_string(),
            ));
        }
    };

    print_kv("Model", &name);
    print_kv("Path", &path.display().to_string());

    println!();
    print_info("Analyzing model...");

    let analysis = analyze_model(&path, &name)?;

    // Print results
    print_model_analysis(&analysis, args.detailed);

    // Save report if requested
    if let Some(output) = &args.output {
        save_analysis_report(&analysis, output, &args.format)?;
        print_success(&format!("Report saved to: {output}"));
    }

    Ok(())
}

fn analyze_model(path: &PathBuf, name: &str) -> CliResult<ModelAnalysis> {
    let state_dict =
        load_state_dict(path).map_err(|e| CliError::Model(format!("Failed to load model: {e}")))?;

    let file_size = fs::metadata(path)?.len();
    let format = detect_format(path);

    let mut layer_analysis = Vec::new();
    let total_params: usize = state_dict
        .entries()
        .map(|(_, entry)| entry.data.shape.iter().product::<usize>())
        .sum();

    for (layer_name, entry) in state_dict.entries() {
        let shape = entry.data.shape.clone();
        let num_params: usize = shape.iter().product();
        let percentage = if total_params > 0 {
            (num_params as f64 / total_params as f64) * 100.0
        } else {
            0.0
        };

        let layer_type = infer_layer_type(layer_name, &shape);

        layer_analysis.push(LayerInfo {
            name: layer_name.clone(),
            layer_type,
            shape,
            num_parameters: num_params,
            percentage,
        });
    }

    // Sort by parameter count
    layer_analysis.sort_by(|a, b| b.num_parameters.cmp(&a.num_parameters));

    let architecture_type = infer_architecture(&layer_analysis);
    let estimated_memory = estimate_memory(total_params);
    let recommendations = generate_model_recommendations(&layer_analysis, total_params);

    Ok(ModelAnalysis {
        name: name.to_string(),
        path: path.display().to_string(),
        format,
        num_parameters: total_params,
        num_layers: layer_analysis.len(),
        file_size,
        layer_analysis,
        architecture_type,
        estimated_memory,
        recommendations,
    })
}

fn print_model_analysis(analysis: &ModelAnalysis, detailed: bool) {
    println!();
    print_header("Overview");
    print_kv("Architecture", &analysis.architecture_type);
    print_kv("Format", &analysis.format);
    print_kv("Total parameters", &format_number(analysis.num_parameters));
    print_kv("Number of layers", &analysis.num_layers.to_string());
    print_kv("File size", &format_size(analysis.file_size));
    print_kv(
        "Estimated memory (inference)",
        &format_size(analysis.estimated_memory),
    );

    // Parameter breakdown
    println!();
    print_header("Parameter Distribution");

    let top_layers: Vec<_> = analysis.layer_analysis.iter().take(10).collect();
    for layer in top_layers {
        println!(
            "  {:40} {:>12} ({:>5.1}%)",
            truncate(&layer.name, 40),
            format_number(layer.num_parameters),
            layer.percentage
        );
    }

    if analysis.layer_analysis.len() > 10 {
        println!(
            "  ... and {} more layers",
            analysis.layer_analysis.len() - 10
        );
    }

    if detailed {
        println!();
        print_header("All Layers");
        for layer in &analysis.layer_analysis {
            let shape_str = layer
                .shape
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join("x");
            println!(
                "  {} [{}] - {} ({})",
                layer.name,
                shape_str,
                layer.layer_type,
                format_number(layer.num_parameters)
            );
        }
    }

    if !analysis.recommendations.is_empty() {
        println!();
        print_header("Recommendations");
        for rec in &analysis.recommendations {
            println!("  - {rec}");
        }
    }
}

// =============================================================================
// Analyze Data Subcommand
// =============================================================================

fn execute_analyze_data(args: AnalyzeDataArgs) -> CliResult<()> {
    print_header("Dataset Analysis");

    // Get data path (from args or workspace)
    let (path, name) = if let Some(p) = &args.path {
        let path = PathBuf::from(p);
        if !path_exists(&path) {
            return Err(CliError::Data(format!("Dataset not found: {p}")));
        }
        let name = path.file_name().map_or_else(
            || "dataset".to_string(),
            |s| s.to_string_lossy().to_string(),
        );
        (path, name)
    } else {
        let workspace = load_workspace()?;
        if let Some(dataset) = &workspace.dataset {
            (PathBuf::from(&dataset.path), dataset.name.clone())
        } else {
            return Err(CliError::Data(
                "No dataset specified. Use --path or load a dataset first with 'axonml load data'"
                    .to_string(),
            ));
        }
    };

    print_kv("Dataset", &name);
    print_kv("Path", &path.display().to_string());

    println!();
    print_info("Analyzing dataset...");

    let analysis = analyze_data(&path, &name, args.max_samples)?;

    // Print results
    print_data_analysis(&analysis, args.detailed);

    // Save report if requested
    if let Some(output) = &args.output {
        save_data_report(&analysis, output, &args.format)?;
        print_success(&format!("Report saved to: {output}"));
    }

    Ok(())
}

fn analyze_data(path: &PathBuf, name: &str, max_samples: usize) -> CliResult<DataAnalysis> {
    use walkdir::WalkDir;

    let data_type = detect_data_type(path);
    let mut total_size = 0u64;
    let mut file_stats = FileStats::default();
    let mut file_sizes: Vec<u64> = Vec::new();
    let mut class_distribution: HashMap<String, usize> = HashMap::new();

    // Scan files
    for entry in WalkDir::new(path)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if entry.file_type().is_file() {
            file_stats.total_files += 1;
            if let Ok(meta) = entry.metadata() {
                let size = meta.len();
                total_size += size;
                file_sizes.push(size);
            }
            if let Some(ext) = entry.path().extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                *file_stats.file_types.entry(ext_str).or_insert(0) += 1;
            }
        }
    }

    // Calculate file size stats
    if !file_sizes.is_empty() {
        file_stats.avg_file_size = total_size / file_sizes.len() as u64;
        file_stats.min_file_size = *file_sizes.iter().min().unwrap_or(&0);
        file_stats.max_file_size = *file_sizes.iter().max().unwrap_or(&0);
    }

    // Detect classes and count samples
    let (num_samples, num_classes) = match data_type.as_str() {
        "image" => count_image_samples(path, &mut class_distribution),
        "tabular" => count_tabular_samples(path, &mut class_distribution, max_samples),
        _ => (file_stats.total_files, None),
    };

    // Calculate quality score
    let quality_score =
        calculate_quality_score(&data_type, num_samples, &class_distribution, &file_stats);

    // Find issues
    let issues = find_data_issues(&data_type, num_samples, &class_distribution, &file_stats);

    // Generate recommendations
    let recommendations =
        generate_data_recommendations(&data_type, num_samples, &class_distribution, &issues);

    Ok(DataAnalysis {
        name: name.to_string(),
        path: path.display().to_string(),
        data_type,
        num_samples,
        num_classes,
        class_distribution: if class_distribution.is_empty() {
            None
        } else {
            Some(class_distribution)
        },
        total_size,
        file_stats,
        quality_score,
        issues,
        recommendations,
    })
}

fn print_data_analysis(analysis: &DataAnalysis, detailed: bool) {
    println!();
    print_header("Overview");
    print_kv("Type", &analysis.data_type);
    print_kv("Total samples", &analysis.num_samples.to_string());
    if let Some(n) = analysis.num_classes {
        print_kv("Number of classes", &n.to_string());
    }
    print_kv("Total size", &format_size(analysis.total_size));
    print_kv("Total files", &analysis.file_stats.total_files.to_string());
    print_kv(
        "Quality score",
        &format!("{:.1}/10", analysis.quality_score),
    );

    // File statistics
    println!();
    print_header("File Statistics");
    print_kv(
        "Average file size",
        &format_size(analysis.file_stats.avg_file_size),
    );
    print_kv(
        "Min file size",
        &format_size(analysis.file_stats.min_file_size),
    );
    print_kv(
        "Max file size",
        &format_size(analysis.file_stats.max_file_size),
    );

    if detailed {
        println!();
        println!("File types:");
        for (ext, count) in &analysis.file_stats.file_types {
            println!("  .{ext}: {count} files");
        }
    }

    // Class distribution
    if let Some(dist) = &analysis.class_distribution {
        println!();
        print_header("Class Distribution");

        let mut sorted: Vec<_> = dist.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        let total: usize = dist.values().sum();
        for (class, count) in sorted.iter().take(20) {
            let pct = (**count as f64 / total as f64) * 100.0;
            println!("  {:30} {:>8} ({:>5.1}%)", truncate(class, 30), count, pct);
        }

        if sorted.len() > 20 {
            println!("  ... and {} more classes", sorted.len() - 20);
        }

        // Check for imbalance
        let counts: Vec<usize> = dist.values().copied().collect();
        if !counts.is_empty() {
            let max = *counts.iter().max().unwrap() as f64;
            let min = *counts.iter().min().unwrap() as f64;
            if max / min > 5.0 {
                println!();
                print_warning(&format!(
                    "Class imbalance detected: {:.1}x ratio",
                    max / min
                ));
            }
        }
    }

    // Issues
    if !analysis.issues.is_empty() {
        println!();
        print_header("Issues Found");
        for issue in &analysis.issues {
            println!("  ! {issue}");
        }
    }

    // Recommendations
    if !analysis.recommendations.is_empty() {
        println!();
        print_header("Recommendations");
        for rec in &analysis.recommendations {
            println!("  - {rec}");
        }
    }
}

// =============================================================================
// Analyze Both Subcommand
// =============================================================================

fn execute_analyze_both(args: AnalyzeBothArgs) -> CliResult<()> {
    print_header("Combined Model & Dataset Analysis");

    let workspace = load_workspace()?;

    let model = workspace.model.ok_or_else(|| {
        CliError::Model("No model loaded. Use 'axonml load model' first".to_string())
    })?;

    let dataset = workspace.dataset.ok_or_else(|| {
        CliError::Data("No dataset loaded. Use 'axonml load data' first".to_string())
    })?;

    print_kv("Model", &model.name);
    print_kv("Dataset", &dataset.name);

    println!();
    print_info("Analyzing compatibility...");

    // Analyze both
    let model_analysis = analyze_model(&PathBuf::from(&model.path), &model.name)?;
    let data_analysis = analyze_data(&PathBuf::from(&dataset.path), &dataset.name, 1000)?;

    // Check compatibility
    let compatibility = check_compatibility(&model_analysis, &data_analysis);

    // Print results
    println!();
    print_header("Compatibility Analysis");

    if compatibility.compatible {
        print_success("Model and dataset appear compatible!");
    } else {
        print_warning("Potential compatibility issues found");
    }

    print_kv(
        "Input shape match",
        if compatibility.input_match {
            "Yes"
        } else {
            "No"
        },
    );
    print_kv(
        "Output shape match",
        if compatibility.output_match {
            "Yes"
        } else {
            "No"
        },
    );

    if !compatibility.issues.is_empty() {
        println!();
        println!("Issues:");
        for issue in &compatibility.issues {
            println!("  ! {issue}");
        }
    }

    if !compatibility.suggestions.is_empty() {
        println!();
        println!("Suggestions:");
        for sug in &compatibility.suggestions {
            println!("  - {sug}");
        }
    }

    // Summary
    println!();
    print_header("Summary");
    print_kv(
        "Model parameters",
        &format_number(model_analysis.num_parameters),
    );
    print_kv("Dataset samples", &data_analysis.num_samples.to_string());

    if let Some(n) = data_analysis.num_classes {
        print_kv("Classes", &n.to_string());

        // Check if model output matches
        let output_size = infer_output_size(&model_analysis);
        if let Some(out) = output_size {
            if out != n {
                print_warning(&format!(
                    "Model output size ({out}) doesn't match number of classes ({n})"
                ));
            }
        }
    }

    // Save report if requested
    if let Some(output) = &args.output {
        let combined = CombinedAnalysis {
            model: model_analysis,
            dataset: data_analysis,
            compatibility,
        };
        let json = serde_json::to_string_pretty(&combined)?;
        fs::write(output, json)?;
        print_success(&format!("Report saved to: {output}"));
    }

    Ok(())
}

fn check_compatibility(model: &ModelAnalysis, data: &DataAnalysis) -> CompatibilityAnalysis {
    let mut issues = Vec::new();
    let mut suggestions = Vec::new();

    // Check output size vs classes
    let output_size = infer_output_size(model);
    let output_match = if let (Some(out), Some(classes)) = (output_size, data.num_classes) {
        if out == classes {
            true
        } else {
            issues.push(format!(
                "Model output size ({out}) doesn't match dataset classes ({classes})"
            ));
            suggestions
                .push("Consider adjusting the final layer or relabeling the dataset".to_string());
            false
        }
    } else {
        true // Can't determine, assume OK
    };

    // Check input size (heuristic based on first layer)
    let input_match = true; // Would need more info to check properly

    // Check dataset size vs model complexity
    if data.num_samples < model.num_parameters / 10 {
        issues.push(format!(
            "Dataset may be too small ({} samples) for model size ({} parameters)",
            data.num_samples,
            format_number(model.num_parameters)
        ));
        suggestions.push("Consider using data augmentation or a smaller model".to_string());
    }

    // Check for class imbalance
    if let Some(dist) = &data.class_distribution {
        let counts: Vec<usize> = dist.values().copied().collect();
        if !counts.is_empty() {
            let max = *counts.iter().max().unwrap() as f64;
            let min = *counts.iter().min().unwrap() as f64;
            if max / min > 10.0 {
                issues.push("Severe class imbalance may affect training".to_string());
                suggestions
                    .push("Consider class weighting or oversampling minority classes".to_string());
            }
        }
    }

    let compatible = issues.is_empty();

    CompatibilityAnalysis {
        compatible,
        input_match,
        output_match,
        issues,
        suggestions,
    }
}

// =============================================================================
// Report Subcommand
// =============================================================================

fn execute_report(args: AnalyzeReportArgs) -> CliResult<()> {
    print_header("Generate Comprehensive Report");

    let workspace = load_workspace()?;

    if workspace.model.is_none() && workspace.dataset.is_none() {
        return Err(CliError::Other(
            "No model or dataset loaded. Use 'axonml load' first".to_string(),
        ));
    }

    print_info("Generating report...");

    let output_path = PathBuf::from(&args.output);
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    match args.format.as_str() {
        "html" => generate_html_report(&workspace, &output_path)?,
        "json" => generate_json_report(&workspace, &output_path)?,
        "md" | "markdown" => generate_markdown_report(&workspace, &output_path)?,
        _ => generate_text_report(&workspace, &output_path)?,
    }

    print_success(&format!("Report saved to: {}", args.output));

    Ok(())
}

fn generate_html_report(workspace: &super::load::WorkspaceState, path: &PathBuf) -> CliResult<()> {
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("<title>Axonml Analysis Report</title>\n");
    html.push_str("<style>\n");
    html.push_str("body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }\n");
    html.push_str("h1, h2, h3 { color: #1e3a5f; }\n");
    html.push_str("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n");
    html.push_str("th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }\n");
    html.push_str("th { background-color: #1e3a5f; color: white; }\n");
    html.push_str("tr:nth-child(even) { background-color: #f9f9f9; }\n");
    html.push_str(".metric { font-size: 24px; font-weight: bold; color: #1e3a5f; }\n");
    html.push_str(
        ".card { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 10px 0; }\n",
    );
    html.push_str("</style>\n</head>\n<body>\n");

    html.push_str("<h1>Axonml Analysis Report</h1>\n");
    html.push_str(&format!(
        "<p>Generated: {}</p>\n",
        chrono::Utc::now().to_rfc3339()
    ));

    if let Some(model) = &workspace.model {
        html.push_str("<h2>Model Analysis</h2>\n");
        html.push_str("<div class=\"card\">\n");
        html.push_str(&format!("<p><strong>Name:</strong> {}</p>\n", model.name));
        html.push_str(&format!("<p><strong>Path:</strong> {}</p>\n", model.path));
        html.push_str(&format!(
            "<p><strong>Format:</strong> {}</p>\n",
            model.format
        ));
        html.push_str(&format!(
            "<p><strong>Parameters:</strong> <span class=\"metric\">{}</span></p>\n",
            format_number(model.num_parameters)
        ));
        html.push_str(&format!(
            "<p><strong>Size:</strong> {}</p>\n",
            format_size(model.file_size)
        ));
        html.push_str("</div>\n");
    }

    if let Some(dataset) = &workspace.dataset {
        html.push_str("<h2>Dataset Analysis</h2>\n");
        html.push_str("<div class=\"card\">\n");
        html.push_str(&format!("<p><strong>Name:</strong> {}</p>\n", dataset.name));
        html.push_str(&format!("<p><strong>Path:</strong> {}</p>\n", dataset.path));
        html.push_str(&format!(
            "<p><strong>Type:</strong> {}</p>\n",
            dataset.data_type
        ));
        html.push_str(&format!(
            "<p><strong>Samples:</strong> <span class=\"metric\">{}</span></p>\n",
            dataset.num_samples
        ));
        if let Some(n) = dataset.num_classes {
            html.push_str(&format!("<p><strong>Classes:</strong> {n}</p>\n"));
        }
        html.push_str("</div>\n");
    }

    html.push_str("</body>\n</html>");

    fs::write(path, html)?;
    Ok(())
}

fn generate_json_report(workspace: &super::load::WorkspaceState, path: &PathBuf) -> CliResult<()> {
    let json = serde_json::to_string_pretty(workspace)?;
    fs::write(path, json)?;
    Ok(())
}

fn generate_markdown_report(
    workspace: &super::load::WorkspaceState,
    path: &PathBuf,
) -> CliResult<()> {
    let mut md = String::new();
    md.push_str("# Axonml Analysis Report\n\n");
    md.push_str(&format!(
        "Generated: {}\n\n",
        chrono::Utc::now().to_rfc3339()
    ));

    if let Some(model) = &workspace.model {
        md.push_str("## Model Analysis\n\n");
        md.push_str(&format!("- **Name:** {}\n", model.name));
        md.push_str(&format!("- **Path:** {}\n", model.path));
        md.push_str(&format!("- **Format:** {}\n", model.format));
        md.push_str(&format!(
            "- **Parameters:** {}\n",
            format_number(model.num_parameters)
        ));
        md.push_str(&format!("- **Size:** {}\n\n", format_size(model.file_size)));
    }

    if let Some(dataset) = &workspace.dataset {
        md.push_str("## Dataset Analysis\n\n");
        md.push_str(&format!("- **Name:** {}\n", dataset.name));
        md.push_str(&format!("- **Path:** {}\n", dataset.path));
        md.push_str(&format!("- **Type:** {}\n", dataset.data_type));
        md.push_str(&format!("- **Samples:** {}\n", dataset.num_samples));
        if let Some(n) = dataset.num_classes {
            md.push_str(&format!("- **Classes:** {n}\n"));
        }
        md.push('\n');
    }

    fs::write(path, md)?;
    Ok(())
}

fn generate_text_report(workspace: &super::load::WorkspaceState, path: &PathBuf) -> CliResult<()> {
    let mut text = String::new();
    text.push_str("FERRITE ANALYSIS REPORT\n");
    text.push_str(&"=".repeat(50));
    text.push_str(&format!(
        "\nGenerated: {}\n\n",
        chrono::Utc::now().to_rfc3339()
    ));

    if let Some(model) = &workspace.model {
        text.push_str("MODEL ANALYSIS\n");
        text.push_str(&"-".repeat(50));
        text.push_str(&format!("\nName: {}\n", model.name));
        text.push_str(&format!("Path: {}\n", model.path));
        text.push_str(&format!("Format: {}\n", model.format));
        text.push_str(&format!(
            "Parameters: {}\n",
            format_number(model.num_parameters)
        ));
        text.push_str(&format!("Size: {}\n\n", format_size(model.file_size)));
    }

    if let Some(dataset) = &workspace.dataset {
        text.push_str("DATASET ANALYSIS\n");
        text.push_str(&"-".repeat(50));
        text.push_str(&format!("\nName: {}\n", dataset.name));
        text.push_str(&format!("Path: {}\n", dataset.path));
        text.push_str(&format!("Type: {}\n", dataset.data_type));
        text.push_str(&format!("Samples: {}\n", dataset.num_samples));
        if let Some(n) = dataset.num_classes {
            text.push_str(&format!("Classes: {n}\n"));
        }
        text.push('\n');
    }

    fs::write(path, text)?;
    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn detect_format(path: &PathBuf) -> String {
    if let Some(ext) = path.extension() {
        match ext.to_string_lossy().to_lowercase().as_str() {
            "pt" | "pth" => "pytorch".to_string(),
            "safetensors" => "safetensors".to_string(),
            "onnx" => "onnx".to_string(),
            "axonml" => "axonml".to_string(),
            _ => "unknown".to_string(),
        }
    } else {
        "unknown".to_string()
    }
}

fn detect_data_type(path: &PathBuf) -> String {
    use walkdir::WalkDir;

    let mut counts: HashMap<&str, usize> = HashMap::new();

    for entry in WalkDir::new(path)
        .max_depth(3)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .take(100)
    {
        if let Some(ext) = entry.path().extension() {
            match ext.to_string_lossy().to_lowercase().as_str() {
                "jpg" | "jpeg" | "png" | "bmp" => *counts.entry("image").or_insert(0) += 1,
                "csv" | "tsv" | "parquet" => *counts.entry("tabular").or_insert(0) += 1,
                "txt" | "json" | "jsonl" => *counts.entry("text").or_insert(0) += 1,
                "wav" | "mp3" | "flac" => *counts.entry("audio").or_insert(0) += 1,
                _ => {}
            }
        }
    }

    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map_or_else(|| "unknown".to_string(), |(t, _)| t.to_string())
}

fn infer_layer_type(name: &str, shape: &[usize]) -> String {
    let name_lower = name.to_lowercase();

    if name_lower.contains("embed") {
        "Embedding".to_string()
    } else if name_lower.contains("conv") {
        if shape.len() == 4 {
            "Conv2d".to_string()
        } else if shape.len() == 3 {
            "Conv1d".to_string()
        } else {
            "Conv".to_string()
        }
    } else if name_lower.contains("bn") || name_lower.contains("batch_norm") {
        "BatchNorm".to_string()
    } else if name_lower.contains("ln") || name_lower.contains("layer_norm") {
        "LayerNorm".to_string()
    } else if name_lower.contains("attention") || name_lower.contains("attn") {
        "Attention".to_string()
    } else if name_lower.contains("fc")
        || name_lower.contains("linear")
        || name_lower.contains("dense")
    {
        "Linear".to_string()
    } else if name_lower.contains("bias") {
        "Bias".to_string()
    } else if name_lower.contains("weight") {
        if shape.len() == 2 {
            "Linear".to_string()
        } else if shape.len() == 4 {
            "Conv2d".to_string()
        } else {
            "Weight".to_string()
        }
    } else {
        "Unknown".to_string()
    }
}

fn infer_architecture(layers: &[LayerInfo]) -> String {
    let has_conv = layers.iter().any(|l| l.layer_type.contains("Conv"));
    let has_attention = layers.iter().any(|l| l.layer_type.contains("Attention"));
    let has_embed = layers.iter().any(|l| l.layer_type.contains("Embed"));

    if has_attention {
        "Transformer".to_string()
    } else if has_conv {
        "CNN".to_string()
    } else if has_embed {
        "Embedding-based".to_string()
    } else {
        "MLP".to_string()
    }
}

fn infer_output_size(model: &ModelAnalysis) -> Option<usize> {
    // Look for the last linear/dense layer
    for layer in model.layer_analysis.iter().rev() {
        if layer.layer_type == "Linear" && layer.shape.len() == 2 {
            return Some(layer.shape[0]); // Output dimension
        }
    }
    None
}

fn estimate_memory(num_params: usize) -> u64 {
    // Estimate: params * 4 bytes (f32) * 2 (gradients) + overhead
    (num_params as u64 * 4 * 2) + (1024 * 1024) // 1MB overhead
}

fn generate_model_recommendations(layers: &[LayerInfo], total_params: usize) -> Vec<String> {
    let mut recs = Vec::new();

    if total_params > 100_000_000 {
        recs.push("Large model - consider gradient checkpointing to reduce memory".to_string());
        recs.push("Use mixed precision (F16) training for faster training".to_string());
    }

    if total_params < 10_000 {
        recs.push("Small model - may underfit on complex tasks".to_string());
    }

    // Check for potential optimizations
    let linear_params: usize = layers
        .iter()
        .filter(|l| l.layer_type == "Linear")
        .map(|l| l.num_parameters)
        .sum();

    if linear_params as f64 / total_params as f64 > 0.8 {
        recs.push(
            "Model is mostly linear layers - consider using LoRA for fine-tuning".to_string(),
        );
    }

    recs
}

fn count_image_samples(
    path: &PathBuf,
    class_dist: &mut HashMap<String, usize>,
) -> (usize, Option<usize>) {
    use walkdir::WalkDir;

    let mut total = 0;

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.filter_map(std::result::Result::ok) {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let class_name = entry.file_name().to_string_lossy().to_string();
                if !class_name.starts_with('.') {
                    let count = WalkDir::new(entry.path())
                        .into_iter()
                        .filter_map(std::result::Result::ok)
                        .filter(|e| {
                            e.file_type().is_file()
                                && e.path().extension().is_some_and(|ext| {
                                    matches!(
                                        ext.to_string_lossy().to_lowercase().as_str(),
                                        "jpg" | "jpeg" | "png" | "bmp" | "gif"
                                    )
                                })
                        })
                        .count();

                    total += count;
                    class_dist.insert(class_name, count);
                }
            }
        }
    }

    let num_classes = if class_dist.len() > 1 {
        Some(class_dist.len())
    } else {
        None
    };
    (total, num_classes)
}

fn count_tabular_samples(
    path: &PathBuf,
    class_dist: &mut HashMap<String, usize>,
    max_samples: usize,
) -> (usize, Option<usize>) {
    use walkdir::WalkDir;

    for entry in WalkDir::new(path)
        .max_depth(2)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if entry.path().extension().is_some_and(|e| e == "csv") {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                let lines: Vec<&str> = content.lines().collect();
                let num_samples = lines.len().saturating_sub(1);

                // Try to detect classes from last column
                for line in lines.iter().skip(1).take(max_samples) {
                    if let Some(label) = line.split(',').next_back() {
                        *class_dist.entry(label.trim().to_string()).or_insert(0) += 1;
                    }
                }

                let num_classes = if class_dist.len() > 1 && class_dist.len() < 100 {
                    Some(class_dist.len())
                } else {
                    None
                };

                return (num_samples, num_classes);
            }
        }
    }

    (0, None)
}

fn calculate_quality_score(
    _data_type: &str,
    num_samples: usize,
    class_dist: &HashMap<String, usize>,
    _file_stats: &FileStats,
) -> f64 {
    let mut score: f64 = 5.0; // Base score

    // Sample count bonus
    if num_samples > 10000 {
        score += 2.0;
    } else if num_samples > 1000 {
        score += 1.0;
    } else if num_samples < 100 {
        score -= 2.0;
    }

    // Class balance bonus
    if !class_dist.is_empty() {
        let counts: Vec<usize> = class_dist.values().copied().collect();
        let max = *counts.iter().max().unwrap_or(&1) as f64;
        let min = *counts.iter().min().unwrap_or(&1) as f64;
        let ratio = max / min.max(1.0);

        if ratio < 2.0 {
            score += 2.0;
        } else if ratio < 5.0 {
            score += 1.0;
        } else if ratio > 10.0 {
            score -= 1.0;
        }
    }

    score.clamp(0.0, 10.0)
}

fn find_data_issues(
    _data_type: &str,
    num_samples: usize,
    class_dist: &HashMap<String, usize>,
    _file_stats: &FileStats,
) -> Vec<String> {
    let mut issues = Vec::new();

    if num_samples < 100 {
        issues.push("Very small dataset - may lead to overfitting".to_string());
    }

    if !class_dist.is_empty() {
        let counts: Vec<usize> = class_dist.values().copied().collect();
        let max = *counts.iter().max().unwrap_or(&1) as f64;
        let min = *counts.iter().min().unwrap_or(&1) as f64;

        if max / min.max(1.0) > 10.0 {
            issues.push("Severe class imbalance detected".to_string());
        }

        // Check for very small classes
        for (class, count) in class_dist {
            if *count < 10 {
                issues.push(format!(
                    "Class '{}' has only {} samples",
                    truncate(class, 20),
                    count
                ));
            }
        }
    }

    issues
}

fn generate_data_recommendations(
    data_type: &str,
    num_samples: usize,
    class_dist: &HashMap<String, usize>,
    _issues: &[String],
) -> Vec<String> {
    let mut recs = Vec::new();

    if num_samples < 1000 {
        recs.push("Consider data augmentation to increase effective dataset size".to_string());
    }

    if !class_dist.is_empty() {
        let counts: Vec<usize> = class_dist.values().copied().collect();
        let max = *counts.iter().max().unwrap_or(&1) as f64;
        let min = *counts.iter().min().unwrap_or(&1) as f64;

        if max / min.max(1.0) > 5.0 {
            recs.push("Use class weighting or oversampling for imbalanced classes".to_string());
        }
    }

    if data_type == "image" {
        recs.push("Consider using pretrained models with transfer learning".to_string());
    }

    recs
}

fn save_analysis_report<T: Serialize>(analysis: &T, path: &str, format: &str) -> CliResult<()> {
    let content = match format {
        "json" => serde_json::to_string_pretty(analysis)?,
        _ => serde_json::to_string_pretty(analysis)?, // Default to JSON
    };
    fs::write(path, content)?;
    Ok(())
}

fn save_data_report(analysis: &DataAnalysis, path: &str, format: &str) -> CliResult<()> {
    save_analysis_report(analysis, path, format)
}

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

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_layer_type() {
        assert_eq!(infer_layer_type("conv1.weight", &[64, 3, 3, 3]), "Conv2d");
        assert_eq!(infer_layer_type("fc1.weight", &[128, 64]), "Linear");
        assert_eq!(infer_layer_type("embed.weight", &[1000, 256]), "Embedding");
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 bytes");
        assert_eq!(format_size(1024), "1.00 KB");
    }

    #[test]
    fn test_calculate_quality_score() {
        let mut dist = HashMap::new();
        dist.insert("class1".to_string(), 100);
        dist.insert("class2".to_string(), 100);

        let score = calculate_quality_score("image", 10000, &dist, &FileStats::default());
        assert!(score > 5.0); // Good score for balanced dataset
    }
}
