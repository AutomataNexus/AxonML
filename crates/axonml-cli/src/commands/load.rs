//! Load - Load Models and Datasets
//!
//! Central command for loading models and datasets into Axonml's
//! workspace for training, analysis, and inference.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::fs;
use std::path::PathBuf;

use axonml_serialize::{load_state_dict, StateDict};
use serde::{Deserialize, Serialize};

use super::utils::{path_exists, print_header, print_info, print_kv, print_success};
use crate::cli::{LoadArgs, LoadBothArgs, LoadDataArgs, LoadModelArgs, LoadSubcommand};
use crate::error::{CliError, CliResult};

// =============================================================================
// Workspace State
// =============================================================================

/// Axonml workspace state file
const WORKSPACE_FILE: &str = ".axonml/workspace.json";

/// Current workspace state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceState {
    pub model: Option<LoadedModel>,
    pub dataset: Option<LoadedDataset>,
    pub last_updated: String,
}

/// Information about a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModel {
    pub path: String,
    pub name: String,
    pub format: String,
    pub num_parameters: usize,
    pub file_size: u64,
    pub loaded_at: String,
}

/// Information about a loaded dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedDataset {
    pub path: String,
    pub name: String,
    pub data_type: String,
    pub num_samples: usize,
    pub num_classes: Option<usize>,
    pub loaded_at: String,
}

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `load` command
pub fn execute(args: LoadArgs) -> CliResult<()> {
    match args.action {
        LoadSubcommand::Model(model_args) => execute_load_model(model_args),
        LoadSubcommand::Data(data_args) => execute_load_data(data_args),
        LoadSubcommand::Both(both_args) => execute_load_both(both_args),
        LoadSubcommand::Status => execute_status(),
        LoadSubcommand::Clear => execute_clear(),
    }
}

// =============================================================================
// Load Model Subcommand
// =============================================================================

fn execute_load_model(args: LoadModelArgs) -> CliResult<()> {
    print_header("Load Model");

    let path = PathBuf::from(&args.path);
    if !path_exists(&path) {
        return Err(CliError::Model(format!("Model not found: {}", args.path)));
    }

    print_kv("Path", &args.path);

    // Detect format
    let format = detect_model_format(&path);
    print_kv("Format", &format);

    // Get file info
    let file_size = fs::metadata(&path)?.len();
    print_kv("Size", &format_size(file_size));

    println!();
    print_info("Loading model...");

    // Load and validate
    let state_dict = load_state_dict(&path)
        .map_err(|e| CliError::Model(format!("Failed to load model: {e}")))?;

    let num_parameters = count_parameters(&state_dict);
    let num_layers = state_dict.entries().count();

    print_kv("Parameters", &format_number(num_parameters));
    print_kv("Layers", &num_layers.to_string());

    // Determine name
    let name = args.name.clone().unwrap_or_else(|| {
        path.file_stem().map_or_else(|| "model".to_string(), |s| s.to_string_lossy().to_string())
    });

    // Update workspace
    let loaded = LoadedModel {
        path: args.path.clone(),
        name: name.clone(),
        format,
        num_parameters,
        file_size,
        loaded_at: chrono::Utc::now().to_rfc3339(),
    };

    let mut workspace = load_workspace()?;
    workspace.model = Some(loaded);
    workspace.last_updated = chrono::Utc::now().to_rfc3339();
    save_workspace(&workspace)?;

    println!();
    print_success(&format!("Model '{name}' loaded successfully!"));
    print_info("Use 'axonml analyze model' for detailed analysis");
    print_info("Use 'axonml load status' to see loaded items");

    Ok(())
}

// =============================================================================
// Load Data Subcommand
// =============================================================================

fn execute_load_data(args: LoadDataArgs) -> CliResult<()> {
    print_header("Load Dataset");

    let path = PathBuf::from(&args.path);
    if !path_exists(&path) {
        return Err(CliError::Data(format!("Dataset not found: {}", args.path)));
    }

    print_kv("Path", &args.path);

    // Detect type
    let data_type = detect_data_type(&path);
    print_kv("Type", &data_type);

    println!();
    print_info("Analyzing dataset...");

    // Quick analysis
    let (num_samples, num_classes, total_size) = quick_analyze_dataset(&path, &data_type)?;

    print_kv("Samples", &num_samples.to_string());
    if let Some(n) = num_classes {
        print_kv("Classes", &n.to_string());
    }
    print_kv("Size", &format_size(total_size));

    // Determine name
    let name = args.name.clone().unwrap_or_else(|| {
        path.file_name().map_or_else(|| "dataset".to_string(), |s| s.to_string_lossy().to_string())
    });

    // Update workspace
    let loaded = LoadedDataset {
        path: args.path.clone(),
        name: name.clone(),
        data_type,
        num_samples,
        num_classes,
        loaded_at: chrono::Utc::now().to_rfc3339(),
    };

    let mut workspace = load_workspace()?;
    workspace.dataset = Some(loaded);
    workspace.last_updated = chrono::Utc::now().to_rfc3339();
    save_workspace(&workspace)?;

    println!();
    print_success(&format!("Dataset '{name}' loaded successfully!"));
    print_info("Use 'axonml analyze data' for detailed analysis");
    print_info("Use 'axonml load status' to see loaded items");

    Ok(())
}

// =============================================================================
// Load Both Subcommand
// =============================================================================

fn execute_load_both(args: LoadBothArgs) -> CliResult<()> {
    print_header("Load Model and Dataset");

    // Load model
    let model_path = PathBuf::from(&args.model);
    if !path_exists(&model_path) {
        return Err(CliError::Model(format!("Model not found: {}", args.model)));
    }

    let data_path = PathBuf::from(&args.data);
    if !path_exists(&data_path) {
        return Err(CliError::Data(format!("Dataset not found: {}", args.data)));
    }

    print_kv("Model", &args.model);
    print_kv("Dataset", &args.data);

    println!();

    // Load model
    print_info("Loading model...");
    let model_format = detect_model_format(&model_path);
    let model_size = fs::metadata(&model_path)?.len();
    let state_dict = load_state_dict(&model_path)
        .map_err(|e| CliError::Model(format!("Failed to load model: {e}")))?;
    let num_parameters = count_parameters(&state_dict);

    let model_name = model_path
        .file_stem().map_or_else(|| "model".to_string(), |s| s.to_string_lossy().to_string());

    print_kv("  Parameters", &format_number(num_parameters));

    // Load dataset
    print_info("Loading dataset...");
    let data_type = detect_data_type(&data_path);
    let (num_samples, num_classes, _data_size) = quick_analyze_dataset(&data_path, &data_type)?;

    let data_name = data_path
        .file_name().map_or_else(|| "dataset".to_string(), |s| s.to_string_lossy().to_string());

    print_kv("  Samples", &num_samples.to_string());

    // Update workspace
    let model = LoadedModel {
        path: args.model.clone(),
        name: model_name.clone(),
        format: model_format,
        num_parameters,
        file_size: model_size,
        loaded_at: chrono::Utc::now().to_rfc3339(),
    };

    let dataset = LoadedDataset {
        path: args.data.clone(),
        name: data_name.clone(),
        data_type,
        num_samples,
        num_classes,
        loaded_at: chrono::Utc::now().to_rfc3339(),
    };

    let workspace = WorkspaceState {
        model: Some(model),
        dataset: Some(dataset),
        last_updated: chrono::Utc::now().to_rfc3339(),
    };
    save_workspace(&workspace)?;

    println!();
    print_success("Model and dataset loaded successfully!");
    print_info("Use 'axonml analyze both' for compatibility analysis");
    print_info("Use 'axonml train' to start training");

    Ok(())
}

// =============================================================================
// Status Subcommand
// =============================================================================

fn execute_status() -> CliResult<()> {
    print_header("Workspace Status");

    let workspace = load_workspace()?;

    if workspace.model.is_none() && workspace.dataset.is_none() {
        println!();
        print_info("No model or dataset loaded");
        print_info("Use 'axonml load model <path>' to load a model");
        print_info("Use 'axonml load data <path>' to load a dataset");
        return Ok(());
    }

    if let Some(model) = &workspace.model {
        println!();
        print_header("Loaded Model");
        print_kv("Name", &model.name);
        print_kv("Path", &model.path);
        print_kv("Format", &model.format);
        print_kv("Parameters", &format_number(model.num_parameters));
        print_kv("Size", &format_size(model.file_size));
        print_kv("Loaded at", &model.loaded_at);
    }

    if let Some(dataset) = &workspace.dataset {
        println!();
        print_header("Loaded Dataset");
        print_kv("Name", &dataset.name);
        print_kv("Path", &dataset.path);
        print_kv("Type", &dataset.data_type);
        print_kv("Samples", &dataset.num_samples.to_string());
        if let Some(n) = dataset.num_classes {
            print_kv("Classes", &n.to_string());
        }
        print_kv("Loaded at", &dataset.loaded_at);
    }

    println!();
    print_kv("Last updated", &workspace.last_updated);

    Ok(())
}

// =============================================================================
// Clear Subcommand
// =============================================================================

fn execute_clear() -> CliResult<()> {
    print_header("Clear Workspace");

    let workspace_path = PathBuf::from(WORKSPACE_FILE);
    if workspace_path.exists() {
        fs::remove_file(&workspace_path)?;
        print_success("Workspace cleared");
    } else {
        print_info("Workspace already empty");
    }

    Ok(())
}

// =============================================================================
// Workspace Management
// =============================================================================

pub fn load_workspace() -> CliResult<WorkspaceState> {
    let path = PathBuf::from(WORKSPACE_FILE);
    if path.exists() {
        let content = fs::read_to_string(&path)?;
        let workspace: WorkspaceState = serde_json::from_str(&content)?;
        Ok(workspace)
    } else {
        Ok(WorkspaceState::default())
    }
}

fn save_workspace(workspace: &WorkspaceState) -> CliResult<()> {
    let path = PathBuf::from(WORKSPACE_FILE);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let content = serde_json::to_string_pretty(workspace)?;
    fs::write(path, content)?;
    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

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

fn detect_data_type(path: &PathBuf) -> String {
    use walkdir::WalkDir;

    let mut image_count = 0;
    let mut csv_count = 0;
    let mut text_count = 0;

    let walker = WalkDir::new(path).max_depth(3).into_iter();
    for entry in walker.filter_map(std::result::Result::ok).take(100) {
        if let Some(ext) = entry.path().extension() {
            match ext.to_string_lossy().to_lowercase().as_str() {
                "jpg" | "jpeg" | "png" | "bmp" | "gif" => image_count += 1,
                "csv" | "tsv" | "parquet" | "json" => csv_count += 1,
                "txt" | "md" => text_count += 1,
                _ => {}
            }
        }
    }

    if image_count > csv_count && image_count > text_count {
        "image".to_string()
    } else if csv_count > text_count {
        "tabular".to_string()
    } else if text_count > 0 {
        "text".to_string()
    } else {
        "unknown".to_string()
    }
}

fn quick_analyze_dataset(
    path: &PathBuf,
    data_type: &str,
) -> CliResult<(usize, Option<usize>, u64)> {
    use walkdir::WalkDir;

    let mut num_samples = 0;
    let mut total_size = 0u64;
    let mut classes: std::collections::HashSet<String> = std::collections::HashSet::new();

    match data_type {
        "image" => {
            // Check for class subdirectories
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.filter_map(std::result::Result::ok) {
                    if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        let class_name = entry.file_name().to_string_lossy().to_string();
                        if !class_name.starts_with('.') {
                            classes.insert(class_name);
                            // Count images in class
                            for img in WalkDir::new(entry.path())
                                .into_iter()
                                .filter_map(std::result::Result::ok)
                            {
                                if img.file_type().is_file() {
                                    if let Some(ext) = img.path().extension() {
                                        if matches!(
                                            ext.to_string_lossy().to_lowercase().as_str(),
                                            "jpg" | "jpeg" | "png" | "bmp" | "gif"
                                        ) {
                                            num_samples += 1;
                                            if let Ok(meta) = img.metadata() {
                                                total_size += meta.len();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        "tabular" => {
            // Find and count CSV rows
            for entry in WalkDir::new(path)
                .max_depth(2)
                .into_iter()
                .filter_map(std::result::Result::ok)
            {
                if entry
                    .path()
                    .extension()
                    .is_some_and(|e| e == "csv")
                {
                    if let Ok(content) = fs::read_to_string(entry.path()) {
                        num_samples = content.lines().count().saturating_sub(1);
                        total_size = content.len() as u64;

                        // Try to detect classes from last column
                        for line in content.lines().skip(1).take(1000) {
                            if let Some(label) = line.split(',').next_back() {
                                classes.insert(label.trim().to_string());
                            }
                        }
                        break;
                    }
                }
            }
        }
        _ => {
            // Generic file count
            for entry in WalkDir::new(path).into_iter().filter_map(std::result::Result::ok) {
                if entry.file_type().is_file() {
                    num_samples += 1;
                    if let Ok(meta) = entry.metadata() {
                        total_size += meta.len();
                    }
                }
            }
        }
    }

    let num_classes = if classes.len() > 1 && classes.len() < 1000 {
        Some(classes.len())
    } else {
        None
    };

    Ok((num_samples, num_classes, total_size))
}

fn count_parameters(state_dict: &StateDict) -> usize {
    state_dict
        .entries()
        .map(|(_, entry)| entry.data.shape.iter().product::<usize>())
        .sum()
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 bytes");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.50K");
        assert_eq!(format_number(1_500_000), "1.50M");
    }

    #[test]
    fn test_detect_model_format() {
        assert_eq!(detect_model_format(&PathBuf::from("model.pt")), "pytorch");
        assert_eq!(
            detect_model_format(&PathBuf::from("model.safetensors")),
            "safetensors"
        );
        assert_eq!(
            detect_model_format(&PathBuf::from("model.axonml")),
            "axonml"
        );
    }
}
