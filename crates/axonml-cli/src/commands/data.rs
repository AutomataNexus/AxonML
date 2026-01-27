//! Data - Dataset Analysis and Management Command
//!
//! Handles dataset upload, analysis, validation, and configuration
//! generation for training with Axonml.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use super::utils::{
    ensure_dir, path_exists, print_header, print_info, print_kv, print_success, print_warning,
};
use crate::cli::{
    DataAnalyzeArgs, DataArgs, DataConfigArgs, DataListArgs, DataPreviewArgs, DataSubcommand,
    DataUploadArgs, DataValidateArgs,
};
use crate::error::{CliError, CliResult};

// =============================================================================
// Data Structures
// =============================================================================

/// Detected dataset type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetType {
    Image,
    Tabular,
    Text,
    Audio,
    Mixed,
    Unknown,
}

impl std::fmt::Display for DatasetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetType::Image => write!(f, "image"),
            DatasetType::Tabular => write!(f, "tabular"),
            DatasetType::Text => write!(f, "text"),
            DatasetType::Audio => write!(f, "audio"),
            DatasetType::Mixed => write!(f, "mixed"),
            DatasetType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Dataset analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAnalysis {
    pub name: String,
    pub path: String,
    pub data_type: DatasetType,
    pub task_type: String,
    pub num_samples: usize,
    pub num_classes: Option<usize>,
    pub class_distribution: Option<HashMap<String, usize>>,
    pub input_shape: Option<Vec<usize>>,
    pub feature_names: Option<Vec<String>>,
    pub statistics: DataStatistics,
    pub recommendations: TrainingRecommendations,
}

/// Statistical information about the dataset
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataStatistics {
    pub total_size_bytes: u64,
    pub num_files: usize,
    pub file_types: HashMap<String, usize>,
    pub missing_values: usize,
    pub duplicate_samples: usize,
    pub mean_values: Option<Vec<f64>>,
    pub std_values: Option<Vec<f64>>,
    pub min_values: Option<Vec<f64>>,
    pub max_values: Option<Vec<f64>>,
}

/// Recommended training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecommendations {
    pub architecture: String,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub optimizer: String,
    pub loss_function: String,
    pub transforms: Vec<String>,
    pub augmentations: Vec<String>,
    pub notes: Vec<String>,
}

impl Default for TrainingRecommendations {
    fn default() -> Self {
        Self {
            architecture: "mlp".to_string(),
            batch_size: 32,
            learning_rate: 0.001,
            epochs: 10,
            optimizer: "adam".to_string(),
            loss_function: "cross_entropy".to_string(),
            transforms: vec!["normalize".to_string()],
            augmentations: vec![],
            notes: vec![],
        }
    }
}

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `data` command
pub fn execute(args: DataArgs) -> CliResult<()> {
    match args.action {
        DataSubcommand::Upload(upload_args) => execute_upload(upload_args),
        DataSubcommand::Analyze(analyze_args) => execute_analyze(analyze_args),
        DataSubcommand::List(list_args) => execute_list(list_args),
        DataSubcommand::Config(config_args) => execute_config(config_args),
        DataSubcommand::Preview(preview_args) => execute_preview(preview_args),
        DataSubcommand::Validate(validate_args) => execute_validate(validate_args),
    }
}

// =============================================================================
// Upload Subcommand
// =============================================================================

fn execute_upload(args: DataUploadArgs) -> CliResult<()> {
    print_header("Dataset Upload");

    let source_path = PathBuf::from(&args.path);
    if !path_exists(&source_path) {
        return Err(CliError::Data(format!("Path not found: {}", args.path)));
    }

    // Determine dataset name
    let dataset_name = args.name.clone().unwrap_or_else(|| {
        source_path.file_stem().map_or_else(
            || "dataset".to_string(),
            |s| s.to_string_lossy().to_string(),
        )
    });

    print_kv("Source", &args.path);
    print_kv("Dataset name", &dataset_name);

    // Ensure output directory exists
    ensure_dir(&args.output)?;

    // Create dataset directory
    let dest_dir = PathBuf::from(&args.output).join(&dataset_name);
    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir)?;
    }

    println!();
    print_info("Copying dataset files...");

    // Copy files
    let file_count = copy_dataset(&source_path, &dest_dir)?;
    print_success(&format!("Copied {file_count} files"));

    // Auto-analyze if requested
    if args.analyze {
        println!();
        let analysis = analyze_dataset(&dest_dir, args.data_type.as_deref(), 1000)?;
        print_analysis_summary(&analysis);

        // Save analysis
        let analysis_path = dest_dir.join("dataset_analysis.json");
        let json = serde_json::to_string_pretty(&analysis)?;
        fs::write(&analysis_path, json)?;
        print_info(&format!("Analysis saved to: {}", analysis_path.display()));
    }

    println!();
    print_success("Dataset uploaded successfully!");
    print_info(&format!("Location: {}", dest_dir.display()));
    print_info("Use 'axonml data analyze' for detailed analysis");
    print_info("Use 'axonml data config' to generate training config");

    Ok(())
}

fn copy_dataset(source: &PathBuf, dest: &PathBuf) -> CliResult<usize> {
    let mut count = 0;

    if source.is_file() {
        let dest_file = dest.join(source.file_name().unwrap());
        fs::copy(source, dest_file)?;
        count = 1;
    } else if source.is_dir() {
        for entry in WalkDir::new(source).min_depth(1) {
            let entry = entry.map_err(|e| CliError::Io(e.into()))?;
            let path = entry.path();
            let relative = path.strip_prefix(source).unwrap();
            let dest_path = dest.join(relative);

            if path.is_dir() {
                fs::create_dir_all(&dest_path)?;
            } else {
                if let Some(parent) = dest_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(path, &dest_path)?;
                count += 1;
            }
        }
    }

    Ok(count)
}

// =============================================================================
// Analyze Subcommand
// =============================================================================

fn execute_analyze(args: DataAnalyzeArgs) -> CliResult<()> {
    print_header("Dataset Analysis");

    let path = PathBuf::from(&args.path);
    if !path_exists(&path) {
        return Err(CliError::Data(format!("Path not found: {}", args.path)));
    }

    print_kv("Path", &args.path);
    print_kv("Max samples", &args.max_samples.to_string());
    println!();

    print_info("Analyzing dataset...");
    let analysis = analyze_dataset(&path, args.data_type.as_deref(), args.max_samples)?;

    if args.format == "json" {
        let json = serde_json::to_string_pretty(&analysis)?;
        if let Some(output) = &args.output {
            fs::write(output, &json)?;
            print_success(&format!("Analysis saved to: {output}"));
        } else {
            println!("{json}");
        }
    } else {
        print_analysis_summary(&analysis);

        if args.detailed {
            print_detailed_statistics(&analysis);
        }

        if args.recommend {
            print_recommendations(&analysis);
        }

        if let Some(output) = &args.output {
            let json = serde_json::to_string_pretty(&analysis)?;
            fs::write(output, &json)?;
            print_success(&format!("Analysis saved to: {output}"));
        }
    }

    Ok(())
}

fn analyze_dataset(
    path: &PathBuf,
    type_hint: Option<&str>,
    max_samples: usize,
) -> CliResult<DatasetAnalysis> {
    let name = path.file_name().map_or_else(
        || "dataset".to_string(),
        |s| s.to_string_lossy().to_string(),
    );

    // Detect dataset type
    let data_type = type_hint.map_or_else(|| detect_data_type(path), parse_data_type);

    // Scan files
    let (statistics, file_info) = scan_files(path)?;

    // Count samples
    let num_samples = estimate_sample_count(path, &data_type, &file_info);

    // Detect classes for classification tasks
    let (num_classes, class_distribution) = detect_classes(path, &data_type);

    // Detect input shape
    let input_shape = detect_input_shape(path, &data_type, &file_info);

    // Determine task type
    let task_type = infer_task_type(&data_type, num_classes);

    // Generate recommendations
    let recommendations = generate_recommendations(
        &data_type,
        &task_type,
        num_samples,
        num_classes,
        &input_shape,
    );

    Ok(DatasetAnalysis {
        name,
        path: path.display().to_string(),
        data_type,
        task_type,
        num_samples: num_samples.min(max_samples),
        num_classes,
        class_distribution,
        input_shape,
        feature_names: None,
        statistics,
        recommendations,
    })
}

fn parse_data_type(s: &str) -> DatasetType {
    match s.to_lowercase().as_str() {
        "image" | "img" | "vision" => DatasetType::Image,
        "tabular" | "csv" | "table" => DatasetType::Tabular,
        "text" | "nlp" | "language" => DatasetType::Text,
        "audio" | "sound" | "speech" => DatasetType::Audio,
        "mixed" => DatasetType::Mixed,
        _ => DatasetType::Unknown,
    }
}

fn detect_data_type(path: &PathBuf) -> DatasetType {
    let mut image_count = 0;
    let mut csv_count = 0;
    let mut text_count = 0;
    let mut audio_count = 0;

    let walker = WalkDir::new(path).max_depth(3).into_iter();
    for entry in walker.filter_map(std::result::Result::ok).take(100) {
        if let Some(ext) = entry.path().extension() {
            match ext.to_string_lossy().to_lowercase().as_str() {
                "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" => image_count += 1,
                "csv" | "tsv" | "parquet" | "json" | "jsonl" => csv_count += 1,
                "txt" | "md" | "xml" => text_count += 1,
                "wav" | "mp3" | "flac" | "ogg" => audio_count += 1,
                _ => {}
            }
        }
    }

    if image_count > csv_count && image_count > text_count && image_count > audio_count {
        DatasetType::Image
    } else if csv_count > text_count && csv_count > audio_count {
        DatasetType::Tabular
    } else if text_count > audio_count {
        DatasetType::Text
    } else if audio_count > 0 {
        DatasetType::Audio
    } else {
        DatasetType::Unknown
    }
}

fn scan_files(path: &PathBuf) -> CliResult<(DataStatistics, HashMap<String, usize>)> {
    let mut stats = DataStatistics::default();
    let mut file_types: HashMap<String, usize> = HashMap::new();

    for entry in WalkDir::new(path)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if entry.file_type().is_file() {
            stats.num_files += 1;
            if let Ok(metadata) = entry.metadata() {
                stats.total_size_bytes += metadata.len();
            }
            if let Some(ext) = entry.path().extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                *file_types.entry(ext_str).or_insert(0) += 1;
            }
        }
    }

    stats.file_types = file_types.clone();
    Ok((stats, file_types))
}

fn estimate_sample_count(
    path: &PathBuf,
    data_type: &DatasetType,
    file_info: &HashMap<String, usize>,
) -> usize {
    match data_type {
        DatasetType::Image => {
            // Count image files
            file_info
                .iter()
                .filter(|(ext, _)| {
                    matches!(
                        ext.as_str(),
                        "jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp"
                    )
                })
                .map(|(_, count)| count)
                .sum()
        }
        DatasetType::Tabular => {
            // Try to read CSV row count
            for entry in WalkDir::new(path)
                .max_depth(2)
                .into_iter()
                .filter_map(std::result::Result::ok)
            {
                if entry.path().extension().is_some_and(|e| e == "csv") {
                    if let Ok(content) = fs::read_to_string(entry.path()) {
                        return content.lines().count().saturating_sub(1); // Subtract header
                    }
                }
            }
            0
        }
        DatasetType::Text => file_info
            .iter()
            .filter(|(ext, _)| matches!(ext.as_str(), "txt" | "json" | "jsonl"))
            .map(|(_, count)| count)
            .sum(),
        DatasetType::Audio => file_info
            .iter()
            .filter(|(ext, _)| matches!(ext.as_str(), "wav" | "mp3" | "flac" | "ogg"))
            .map(|(_, count)| count)
            .sum(),
        _ => file_info.values().sum(),
    }
}

fn detect_classes(
    path: &PathBuf,
    data_type: &DatasetType,
) -> (Option<usize>, Option<HashMap<String, usize>>) {
    match data_type {
        DatasetType::Image => {
            // Look for class subdirectories
            let mut classes: HashMap<String, usize> = HashMap::new();
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.filter_map(std::result::Result::ok) {
                    if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        let class_name = entry.file_name().to_string_lossy().to_string();
                        if !class_name.starts_with('.') {
                            // Count files in this class directory
                            let count = WalkDir::new(entry.path())
                                .into_iter()
                                .filter_map(std::result::Result::ok)
                                .filter(|e| e.file_type().is_file())
                                .count();
                            classes.insert(class_name, count);
                        }
                    }
                }
            }
            if classes.is_empty() {
                (None, None)
            } else {
                let num = classes.len();
                (Some(num), Some(classes))
            }
        }
        DatasetType::Tabular => {
            // Try to detect from CSV label column
            for entry in WalkDir::new(path)
                .max_depth(2)
                .into_iter()
                .filter_map(std::result::Result::ok)
            {
                if entry.path().extension().is_some_and(|e| e == "csv") {
                    if let Ok(content) = fs::read_to_string(entry.path()) {
                        let lines: Vec<&str> = content.lines().collect();
                        if lines.len() > 1 {
                            // Try last column as label
                            let mut classes: HashMap<String, usize> = HashMap::new();
                            for line in lines.iter().skip(1).take(1000) {
                                if let Some(label) = line.split(',').next_back() {
                                    *classes.entry(label.trim().to_string()).or_insert(0) += 1;
                                }
                            }
                            if classes.len() > 1 && classes.len() < 100 {
                                let num = classes.len();
                                return (Some(num), Some(classes));
                            }
                        }
                    }
                }
            }
            (None, None)
        }
        _ => (None, None),
    }
}

fn detect_input_shape(
    path: &PathBuf,
    data_type: &DatasetType,
    _file_info: &HashMap<String, usize>,
) -> Option<Vec<usize>> {
    match data_type {
        DatasetType::Image => {
            // Default common image sizes
            Some(vec![3, 224, 224])
        }
        DatasetType::Tabular => {
            // Try to detect from CSV
            for entry in WalkDir::new(path)
                .max_depth(2)
                .into_iter()
                .filter_map(std::result::Result::ok)
            {
                if entry.path().extension().is_some_and(|e| e == "csv") {
                    if let Ok(content) = fs::read_to_string(entry.path()) {
                        if let Some(first_line) = content.lines().next() {
                            let num_cols = first_line.split(',').count();
                            return Some(vec![num_cols.saturating_sub(1)]); // Subtract label column
                        }
                    }
                }
            }
            None
        }
        DatasetType::Audio => {
            // Default audio shape (mono, 16kHz, 1 second)
            Some(vec![1, 16000])
        }
        _ => None,
    }
}

fn infer_task_type(data_type: &DatasetType, num_classes: Option<usize>) -> String {
    match (data_type, num_classes) {
        (DatasetType::Image, Some(n)) if n > 1 => "classification".to_string(),
        (DatasetType::Tabular, Some(n)) if n > 1 && n < 20 => "classification".to_string(),
        (DatasetType::Tabular, _) => "regression".to_string(),
        (DatasetType::Text, Some(n)) if n > 1 => "classification".to_string(),
        (DatasetType::Text, _) => "language_modeling".to_string(),
        (DatasetType::Audio, Some(n)) if n > 1 => "classification".to_string(),
        _ => "unknown".to_string(),
    }
}

fn generate_recommendations(
    data_type: &DatasetType,
    task_type: &str,
    num_samples: usize,
    num_classes: Option<usize>,
    _input_shape: &Option<Vec<usize>>,
) -> TrainingRecommendations {
    let mut rec = TrainingRecommendations::default();

    // Architecture
    rec.architecture = match data_type {
        DatasetType::Image => "cnn".to_string(),
        DatasetType::Tabular => "mlp".to_string(),
        DatasetType::Text => "transformer".to_string(),
        DatasetType::Audio => "cnn".to_string(),
        _ => "mlp".to_string(),
    };

    // Batch size (smaller for small datasets)
    rec.batch_size = if num_samples < 1000 {
        16
    } else if num_samples < 10000 {
        32
    } else {
        64
    };

    // Learning rate
    rec.learning_rate = if rec.architecture == "transformer" {
        0.0001
    } else {
        0.001
    };

    // Epochs (more for small datasets)
    rec.epochs = if num_samples < 1000 {
        50
    } else if num_samples < 10000 {
        20
    } else {
        10
    };

    // Loss function
    rec.loss_function = match task_type {
        "classification" => "cross_entropy".to_string(),
        "regression" => "mse".to_string(),
        "language_modeling" => "cross_entropy".to_string(),
        _ => "cross_entropy".to_string(),
    };

    // Transforms
    rec.transforms = match data_type {
        DatasetType::Image => vec![
            "resize(224, 224)".to_string(),
            "to_tensor".to_string(),
            "normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])".to_string(),
        ],
        DatasetType::Tabular => vec!["normalize".to_string()],
        DatasetType::Audio => vec![
            "resample(16000)".to_string(),
            "mel_spectrogram".to_string(),
            "normalize".to_string(),
        ],
        _ => vec!["normalize".to_string()],
    };

    // Augmentations
    rec.augmentations = match data_type {
        DatasetType::Image => vec![
            "random_horizontal_flip".to_string(),
            "random_rotation(10)".to_string(),
            "color_jitter".to_string(),
        ],
        DatasetType::Audio => vec!["add_noise(0.01)".to_string(), "time_shift".to_string()],
        _ => vec![],
    };

    // Notes
    if num_samples < 1000 {
        rec.notes
            .push("Small dataset: Consider data augmentation and regularization".to_string());
    }
    if let Some(n) = num_classes {
        if n > 100 {
            rec.notes.push(format!(
                "Large number of classes ({n}): May need larger model capacity"
            ));
        }
    }

    rec
}

// =============================================================================
// List Subcommand
// =============================================================================

fn execute_list(args: DataListArgs) -> CliResult<()> {
    print_header("Available Datasets");

    let path = PathBuf::from(&args.path);
    if !path_exists(&path) {
        print_warning(&format!("Data directory not found: {}", args.path));
        print_info("Use 'axonml data upload' to add a dataset");
        return Ok(());
    }

    let mut found = false;
    if let Ok(entries) = fs::read_dir(&path) {
        for entry in entries.filter_map(std::result::Result::ok) {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    found = true;
                    print_dataset_info(&entry.path(), &name, args.detailed)?;
                }
            }
        }
    }

    if !found {
        print_info("No datasets found");
        print_info("Use 'axonml data upload' to add a dataset");
    }

    Ok(())
}

fn print_dataset_info(path: &PathBuf, name: &str, detailed: bool) -> CliResult<()> {
    let (stats, _) = scan_files(path)?;
    let data_type = detect_data_type(path);

    println!();
    print_kv("Name", name);
    print_kv("Type", &data_type.to_string());
    print_kv("Files", &stats.num_files.to_string());
    print_kv("Size", &format_size(stats.total_size_bytes));

    if detailed {
        let (num_classes, _) = detect_classes(path, &data_type);
        if let Some(n) = num_classes {
            print_kv("Classes", &n.to_string());
        }
    }

    Ok(())
}

// =============================================================================
// Config Subcommand
// =============================================================================

fn execute_config(args: DataConfigArgs) -> CliResult<()> {
    print_header("Generate Data Configuration");

    let path = PathBuf::from(&args.path);
    if !path_exists(&path) {
        return Err(CliError::Data(format!("Path not found: {}", args.path)));
    }

    print_info("Analyzing dataset...");
    let analysis = analyze_dataset(&path, None, 1000)?;

    let config = generate_data_config(&analysis);

    if args.format == "json" {
        let json = serde_json::to_string_pretty(&config)?;
        fs::write(&args.output, json)?;
    } else {
        let toml = toml::to_string_pretty(&config).map_err(|e| CliError::Config(e.to_string()))?;
        fs::write(&args.output, toml)?;
    }

    print_success(&format!("Configuration saved to: {}", args.output));
    print_info("Add this to your axonml.toml or use with --config");

    Ok(())
}

fn generate_data_config(analysis: &DatasetAnalysis) -> toml::Value {
    let mut config = toml::map::Map::new();

    config.insert(
        "name".to_string(),
        toml::Value::String(analysis.name.clone()),
    );
    config.insert(
        "path".to_string(),
        toml::Value::String(analysis.path.clone()),
    );
    config.insert(
        "type".to_string(),
        toml::Value::String(analysis.data_type.to_string()),
    );
    config.insert(
        "task".to_string(),
        toml::Value::String(analysis.task_type.clone()),
    );

    if let Some(n) = analysis.num_classes {
        config.insert("num_classes".to_string(), toml::Value::Integer(n as i64));
    }

    if let Some(shape) = &analysis.input_shape {
        let shape_arr: Vec<toml::Value> = shape
            .iter()
            .map(|&d| toml::Value::Integer(d as i64))
            .collect();
        config.insert("input_shape".to_string(), toml::Value::Array(shape_arr));
    }

    // Add recommendations
    let rec = &analysis.recommendations;
    let mut training = toml::map::Map::new();
    training.insert(
        "batch_size".to_string(),
        toml::Value::Integer(rec.batch_size as i64),
    );
    training.insert(
        "learning_rate".to_string(),
        toml::Value::Float(rec.learning_rate),
    );
    training.insert(
        "epochs".to_string(),
        toml::Value::Integer(rec.epochs as i64),
    );
    training.insert(
        "optimizer".to_string(),
        toml::Value::String(rec.optimizer.clone()),
    );
    training.insert(
        "loss".to_string(),
        toml::Value::String(rec.loss_function.clone()),
    );

    config.insert("training".to_string(), toml::Value::Table(training));

    // Add transforms
    let transforms: Vec<toml::Value> = rec
        .transforms
        .iter()
        .map(|t| toml::Value::String(t.clone()))
        .collect();
    config.insert("transforms".to_string(), toml::Value::Array(transforms));

    // Add augmentations
    if !rec.augmentations.is_empty() {
        let augs: Vec<toml::Value> = rec
            .augmentations
            .iter()
            .map(|a| toml::Value::String(a.clone()))
            .collect();
        config.insert("augmentations".to_string(), toml::Value::Array(augs));
    }

    toml::Value::Table(config)
}

// =============================================================================
// Preview Subcommand
// =============================================================================

fn execute_preview(args: DataPreviewArgs) -> CliResult<()> {
    print_header("Dataset Preview");

    let path = PathBuf::from(&args.path);
    if !path_exists(&path) {
        return Err(CliError::Data(format!("Path not found: {}", args.path)));
    }

    let data_type = detect_data_type(&path);
    print_kv("Type", &data_type.to_string());
    println!();

    match data_type {
        DatasetType::Tabular => preview_tabular(&path, args.num_samples)?,
        DatasetType::Image => preview_image(&path, args.num_samples)?,
        DatasetType::Text => preview_text(&path, args.num_samples)?,
        _ => preview_files(&path, args.num_samples)?,
    }

    Ok(())
}

fn preview_tabular(path: &PathBuf, num_samples: usize) -> CliResult<()> {
    for entry in WalkDir::new(path)
        .max_depth(2)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if entry.path().extension().is_some_and(|e| e == "csv") {
            let content = fs::read_to_string(entry.path())?;
            let lines: Vec<&str> = content.lines().collect();

            if let Some(header) = lines.first() {
                println!("Columns: {header}");
                println!();
            }

            println!("Sample rows:");
            for (i, line) in lines.iter().skip(1).take(num_samples).enumerate() {
                println!("  [{}] {}", i + 1, truncate(line, 100));
            }

            return Ok(());
        }
    }

    print_warning("No CSV files found");
    Ok(())
}

fn preview_image(path: &PathBuf, num_samples: usize) -> CliResult<()> {
    let mut count = 0;

    println!("Sample images:");
    for entry in WalkDir::new(path)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if let Some(ext) = entry.path().extension() {
            if matches!(
                ext.to_string_lossy().to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "bmp"
            ) && count < num_samples
            {
                let relative = entry.path().strip_prefix(path).unwrap_or(entry.path());
                println!("  [{}] {}", count + 1, relative.display());
                count += 1;
            }
        }
    }

    if count == 0 {
        print_warning("No image files found");
    }

    Ok(())
}

fn preview_text(path: &PathBuf, num_samples: usize) -> CliResult<()> {
    let mut count = 0;

    println!("Sample text:");
    for entry in WalkDir::new(path)
        .max_depth(2)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if let Some(ext) = entry.path().extension() {
            if matches!(
                ext.to_string_lossy().to_lowercase().as_str(),
                "txt" | "json" | "jsonl"
            ) && count < num_samples
            {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    let preview = truncate(content.lines().next().unwrap_or(""), 100);
                    println!("  [{}] {}", count + 1, preview);
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        print_warning("No text files found");
    }

    Ok(())
}

fn preview_files(path: &PathBuf, num_samples: usize) -> CliResult<()> {
    let mut count = 0;

    println!("Sample files:");
    for entry in WalkDir::new(path)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if entry.file_type().is_file() && count < num_samples {
            let relative = entry.path().strip_prefix(path).unwrap_or(entry.path());
            println!("  [{}] {}", count + 1, relative.display());
            count += 1;
        }
    }

    Ok(())
}

// =============================================================================
// Validate Subcommand
// =============================================================================

fn execute_validate(args: DataValidateArgs) -> CliResult<()> {
    print_header("Dataset Validation");

    let path = PathBuf::from(&args.path);
    if !path_exists(&path) {
        return Err(CliError::Data(format!("Path not found: {}", args.path)));
    }

    let mut issues: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let data_type = detect_data_type(&path);
    print_kv("Type", &data_type.to_string());
    println!();

    print_info("Validating structure...");

    // Check class count
    let (num_classes, class_dist) = detect_classes(&path, &data_type);
    if let Some(expected) = args.num_classes {
        if let Some(actual) = num_classes {
            if actual != expected {
                issues.push(format!(
                    "Class count mismatch: expected {expected}, found {actual}"
                ));
            }
        }
    }

    // Check class balance
    if args.check_balance {
        if let Some(dist) = &class_dist {
            let counts: Vec<usize> = dist.values().copied().collect();
            if !counts.is_empty() {
                let max = *counts.iter().max().unwrap() as f64;
                let min = *counts.iter().min().unwrap() as f64;
                if max / min > 10.0 {
                    warnings.push(format!(
                        "Class imbalance detected: ratio {:.1}x (max: {}, min: {})",
                        max / min,
                        max as usize,
                        min as usize
                    ));
                }
            }
        }
    }

    // Print results
    println!();
    if issues.is_empty() && warnings.is_empty() {
        print_success("Validation passed - no issues found");
    } else {
        for issue in &issues {
            println!("  ERROR: {issue}");
        }
        for warning in &warnings {
            print_warning(warning);
        }

        if !issues.is_empty() {
            return Err(CliError::Data("Validation failed".to_string()));
        }
    }

    // Print summary
    if let Some(n) = num_classes {
        print_kv("Classes", &n.to_string());
    }

    if let Some(dist) = class_dist {
        println!();
        println!("Class distribution:");
        for (class, count) in &dist {
            println!("  {class}: {count} samples");
        }
    }

    Ok(())
}

// =============================================================================
// Output Helpers
// =============================================================================

fn print_analysis_summary(analysis: &DatasetAnalysis) {
    println!();
    print_header("Analysis Summary");
    print_kv("Dataset", &analysis.name);
    print_kv("Type", &analysis.data_type.to_string());
    print_kv("Task", &analysis.task_type);
    print_kv("Samples", &analysis.num_samples.to_string());

    if let Some(n) = analysis.num_classes {
        print_kv("Classes", &n.to_string());
    }

    if let Some(shape) = &analysis.input_shape {
        let shape_str = shape
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(" x ");
        print_kv("Input shape", &shape_str);
    }

    print_kv("Files", &analysis.statistics.num_files.to_string());
    print_kv(
        "Total size",
        &format_size(analysis.statistics.total_size_bytes),
    );
}

fn print_detailed_statistics(analysis: &DatasetAnalysis) {
    println!();
    print_header("File Statistics");

    for (ext, count) in &analysis.statistics.file_types {
        println!("  .{ext}: {count} files");
    }
}

fn print_recommendations(analysis: &DatasetAnalysis) {
    println!();
    print_header("Training Recommendations");

    let rec = &analysis.recommendations;
    print_kv("Architecture", &rec.architecture);
    print_kv("Batch size", &rec.batch_size.to_string());
    print_kv("Learning rate", &format!("{:.6}", rec.learning_rate));
    print_kv("Epochs", &rec.epochs.to_string());
    print_kv("Optimizer", &rec.optimizer);
    print_kv("Loss function", &rec.loss_function);

    if !rec.transforms.is_empty() {
        println!();
        println!("Recommended transforms:");
        for t in &rec.transforms {
            println!("  - {t}");
        }
    }

    if !rec.augmentations.is_empty() {
        println!();
        println!("Recommended augmentations:");
        for a in &rec.augmentations {
            println!("  - {a}");
        }
    }

    if !rec.notes.is_empty() {
        println!();
        println!("Notes:");
        for note in &rec.notes {
            println!("  * {note}");
        }
    }
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
    fn test_parse_data_type() {
        assert_eq!(parse_data_type("image"), DatasetType::Image);
        assert_eq!(parse_data_type("tabular"), DatasetType::Tabular);
        assert_eq!(parse_data_type("text"), DatasetType::Text);
        assert_eq!(parse_data_type("audio"), DatasetType::Audio);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 bytes");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_infer_task_type() {
        assert_eq!(
            infer_task_type(&DatasetType::Image, Some(10)),
            "classification"
        );
        assert_eq!(infer_task_type(&DatasetType::Tabular, None), "regression");
    }
}
