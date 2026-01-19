//! Rename - Rename Models and Datasets
//!
//! Provides utilities for renaming models and datasets with proper
//! metadata updates.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::fs;
use std::path::PathBuf;

use super::utils::{path_exists, print_header, print_info, print_kv, print_success, print_warning};
use crate::cli::{RenameArgs, RenameDataArgs, RenameModelArgs, RenameSubcommand};
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `rename` command
pub fn execute(args: RenameArgs) -> CliResult<()> {
    match args.action {
        RenameSubcommand::Model(model_args) => execute_rename_model(model_args),
        RenameSubcommand::Data(data_args) => execute_rename_data(data_args),
    }
}

// =============================================================================
// Rename Model Subcommand
// =============================================================================

fn execute_rename_model(args: RenameModelArgs) -> CliResult<()> {
    print_header("Rename Model");

    let source_path = PathBuf::from(&args.path);
    if !path_exists(&source_path) {
        return Err(CliError::Model(format!("Model not found: {}", args.path)));
    }

    // Determine new path
    let default_parent = PathBuf::from(".");
    let parent = source_path.parent().unwrap_or(&default_parent);
    let extension = source_path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    let new_path = parent.join(format!("{}{}", args.new_name, extension));

    print_kv("Source", &args.path);
    print_kv("New name", &args.new_name);
    print_kv("New path", &new_path.display().to_string());

    // Check if destination exists
    if path_exists(&new_path) && !args.force {
        return Err(CliError::Other(format!(
            "Destination already exists: {}. Use --force to overwrite.",
            new_path.display()
        )));
    }

    println!();

    // Rename the model file
    if path_exists(&new_path) {
        fs::remove_file(&new_path)?;
    }
    fs::rename(&source_path, &new_path)?;
    print_success(&format!("Renamed model to: {}", new_path.display()));

    // Also rename associated metadata file if it exists
    let meta_source = source_path.with_extension("meta.json");
    if meta_source.exists() {
        let meta_dest = new_path.with_extension("meta.json");
        if path_exists(&meta_dest) {
            fs::remove_file(&meta_dest)?;
        }
        fs::rename(&meta_source, &meta_dest)?;

        // Update metadata content
        if let Ok(content) = fs::read_to_string(&meta_dest) {
            if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(obj) = json.as_object_mut() {
                    obj.insert("name".to_string(), serde_json::json!(args.new_name));
                    obj.insert(
                        "renamed_at".to_string(),
                        serde_json::json!(chrono::Utc::now().to_rfc3339()),
                    );
                    if let Ok(updated) = serde_json::to_string_pretty(&json) {
                        let _ = fs::write(&meta_dest, updated);
                    }
                }
            }
        }

        print_info("Updated metadata file");
    }

    // Handle checkpoint files if this is a checkpoint directory
    let checkpoint_dir = source_path.with_extension("");
    if checkpoint_dir.is_dir() {
        let new_checkpoint_dir = parent.join(&args.new_name);
        if path_exists(&new_checkpoint_dir) && !args.force {
            print_warning(&format!(
                "Checkpoint directory exists: {}. Skipping.",
                new_checkpoint_dir.display()
            ));
        } else {
            if path_exists(&new_checkpoint_dir) {
                fs::remove_dir_all(&new_checkpoint_dir)?;
            }
            fs::rename(&checkpoint_dir, &new_checkpoint_dir)?;
            print_info(&format!(
                "Renamed checkpoint directory to: {}",
                new_checkpoint_dir.display()
            ));
        }
    }

    Ok(())
}

// =============================================================================
// Rename Data Subcommand
// =============================================================================

fn execute_rename_data(args: RenameDataArgs) -> CliResult<()> {
    print_header("Rename Dataset");

    let source_path = PathBuf::from(&args.path);
    if !path_exists(&source_path) {
        return Err(CliError::Data(format!("Dataset not found: {}", args.path)));
    }

    // Determine new path
    let default_parent = PathBuf::from(".");
    let parent = source_path.parent().unwrap_or(&default_parent);
    let new_path = parent.join(&args.new_name);

    print_kv("Source", &args.path);
    print_kv("New name", &args.new_name);
    print_kv("New path", &new_path.display().to_string());

    // Check if destination exists
    if path_exists(&new_path) && !args.force {
        return Err(CliError::Other(format!(
            "Destination already exists: {}. Use --force to overwrite.",
            new_path.display()
        )));
    }

    println!();

    // Check if it's a directory or file
    if source_path.is_dir() {
        // Rename directory
        if path_exists(&new_path) {
            fs::remove_dir_all(&new_path)?;
        }
        fs::rename(&source_path, &new_path)?;
        print_success(&format!(
            "Renamed dataset directory to: {}",
            new_path.display()
        ));

        // Update analysis file if it exists
        let analysis_path = new_path.join("dataset_analysis.json");
        if analysis_path.exists() {
            if let Ok(content) = fs::read_to_string(&analysis_path) {
                if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(obj) = json.as_object_mut() {
                        obj.insert("name".to_string(), serde_json::json!(args.new_name));
                        obj.insert(
                            "path".to_string(),
                            serde_json::json!(new_path.display().to_string()),
                        );
                        obj.insert(
                            "renamed_at".to_string(),
                            serde_json::json!(chrono::Utc::now().to_rfc3339()),
                        );
                        if let Ok(updated) = serde_json::to_string_pretty(&json) {
                            let _ = fs::write(&analysis_path, updated);
                        }
                    }
                }
            }
            print_info("Updated dataset analysis file");
        }

        // Count files
        let file_count = walkdir::WalkDir::new(&new_path)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            .count();
        print_kv("Files", &file_count.to_string());
    } else {
        // Rename single file (e.g., CSV dataset)
        let extension = source_path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default();
        let new_file_path = parent.join(format!("{}{}", args.new_name, extension));

        if path_exists(&new_file_path) {
            fs::remove_file(&new_file_path)?;
        }
        fs::rename(&source_path, &new_file_path)?;
        print_success(&format!(
            "Renamed dataset file to: {}",
            new_file_path.display()
        ));
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_rename_file() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("old_model.axonml");
        fs::write(&source, "test data").unwrap();

        // Manually test the rename logic
        let new_path = dir.path().join("new_model.axonml");
        fs::rename(&source, &new_path).unwrap();

        assert!(!source.exists());
        assert!(new_path.exists());
        assert_eq!(fs::read_to_string(&new_path).unwrap(), "test data");
    }

    #[test]
    fn test_rename_directory() {
        let dir = tempdir().unwrap();
        let source_dir = dir.path().join("old_dataset");
        fs::create_dir(&source_dir).unwrap();
        fs::write(source_dir.join("data.csv"), "col1,col2\n1,2").unwrap();

        let new_dir = dir.path().join("new_dataset");
        fs::rename(&source_dir, &new_dir).unwrap();

        assert!(!source_dir.exists());
        assert!(new_dir.exists());
        assert!(new_dir.join("data.csv").exists());
    }
}
