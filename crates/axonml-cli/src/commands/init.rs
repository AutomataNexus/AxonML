//! Init - Initialize Axonml in Existing Directory
//!
//! Initializes a Axonml project in an existing directory.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::env;
use std::fs;
use std::path::PathBuf;

use super::utils::{ensure_dir, print_info, print_step, print_success, print_warning};
use crate::cli::InitArgs;
use crate::config::ProjectConfig;
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `init` command
pub fn execute(args: InitArgs) -> CliResult<()> {
    let current_dir = env::current_dir()?;

    // Get project name from args or directory name
    let project_name = args.name.unwrap_or_else(|| {
        current_dir
            .file_name()
            .and_then(|n| n.to_str())
            .map_or_else(|| "axonml-project".to_string(), String::from)
    });

    // Check if axonml.toml already exists
    let config_path = current_dir.join("axonml.toml");
    if config_path.exists() && !args.force {
        return Err(CliError::ProjectExists(
            "axonml.toml already exists. Use --force to overwrite.".to_string(),
        ));
    }

    println!();
    print_info(&format!("Initializing Axonml project: {project_name}"));
    println!();

    // Create necessary directories
    print_step(1, 4, "Creating directory structure...");
    create_directories(&current_dir)?;

    // Create configuration
    print_step(2, 4, "Creating configuration...");
    create_config(&current_dir, &project_name)?;

    // Create source templates if src/ doesn't exist
    print_step(3, 4, "Setting up source structure...");
    create_source_structure(&current_dir, &project_name)?;

    // Initialize git if not exists and not disabled
    if args.no_git {
        print_step(4, 4, "Skipping git setup...");
    } else {
        print_step(4, 4, "Setting up git...");
        setup_git(&current_dir)?;
    }

    println!();
    print_success(&format!(
        "Initialized Axonml project '{project_name}' in current directory"
    ));
    println!();
    print_info("Get started with:");
    println!("  axonml train");
    println!();

    Ok(())
}

// =============================================================================
// Directory Creation
// =============================================================================

fn create_directories(base_path: &PathBuf) -> CliResult<()> {
    let dirs = [
        "src",
        "src/models",
        "src/data",
        "configs",
        "checkpoints",
        "logs",
        "outputs",
    ];

    for dir in &dirs {
        let dir_path = base_path.join(dir);
        if !dir_path.exists() {
            ensure_dir(&dir_path)?;
        }
    }

    // Create data directories only if they don't exist
    let data_dirs = ["data", "data/train", "data/val", "data/test"];
    for dir in &data_dirs {
        let dir_path = base_path.join(dir);
        if !dir_path.exists() {
            ensure_dir(&dir_path)?;
        }
    }

    Ok(())
}

// =============================================================================
// Configuration
// =============================================================================

fn create_config(base_path: &PathBuf, project_name: &str) -> CliResult<()> {
    // Create axonml.toml
    let config_path = base_path.join("axonml.toml");
    let config = ProjectConfig::new(project_name);
    config.save(&config_path)?;

    // Create .gitignore if it doesn't exist
    let gitignore_path = base_path.join(".gitignore");
    if gitignore_path.exists() {
        // Append Axonml-specific entries if not present
        let existing = fs::read_to_string(&gitignore_path)?;
        if !existing.contains("# Axonml") {
            let axonml_entries = "\n# Axonml\n/checkpoints/\n/logs/\n/outputs/\n*.axonml\n";
            fs::write(&gitignore_path, format!("{existing}{axonml_entries}"))?;
            print_warning("Appended Axonml entries to existing .gitignore");
        }
    } else {
        let gitignore = create_gitignore();
        fs::write(&gitignore_path, gitignore)?;
    }

    // Create example training config
    let train_config_path = base_path.join("configs/train.toml");
    if !train_config_path.exists() {
        let train_config = create_train_config();
        fs::write(&train_config_path, train_config)?;
    }

    Ok(())
}

fn create_gitignore() -> &'static str {
    r"# Axonml project gitignore

# Output directories
/checkpoints/
/logs/
/outputs/

# Data (typically large)
/data/train/
/data/val/
/data/test/

# Python virtual environment
venv/
.venv/
__pycache__/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Model files (typically large)
*.axonml
*.onnx
*.pt
*.pth
*.safetensors

# Temporary files
*.tmp
*.log
"
}

fn create_train_config() -> &'static str {
    r#"# Training Configuration

[training]
epochs = 10
batch_size = 32
learning_rate = 0.001
device = "cpu"

[training.optimizer]
name = "adam"
weight_decay = 0.0001

[model]
architecture = "custom"

[data]
train_path = "data/train"
val_path = "data/val"
shuffle = true
"#
}

// =============================================================================
// Source Structure
// =============================================================================

fn create_source_structure(base_path: &PathBuf, project_name: &str) -> CliResult<()> {
    // Only create source files if they don't exist
    let main_path = base_path.join("src/main.rs");
    if !main_path.exists() {
        let main_content = format!(
            r#"//! {project_name} - Main Entry Point

use axonml::prelude::*;

mod models;
mod data;

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    println!("Starting {project_name} training...");

    // Your training code here

    Ok(())
}}
"#
        );
        fs::write(&main_path, main_content)?;
    }

    let models_path = base_path.join("src/models/mod.rs");
    if !models_path.exists() {
        let models_content = format!(
            r"//! Model definitions for {project_name}

use axonml::prelude::*;

// Define your models here
"
        );
        fs::write(&models_path, models_content)?;
    }

    let data_path = base_path.join("src/data/mod.rs");
    if !data_path.exists() {
        let data_content = format!(
            r"//! Data loading for {project_name}

use axonml::prelude::*;

// Define your data loaders here
"
        );
        fs::write(&data_path, data_content)?;
    }

    // Create .gitkeep files for empty directories
    let gitkeep_dirs = ["checkpoints/.gitkeep", "logs/.gitkeep", "outputs/.gitkeep"];
    for path in &gitkeep_dirs {
        let full_path = base_path.join(path);
        if !full_path.exists() {
            fs::write(&full_path, "")?;
        }
    }

    Ok(())
}

// =============================================================================
// Git Setup
// =============================================================================

fn setup_git(base_path: &PathBuf) -> CliResult<()> {
    use std::process::Command;

    // Check if already a git repo
    let git_dir = base_path.join(".git");
    if git_dir.exists() {
        return Ok(());
    }

    // Try to initialize git
    let output = Command::new("git")
        .arg("init")
        .current_dir(base_path)
        .output();

    if let Ok(result) = output {
        if result.status.success() {
            // Make initial commit if there are files
            let _ = Command::new("git")
                .args(["add", "."])
                .current_dir(base_path)
                .output();

            let _ = Command::new("git")
                .args(["commit", "-m", "Initialize Axonml project"])
                .current_dir(base_path)
                .output();
        }
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
    fn test_create_directories() {
        let temp = tempdir().unwrap();
        let base = temp.path().to_path_buf();

        create_directories(&base).unwrap();

        assert!(base.join("src").exists());
        assert!(base.join("configs").exists());
        assert!(base.join("checkpoints").exists());
    }

    #[test]
    fn test_create_config() {
        let temp = tempdir().unwrap();
        let base = temp.path().to_path_buf();
        std::fs::create_dir_all(base.join("configs")).unwrap();

        create_config(&base, "test-project").unwrap();

        assert!(base.join("axonml.toml").exists());
        assert!(base.join(".gitignore").exists());
    }
}
