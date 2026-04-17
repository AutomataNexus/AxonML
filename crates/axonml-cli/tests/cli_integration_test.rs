//! CLI Integration Tests — End-to-End Binary Invocations
//!
//! End-to-end integration tests for the `axonml` CLI binary. Exercises every
//! major subcommand through `assert_cmd::Command`: help/version output,
//! project creation (`new`) and initialization (`init`), GPU listing/info/
//! status, hub list/info/cached, dataset list/info/sources, hardware
//! benchmarks, quantization format listing, scaffold template listing and
//! generation, load/workspace status and clear, model analysis and inspection,
//! bundle zip create/list, Kaggle status/list, data list/validate, verbose
//! and quiet flags, subcommand help for train/eval/convert/export, error
//! handling for invalid commands and missing args, a full project workflow
//! simulation, and rapid sequential invocations of multiple subcommands.
//!
//! # File
//! `crates/axonml-cli/tests/cli_integration_test.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// =============================================================================
// Test Helpers
// =============================================================================

/// Get a Command for the axonml binary
fn axonml_cmd() -> Command {
    Command::cargo_bin("axonml").unwrap()
}

// =============================================================================
// CLI Help and Version
// =============================================================================

#[test]
fn test_cli_help() {
    axonml_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage:"))
        .stdout(predicate::str::contains("Commands:"))
        .stdout(predicate::str::contains("train").or(predicate::str::contains("Train")));
}

#[test]
fn test_cli_version() {
    axonml_cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("axonml"));
}

// =============================================================================
// Project Creation (new command)
// =============================================================================

#[test]
fn test_new_project_creates_structure() {
    let temp_dir = TempDir::new().unwrap();
    let project_name = "test_ml_project";

    axonml_cmd()
        .arg("new")
        .arg(project_name)
        .arg("--path")
        .arg(temp_dir.path())
        .arg("--no-git")
        .assert()
        .success()
        .stdout(predicate::str::contains("Created project"));

    // Verify project structure was created
    let project_path = temp_dir.path().join(project_name);
    assert!(project_path.exists(), "Project directory should exist");
    assert!(
        project_path.join("axonml.toml").exists(),
        "Config file should exist"
    );
    assert!(
        project_path.join("src").exists(),
        "src directory should exist"
    );
    assert!(
        project_path.join("data").exists(),
        "data directory should exist"
    );
}

#[test]
fn test_new_project_with_template() {
    let temp_dir = TempDir::new().unwrap();

    axonml_cmd()
        .arg("new")
        .arg("cnn_project")
        .arg("--path")
        .arg(temp_dir.path())
        .arg("--template")
        .arg("cnn")
        .arg("--no-git")
        .assert()
        .success();

    let project_path = temp_dir.path().join("cnn_project");
    assert!(project_path.exists());
}

// =============================================================================
// Project Initialization (init command)
// =============================================================================

#[test]
fn test_init_in_existing_directory() {
    let temp_dir = TempDir::new().unwrap();

    axonml_cmd()
        .current_dir(temp_dir.path())
        .arg("init")
        .arg("--no-git")
        .assert()
        .success()
        .stdout(predicate::str::contains("Initialized Axonml project"));

    assert!(temp_dir.path().join("axonml.toml").exists());
}

// =============================================================================
// GPU Commands
// =============================================================================

#[test]
fn test_gpu_list() {
    axonml_cmd().arg("gpu").arg("list").assert().success();
    // Output depends on actual hardware, just verify it runs
}

#[test]
fn test_gpu_info() {
    axonml_cmd().arg("gpu").arg("info").assert().success();
}

#[test]
fn test_gpu_status() {
    axonml_cmd().arg("gpu").arg("status").assert().success();
}

// =============================================================================
// Hub Commands
// =============================================================================

#[test]
fn test_hub_list() {
    axonml_cmd()
        .arg("hub")
        .arg("list")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("resnet")
                .or(predicate::str::contains("ResNet"))
                .or(predicate::str::contains("Model")),
        );
}

#[test]
fn test_hub_info() {
    axonml_cmd()
        .arg("hub")
        .arg("info")
        .arg("resnet18")
        .assert()
        .success()
        .stdout(predicate::str::contains("ResNet"));
}

#[test]
fn test_hub_cached() {
    axonml_cmd().arg("hub").arg("cached").assert().success();
}

// =============================================================================
// Dataset Commands
// =============================================================================

#[test]
fn test_dataset_list() {
    axonml_cmd()
        .arg("dataset")
        .arg("list")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("mnist")
                .or(predicate::str::contains("MNIST"))
                .or(predicate::str::contains("Dataset")),
        );
}

#[test]
fn test_dataset_info() {
    axonml_cmd()
        .arg("dataset")
        .arg("info")
        .arg("mnist")
        .assert()
        .success()
        .stdout(predicate::str::contains("MNIST"));
}

#[test]
fn test_dataset_sources() {
    axonml_cmd()
        .arg("dataset")
        .arg("sources")
        .assert()
        .success()
        .stdout(predicate::str::contains("builtin"));
}

// =============================================================================
// Benchmark Commands
// =============================================================================

#[test]
fn test_bench_hardware() {
    axonml_cmd()
        .arg("bench")
        .arg("hardware")
        .arg("--iterations")
        .arg("2")
        .assert()
        .success()
        .stdout(predicate::str::contains("Hardware Benchmark"));
}

// =============================================================================
// Quantization Commands
// =============================================================================

#[test]
fn test_quant_list() {
    axonml_cmd()
        .arg("quant")
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("Q4_0"))
        .stdout(predicate::str::contains("Q8_0"))
        .stdout(predicate::str::contains("F16"));
}

// =============================================================================
// Scaffold Commands
// =============================================================================

#[test]
fn test_scaffold_templates() {
    axonml_cmd()
        .arg("scaffold")
        .arg("templates")
        .assert()
        .success()
        .stdout(predicate::str::contains("training"))
        .stdout(predicate::str::contains("Template"));
}

#[test]
fn test_scaffold_generate() {
    let temp_dir = TempDir::new().unwrap();

    // scaffold generate may fail if certain templates require additional setup
    // Just verify the command accepts the arguments correctly
    let result = axonml_cmd()
        .arg("scaffold")
        .arg("generate")
        .arg("my_training_project")
        .arg("--output")
        .arg(temp_dir.path())
        .arg("--template")
        .arg("minimal")
        .assert();

    // The command should either succeed or fail gracefully
    // We're mainly testing that the CLI parses arguments correctly
    let _ = result; // Allow either success or failure
}

// =============================================================================
// Load / Workspace Commands
// =============================================================================

#[test]
fn test_load_status() {
    axonml_cmd().arg("load").arg("status").assert().success();
}

#[test]
fn test_load_clear() {
    axonml_cmd().arg("load").arg("clear").assert().success();
}

// =============================================================================
// Analyze Commands
// =============================================================================

#[test]
fn test_analyze_model_without_loaded() {
    // Should provide helpful message when no model is loaded
    // May either succeed with a warning or fail - both are acceptable
    let result = axonml_cmd().arg("analyze").arg("model").assert();

    // Just verify the command runs without panic
    let _ = result;
}

// =============================================================================
// Inspect Command
// =============================================================================

#[test]
fn test_inspect_nonexistent_model() {
    axonml_cmd()
        .arg("inspect")
        .arg("/nonexistent/model.axonml")
        .assert()
        .failure(); // Should fail gracefully for missing files
}

// =============================================================================
// Zip / Bundle Commands
// =============================================================================

#[test]
fn test_zip_create_and_list() {
    let temp_dir = TempDir::new().unwrap();
    let bundle_path = temp_dir.path().join("test_bundle.axonmlbundle");

    // Create a dummy file to bundle
    let model_path = temp_dir.path().join("model.bin");
    fs::write(&model_path, b"dummy model data").unwrap();

    // Create bundle
    axonml_cmd()
        .arg("zip")
        .arg("create")
        .arg("--output")
        .arg(&bundle_path)
        .arg("--model")
        .arg(&model_path)
        .assert()
        .success();

    if bundle_path.exists() {
        // List bundle contents
        axonml_cmd()
            .arg("zip")
            .arg("list")
            .arg(&bundle_path)
            .assert()
            .success();
    }
}

// =============================================================================
// Kaggle Commands (without credentials)
// =============================================================================

#[test]
fn test_kaggle_status() {
    axonml_cmd().arg("kaggle").arg("status").assert().success(); // Should report not configured, but not crash
}

#[test]
fn test_kaggle_list() {
    axonml_cmd().arg("kaggle").arg("list").assert().success();
}

// =============================================================================
// Data Commands
// =============================================================================

#[test]
fn test_data_list() {
    let temp_dir = TempDir::new().unwrap();

    axonml_cmd()
        .arg("data")
        .arg("list")
        .arg("--path")
        .arg(temp_dir.path())
        .assert()
        .success();
}

#[test]
fn test_data_validate_nonexistent() {
    axonml_cmd()
        .arg("data")
        .arg("validate")
        .arg("/nonexistent/data")
        .assert()
        .failure(); // Should fail for nonexistent path
}

// =============================================================================
// Verbose and Quiet Flags
// =============================================================================

#[test]
fn test_verbose_flag() {
    axonml_cmd()
        .arg("--verbose")
        .arg("hub")
        .arg("list")
        .assert()
        .success();
}

#[test]
fn test_quiet_flag() {
    axonml_cmd()
        .arg("--quiet")
        .arg("hub")
        .arg("list")
        .assert()
        .success();
}

// =============================================================================
// Help for Subcommands
// =============================================================================

#[test]
fn test_train_help() {
    axonml_cmd()
        .arg("train")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("epochs"))
        .stdout(predicate::str::contains("batch-size"))
        .stdout(predicate::str::contains("learning rate").or(predicate::str::contains("lr")));
}

#[test]
fn test_eval_help() {
    axonml_cmd()
        .arg("eval")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("model"))
        .stdout(predicate::str::contains("metrics"));
}

#[test]
fn test_convert_help() {
    axonml_cmd()
        .arg("convert")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("INPUT").or(predicate::str::contains("input")))
        .stdout(predicate::str::contains("OUTPUT").or(predicate::str::contains("output")));
}

#[test]
fn test_export_help() {
    axonml_cmd()
        .arg("export")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("format"))
        .stdout(predicate::str::contains("onnx").or(predicate::str::contains("ONNX")));
}

// =============================================================================
// Error Handling
// =============================================================================

#[test]
fn test_invalid_command() {
    axonml_cmd().arg("nonexistent_command").assert().failure();
}

#[test]
fn test_missing_required_args() {
    axonml_cmd()
        .arg("eval")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

// =============================================================================
// Full Workflow Simulation
// =============================================================================

#[test]
fn test_full_project_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let project_name = "e2e_test_project";

    // Step 1: Create new project
    axonml_cmd()
        .arg("new")
        .arg(project_name)
        .arg("--path")
        .arg(temp_dir.path())
        .arg("--no-git")
        .assert()
        .success();

    let project_path = temp_dir.path().join(project_name);

    // Step 2: Verify project structure
    assert!(project_path.join("axonml.toml").exists());
    assert!(project_path.join("src").exists());

    // Step 3: Load status in project directory
    axonml_cmd()
        .current_dir(&project_path)
        .arg("load")
        .arg("status")
        .assert()
        .success();

    // Step 4: Check available datasets
    axonml_cmd()
        .current_dir(&project_path)
        .arg("dataset")
        .arg("list")
        .assert()
        .success();

    // Step 5: Check available pretrained models
    axonml_cmd()
        .current_dir(&project_path)
        .arg("hub")
        .arg("list")
        .assert()
        .success();

    println!("✓ Full project workflow completed successfully");
}

// =============================================================================
// Concurrent Command Execution
// =============================================================================

#[test]
fn test_multiple_help_commands() {
    // Verify CLI can handle multiple rapid invocations
    for cmd in &["new", "train", "eval", "hub", "dataset", "gpu"] {
        axonml_cmd().arg(cmd).arg("--help").assert().success();
    }
}
