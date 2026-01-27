//! TUI - Terminal User Interface Command
//!
//! Launches the Axonml TUI for interactive model and data exploration.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use crate::cli::TuiArgs;
use crate::error::CliResult;
use colored::Colorize;
use std::path::PathBuf;

// =============================================================================
// TUI Command Execution
// =============================================================================

/// Execute the tui command
pub fn execute(args: TuiArgs) -> CliResult<()> {
    println!("{} {}", "Axonml TUI".cyan().bold(), "v0.1.0".dimmed());
    println!();

    // Convert paths
    let model_path = args.model.map(PathBuf::from);
    let data_path = args.data.map(PathBuf::from);

    // Validate paths if provided
    if let Some(ref path) = model_path {
        if !path.exists() {
            eprintln!(
                "{} Model file not found: {}",
                "warning:".yellow().bold(),
                path.display()
            );
        }
    }

    if let Some(ref path) = data_path {
        if !path.exists() {
            eprintln!(
                "{} Dataset path not found: {}",
                "warning:".yellow().bold(),
                path.display()
            );
        }
    }

    // Launch the TUI
    println!("{}", "Starting terminal user interface...".dimmed());
    println!("{}", "Press 'q' to quit, '?' for help".dimmed());
    println!();

    // Run the TUI application
    axonml_tui::run(model_path, data_path)
        .map_err(|e| crate::error::CliError::Other(format!("TUI error: {}", e)))?;

    println!();
    println!("{}", "TUI session ended.".green());

    Ok(())
}
