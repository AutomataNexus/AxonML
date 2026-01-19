//! Axonml TUI - Terminal User Interface for ML Framework
//!
//! A comprehensive terminal-based user interface for the Axonml ML framework,
//! providing visualization of models, datasets, training progress, and more.
//!
//! # Features
//! - Model architecture visualization
//! - Dataset structure and statistics
//! - Real-time training progress monitoring
//! - Loss and accuracy graphs
//! - File browser for models and datasets
//!
//! # Example
//! ```no_run
//! use axonml_tui::run;
//!
//! fn main() -> std::io::Result<()> {
//!     run(None, None)
//! }
//! ```
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// TUI-specific allowances
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unused_self)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::single_match_else)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::option_map_or_none)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::redundant_else)]
#![allow(clippy::needless_raw_string_hashes)]
#![allow(clippy::needless_bool)]
#![allow(clippy::bool_comparison)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::return_self_not_must_use)]

pub mod app;
pub mod event;
pub mod theme;
pub mod ui;
pub mod views;

pub use app::{App, Tab};
pub use theme::AxonmlTheme;
pub use views::{DataView, FilesView, GraphsView, HelpView, ModelView, TrainingView};

use std::io::{self, stdout};
use std::path::PathBuf;
use std::time::Duration;

use crossterm::{
    event::Event,
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

// =============================================================================
// Main Entry Point
// =============================================================================

/// Run the Axonml TUI application
///
/// # Arguments
/// * `model_path` - Optional path to a model file to load on startup
/// * `data_path` - Optional path to a dataset directory to load on startup
///
/// # Errors
/// Returns an error if terminal initialization fails
pub fn run(model_path: Option<PathBuf>, data_path: Option<PathBuf>) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create application state
    let mut app = App::new();

    // Load model if provided
    if let Some(path) = model_path {
        app.load_model(path);
    }

    // Load dataset if provided
    if let Some(path) = data_path {
        app.load_dataset(path);
    }

    // Main loop
    let result = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

/// Main application loop
fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    loop {
        // Render
        terminal.draw(|frame| {
            ui::render(app, frame);
        })?;

        // Handle events
        if event::poll_event(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read_event()? {
                event::handle_key_event(app, key);
            }
        }

        // Check if we should quit
        if app.should_quit {
            return Ok(());
        }

        // Update training view if active (for real-time updates)
        if app.active_tab == Tab::Training {
            app.training_view.tick();
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_creation() {
        let app = App::new();
        assert_eq!(app.active_tab, Tab::Model);
        assert!(!app.should_quit);
    }
}
