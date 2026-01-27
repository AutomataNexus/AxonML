//! App - Main Application State for Axonml TUI
//!
//! Manages the overall application state, tabs, and navigation.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::path::PathBuf;

use crate::views::{DataView, FilesView, GraphsView, HelpView, ModelView, TrainingView};

// =============================================================================
// Application State
// =============================================================================

/// Main application state
pub struct App {
    /// Current active tab
    pub active_tab: Tab,

    /// Whether the app should quit
    pub should_quit: bool,

    /// Model view state
    pub model_view: ModelView,

    /// Data view state
    pub data_view: DataView,

    /// Training view state
    pub training_view: TrainingView,

    /// Graphs view state
    pub graphs_view: GraphsView,

    /// Files view state
    pub files_view: FilesView,

    /// Help view state
    pub help_view: HelpView,

    /// Status message
    pub status_message: Option<StatusMessage>,

    /// Show help overlay
    pub show_help: bool,
}

/// Available tabs in the TUI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    /// Model architecture view
    Model,
    /// Data/dataset view
    Data,
    /// Training progress view
    Training,
    /// Graphs and charts view
    Graphs,
    /// File browser view
    Files,
    /// Help view
    Help,
}

impl Tab {
    /// Get all tabs in order
    pub fn all() -> &'static [Tab] {
        &[
            Tab::Model,
            Tab::Data,
            Tab::Training,
            Tab::Graphs,
            Tab::Files,
            Tab::Help,
        ]
    }

    /// Get tab name
    pub fn name(&self) -> &'static str {
        match self {
            Tab::Model => "Model",
            Tab::Data => "Data",
            Tab::Training => "Training",
            Tab::Graphs => "Graphs",
            Tab::Files => "Files",
            Tab::Help => "Help",
        }
    }

    /// Get tab shortcut key
    pub fn key(&self) -> char {
        match self {
            Tab::Model => '1',
            Tab::Data => '2',
            Tab::Training => '3',
            Tab::Graphs => '4',
            Tab::Files => '5',
            Tab::Help => '?',
        }
    }

    /// Get next tab
    pub fn next(&self) -> Tab {
        match self {
            Tab::Model => Tab::Data,
            Tab::Data => Tab::Training,
            Tab::Training => Tab::Graphs,
            Tab::Graphs => Tab::Files,
            Tab::Files => Tab::Help,
            Tab::Help => Tab::Model,
        }
    }

    /// Get previous tab
    pub fn prev(&self) -> Tab {
        match self {
            Tab::Model => Tab::Help,
            Tab::Data => Tab::Model,
            Tab::Training => Tab::Data,
            Tab::Graphs => Tab::Training,
            Tab::Files => Tab::Graphs,
            Tab::Help => Tab::Files,
        }
    }
}

/// Status message with level
#[derive(Debug, Clone)]
pub struct StatusMessage {
    /// The message text
    pub text: String,
    /// The severity level of the message
    pub level: StatusLevel,
}

/// Status message level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusLevel {
    /// Informational message
    Info,
    /// Success message
    Success,
    /// Warning message
    Warning,
    /// Error message
    Error,
}

// =============================================================================
// Implementation
// =============================================================================

impl App {
    /// Create a new application instance
    pub fn new() -> Self {
        Self {
            active_tab: Tab::Model,
            should_quit: false,
            model_view: ModelView::new(),
            data_view: DataView::new(),
            training_view: TrainingView::new(),
            graphs_view: GraphsView::new(),
            files_view: FilesView::new(),
            help_view: HelpView::new(),
            status_message: None,
            show_help: false,
        }
    }

    /// Switch to the next tab
    pub fn next_tab(&mut self) {
        self.active_tab = self.active_tab.next();
    }

    /// Switch to the previous tab
    pub fn prev_tab(&mut self) {
        self.active_tab = self.active_tab.prev();
    }

    /// Switch to a specific tab
    pub fn go_to_tab(&mut self, tab: Tab) {
        self.active_tab = tab;
    }

    /// Set status message
    pub fn set_status(&mut self, text: impl Into<String>, level: StatusLevel) {
        self.status_message = Some(StatusMessage {
            text: text.into(),
            level,
        });
    }

    /// Clear status message
    pub fn clear_status(&mut self) {
        self.status_message = None;
    }

    /// Toggle help overlay
    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    /// Load a model file
    pub fn load_model(&mut self, path: PathBuf) {
        match self.model_view.load_model(&path) {
            Ok(()) => {
                self.set_status(
                    format!("Loaded model: {}", path.display()),
                    StatusLevel::Success,
                );
                self.active_tab = Tab::Model;
            }
            Err(e) => {
                self.set_status(format!("Failed to load model: {}", e), StatusLevel::Error);
            }
        }
    }

    /// Load a dataset directory
    pub fn load_dataset(&mut self, path: PathBuf) {
        match self.data_view.load_dataset(&path) {
            Ok(()) => {
                self.set_status(
                    format!("Loaded dataset: {}", path.display()),
                    StatusLevel::Success,
                );
                self.active_tab = Tab::Data;
            }
            Err(e) => {
                self.set_status(format!("Failed to load dataset: {}", e), StatusLevel::Error);
            }
        }
    }

    /// Start watching a training log
    pub fn watch_training(&mut self, path: PathBuf) {
        match self.training_view.watch_log(&path) {
            Ok(()) => {
                self.set_status(
                    format!("Watching training: {}", path.display()),
                    StatusLevel::Success,
                );
                self.active_tab = Tab::Training;
            }
            Err(e) => {
                self.set_status(
                    format!("Failed to watch training: {}", e),
                    StatusLevel::Error,
                );
            }
        }
    }

    /// Request quit
    pub fn quit(&mut self) {
        self.should_quit = true;
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tab_navigation() {
        let mut app = App::new();
        assert_eq!(app.active_tab, Tab::Model);

        app.next_tab();
        assert_eq!(app.active_tab, Tab::Data);

        app.prev_tab();
        assert_eq!(app.active_tab, Tab::Model);
    }

    #[test]
    fn test_tab_cycle() {
        let tab = Tab::Help;
        assert_eq!(tab.next(), Tab::Model);

        let tab = Tab::Model;
        assert_eq!(tab.prev(), Tab::Help);
    }

    #[test]
    fn test_status_message() {
        let mut app = App::new();
        assert!(app.status_message.is_none());

        app.set_status("Test message", StatusLevel::Info);
        assert!(app.status_message.is_some());

        app.clear_status();
        assert!(app.status_message.is_none());
    }
}
