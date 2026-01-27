//! Event - Event Handling for Axonml TUI
//!
//! Handles keyboard input and other terminal events.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use std::time::Duration;

use crate::app::{App, Tab};

// =============================================================================
// Event Handler
// =============================================================================

/// Poll for events with timeout
pub fn poll_event(timeout: Duration) -> std::io::Result<bool> {
    event::poll(timeout)
}

/// Read the next event
pub fn read_event() -> std::io::Result<Event> {
    event::read()
}

/// Handle a key event
pub fn handle_key_event(app: &mut App, key: KeyEvent) {
    // If help overlay is shown, close it on any key
    if app.show_help {
        app.show_help = false;
        return;
    }

    // Global keybindings
    match key.code {
        // Quit
        KeyCode::Char('q') => {
            app.quit();
            return;
        }
        // Help
        KeyCode::Char('?') => {
            app.toggle_help();
            return;
        }
        // Tab navigation
        KeyCode::Tab => {
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                app.prev_tab();
            } else {
                app.next_tab();
            }
            return;
        }
        KeyCode::BackTab => {
            app.prev_tab();
            return;
        }
        // Direct tab access
        KeyCode::Char('1') => {
            app.go_to_tab(Tab::Model);
            return;
        }
        KeyCode::Char('2') => {
            app.go_to_tab(Tab::Data);
            return;
        }
        KeyCode::Char('3') => {
            app.go_to_tab(Tab::Training);
            return;
        }
        KeyCode::Char('4') => {
            app.go_to_tab(Tab::Graphs);
            return;
        }
        KeyCode::Char('5') => {
            app.go_to_tab(Tab::Files);
            return;
        }
        _ => {}
    }

    // Tab-specific keybindings
    match app.active_tab {
        Tab::Model => handle_model_keys(app, key),
        Tab::Data => handle_data_keys(app, key),
        Tab::Training => handle_training_keys(app, key),
        Tab::Graphs => handle_graphs_keys(app, key),
        Tab::Files => handle_files_keys(app, key),
        Tab::Help => handle_help_keys(app, key),
    }
}

// =============================================================================
// Tab-Specific Handlers
// =============================================================================

fn handle_model_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.model_view.select_prev();
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.model_view.select_next();
        }
        KeyCode::Enter | KeyCode::Char('d') => {
            app.model_view.toggle_details();
        }
        _ => {}
    }
}

fn handle_data_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.data_view.select_prev();
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.data_view.select_next();
        }
        KeyCode::Left | KeyCode::Char('h') => {
            app.data_view.prev_panel();
        }
        KeyCode::Right | KeyCode::Char('l') => {
            app.data_view.next_panel();
        }
        _ => {}
    }
}

fn handle_training_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.training_view.scroll_up();
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.training_view.scroll_down();
        }
        KeyCode::Char('p') => {
            app.training_view.toggle_pause();
        }
        KeyCode::Char('r') => {
            app.training_view.refresh();
        }
        _ => {}
    }
}

fn handle_graphs_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Left | KeyCode::Char('h') => {
            app.graphs_view.prev_chart();
        }
        KeyCode::Right | KeyCode::Char('l') => {
            app.graphs_view.next_chart();
        }
        KeyCode::Char('z') => {
            app.graphs_view.toggle_zoom();
        }
        _ => {}
    }
}

fn handle_files_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.files_view.select_prev();
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.files_view.select_next();
        }
        KeyCode::Enter => {
            if let Some(path) = app.files_view.open_selected() {
                // Determine file type and load appropriately
                let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                match extension {
                    "axonml" | "onnx" | "safetensors" | "pt" | "pth" => {
                        app.load_model(path);
                    }
                    "npz" | "csv" | "parquet" => {
                        app.load_dataset(path);
                    }
                    _ => {
                        // Just update the preview
                    }
                }
            }
        }
        KeyCode::Backspace | KeyCode::Char('u') => {
            app.files_view.go_up();
        }
        KeyCode::Char('~') => {
            app.files_view.go_home();
        }
        _ => {}
    }
}

fn handle_help_keys(_app: &mut App, _key: KeyEvent) {
    // Help view doesn't need special handling
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quit_key() {
        let mut app = App::new();
        assert!(!app.should_quit);

        handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE),
        );

        assert!(app.should_quit);
    }

    #[test]
    fn test_tab_navigation() {
        let mut app = App::new();
        assert_eq!(app.active_tab, Tab::Model);

        handle_key_event(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.active_tab, Tab::Data);

        handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE),
        );
        assert_eq!(app.active_tab, Tab::Model);
    }

    #[test]
    fn test_direct_tab_access() {
        let mut app = App::new();

        handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE),
        );
        assert_eq!(app.active_tab, Tab::Training);

        handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE),
        );
        assert_eq!(app.active_tab, Tab::Model);
    }
}
