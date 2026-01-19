//! Help View - Keyboard Shortcuts and Commands
//!
//! Displays all available keyboard shortcuts and commands organized
//! by category for easy reference.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::theme::AxonmlTheme;

// =============================================================================
// Types
// =============================================================================

/// Key binding entry
#[derive(Debug, Clone)]
pub struct KeyBinding {
    pub key: &'static str,
    pub description: &'static str,
}

/// Help category
#[derive(Debug, Clone)]
pub struct HelpCategory {
    pub name: &'static str,
    pub bindings: Vec<KeyBinding>,
}

// =============================================================================
// Help View
// =============================================================================

/// Help view displaying keyboard shortcuts and commands
pub struct HelpView {
    /// Help categories
    pub categories: Vec<HelpCategory>,

    /// Scroll offset
    pub scroll_offset: usize,

    /// Show compact view
    pub compact: bool,
}

impl HelpView {
    /// Create a new help view
    pub fn new() -> Self {
        Self {
            categories: Self::default_categories(),
            scroll_offset: 0,
            compact: false,
        }
    }

    /// Get default key binding categories
    fn default_categories() -> Vec<HelpCategory> {
        vec![
            HelpCategory {
                name: "Navigation",
                bindings: vec![
                    KeyBinding { key: "Tab / Shift+Tab", description: "Switch between views" },
                    KeyBinding { key: "1-6", description: "Jump to specific view (Model, Data, Training, Graphs, Files, Help)" },
                    KeyBinding { key: "j / Down", description: "Move selection down" },
                    KeyBinding { key: "k / Up", description: "Move selection up" },
                    KeyBinding { key: "h / Left", description: "Collapse / Previous" },
                    KeyBinding { key: "l / Right", description: "Expand / Next" },
                    KeyBinding { key: "g / Home", description: "Go to first item" },
                    KeyBinding { key: "G / End", description: "Go to last item" },
                    KeyBinding { key: "Ctrl+u / PageUp", description: "Page up" },
                    KeyBinding { key: "Ctrl+d / PageDown", description: "Page down" },
                ],
            },
            HelpCategory {
                name: "Model View",
                bindings: vec![
                    KeyBinding { key: "o", description: "Open model file" },
                    KeyBinding { key: "Enter", description: "View layer details" },
                    KeyBinding { key: "d", description: "Toggle detailed view" },
                    KeyBinding { key: "e", description: "Export model info" },
                    KeyBinding { key: "v", description: "Visualize model graph" },
                ],
            },
            HelpCategory {
                name: "Data View",
                bindings: vec![
                    KeyBinding { key: "o", description: "Open dataset file" },
                    KeyBinding { key: "Tab", description: "Switch panel (classes/features)" },
                    KeyBinding { key: "s", description: "View data statistics" },
                    KeyBinding { key: "p", description: "Preview samples" },
                ],
            },
            HelpCategory {
                name: "Training View",
                bindings: vec![
                    KeyBinding { key: "t", description: "Start training" },
                    KeyBinding { key: "Space", description: "Pause/Resume training" },
                    KeyBinding { key: "s", description: "Stop training" },
                    KeyBinding { key: "c", description: "Open training config" },
                    KeyBinding { key: "Ctrl+s", description: "Save checkpoint" },
                    KeyBinding { key: "d", description: "Toggle detailed metrics" },
                ],
            },
            HelpCategory {
                name: "Graphs View",
                bindings: vec![
                    KeyBinding { key: "< / >", description: "Switch chart type" },
                    KeyBinding { key: "+/-", description: "Zoom in/out" },
                    KeyBinding { key: "r", description: "Reset zoom" },
                    KeyBinding { key: "e", description: "Export chart as image" },
                    KeyBinding { key: "l", description: "Toggle legend" },
                ],
            },
            HelpCategory {
                name: "Files View",
                bindings: vec![
                    KeyBinding { key: "Enter", description: "Open file/Toggle directory" },
                    KeyBinding { key: "Backspace", description: "Go to parent directory" },
                    KeyBinding { key: ".", description: "Toggle hidden files" },
                    KeyBinding { key: "/", description: "Search files" },
                    KeyBinding { key: "n", description: "New file/directory" },
                    KeyBinding { key: "Delete", description: "Delete file" },
                    KeyBinding { key: "r", description: "Rename file" },
                ],
            },
            HelpCategory {
                name: "Global",
                bindings: vec![
                    KeyBinding { key: "?", description: "Show/Hide help" },
                    KeyBinding { key: ":", description: "Command mode" },
                    KeyBinding { key: "Ctrl+c / q", description: "Quit application" },
                    KeyBinding { key: "Ctrl+l", description: "Redraw screen" },
                    KeyBinding { key: "Esc", description: "Cancel / Close popup" },
                ],
            },
        ]
    }

    /// Scroll up
    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    /// Scroll down
    pub fn scroll_down(&mut self) {
        self.scroll_offset += 1;
    }

    /// Toggle compact view
    pub fn toggle_compact(&mut self) {
        self.compact = !self.compact;
    }

    /// Render the help view
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),  // Header
                Constraint::Min(10),    // Keybindings
                Constraint::Length(3),  // Footer
            ])
            .split(area);

        self.render_header(frame, chunks[0]);
        self.render_keybindings(frame, chunks[1]);
        self.render_footer(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let header_text = vec![
            Line::from(Span::styled(
                "Axonml TUI - Keyboard Shortcuts",
                AxonmlTheme::title(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Press '?' to toggle help, 'q' to close",
                AxonmlTheme::muted(),
            )),
        ];

        let header = Paragraph::new(header_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Help ", AxonmlTheme::header())),
            )
            .alignment(Alignment::Center);

        frame.render_widget(header, area);
    }

    fn render_keybindings(&self, frame: &mut Frame, area: Rect) {
        // Calculate column layout based on available width
        let num_columns = if area.width > 120 { 3 } else if area.width > 80 { 2 } else { 1 };

        let column_constraints: Vec<Constraint> = (0..num_columns)
            .map(|_| Constraint::Percentage(100 / num_columns as u16))
            .collect();

        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(column_constraints)
            .split(area);

        // Distribute categories across columns
        let categories_per_column = (self.categories.len() + num_columns - 1) / num_columns;

        for (col_idx, column_area) in columns.iter().enumerate() {
            let start_idx = col_idx * categories_per_column;
            let end_idx = (start_idx + categories_per_column).min(self.categories.len());

            if start_idx >= self.categories.len() {
                continue;
            }

            let column_categories = &self.categories[start_idx..end_idx];
            self.render_column(frame, *column_area, column_categories);
        }
    }

    fn render_column(&self, frame: &mut Frame, area: Rect, categories: &[HelpCategory]) {
        let mut lines: Vec<Line> = Vec::new();

        for category in categories {
            // Category header
            lines.push(Line::from(Span::styled(
                format!(" {} ", category.name),
                AxonmlTheme::header().add_modifier(Modifier::UNDERLINED),
            )));
            lines.push(Line::from(""));

            // Key bindings
            for binding in &category.bindings {
                lines.push(Line::from(vec![
                    Span::styled(format!("  {:20}", binding.key), AxonmlTheme::key()),
                    Span::styled(binding.description, AxonmlTheme::key_desc()),
                ]));
            }

            lines.push(Line::from(""));
        }

        let content = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border()),
            );

        frame.render_widget(content, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer_text = Line::from(vec![
            Span::styled("Tip: ", AxonmlTheme::muted()),
            Span::styled(
                "Use Tab to switch views, number keys (1-6) for quick access",
                AxonmlTheme::info(),
            ),
        ]);

        let footer = Paragraph::new(footer_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Quick Tips ", AxonmlTheme::header())),
            )
            .alignment(Alignment::Center);

        frame.render_widget(footer, area);
    }
}

impl Default for HelpView {
    fn default() -> Self {
        Self::new()
    }
}
