//! UI - Main UI Rendering for Axonml TUI
//!
//! Renders the complete TUI interface including tabs, status bar, and views.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs},
    Frame,
};

use crate::app::{App, StatusLevel, Tab};
use crate::theme::{AxonmlTheme, CIRCUIT_UNDERLINE, FERRITE_LOGO_SMALL};

// =============================================================================
// Main UI Render
// =============================================================================

/// Render the complete UI
pub fn render(app: &mut App, frame: &mut Frame) {
    let area = frame.area();

    // Main layout: header, content, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header with tabs
            Constraint::Min(10),   // Main content
            Constraint::Length(3), // Status bar
        ])
        .split(area);

    render_header(app, frame, chunks[0]);
    render_content(app, frame, chunks[1]);
    render_footer(app, frame, chunks[2]);

    // Render help overlay if active
    if app.show_help {
        render_help_overlay(frame, area);
    }
}

// =============================================================================
// Header
// =============================================================================

fn render_header(app: &App, frame: &mut Frame, area: Rect) {
    let titles: Vec<Line> = Tab::all()
        .iter()
        .map(|tab| {
            let style = if *tab == app.active_tab {
                AxonmlTheme::tab_active()
            } else {
                AxonmlTheme::tab_inactive()
            };
            Line::from(Span::styled(
                format!(" {} [{}] ", tab.name(), tab.key()),
                style,
            ))
        })
        .collect();

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(AxonmlTheme::border())
                .title(Span::styled(
                    format!(" {} ", FERRITE_LOGO_SMALL),
                    AxonmlTheme::header(),
                )),
        )
        .select(Tab::all().iter().position(|t| *t == app.active_tab).unwrap_or(0))
        .highlight_style(AxonmlTheme::tab_active().add_modifier(Modifier::UNDERLINED));

    frame.render_widget(tabs, area);
}

// =============================================================================
// Content
// =============================================================================

fn render_content(app: &mut App, frame: &mut Frame, area: Rect) {
    match app.active_tab {
        Tab::Model => app.model_view.render(frame, area),
        Tab::Data => app.data_view.render(frame, area),
        Tab::Training => app.training_view.render(frame, area),
        Tab::Graphs => app.graphs_view.render(frame, area),
        Tab::Files => app.files_view.render(frame, area),
        Tab::Help => app.help_view.render(frame, area),
    }
}

// =============================================================================
// Footer / Status Bar
// =============================================================================

fn render_footer(app: &App, frame: &mut Frame, area: Rect) {
    let status_text = if let Some(msg) = &app.status_message {
        let style = match msg.level {
            StatusLevel::Info => AxonmlTheme::info(),
            StatusLevel::Success => AxonmlTheme::success(),
            StatusLevel::Warning => AxonmlTheme::warning(),
            StatusLevel::Error => AxonmlTheme::error(),
        };
        vec![
            Line::from(Span::styled(&msg.text, style)),
            Line::from(vec![
                Span::styled(" Tab/Shift+Tab ", AxonmlTheme::key()),
                Span::styled("switch tabs  ", AxonmlTheme::key_desc()),
                Span::styled(" ? ", AxonmlTheme::key()),
                Span::styled("help  ", AxonmlTheme::key_desc()),
                Span::styled(" q ", AxonmlTheme::key()),
                Span::styled("quit", AxonmlTheme::key_desc()),
            ]),
        ]
    } else {
        vec![
            Line::from(Span::styled(CIRCUIT_UNDERLINE, AxonmlTheme::accent())),
            Line::from(vec![
                Span::styled(" Tab/Shift+Tab ", AxonmlTheme::key()),
                Span::styled("switch tabs  ", AxonmlTheme::key_desc()),
                Span::styled(" ? ", AxonmlTheme::key()),
                Span::styled("help  ", AxonmlTheme::key_desc()),
                Span::styled(" q ", AxonmlTheme::key()),
                Span::styled("quit", AxonmlTheme::key_desc()),
            ]),
        ]
    };

    let status = Paragraph::new(status_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(AxonmlTheme::border()),
        );

    frame.render_widget(status, area);
}

// =============================================================================
// Help Overlay
// =============================================================================

fn render_help_overlay(frame: &mut Frame, area: Rect) {
    // Center the help popup
    let popup_area = centered_rect(60, 70, area);

    // Clear the background
    frame.render_widget(
        Block::default().style(AxonmlTheme::default()),
        popup_area,
    );

    let help_text = vec![
        Line::from(""),
        Line::from(Span::styled("  Keyboard Shortcuts", AxonmlTheme::header())),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Tab        ", AxonmlTheme::key()),
            Span::styled("Next tab", AxonmlTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("  Shift+Tab  ", AxonmlTheme::key()),
            Span::styled("Previous tab", AxonmlTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("  1-5        ", AxonmlTheme::key()),
            Span::styled("Go to tab", AxonmlTheme::key_desc()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ↑/k        ", AxonmlTheme::key()),
            Span::styled("Move up", AxonmlTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("  ↓/j        ", AxonmlTheme::key()),
            Span::styled("Move down", AxonmlTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("  Enter      ", AxonmlTheme::key()),
            Span::styled("Select/Open", AxonmlTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("  Esc        ", AxonmlTheme::key()),
            Span::styled("Back/Cancel", AxonmlTheme::key_desc()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  o          ", AxonmlTheme::key()),
            Span::styled("Open file", AxonmlTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("  r          ", AxonmlTheme::key()),
            Span::styled("Refresh", AxonmlTheme::key_desc()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ?          ", AxonmlTheme::key()),
            Span::styled("Toggle help", AxonmlTheme::key_desc()),
        ]),
        Line::from(vec![
            Span::styled("  q          ", AxonmlTheme::key()),
            Span::styled("Quit", AxonmlTheme::key_desc()),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Press any key to close",
            AxonmlTheme::muted(),
        )),
    ];

    let help = Paragraph::new(help_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(AxonmlTheme::border_active())
                .title(Span::styled(" Help ", AxonmlTheme::header())),
        );

    frame.render_widget(help, popup_area);
}

// =============================================================================
// Helpers
// =============================================================================

/// Create a centered rect
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
