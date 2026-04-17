//! Graphs View - Display Loss Curves and Accuracy Charts
//!
//! # File
//! `crates/axonml-tui/src/views/graphs.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
};

use crate::theme::{AxonmlTheme, INFO, TEAL, TERRACOTTA};

// =============================================================================
// Types
// =============================================================================

/// Chart data point
pub type DataPoint = (f64, f64);

/// Chart data series for custom metrics.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DataSeries {
    /// Series display name
    pub name: String,
    /// Data points (x, y)
    pub data: Vec<DataPoint>,
    /// Line color
    pub color: ratatui::style::Color,
    /// Point marker style
    pub marker: symbols::Marker,
}

#[allow(dead_code)]
impl DataSeries {
    /// Creates a new data series.
    pub fn new(name: &str, color: ratatui::style::Color) -> Self {
        Self {
            name: name.to_string(),
            data: Vec::new(),
            color,
            marker: symbols::Marker::Braille,
        }
    }

    /// Adds a data point.
    pub fn push(&mut self, x: f64, y: f64) {
        self.data.push((x, y));
    }
}

/// Chart type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartType {
    Loss,
    Accuracy,
    LearningRate,
}

impl ChartType {
    fn as_str(&self) -> &'static str {
        match self {
            ChartType::Loss => "Loss",
            ChartType::Accuracy => "Accuracy",
            ChartType::LearningRate => "Learning Rate",
        }
    }
}

// =============================================================================
// Graphs View
// =============================================================================

/// Graphs view state for training visualization
pub struct GraphsView {
    /// Training loss data
    pub train_loss: Vec<DataPoint>,

    /// Validation loss data
    pub val_loss: Vec<DataPoint>,

    /// Training accuracy data
    pub train_acc: Vec<DataPoint>,

    /// Validation accuracy data
    pub val_acc: Vec<DataPoint>,

    /// Learning rate schedule
    pub learning_rate: Vec<DataPoint>,

    /// Currently selected chart
    pub active_chart: ChartType,

    /// X-axis bounds
    pub x_bounds: [f64; 2],

    /// Full (unzoomed) X-axis bounds
    pub x_bounds_full: [f64; 2],

    /// Y-axis bounds for loss
    pub loss_bounds: [f64; 2],

    /// Y-axis bounds for accuracy
    pub acc_bounds: [f64; 2],

    /// Whether zoom is active (show last 50% of data)
    pub zoomed: bool,
}

impl GraphsView {
    /// Create a new graphs view with demo data
    pub fn new() -> Self {
        let mut view = Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            train_acc: Vec::new(),
            val_acc: Vec::new(),
            learning_rate: Vec::new(),
            active_chart: ChartType::Loss,
            x_bounds: [0.0, 20.0],
            x_bounds_full: [0.0, 20.0],
            loss_bounds: [0.0, 2.5],
            acc_bounds: [0.0, 100.0],
            zoomed: false,
        };

        view.load_demo_data();
        view
    }

    /// Load demo training data for visualization
    pub fn load_demo_data(&mut self) {
        // Simulated training data over 15 epochs
        self.train_loss = vec![
            (1.0, 2.31),
            (2.0, 1.85),
            (3.0, 1.23),
            (4.0, 0.86),
            (5.0, 0.61),
            (6.0, 0.48),
            (7.0, 0.39),
            (8.0, 0.32),
            (9.0, 0.27),
            (10.0, 0.23),
            (11.0, 0.20),
            (12.0, 0.17),
            (13.0, 0.15),
            (14.0, 0.13),
            (15.0, 0.12),
        ];

        self.val_loss = vec![
            (1.0, 2.30),
            (2.0, 1.76),
            (3.0, 1.19),
            (4.0, 0.82),
            (5.0, 0.60),
            (6.0, 0.49),
            (7.0, 0.41),
            (8.0, 0.36),
            (9.0, 0.32),
            (10.0, 0.29),
            (11.0, 0.27),
            (12.0, 0.25),
            (13.0, 0.24),
            (14.0, 0.23),
            (15.0, 0.22),
        ];

        self.train_acc = vec![
            (1.0, 11.2),
            (2.0, 34.2),
            (3.0, 56.7),
            (4.0, 71.2),
            (5.0, 79.8),
            (6.0, 84.5),
            (7.0, 87.8),
            (8.0, 90.2),
            (9.0, 91.8),
            (10.0, 93.1),
            (11.0, 94.0),
            (12.0, 94.7),
            (13.0, 95.2),
            (14.0, 95.6),
            (15.0, 95.9),
        ];

        self.val_acc = vec![
            (1.0, 11.8),
            (2.0, 35.8),
            (3.0, 58.2),
            (4.0, 72.4),
            (5.0, 80.5),
            (6.0, 84.2),
            (7.0, 86.9),
            (8.0, 89.1),
            (9.0, 90.5),
            (10.0, 91.4),
            (11.0, 92.1),
            (12.0, 92.6),
            (13.0, 93.0),
            (14.0, 93.3),
            (15.0, 93.5),
        ];

        self.learning_rate = vec![
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 1.0),
            (4.0, 1.0),
            (5.0, 0.5),
            (6.0, 0.5),
            (7.0, 0.5),
            (8.0, 0.25),
            (9.0, 0.25),
            (10.0, 0.25),
            (11.0, 0.125),
            (12.0, 0.125),
            (13.0, 0.125),
            (14.0, 0.0625),
            (15.0, 0.0625),
        ];

        self.x_bounds = [0.0, 16.0];
        self.x_bounds_full = [0.0, 16.0];
        self.zoomed = false;
    }

    /// Switch to next chart type
    pub fn next_chart(&mut self) {
        self.active_chart = match self.active_chart {
            ChartType::Loss => ChartType::Accuracy,
            ChartType::Accuracy => ChartType::LearningRate,
            ChartType::LearningRate => ChartType::Loss,
        };
    }

    /// Switch to previous chart type
    pub fn prev_chart(&mut self) {
        self.active_chart = match self.active_chart {
            ChartType::Loss => ChartType::LearningRate,
            ChartType::Accuracy => ChartType::Loss,
            ChartType::LearningRate => ChartType::Accuracy,
        };
    }

    /// Toggle zoom mode: zooms into the last 50% of the x-axis range
    /// to see recent training detail, or zooms back out to full range.
    pub fn toggle_zoom(&mut self) {
        self.zoomed = !self.zoomed;
        if self.zoomed {
            let mid = self.x_bounds_full[0].midpoint(self.x_bounds_full[1]);
            self.x_bounds = [mid, self.x_bounds_full[1]];
        } else {
            self.x_bounds = self.x_bounds_full;
        }
    }

    /// Render the graphs view
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Chart selector
                Constraint::Min(15),   // Main chart
                Constraint::Length(5), // Legend/info
            ])
            .split(area);

        self.render_selector(frame, chunks[0]);
        self.render_chart(frame, chunks[1]);
        self.render_legend(frame, chunks[2]);
    }

    fn render_selector(&self, frame: &mut Frame, area: Rect) {
        let tabs: Vec<Span> = [
            ChartType::Loss,
            ChartType::Accuracy,
            ChartType::LearningRate,
        ]
        .iter()
        .map(|ct| {
            let style = if *ct == self.active_chart {
                AxonmlTheme::tab_active()
            } else {
                AxonmlTheme::tab_inactive()
            };
            Span::styled(format!(" {} ", ct.as_str()), style)
        })
        .collect();

        let selector = Paragraph::new(Line::from(tabs)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(AxonmlTheme::border())
                .title(Span::styled(
                    " Chart Type (</> to switch) ",
                    AxonmlTheme::header(),
                )),
        );

        frame.render_widget(selector, area);
    }

    fn render_chart(&self, frame: &mut Frame, area: Rect) {
        match self.active_chart {
            ChartType::Loss => self.render_loss_chart(frame, area),
            ChartType::Accuracy => self.render_accuracy_chart(frame, area),
            ChartType::LearningRate => self.render_lr_chart(frame, area),
        }
    }

    fn render_loss_chart(&self, frame: &mut Frame, area: Rect) {
        let datasets = vec![
            Dataset::default()
                .name("Train Loss")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(TEAL))
                .data(&self.train_loss),
            Dataset::default()
                .name("Val Loss")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(TERRACOTTA))
                .data(&self.val_loss),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border_focused())
                    .title(Span::styled(" Loss Curves ", AxonmlTheme::header())),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Epoch", AxonmlTheme::graph_label()))
                    .style(AxonmlTheme::graph_axis())
                    .bounds(self.x_bounds)
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw("5"),
                        Span::raw("10"),
                        Span::raw("15"),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("Loss", AxonmlTheme::graph_label()))
                    .style(AxonmlTheme::graph_axis())
                    .bounds(self.loss_bounds)
                    .labels(vec![Span::raw("0.0"), Span::raw("1.0"), Span::raw("2.0")]),
            );

        frame.render_widget(chart, area);
    }

    fn render_accuracy_chart(&self, frame: &mut Frame, area: Rect) {
        let datasets = vec![
            Dataset::default()
                .name("Train Acc")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(TEAL))
                .data(&self.train_acc),
            Dataset::default()
                .name("Val Acc")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(TERRACOTTA))
                .data(&self.val_acc),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border_focused())
                    .title(Span::styled(" Accuracy Curves ", AxonmlTheme::header())),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Epoch", AxonmlTheme::graph_label()))
                    .style(AxonmlTheme::graph_axis())
                    .bounds(self.x_bounds)
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw("5"),
                        Span::raw("10"),
                        Span::raw("15"),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("Accuracy %", AxonmlTheme::graph_label()))
                    .style(AxonmlTheme::graph_axis())
                    .bounds(self.acc_bounds)
                    .labels(vec![Span::raw("0"), Span::raw("50"), Span::raw("100")]),
            );

        frame.render_widget(chart, area);
    }

    fn render_lr_chart(&self, frame: &mut Frame, area: Rect) {
        let datasets = vec![
            Dataset::default()
                .name("Learning Rate")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(INFO))
                .data(&self.learning_rate),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border_focused())
                    .title(Span::styled(
                        " Learning Rate Schedule ",
                        AxonmlTheme::header(),
                    )),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Epoch", AxonmlTheme::graph_label()))
                    .style(AxonmlTheme::graph_axis())
                    .bounds(self.x_bounds)
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw("5"),
                        Span::raw("10"),
                        Span::raw("15"),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("LR (relative)", AxonmlTheme::graph_label()))
                    .style(AxonmlTheme::graph_axis())
                    .bounds([0.0, 1.2])
                    .labels(vec![Span::raw("0"), Span::raw("0.5"), Span::raw("1.0")]),
            );

        frame.render_widget(chart, area);
    }

    fn render_legend(&self, frame: &mut Frame, area: Rect) {
        let legend_text = match self.active_chart {
            ChartType::Loss => vec![
                Line::from(vec![
                    Span::styled("\u{2588}\u{2588}", Style::default().fg(TEAL)),
                    Span::styled(" Train Loss", AxonmlTheme::graph_label()),
                    Span::raw("    "),
                    Span::styled("\u{2588}\u{2588}", Style::default().fg(TERRACOTTA)),
                    Span::styled(" Val Loss", AxonmlTheme::graph_label()),
                ]),
                Line::from(vec![
                    Span::styled("Latest: ", AxonmlTheme::muted()),
                    Span::styled(
                        format!(
                            "Train {:.4}",
                            self.train_loss.last().map(|p| p.1).unwrap_or(0.0)
                        ),
                        AxonmlTheme::metric_value(),
                    ),
                    Span::raw("  "),
                    Span::styled(
                        format!(
                            "Val {:.4}",
                            self.val_loss.last().map(|p| p.1).unwrap_or(0.0)
                        ),
                        AxonmlTheme::accent(),
                    ),
                ]),
            ],
            ChartType::Accuracy => vec![
                Line::from(vec![
                    Span::styled("\u{2588}\u{2588}", Style::default().fg(TEAL)),
                    Span::styled(" Train Acc", AxonmlTheme::graph_label()),
                    Span::raw("    "),
                    Span::styled("\u{2588}\u{2588}", Style::default().fg(TERRACOTTA)),
                    Span::styled(" Val Acc", AxonmlTheme::graph_label()),
                ]),
                Line::from(vec![
                    Span::styled("Latest: ", AxonmlTheme::muted()),
                    Span::styled(
                        format!(
                            "Train {:.1}%",
                            self.train_acc.last().map(|p| p.1).unwrap_or(0.0)
                        ),
                        AxonmlTheme::success(),
                    ),
                    Span::raw("  "),
                    Span::styled(
                        format!(
                            "Val {:.1}%",
                            self.val_acc.last().map(|p| p.1).unwrap_or(0.0)
                        ),
                        AxonmlTheme::success(),
                    ),
                ]),
            ],
            ChartType::LearningRate => vec![
                Line::from(vec![
                    Span::styled("\u{2588}\u{2588}", Style::default().fg(INFO)),
                    Span::styled(" Learning Rate (normalized)", AxonmlTheme::graph_label()),
                ]),
                Line::from(vec![
                    Span::styled("Current: ", AxonmlTheme::muted()),
                    Span::styled(
                        format!(
                            "{:.4}x initial",
                            self.learning_rate.last().map(|p| p.1).unwrap_or(1.0)
                        ),
                        AxonmlTheme::metric_value(),
                    ),
                ]),
            ],
        };

        let legend = Paragraph::new(legend_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(AxonmlTheme::border())
                .title(Span::styled(" Legend ", AxonmlTheme::header())),
        );

        frame.render_widget(legend, area);
    }
}

impl Default for GraphsView {
    fn default() -> Self {
        Self::new()
    }
}
