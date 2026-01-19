//! Data View - Display Dataset Structure and Statistics
//!
//! Shows dataset information including sample counts, class distributions,
//! feature dimensions, and data splits.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::path::Path;

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Row, Table},
    Frame,
};

use crate::theme::AxonmlTheme;

// =============================================================================
// Types
// =============================================================================

/// Class distribution information
#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub name: String,
    pub count: usize,
    pub percentage: f32,
}

/// Data split information
#[derive(Debug, Clone)]
pub struct SplitInfo {
    pub name: String,
    pub samples: usize,
    pub percentage: f32,
}

/// Feature information
#[derive(Debug, Clone)]
pub struct FeatureInfo {
    pub name: String,
    pub dtype: String,
    pub shape: String,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub mean: Option<f32>,
}

/// Dataset information
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub total_samples: usize,
    pub num_classes: usize,
    pub feature_dim: String,
    pub classes: Vec<ClassInfo>,
    pub splits: Vec<SplitInfo>,
    pub features: Vec<FeatureInfo>,
    pub file_path: String,
    pub file_size: u64,
}

// =============================================================================
// Data View
// =============================================================================

/// Dataset view state
pub struct DataView {
    /// Loaded dataset info
    pub dataset: Option<DatasetInfo>,

    /// Selected class index
    pub selected_class: usize,

    /// List state for class navigation
    pub list_state: ListState,

    /// Active panel (0=classes, 1=features)
    pub active_panel: usize,
}

impl DataView {
    /// Create a new data view with demo data
    pub fn new() -> Self {
        let mut view = Self {
            dataset: None,
            selected_class: 0,
            list_state: ListState::default(),
            active_panel: 0,
        };

        // Load demo data by default
        view.load_demo_data();
        view.list_state.select(Some(0));
        view
    }

    /// Load demo dataset for visualization
    pub fn load_demo_data(&mut self) {
        let dataset = DatasetInfo {
            name: "MNIST".to_string(),
            total_samples: 70_000,
            num_classes: 10,
            feature_dim: "[28, 28, 1]".to_string(),
            classes: vec![
                ClassInfo { name: "0".to_string(), count: 6903, percentage: 9.86 },
                ClassInfo { name: "1".to_string(), count: 7877, percentage: 11.25 },
                ClassInfo { name: "2".to_string(), count: 6990, percentage: 9.99 },
                ClassInfo { name: "3".to_string(), count: 7141, percentage: 10.20 },
                ClassInfo { name: "4".to_string(), count: 6824, percentage: 9.75 },
                ClassInfo { name: "5".to_string(), count: 6313, percentage: 9.02 },
                ClassInfo { name: "6".to_string(), count: 6876, percentage: 9.82 },
                ClassInfo { name: "7".to_string(), count: 7293, percentage: 10.42 },
                ClassInfo { name: "8".to_string(), count: 6825, percentage: 9.75 },
                ClassInfo { name: "9".to_string(), count: 6958, percentage: 9.94 },
            ],
            splits: vec![
                SplitInfo { name: "Train".to_string(), samples: 60_000, percentage: 85.71 },
                SplitInfo { name: "Test".to_string(), samples: 10_000, percentage: 14.29 },
            ],
            features: vec![
                FeatureInfo {
                    name: "pixels".to_string(),
                    dtype: "f32".to_string(),
                    shape: "[28, 28]".to_string(),
                    min: Some(0.0),
                    max: Some(255.0),
                    mean: Some(33.32),
                },
                FeatureInfo {
                    name: "label".to_string(),
                    dtype: "u8".to_string(),
                    shape: "[1]".to_string(),
                    min: Some(0.0),
                    max: Some(9.0),
                    mean: None,
                },
            ],
            file_path: "/data/mnist.npz".to_string(),
            file_size: 11_490_434,
        };

        self.dataset = Some(dataset);
    }

    /// Move selection up
    pub fn select_prev(&mut self) {
        if self.dataset.is_some() && self.selected_class > 0 {
            self.selected_class -= 1;
            self.list_state.select(Some(self.selected_class));
        }
    }

    /// Move selection down
    pub fn select_next(&mut self) {
        if let Some(dataset) = &self.dataset {
            if self.selected_class < dataset.classes.len() - 1 {
                self.selected_class += 1;
                self.list_state.select(Some(self.selected_class));
            }
        }
    }

    /// Load a dataset from a file path
    pub fn load_dataset(&mut self, _path: &Path) -> Result<(), String> {
        // For now, just load demo data
        // In real implementation, would parse actual dataset files
        self.load_demo_data();
        Ok(())
    }

    /// Switch active panel
    pub fn switch_panel(&mut self) {
        self.active_panel = (self.active_panel + 1) % 2;
    }

    /// Go to previous panel
    pub fn prev_panel(&mut self) {
        if self.active_panel > 0 {
            self.active_panel -= 1;
        } else {
            self.active_panel = 1; // Wrap around
        }
    }

    /// Go to next panel
    pub fn next_panel(&mut self) {
        self.active_panel = (self.active_panel + 1) % 2;
    }

    /// Render the data view
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        if let Some(dataset) = &self.dataset.clone() {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(6),  // Header
                    Constraint::Min(10),    // Main content
                    Constraint::Length(6),  // Splits panel
                ])
                .split(area);

            self.render_header(frame, chunks[0], dataset);
            self.render_main(frame, chunks[1], dataset);
            self.render_splits(frame, chunks[2], dataset);
        } else {
            self.render_empty(frame, area);
        }
    }

    fn render_header(&self, frame: &mut Frame, area: Rect, dataset: &DatasetInfo) {
        let header_text = vec![
            Line::from(vec![
                Span::styled("Dataset: ", AxonmlTheme::muted()),
                Span::styled(&dataset.name, AxonmlTheme::title()),
            ]),
            Line::from(vec![
                Span::styled("Path: ", AxonmlTheme::muted()),
                Span::styled(&dataset.file_path, AxonmlTheme::accent()),
            ]),
            Line::from(vec![
                Span::styled("Total Samples: ", AxonmlTheme::muted()),
                Span::styled(format_number(dataset.total_samples), AxonmlTheme::metric_value()),
                Span::raw("  "),
                Span::styled("Classes: ", AxonmlTheme::muted()),
                Span::styled(dataset.num_classes.to_string(), AxonmlTheme::metric_value()),
                Span::raw("  "),
                Span::styled("Feature Shape: ", AxonmlTheme::muted()),
                Span::styled(&dataset.feature_dim, AxonmlTheme::layer_shape()),
            ]),
            Line::from(vec![
                Span::styled("File Size: ", AxonmlTheme::muted()),
                Span::styled(format_size(dataset.file_size), AxonmlTheme::accent()),
            ]),
        ];

        let header = Paragraph::new(header_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Dataset Info ", AxonmlTheme::header())),
            );

        frame.render_widget(header, area);
    }

    fn render_main(&mut self, frame: &mut Frame, area: Rect, dataset: &DatasetInfo) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),  // Class distribution
                Constraint::Percentage(50),  // Features
            ])
            .split(area);

        self.render_class_distribution(frame, chunks[0], dataset);
        self.render_features(frame, chunks[1], dataset);
    }

    fn render_class_distribution(&mut self, frame: &mut Frame, area: Rect, dataset: &DatasetInfo) {
        let items: Vec<ListItem> = dataset
            .classes
            .iter()
            .enumerate()
            .map(|(i, class)| {
                let style = if i == self.selected_class && self.active_panel == 0 {
                    AxonmlTheme::selected()
                } else {
                    Style::default()
                };

                // Create a simple bar using unicode blocks
                let bar_width = (class.percentage * 0.3) as usize;
                let bar = "\u{2588}".repeat(bar_width.min(15));

                let content = Line::from(vec![
                    Span::styled(
                        format!("Class {:>2}: ", class.name),
                        AxonmlTheme::layer_type(),
                    ),
                    Span::styled(
                        format!("{:>6} ", format_number(class.count)),
                        AxonmlTheme::metric_value(),
                    ),
                    Span::styled(
                        format!("({:>5.1}%) ", class.percentage),
                        AxonmlTheme::muted(),
                    ),
                    Span::styled(bar, AxonmlTheme::graph_primary()),
                ]);

                ListItem::new(content).style(style)
            })
            .collect();

        let border_style = if self.active_panel == 0 {
            AxonmlTheme::border_focused()
        } else {
            AxonmlTheme::border()
        };

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(border_style)
                    .title(Span::styled(" Class Distribution ", AxonmlTheme::header())),
            )
            .highlight_style(AxonmlTheme::selected());

        frame.render_stateful_widget(list, area, &mut self.list_state);
    }

    fn render_features(&self, frame: &mut Frame, area: Rect, dataset: &DatasetInfo) {
        let rows: Vec<Row> = dataset
            .features
            .iter()
            .map(|feature| {
                Row::new(vec![
                    Span::styled(&feature.name, AxonmlTheme::layer_type()),
                    Span::styled(&feature.dtype, AxonmlTheme::accent()),
                    Span::styled(&feature.shape, AxonmlTheme::layer_shape()),
                    Span::styled(
                        feature.min.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::muted(),
                    ),
                    Span::styled(
                        feature.max.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::muted(),
                    ),
                    Span::styled(
                        feature.mean.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::metric_value(),
                    ),
                ])
            })
            .collect();

        let border_style = if self.active_panel == 1 {
            AxonmlTheme::border_focused()
        } else {
            AxonmlTheme::border()
        };

        let table = Table::new(
            rows,
            [
                Constraint::Length(10),  // Name
                Constraint::Length(6),   // Type
                Constraint::Length(10),  // Shape
                Constraint::Length(8),   // Min
                Constraint::Length(8),   // Max
                Constraint::Length(8),   // Mean
            ],
        )
        .header(
            Row::new(vec!["Name", "Type", "Shape", "Min", "Max", "Mean"])
                .style(AxonmlTheme::header())
                .bottom_margin(1),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(border_style)
                .title(Span::styled(" Features ", AxonmlTheme::header())),
        );

        frame.render_widget(table, area);
    }

    fn render_splits(&self, frame: &mut Frame, area: Rect, dataset: &DatasetInfo) {
        let split_text: Vec<Line> = dataset
            .splits
            .iter()
            .map(|split| {
                let bar_width = (split.percentage * 0.4) as usize;
                let bar = "\u{2588}".repeat(bar_width.min(40));

                Line::from(vec![
                    Span::styled(format!("{:<8}", split.name), AxonmlTheme::layer_type()),
                    Span::styled(
                        format!("{:>8} samples ", format_number(split.samples)),
                        AxonmlTheme::metric_value(),
                    ),
                    Span::styled(format!("({:>5.1}%) ", split.percentage), AxonmlTheme::muted()),
                    Span::styled(bar, AxonmlTheme::graph_secondary()),
                ])
            })
            .collect();

        let splits = Paragraph::new(split_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Data Splits ", AxonmlTheme::header())),
            );

        frame.render_widget(splits, area);
    }

    fn render_empty(&self, frame: &mut Frame, area: Rect) {
        let text = vec![
            Line::from(""),
            Line::from(Span::styled(
                "No dataset loaded",
                AxonmlTheme::muted(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Press 'o' to open a dataset file",
                AxonmlTheme::info(),
            )),
            Line::from(Span::styled(
                "Supported formats: .npz, .csv, .parquet",
                AxonmlTheme::muted(),
            )),
        ];

        let paragraph = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Dataset ", AxonmlTheme::header())),
            )
            .alignment(ratatui::layout::Alignment::Center);

        frame.render_widget(paragraph, area);
    }
}

impl Default for DataView {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
