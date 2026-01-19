//! Model View - Display Neural Network Architecture
//!
//! Shows model layers, parameters, shapes, and structure.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::path::Path;

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
    Frame,
};

use crate::theme::AxonmlTheme;

// =============================================================================
// Types
// =============================================================================

/// Layer information for display
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub input_shape: String,
    pub output_shape: String,
    pub params: usize,
    pub trainable: bool,
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub layers: Vec<LayerInfo>,
    pub total_params: usize,
    pub trainable_params: usize,
    pub file_size: u64,
    pub format: String,
}

// =============================================================================
// Model View
// =============================================================================

/// Model architecture view state
pub struct ModelView {
    /// Loaded model info
    pub model: Option<ModelInfo>,

    /// Selected layer index
    pub selected_layer: usize,

    /// List state for layer navigation
    pub list_state: ListState,

    /// Scroll state
    pub scroll_state: ScrollbarState,

    /// Show detailed view
    pub show_details: bool,
}

impl ModelView {
    /// Create a new model view
    pub fn new() -> Self {
        let mut list_state = ListState::default();
        list_state.select(Some(0));

        Self {
            model: None,
            selected_layer: 0,
            list_state,
            scroll_state: ScrollbarState::default(),
            show_details: false,
        }
    }

    /// Load a model from file
    pub fn load_model(&mut self, path: &Path) -> Result<(), String> {
        // For now, create a demo model structure
        // In real implementation, this would parse actual model files
        let model = self.parse_model_file(path)?;
        let layer_count = model.layers.len();
        self.model = Some(model);
        self.selected_layer = 0;
        self.list_state.select(Some(0));
        self.scroll_state = ScrollbarState::default().content_length(layer_count);
        Ok(())
    }

    /// Parse model file (placeholder - would use axonml-serialize in real impl)
    fn parse_model_file(&self, path: &Path) -> Result<ModelInfo, String> {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model");

        // Demo model structure for visualization
        let layers = vec![
            LayerInfo {
                name: "input".to_string(),
                layer_type: "Input".to_string(),
                input_shape: "[batch, 784]".to_string(),
                output_shape: "[batch, 784]".to_string(),
                params: 0,
                trainable: false,
            },
            LayerInfo {
                name: "fc1".to_string(),
                layer_type: "Linear".to_string(),
                input_shape: "[batch, 784]".to_string(),
                output_shape: "[batch, 256]".to_string(),
                params: 200_960,
                trainable: true,
            },
            LayerInfo {
                name: "relu1".to_string(),
                layer_type: "ReLU".to_string(),
                input_shape: "[batch, 256]".to_string(),
                output_shape: "[batch, 256]".to_string(),
                params: 0,
                trainable: false,
            },
            LayerInfo {
                name: "dropout1".to_string(),
                layer_type: "Dropout(0.2)".to_string(),
                input_shape: "[batch, 256]".to_string(),
                output_shape: "[batch, 256]".to_string(),
                params: 0,
                trainable: false,
            },
            LayerInfo {
                name: "fc2".to_string(),
                layer_type: "Linear".to_string(),
                input_shape: "[batch, 256]".to_string(),
                output_shape: "[batch, 128]".to_string(),
                params: 32_896,
                trainable: true,
            },
            LayerInfo {
                name: "relu2".to_string(),
                layer_type: "ReLU".to_string(),
                input_shape: "[batch, 128]".to_string(),
                output_shape: "[batch, 128]".to_string(),
                params: 0,
                trainable: false,
            },
            LayerInfo {
                name: "fc3".to_string(),
                layer_type: "Linear".to_string(),
                input_shape: "[batch, 128]".to_string(),
                output_shape: "[batch, 10]".to_string(),
                params: 1_290,
                trainable: true,
            },
            LayerInfo {
                name: "softmax".to_string(),
                layer_type: "Softmax".to_string(),
                input_shape: "[batch, 10]".to_string(),
                output_shape: "[batch, 10]".to_string(),
                params: 0,
                trainable: false,
            },
        ];

        let total_params: usize = layers.iter().map(|l| l.params).sum();
        let trainable_params: usize = layers
            .iter()
            .filter(|l| l.trainable)
            .map(|l| l.params)
            .sum();

        Ok(ModelInfo {
            name: file_name.to_string(),
            layers,
            total_params,
            trainable_params,
            file_size: 940_584,
            format: "Axonml".to_string(),
        })
    }

    /// Move selection up
    pub fn select_prev(&mut self) {
        if self.model.is_some() && self.selected_layer > 0 {
            self.selected_layer -= 1;
            self.list_state.select(Some(self.selected_layer));
            self.scroll_state = self.scroll_state.position(self.selected_layer);
        }
    }

    /// Move selection down
    pub fn select_next(&mut self) {
        if let Some(model) = &self.model {
            if self.selected_layer < model.layers.len() - 1 {
                self.selected_layer += 1;
                self.list_state.select(Some(self.selected_layer));
                self.scroll_state = self.scroll_state.position(self.selected_layer);
            }
        }
    }

    /// Toggle detailed view
    pub fn toggle_details(&mut self) {
        self.show_details = !self.show_details;
    }

    /// Render the model view
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        if let Some(model) = self.model.clone() {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(5),  // Header
                    Constraint::Min(10),    // Layers list
                    Constraint::Length(8),  // Details panel
                ])
                .split(area);

            self.render_header(frame, chunks[0], &model);
            self.render_layers(frame, chunks[1], &model);
            self.render_details(frame, chunks[2], &model);
        } else {
            self.render_empty(frame, area);
        }
    }

    fn render_header(&self, frame: &mut Frame, area: Rect, model: &ModelInfo) {
        let header_text = vec![
            Line::from(vec![
                Span::styled("Model: ", AxonmlTheme::muted()),
                Span::styled(&model.name, AxonmlTheme::title()),
            ]),
            Line::from(vec![
                Span::styled("Format: ", AxonmlTheme::muted()),
                Span::styled(&model.format, AxonmlTheme::accent()),
                Span::raw("  "),
                Span::styled("Size: ", AxonmlTheme::muted()),
                Span::styled(format_size(model.file_size), AxonmlTheme::accent()),
            ]),
            Line::from(vec![
                Span::styled("Total Params: ", AxonmlTheme::muted()),
                Span::styled(format_number(model.total_params), AxonmlTheme::metric_value()),
                Span::raw("  "),
                Span::styled("Trainable: ", AxonmlTheme::muted()),
                Span::styled(format_number(model.trainable_params), AxonmlTheme::success()),
            ]),
        ];

        let header = Paragraph::new(header_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Model Info ", AxonmlTheme::header())),
            );

        frame.render_widget(header, area);
    }

    fn render_layers(&mut self, frame: &mut Frame, area: Rect, model: &ModelInfo) {
        let items: Vec<ListItem> = model
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let style = if i == self.selected_layer {
                    AxonmlTheme::selected()
                } else {
                    Style::default()
                };

                let content = Line::from(vec![
                    Span::styled(
                        format!("{:>2}. ", i + 1),
                        AxonmlTheme::muted(),
                    ),
                    Span::styled(
                        format!("{:<12}", layer.name),
                        AxonmlTheme::layer_type(),
                    ),
                    Span::styled(
                        format!("{:<15}", layer.layer_type),
                        AxonmlTheme::accent(),
                    ),
                    Span::styled(
                        format!("{:>15}", layer.output_shape),
                        AxonmlTheme::layer_shape(),
                    ),
                    Span::styled(
                        format!("{:>12}", format_number(layer.params)),
                        if layer.trainable {
                            AxonmlTheme::success()
                        } else {
                            AxonmlTheme::muted()
                        },
                    ),
                ]);

                ListItem::new(content).style(style)
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border_focused())
                    .title(Span::styled(" Layers ", AxonmlTheme::header())),
            )
            .highlight_style(AxonmlTheme::selected());

        frame.render_stateful_widget(list, area, &mut self.list_state);

        // Render scrollbar
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("▲"))
            .end_symbol(Some("▼"));

        frame.render_stateful_widget(
            scrollbar,
            area.inner(ratatui::layout::Margin { vertical: 1, horizontal: 0 }),
            &mut self.scroll_state,
        );
    }

    fn render_details(&self, frame: &mut Frame, area: Rect, model: &ModelInfo) {
        let layer = &model.layers[self.selected_layer];

        let details = vec![
            Line::from(vec![
                Span::styled("Layer: ", AxonmlTheme::muted()),
                Span::styled(&layer.name, AxonmlTheme::title()),
                Span::styled(" (", AxonmlTheme::muted()),
                Span::styled(&layer.layer_type, AxonmlTheme::accent()),
                Span::styled(")", AxonmlTheme::muted()),
            ]),
            Line::from(vec![
                Span::styled("Input:  ", AxonmlTheme::muted()),
                Span::styled(&layer.input_shape, AxonmlTheme::layer_shape()),
            ]),
            Line::from(vec![
                Span::styled("Output: ", AxonmlTheme::muted()),
                Span::styled(&layer.output_shape, AxonmlTheme::layer_shape()),
            ]),
            Line::from(vec![
                Span::styled("Params: ", AxonmlTheme::muted()),
                Span::styled(format_number(layer.params), AxonmlTheme::metric_value()),
                Span::raw("  "),
                Span::styled("Trainable: ", AxonmlTheme::muted()),
                Span::styled(
                    if layer.trainable { "Yes" } else { "No" },
                    if layer.trainable {
                        AxonmlTheme::success()
                    } else {
                        AxonmlTheme::muted()
                    },
                ),
            ]),
        ];

        let details_widget = Paragraph::new(details)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Layer Details ", AxonmlTheme::header())),
            );

        frame.render_widget(details_widget, area);
    }

    fn render_empty(&self, frame: &mut Frame, area: Rect) {
        let text = vec![
            Line::from(""),
            Line::from(Span::styled(
                "No model loaded",
                AxonmlTheme::muted(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Press 'o' to open a model file",
                AxonmlTheme::info(),
            )),
            Line::from(Span::styled(
                "or use: axonml tui --model <path>",
                AxonmlTheme::muted(),
            )),
        ];

        let paragraph = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Model Architecture ", AxonmlTheme::header())),
            )
            .alignment(ratatui::layout::Alignment::Center);

        frame.render_widget(paragraph, area);
    }
}

impl Default for ModelView {
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
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
