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
use axonml_serialize::load_state_dict;

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

    /// Parse model file using axonml-serialize
    fn parse_model_file(&self, path: &Path) -> Result<ModelInfo, String> {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model");

        // Get file size
        let file_size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);

        // Detect format from extension
        let format = path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext.to_lowercase().as_str() {
                "axonml" => "Axonml",
                "safetensors" => "SafeTensors",
                "json" => "JSON",
                "pt" | "pth" => "PyTorch",
                "onnx" => "ONNX",
                _ => "Unknown",
            })
            .unwrap_or("Unknown")
            .to_string();

        // Load the actual state dict
        let state_dict = load_state_dict(path)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        // Group parameters by layer prefix and extract layer info
        let mut layer_map: std::collections::BTreeMap<String, Vec<(String, Vec<usize>, usize, bool)>> =
            std::collections::BTreeMap::new();

        for (param_name, entry) in state_dict.entries() {
            let shape = entry.data.shape.clone();
            let num_params: usize = shape.iter().product();

            // Extract layer name from parameter name (e.g., "layer1.conv.weight" -> "layer1.conv")
            let layer_name = if let Some(idx) = param_name.rfind('.') {
                param_name[..idx].to_string()
            } else {
                param_name.clone()
            };

            layer_map
                .entry(layer_name)
                .or_default()
                .push((param_name.clone(), shape, num_params, entry.requires_grad));
        }

        // Create LayerInfo for each unique layer
        let mut layers = Vec::new();
        for (layer_name, params) in layer_map {
            let layer_num_params: usize = params.iter().map(|(_, _, p, _)| *p).sum();
            let trainable = params.iter().any(|(_, _, _, t)| *t);

            // Infer layer type from parameter names
            let layer_type = infer_layer_type(&params);

            // Get shape from weight parameter if available
            let (input_shape, output_shape) = infer_shapes(&params);

            layers.push(LayerInfo {
                name: layer_name,
                layer_type,
                input_shape,
                output_shape,
                params: layer_num_params,
                trainable,
            });
        }

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
            file_size,
            format,
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

/// Infer layer type from parameter names
fn infer_layer_type(params: &[(String, Vec<usize>, usize, bool)]) -> String {
    for (name, shape, _, _) in params {
        if name.ends_with(".weight") {
            let dims = shape.len();
            if dims == 4 {
                return "Conv2d".to_string();
            } else if dims == 2 {
                return "Linear".to_string();
            } else if dims == 1 {
                return "BatchNorm".to_string();
            }
        }
        if name.ends_with(".gamma") || name.ends_with(".beta") {
            return "LayerNorm".to_string();
        }
        if name.ends_with(".embedding") {
            return "Embedding".to_string();
        }
    }
    "Unknown".to_string()
}

/// Infer input and output shapes from parameters
fn infer_shapes(params: &[(String, Vec<usize>, usize, bool)]) -> (String, String) {
    for (name, shape, _, _) in params {
        if name.ends_with(".weight") && shape.len() >= 2 {
            // For Linear: [out_features, in_features]
            // For Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
            if shape.len() == 2 {
                return (
                    format!("[batch, {}]", shape[1]),
                    format!("[batch, {}]", shape[0]),
                );
            } else if shape.len() == 4 {
                return (
                    format!("[batch, {}, H, W]", shape[1]),
                    format!("[batch, {}, H', W']", shape[0]),
                );
            }
        }
    }
    ("-".to_string(), "-".to_string())
}
