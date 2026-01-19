//! Files View - File Browser for Model and Data Files
//!
//! Provides a tree-like file browser to navigate directories and
//! open model/data files for viewing and training.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::path::PathBuf;

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};

use crate::theme::AxonmlTheme;

// =============================================================================
// Types
// =============================================================================

/// File entry type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileType {
    Directory,
    Model,      // .axonml, .onnx, .pt, .safetensors
    Dataset,    // .npz, .csv, .parquet
    Config,     // .toml, .yaml, .json
    Other,
}

impl FileType {
    /// Get icon for file type
    #[allow(dead_code)]
    pub fn icon(&self) -> &'static str {
        match self {
            FileType::Directory => "\u{1F4C1}",  // Folder
            FileType::Model => "\u{1F9E0}",      // Brain (model)
            FileType::Dataset => "\u{1F4CA}",    // Chart (data)
            FileType::Config => "\u{2699}",      // Gear (config)
            FileType::Other => "\u{1F4C4}",      // Document
        }
    }

    /// Determine file type from extension
    #[allow(dead_code)]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "axonml" | "onnx" | "pt" | "pth" | "safetensors" | "h5" | "keras" => FileType::Model,
            "npz" | "npy" | "csv" | "parquet" | "arrow" | "tfrecord" => FileType::Dataset,
            "toml" | "yaml" | "yml" | "json" => FileType::Config,
            _ => FileType::Other,
        }
    }
}

/// File entry in the browser
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub name: String,
    pub path: PathBuf,
    pub file_type: FileType,
    pub size: Option<u64>,
    pub is_expanded: bool,
    pub depth: usize,
}

impl FileEntry {
    /// Create a new file entry
    #[allow(dead_code)]
    pub fn new(name: String, path: PathBuf, file_type: FileType, depth: usize) -> Self {
        Self {
            name,
            path,
            file_type,
            size: None,
            is_expanded: false,
            depth,
        }
    }
}

// =============================================================================
// Files View
// =============================================================================

/// File browser view state
pub struct FilesView {
    /// Current directory path
    pub current_dir: PathBuf,

    /// List of file entries
    pub entries: Vec<FileEntry>,

    /// Selected entry index
    pub selected: usize,

    /// List state for navigation
    pub list_state: ListState,

    /// Show hidden files
    pub show_hidden: bool,

    /// Filter pattern
    pub filter: Option<String>,

    /// Currently previewed file info
    pub preview: Option<String>,
}

impl FilesView {
    /// Create a new files view with demo data
    pub fn new() -> Self {
        let mut view = Self {
            current_dir: PathBuf::from("/home/user/ml-projects"),
            entries: Vec::new(),
            selected: 0,
            list_state: ListState::default(),
            show_hidden: false,
            filter: None,
            preview: None,
        };

        view.load_demo_entries();
        view.list_state.select(Some(0));
        view
    }

    /// Load demo file entries for visualization
    pub fn load_demo_entries(&mut self) {
        self.entries = vec![
            FileEntry {
                name: "..".to_string(),
                path: PathBuf::from("/home/user"),
                file_type: FileType::Directory,
                size: None,
                is_expanded: false,
                depth: 0,
            },
            FileEntry {
                name: "models".to_string(),
                path: PathBuf::from("/home/user/ml-projects/models"),
                file_type: FileType::Directory,
                size: None,
                is_expanded: true,
                depth: 0,
            },
            FileEntry {
                name: "mnist_classifier.axonml".to_string(),
                path: PathBuf::from("/home/user/ml-projects/models/mnist_classifier.axonml"),
                file_type: FileType::Model,
                size: Some(940_584),
                is_expanded: false,
                depth: 1,
            },
            FileEntry {
                name: "resnet50.onnx".to_string(),
                path: PathBuf::from("/home/user/ml-projects/models/resnet50.onnx"),
                file_type: FileType::Model,
                size: Some(97_800_000),
                is_expanded: false,
                depth: 1,
            },
            FileEntry {
                name: "bert_base.safetensors".to_string(),
                path: PathBuf::from("/home/user/ml-projects/models/bert_base.safetensors"),
                file_type: FileType::Model,
                size: Some(438_000_000),
                is_expanded: false,
                depth: 1,
            },
            FileEntry {
                name: "datasets".to_string(),
                path: PathBuf::from("/home/user/ml-projects/datasets"),
                file_type: FileType::Directory,
                size: None,
                is_expanded: true,
                depth: 0,
            },
            FileEntry {
                name: "mnist.npz".to_string(),
                path: PathBuf::from("/home/user/ml-projects/datasets/mnist.npz"),
                file_type: FileType::Dataset,
                size: Some(11_490_434),
                is_expanded: false,
                depth: 1,
            },
            FileEntry {
                name: "cifar10.npz".to_string(),
                path: PathBuf::from("/home/user/ml-projects/datasets/cifar10.npz"),
                file_type: FileType::Dataset,
                size: Some(170_498_071),
                is_expanded: false,
                depth: 1,
            },
            FileEntry {
                name: "imagenet_labels.csv".to_string(),
                path: PathBuf::from("/home/user/ml-projects/datasets/imagenet_labels.csv"),
                file_type: FileType::Dataset,
                size: Some(21_384),
                is_expanded: false,
                depth: 1,
            },
            FileEntry {
                name: "configs".to_string(),
                path: PathBuf::from("/home/user/ml-projects/configs"),
                file_type: FileType::Directory,
                size: None,
                is_expanded: false,
                depth: 0,
            },
            FileEntry {
                name: "train_config.toml".to_string(),
                path: PathBuf::from("/home/user/ml-projects/train_config.toml"),
                file_type: FileType::Config,
                size: Some(1_024),
                is_expanded: false,
                depth: 0,
            },
            FileEntry {
                name: "README.md".to_string(),
                path: PathBuf::from("/home/user/ml-projects/README.md"),
                file_type: FileType::Other,
                size: Some(2_048),
                is_expanded: false,
                depth: 0,
            },
        ];
    }

    /// Move selection up
    pub fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.list_state.select(Some(self.selected));
            self.update_preview();
        }
    }

    /// Move selection down
    pub fn select_next(&mut self) {
        if self.selected < self.entries.len() - 1 {
            self.selected += 1;
            self.list_state.select(Some(self.selected));
            self.update_preview();
        }
    }

    /// Toggle directory expansion or open file
    pub fn toggle_or_open(&mut self) {
        if let Some(entry) = self.entries.get_mut(self.selected) {
            if entry.file_type == FileType::Directory {
                entry.is_expanded = !entry.is_expanded;
                // In real implementation, would reload children
            }
            // In real implementation, would open file for viewing
        }
    }

    /// Go to parent directory
    pub fn go_parent(&mut self) {
        if let Some(parent) = self.current_dir.parent() {
            self.current_dir = parent.to_path_buf();
            // In real implementation, would reload entries
        }
    }

    /// Go up to parent directory (alias for go_parent)
    pub fn go_up(&mut self) {
        self.go_parent();
    }

    /// Go to home directory
    pub fn go_home(&mut self) {
        if let Some(home) = dirs::home_dir() {
            self.current_dir = home;
            // In real implementation, would reload entries
        }
    }

    /// Open selected file and return its path if it's a file (not directory)
    pub fn open_selected(&mut self) -> Option<PathBuf> {
        if let Some(entry) = self.entries.get(self.selected) {
            if entry.file_type == FileType::Directory {
                // Toggle expansion for directories
                self.toggle_or_open();
            } else {
                return Some(entry.path.clone());
            }
        }
        None
    }

    /// Toggle hidden files
    pub fn toggle_hidden(&mut self) {
        self.show_hidden = !self.show_hidden;
        // In real implementation, would reload entries
    }

    /// Update preview for selected file
    fn update_preview(&mut self) {
        if let Some(entry) = self.entries.get(self.selected) {
            self.preview = Some(format!(
                "Path: {}\nType: {:?}\nSize: {}",
                entry.path.display(),
                entry.file_type,
                entry.size.map(format_size).unwrap_or_else(|| "-".to_string())
            ));
        }
    }

    /// Render the files view
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60),  // File list
                Constraint::Percentage(40),  // Preview
            ])
            .split(area);

        self.render_file_list(frame, chunks[0]);
        self.render_preview(frame, chunks[1]);
    }

    fn render_file_list(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Path bar
                Constraint::Min(10),    // File list
            ])
            .split(area);

        // Path bar
        let path_text = Line::from(vec![
            Span::styled("Path: ", AxonmlTheme::muted()),
            Span::styled(self.current_dir.display().to_string(), AxonmlTheme::accent()),
        ]);

        let path_bar = Paragraph::new(path_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Location ", AxonmlTheme::header())),
            );

        frame.render_widget(path_bar, chunks[0]);

        // File list
        let items: Vec<ListItem> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let style = if i == self.selected {
                    AxonmlTheme::selected()
                } else {
                    Style::default()
                };

                // Indentation for tree structure
                let indent = "  ".repeat(entry.depth);

                // Expansion indicator for directories
                let expand_char = if entry.file_type == FileType::Directory {
                    if entry.is_expanded { "\u{25BC} " } else { "\u{25B6} " }
                } else {
                    "  "
                };

                // File type styling
                let name_style = match entry.file_type {
                    FileType::Directory => AxonmlTheme::layer_type(),
                    FileType::Model => AxonmlTheme::success(),
                    FileType::Dataset => AxonmlTheme::info(),
                    FileType::Config => AxonmlTheme::warning(),
                    FileType::Other => AxonmlTheme::muted(),
                };

                let size_str = entry
                    .size
                    .map(|s| format!("{:>10}", format_size(s)))
                    .unwrap_or_else(|| "         -".to_string());

                let content = Line::from(vec![
                    Span::raw(indent),
                    Span::styled(expand_char, AxonmlTheme::muted()),
                    Span::styled(&entry.name, name_style),
                    Span::styled(format!("  {}", size_str), AxonmlTheme::muted()),
                ]);

                ListItem::new(content).style(style)
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border_focused())
                    .title(Span::styled(" Files ", AxonmlTheme::header())),
            )
            .highlight_style(AxonmlTheme::selected());

        frame.render_stateful_widget(list, chunks[1], &mut self.list_state);
    }

    fn render_preview(&self, frame: &mut Frame, area: Rect) {
        let selected_entry = self.entries.get(self.selected);

        let preview_text = if let Some(entry) = selected_entry {
            let type_info = match entry.file_type {
                FileType::Directory => "Directory - Press Enter to expand/collapse",
                FileType::Model => "Neural Network Model\nSupported: Axonml, ONNX, SafeTensors\nPress Enter to load in Model view",
                FileType::Dataset => "Dataset File\nSupported: NPZ, CSV, Parquet\nPress Enter to load in Data view",
                FileType::Config => "Configuration File\nPress Enter to edit",
                FileType::Other => "Unknown file type",
            };

            vec![
                Line::from(vec![
                    Span::styled("Name: ", AxonmlTheme::muted()),
                    Span::styled(&entry.name, AxonmlTheme::title()),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Type: ", AxonmlTheme::muted()),
                    Span::styled(format!("{:?}", entry.file_type), AxonmlTheme::accent()),
                ]),
                Line::from(vec![
                    Span::styled("Size: ", AxonmlTheme::muted()),
                    Span::styled(
                        entry.size.map(format_size).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::metric_value(),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Path: ", AxonmlTheme::muted()),
                    Span::styled(entry.path.display().to_string(), AxonmlTheme::layer_shape()),
                ]),
                Line::from(""),
                Line::from(Span::styled(type_info, AxonmlTheme::info())),
            ]
        } else {
            vec![Line::from(Span::styled("No file selected", AxonmlTheme::muted()))]
        };

        let preview = Paragraph::new(preview_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Preview ", AxonmlTheme::header())),
            )
            .wrap(ratatui::widgets::Wrap { trim: true });

        frame.render_widget(preview, area);
    }
}

impl Default for FilesView {
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
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
