//! Training View - Monitor Training Progress
//!
//! Displays real-time training metrics including epochs, loss, accuracy,
//! learning rate, and includes a sparkline for loss history.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::path::Path;

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Frame,
};

use crate::theme::AxonmlTheme;

// =============================================================================
// Types
// =============================================================================

/// Training status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingStatus {
    Idle,
    Running,
    Paused,
    Completed,
    Failed,
}

impl TrainingStatus {
    fn as_str(&self) -> &'static str {
        match self {
            TrainingStatus::Idle => "Idle",
            TrainingStatus::Running => "Running",
            TrainingStatus::Paused => "Paused",
            TrainingStatus::Completed => "Completed",
            TrainingStatus::Failed => "Failed",
        }
    }

    fn style(&self) -> Style {
        match self {
            TrainingStatus::Idle => AxonmlTheme::muted(),
            TrainingStatus::Running => AxonmlTheme::success(),
            TrainingStatus::Paused => AxonmlTheme::warning(),
            TrainingStatus::Completed => AxonmlTheme::info(),
            TrainingStatus::Failed => AxonmlTheme::error(),
        }
    }
}

/// Training metrics for a single epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    pub train_acc: Option<f32>,
    pub val_acc: Option<f32>,
    pub learning_rate: f64,
    pub duration_secs: f32,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub total_epochs: usize,
    pub batch_size: usize,
    pub optimizer: String,
    pub initial_lr: f64,
    pub model_name: String,
}

/// Training session information
#[derive(Debug, Clone)]
pub struct TrainingSession {
    pub config: TrainingConfig,
    pub current_epoch: usize,
    pub current_batch: usize,
    pub total_batches: usize,
    pub status: TrainingStatus,
    pub metrics_history: Vec<EpochMetrics>,
    pub loss_history: Vec<u64>,  // Scaled for sparkline
    pub best_val_loss: Option<f32>,
    pub best_epoch: Option<usize>,
    pub elapsed_secs: f64,
    pub eta_secs: Option<f64>,
}

// =============================================================================
// Training View
// =============================================================================

/// Training progress view state
pub struct TrainingView {
    /// Current training session
    pub session: Option<TrainingSession>,

    /// Show detailed metrics
    pub show_details: bool,
}

impl TrainingView {
    /// Create a new training view with demo data
    pub fn new() -> Self {
        let mut view = Self {
            session: None,
            show_details: false,
        };

        // Load demo data
        view.load_demo_session();
        view
    }

    /// Load a demo training session for visualization
    pub fn load_demo_session(&mut self) {
        let metrics_history = vec![
            EpochMetrics { epoch: 1, train_loss: 2.312, val_loss: Some(2.298), train_acc: Some(0.112), val_acc: Some(0.118), learning_rate: 0.001, duration_secs: 45.2 },
            EpochMetrics { epoch: 2, train_loss: 1.845, val_loss: Some(1.756), train_acc: Some(0.342), val_acc: Some(0.358), learning_rate: 0.001, duration_secs: 44.8 },
            EpochMetrics { epoch: 3, train_loss: 1.234, val_loss: Some(1.189), train_acc: Some(0.567), val_acc: Some(0.582), learning_rate: 0.001, duration_secs: 45.1 },
            EpochMetrics { epoch: 4, train_loss: 0.856, val_loss: Some(0.823), train_acc: Some(0.712), val_acc: Some(0.724), learning_rate: 0.001, duration_secs: 44.9 },
            EpochMetrics { epoch: 5, train_loss: 0.612, val_loss: Some(0.598), train_acc: Some(0.798), val_acc: Some(0.805), learning_rate: 0.0005, duration_secs: 45.3 },
            EpochMetrics { epoch: 6, train_loss: 0.478, val_loss: Some(0.489), train_acc: Some(0.845), val_acc: Some(0.842), learning_rate: 0.0005, duration_secs: 45.0 },
            EpochMetrics { epoch: 7, train_loss: 0.389, val_loss: Some(0.412), train_acc: Some(0.878), val_acc: Some(0.869), learning_rate: 0.0005, duration_secs: 44.7 },
            EpochMetrics { epoch: 8, train_loss: 0.321, val_loss: Some(0.358), train_acc: Some(0.902), val_acc: Some(0.891), learning_rate: 0.00025, duration_secs: 45.2 },
        ];

        // Scale loss values for sparkline (0-100 range)
        let loss_history: Vec<u64> = metrics_history
            .iter()
            .map(|m| ((2.5 - m.train_loss) / 2.5 * 100.0).max(0.0) as u64)
            .collect();

        let session = TrainingSession {
            config: TrainingConfig {
                total_epochs: 20,
                batch_size: 64,
                optimizer: "Adam".to_string(),
                initial_lr: 0.001,
                model_name: "mnist_classifier".to_string(),
            },
            current_epoch: 8,
            current_batch: 720,
            total_batches: 938,
            status: TrainingStatus::Running,
            metrics_history,
            loss_history,
            best_val_loss: Some(0.358),
            best_epoch: Some(8),
            elapsed_secs: 361.2,
            eta_secs: Some(540.0),
        };

        self.session = Some(session);
    }

    /// Toggle detailed view
    pub fn toggle_details(&mut self) {
        self.show_details = !self.show_details;
    }

    /// Pause/resume training
    pub fn toggle_pause(&mut self) {
        if let Some(session) = &mut self.session {
            session.status = match session.status {
                TrainingStatus::Running => TrainingStatus::Paused,
                TrainingStatus::Paused => TrainingStatus::Running,
                other => other,
            };
        }
    }

    /// Watch a training log file
    pub fn watch_log(&mut self, path: &Path) -> Result<(), String> {
        // Try to parse the training log file
        match self.parse_training_log(path) {
            Ok(session) => {
                self.session = Some(session);
                Ok(())
            }
            Err(e) => {
                // Log file not found or invalid - fall back to demo
                eprintln!("Warning: Could not parse training log: {}", e);
                self.load_demo_session();
                Err(e)
            }
        }
    }

    /// Parse a training log file to extract metrics
    fn parse_training_log(&self, path: &Path) -> Result<TrainingSession, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read log file: {}", e))?;

        let mut metrics_history = Vec::new();
        let mut config = TrainingConfig {
            total_epochs: 10,
            batch_size: 32,
            optimizer: "Adam".to_string(),
            initial_lr: 0.001,
            model_name: "model".to_string(),
        };

        // Parse JSON log format (one JSON object per line)
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            // Try to parse as JSON
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                // Extract epoch info
                if let Some(epoch) = json.get("epoch").and_then(|v| v.as_u64()) {
                    let train_loss = json.get("train_loss")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;
                    let val_loss = json.get("val_loss")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32);
                    let train_acc = json.get("train_acc")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32);
                    let val_acc = json.get("val_acc")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32);
                    let learning_rate = json.get("learning_rate")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.001);
                    let duration = json.get("duration_secs")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;

                    metrics_history.push(EpochMetrics {
                        epoch: epoch as usize,
                        train_loss,
                        val_loss,
                        train_acc,
                        val_acc,
                        learning_rate,
                        duration_secs: duration,
                    });
                }

                // Extract config info
                if let Some(total) = json.get("total_epochs").and_then(|v| v.as_u64()) {
                    config.total_epochs = total as usize;
                }
                if let Some(bs) = json.get("batch_size").and_then(|v| v.as_u64()) {
                    config.batch_size = bs as usize;
                }
                if let Some(opt) = json.get("optimizer").and_then(|v| v.as_str()) {
                    config.optimizer = opt.to_string();
                }
                if let Some(name) = json.get("model_name").and_then(|v| v.as_str()) {
                    config.model_name = name.to_string();
                }
            }
        }

        if metrics_history.is_empty() {
            return Err("No training metrics found in log file".to_string());
        }

        // Calculate derived values
        let current_epoch = metrics_history.last().map(|m| m.epoch).unwrap_or(1);
        let loss_history: Vec<u64> = metrics_history
            .iter()
            .map(|m| {
                let max_loss = 3.0f32;
                ((max_loss - m.train_loss.min(max_loss)) / max_loss * 100.0) as u64
            })
            .collect();

        let (best_val_loss, best_epoch) = metrics_history
            .iter()
            .filter_map(|m| m.val_loss.map(|v| (v, m.epoch)))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0.0, 1));

        let total_duration: f32 = metrics_history.iter().map(|m| m.duration_secs).sum();
        let avg_epoch_time = total_duration / metrics_history.len() as f32;
        let total_epochs = config.total_epochs;
        let remaining_epochs = total_epochs.saturating_sub(current_epoch);
        let eta = (remaining_epochs as f32 * avg_epoch_time) as f64;

        Ok(TrainingSession {
            config,
            current_epoch,
            current_batch: 0,
            total_batches: 100,
            status: if current_epoch >= total_epochs {
                TrainingStatus::Completed
            } else {
                TrainingStatus::Idle
            },
            metrics_history,
            loss_history,
            best_val_loss: Some(best_val_loss),
            best_epoch: Some(best_epoch),
            elapsed_secs: total_duration as f64,
            eta_secs: if eta > 0.0 { Some(eta) } else { None },
        })
    }

    /// Scroll up in the metrics history
    pub fn scroll_up(&mut self) {
        // Reserved for scrolling through history
    }

    /// Scroll down in the metrics history
    pub fn scroll_down(&mut self) {
        // Reserved for scrolling through history
    }

    /// Refresh training data
    pub fn refresh(&mut self) {
        // Reload demo data for now
        self.load_demo_session();
    }

    /// Tick update for real-time animation
    pub fn tick(&mut self) {
        // In real implementation, would update metrics from training process
        // For demo, we simulate small progress updates
        if let Some(session) = &mut self.session {
            if session.status == TrainingStatus::Running {
                // Simulate batch progress
                if session.current_batch < session.total_batches {
                    session.current_batch += 1;
                } else {
                    // Next epoch
                    session.current_batch = 0;
                    if session.current_epoch < session.config.total_epochs {
                        session.current_epoch += 1;
                    }
                }
                session.elapsed_secs += 0.1;
            }
        }
    }

    /// Render the training view
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        if let Some(session) = &self.session.clone() {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(7),  // Header with status
                    Constraint::Length(3),  // Epoch progress bar
                    Constraint::Length(3),  // Batch progress bar
                    Constraint::Min(8),     // Metrics and sparkline
                    Constraint::Length(10), // Epoch history
                ])
                .split(area);

            self.render_header(frame, chunks[0], session);
            self.render_epoch_progress(frame, chunks[1], session);
            self.render_batch_progress(frame, chunks[2], session);
            self.render_metrics(frame, chunks[3], session);
            self.render_history(frame, chunks[4], session);
        } else {
            self.render_empty(frame, area);
        }
    }

    fn render_header(&self, frame: &mut Frame, area: Rect, session: &TrainingSession) {
        let status_style = session.status.style();

        let header_text = vec![
            Line::from(vec![
                Span::styled("Model: ", AxonmlTheme::muted()),
                Span::styled(&session.config.model_name, AxonmlTheme::title()),
                Span::raw("  "),
                Span::styled("Status: ", AxonmlTheme::muted()),
                Span::styled(session.status.as_str(), status_style),
            ]),
            Line::from(vec![
                Span::styled("Optimizer: ", AxonmlTheme::muted()),
                Span::styled(&session.config.optimizer, AxonmlTheme::accent()),
                Span::raw("  "),
                Span::styled("Batch Size: ", AxonmlTheme::muted()),
                Span::styled(session.config.batch_size.to_string(), AxonmlTheme::metric_value()),
                Span::raw("  "),
                Span::styled("LR: ", AxonmlTheme::muted()),
                Span::styled(
                    format!("{:.6}", session.metrics_history.last().map(|m| m.learning_rate).unwrap_or(session.config.initial_lr)),
                    AxonmlTheme::metric_value(),
                ),
            ]),
            Line::from(vec![
                Span::styled("Elapsed: ", AxonmlTheme::muted()),
                Span::styled(format_duration(session.elapsed_secs), AxonmlTheme::accent()),
                Span::raw("  "),
                Span::styled("ETA: ", AxonmlTheme::muted()),
                Span::styled(
                    session.eta_secs.map(format_duration).unwrap_or_else(|| "--:--".to_string()),
                    AxonmlTheme::accent(),
                ),
            ]),
            Line::from(vec![
                Span::styled("Best Val Loss: ", AxonmlTheme::muted()),
                Span::styled(
                    session.best_val_loss.map(|v| format!("{:.4}", v)).unwrap_or_else(|| "-".to_string()),
                    AxonmlTheme::success(),
                ),
                Span::raw("  "),
                Span::styled("@ Epoch: ", AxonmlTheme::muted()),
                Span::styled(
                    session.best_epoch.map(|e| e.to_string()).unwrap_or_else(|| "-".to_string()),
                    AxonmlTheme::success(),
                ),
            ]),
        ];

        let header = Paragraph::new(header_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Training Session ", AxonmlTheme::header())),
            );

        frame.render_widget(header, area);
    }

    fn render_epoch_progress(&self, frame: &mut Frame, area: Rect, session: &TrainingSession) {
        let progress = session.current_epoch as f64 / session.config.total_epochs as f64;

        let gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(
                        format!(" Epoch {}/{} ", session.current_epoch, session.config.total_epochs),
                        AxonmlTheme::epoch(),
                    )),
            )
            .gauge_style(AxonmlTheme::graph_primary())
            .ratio(progress)
            .label(format!("{:.1}%", progress * 100.0));

        frame.render_widget(gauge, area);
    }

    fn render_batch_progress(&self, frame: &mut Frame, area: Rect, session: &TrainingSession) {
        let progress = session.current_batch as f64 / session.total_batches as f64;

        let gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(
                        format!(" Batch {}/{} ", session.current_batch, session.total_batches),
                        AxonmlTheme::muted(),
                    )),
            )
            .gauge_style(AxonmlTheme::graph_secondary())
            .ratio(progress)
            .label(format!("{:.1}%", progress * 100.0));

        frame.render_widget(gauge, area);
    }

    fn render_metrics(&self, frame: &mut Frame, area: Rect, session: &TrainingSession) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),  // Current metrics
                Constraint::Percentage(50),  // Loss sparkline
            ])
            .split(area);

        // Current metrics
        let latest = session.metrics_history.last();
        let metrics_text = if let Some(m) = latest {
            let loss_style = if m.train_loss < 0.5 {
                AxonmlTheme::loss_good()
            } else if m.train_loss < 1.0 {
                AxonmlTheme::loss_neutral()
            } else {
                AxonmlTheme::loss_bad()
            };

            vec![
                Line::from(vec![
                    Span::styled("Train Loss:  ", AxonmlTheme::metric_label()),
                    Span::styled(format!("{:.4}", m.train_loss), loss_style),
                ]),
                Line::from(vec![
                    Span::styled("Val Loss:    ", AxonmlTheme::metric_label()),
                    Span::styled(
                        m.val_loss.map(|v| format!("{:.4}", v)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::metric_value(),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Train Acc:   ", AxonmlTheme::metric_label()),
                    Span::styled(
                        m.train_acc.map(|v| format!("{:.2}%", v * 100.0)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::success(),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Val Acc:     ", AxonmlTheme::metric_label()),
                    Span::styled(
                        m.val_acc.map(|v| format!("{:.2}%", v * 100.0)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::success(),
                    ),
                ]),
            ]
        } else {
            vec![Line::from(Span::styled("No metrics yet", AxonmlTheme::muted()))]
        };

        let metrics = Paragraph::new(metrics_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border_focused())
                    .title(Span::styled(" Current Metrics ", AxonmlTheme::header())),
            );

        frame.render_widget(metrics, chunks[0]);

        // Loss sparkline
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Loss Trend (inverted) ", AxonmlTheme::header())),
            )
            .data(&session.loss_history)
            .style(AxonmlTheme::graph_primary());

        frame.render_widget(sparkline, chunks[1]);
    }

    fn render_history(&self, frame: &mut Frame, area: Rect, session: &TrainingSession) {
        let history_lines: Vec<Line> = session
            .metrics_history
            .iter()
            .rev()
            .take(6)
            .map(|m| {
                Line::from(vec![
                    Span::styled(format!("Epoch {:>2} ", m.epoch), AxonmlTheme::epoch()),
                    Span::styled("Loss: ", AxonmlTheme::muted()),
                    Span::styled(format!("{:.4}", m.train_loss), AxonmlTheme::metric_value()),
                    Span::raw(" / "),
                    Span::styled(
                        m.val_loss.map(|v| format!("{:.4}", v)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::accent(),
                    ),
                    Span::raw("  "),
                    Span::styled("Acc: ", AxonmlTheme::muted()),
                    Span::styled(
                        m.train_acc.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::success(),
                    ),
                    Span::raw(" / "),
                    Span::styled(
                        m.val_acc.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "-".to_string()),
                        AxonmlTheme::success(),
                    ),
                    Span::raw("  "),
                    Span::styled(format!("({:.1}s)", m.duration_secs), AxonmlTheme::muted()),
                ])
            })
            .collect();

        let history = Paragraph::new(history_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Epoch History (recent) ", AxonmlTheme::header())),
            );

        frame.render_widget(history, area);
    }

    fn render_empty(&self, frame: &mut Frame, area: Rect) {
        let text = vec![
            Line::from(""),
            Line::from(Span::styled(
                "No training session active",
                AxonmlTheme::muted(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Press 't' to start training",
                AxonmlTheme::info(),
            )),
            Line::from(Span::styled(
                "or load a model first with 'o'",
                AxonmlTheme::muted(),
            )),
        ];

        let paragraph = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(AxonmlTheme::border())
                    .title(Span::styled(" Training ", AxonmlTheme::header())),
            )
            .alignment(Alignment::Center);

        frame.render_widget(paragraph, area);
    }
}

impl Default for TrainingView {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn format_duration(secs: f64) -> String {
    let total_secs = secs as u64;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    } else {
        format!("{:02}:{:02}", minutes, seconds)
    }
}
