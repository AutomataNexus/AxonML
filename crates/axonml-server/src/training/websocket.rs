//! WebSocket metrics streaming for AxonML
//!
//! Provides real-time training metrics via WebSocket connections.

use crate::db::runs::TrainingMetrics;
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::sync::broadcast;

/// Metrics message for WebSocket
#[derive(Debug, Clone, Serialize)]
pub struct MetricsMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub data: MetricsData,
}

/// Metrics data payload
#[derive(Debug, Clone, Serialize)]
pub struct MetricsData {
    pub epoch: u32,
    pub step: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_util: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<f64>,
    pub timestamp: String,
}

/// Status message for WebSocket
#[derive(Debug, Clone, Serialize)]
pub struct StatusMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub data: StatusData,
}

/// Status data payload
#[derive(Debug, Clone, Serialize)]
pub struct StatusData {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
}

/// Metrics streamer for WebSocket connections
pub struct MetricsStreamer;

impl MetricsStreamer {
    /// Stream metrics to a WebSocket connection
    pub async fn stream(
        socket: WebSocket,
        mut receiver: broadcast::Receiver<TrainingMetrics>,
    ) {
        let (mut sender, mut ws_receiver) = socket.split();

        // Spawn task to handle incoming messages (ping/pong, close)
        let recv_handle = tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                if let Ok(msg) = msg {
                    if matches!(msg, Message::Close(_)) {
                        break;
                    }
                } else {
                    break;
                }
            }
        });

        // Send metrics as they arrive
        loop {
            // Check if receiver task is finished
            if recv_handle.is_finished() {
                break;
            }

            match receiver.recv().await {
                Ok(metrics) => {
                    let message = MetricsMessage {
                        msg_type: "metrics".to_string(),
                        data: MetricsData {
                            epoch: metrics.epoch,
                            step: metrics.step,
                            loss: metrics.loss,
                            accuracy: metrics.accuracy,
                            lr: metrics.lr,
                            gpu_util: metrics.gpu_util,
                            memory_mb: metrics.memory_mb,
                            timestamp: metrics.timestamp.to_rfc3339(),
                        },
                    };

                    let json = serde_json::to_string(&message).unwrap_or_default();
                    if sender.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => {
                    // Run completed, send status message
                    let status = StatusMessage {
                        msg_type: "status".to_string(),
                        data: StatusData {
                            status: "completed".to_string(),
                            completed_at: Some(chrono::Utc::now().to_rfc3339()),
                        },
                    };

                    let json = serde_json::to_string(&status).unwrap_or_default();
                    let _ = sender.send(Message::Text(json)).await;
                    break;
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    // Missed some messages, continue
                    continue;
                }
            }
        }

        recv_handle.abort();
    }

    /// Send a single metrics update
    pub fn format_metrics(metrics: &TrainingMetrics) -> String {
        let message = MetricsMessage {
            msg_type: "metrics".to_string(),
            data: MetricsData {
                epoch: metrics.epoch,
                step: metrics.step,
                loss: metrics.loss,
                accuracy: metrics.accuracy,
                lr: metrics.lr,
                gpu_util: metrics.gpu_util,
                memory_mb: metrics.memory_mb,
                timestamp: metrics.timestamp.to_rfc3339(),
            },
        };

        serde_json::to_string(&message).unwrap_or_default()
    }

    /// Send a status update
    pub fn format_status(status: &str, completed_at: Option<&str>) -> String {
        let message = StatusMessage {
            msg_type: "status".to_string(),
            data: StatusData {
                status: status.to_string(),
                completed_at: completed_at.map(String::from),
            },
        };

        serde_json::to_string(&message).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_metrics() {
        let metrics = TrainingMetrics {
            epoch: 5,
            step: 1000,
            loss: Some(0.5),
            accuracy: Some(0.9),
            lr: Some(0.001),
            gpu_util: None,
            memory_mb: None,
            custom: serde_json::json!({}),
            timestamp: chrono::Utc::now(),
        };

        let json = MetricsStreamer::format_metrics(&metrics);
        assert!(json.contains("\"type\":\"metrics\""));
        assert!(json.contains("\"epoch\":5"));
    }

    #[test]
    fn test_format_status() {
        let json = MetricsStreamer::format_status("completed", Some("2024-01-15T10:30:00Z"));
        assert!(json.contains("\"type\":\"status\""));
        assert!(json.contains("\"status\":\"completed\""));
    }
}
