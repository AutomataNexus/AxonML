//! Training Monitor — live browser-based training dashboard
//!
//! # File
//! `crates/axonml/src/monitor.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 10, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

// =============================================================================
// Types
// =============================================================================

/// A single epoch's metrics.
#[derive(Clone, Debug)]
pub struct EpochMetrics {
    /// Epoch number (1-based).
    pub epoch: usize,
    /// Total training loss.
    pub train_loss: f32,
    /// Optional validation loss.
    pub val_loss: Option<f32>,
    /// Additional named metrics (e.g., "cls_loss", "bbox_loss", "img_s").
    pub extras: Vec<(String, f32)>,
}

/// Shared state for the monitor.
#[derive(Clone)]
struct MonitorState {
    model_name: String,
    param_count: usize,
    total_epochs: usize,
    batch_size: usize,
    status: String,
    best_loss: f32,
    epochs: Vec<EpochMetrics>,
}

/// Live training monitor that serves a browser dashboard.
///
/// # Usage
/// ```no_run
/// use axonml::monitor::TrainingMonitor;
///
/// let monitor = TrainingMonitor::new("BlazeFace", 125506)
///     .total_epochs(50)
///     .batch_size(32)
///     .launch();
///
/// // In training loop:
/// monitor.log_epoch(1, 0.5, None, vec![("cls", 0.03), ("bbox", 0.47)]);
/// monitor.set_status("complete");
/// ```
pub struct TrainingMonitor {
    state: Arc<Mutex<MonitorState>>,
    port: u16,
}

// =============================================================================
// Implementation
// =============================================================================

impl TrainingMonitor {
    /// Create a new monitor builder.
    pub fn new(model_name: &str, param_count: usize) -> MonitorBuilder {
        MonitorBuilder {
            model_name: model_name.to_string(),
            param_count,
            total_epochs: 50,
            batch_size: 1,
        }
    }

    /// Log metrics for a completed epoch.
    pub fn log_epoch(
        &self,
        epoch: usize,
        train_loss: f32,
        val_loss: Option<f32>,
        extras: Vec<(&str, f32)>,
    ) {
        let mut state = self.state.lock().unwrap();
        let metrics = EpochMetrics {
            epoch,
            train_loss,
            val_loss,
            extras: extras
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        };
        state.epochs.push(metrics);
        if train_loss < state.best_loss {
            state.best_loss = train_loss;
        }
        state.status = "training".to_string();
    }

    /// Update the monitor status (e.g., "training", "complete", "evaluating").
    pub fn set_status(&self, status: &str) {
        self.state.lock().unwrap().status = status.to_string();
    }

    /// Get the port the monitor is running on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Build the JSON metrics response.
    fn build_json(state: &MonitorState) -> String {
        let mut json = String::with_capacity(4096);
        json.push_str("{\n");
        json.push_str(&format!("  \"model\": \"{}\",\n", state.model_name));
        json.push_str(&format!("  \"params\": {},\n", state.param_count));
        json.push_str(&format!("  \"total_epochs\": {},\n", state.total_epochs));
        json.push_str(&format!("  \"batch_size\": {},\n", state.batch_size));
        json.push_str(&format!("  \"status\": \"{}\",\n", state.status));
        json.push_str(&format!("  \"best_loss\": {:.6},\n", state.best_loss));
        json.push_str(&format!(
            "  \"current_epoch\": {},\n",
            state.epochs.last().map_or(0, |e| e.epoch)
        ));
        json.push_str("  \"epochs\": [\n");
        for (i, ep) in state.epochs.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"epoch\":{},\"train_loss\":{:.6}",
                ep.epoch, ep.train_loss
            ));
            if let Some(vl) = ep.val_loss {
                json.push_str(&format!(",\"val_loss\":{vl:.6}"));
            }
            for (key, val) in &ep.extras {
                json.push_str(&format!(",\"{key}\":{val:.6}"));
            }
            json.push('}');
            if i + 1 < state.epochs.len() {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("  ]\n}");
        json
    }
}

/// Builder for configuring a `TrainingMonitor`.
pub struct MonitorBuilder {
    model_name: String,
    param_count: usize,
    total_epochs: usize,
    batch_size: usize,
}

impl MonitorBuilder {
    /// Set total number of epochs.
    pub fn total_epochs(mut self, n: usize) -> Self {
        self.total_epochs = n;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Launch the monitor — starts HTTP server and opens Chromium.
    pub fn launch(self) -> TrainingMonitor {
        let state = Arc::new(Mutex::new(MonitorState {
            model_name: self.model_name,
            param_count: self.param_count,
            total_epochs: self.total_epochs,
            batch_size: self.batch_size,
            status: "initializing".to_string(),
            best_loss: f32::MAX,
            epochs: Vec::new(),
        }));

        // Bind to port 0 to get an auto-assigned free port
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind monitor port");
        let port = listener.local_addr().unwrap().port();

        let server_state = state.clone();
        thread::spawn(move || {
            serve_http(listener, server_state);
        });

        // Open Chromium
        let url = format!("http://127.0.0.1:{port}");
        println!("Training monitor: {url}");
        let _ = std::process::Command::new("chromium-browser")
            .arg("--no-sandbox")
            .arg("--new-window")
            .arg(&url)
            .spawn()
            .or_else(|_| {
                std::process::Command::new("chromium")
                    .arg("--no-sandbox")
                    .arg("--new-window")
                    .arg(&url)
                    .spawn()
            })
            .or_else(|_| {
                std::process::Command::new("google-chrome")
                    .arg("--no-sandbox")
                    .arg("--new-window")
                    .arg(&url)
                    .spawn()
            })
            .or_else(|_| {
                // WSL2: try opening in Windows browser
                std::process::Command::new("cmd.exe")
                    .args(["/C", "start", &url])
                    .spawn()
            });

        TrainingMonitor { state, port }
    }
}

// =============================================================================
// HTTP Server (pure std, no deps)
// =============================================================================

fn serve_http(listener: TcpListener, state: Arc<Mutex<MonitorState>>) {
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap_or(0);
        let request = String::from_utf8_lossy(&buf[..n]);

        let (status, content_type, body) =
            if request.starts_with("GET /api/metrics") || request.starts_with("GET /api/state") {
                let state = state.lock().unwrap();
                let json = TrainingMonitor::build_json(&state);
                ("200 OK", "application/json", json)
            } else if request.starts_with("POST /api/epoch") {
                // REST API: accept metrics from any language (Python, JS, etc.)
                // POST /api/epoch  {"epoch":1,"train_loss":0.5,"val_loss":null,"extras":{"cls":0.1}}
                // POST /api/status {"status":"complete"}
                let body_str = extract_http_body(&request);
                match parse_epoch_post(&body_str) {
                    Ok(metrics) => {
                        let mut state = state.lock().unwrap();
                        if metrics.train_loss < state.best_loss {
                            state.best_loss = metrics.train_loss;
                        }
                        state.status = "training".to_string();
                        state.epochs.push(metrics);
                        ("200 OK", "application/json", r#"{"ok":true}"#.to_string())
                    }
                    Err(e) => (
                        "400 Bad Request",
                        "application/json",
                        format!("{{\"error\":\"{e}\"}}"),
                    ),
                }
            } else if request.starts_with("POST /api/status") {
                let body_str = extract_http_body(&request);
                if let Some(s) = extract_json_value(&body_str, "status") {
                    state.lock().unwrap().status = s;
                    ("200 OK", "application/json", r#"{"ok":true}"#.to_string())
                } else {
                    (
                        "400 Bad Request",
                        "application/json",
                        r#"{"error":"missing status"}"#.to_string(),
                    )
                }
            } else if request.starts_with("POST /api/config") {
                // Allow updating model_name, total_epochs, batch_size, param_count
                let body_str = extract_http_body(&request);
                let mut state = state.lock().unwrap();
                if let Some(v) = extract_json_value(&body_str, "model") {
                    state.model_name = v;
                }
                if let Some(v) = extract_json_value(&body_str, "total_epochs") {
                    if let Ok(n) = v.parse::<usize>() {
                        state.total_epochs = n;
                    }
                }
                if let Some(v) = extract_json_value(&body_str, "batch_size") {
                    if let Ok(n) = v.parse::<usize>() {
                        state.batch_size = n;
                    }
                }
                if let Some(v) = extract_json_value(&body_str, "params") {
                    if let Ok(n) = v.parse::<usize>() {
                        state.param_count = n;
                    }
                }
                ("200 OK", "application/json", r#"{"ok":true}"#.to_string())
            } else if request.starts_with("GET / ") || request.starts_with("GET / HTTP") {
                ("200 OK", "text/html", DASHBOARD_HTML.to_string())
            } else {
                ("404 Not Found", "text/plain", "Not Found".to_string())
            };

        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    }
}

/// Extract HTTP body after the \r\n\r\n separator.
fn extract_http_body(request: &str) -> String {
    if let Some(pos) = request.find("\r\n\r\n") {
        request[pos + 4..].to_string()
    } else {
        String::new()
    }
}

/// Minimal JSON string value extractor (no serde needed).
fn extract_json_value(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":");
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    if rest.starts_with('"') {
        // String value
        let inner = &rest[1..];
        let end = inner.find('"')?;
        Some(inner[..end].to_string())
    } else if rest.starts_with("null") {
        None
    } else {
        // Number value
        let num: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        Some(num)
    }
}

/// Parse a POST /api/epoch JSON body into EpochMetrics.
fn parse_epoch_post(body: &str) -> Result<EpochMetrics, &'static str> {
    let epoch_str = extract_json_value(body, "epoch").ok_or("missing epoch")?;
    let epoch: usize = epoch_str.parse().map_err(|_| "invalid epoch")?;

    let loss_str = extract_json_value(body, "train_loss").ok_or("missing train_loss")?;
    let train_loss: f32 = loss_str.parse().map_err(|_| "invalid train_loss")?;

    let val_loss = extract_json_value(body, "val_loss").and_then(|v| v.parse::<f32>().ok());

    // Parse extras object: {"extras":{"cls":0.1,"bbox":0.4}}
    let mut extras = Vec::new();
    if let Some(pos) = body.find("\"extras\"") {
        let rest = &body[pos..];
        if let Some(brace_start) = rest.find('{') {
            if let Some(brace_end) = rest[brace_start..].find('}') {
                let inner = &rest[brace_start + 1..brace_start + brace_end];
                // Parse key:value pairs
                for part in inner.split(',') {
                    let kv: Vec<&str> = part.splitn(2, ':').collect();
                    if kv.len() == 2 {
                        let key = kv[0].trim().trim_matches('"');
                        if let Ok(val) = kv[1].trim().parse::<f32>() {
                            extras.push((key.to_string(), val));
                        }
                    }
                }
            }
        }
    }

    Ok(EpochMetrics {
        epoch,
        train_loss,
        val_loss,
        extras,
    })
}

// =============================================================================
// Dashboard HTML — Prometheus NexusEdge theme (light, warm neutrals, teal accent)
// =============================================================================

const DASHBOARD_HTML: &str = include_str!("monitor_dashboard.html");

// Old HTML removed — now loaded from monitor_dashboard.html via include_str!

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_json_output() {
        let state = MonitorState {
            model_name: "TestModel".to_string(),
            param_count: 1000,
            total_epochs: 10,
            batch_size: 8,
            status: "training".to_string(),
            best_loss: 0.5,
            epochs: vec![
                EpochMetrics {
                    epoch: 1,
                    train_loss: 0.8,
                    val_loss: Some(0.9),
                    extras: vec![("cls".to_string(), 0.1)],
                },
                EpochMetrics {
                    epoch: 2,
                    train_loss: 0.5,
                    val_loss: None,
                    extras: vec![],
                },
            ],
        };
        let json = TrainingMonitor::build_json(&state);
        assert!(json.contains("\"model\": \"TestModel\""));
        assert!(json.contains("\"best_loss\": 0.500000"));
        assert!(json.contains("\"epoch\":1"));
        assert!(json.contains("\"val_loss\":0.900000"));
        assert!(json.contains("\"epoch\":2"));
    }

    #[test]
    fn test_log_epoch_updates_best() {
        let monitor = TrainingMonitor::new("Test", 100)
            .total_epochs(5)
            .batch_size(4)
            .launch();
        monitor.log_epoch(1, 0.8, None, vec![]);
        monitor.log_epoch(2, 0.5, None, vec![]);
        monitor.log_epoch(3, 0.6, None, vec![]);
        let state = monitor.state.lock().unwrap();
        assert_eq!(state.epochs.len(), 3);
        assert!((state.best_loss - 0.5).abs() < 1e-6);
    }
}
