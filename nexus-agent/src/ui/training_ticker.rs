//! nexus-training-ticker — Live Training Run Desktop Widget
//!
//! eframe-based frameless desktop widget specialised for watching an AxonML
//! training run. Mirrors the visual language of the other tickers
//! (`ticker.rs`, `tech_ticker.rs`, `ferrum_ticker.rs`) — frameless
//! transparent rounded pill, custom drag-region titlebar, dark/light theme
//! toggle — but trades the CI-monitor backend for a training-lifecycle
//! backend:
//!
//! * **Control socket** — connects to the per-run Unix socket installed by
//!   [`llm_training::lifecycle`]. The socket path comes from the
//!   `AXONML_TICKER_SOCKET` env var when spawned by the training binary,
//!   or falls back to the `/tmp/axonml-train-latest.sock` symlink when
//!   launched standalone. An `info` / `status` query on connect pulls
//!   `model_name`, `monitor_port`, `started_at`, `total_epochs`,
//!   `batch_size`, `param_count`. Buttons send `pause`, `resume`, `stop`,
//!   `checkpoint`.
//! * **Monitor HTTP** — once the monitor port is known, a background task
//!   polls `http://127.0.0.1:<port>/api/metrics` every 2 s. New epochs
//!   append a one-line console entry (`[epoch N] train=... val=... extras`)
//!   and push to the `train_history` / `val_history` / `extras_history`
//!   vectors that feed the dotted-line loss plot.
//! * **Auto-reconnect** — if the socket read fails (training process exited
//!   cleanly or crashed), the ticker flips to `DISCONNECTED`, retries
//!   `LATEST_SOCKET` every 5 s, and re-attaches to the next run.
//!
//! # Layout
//!
//! Everything is sized from `ui.available_rect_before_wrap()` — no hardcoded
//! pixel positions. Default window size is 400×500 px and the widget must
//! reflow cleanly on resize:
//!
//! * titlebar                                  (fixed small height)
//! * status line + model name + buttons row    (wraps)
//! * dotted-line loss plot                     (40% of remaining height)
//! * scrolling console log                     (remaining height)
//!
//! Plot is rendered manually via `ui.painter()` — two dotted series
//! (train + val) over epoch index, with a teal/amber legend — so no
//! `egui_plot` dependency is required.
//!
//! # Launch
//!
//! Spawned automatically by training binaries invoked with `--ticker`
//! (see `llm-training/src/lifecycle.rs`), or standalone:
//!
//! ```bash
//! nexus-training-ticker &
//! ```
//!
//! # File
//! `nexus-agent/src/ui/training_ticker.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 18, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use eframe::egui;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// =============================================================================
// Constants
// =============================================================================

const TICKER_WIDTH: f32 = 400.0;
const TICKER_HEIGHT: f32 = 500.0;
const MIN_WIDTH: f32 = 320.0;
const MIN_HEIGHT: f32 = 320.0;
const MAX_LOG_LINES: usize = 200;
const MAX_EPOCHS_PLOTTED: usize = 10_000;
const METRICS_POLL_SECS: f32 = 2.0;
const RECONNECT_POLL_SECS: f32 = 5.0;
const LATEST_SOCKET: &str = "/tmp/axonml-train-latest.sock";
const THEME_FILE: &str = "/tmp/.nexus-training-ticker-theme";
const SOCKET_ENV: &str = "AXONML_TICKER_SOCKET";

// =============================================================================
// Entry Point
// =============================================================================

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([TICKER_WIDTH, TICKER_HEIGHT])
            .with_min_inner_size([MIN_WIDTH, MIN_HEIGHT])
            .with_always_on_top()
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(true)
            .with_title("nexus-training-ticker"),
        ..Default::default()
    };

    eframe::run_native(
        "nexus-training-ticker",
        options,
        Box::new(|cc| Ok(Box::new(TrainingTickerApp::new(cc)))),
    )
}

// =============================================================================
// State
// =============================================================================

#[derive(Clone, PartialEq, Debug)]
enum Connection {
    Disconnected,
    Connected,
    Paused,
    Stopping,
    Complete,
}

impl Connection {
    fn label(&self) -> &'static str {
        match self {
            Self::Disconnected => "DISCONNECTED",
            Self::Connected => "TRAINING",
            Self::Paused => "PAUSED",
            Self::Stopping => "STOPPING",
            Self::Complete => "COMPLETE",
        }
    }
}

#[derive(Clone)]
struct LogEntry {
    timestamp: String,
    message: String,
    is_error: bool,
}

/// One epoch's worth of metrics mirrored from `/api/metrics`.
#[derive(Clone)]
struct EpochPoint {
    epoch: usize,
    train_loss: f32,
    val_loss: Option<f32>,
    extras: Vec<(String, f32)>,
}

/// Info pulled from the socket `status` command.
#[derive(Clone, Default)]
struct RunInfo {
    model_name: String,
    monitor_port: u16,
    started_at: u64,
    total_epochs: usize,
    batch_size: usize,
    param_count: usize,
    current_epoch: usize,
    global_step: u64,
    last_loss: f32,
}

struct SharedState {
    connection: Connection,
    socket_path: String,
    info: RunInfo,
    /// All epoch points, in arrival order. Capped at `MAX_EPOCHS_PLOTTED`.
    epochs: Vec<EpochPoint>,
    /// Last epoch number we already emitted a console line for.
    last_logged_epoch: usize,
    log: Vec<LogEntry>,
}

impl SharedState {
    fn push_log(&mut self, msg: impl Into<String>, is_error: bool) {
        self.log.push(LogEntry {
            timestamp: now_hms(),
            message: msg.into(),
            is_error,
        });
        let excess = self.log.len().saturating_sub(MAX_LOG_LINES);
        if excess > 0 {
            self.log.drain(0..excess);
        }
    }
}

struct TrainingTickerApp {
    state: Arc<Mutex<SharedState>>,
    last_metrics_poll: Instant,
    last_reconnect_try: Instant,
}

impl TrainingTickerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let socket_path = std::env::var(SOCKET_ENV)
            .unwrap_or_else(|_| LATEST_SOCKET.to_string());
        let state = Arc::new(Mutex::new(SharedState {
            connection: Connection::Disconnected,
            socket_path: socket_path.clone(),
            info: RunInfo::default(),
            epochs: Vec::new(),
            last_logged_epoch: 0,
            log: vec![LogEntry {
                timestamp: now_hms(),
                message: format!("training ticker started — socket {socket_path}"),
                is_error: false,
            }],
        }));
        Self {
            state,
            last_metrics_poll: Instant::now() - Duration::from_secs(60),
            last_reconnect_try: Instant::now() - Duration::from_secs(60),
        }
    }

    /// Attempt to fetch and apply `status` from the control socket.
    /// Returns true when a fresh status was read.
    fn try_status(&self) -> bool {
        let path = self.state.lock().unwrap().socket_path.clone();
        match query_socket(&path, "status") {
            Ok(line) => {
                if let Some(info) = parse_status_json(&line) {
                    if let Ok(mut s) = self.state.lock() {
                        let prev = s.connection.clone();
                        let paused = info.paused;
                        let stopping = info.stopping;
                        s.info = info.into_run_info();
                        s.connection = if stopping {
                            Connection::Stopping
                        } else if paused {
                            Connection::Paused
                        } else {
                            Connection::Connected
                        };
                        if prev == Connection::Disconnected {
                            let model_label = if s.info.model_name.is_empty() {
                                "unknown".to_string()
                            } else {
                                s.info.model_name.clone()
                            };
                            let monitor_port = s.info.monitor_port;
                            s.push_log(
                                format!(
                                    "connected — model {model_label} (monitor :{monitor_port})"
                                ),
                                false,
                            );
                        }
                    }
                    return true;
                }
                false
            }
            Err(_) => {
                // Socket is gone — drop to disconnected and forget the port.
                if let Ok(mut s) = self.state.lock() {
                    if s.connection != Connection::Disconnected
                        && s.connection != Connection::Complete
                    {
                        s.connection = Connection::Disconnected;
                        s.info.monitor_port = 0;
                        s.push_log("socket unreachable — waiting to re-attach", true);
                    }
                }
                false
            }
        }
    }

    /// Poll the browser monitor's `/api/metrics` endpoint and merge new epochs.
    fn poll_metrics(&self) {
        let port = {
            let s = self.state.lock().unwrap();
            if s.info.monitor_port == 0 {
                return;
            }
            s.info.monitor_port
        };
        let url = format!("http://127.0.0.1:{port}/api/metrics");
        let body = match http_get(&url, Duration::from_secs(2)) {
            Ok(b) => b,
            Err(_) => return,
        };
        let points = parse_metrics_json(&body);
        if points.is_empty() {
            return;
        }
        let status_from_json = extract_json_string(&body, "status");
        if let Ok(mut s) = self.state.lock() {
            let prev_last = s.last_logged_epoch;
            for p in &points {
                if p.epoch > prev_last {
                    let mut line = format!(
                        "[epoch {}] train={:.4}",
                        p.epoch, p.train_loss
                    );
                    if let Some(v) = p.val_loss {
                        line.push_str(&format!(" val={v:.4}"));
                    }
                    for (k, v) in &p.extras {
                        line.push_str(&format!(" {k}={v:.4}"));
                    }
                    s.push_log(line, false);
                }
            }
            // Replace stored history (monitor is source of truth).
            s.epochs = points;
            if s.epochs.len() > MAX_EPOCHS_PLOTTED {
                let drop = s.epochs.len() - MAX_EPOCHS_PLOTTED;
                s.epochs.drain(0..drop);
            }
            if let Some(last) = s.epochs.last() {
                s.last_logged_epoch = s.last_logged_epoch.max(last.epoch);
            }
            if let Some(st) = status_from_json {
                if st == "complete" && s.connection != Connection::Complete {
                    s.connection = Connection::Complete;
                    s.push_log("run complete", false);
                }
            }
        }
    }

    fn send_control(&self, cmd: &'static str) {
        let path = self.state.lock().unwrap().socket_path.clone();
        let result = query_socket(&path, cmd);
        if let Ok(mut s) = self.state.lock() {
            match result {
                Ok(line) => {
                    let msg = line.trim();
                    s.push_log(format!("> {cmd}  ({msg})"), false);
                    match cmd {
                        "pause" => s.connection = Connection::Paused,
                        "resume" => s.connection = Connection::Connected,
                        "stop" => s.connection = Connection::Stopping,
                        _ => {}
                    }
                }
                Err(e) => {
                    s.push_log(format!("> {cmd}  (send failed: {e})"), true);
                }
            }
        }
    }
}

// =============================================================================
// Socket + HTTP helpers
// =============================================================================

/// Send a single-line command to the control socket and read one reply line.
fn query_socket(path: &str, cmd: &str) -> Result<String, String> {
    // Resolve symlink + check file exists first; UnixStream::connect is OK
    // with symlinks but we want a clearer error message for the log.
    let meta = std::fs::metadata(path).map_err(|e| format!("no socket at {path}: {e}"))?;
    if !meta.file_type().is_socket() && !meta.file_type().is_symlink() {
        // metadata follows symlinks, so `meta.file_type().is_socket()` is
        // what matters. Accept socket; fail on anything else.
        return Err(format!("{path} is not a unix socket"));
    }
    let mut stream = UnixStream::connect(path).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .ok();
    stream
        .write_all(format!("{cmd}\n").as_bytes())
        .map_err(|e| e.to_string())?;
    stream.flush().ok();
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    Ok(line)
}

/// Blocking HTTP GET using plain std::net. Returns the body only.
fn http_get(url: &str, timeout: Duration) -> Result<String, String> {
    // Parse `http://host:port/path` by hand — no reqwest dependency needed
    // for localhost loopback.
    let without_scheme = url.strip_prefix("http://").ok_or("not http://")?;
    let (host_port, path) = match without_scheme.find('/') {
        Some(i) => (&without_scheme[..i], &without_scheme[i..]),
        None => (without_scheme, "/"),
    };
    let addr: std::net::SocketAddr = host_port
        .parse()
        .map_err(|e: std::net::AddrParseError| e.to_string())?;
    let mut stream = std::net::TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let req = format!(
        "GET {path} HTTP/1.0\r\nHost: {host_port}\r\nConnection: close\r\n\r\n"
    );
    use std::io::{Read, Write};
    stream
        .write_all(req.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::with_capacity(16384);
    stream.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf).to_string();
    match text.find("\r\n\r\n") {
        Some(i) => Ok(text[i + 4..].to_string()),
        None => Err("malformed http response".to_string()),
    }
}

// =============================================================================
// JSON parsing (minimal, hand-rolled)
// =============================================================================

/// The direct fields we pull from the `status` socket reply. Mirrors the
/// JSON object built in `llm-training/src/lifecycle.rs::handle_connection`.
#[derive(Clone, Default)]
struct StatusBlob {
    model_name: String,
    monitor_port: u16,
    started_at: u64,
    total_epochs: usize,
    batch_size: usize,
    params: usize,
    epoch: usize,
    global_step: u64,
    last_loss: f32,
    paused: bool,
    stopping: bool,
    monitor_url: String,
}

impl StatusBlob {
    fn into_run_info(self) -> RunInfo {
        // If monitor_port is zero but we got a "monitor" URL with a port,
        // prefer parsing that.
        let port = if self.monitor_port != 0 {
            self.monitor_port
        } else {
            self.monitor_url
                .rsplit_once(':')
                .and_then(|(_, tail)| tail.split('/').next())
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0)
        };
        RunInfo {
            model_name: self.model_name,
            monitor_port: port,
            started_at: self.started_at,
            total_epochs: self.total_epochs,
            batch_size: self.batch_size,
            param_count: self.params,
            current_epoch: self.epoch,
            global_step: self.global_step,
            last_loss: self.last_loss,
        }
    }
}

fn parse_status_json(text: &str) -> Option<StatusBlob> {
    if !text.contains("\"model\"") && !text.contains("\"monitor\"") {
        return None;
    }
    Some(StatusBlob {
        model_name: extract_json_string(text, "model").unwrap_or_default(),
        monitor_port: extract_json_number::<u16>(text, "monitor_port").unwrap_or(0),
        started_at: extract_json_number::<u64>(text, "started_at").unwrap_or(0),
        total_epochs: extract_json_number::<usize>(text, "total_epochs").unwrap_or(0),
        batch_size: extract_json_number::<usize>(text, "batch_size").unwrap_or(0),
        params: extract_json_number::<usize>(text, "params").unwrap_or(0),
        epoch: extract_json_number::<usize>(text, "epoch").unwrap_or(0),
        global_step: extract_json_number::<u64>(text, "global_step").unwrap_or(0),
        last_loss: extract_json_number::<f32>(text, "last_loss").unwrap_or(0.0),
        paused: extract_json_bool(text, "paused").unwrap_or(false),
        stopping: extract_json_bool(text, "stopping").unwrap_or(false),
        monitor_url: extract_json_string(text, "monitor").unwrap_or_default(),
    })
}

/// Walk the `"epochs":[...]` array from `/api/metrics` into `EpochPoint`s.
fn parse_metrics_json(body: &str) -> Vec<EpochPoint> {
    let epochs_start = match body.find("\"epochs\"") {
        Some(i) => i,
        None => return Vec::new(),
    };
    let arr_start = match body[epochs_start..].find('[') {
        Some(i) => epochs_start + i,
        None => return Vec::new(),
    };
    let arr_end = match body[arr_start..].rfind(']') {
        Some(i) => arr_start + i,
        None => return Vec::new(),
    };
    let arr = &body[arr_start + 1..arr_end];

    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut obj_start = None::<usize>;
    for (i, c) in arr.char_indices() {
        match c {
            '{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(s) = obj_start {
                        let obj = &arr[s..=i];
                        if let Some(p) = parse_epoch_obj(obj) {
                            out.push(p);
                        }
                        obj_start = None;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn parse_epoch_obj(obj: &str) -> Option<EpochPoint> {
    let epoch = extract_json_number::<usize>(obj, "epoch")?;
    let train_loss = extract_json_number::<f32>(obj, "train_loss").unwrap_or(0.0);
    let val_loss = extract_json_number::<f32>(obj, "val_loss");

    // Everything else is "extras" — any numeric key that isn't epoch /
    // train_loss / val_loss. Walk the key:value pairs at the top level of
    // this object (no nesting expected).
    let mut extras = Vec::new();
    // Strip the outer braces, then iterate comma-separated entries. The
    // monitor serialiser doesn't nest, so a naive split is safe.
    let inner = obj.trim_start_matches('{').trim_end_matches('}');
    for part in split_top_level_commas(inner) {
        let mut it = part.splitn(2, ':');
        let raw_k = it.next()?.trim();
        let raw_v = it.next()?.trim();
        let key = raw_k.trim_matches('"');
        if key == "epoch" || key == "train_loss" || key == "val_loss" {
            continue;
        }
        if let Ok(f) = raw_v.parse::<f32>() {
            extras.push((key.to_string(), f));
        }
    }

    Some(EpochPoint {
        epoch,
        train_loss,
        val_loss,
        extras,
    })
}

/// Split a flat "k:v,k:v,..." string on top-level commas, skipping commas
/// inside quoted strings or nested braces/brackets.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut last = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'"' => in_str = !in_str,
            b'{' | b'[' if !in_str => depth += 1,
            b'}' | b']' if !in_str => depth -= 1,
            b',' if !in_str && depth == 0 => {
                out.push(&s[last..i]);
                last = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if last < bytes.len() {
        out.push(&s[last..]);
    }
    out
}

fn extract_json_string(text: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let start = text.find(&pat)? + pat.len();
    let rest = &text[start..];
    let colon = rest.find(':')? + 1;
    let after = rest[colon..].trim_start();
    let inner = after.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

fn extract_json_number<T: std::str::FromStr>(text: &str, key: &str) -> Option<T> {
    let pat = format!("\"{key}\"");
    let start = text.find(&pat)? + pat.len();
    let rest = &text[start..];
    let colon = rest.find(':')? + 1;
    let after = rest[colon..].trim_start();
    // Handle quoted number, null, or bare number.
    if let Some(stripped) = after.strip_prefix('"') {
        let end = stripped.find('"')?;
        return stripped[..end].parse().ok();
    }
    if after.starts_with("null") {
        return None;
    }
    let num: String = after
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E' || *c == '+')
        .collect();
    num.parse().ok()
}

fn extract_json_bool(text: &str, key: &str) -> Option<bool> {
    let pat = format!("\"{key}\"");
    let start = text.find(&pat)? + pat.len();
    let rest = &text[start..];
    let colon = rest.find(':')? + 1;
    let after = rest[colon..].trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

// =============================================================================
// Time formatting
// =============================================================================

fn now_hms() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

fn format_uptime(started_at: u64) -> String {
    if started_at == 0 {
        return "–".to_string();
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let secs = now.saturating_sub(started_at);
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h{m:02}m")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

// =============================================================================
// UI — eframe::App
// =============================================================================

impl eframe::App for TrainingTickerApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint for the pulse animation and so polling stays responsive.
        ctx.request_repaint_after(Duration::from_millis(200));

        // Load persisted theme on first frame.
        static LOADED: std::sync::Once = std::sync::Once::new();
        LOADED.call_once(|| set_active_theme(Theme::load()));

        let is_light = ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) == 1;
        let mut visuals = if is_light { egui::Visuals::light() } else { egui::Visuals::dark() };
        visuals.panel_fill = egui::Color32::TRANSPARENT;
        visuals.window_fill = egui::Color32::TRANSPARENT;
        visuals.override_text_color = Some(text_color());
        ctx.set_visuals(visuals);

        // ---- Background polls (run on the UI thread — both calls are short
        // blocking I/O against localhost, guarded by ~2s timeouts) ----
        let now = Instant::now();
        let (conn, has_port) = {
            let s = self.state.lock().unwrap();
            (s.connection.clone(), s.info.monitor_port != 0)
        };
        if conn == Connection::Disconnected
            && now.duration_since(self.last_reconnect_try).as_secs_f32() >= RECONNECT_POLL_SECS
        {
            self.last_reconnect_try = now;
            self.try_status();
        } else if conn != Connection::Disconnected
            && now.duration_since(self.last_metrics_poll).as_secs_f32() >= METRICS_POLL_SECS
        {
            self.last_metrics_poll = now;
            // Refresh status every metrics tick so pause/stop state stays
            // in sync with external controllers (train_ctl, SIGUSR1, etc.).
            self.try_status();
            if has_port {
                self.poll_metrics();
            }
        }

        let snap = self.state.lock().unwrap().snapshot();

        // ---- Outer frame ----
        let outer = egui::Frame::none()
            .fill(bg_color())
            .rounding(egui::Rounding::same(10.0))
            .stroke(egui::Stroke::new(
                1.0,
                if is_light {
                    egui::Color32::from_rgba_unmultiplied(
                        text_dim().r(), text_dim().g(), text_dim().b(), 90,
                    )
                } else {
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, 18)
                },
            ))
            .shadow(egui::epaint::Shadow {
                offset: egui::vec2(0.0, 2.0),
                blur: 8.0,
                spread: 0.0,
                color: egui::Color32::from_rgba_unmultiplied(
                    0, 0, 0, if is_light { 32 } else { 120 },
                ),
            })
            .inner_margin(egui::Margin {
                left: 10.0, right: 10.0, top: 8.0, bottom: 10.0,
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(egui::Margin::same(6.0)))
            .show(ctx, |ui| {
                outer.show(ui, |ui| {
                    ui.set_clip_rect(ui.max_rect());
                    self.render_content(ui, ctx, &snap, is_light);
                });
            });
    }
}

impl TrainingTickerApp {
    fn render_content(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        snap: &Snapshot,
        is_light: bool,
    ) {
        // ── Custom titlebar ──
        let pulse = snap.connection == Connection::Connected
            || snap.connection == Connection::Stopping;
        let title_resp = ui
            .horizontal(|ui| {
                draw_led(ui, status_color(&snap.connection), 5.0, pulse);
                ui.label(
                    egui::RichText::new("NEXUS-TRAIN")
                        .strong()
                        .size(11.0)
                        .color(text_color()),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("✕").size(12.0).color(text_dim()),
                            )
                            .frame(false)
                            .min_size(egui::vec2(20.0, 20.0)),
                        )
                        .on_hover_text("close")
                        .clicked()
                    {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        std::process::exit(0);
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("—").size(12.0).color(text_dim()),
                            )
                            .frame(false)
                            .min_size(egui::vec2(20.0, 20.0)),
                        )
                        .on_hover_text("minimize")
                        .clicked()
                    {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                        let _ = std::process::Command::new("sh")
                            .arg("-c")
                            .arg("xdotool search --name '^nexus-training-ticker$' windowminimize 2>/dev/null")
                            .spawn();
                    }
                    // Theme toggle
                    let is_light_now =
                        ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) == 1;
                    let glyph = if is_light_now { "☾" } else { "☀" };
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(glyph).size(11.0).color(amber()).strong(),
                            )
                            .frame(false)
                            .min_size(egui::vec2(16.0, 16.0)),
                        )
                        .on_hover_text("Toggle light/dark theme")
                        .clicked()
                    {
                        let new_theme = if is_light_now { Theme::Dark } else { Theme::Light };
                        set_active_theme(new_theme);
                        new_theme.save();
                    }
                });
            })
            .response;
        let drag_sense = ui.interact(
            title_resp.rect,
            egui::Id::new("train-ticker-drag"),
            egui::Sense::click_and_drag(),
        );
        if drag_sense.drag_started_by(egui::PointerButton::Primary) {
            ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
        }
        ui.separator();

        // ── Status row: [LED] STATE   model   uptime ──
        ui.horizontal_wrapped(|ui| {
            draw_led(ui, status_color(&snap.connection), 5.0, pulse);
            ui.label(
                egui::RichText::new(snap.connection.label())
                    .strong()
                    .size(11.0)
                    .color(text_color()),
            );
            let model_display = if snap.info.model_name.is_empty() {
                "—".to_string()
            } else {
                snap.info.model_name.clone()
            };
            ui.label(
                egui::RichText::new(format!(" {model_display}"))
                    .size(10.0)
                    .color(text_dim()),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format_uptime(snap.info.started_at))
                        .size(9.0)
                        .color(text_dim()),
                );
            });
        });

        // ── Run summary: epoch / step / best loss / params ──
        ui.horizontal_wrapped(|ui| {
            let epoch_txt = if snap.info.total_epochs > 0 {
                format!(
                    "epoch {}/{}",
                    snap.info.current_epoch, snap.info.total_epochs
                )
            } else {
                format!("epoch {}", snap.info.current_epoch)
            };
            ui.label(egui::RichText::new(epoch_txt).size(9.0).color(text_dim()));
            ui.label(
                egui::RichText::new(format!("step {}", snap.info.global_step))
                    .size(9.0)
                    .color(text_dim()),
            );
            ui.label(
                egui::RichText::new(format!("loss {:.4}", snap.info.last_loss))
                    .size(9.0)
                    .color(text_dim()),
            );
            if snap.info.param_count > 0 {
                ui.label(
                    egui::RichText::new(format!("{} params", format_count(snap.info.param_count)))
                        .size(9.0)
                        .color(text_dim()),
                );
            }
            if snap.info.batch_size > 0 {
                ui.label(
                    egui::RichText::new(format!("bs={}", snap.info.batch_size))
                        .size(9.0)
                        .color(text_dim()),
                );
            }
        });

        // ── Control buttons ──
        ui.horizontal(|ui| {
            let is_paused = snap.connection == Connection::Paused;
            let is_stopping = snap.connection == Connection::Stopping;
            let is_live = matches!(
                snap.connection,
                Connection::Connected | Connection::Paused
            );

            let pause_clicked = ui
                .add_enabled(
                    is_live && !is_paused && !is_stopping,
                    egui::Button::new(
                        egui::RichText::new("Pause")
                            .size(10.0)
                            .color(if !is_live || is_paused || is_stopping {
                                text_dim()
                            } else {
                                amber()
                            }),
                    ),
                )
                .on_hover_text("Pause training after current step")
                .clicked();
            let resume_clicked = ui
                .add_enabled(
                    is_paused,
                    egui::Button::new(
                        egui::RichText::new("Resume")
                            .size(10.0)
                            .color(if is_paused { teal() } else { text_dim() }),
                    ),
                )
                .on_hover_text("Resume a paused run")
                .clicked();
            let stop_clicked = ui
                .add_enabled(
                    is_live && !is_stopping,
                    egui::Button::new(
                        egui::RichText::new("Stop / Cancel")
                            .size(10.0)
                            .color(if is_live && !is_stopping {
                                terracotta()
                            } else {
                                text_dim()
                            }),
                    ),
                )
                .on_hover_text("Request final checkpoint then exit")
                .clicked();
            let ckpt_clicked = ui
                .add_enabled(
                    is_live,
                    egui::Button::new(
                        egui::RichText::new("Checkpoint")
                            .size(10.0)
                            .color(if is_live { teal() } else { text_dim() }),
                    ),
                )
                .on_hover_text("Flush a checkpoint right now")
                .clicked();

            if pause_clicked {
                self.send_control("pause");
            }
            if resume_clicked {
                self.send_control("resume");
            }
            if stop_clicked {
                self.send_control("stop");
            }
            if ckpt_clicked {
                self.send_control("checkpoint");
            }
        });

        ui.add_space(2.0);
        ui.separator();

        // ── Dotted-line loss plot ──
        let avail = ui.available_rect_before_wrap();
        let plot_h = (avail.height() * 0.42).clamp(80.0, 260.0);
        draw_loss_plot(ui, plot_h, &snap.epochs, is_light);

        // ── Log toolbar ──
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("log")
                    .size(9.0)
                    .color(text_dim()),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(
                        egui::Button::new(
                            egui::RichText::new("clear").size(9.0).color(terracotta()),
                        )
                        .frame(false),
                    )
                    .on_hover_text("Clear the log buffer")
                    .clicked()
                {
                    if let Ok(mut s) = self.state.lock() {
                        s.log.clear();
                        s.push_log("log cleared", false);
                    }
                }
                ui.add_space(6.0);
                if ui
                    .add(
                        egui::Button::new(
                            egui::RichText::new("copy all").size(9.0).color(teal()),
                        )
                        .frame(false),
                    )
                    .on_hover_text("Copy entire log to clipboard")
                    .clicked()
                {
                    let full: String = snap
                        .log
                        .iter()
                        .map(|(ts, msg, _)| format!("{ts}  {msg}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    ctx.copy_text(full);
                }
            });
        });

        // ── Scrolling log (fills remaining height) ──
        let avail_after = ui.available_height().max(40.0);
        egui::ScrollArea::vertical()
            .max_height(avail_after)
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .scroll_bar_visibility(
                egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
            )
            .show(ui, |ui| {
                for (ts, msg, is_err) in &snap.log {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(ts).size(8.0).color(slate()));
                        let color = if *is_err { terracotta() } else { text_color() };
                        ui.label(egui::RichText::new(msg).size(9.0).color(color));
                    });
                }
            });
    }
}

// =============================================================================
// Plot — dotted loss curves
// =============================================================================

/// Draw train/val loss (and first extra, if any) as dotted lines scaled to
/// a rect of the requested height. Renders nothing when there are no points.
fn draw_loss_plot(
    ui: &mut egui::Ui,
    height: f32,
    epochs: &[EpochPoint],
    _is_light: bool,
) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);

    // Background
    painter.rect_filled(rect, egui::Rounding::same(4.0), bg_row());

    if epochs.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "waiting for epochs…",
            egui::FontId::proportional(10.0),
            slate(),
        );
        return;
    }

    // Pick which series we have. Train is always there; val optional.
    let has_val = epochs.iter().any(|p| p.val_loss.is_some());
    // One additional extra series: use the first extra name we see, if any.
    let extra_key: Option<String> = epochs
        .iter()
        .find_map(|p| p.extras.first().map(|(k, _)| k.clone()));

    // Gather values to compute y-range across all plotted series.
    let mut all_vals: Vec<f32> = epochs.iter().map(|p| p.train_loss).collect();
    if has_val {
        all_vals.extend(epochs.iter().filter_map(|p| p.val_loss));
    }
    if let Some(key) = &extra_key {
        all_vals.extend(epochs.iter().filter_map(|p| {
            p.extras.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
        }));
    }
    let all_vals: Vec<f32> = all_vals
        .into_iter()
        .filter(|v| v.is_finite())
        .collect();
    if all_vals.is_empty() {
        return;
    }
    let y_min = all_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = all_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_span = (y_max - y_min).max(1e-6);

    let x_min = epochs.first().map(|p| p.epoch as f32).unwrap_or(0.0);
    let x_max = epochs.last().map(|p| p.epoch as f32).unwrap_or(1.0);
    let x_span = (x_max - x_min).max(1e-6);

    // Drawable area inside `rect` — leave ~18 px on bottom for legend and
    // ~4 px padding on the other three sides.
    let pad_l = 4.0;
    let pad_r = 4.0;
    let pad_t = 4.0;
    let pad_b = 18.0;
    let inner = egui::Rect::from_min_max(
        egui::pos2(rect.min.x + pad_l, rect.min.y + pad_t),
        egui::pos2(rect.max.x - pad_r, rect.max.y - pad_b),
    );

    let map = |ep: usize, y: f32| -> egui::Pos2 {
        let t = (ep as f32 - x_min) / x_span;
        let v = (y - y_min) / y_span;
        egui::pos2(
            inner.min.x + t * inner.width(),
            // Flip Y — higher loss is drawn higher on screen
            inner.max.y - v * inner.height(),
        )
    };

    // Faint grid: 4 horizontal rules for visual reference.
    let grid_color = egui::Color32::from_rgba_unmultiplied(
        slate().r(), slate().g(), slate().b(), 40,
    );
    for i in 0..=3 {
        let t = i as f32 / 3.0;
        let y = inner.min.y + t * inner.height();
        painter.line_segment(
            [egui::pos2(inner.min.x, y), egui::pos2(inner.max.x, y)],
            egui::Stroke::new(1.0, grid_color),
        );
    }

    // ---- Draw series as dotted lines ----
    let train_points: Vec<egui::Pos2> =
        epochs.iter().map(|p| map(p.epoch, p.train_loss)).collect();
    draw_dotted_polyline(&painter, &train_points, teal());

    if has_val {
        let val_points: Vec<egui::Pos2> = epochs
            .iter()
            .filter_map(|p| p.val_loss.map(|v| map(p.epoch, v)))
            .collect();
        draw_dotted_polyline(&painter, &val_points, amber());
    }

    if let Some(key) = &extra_key {
        let ex_points: Vec<egui::Pos2> = epochs
            .iter()
            .filter_map(|p| {
                p.extras
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| map(p.epoch, *v))
            })
            .collect();
        draw_dotted_polyline(&painter, &ex_points, slate());
    }

    // ---- Legend + axis labels ----
    let legend_y = rect.max.y - 12.0;
    let mut x_cursor = rect.min.x + 8.0;
    let draw_swatch = |label: &str, color: egui::Color32, cursor: &mut f32| {
        painter.circle_filled(egui::pos2(*cursor, legend_y), 2.5, color);
        let galley = painter.layout_no_wrap(
            label.to_string(),
            egui::FontId::proportional(9.0),
            text_dim(),
        );
        painter.galley(egui::pos2(*cursor + 6.0, legend_y - 6.0), galley, text_dim());
        *cursor += 8.0 + label.len() as f32 * 5.5 + 10.0;
    };
    draw_swatch("train", teal(), &mut x_cursor);
    if has_val {
        draw_swatch("val", amber(), &mut x_cursor);
    }
    if let Some(key) = &extra_key {
        draw_swatch(key, slate(), &mut x_cursor);
    }

    // y-axis label (min / max loss, right-aligned so it doesn't collide
    // with the legend on the left)
    let y_galley = painter.layout_no_wrap(
        format!("{y_max:.3}  →  {y_min:.3}"),
        egui::FontId::proportional(8.0),
        slate(),
    );
    let y_pos = egui::pos2(rect.max.x - 8.0 - y_galley.size().x, rect.min.y + 2.0);
    painter.galley(y_pos, y_galley, slate());
}

/// Render a polyline as a sequence of short dots (three pixels per dot,
/// spaced). We draw per pixel along the path to get an evenly-spaced dotted
/// effect regardless of the underlying segment geometry.
fn draw_dotted_polyline(
    painter: &egui::Painter,
    points: &[egui::Pos2],
    color: egui::Color32,
) {
    if points.len() < 2 {
        if let Some(p) = points.first() {
            painter.circle_filled(*p, 1.5, color);
        }
        return;
    }
    const DOT_SPACING: f32 = 5.0;
    const DOT_RADIUS: f32 = 1.4;
    let mut acc = 0.0f32;
    for win in points.windows(2) {
        let a = win[0];
        let b = win[1];
        let seg_len = (b - a).length().max(1e-3);
        let dir = (b - a) / seg_len;
        // Start from the leftover offset on this segment.
        let mut t = -acc;
        while t < seg_len {
            if t >= 0.0 {
                let p = a + dir * t;
                painter.circle_filled(p, DOT_RADIUS, color);
            }
            t += DOT_SPACING;
        }
        // Carry over how far past the last dot we went so adjacent
        // segments stay evenly spaced.
        acc = (seg_len + acc) % DOT_SPACING;
    }
}

// =============================================================================
// Theme — mirrors the palette in ticker.rs / tech_ticker.rs
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Theme {
    Dark,
    Light,
}

impl Theme {
    fn load() -> Self {
        match std::fs::read_to_string(THEME_FILE)
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("light") => Self::Light,
            _ => Self::Dark,
        }
    }
    fn save(&self) {
        let _ = std::fs::write(
            THEME_FILE,
            match self {
                Self::Dark => "dark",
                Self::Light => "light",
            },
        );
    }
}

static ACTIVE_THEME: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
fn set_active_theme(t: Theme) {
    ACTIVE_THEME.store(
        match t {
            Theme::Dark => 0,
            Theme::Light => 1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

#[derive(Clone, Copy)]
struct Palette {
    bg: egui::Color32,
    bg_row: egui::Color32,
    text: egui::Color32,
    text_dim: egui::Color32,
    warn: egui::Color32,
    slate: egui::Color32,
}
const DARK: Palette = Palette {
    bg: egui::Color32::from_rgb(45, 42, 38),
    bg_row: egui::Color32::from_rgb(55, 50, 46),
    text: egui::Color32::from_rgb(245, 240, 235),
    text_dim: egui::Color32::from_rgb(155, 145, 138),
    warn: egui::Color32::from_rgb(245, 180, 60),
    slate: egui::Color32::from_rgb(150, 145, 138),
};
const LIGHT: Palette = Palette {
    bg: egui::Color32::from_rgb(250, 249, 245),
    bg_row: egui::Color32::from_rgb(241, 236, 224),
    text: egui::Color32::from_rgb(61, 57, 41),
    text_dim: egui::Color32::from_rgb(141, 132, 119),
    warn: egui::Color32::from_rgb(201, 133, 50),
    slate: egui::Color32::from_rgb(180, 172, 158),
};

fn active_palette() -> Palette {
    match ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) {
        1 => LIGHT,
        _ => DARK,
    }
}

fn bg_color() -> egui::Color32 { active_palette().bg }
fn bg_row() -> egui::Color32 { active_palette().bg_row }
fn text_color() -> egui::Color32 { active_palette().text }
fn text_dim() -> egui::Color32 { active_palette().text_dim }
fn amber() -> egui::Color32 { active_palette().warn }
fn slate() -> egui::Color32 { active_palette().slate }
fn teal() -> egui::Color32 { egui::Color32::from_rgb(20, 184, 166) }
fn terracotta() -> egui::Color32 {
    if active_palette().bg == LIGHT.bg {
        egui::Color32::from_rgb(200, 50, 40)
    } else {
        egui::Color32::from_rgb(205, 92, 68)
    }
}

fn status_color(c: &Connection) -> egui::Color32 {
    match c {
        Connection::Disconnected => slate(),
        Connection::Connected => teal(),
        Connection::Paused => amber(),
        Connection::Stopping => terracotta(),
        Connection::Complete => teal(),
    }
}

/// Pulsing LED matching the look of `ticker.rs::draw_led`.
fn draw_led(ui: &mut egui::Ui, color: egui::Color32, radius: f32, pulse: bool) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(radius * 2.5, radius * 2.5),
        egui::Sense::hover(),
    );
    let center = rect.center();
    let alpha = if pulse {
        let t = ui.input(|i| i.time) as f32;
        let wave = (t * 2.0).sin() * 0.2 + 0.8;
        (wave * 255.0) as u8
    } else {
        255
    };
    let c = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
    let glow = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha / 4);
    ui.painter().circle_filled(center, radius * 1.6, glow);
    ui.painter().circle_filled(center, radius, c);
    let highlight = egui::Color32::from_rgba_unmultiplied(255, 255, 255, alpha / 3);
    ui.painter().circle_filled(
        center + egui::vec2(-radius * 0.2, -radius * 0.2),
        radius * 0.35,
        highlight,
    );
}

// =============================================================================
// Snapshot
// =============================================================================

struct Snapshot {
    connection: Connection,
    info: RunInfo,
    epochs: Vec<EpochPoint>,
    log: Vec<(String, String, bool)>,
}

impl SharedState {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            connection: self.connection.clone(),
            info: self.info.clone(),
            epochs: self.epochs.clone(),
            log: self
                .log
                .iter()
                .map(|e| (e.timestamp.clone(), e.message.clone(), e.is_error))
                .collect(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_json_parses() {
        let s = r#"{"model":"Trident-Coder-1b","pid":1234,"output_dir":"/tmp/ckpt","epoch":3,"global_step":500,"last_loss":0.123456,"paused":false,"stopping":false,"started_at":1700000000,"uptime_sec":42,"params":1200000000,"total_epochs":1,"batch_size":4,"monitor":"http://127.0.0.1:34567"}"#;
        let blob = parse_status_json(s).unwrap();
        assert_eq!(blob.model_name, "Trident-Coder-1b");
        assert_eq!(blob.epoch, 3);
        assert_eq!(blob.global_step, 500);
        let info = blob.into_run_info();
        assert_eq!(info.monitor_port, 34567);
        assert_eq!(info.total_epochs, 1);
        assert_eq!(info.param_count, 1_200_000_000);
    }

    #[test]
    fn metrics_json_parses() {
        let body = r#"{
  "model": "X",
  "params": 1000,
  "total_epochs": 3,
  "batch_size": 8,
  "status": "training",
  "best_loss": 0.500000,
  "current_epoch": 2,
  "epochs": [
    {"epoch":1,"train_loss":0.800000,"val_loss":0.900000,"cls":0.100000},
    {"epoch":2,"train_loss":0.500000}
  ]
}"#;
        let pts = parse_metrics_json(body);
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].epoch, 1);
        assert!((pts[0].train_loss - 0.8).abs() < 1e-5);
        assert!(pts[0].val_loss.is_some());
        assert_eq!(pts[0].extras.len(), 1);
        assert_eq!(pts[0].extras[0].0, "cls");
        assert_eq!(pts[1].epoch, 2);
        assert!(pts[1].val_loss.is_none());
    }

    #[test]
    fn dotted_polyline_handles_short_inputs() {
        // Would panic on empty / single points previously; exercise them.
        let ctx = egui::Context::default();
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let rect = ui.max_rect();
                let painter = ui.painter_at(rect);
                draw_dotted_polyline(&painter, &[], teal());
                draw_dotted_polyline(
                    &painter,
                    &[egui::pos2(rect.min.x, rect.min.y)],
                    teal(),
                );
            });
        });
    }
}
