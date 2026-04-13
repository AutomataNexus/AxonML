//! tech-ticker — always-on-top compact widget showing field-tech stats.
//!
//! Per-tech, one row:
//!   - Online LED (tailscale status on Andrew's machine)
//!   - Tech name
//!   - Rate-limit usage bar (X/MAX from server's rate_limits.json)
//!   - Last-diagnostic relative timestamp
//!   - Binary-update-pending indicator (source .exe mtime vs last-pushed cache)
//!
//! Data sources:
//!   1. `tailscale status` — local command, polled every 10s
//!   2. `scp devops@100.67.227.31:~/.nexusoracle/rate_limits.json` — polled every 30s
//!      (uses SSH key auth — same creds we use elsewhere)
//!   3. /var/lib/tailscale-monitor/pushed-<name> — local file mtime, read every 10s
//!
//! Launch: tech-ticker &

use eframe::egui;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

const WIDTH: f32 = 300.0;
const HEIGHT: f32 = 480.0;
const TAILSCALE_POLL_SECS: u64 = 10;
const RATE_LIMITS_POLL_SECS: u64 = 30;

const SERVER_SSH: &str = "devops@100.67.227.31";
const SERVER_HOST: &str = "100.67.227.31";
const SERVER_USER: &str = "devops";
const SERVER_RATE_LIMITS_PATH: &str = "/home/devops/.nexusoracle/rate_limits.json";
const LOCAL_CACHE_COPY: &str = "/tmp/.tech-ticker-rate-limits.json";
const MONITOR_CACHE_DIR: &str = "/var/lib/tailscale-monitor";
const MONITOR_CACHE_DIR_ALT: &str = "/home/devops/.tailscale-monitor";
const SRC_EXE: &str = "/opt/NexusOracle/apps/oracle-chat/src-tauri/target/x86_64-pc-windows-gnu/release/oracle-chat-tauri.exe";
const RESTART_SCRIPT: &str = "/opt/NexusOracle/restartoracle.sh";

/// Daemon base URL — local to the AN server over Tailscale.
const DAEMON_URL: &str = "http://100.67.227.31:4780";

/// Server-side relay (tailscale-relay) — pushes + event log + push state.
const RELAY_URL: &str = "http://100.67.227.31:4790";
const RELAY_POLL_SECS: u64 = 10;
/// Max events kept in memory for the dropdown panel.
const EVENTS_IN_MEMORY: usize = 200;
/// Stored between runs so we can compute "while you were away".
const LAST_SEEN_FILE: &str = "/tmp/.tech-ticker-last-seen";
/// Owner API key used by the ticker to hit owner-only endpoints.
/// Same key Andrew uses; owners bypass rate limits.
const OWNER_API_KEY: &str = "qUUlVoumbb3xXQFqv3jNdr-8wAiDt_oT6IYiw6D6ogw";

/// Techs to show, in display order.
/// Format: (display_name, tailscale_host, wsl_ssh_user)
///
/// Owners (Andrew, DevOps) are deliberately excluded — they bypass the rate
/// limiter so their row would always be empty.
///
/// `wsl_ssh_user` is used when the "SSH" button opens a Windows Terminal tab
/// into the tech's WSL — empty string disables the button for that row
/// (techs not on Tailscale yet).
const TECHS: &[(&str, &str, &str)] = &[
    ("Nick",   "nick",     "getac"),
    ("Leon",   "leon-wsl", "ulvenr"),
    ("John",   "john",     ""),
    ("Keenan", "keenan",   ""),
    ("Denior", "denior",   ""),
];

// ─────────────────────────────────────────────────────────────────────────────
// NexusStratum palette
// ─────────────────────────────────────────────────────────────────────────────

const CREAM: egui::Color32 = egui::Color32::from_rgb(245, 240, 235);
const TEAL: egui::Color32 = egui::Color32::from_rgb(20, 184, 166);
const TERRACOTTA: egui::Color32 = egui::Color32::from_rgb(205, 92, 68);
const AMBER: egui::Color32 = egui::Color32::from_rgb(245, 180, 60);
const SLATE: egui::Color32 = egui::Color32::from_rgb(150, 145, 138);
const BG_DARK: egui::Color32 = egui::Color32::from_rgb(45, 42, 38);
const BG_ROW: egui::Color32 = egui::Color32::from_rgb(55, 50, 46);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(155, 145, 138);

// ─────────────────────────────────────────────────────────────────────────────
// Entry
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([WIDTH, HEIGHT])
            .with_min_inner_size([WIDTH, 120.0])
            .with_always_on_top()
            .with_decorations(true)
            .with_transparent(false)
            .with_title("tech-monitor"),
        ..Default::default()
    };
    eframe::run_native(
        "tech-ticker",
        options,
        Box::new(|cc| Ok(Box::new(TechApp::new(cc)))),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Default)]
struct TechState {
    online: bool,
    /// Number of diagnostic timestamps still inside the window.
    used: usize,
    /// Configured max from the server (defaults to 3 if unknown).
    max: usize,
    /// Window length in hours (for the "Xh" label).
    window_hours: u64,
    /// Most-recent diagnostic timestamp, shown as relative.
    last_request: Option<chrono::DateTime<chrono::Utc>>,
    /// True when the source .exe on Andrew's machine is newer than what
    /// we've pushed to this tech — i.e. an update is waiting for them.
    update_pending: bool,
}

/// Transient UI feedback for a running action (Reset / Build).
#[derive(Clone, Default)]
struct ActionState {
    /// Short label shown while the action is running (e.g. "building...").
    in_flight: Option<String>,
    /// Last action result, shown briefly in the header.
    last_result: Option<(String, bool)>, // (msg, is_error)
    last_result_at: Option<std::time::Instant>,
}

/// Mirrors the relay's /push-state JSON.
#[derive(Clone, Default, Deserialize)]
struct RelayPushState {
    #[serde(default)]
    staged_build_mtime: u64,
    #[serde(default)]
    techs: HashMap<String, RelayTechState>,
}
#[derive(Clone, Default, Deserialize)]
struct RelayTechState {
    #[serde(default)]
    last_pushed_build_mtime: u64,
    #[serde(default)]
    last_pushed_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    online: bool,
}

/// One event pulled from the relay's /events log (JSONL).
#[derive(Clone, Deserialize)]
#[serde(tag = "kind")]
#[serde(rename_all = "snake_case")]
enum RelayEvent {
    RelayStarted    { ts: chrono::DateTime<chrono::Utc>, #[serde(default)] pid: u32 },
    TechOnline      { ts: chrono::DateTime<chrono::Utc>, tech: String, #[serde(default)] host: String },
    TechOffline     { ts: chrono::DateTime<chrono::Utc>, tech: String, #[serde(default)] host: String },
    PushOk          { ts: chrono::DateTime<chrono::Utc>, tech: String, #[serde(default)] build_ts: Option<chrono::DateTime<chrono::Utc>> },
    PushFailed      { ts: chrono::DateTime<chrono::Utc>, tech: String, #[serde(default)] error: String },
    RateLimitHit    { ts: chrono::DateTime<chrono::Utc>, #[serde(default)] log: String },
    RateLimitReset  { ts: chrono::DateTime<chrono::Utc>, #[serde(default)] log: String },
    AuthError       { ts: chrono::DateTime<chrono::Utc>, #[serde(default)] log: String },
    DiagnosticSession { ts: chrono::DateTime<chrono::Utc>, equipment: String, #[serde(default)] file: String },
    #[serde(other)]
    Other,
}

impl RelayEvent {
    fn ts(&self) -> chrono::DateTime<chrono::Utc> {
        match self {
            RelayEvent::RelayStarted { ts, .. }
            | RelayEvent::TechOnline { ts, .. }
            | RelayEvent::TechOffline { ts, .. }
            | RelayEvent::PushOk { ts, .. }
            | RelayEvent::PushFailed { ts, .. }
            | RelayEvent::RateLimitHit { ts, .. }
            | RelayEvent::RateLimitReset { ts, .. }
            | RelayEvent::AuthError { ts, .. }
            | RelayEvent::DiagnosticSession { ts, .. } => *ts,
            RelayEvent::Other => chrono::Utc::now(),
        }
    }
    fn summary(&self) -> String {
        match self {
            RelayEvent::RelayStarted { .. } => "relay started".into(),
            RelayEvent::TechOnline { tech, .. } => format!("{tech} online"),
            RelayEvent::TechOffline { tech, .. } => format!("{tech} offline"),
            RelayEvent::PushOk { tech, .. } => format!("pushed → {tech}"),
            RelayEvent::PushFailed { tech, error, .. } => format!("push {tech} FAILED: {error}"),
            RelayEvent::RateLimitHit { .. } => "rate-limit hit".into(),
            RelayEvent::RateLimitReset { .. } => "rate-limit reset".into(),
            RelayEvent::AuthError { .. } => "auth error".into(),
            RelayEvent::DiagnosticSession { equipment, .. } => format!("session: {equipment}"),
            RelayEvent::Other => "?".into(),
        }
    }
    fn is_alert(&self) -> bool {
        matches!(self, RelayEvent::PushFailed { .. } | RelayEvent::AuthError { .. })
    }
}

struct Shared {
    techs: HashMap<String, TechState>,
    last_rate_limits_fetch: Option<chrono::DateTime<chrono::Local>>,
    last_rate_limits_error: Option<String>,
    action: ActionState,
    /// Recent events pulled from the relay (newest last), capped at
    /// EVENTS_IN_MEMORY.
    events: Vec<RelayEvent>,
    /// Timestamp of the newest event we've "acknowledged" (persists to
    /// LAST_SEEN_FILE). Events newer than this count toward the banner.
    last_seen: chrono::DateTime<chrono::Utc>,
    /// Relay /push-state last successful fetch + error string.
    relay_state: Option<RelayPushState>,
    last_relay_error: Option<String>,
    /// Show or hide the events dropdown below the header.
    show_events: bool,
    /// Per-tech disable state, keyed lowercase — mirrors the daemon's
    /// ~/.nexusoracle/tech_overrides.json. Populated by poll_tech_overrides.
    disabled_techs: HashMap<String, String>, // name-lowercase → reason
}

#[derive(Clone, Default, Deserialize)]
struct RemoteOverride {
    #[serde(default)]
    disabled: bool,
    #[serde(default)]
    reason: String,
}

struct TechApp {
    state: Arc<Mutex<Shared>>,
    runtime: tokio::runtime::Runtime,
}

impl TechApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let techs: HashMap<String, TechState> = TECHS
            .iter()
            .map(|(name, _, _)| (name.to_string(), TechState::default()))
            .collect();
        // Restore last_seen from disk so the banner only shows events
        // Andrew hasn't acknowledged yet, even after a reboot.
        let last_seen = std::fs::read_to_string(LAST_SEEN_FILE)
            .ok()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s.trim()).ok())
            .map(|d| d.with_timezone(&chrono::Utc))
            .unwrap_or_else(|| chrono::Utc::now() - chrono::Duration::days(7));

        let state = Arc::new(Mutex::new(Shared {
            techs,
            last_rate_limits_fetch: None,
            last_rate_limits_error: None,
            action: ActionState::default(),
            events: Vec::new(),
            last_seen,
            relay_state: None,
            last_relay_error: None,
            show_events: false,
        }));

        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");

        // Tailscale poller (local fallback — relay is authoritative when
        // reachable, this catches the case where the relay is down).
        let s1 = state.clone();
        runtime.spawn(async move {
            loop {
                refresh_tailscale(&s1).await;
                refresh_update_pending(&s1).await;
                tokio::time::sleep(std::time::Duration::from_secs(TAILSCALE_POLL_SECS)).await;
            }
        });

        // Rate-limits poller (daemon-owned data not in the relay)
        let s2 = state.clone();
        runtime.spawn(async move {
            loop {
                refresh_rate_limits(&s2).await;
                tokio::time::sleep(std::time::Duration::from_secs(RATE_LIMITS_POLL_SECS))
                    .await;
            }
        });

        // Relay poller — events + push state
        let s3 = state.clone();
        runtime.spawn(async move {
            loop {
                refresh_relay(&s3).await;
                tokio::time::sleep(std::time::Duration::from_secs(RELAY_POLL_SECS)).await;
            }
        });

        Self { state, runtime }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data refresh
// ─────────────────────────────────────────────────────────────────────────────

async fn refresh_tailscale(state: &Arc<Mutex<Shared>>) {
    let output = tokio::process::Command::new("tailscale")
        .arg("status")
        .output()
        .await;
    let Ok(out) = output else { return };
    let stdout = String::from_utf8_lossy(&out.stdout);

    let Ok(mut s) = state.lock() else { return };
    for (name, host, _wsl) in TECHS {
        let online = stdout
            .lines()
            .filter(|l| {
                let cols: Vec<&str> = l.split_whitespace().collect();
                cols.len() >= 2 && cols[1].eq_ignore_ascii_case(host)
            })
            .any(|l| !l.contains("offline"));
        if let Some(t) = s.techs.get_mut(*name) {
            t.online = online;
        }
    }
}

async fn refresh_update_pending(state: &Arc<Mutex<Shared>>) {
    let src_mtime = match std::fs::metadata(SRC_EXE).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return,
    };
    let Ok(mut s) = state.lock() else { return };
    for (name, _host, _wsl) in TECHS {
        // Cache lives in one of two places depending on permissions
        let c1 = format!("{MONITOR_CACHE_DIR}/pushed-{}", name.to_lowercase());
        let c2 = format!("{MONITOR_CACHE_DIR_ALT}/pushed-{}", name.to_lowercase());
        let cache_path = if std::path::Path::new(&c1).exists() {
            c1
        } else {
            c2
        };
        let last_pushed = std::fs::read_to_string(&cache_path)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        let pending = match last_pushed {
            None => true, // never pushed
            Some(pushed_unix) => {
                let src_unix = src_mtime
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                src_unix > pushed_unix
            }
        };
        if let Some(t) = s.techs.get_mut(*name) {
            t.update_pending = pending;
        }
    }
}

#[derive(Deserialize)]
struct RateLimitsRaw(HashMap<String, Vec<String>>);

async fn refresh_relay(state: &Arc<Mutex<Shared>>) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    // /push-state — populates per-tech online + last-pushed-at
    let ps = client
        .get(format!("{RELAY_URL}/push-state"))
        .send()
        .await;
    let ps_result: Result<RelayPushState, String> = match ps {
        Ok(r) if r.status().is_success() => r
            .json::<RelayPushState>()
            .await
            .map_err(|e| e.to_string()),
        Ok(r) => Err(format!("HTTP {}", r.status())),
        Err(e) => Err(e.to_string()),
    };

    // /events — newline-delimited JSON events
    let ev = client.get(format!("{RELAY_URL}/events")).send().await;
    let events_result: Result<Vec<RelayEvent>, String> = match ev {
        Ok(r) if r.status().is_success() => r
            .text()
            .await
            .map_err(|e| e.to_string())
            .map(|body| {
                body.lines()
                    .filter(|l| !l.is_empty())
                    .filter_map(|l| serde_json::from_str::<RelayEvent>(l).ok())
                    .collect()
            }),
        Ok(r) => Err(format!("HTTP {}", r.status())),
        Err(e) => Err(e.to_string()),
    };

    let Ok(mut s) = state.lock() else { return };
    match (ps_result, events_result) {
        (Ok(ps), Ok(mut events)) => {
            s.last_relay_error = None;

            // Apply relay's online/update-pending view on top of whatever
            // the local poller filled in — relay is authoritative.
            for (name, _host, _wsl) in TECHS {
                let rs = ps.techs.get(*name).cloned().unwrap_or_default();
                if let Some(t) = s.techs.get_mut(*name) {
                    t.online = rs.online;
                    t.update_pending = rs.last_pushed_build_mtime < ps.staged_build_mtime;
                }
            }
            s.relay_state = Some(ps);

            // Cap event history so memory doesn't grow forever.
            if events.len() > EVENTS_IN_MEMORY {
                let start = events.len() - EVENTS_IN_MEMORY;
                events.drain(..start);
            }
            s.events = events;
        }
        (ps_res, ev_res) => {
            let err = ps_res
                .err()
                .or(ev_res.err())
                .unwrap_or_else(|| "relay error".into());
            s.last_relay_error = Some(err);
        }
    }
}

/// Write last_seen to disk so the banner stays correct across restarts.
fn persist_last_seen(ts: chrono::DateTime<chrono::Utc>) {
    let _ = std::fs::write(LAST_SEEN_FILE, ts.to_rfc3339());
}

async fn refresh_rate_limits(state: &Arc<Mutex<Shared>>) {
    // scp the JSON down (SSH key auth — same as used elsewhere in this
    // workspace). Fallback: if scp fails, we leave the data stale but record
    // the error for display.
    let scp = tokio::process::Command::new("scp")
        .args([
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            &format!("{SERVER_SSH}:{SERVER_RATE_LIMITS_PATH}"),
            LOCAL_CACHE_COPY,
        ])
        .output()
        .await;

    // Treat "file doesn't exist yet" as a clean zero-state, not an error —
    // the daemon only writes rate_limits.json after the first rate-limited
    // request, so an empty state is expected on a fresh install.
    let scp_err = match scp {
        Ok(o) if o.status.success() => None,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let first_line = stderr.lines().next().unwrap_or("scp failed").trim();
            let missing = stderr.to_lowercase().contains("no such file")
                || stderr.to_lowercase().contains("does not exist");
            if missing {
                None
            } else {
                Some(first_line.to_string())
            }
        }
        Err(e) => Some(e.to_string()),
    };

    let json_result = std::fs::read_to_string(LOCAL_CACHE_COPY)
        .ok()
        .and_then(|raw| serde_json::from_str::<HashMap<String, Vec<String>>>(&raw).ok());

    // Also fetch daemon config for each tech's max/window, via ssh cat.
    let cfg_out = tokio::process::Command::new("ssh")
        .args([
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            SERVER_SSH,
            "cat ~/.nexusoracle/config.toml",
        ])
        .output()
        .await;
    let cfg_text = cfg_out
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let overrides = parse_tech_overrides(&cfg_text);

    let Ok(mut s) = state.lock() else { return };
    s.last_rate_limits_fetch = Some(chrono::Local::now());
    s.last_rate_limits_error = scp_err;

    let now = chrono::Utc::now();

    for (name, _host, _wsl) in TECHS {
        let t = match s.techs.get_mut(*name) {
            Some(t) => t,
            None => continue,
        };

        let (max, window_hours) = overrides
            .get(*name)
            .copied()
            .unwrap_or((3, 4));
        t.max = max;
        t.window_hours = window_hours;

        if let Some(ref timestamps) = json_result {
            let cutoff = now - chrono::Duration::hours(window_hours as i64);
            let parsed: Vec<chrono::DateTime<chrono::Utc>> = timestamps
                .get(*name)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|d| d.with_timezone(&chrono::Utc))
                .filter(|d| *d > cutoff)
                .collect();
            t.used = parsed.len();
            t.last_request = parsed.into_iter().max();
        }
    }
}

/// Very loose TOML parser for `[[tech_access.api_keys]]` blocks to extract
/// per-tech max_diagnostics + window_hours. Avoids pulling in a full TOML
/// dependency for the ticker.
fn parse_tech_overrides(cfg: &str) -> HashMap<String, (usize, u64)> {
    let mut out = HashMap::new();
    let mut current_name: Option<String> = None;
    let mut current_max: Option<usize> = None;
    let mut current_window: Option<u64> = None;

    for line in cfg.lines() {
        let l = line.trim();
        if l == "[[tech_access.api_keys]]" {
            if let Some(n) = current_name.take() {
                out.insert(n, (current_max.unwrap_or(3), current_window.unwrap_or(4)));
            }
            current_max = None;
            current_window = None;
            continue;
        }
        if let Some(v) = l.strip_prefix("name = ") {
            let v = v.trim().trim_matches('"');
            current_name = Some(v.to_string());
        } else if let Some(v) = l.strip_prefix("max_diagnostics = ") {
            current_max = v.trim().parse().ok();
        } else if let Some(v) = l.strip_prefix("window_hours = ") {
            current_window = v.trim().parse().ok();
        } else if l.starts_with('[') && !l.starts_with("[[tech_access") {
            // Entered a different section — flush the last api_keys block.
            if let Some(n) = current_name.take() {
                out.insert(n, (current_max.unwrap_or(3), current_window.unwrap_or(4)));
            }
            current_max = None;
            current_window = None;
        }
    }
    // Flush trailing block.
    if let Some(n) = current_name.take() {
        out.insert(n, (current_max.unwrap_or(3), current_window.unwrap_or(4)));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────────────────────────────────

impl eframe::App for TechApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint for LED pulse
        ctx.request_repaint_after(std::time::Duration::from_millis(80));

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = BG_DARK;
        visuals.window_fill = BG_DARK;
        visuals.override_text_color = Some(CREAM);
        ctx.set_visuals(visuals);

        let snapshot: Vec<(String, &'static str, &'static str, TechState)> = {
            let s = self.state.lock().unwrap();
            TECHS
                .iter()
                .map(|(name, host, wsl)| {
                    let t = s.techs.get(*name).cloned().unwrap_or_default();
                    (name.to_string(), *host, *wsl, t)
                })
                .collect()
        };
        let (last_fetch, last_err, action, events, last_seen, relay_err, show_events) = {
            let s = self.state.lock().unwrap();
            (
                s.last_rate_limits_fetch,
                s.last_rate_limits_error.clone(),
                s.action.clone(),
                s.events.clone(),
                s.last_seen,
                s.last_relay_error.clone(),
                s.show_events,
            )
        };
        let new_events: Vec<RelayEvent> = events
            .iter()
            .filter(|e| e.ts() > last_seen)
            .cloned()
            .collect();
        let new_count = new_events.len();

        // Expire stale "last_result" after 8s so transient status messages clear.
        if let Some(t) = action.last_result_at {
            if t.elapsed() > std::time::Duration::from_secs(8) {
                if let Ok(mut s) = self.state.lock() {
                    s.action.last_result = None;
                    s.action.last_result_at = None;
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Header row
            ui.horizontal(|ui| {
                draw_led(ui, TEAL, 5.0, true);
                ui.label(
                    egui::RichText::new("TECH MONITOR")
                        .strong()
                        .size(11.0)
                        .color(CREAM),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // "Build" button (rebuild exe + SIGUSR1 monitor)
                    let building = action.in_flight.as_deref() == Some("build");
                    let btn_label = if building { "building..." } else { "Build" };
                    let btn = egui::Button::new(
                        egui::RichText::new(btn_label).size(9.0).color(CREAM),
                    )
                    .fill(if building { AMBER.linear_multiply(0.5) } else { TEAL.linear_multiply(0.25) })
                    .stroke(egui::Stroke::new(1.0, TEAL))
                    .rounding(3.0)
                    .min_size(egui::vec2(54.0, 18.0));
                    if ui.add_enabled(!building, btn).on_hover_text(
                        "Run restartoracle.sh: rebuild the .exe then trigger an immediate push to any online tech",
                    ).clicked() {
                        self.trigger_build();
                    }
                    ui.add_space(4.0);

                    if last_err.is_some() {
                        ui.label(
                            egui::RichText::new("⚠ sync")
                                .size(9.0)
                                .color(TERRACOTTA),
                        )
                        .on_hover_text(last_err.clone().unwrap_or_default());
                    } else if let Some(t) = last_fetch {
                        ui.label(
                            egui::RichText::new(t.format("%H:%M").to_string())
                                .size(9.0)
                                .color(TEXT_DIM),
                        );
                    }
                });
            });

            // Second header line: current action status / result (very subtle)
            if let Some(msg) = &action.in_flight {
                ui.label(
                    egui::RichText::new(format!("⚙ {msg}"))
                        .size(9.0)
                        .color(AMBER),
                );
            } else if let Some((msg, is_err)) = &action.last_result {
                let color = if *is_err { TERRACOTTA } else { TEAL };
                ui.label(egui::RichText::new(msg).size(9.0).color(color));
            }

            // ── "While you were away" banner + collapsible events panel ──
            // Always show a small events toggle; when new events exist, it
            // gets highlighted in amber with the count.
            ui.horizontal(|ui| {
                let oldest_new = new_events.iter().map(|e| e.ts()).min();
                let (label, color) = if new_count > 0 {
                    let since = oldest_new
                        .map(|t| t.with_timezone(&chrono::Local).format("%H:%M").to_string())
                        .unwrap_or_default();
                    (
                        format!("⚠ {new_count} events since {since}"),
                        AMBER,
                    )
                } else if relay_err.is_some() {
                    ("⚠ relay offline".to_string(), TERRACOTTA)
                } else {
                    (format!("{} events", events.len()), TEXT_DIM)
                };
                let btn = egui::Button::new(
                    egui::RichText::new(&label).size(9.0).color(color),
                )
                .frame(false);
                if ui.add(btn)
                    .on_hover_text(
                        relay_err.clone()
                            .map(|e| format!("relay error: {e}"))
                            .unwrap_or_else(|| "click to expand event log".into())
                    )
                    .clicked()
                {
                    if let Ok(mut s) = self.state.lock() {
                        s.show_events = !s.show_events;
                    }
                }
                if new_count > 0 {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(
                            egui::Button::new(
                                egui::RichText::new("dismiss").size(9.0).color(TEAL),
                            )
                            .frame(false),
                        )
                        .on_hover_text("mark all events as seen")
                        .clicked()
                        {
                            let newest = events.iter().map(|e| e.ts()).max();
                            if let Some(ts) = newest {
                                if let Ok(mut s) = self.state.lock() {
                                    s.last_seen = ts;
                                }
                                persist_last_seen(ts);
                            }
                        }
                    });
                }
            });

            if show_events {
                egui::ScrollArea::vertical()
                    .max_height(100.0)
                    .show(ui, |ui| {
                        for ev in events.iter().rev().take(30) {
                            let ts_local =
                                ev.ts().with_timezone(&chrono::Local).format("%H:%M");
                            let color = if ev.is_alert() { TERRACOTTA } else if ev.ts() > last_seen { AMBER } else { TEXT_DIM };
                            ui.label(
                                egui::RichText::new(format!("{ts_local}  {}", ev.summary()))
                                    .size(9.0)
                                    .color(color),
                            );
                        }
                    });
            }

            ui.separator();

            // Per-tech rows
            for (name, _host, wsl, t) in &snapshot {
                self.draw_tech_row(ui, name, wsl, t);
            }

            // Footer: legend
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                draw_led(ui, TEAL, 3.0, false);
                ui.label(egui::RichText::new("online").size(8.0).color(TEXT_DIM));
                ui.add_space(4.0);
                draw_led(ui, SLATE, 3.0, false);
                ui.label(egui::RichText::new("offline").size(8.0).color(TEXT_DIM));
                ui.add_space(4.0);
                draw_led(ui, AMBER, 3.0, true);
                ui.label(
                    egui::RichText::new("update pending")
                        .size(8.0)
                        .color(TEXT_DIM),
                );
            });
        });

        // Keep the runtime alive
        let _ = &self.runtime;
    }
}

impl TechApp {
    fn draw_tech_row(
        &self,
        ui: &mut egui::Ui,
        name: &str,
        wsl_user: &str,
        t: &TechState,
    ) {
        let usage_color = if t.max > 0 && t.used >= t.max {
            TERRACOTTA
        } else if t.max > 0 && t.used as f32 / t.max as f32 >= 0.66 {
            AMBER
        } else {
            TEAL
        };

        egui::Frame::none()
            .fill(BG_ROW)
            .rounding(4.0)
            .inner_margin(egui::Margin::symmetric(8.0, 6.0))
            .show(ui, |ui| {
                // Line 1: LED + name + [update-LED] [SSH] [Reset] ...spacer... usage
                ui.horizontal(|ui| {
                    let color = if t.online { TEAL } else { SLATE };
                    draw_led(ui, color, 5.0, t.online);
                    ui.label(
                        egui::RichText::new(name)
                            .strong()
                            .size(12.0)
                            .color(CREAM),
                    );
                    if t.update_pending {
                        ui.add_space(2.0);
                        draw_led(ui, AMBER, 3.5, true);
                    }

                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            // Usage counter (always)
                            ui.label(
                                egui::RichText::new(format!("{}/{}", t.used, t.max))
                                    .strong()
                                    .size(13.0)
                                    .color(usage_color),
                            );

                            ui.add_space(4.0);

                            // Reset button — always shown
                            let reset_btn = egui::Button::new(
                                egui::RichText::new("↺").size(11.0).color(TEAL),
                            )
                            .frame(false)
                            .min_size(egui::vec2(18.0, 18.0));
                            if ui.add(reset_btn)
                                .on_hover_text(format!(
                                    "Clear {name}'s rate-limit counter on the daemon"
                                ))
                                .clicked()
                            {
                                self.trigger_reset(name.to_string());
                            }

                            // SSH button — only if tech has a known WSL user + is online
                            let ssh_enabled = !wsl_user.is_empty() && t.online;
                            let ssh_color = if ssh_enabled { TEAL } else { SLATE };
                            let ssh_btn = egui::Button::new(
                                egui::RichText::new("⌨").size(11.0).color(ssh_color),
                            )
                            .frame(false)
                            .min_size(egui::vec2(18.0, 18.0));
                            let ssh_resp = ui.add_enabled(ssh_enabled, ssh_btn);
                            let hover = if wsl_user.is_empty() {
                                format!("{name}: not on Tailscale yet")
                            } else if !t.online {
                                format!("{name} is offline")
                            } else {
                                format!("SSH to {wsl_user}@{name} in a new terminal tab")
                            };
                            if ssh_resp.on_hover_text(hover).clicked() {
                                self.trigger_ssh(name.to_string(), wsl_user.to_string());
                            }
                        },
                    );
                });

                // Line 2: last-used (prominent) ...spacer... window
                ui.horizontal(|ui| {
                    let (last_label, last_color) = match t.last_request {
                        Some(ts) => (format!("last used {}", rel_time(ts)), CREAM),
                        None => ("never used".to_string(), TEXT_DIM),
                    };
                    ui.label(
                        egui::RichText::new(last_label)
                            .size(10.0)
                            .color(last_color),
                    );
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            ui.label(
                                egui::RichText::new(format!("{}h window", t.window_hours))
                                    .size(9.0)
                                    .color(TEXT_DIM),
                            );
                        },
                    );
                });

                // Line 3: thin usage bar
                let (bar_rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 2.5),
                    egui::Sense::hover(),
                );
                ui.painter().rect_filled(
                    bar_rect,
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(90, 85, 80, 200),
                );
                if t.max > 0 {
                    let frac = (t.used as f32 / t.max as f32).clamp(0.0, 1.0);
                    let fill_rect = egui::Rect::from_min_size(
                        bar_rect.min,
                        egui::vec2(bar_rect.width() * frac, bar_rect.height()),
                    );
                    ui.painter().rect_filled(fill_rect, 1.0, usage_color);
                }
            });

        ui.add_space(3.0);
    }

    // ── Actions ──────────────────────────────────────────────────────────────

    fn trigger_reset(&self, tech_name: String) {
        let state = self.state.clone();
        if let Ok(mut s) = state.lock() {
            s.action.in_flight = Some(format!("reset {tech_name}"));
        }
        self.runtime.spawn(async move {
            let client = reqwest::Client::new();
            let res = client
                .post(format!("{DAEMON_URL}/api/v1/rate-limits/reset"))
                .header("Authorization", format!("Bearer {OWNER_API_KEY}"))
                .json(&serde_json::json!({ "tech_name": tech_name }))
                .timeout(std::time::Duration::from_secs(10))
                .send()
                .await;
            if let Ok(mut s) = state.lock() {
                s.action.in_flight = None;
                s.action.last_result = Some(match res {
                    Ok(r) if r.status().is_success() => (
                        format!("reset {tech_name}: OK"),
                        false,
                    ),
                    Ok(r) => (format!("reset {tech_name} → HTTP {}", r.status()), true),
                    Err(e) => (format!("reset {tech_name}: {e}"), true),
                });
                s.action.last_result_at = Some(std::time::Instant::now());
            }
        });
    }

    fn trigger_ssh(&self, tech_name: String, wsl_user: String) {
        // Spawn a new Windows Terminal tab that SSHes into the tech's WSL via
        // Tailscale, using sshpass for non-interactive auth (same password we
        // use throughout the workspace for tech WSLs).
        let password = "Invertedskynet2$";
        let target = format!("{wsl_user}@{tech_name}");
        let inner = format!(
            "sshpass -p '{password}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {target}"
        );
        let _ = std::process::Command::new("wt.exe")
            .args([
                "-w",
                "0",
                "new-tab",
                "--title",
                &format!("ssh {tech_name}"),
                "wsl.exe",
                "-d",
                "Ubuntu",
                "-e",
                "bash",
                "-lc",
                &inner,
            ])
            .spawn();
    }

    fn trigger_build(&self) {
        let state = self.state.clone();
        if let Ok(mut s) = state.lock() {
            s.action.in_flight = Some("build".to_string());
        }
        self.runtime.spawn(async move {
            let out = tokio::process::Command::new("bash")
                .arg(RESTART_SCRIPT)
                .output()
                .await;
            if let Ok(mut s) = state.lock() {
                s.action.in_flight = None;
                s.action.last_result = Some(match out {
                    Ok(o) if o.status.success() => ("build OK — pushing to online techs".to_string(), false),
                    Ok(o) => {
                        let stderr = String::from_utf8_lossy(&o.stderr);
                        let first = stderr.lines().filter(|l| !l.is_empty()).next().unwrap_or("build failed");
                        (format!("build: {first}"), true)
                    }
                    Err(e) => (format!("build spawn failed: {e}"), true),
                });
                s.action.last_result_at = Some(std::time::Instant::now());
            }
        });
    }
}

/// Relative "Nh ago" / "Nm ago" string.
fn rel_time(when: chrono::DateTime<chrono::Utc>) -> String {
    let delta = chrono::Utc::now() - when;
    let total = delta.num_seconds();
    if total < 60 {
        format!("{}s ago", total.max(0))
    } else if total < 3600 {
        format!("{}m ago", total / 60)
    } else if total < 86400 {
        format!("{}h ago", total / 3600)
    } else {
        format!("{}d ago", total / 86400)
    }
}

fn draw_led(ui: &mut egui::Ui, color: egui::Color32, radius: f32, pulse: bool) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(radius * 2.5, radius * 2.5), egui::Sense::hover());
    let center = rect.center();
    let alpha = if pulse {
        let t = ui.input(|i| i.time) as f32;
        let wave = (t * 2.0).sin() * 0.2 + 0.8;
        (wave * 255.0) as u8
    } else {
        220
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
