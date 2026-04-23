//! tech-ticker — Always-On-Top Field-Tech Monitor
//!
//! eframe-based compact widget showing per-tech status on one row each:
//! online LED (driven by `tailscale status` + the relay's authoritative
//! `/push-state`), tech name, rate-limit usage counter `X/MAX` from the
//! server's `rate_limits.json`, last-diagnostic relative timestamp, a
//! binary-update-pending amber LED, and per-tech action buttons (Reset
//! rate-limit counter, SSH to tech's WSL via Windows Terminal, Disable /
//! Enable tech-access).
//!
//! Four background pollers via tokio:
//! * `refresh_tailscale` — runs `tailscale status` every 10s (local fallback
//!   for relay outage).
//! * `refresh_rate_limits` — pulls `rate_limits.json` + TOML `daemon_config`
//!   from the relay every 30s, parses `[[tech_access.api_keys]]` blocks via
//!   `parse_tech_overrides` to get per-tech `max_diagnostics` and
//!   `window_hours`, then filters timestamps inside the window.
//! * `refresh_relay` — every 10s pulls `/push-state` (authoritative online +
//!   last-pushed-build-mtime) and `/events` JSONL into the
//!   `RelayEvent` enum (`TechOnline`, `TechOffline`, `PushOk`, `PushFailed`,
//!   `RateLimitHit`/`Reset`, `AuthError`, `DiagnosticSession`, etc.).
//! * `refresh_tech_overrides` — every 30s hits
//!   `/api/v1/tech-access/overrides` to drive the Disable/Enable button
//!   glyph.
//!
//! UI paints a rounded transparent outer frame with custom titlebar (drag,
//! close, minimize), action header (theme toggle, Build button that runs
//! `/opt/NexusOracle/restartoracle.sh`), "while you were away" events
//! banner with collapsible log panel (new_count + `last_seen` persisted to
//! `/tmp/.tech-ticker-last-seen`), per-tech rows via `draw_tech_row`, and
//! a legend footer. Theme system (`Theme::Dark` / `Theme::Light`) persists
//! to `/tmp/.tech-ticker-theme`.
//!
//! Trigger actions: `trigger_toggle_disable` POSTs to the daemon, optimistically
//! updating `disabled_techs` so the UI flips immediately; `trigger_reset`
//! POSTs to `/api/v1/rate-limits/reset`; `trigger_ssh` spawns `wt.exe` +
//! `sshpass`; `trigger_build` runs `RESTART_SCRIPT`.
//!
//! Launch: tech-ticker &
//!
//! # File
//! `nexus-agent/src/ui/tech_ticker.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use eframe::egui;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// =============================================================================
// Config — Constants, Endpoints, Techs
// =============================================================================

const WIDTH: f32 = 300.0;
const HEIGHT: f32 = 480.0;
const TAILSCALE_POLL_SECS: u64 = 10;
const RATE_LIMITS_POLL_SECS: u64 = 30;

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
    ("John",   "john",     "jdtur"),
    ("Keenan", "keenan",   ""),
    ("Denior", "denior",   ""),
];

// =============================================================================
// Theme — Dark (NexusStratum) + Light (Claude Browser UI)
// =============================================================================

/// Which theme the ticker is currently rendering in. Persisted so it sticks
/// across restarts.
#[derive(Clone, Copy, PartialEq)]
enum Theme {
    Dark,
    Light,
}

const THEME_FILE: &str = "/tmp/.tech-ticker-theme";

impl Theme {
    fn load() -> Self {
        match std::fs::read_to_string(THEME_FILE).ok().as_deref().map(str::trim) {
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

/// A palette is just a bundle of named colors. We keep a static Dark one
/// and a Light one, then swap by which `Palette` we hand to draw helpers.
#[derive(Clone, Copy)]
struct Palette {
    bg:       egui::Color32,
    bg_row:   egui::Color32,
    text:     egui::Color32,
    text_dim: egui::Color32,
    accent:   egui::Color32,   // "good" / primary action (teal dark, claude-coral light)
    warn:     egui::Color32,   // amber — "close to limit" / unacknowledged events
    alert:    egui::Color32,   // terracotta — errors / disabled / over-limit
    slate:    egui::Color32,   // dim/offline/unknown
}

// ── Dark theme (original NexusStratum) ──────────────────────────────────────
const DARK: Palette = Palette {
    bg:       egui::Color32::from_rgb(45, 42, 38),     // #2D2A26
    bg_row:   egui::Color32::from_rgb(55, 50, 46),
    text:     egui::Color32::from_rgb(245, 240, 235),
    text_dim: egui::Color32::from_rgb(155, 145, 138),
    accent:   egui::Color32::from_rgb(20, 184, 166),   // teal
    warn:     egui::Color32::from_rgb(245, 180, 60),   // amber
    alert:    egui::Color32::from_rgb(205, 92, 68),    // terracotta
    slate:    egui::Color32::from_rgb(150, 145, 138),
};

// ── Light theme — Claude browser UI palette ─────────────────────────────────
const LIGHT: Palette = Palette {
    bg:       egui::Color32::from_rgb(250, 249, 245),  // #faf9f5 warm cream
    bg_row:   egui::Color32::from_rgb(237, 231, 216),  // #ede7d8 warmer, reads as a card
    text:     egui::Color32::from_rgb(61, 57, 41),     // #3d3929 warm brown
    text_dim: egui::Color32::from_rgb(141, 132, 119),  // #8d8477 warm grey
    accent:   egui::Color32::from_rgb(201, 100, 66),   // #c96442 Claude coral
    // Deep burnt-orange — legible on the cream bg where pure amber washes out.
    warn:     egui::Color32::from_rgb(176, 88, 22),    // #b05816
    alert:    egui::Color32::from_rgb(170, 50, 40),    // deep red
    slate:    egui::Color32::from_rgb(180, 172, 158),  // soft tan
};

// LED semantic colors — teal / terracotta / slate work in both themes so
// they stay as constants. Amber is too washed-out on a cream background
// though, so it's promoted to a palette-driven lookup.
const TEAL: egui::Color32 = egui::Color32::from_rgb(20, 184, 166);
const TERRACOTTA: egui::Color32 = egui::Color32::from_rgb(205, 92, 68);
const SLATE: egui::Color32 = egui::Color32::from_rgb(150, 145, 138);

/// Warn color — bright amber on dark, deep burnt-orange on light so it
/// reads against the warm cream background.
#[allow(non_snake_case)]
fn AMBER() -> egui::Color32 {
    active_palette().warn
}

/// Process-wide current theme. Reading is hot-path inside paint, so keep it
/// cheap — `AtomicU8` + a tiny decode helper.
static ACTIVE_THEME: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
fn active_palette() -> Palette {
    match ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) {
        1 => LIGHT,
        _ => DARK,
    }
}
fn set_active_theme(t: Theme) {
    let v: u8 = match t {
        Theme::Dark => 0,
        Theme::Light => 1,
    };
    ACTIVE_THEME.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Helper to pick primary-text color using the current palette.
fn c_text() -> egui::Color32 {
    active_palette().text
}
fn c_text_dim() -> egui::Color32 {
    active_palette().text_dim
}
fn c_bg() -> egui::Color32 {
    active_palette().bg
}
fn c_bg_row() -> egui::Color32 {
    active_palette().bg_row
}

// Backwards-compat names for the existing UI code — resolve via the palette
// lookup at call time so theme switches take effect without a refactor.
#[allow(non_snake_case)]
fn CREAM() -> egui::Color32 { c_text() }
#[allow(non_snake_case)]
fn TEXT_DIM() -> egui::Color32 { c_text_dim() }
#[allow(non_snake_case)]
fn BG_DARK() -> egui::Color32 { c_bg() }
#[allow(non_snake_case)]
fn BG_ROW() -> egui::Color32 { c_bg_row() }

// =============================================================================
// Entry Point
// =============================================================================

fn main() -> eframe::Result {
    // Restore persisted theme before the first paint so there's no flash.
    set_active_theme(Theme::load());

    // Frameless + transparent so we can paint a rounded background + custom
    // titlebar ourselves. WSLg's Xwayland path doesn't always pick up DWM's
    // default corner rounding, so drawing it in-app is the reliable option.
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([WIDTH, HEIGHT])
            .with_min_inner_size([WIDTH, 120.0])
            .with_always_on_top()
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(true)
            .with_title("tech-monitor"),
        ..Default::default()
    };
    eframe::run_native(
        "tech-ticker",
        options,
        Box::new(|cc| Ok(Box::new(TechApp::new(cc)))),
    )
}

// =============================================================================
// State — TechState, RelayEvent, Shared, TechApp
// =============================================================================

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
    /// Recent agent sessions pulled from `/api/v1/agent/sessions-recent` (newest
    /// first). Shown in the expandable debug console at the bottom of the
    /// widget.
    recent_sessions: Vec<RecentSession>,
    /// Toggle for the debug console panel.
    show_console: bool,
    /// Which session in `recent_sessions` is expanded (index). None = list view.
    console_expanded_index: Option<usize>,
    last_sessions_error: Option<String>,
}

/// Mirrors the daemon's `/api/v1/agent/sessions-recent` JSON shape.
#[derive(Clone, Default, Deserialize)]
pub struct RecentSession {
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub equipment: String,
    #[serde(default)]
    pub location: String,
    #[serde(default)]
    pub user_request: String,
    #[serde(default)]
    pub analysis: String,
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
            disabled_techs: HashMap::new(),
            recent_sessions: Vec::new(),
            show_console: false,
            console_expanded_index: None,
            last_sessions_error: None,
        }));

        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");

        // Tailscale poller (local fallback — relay is authoritative when
        // reachable, this catches the case where the relay is down).
        let s1 = state.clone();
        runtime.spawn(async move {
            loop {
                refresh_tailscale(&s1).await;
                // refresh_update_pending was the old /var/lib/tailscale-monitor
                // cache reader — now stale-always since the server-side relay
                // took over pushing. Relay's /push-state is authoritative for
                // last_pushed_build_mtime, so we don't poll it here anymore.
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

        // Tech-access overrides poller — populates the Disable/Enable state
        let s4 = state.clone();
        runtime.spawn(async move {
            loop {
                refresh_tech_overrides(&s4).await;
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            }
        });

        // Recent-sessions poller — drives the debug console so Andrew can
        // see field techs' agent questions + responses in real time.
        let s5 = state.clone();
        runtime.spawn(async move {
            loop {
                refresh_recent_sessions(&s5).await;
                tokio::time::sleep(std::time::Duration::from_secs(15)).await;
            }
        });

        Self { state, runtime }
    }
}

// =============================================================================
// Polling / Probes — Tailscale, Rate Limits, Relay, Tech Overrides
// =============================================================================

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

/// Poll the daemon for current tech-access disable overrides so the ticker
/// can render the Disable button in the correct on/off state.
/// Poll the daemon's `/api/v1/agent/sessions-recent` endpoint and stash the
/// newest sessions in Shared for the debug console to render.
async fn refresh_recent_sessions(state: &Arc<Mutex<Shared>>) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let resp = client
        .get(format!("{DAEMON_URL}/api/v1/agent/sessions-recent?limit=25"))
        .header("Authorization", format!("Bearer {OWNER_API_KEY}"))
        .send()
        .await;
    match resp {
        Ok(r) if r.status().is_success() => match r.json::<Vec<RecentSession>>().await {
            Ok(list) => {
                if let Ok(mut s) = state.lock() {
                    s.recent_sessions = list;
                    s.last_sessions_error = None;
                }
            }
            Err(e) => {
                if let Ok(mut s) = state.lock() {
                    s.last_sessions_error = Some(format!("decode: {e}"));
                }
            }
        },
        Ok(r) => {
            if let Ok(mut s) = state.lock() {
                s.last_sessions_error = Some(format!("HTTP {}", r.status()));
            }
        }
        Err(e) => {
            if let Ok(mut s) = state.lock() {
                s.last_sessions_error = Some(e.to_string());
            }
        }
    }
}

async fn refresh_tech_overrides(state: &Arc<Mutex<Shared>>) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let resp = client
        .get(format!("{DAEMON_URL}/api/v1/tech-access/overrides"))
        .header("Authorization", format!("Bearer {OWNER_API_KEY}"))
        .send()
        .await;
    let Ok(r) = resp else { return };
    if !r.status().is_success() {
        return;
    }
    let Ok(map) = r.json::<HashMap<String, RemoteOverride>>().await else {
        return;
    };
    let disabled: HashMap<String, String> = map
        .into_iter()
        .filter(|(_, v)| v.disabled)
        .map(|(k, v)| (k, v.reason))
        .collect();
    if let Ok(mut s) = state.lock() {
        s.disabled_techs = disabled;
    }
}

async fn refresh_rate_limits(state: &Arc<Mutex<Shared>>) {
    // Fetch rate_limits.json + daemon config from the server-side relay's
    // HTTP API. This replaces the old scp+ssh pair that was firing every 30s
    // and drowning the auth log in login events.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let (json_result, scp_err) = match client
        .get(format!("{RELAY_URL}/rate_limits"))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => match r.text().await {
            Ok(raw) => (
                serde_json::from_str::<HashMap<String, Vec<String>>>(&raw).ok(),
                None,
            ),
            Err(e) => (None, Some(format!("read body: {e}"))),
        },
        Ok(r) => (None, Some(format!("relay {}", r.status()))),
        Err(e) => (None, Some(format!("relay unreachable: {e}"))),
    };

    let cfg_text = match client
        .get(format!("{RELAY_URL}/daemon_config"))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r.text().await.unwrap_or_default(),
        _ => String::new(),
    };
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

// =============================================================================
// UI — eframe::App Implementation and Per-Tech Row
// =============================================================================

impl eframe::App for TechApp {
    /// Clear the framebuffer to fully transparent each frame. Without this
    /// override, eframe clears to `visuals.panel_fill` (opaque) and the area
    /// outside our rounded Frame shows up as a dark rectangle.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint for LED pulse
        ctx.request_repaint_after(std::time::Duration::from_millis(80));

        // Pick base visuals depending on active theme. The outer CentralPanel
        // is transparent — we paint a rounded `Frame` inside it to get the
        // custom window chrome.
        let mut visuals = match active_palette().bg == LIGHT.bg {
            true => egui::Visuals::light(),
            false => egui::Visuals::dark(),
        };
        visuals.panel_fill = egui::Color32::TRANSPARENT;
        visuals.window_fill = egui::Color32::TRANSPARENT;
        visuals.override_text_color = Some(CREAM());
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

        let palette = active_palette();
        let is_light = palette.bg == LIGHT.bg;
        // Outer rounded frame — this is our whole window surface. It has a
        // subtle border and drop shadow so it reads as a floating pill
        // against whatever's behind it (now that the OS chrome is gone).
        let outer_frame = egui::Frame::none()
            .fill(palette.bg)
            .rounding(egui::Rounding::same(10.0))
            .stroke(egui::Stroke::new(
                1.0,
                if is_light {
                    egui::Color32::from_rgba_unmultiplied(
                        palette.text_dim.r(),
                        palette.text_dim.g(),
                        palette.text_dim.b(),
                        90,
                    )
                } else {
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, 18)
                },
            ))
            .shadow(egui::epaint::Shadow {
                offset: egui::vec2(0.0, 2.0),
                blur: 8.0,
                spread: 0.0,
                color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, if is_light { 32 } else { 120 }),
            })
            .inner_margin(egui::Margin::symmetric(10.0, 8.0));

        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(egui::Margin::same(6.0)))
            .show(ctx, |ui| { outer_frame.show(ui, |ui| {
            // ── Custom titlebar (drag region + close button) ─────────────
            let title_resp = ui.horizontal(|ui| {
                draw_led(ui, TEAL, 5.0, true);
                ui.label(
                    egui::RichText::new("TECH MONITOR")
                        .strong()
                        .size(11.0)
                        .color(CREAM()),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Close — red-X on hover
                    let close = egui::Button::new(
                        egui::RichText::new("✕").size(12.0).color(TEXT_DIM()),
                    )
                    .frame(false)
                    .min_size(egui::vec2(20.0, 20.0));
                    if ui.add(close).on_hover_text("close").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        std::process::exit(0);
                    }
                    // Minimize
                    let min = egui::Button::new(
                        egui::RichText::new("—").size(12.0).color(TEXT_DIM()),
                    )
                    .frame(false)
                    .min_size(egui::vec2(20.0, 20.0));
                    if ui.add(min).on_hover_text("minimize").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                        let _ = std::process::Command::new("sh")
                            .arg("-c")
                            .arg("xdotool search --name '^tech-monitor$' windowminimize 2>/dev/null")
                            .spawn();
                    }
                });
            }).response;

            // Any drag that starts on the title-row (not on its buttons,
            // since buttons consume the click) moves the OS window.
            let drag_sense = ui.interact(title_resp.rect, egui::Id::new("drag-zone"), egui::Sense::click_and_drag());
            if drag_sense.drag_started_by(egui::PointerButton::Primary) {
                ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }

            ui.separator();

            // ── Row that holds the old "tools" (theme toggle, Build) ─────
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Theme toggle — sun (to switch to light) / moon (to switch to dark).
                    let is_light = active_palette().bg == LIGHT.bg;
                    let (glyph, hover, next) = if is_light {
                        ("☾", "switch to dark theme", Theme::Dark)
                    } else {
                        ("☀", "switch to light theme (Claude palette)", Theme::Light)
                    };
                    let theme_btn = egui::Button::new(
                        egui::RichText::new(glyph).size(11.0).color(TEXT_DIM()),
                    )
                    .frame(false)
                    .min_size(egui::vec2(18.0, 18.0));
                    if ui.add(theme_btn).on_hover_text(hover).clicked() {
                        set_active_theme(next);
                        next.save();
                    }
                    ui.add_space(4.0);

                    // "Build" button (rebuild exe + SIGUSR1 monitor)
                    let building = action.in_flight.as_deref() == Some("build");
                    let btn_label = if building { "building..." } else { "Build" };
                    let btn = egui::Button::new(
                        egui::RichText::new(btn_label).size(9.0).color(CREAM()),
                    )
                    .fill(if building { AMBER().linear_multiply(0.5) } else { TEAL.linear_multiply(0.25) })
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
                                .color(TEXT_DIM()),
                        );
                    }
                });
            });

            // Second header line: current action status / result (very subtle)
            if let Some(msg) = &action.in_flight {
                ui.label(
                    egui::RichText::new(format!("⚙ {msg}"))
                        .size(9.0)
                        .color(AMBER()),
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
                        AMBER(),
                    )
                } else if relay_err.is_some() {
                    ("⚠ relay offline".to_string(), TERRACOTTA)
                } else {
                    (format!("{} events", events.len()), TEXT_DIM())
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
                            let color = if ev.is_alert() { TERRACOTTA } else if ev.ts() > last_seen { AMBER() } else { TEXT_DIM() };
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
            let disabled_techs: HashMap<String, String> = {
                let s = self.state.lock().unwrap();
                s.disabled_techs.clone()
            };
            for (name, _host, wsl, t) in &snapshot {
                let is_disabled = disabled_techs.contains_key(&name.to_lowercase());
                self.draw_tech_row(ui, name, wsl, t, is_disabled);
            }
            // Footer: legend
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                draw_led(ui, TEAL, 3.0, false);
                ui.label(egui::RichText::new("online").size(8.0).color(TEXT_DIM()));
                ui.add_space(4.0);
                draw_led(ui, SLATE, 3.0, false);
                ui.label(egui::RichText::new("offline").size(8.0).color(TEXT_DIM()));
                ui.add_space(4.0);
                draw_led(ui, AMBER(), 3.0, true);
                ui.label(
                    egui::RichText::new("update pending")
                        .size(8.0)
                        .color(TEXT_DIM()),
                );
            });

            // ── Debug console — collapsible at the bottom ──────────────
            ui.add_space(4.0);
            ui.separator();
            self.draw_debug_console(ui);
            }); // closes outer_frame.show
        });

        // Keep the runtime alive
        let _ = &self.runtime;
    }
}

impl TechApp {
    /// Collapsible debug console at the bottom — shows recent techs' agent
    /// questions + responses pulled from the daemon's
    /// `/api/v1/agent/sessions-recent` endpoint, so Andrew can shoulder-surf
    /// field troubleshooting in real time.
    fn draw_debug_console(&self, ui: &mut egui::Ui) {
        let (show, sessions, err, expanded_idx) = {
            let s = self.state.lock().unwrap();
            (
                s.show_console,
                s.recent_sessions.clone(),
                s.last_sessions_error.clone(),
                s.console_expanded_index,
            )
        };

        ui.horizontal(|ui| {
            let arrow = if show { "▾" } else { "▸" };
            let label = format!("{arrow} CONSOLE");
            let btn = ui.add(
                egui::Button::new(
                    egui::RichText::new(label)
                        .size(9.0)
                        .color(active_palette().accent),
                )
                .frame(false),
            );
            if btn.clicked() {
                if let Ok(mut s) = self.state.lock() {
                    s.show_console = !s.show_console;
                }
            }
            ui.add_space(4.0);
            let meta = if let Some(e) = &err {
                egui::RichText::new(format!("⚠ {e}"))
                    .size(8.0)
                    .color(active_palette().alert)
            } else {
                egui::RichText::new(format!("{} sessions", sessions.len()))
                    .size(8.0)
                    .color(active_palette().text_dim)
            };
            ui.label(meta);
        });

        if !show {
            return;
        }

        egui::ScrollArea::vertical()
            .id_salt("debug-console-scroll")
            .max_height(240.0)
            .show(ui, |ui| {
                if sessions.is_empty() {
                    ui.label(
                        egui::RichText::new("no sessions yet — techs' agent requests will appear here")
                            .size(9.0)
                            .color(active_palette().text_dim),
                    );
                    return;
                }
                for (idx, sess) in sessions.iter().enumerate() {
                    let ts_local = chrono::DateTime::parse_from_rfc3339(&sess.timestamp)
                        .ok()
                        .map(|d| d.with_timezone(&chrono::Local).format("%H:%M:%S").to_string())
                        .unwrap_or_else(|| "--:--:--".to_string());
                    let header = format!(
                        "{ts_local}  {}  ({})",
                        sess.equipment, sess.location
                    );
                    let is_expanded = expanded_idx == Some(idx);
                    let arrow = if is_expanded { "▾" } else { "▸" };
                    let row = ui.add(
                        egui::Button::new(
                            egui::RichText::new(format!("{arrow} {header}"))
                                .size(9.0)
                                .color(active_palette().text),
                        )
                        .frame(false),
                    );
                    if row.clicked() {
                        if let Ok(mut s) = self.state.lock() {
                            s.console_expanded_index = if is_expanded { None } else { Some(idx) };
                        }
                    }
                    if !is_expanded {
                        // One-line preview of the tech's question.
                        let preview: String = sess
                            .user_request
                            .chars()
                            .take(80)
                            .collect();
                        ui.label(
                            egui::RichText::new(format!("  Q: {preview}"))
                                .size(8.5)
                                .color(active_palette().text_dim),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("  Q:")
                                .size(9.0)
                                .color(active_palette().accent),
                        );
                        ui.label(
                            egui::RichText::new(&sess.user_request)
                                .size(9.0)
                                .color(active_palette().text),
                        );
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new("  A:")
                                .size(9.0)
                                .color(active_palette().accent),
                        );
                        ui.label(
                            egui::RichText::new(&sess.analysis)
                                .size(9.0)
                                .color(active_palette().text),
                        );
                    }
                    ui.add_space(3.0);
                }
            });
    }

    fn draw_tech_row(
        &self,
        ui: &mut egui::Ui,
        name: &str,
        wsl_user: &str,
        t: &TechState,
        is_disabled: bool,
    ) {
        let usage_color = if is_disabled {
            TERRACOTTA
        } else if t.max > 0 && t.used >= t.max {
            TERRACOTTA
        } else if t.max > 0 && t.used as f32 / t.max as f32 >= 0.66 {
            AMBER()
        } else {
            TEAL
        };

        // Beveled row card: slightly rounder corners, a hairline border in
        // the palette's dim-text color, and a soft drop-shadow to separate
        // rows from the panel background.
        let palette = active_palette();
        let is_light = palette.bg == LIGHT.bg;
        let border = egui::Stroke::new(
            1.0,
            if is_light {
                egui::Color32::from_rgba_unmultiplied(
                    palette.text_dim.r(),
                    palette.text_dim.g(),
                    palette.text_dim.b(),
                    60, // very subtle tan-border
                )
            } else {
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 10) // hairline highlight
            },
        );
        let shadow = egui::epaint::Shadow {
            offset: egui::vec2(0.0, 1.0),
            blur: 4.0,
            spread: 0.0,
            color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, if is_light { 24 } else { 80 }),
        };
        egui::Frame::none()
            .fill(BG_ROW())
            .rounding(egui::Rounding::same(6.0))
            .stroke(border)
            .shadow(shadow)
            .inner_margin(egui::Margin::symmetric(10.0, 7.0))
            .show(ui, |ui| {
                // Line 1: LED + name + [update-LED] [SSH] [Reset] ...spacer... usage
                ui.horizontal(|ui| {
                    let color = if t.online { TEAL } else { SLATE };
                    draw_led(ui, color, 5.0, t.online);
                    ui.label(
                        egui::RichText::new(name)
                            .strong()
                            .size(12.0)
                            .color(CREAM()),
                    );
                    if t.update_pending {
                        ui.add_space(2.0);
                        draw_led(ui, AMBER(), 3.5, true);
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

                            // Disable/Enable toggle — always shown. Glyph and
                            // color reflect current state; click flips it via
                            // the daemon's tech-access endpoint.
                            let (glyph, color, hover_text) = if is_disabled {
                                ("✓", TEAL, format!("Click to ENABLE {name}"))
                            } else {
                                ("⊘", TERRACOTTA, format!("Click to DISABLE {name} (they will get 403 on next request)"))
                            };
                            let toggle_btn = egui::Button::new(
                                egui::RichText::new(glyph).size(12.0).color(color),
                            )
                            .frame(false)
                            .min_size(egui::vec2(18.0, 18.0));
                            if ui.add(toggle_btn).on_hover_text(hover_text).clicked() {
                                self.trigger_toggle_disable(name.to_string(), is_disabled);
                            }
                        },
                    );
                });

                // Line 2: last-used (prominent) ...spacer... window
                ui.horizontal(|ui| {
                    let (last_label, last_color) = match t.last_request {
                        Some(ts) => (format!("last used {}", rel_time(ts)), CREAM()),
                        None => ("never used".to_string(), TEXT_DIM()),
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
                                    .color(TEXT_DIM()),
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

        ui.add_space(5.0);
    }

    // ── Actions ──────────────────────────────────────────────────────────────

    fn trigger_toggle_disable(&self, tech_name: String, currently_disabled: bool) {
        let state = self.state.clone();
        let label = if currently_disabled {
            format!("enable {tech_name}")
        } else {
            format!("disable {tech_name}")
        };
        if let Ok(mut s) = state.lock() {
            s.action.in_flight = Some(label.clone());
        }
        let endpoint = if currently_disabled {
            "tech-access/enable"
        } else {
            "tech-access/disable"
        };
        let body = if currently_disabled {
            serde_json::json!({ "tech_name": tech_name })
        } else {
            serde_json::json!({ "tech_name": tech_name, "reason": "disabled from tech-ticker" })
        };
        let name_for_log = tech_name.clone();
        self.runtime.spawn(async move {
            let client = reqwest::Client::new();
            let res = client
                .post(format!("{DAEMON_URL}/api/v1/{endpoint}"))
                .header("Authorization", format!("Bearer {OWNER_API_KEY}"))
                .json(&body)
                .timeout(std::time::Duration::from_secs(10))
                .send()
                .await;
            if let Ok(mut s) = state.lock() {
                s.action.in_flight = None;
                s.action.last_result = Some(match res {
                    Ok(r) if r.status().is_success() => {
                        // Optimistically update local state so the button
                        // flips before the 30s poll catches up.
                        let key = name_for_log.to_lowercase();
                        if currently_disabled {
                            s.disabled_techs.remove(&key);
                        } else {
                            s.disabled_techs.insert(key, "disabled from tech-ticker".to_string());
                        }
                        (format!("{label}: OK"), false)
                    }
                    Ok(r) => (format!("{label} → HTTP {}", r.status()), true),
                    Err(e) => (format!("{label}: {e}"), true),
                });
                s.action.last_result_at = Some(std::time::Instant::now());
            }
        });
    }

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

// =============================================================================
// Helpers — rel_time, draw_led
// =============================================================================

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
