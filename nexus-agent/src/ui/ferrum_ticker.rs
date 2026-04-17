//! ferrum-ticker — Ferrum Mail + NexusRelay Uptime Monitor
//!
//! eframe-based always-on-top widget that pings the Ferrum Mail stack and
//! Vultr NexusRelay every 8 seconds. The `PROBES` constant declares six
//! live probes split across two groups:
//!
//! * `Group::An` (DO 100.67.227.31): `an_mailbox` (FerumMailbox REST via
//!   Tailscale), `an_saas` (landing page + billing API), `an_public_api`
//!   (nginx → mailbox over HTTPS), `an_webmail` (FerumWebmail Rust WASM),
//!   `an_smtp_ts` (Postfix via Tailscale-side TCP).
//! * `Group::Vultr` (NexusRelay 100.119.45.41 / 137.220.57.248):
//!   `vu_relay` (nexus-relay HTTP :9587), `vu_shield` (:8080 TCP),
//!   `vu_vault` (:8200 TCP), `vu_dns` (:53 unbound), `vu_smtp_ts`
//!   (Postfix Tailscale-side).
//!
//! Each `Probe` carries a `ProbeKind` — `Http` (requires 2xx), `HttpAny`
//! (< 400, tolerates 3xx for webmail), or `Tcp` (requires a connect within
//! `TCP_TIMEOUT`). `probe_with_retry` runs each twice with `RETRY_DELAY`
//! between attempts to ride through transient nginx/TLS blips.
//! `apply_outcome` edge-triggers log lines on UP↔DOWN transitions
//! (categorized `Sev::Info` / `Warn` / `Crit`) and bumps `fail_count`.
//!
//! UI: rounded transparent pill with custom titlebar (drag, close, min,
//! sun/moon theme toggle, clickable fail counter that opens a floating
//! full-log window). `render_group` paints each service row with a pulsing
//! LED (red when down), label, and right-aligned latency / DOWN / — text;
//! `build_tip` generates the hover tooltip with probe URL/target, result
//! code, latency, and last-ok age. Theme persists to
//! `/tmp/.ferrum-ticker-theme`.
//!
//! Same chrome as nexus-ticker / tech-ticker / security-ticker.
//!
//! Launch: ferrum-ticker &
//!
//! # File
//! `nexus-agent/src/ui/ferrum_ticker.rs`
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
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// =============================================================================
// Config — Constants, Probe Definitions
// =============================================================================

const WIDTH: f32 = 460.0;
const HEIGHT: f32 = 460.0;
const POLL_SECS: u64 = 8;
const HTTP_TIMEOUT: Duration = Duration::from_secs(8);
const TCP_TIMEOUT: Duration = Duration::from_secs(4);
/// Wait between retries on a failed probe — short enough to stay within a
/// single poll cycle, long enough to ride through a brief TLS/nginx blip.
const RETRY_DELAY: Duration = Duration::from_millis(500);
const MAX_LOG_LINES: usize = 200;

const THEME_FILE: &str = "/tmp/.ferrum-ticker-theme";

#[derive(Clone, Copy, PartialEq)]
enum Group {
    An,
    Vultr,
}

impl Group {
    fn label(self) -> &'static str {
        match self {
            Group::An => "AN  (DO 100.67.227.31)",
            Group::Vultr => "VULTR  (NexusRelay 100.119.45.41 / 137.220.57.248)",
        }
    }
}

#[derive(Clone, Copy)]
enum ProbeKind {
    /// HTTP GET — OK iff 2xx.
    Http(&'static str),
    /// HTTP GET — OK iff 2xx OR 3xx OR specific tolerated codes (used for
    /// webmail which may serve a directory index / redirect).
    HttpAny(&'static str),
    /// TCP connect — OK if the connection completes inside TCP_TIMEOUT.
    Tcp(&'static str),
}

struct Probe {
    id: &'static str,
    label: &'static str,
    /// Short hover-text describing what this probe is checking.
    desc: &'static str,
    group: Group,
    kind: ProbeKind,
}

// Probe set verified live 2026-04-13. The following endpoints are
// deliberately NOT probed (they exist on the boxes but aren't reachable
// from the laptop, so an LED for them would always be red):
//   - vu_vault    — nexus-vault on Vultr binds 127.0.0.1:8200 only
//   - vu_dns      — unbound on Vultr binds 127.0.0.1:53 only
//   - vu_shield   — nexus-shield on Vultr is bound 0.0.0.0:8080 but the
//                   process isn't responding (even from localhost on the
//                   box). Needs operator investigation; will be added back
//                   once it actually answers /health.
//   - vu_smtp_pub — public-internet :25 from a residential ISP is almost
//                   always blocked outbound. Monitor externally if needed.
//   - an_saas     — port 3737 wasn't found listening on the AN box at
//                   ticker-build time. Add a row when the actual SaaS
//                   listen address is known.
const PROBES: &[Probe] = &[
    // ── AN box (DO) ────────────────────────────────────────────────────────
    Probe {
        id: "an_mailbox",
        label: "mailbox REST :3838",
        desc: "FerumMailbox REST API (inbound SMTP storage + read endpoints) — direct via Tailscale.",
        group: Group::An,
        kind: ProbeKind::Http("http://100.67.227.31:3838/api/v1/health"),
    },
    Probe {
        id: "an_saas",
        label: "SaaS :3737",
        desc: "FerumMailSaaS landing page + billing/marketing API.",
        group: Group::An,
        kind: ProbeKind::HttpAny("http://100.67.227.31:3737/"),
    },
    Probe {
        id: "an_public_api",
        label: "public mail API",
        desc: "Public path through nginx → mailbox: https://ferrum-mail.com/mailbox/api/v1/health",
        group: Group::An,
        kind: ProbeKind::Http("https://ferrum-mail.com/mailbox/api/v1/health"),
    },
    Probe {
        id: "an_webmail",
        label: "webmail",
        desc: "FerumWebmail — Rust WASM frontend at https://ferrum-mail.com/mail/",
        group: Group::An,
        kind: ProbeKind::HttpAny("https://ferrum-mail.com/mail/"),
    },
    Probe {
        id: "an_smtp_ts",
        label: "SMTP :25 (ts)",
        desc: "Postfix on AN box — Tailscale-side. Receives mail forwarded from the Vultr relay.",
        group: Group::An,
        kind: ProbeKind::Tcp("100.67.227.31:25"),
    },
    // ── Vultr NexusRelay ───────────────────────────────────────────────────
    Probe {
        id: "vu_relay",
        label: "direct-MX :9587",
        desc: "nexus-relay HTTP API — outbound SMTP via direct MX (Tailscale-only).",
        group: Group::Vultr,
        kind: ProbeKind::Http("http://100.119.45.41:9587/health"),
    },
    Probe {
        id: "vu_shield",
        label: "shield :8080",
        desc: "nexus-shield endpoint protection (TCP probe — HTTP /health hangs on this box).",
        group: Group::Vultr,
        kind: ProbeKind::Tcp("100.119.45.41:8080"),
    },
    Probe {
        id: "vu_vault",
        label: "vault :8200",
        desc: "nexus-vault local secrets store — now bound to Tailscale interface.",
        group: Group::Vultr,
        kind: ProbeKind::Tcp("100.119.45.41:8200"),
    },
    Probe {
        id: "vu_dns",
        label: "DNS :53 (unbound)",
        desc: "unbound recursive resolver — required for Spamhaus RBL lookups.",
        group: Group::Vultr,
        kind: ProbeKind::Tcp("100.119.45.41:53"),
    },
    Probe {
        id: "vu_smtp_ts",
        label: "SMTP :25 (ts)",
        desc: "Postfix on Vultr — Tailscale-side reachability.",
        group: Group::Vultr,
        kind: ProbeKind::Tcp("100.119.45.41:25"),
    },
    // vu_smtp_pub intentionally omitted — no good way to probe public :25 from a
    // residential ISP (outbound :25 blocked). Add an external uptime monitor if
    // you want this row lit — Uptime Kuma / Better Stack / Healthchecks.io all
    // work. If adding, point the probe at their webhook status URL.
];

// =============================================================================
// Theme — Palette and Dark/Light Persistence
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Theme {
    Dark,
    Light,
}

impl Theme {
    fn load() -> Self {
        match std::fs::read_to_string(THEME_FILE).ok().as_deref().map(str::trim) {
            Some("light") => Self::Light,
            _ => Self::Dark,
        }
    }
    fn save(self) {
        let _ = std::fs::write(
            THEME_FILE,
            match self {
                Self::Dark => "dark",
                Self::Light => "light",
            },
        );
    }
    fn toggled(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Dark,
        }
    }
}

#[derive(Clone, Copy)]
struct Palette {
    bg: egui::Color32,
    bg_row: egui::Color32,
    text: egui::Color32,
    text_dim: egui::Color32,
    teal: egui::Color32,
    amber: egui::Color32,
    terracotta: egui::Color32,
    slate: egui::Color32,
    is_dark: bool,
}

const DARK: Palette = Palette {
    bg: egui::Color32::from_rgb(45, 42, 38),
    bg_row: egui::Color32::from_rgb(55, 50, 46),
    text: egui::Color32::from_rgb(245, 240, 235),
    text_dim: egui::Color32::from_rgb(155, 145, 138),
    teal: egui::Color32::from_rgb(20, 184, 166),
    amber: egui::Color32::from_rgb(245, 180, 60),
    terracotta: egui::Color32::from_rgb(205, 92, 68),
    slate: egui::Color32::from_rgb(110, 105, 100),
    is_dark: true,
};

const LIGHT: Palette = Palette {
    bg: egui::Color32::from_rgb(250, 249, 245),
    bg_row: egui::Color32::from_rgb(237, 231, 216),
    text: egui::Color32::from_rgb(61, 57, 41),
    text_dim: egui::Color32::from_rgb(141, 132, 119),
    teal: egui::Color32::from_rgb(20, 130, 118),
    amber: egui::Color32::from_rgb(176, 88, 22),
    terracotta: egui::Color32::from_rgb(170, 50, 40),
    slate: egui::Color32::from_rgb(180, 172, 158),
    is_dark: false,
};

fn palette(theme: Theme) -> Palette {
    match theme {
        Theme::Dark => DARK,
        Theme::Light => LIGHT,
    }
}

// =============================================================================
// State — ProbeState, LogEntry, SharedState, FerrumApp
// =============================================================================

#[derive(Clone, Default)]
struct ProbeState {
    /// Most recent successful probe time. None if never seen succeed.
    last_ok: Option<Instant>,
    /// Most recent probe time, regardless of outcome.
    last_check: Option<Instant>,
    /// Outcome of the most recent probe.
    ok: Option<bool>,
    /// HTTP status code (for HTTP/HttpAny probes).
    http_code: Option<u16>,
    /// Round-trip duration in milliseconds.
    latency_ms: Option<u64>,
    /// Error text from the most recent failure (for hover detail).
    last_error: Option<String>,
}

impl ProbeState {
    fn led_color(&self, p: Palette) -> egui::Color32 {
        match self.ok {
            Some(true) => p.teal,
            Some(false) => p.terracotta,
            None => p.slate,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Sev {
    Info,
    Warn,
    Crit,
}

#[derive(Clone)]
struct LogEntry {
    timestamp: String,
    message: String,
    severity: Sev,
}

struct SharedState {
    probes: HashMap<&'static str, ProbeState>,
    log: Vec<LogEntry>,
    /// Total count of failures observed since launch (for the header chip).
    fail_count: u64,
}

impl SharedState {
    fn push_log(&mut self, msg: &str, severity: Sev) {
        self.log.push(LogEntry {
            timestamp: now(),
            message: msg.to_string(),
            severity,
        });
        let excess = self.log.len().saturating_sub(MAX_LOG_LINES);
        if excess > 0 {
            self.log.drain(0..excess);
        }
    }
}

struct FerrumApp {
    state: Arc<Mutex<SharedState>>,
    theme: Theme,
    show_log_window: bool,
    _runtime: tokio::runtime::Runtime,
}

impl FerrumApp {
    fn new() -> Self {
        let mut probes: HashMap<&'static str, ProbeState> = HashMap::new();
        for p in PROBES {
            probes.insert(p.id, ProbeState::default());
        }
        let state = Arc::new(Mutex::new(SharedState {
            probes,
            log: vec![LogEntry {
                timestamp: now(),
                message: format!("ferrum-ticker started — polling {} probes every {}s", PROBES.len(), POLL_SECS),
                severity: Sev::Info,
            }],
            fail_count: 0,
        }));

        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");

        let s = state.clone();
        runtime.spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(HTTP_TIMEOUT)
                .danger_accept_invalid_certs(false)
                .build()
                .expect("reqwest client");
            loop {
                run_probes(s.clone(), &client).await;
                tokio::time::sleep(Duration::from_secs(POLL_SECS)).await;
            }
        });

        Self {
            state,
            theme: Theme::load(),
            show_log_window: false,
            _runtime: runtime,
        }
    }
}

// =============================================================================
// Polling / Probes — run_probes, probe_with_retry, probe_once, apply_outcome
// =============================================================================

async fn run_probes(state: Arc<Mutex<SharedState>>, client: &reqwest::Client) {
    // Run all probes concurrently — one slow target shouldn't gate the others.
    let mut handles = Vec::with_capacity(PROBES.len());
    for p in PROBES {
        let client = client.clone();
        handles.push(tokio::spawn(async move { (p.id, probe_with_retry(p, &client).await) }));
    }
    for h in handles {
        if let Ok((id, outcome)) = h.await {
            apply_outcome(&state, id, outcome);
        }
    }
}

/// Run the probe, and on failure retry once after a brief delay. Prevents
/// flapping alerts when Cloudflare edge / nginx / DNS has a transient blip.
async fn probe_with_retry(p: &'static Probe, client: &reqwest::Client) -> Outcome {
    let first = probe_once(p, client).await;
    if first.ok {
        return first;
    }
    tokio::time::sleep(RETRY_DELAY).await;
    let second = probe_once(p, client).await;
    if second.ok {
        return second;
    }
    // Both attempts failed — surface the second (freshest) error.
    second
}

#[derive(Clone)]
struct Outcome {
    ok: bool,
    http_code: Option<u16>,
    latency_ms: u64,
    error: Option<String>,
}

async fn probe_once(p: &Probe, client: &reqwest::Client) -> Outcome {
    let start = Instant::now();
    match p.kind {
        ProbeKind::Http(url) => match client.get(url).send().await {
            Ok(r) => {
                let code = r.status().as_u16();
                let ok = r.status().is_success();
                Outcome {
                    ok,
                    http_code: Some(code),
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: if ok { None } else { Some(format!("HTTP {code}")) },
                }
            }
            Err(e) => Outcome {
                ok: false,
                http_code: None,
                latency_ms: start.elapsed().as_millis() as u64,
                error: Some(strip_url(&e.to_string())),
            },
        },
        ProbeKind::HttpAny(url) => match client.get(url).send().await {
            Ok(r) => {
                let code = r.status().as_u16();
                // Webmail at /mail/ may 200, 301/302 (redirect to /mail/), or
                // 304 if cached. Anything < 400 is "alive".
                let ok = code < 400;
                Outcome {
                    ok,
                    http_code: Some(code),
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: if ok { None } else { Some(format!("HTTP {code}")) },
                }
            }
            Err(e) => Outcome {
                ok: false,
                http_code: None,
                latency_ms: start.elapsed().as_millis() as u64,
                error: Some(strip_url(&e.to_string())),
            },
        },
        ProbeKind::Tcp(addr) => {
            let connect = tokio::net::TcpStream::connect(addr);
            match tokio::time::timeout(TCP_TIMEOUT, connect).await {
                Ok(Ok(_stream)) => Outcome {
                    ok: true,
                    http_code: None,
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: None,
                },
                Ok(Err(e)) => Outcome {
                    ok: false,
                    http_code: None,
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: Some(e.to_string()),
                },
                Err(_) => Outcome {
                    ok: false,
                    http_code: None,
                    latency_ms: TCP_TIMEOUT.as_millis() as u64,
                    error: Some(format!("timeout after {}s", TCP_TIMEOUT.as_secs())),
                },
            }
        }
    }
}

/// reqwest error strings include the full URL which makes log lines noisy and
/// leaks tokens. Trim everything from `https://` onward.
fn strip_url(s: &str) -> String {
    if let Some(idx) = s.find(" for url (") {
        s[..idx].to_string()
    } else {
        s.to_string()
    }
}

fn apply_outcome(state: &Arc<Mutex<SharedState>>, id: &'static str, out: Outcome) {
    let Ok(mut s) = state.lock() else { return };
    let prev_ok = s.probes.get(id).and_then(|st| st.ok);
    let now_t = Instant::now();

    let entry = s.probes.entry(id).or_default();
    entry.last_check = Some(now_t);
    entry.ok = Some(out.ok);
    entry.http_code = out.http_code;
    entry.latency_ms = Some(out.latency_ms);
    entry.last_error = out.error.clone();
    if out.ok {
        entry.last_ok = Some(now_t);
    }

    // Edge-trigger log lines on transitions so a flaky probe doesn't spam.
    let label = PROBES.iter().find(|p| p.id == id).map(|p| p.label).unwrap_or(id);
    match (prev_ok, out.ok) {
        (None, false) => {
            // First-ever probe and it failed — log as info, not warn, since we
            // don't yet know whether this is a real outage or transient.
            let err = out.error.unwrap_or_else(|| "unknown error".into());
            s.push_log(&format!("{label} unreachable on first probe: {err}"), Sev::Warn);
            s.fail_count += 1;
        }
        (Some(true), false) => {
            let err = out.error.unwrap_or_else(|| "unknown error".into());
            s.push_log(&format!("{label} DOWN: {err}"), Sev::Crit);
            s.fail_count += 1;
        }
        (Some(false), true) => {
            s.push_log(&format!("{label} recovered ({}ms)", out.latency_ms), Sev::Info);
        }
        _ => {}
    }
}

// =============================================================================
// Entry Point and eframe::App
// =============================================================================

fn main() -> eframe::Result {
    // Auto-detect WSLg and configure rendering — so the .vbs launcher doesn't
    // need to set env vars. ZINK/Mesa can't pick a GPU inside WSL; force
    // software GL, and point at WSLg's Wayland + X sockets.
    if std::path::Path::new("/mnt/wslg").exists() {
        unsafe {
            if std::env::var("DISPLAY").is_err() {
                std::env::set_var("DISPLAY", ":0");
            }
            if std::env::var("WAYLAND_DISPLAY").is_err() {
                std::env::set_var("WAYLAND_DISPLAY", "wayland-0");
            }
            if std::env::var("XDG_RUNTIME_DIR").is_err() {
                std::env::set_var("XDG_RUNTIME_DIR", "/mnt/wslg/runtime-dir");
            }
            std::env::set_var("LIBGL_ALWAYS_SOFTWARE", "1");
        }
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([WIDTH, HEIGHT])
            .with_min_inner_size([380.0, 280.0])
            .with_always_on_top()
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(true)
            .with_title("ferrum-ticker"),
        ..Default::default()
    };
    eframe::run_native(
        "ferrum-ticker",
        options,
        Box::new(|_cc| Ok(Box::new(FerrumApp::new()))),
    )
}

impl eframe::App for FerrumApp {
    /// REQUIRED for the frameless+transparent pill — without this the GL
    /// surface clears to the visuals' window color and you get an opaque
    /// rectangle behind/around the rounded pill (LESSONS L26).
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0; 4]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(80));

        let p = palette(self.theme);
        let mut visuals = if p.is_dark { egui::Visuals::dark() } else { egui::Visuals::light() };
        visuals.panel_fill = egui::Color32::TRANSPARENT;
        visuals.window_fill = egui::Color32::TRANSPARENT;
        visuals.override_text_color = Some(p.text);
        ctx.set_visuals(visuals);

        let snapshot = self.state.lock().unwrap();
        let probes_snap: HashMap<&'static str, ProbeState> = snapshot.probes.clone();
        let log_snap: Vec<LogEntry> = snapshot.log.clone();
        let fail_count = snapshot.fail_count;
        drop(snapshot);

        let any_down = probes_snap.values().any(|s| s.ok == Some(false));
        let any_unknown = probes_snap.values().any(|s| s.ok.is_none());

        let outer_frame = egui::Frame::none()
            .fill(p.bg)
            .rounding(egui::Rounding::same(10.0))
            .stroke(egui::Stroke::new(
                1.0,
                if p.is_dark {
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, 18)
                } else {
                    egui::Color32::from_rgba_unmultiplied(p.text_dim.r(), p.text_dim.g(), p.text_dim.b(), 90)
                },
            ))
            .shadow(egui::epaint::Shadow {
                offset: egui::vec2(0.0, 2.0),
                blur: 8.0,
                spread: 0.0,
                color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, if p.is_dark { 120 } else { 32 }),
            })
            .inner_margin(egui::Margin::symmetric(10.0, 8.0));

        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(egui::Margin::same(6.0)))
            .show(ctx, |ui| {
                outer_frame.show(ui, |ui| {
                    // Clip child widgets to the rounded outer frame so nothing
                    // paints into the shadow/transparent margin (LESSONS L26).
                    ui.set_clip_rect(ui.max_rect());
                    // ── Custom titlebar ───────────────────────────────────
                    let header = ui.horizontal(|ui| {
                        let main_color = if any_down {
                            p.terracotta
                        } else if any_unknown {
                            p.amber
                        } else {
                            p.teal
                        };
                        draw_led(ui, main_color, 5.0, any_down);
                        ui.label(
                            egui::RichText::new("FERRUM-MAIL")
                                .strong()
                                .size(11.0)
                                .color(p.text),
                        );
                        ui.label(
                            egui::RichText::new(if any_down {
                                "ALERT"
                            } else if any_unknown {
                                "STARTING"
                            } else {
                                "HEALTHY"
                            })
                            .size(9.0)
                            .color(p.text_dim),
                        );

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            // Close
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new("✕").size(12.0).color(p.text_dim),
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
                            // Minimize
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new("—").size(12.0).color(p.text_dim),
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
                                    .arg("xdotool search --name '^ferrum-ticker$' windowminimize 2>/dev/null")
                                    .spawn();
                            }
                            ui.add_space(4.0);
                            // Theme toggle — yellow sun (dark mode) / moon (light mode)
                            let glyph = if self.theme == Theme::Dark { "☀" } else { "☾" };
                            let tip = if self.theme == Theme::Dark {
                                "Switch to light theme"
                            } else {
                                "Switch to dark theme"
                            };
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new(glyph)
                                            .size(11.0)
                                            .color(p.amber)
                                            .strong(),
                                    )
                                    .frame(false)
                                    .min_size(egui::vec2(16.0, 16.0)),
                                )
                                .on_hover_text(tip)
                                .clicked()
                            {
                                self.theme = self.theme.toggled();
                                self.theme.save();
                            }
                            ui.add_space(8.0);
                            // Failure counter — clickable to open log window
                            let fc = format!("fail:{fail_count}");
                            let fc_color = if fail_count == 0 { p.teal } else { p.amber };
                            let fc_tip = if fail_count == 0 {
                                "no probe failures since launch — click anyway to view log".to_string()
                            } else {
                                format!("{fail_count} probe failure(s) since launch — click for log")
                            };
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new(fc).size(9.0).color(fc_color),
                                    )
                                    .frame(false),
                                )
                                .on_hover_text(&fc_tip)
                                .clicked()
                            {
                                self.show_log_window = !self.show_log_window;
                            }
                        });
                    })
                    .response;

                    let drag_sense = ui.interact(
                        header.rect,
                        egui::Id::new("ferrum-drag"),
                        egui::Sense::click_and_drag(),
                    );
                    if drag_sense.drag_started_by(egui::PointerButton::Primary) {
                        ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                    }

                    ui.separator();

                    // ── Probe sections ────────────────────────────────────
                    render_group(ui, &p, Group::An, &probes_snap);
                    ui.add_space(4.0);
                    render_group(ui, &p, Group::Vultr, &probes_snap);

                    ui.separator();

                    // ── Log strip ────────────────────────────────────────
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("recent ({} lines)", log_snap.len()))
                                .size(8.0)
                                .color(p.text_dim),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("copy all")
                                            .size(9.0)
                                            .color(p.teal),
                                    )
                                    .on_hover_text("Copy entire log to clipboard")
                                    .clicked()
                                {
                                    let buf: String = log_snap
                                        .iter()
                                        .map(|e| {
                                            let tag = match e.severity {
                                                Sev::Info => "INFO",
                                                Sev::Warn => "WARN",
                                                Sev::Crit => "CRIT",
                                            };
                                            format!("{}  {}  {}", e.timestamp, tag, e.message)
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n");
                                    ui.ctx().copy_text(buf);
                                }
                                if ui
                                    .button(
                                        egui::RichText::new("clear")
                                            .size(9.0)
                                            .color(p.amber),
                                    )
                                    .on_hover_text("Clear the log buffer")
                                    .clicked()
                                {
                                    if let Ok(mut s) = self.state.lock() {
                                        s.log.clear();
                                    }
                                }
                            },
                        );
                    });
                    let avail = ui.available_height();
                    egui::ScrollArea::vertical()
                        .max_height(avail.max(40.0))
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for entry in &log_snap {
                                let color = match entry.severity {
                                    Sev::Info => p.text,
                                    Sev::Warn => p.amber,
                                    Sev::Crit => p.terracotta,
                                };
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(&entry.timestamp)
                                                .size(8.0)
                                                .color(p.slate),
                                        )
                                        .selectable(true),
                                    );
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(&entry.message)
                                                .size(9.0)
                                                .color(color),
                                        )
                                        .selectable(true)
                                        .wrap(),
                                    );
                                });
                            }
                        });
                });
            });

        // ── Optional floating log/detail window ───────────────────────────
        if self.show_log_window {
            let mut still_open = true;
            egui::Window::new("Ferrum probe log — full detail")
                .open(&mut still_open)
                .default_size([520.0, 360.0])
                .resizable(true)
                .collapsible(false)
                .frame(
                    egui::Frame::none()
                        .fill(p.bg)
                        .rounding(egui::Rounding::same(8.0))
                        .stroke(egui::Stroke::new(1.0, p.teal))
                        .inner_margin(egui::Margin::same(10.0)),
                )
                .show(ctx, |ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} failure(s) since launch · {} log entries · poll every {}s",
                            fail_count,
                            log_snap.len(),
                            POLL_SECS
                        ))
                        .size(10.0)
                        .color(p.text),
                    );
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for entry in log_snap.iter().rev() {
                            let color = match entry.severity {
                                Sev::Info => p.text,
                                Sev::Warn => p.amber,
                                Sev::Crit => p.terracotta,
                            };
                            let tag = match entry.severity {
                                Sev::Info => "INFO",
                                Sev::Warn => "WARN",
                                Sev::Crit => "CRIT",
                            };
                            ui.horizontal_wrapped(|ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&entry.timestamp)
                                            .size(8.0)
                                            .color(p.slate),
                                    )
                                    .selectable(true),
                                );
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(tag)
                                            .size(8.0)
                                            .color(color)
                                            .strong(),
                                    )
                                    .selectable(true),
                                );
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&entry.message)
                                            .size(9.0)
                                            .color(color),
                                    )
                                    .selectable(true)
                                    .wrap(),
                                );
                            });
                            ui.add_space(2.0);
                        }
                    });
                    ui.separator();
                    if ui
                        .button(egui::RichText::new("copy").size(9.0).color(p.teal))
                        .clicked()
                    {
                        let buf: String = log_snap
                            .iter()
                            .map(|e| {
                                let tag = match e.severity {
                                    Sev::Info => "INFO",
                                    Sev::Warn => "WARN",
                                    Sev::Crit => "CRIT",
                                };
                                format!("{}  {}  {}", e.timestamp, tag, e.message)
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        ui.ctx().copy_text(buf);
                    }
                });
            if !still_open {
                self.show_log_window = false;
            }
        }
    }
}

// =============================================================================
// Widgets / UI Components — Group Rendering, Tooltip Builder
// =============================================================================

fn render_group(
    ui: &mut egui::Ui,
    p: &Palette,
    group: Group,
    probes: &HashMap<&'static str, ProbeState>,
) {
    ui.label(
        egui::RichText::new(group.label())
            .size(9.0)
            .color(p.text_dim)
            .strong(),
    );
    let frame = egui::Frame::none()
        .fill(p.bg_row)
        .rounding(egui::Rounding::same(6.0))
        .inner_margin(egui::Margin::symmetric(8.0, 6.0));
    frame.show(ui, |ui| {
        for probe in PROBES.iter().filter(|p| p.group == group) {
            let st = probes.get(probe.id).cloned().unwrap_or_default();
            let color = st.led_color(*p);
            let pulsing = st.ok == Some(false);
            let tip = build_tip(probe, &st);

            ui.horizontal(|ui| {
                draw_led(ui, color, 4.0, pulsing).on_hover_text(&tip);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(probe.label).size(9.5).color(p.text),
                    )
                    .selectable(false),
                )
                .on_hover_text(&tip);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let right = match (st.ok, st.latency_ms) {
                        (Some(true), Some(ms)) => format!("{ms}ms"),
                        (Some(false), _) => "DOWN".to_string(),
                        _ => "—".to_string(),
                    };
                    let right_color = match st.ok {
                        Some(true) => p.text_dim,
                        Some(false) => p.terracotta,
                        None => p.slate,
                    };
                    ui.label(
                        egui::RichText::new(right)
                            .size(8.5)
                            .color(right_color)
                            .monospace(),
                    )
                    .on_hover_text(&tip);
                });
            });
        }
    });
}

fn build_tip(probe: &Probe, st: &ProbeState) -> String {
    let mut out = String::new();
    out.push_str(probe.label);
    out.push('\n');
    out.push_str(probe.desc);
    out.push_str("\n\n");
    let target = match probe.kind {
        ProbeKind::Http(u) | ProbeKind::HttpAny(u) => format!("GET {u}"),
        ProbeKind::Tcp(a) => format!("TCP {a}"),
    };
    out.push_str(&target);
    out.push('\n');
    match (st.ok, st.http_code, st.latency_ms) {
        (Some(true), Some(code), Some(ms)) => {
            out.push_str(&format!("→ HTTP {code} · {ms}ms"));
        }
        (Some(true), None, Some(ms)) => {
            out.push_str(&format!("→ connected in {ms}ms"));
        }
        (Some(false), _, _) => {
            let err = st.last_error.as_deref().unwrap_or("unknown error");
            out.push_str(&format!("→ FAIL: {err}"));
        }
        _ => out.push_str("→ no probe data yet"),
    }
    if let Some(last_ok) = st.last_ok {
        let ago = last_ok.elapsed().as_secs();
        out.push_str(&format!("\nlast ok: {}s ago", ago));
    }
    out
}

// =============================================================================
// Helpers
// =============================================================================

fn now() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

fn draw_led(ui: &mut egui::Ui, color: egui::Color32, radius: f32, pulse: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(radius * 2.5, radius * 2.5),
        egui::Sense::click(),
    );
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
    response
}
