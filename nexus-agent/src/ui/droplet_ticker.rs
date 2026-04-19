//! nexus-droplet-ticker — DigitalOcean GPU Droplet Desktop Widget
//!
//! eframe-based frameless desktop widget for monitoring + one-click
//! lifecycle control of the user's ephemeral DigitalOcean GPU droplets
//! (H100 / H200 class, ~$3-4/hr). The lifecycle pattern is
//! **spin-up → train → destroy**; the ticker keeps the user aware of
//! running cost in real time and gives them lifecycle buttons so a
//! forgotten droplet can't silently bleed credit.
//!
//! Mirrors the visual language of `training_ticker.rs` (frameless
//! transparent rounded pill, custom drag-region titlebar, dark/light
//! theme toggle, hand-rolled JSON parser, dotted-line history plot),
//! but swaps the training control socket for:
//!
//! * **DigitalOcean REST API** — called via `/usr/bin/curl` so no new
//!   crate dependencies are introduced. Every call captures the HTTP
//!   status code (`--write-out '%{http_code}'`) and appends to the
//!   action log. Token sourced from `$DO_API_TOKEN`; if unset the
//!   ticker runs in read-only "no-token" mode with write buttons
//!   disabled.
//! * **SSH-exec'd GPU + CPU probes** — poll every 4 s by running
//!   `nvidia-smi` + `/proc/stat` + `/proc/meminfo` via `/usr/bin/ssh`
//!   against `$NEXUS_DROPLET_HOST` (key-based auth expected from the
//!   local user's ssh-agent).
//! * **Billing math** — session charge = `size.price_hourly * uptime`
//!   since the droplet most recent `status=active` transition.
//!   MTD charge = `month_to_date_usage` from
//!   `/v2/customers/my/balance`. Ticks every 10s.
//!
//! # Environment variables
//!
//! * `DO_API_TOKEN`              — required for write actions (POST/DELETE).
//!                                 Read-only without it.
//! * `NEXUS_DROPLET_HOST`        — SSH target for live metrics,
//!                                 e.g. `root@192.241.187.102`.
//! * `NEXUS_DROPLET_ID`          — optional. Pins the ticker to one droplet
//!                                 by id. Without it, the ticker auto-attaches
//!                                 to the single GPU droplet on the account,
//!                                 or offers a picker if there's more than one.
//! * `NEXUS_DROPLET_SNAPSHOT_ID` — snapshot image id used by the `Start` button
//!                                 when no droplet is active.
//! * `NEXUS_DROPLET_SIZE`        — slug for new droplets; default `gpu-h200x1-141gb`.
//! * `NEXUS_DROPLET_REGION`      — region slug; default `nyc2`.
//! * `NEXUS_DROPLET_NAME`        — name for newly provisioned droplets;
//!                                 default `nexus-gpu-<yyyymmddThhmmss>`.
//!
//! # Layout
//!
//! Everything sized from `ui.available_rect_before_wrap()` — no
//! hardcoded pixel positions. Default window 480×520; resizable down
//! to 360×400 with reflow.
//!
//! 1. Titlebar                              (drag, ─, ☀/☾, ✕)
//! 2. Droplet identity                      (name · region · size · status LED)
//! 3. Metrics strip                         (GPU%, VRAM%, CPU%, RAM%, temp, W)
//! 4. GPU% history dotted plot              (last 5 min; collapses below 420px tall)
//! 5. Billing panel                         (session $ · runtime · rate · MTD $)
//! 6. Action row                            (Start · Stop · Shutdown · Destroy)
//! 7. Log panel                             (scrollable ring buffer)
//!
//! # Launch
//!
//! ```bash
//! DO_API_TOKEN=dop_v1_xxxx NEXUS_DROPLET_HOST=root@1.2.3.4 \
//!   nexus-droplet-ticker &
//! ```
//!
//! Single-instance via `/tmp/.nexus-droplet-ticker.lock`. Second start
//! exits immediately.
//!
//! # systemd user service (auto-start on login, always-on while GPU is live)
//!
//! ```ini
//! # ~/.config/systemd/user/nexus-droplet-ticker.service
//! [Unit]
//! Description=nexus-droplet-ticker — DO GPU droplet monitor
//! After=graphical-session.target
//!
//! [Service]
//! Environment=DO_API_TOKEN=dop_v1_xxxx
//! Environment=NEXUS_DROPLET_HOST=root@1.2.3.4
//! Environment=NEXUS_DROPLET_SNAPSHOT_ID=123456789
//! ExecStart=/opt/AxonML/nexus-agent/target/release/nexus-droplet-ticker
//! Restart=always
//! RestartSec=5
//!
//! [Install]
//! WantedBy=default.target
//! ```
//!
//! Install:
//! ```bash
//! systemctl --user daemon-reload
//! systemctl --user enable --now nexus-droplet-ticker.service
//! ```
//!
//! # File
//! `nexus-agent/src/ui/droplet_ticker.rs`
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
use std::io::Write;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// =============================================================================
// Constants
// =============================================================================

const TICKER_WIDTH: f32 = 480.0;
const TICKER_HEIGHT: f32 = 520.0;
const MIN_WIDTH: f32 = 360.0;
const MIN_HEIGHT: f32 = 400.0;
const PLOT_COLLAPSE_BELOW: f32 = 420.0;

const MAX_LOG_LINES: usize = 300;
const MAX_GPU_HISTORY: usize = 300; // 300 samples × 4s = 20 min ceiling; plot the last 5 min

const METRICS_POLL_SECS: f32 = 4.0;
const DROPLET_POLL_SECS: f32 = 10.0;
const BILLING_POLL_SECS: f32 = 60.0; // balance endpoint is rate-limited

const LOCK_FILE: &str = "/tmp/.nexus-droplet-ticker.lock";
const THEME_FILE: &str = "/tmp/.nexus-droplet-ticker-theme";
const STATE_DIR_REL: &str = ".config/nexus-droplet-ticker";
const STATE_FILE_NAME: &str = "state.toml";

const DO_API_BASE: &str = "https://api.digitalocean.com/v2";

// =============================================================================
// Entry Point
// =============================================================================

fn main() -> eframe::Result {
    // ---- Single-instance lock ----
    if let Err(e) = acquire_single_instance_lock() {
        eprintln!("nexus-droplet-ticker: another instance is already running ({e}); exiting.");
        std::process::exit(0);
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([TICKER_WIDTH, TICKER_HEIGHT])
            .with_min_inner_size([MIN_WIDTH, MIN_HEIGHT])
            .with_always_on_top()
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(true)
            .with_title("nexus-droplet-ticker"),
        ..Default::default()
    };

    eframe::run_native(
        "nexus-droplet-ticker",
        options,
        Box::new(|cc| Ok(Box::new(DropletTickerApp::new(cc)))),
    )
}

/// Writes this process's PID into `LOCK_FILE`. If the file already
/// exists and the PID inside is still alive, refuse to start. Otherwise
/// overwrite and proceed. We don't use `flock` because the process may
/// be killed via SIGKILL and leave a stale lock — a PID-liveness check
/// recovers gracefully.
fn acquire_single_instance_lock() -> Result<(), String> {
    if let Ok(txt) = std::fs::read_to_string(LOCK_FILE) {
        if let Ok(pid) = txt.trim().parse::<i32>() {
            // Signal 0 == liveness probe on Linux.
            let alive = libc_kill_stub(pid);
            if alive {
                return Err(format!("pid {pid} is alive"));
            }
        }
    }
    let mut f = std::fs::File::create(LOCK_FILE).map_err(|e| e.to_string())?;
    write!(f, "{}", std::process::id()).map_err(|e| e.to_string())?;
    Ok(())
}

/// Minimal PID-liveness probe without pulling in `libc`. Issues a
/// `kill -0 <pid>` via /bin/kill and checks the exit code. Zero means
/// the process is alive (or we don't have permission to signal it,
/// which for our purposes still means "don't clobber").
fn libc_kill_stub(pid: i32) -> bool {
    Command::new("/bin/kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// =============================================================================
// State
// =============================================================================

#[derive(Clone, PartialEq, Debug)]
enum DropletState {
    Unknown,
    NoToken,
    NoDroplet,
    New,        // newly created, still provisioning
    Active,
    Off,
    Archive,
    Transitioning(String), // e.g. "powering-off", "destroying"
    Error(String),
}

impl DropletState {
    fn label(&self) -> String {
        match self {
            Self::Unknown => "UNKNOWN".to_string(),
            Self::NoToken => "NO TOKEN".to_string(),
            Self::NoDroplet => "NO DROPLET".to_string(),
            Self::New => "PROVISIONING".to_string(),
            Self::Active => "ACTIVE".to_string(),
            Self::Off => "OFF".to_string(),
            Self::Archive => "ARCHIVE".to_string(),
            Self::Transitioning(s) => s.to_uppercase(),
            Self::Error(_) => "ERROR".to_string(),
        }
    }
    fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
    fn is_off(&self) -> bool {
        matches!(self, Self::Off)
    }
    fn is_transitional(&self) -> bool {
        matches!(self, Self::New | Self::Transitioning(_))
    }
}

#[derive(Clone, Default)]
#[allow(dead_code)] // memory_mb / vcpus / disk_gb are parsed for future expansion
struct Droplet {
    id: u64,
    name: String,
    region: String,
    size_slug: String,
    price_hourly: f64,
    status: String,              // raw DO status: "new" / "active" / "off" / "archive"
    ip_public: String,
    memory_mb: u64,
    vcpus: u32,
    disk_gb: u64,
    gpu_label: String,           // human-facing size summary ("H200 · 141GB VRAM")
    /// Unix-seconds timestamp when we first observed this droplet in
    /// `status=active` during this ticker session. Reset when the
    /// droplet transitions away from active.
    active_since: u64,
}

#[derive(Clone, Default)]
struct LiveMetrics {
    /// Last time we got a fresh sample, as unix seconds.
    sampled_at: u64,
    gpu_util_pct: f32,           // 0..=100
    vram_used_mb: u64,
    vram_total_mb: u64,
    gpu_temp_c: f32,
    gpu_power_w: f32,
    cpu_util_pct: f32,           // 0..=100, from /proc/stat delta
    ram_used_mb: u64,
    ram_total_mb: u64,
}

/// A single GPU% sample for the history plot.
#[derive(Clone, Copy)]
struct GpuSample {
    ts: u64,       // unix seconds
    gpu_pct: f32,  // 0..=100
}

#[derive(Clone)]
struct LogEntry {
    timestamp: String,
    message: String,
    is_error: bool,
}

/// CPU delta helpers — `/proc/stat` gives cumulative jiffies; we store
/// the previous snapshot so we can compute % between polls.
#[derive(Clone, Copy, Default)]
struct CpuSnapshot {
    total: u64,
    idle: u64,
}

#[derive(Clone, Default)]
#[allow(dead_code)] // generated_at reserved for stale-data surfacing
struct Billing {
    month_to_date_usage: f64,    // USD
    account_balance: f64,        // USD
    generated_at: String,
}

struct SharedState {
    token_present: bool,
    state: DropletState,
    droplet: Droplet,
    metrics: LiveMetrics,
    gpu_history: Vec<GpuSample>,
    last_cpu_snapshot: Option<CpuSnapshot>,
    billing: Billing,
    log: Vec<LogEntry>,
    destroy_confirm_open: bool,
    destroy_snapshot_first: bool,
    /// Cached droplet picker list when more than one GPU droplet is on the account.
    droplet_options: Vec<Droplet>,
    /// If Some, a worker action is in flight and the button row should grey out.
    pending_action: Option<String>,
    /// Persisted state (droplet id we're pinned to).
    pinned_droplet_id: Option<u64>,
    /// Session runtime seconds, computed in UI render.
    session_seconds: u64,
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

struct DropletTickerApp {
    state: Arc<Mutex<SharedState>>,
    last_droplet_poll: Instant,
    last_metrics_poll: Instant,
    last_billing_poll: Instant,
    ssh_host: String,
}

impl DropletTickerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let token_present = std::env::var("DO_API_TOKEN").is_ok();
        let pinned = load_persisted_droplet_id();

        let boot_msg = if token_present {
            "ticker started — DO_API_TOKEN present; polling DigitalOcean".to_string()
        } else {
            "ticker started — DO_API_TOKEN missing; read-only mode (actions disabled)".to_string()
        };

        let state = Arc::new(Mutex::new(SharedState {
            token_present,
            state: if token_present { DropletState::Unknown } else { DropletState::NoToken },
            droplet: Droplet::default(),
            metrics: LiveMetrics::default(),
            gpu_history: Vec::new(),
            last_cpu_snapshot: None,
            billing: Billing::default(),
            log: vec![LogEntry {
                timestamp: now_hms(),
                message: boot_msg,
                is_error: false,
            }],
            destroy_confirm_open: false,
            destroy_snapshot_first: true,
            droplet_options: Vec::new(),
            pending_action: None,
            pinned_droplet_id: pinned,
            session_seconds: 0,
        }));

        let ssh_host = std::env::var("NEXUS_DROPLET_HOST").unwrap_or_default();
        if let Ok(mut s) = state.lock() {
            if !ssh_host.is_empty() {
                s.push_log(format!("ssh target: {ssh_host}"), false);
            } else {
                s.push_log("NEXUS_DROPLET_HOST unset — live GPU metrics disabled", false);
            }
            if let Some(id) = pinned {
                s.push_log(format!("restored pinned droplet {id} from state file"), false);
            }
        }

        Self {
            state,
            last_droplet_poll: Instant::now() - Duration::from_secs(60),
            last_metrics_poll: Instant::now() - Duration::from_secs(60),
            last_billing_poll: Instant::now() - Duration::from_secs(60),
            ssh_host,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Droplet discovery + state refresh
    // ─────────────────────────────────────────────────────────────────────

    fn refresh_droplet(&self) {
        let (token_present, pinned, pinned_existing) = {
            let s = self.state.lock().unwrap();
            (s.token_present, s.pinned_droplet_id, s.droplet.id)
        };
        if !token_present {
            return;
        }

        // Prefer the pinned id → single-droplet fetch.
        let pinned = pinned.or(if pinned_existing != 0 { Some(pinned_existing) } else { None })
            .or_else(|| std::env::var("NEXUS_DROPLET_ID").ok().and_then(|s| s.parse().ok()));

        if let Some(id) = pinned {
            let url = format!("{DO_API_BASE}/droplets/{id}");
            match do_get(&url) {
                Ok((200, body)) => {
                    if let Some(d) = parse_single_droplet(&body) {
                        self.apply_droplet(d);
                    }
                }
                Ok((404, _)) => {
                    if let Ok(mut s) = self.state.lock() {
                        s.state = DropletState::NoDroplet;
                        s.push_log(format!("pinned droplet {id} returned 404 — dropping pin"), true);
                        s.pinned_droplet_id = None;
                        clear_persisted_droplet_id();
                    }
                }
                Ok((code, _)) => {
                    if let Ok(mut s) = self.state.lock() {
                        s.state = DropletState::Error(format!("GET droplets/{id} → {code}"));
                    }
                }
                Err(e) => {
                    if let Ok(mut s) = self.state.lock() {
                        s.push_log(format!("curl error on GET droplets/{id}: {e}"), true);
                    }
                }
            }
            return;
        }

        // No pin yet — list all droplets, filter to GPU types.
        let url = format!("{DO_API_BASE}/droplets?per_page=200");
        match do_get(&url) {
            Ok((200, body)) => {
                let droplets = parse_droplet_list(&body);
                let gpu: Vec<Droplet> = droplets
                    .into_iter()
                    .filter(|d| d.size_slug.starts_with("gpu-"))
                    .collect();
                if let Ok(mut s) = self.state.lock() {
                    s.droplet_options = gpu.clone();
                    match gpu.len() {
                        0 => {
                            s.state = DropletState::NoDroplet;
                            s.droplet = Droplet::default();
                        }
                        1 => {
                            // Auto-pin the lone droplet.
                            let d = gpu.into_iter().next().unwrap();
                            let id = d.id;
                            drop(s);
                            self.apply_droplet(d);
                            save_persisted_droplet_id(id);
                            if let Ok(mut s) = self.state.lock() {
                                s.pinned_droplet_id = Some(id);
                                s.push_log(format!("auto-pinned droplet {id}"), false);
                            }
                        }
                        n => {
                            s.state = DropletState::NoDroplet;
                            s.push_log(
                                format!("{n} GPU droplets found — pick one from the picker"),
                                false,
                            );
                        }
                    }
                }
            }
            Ok((code, _)) => {
                if let Ok(mut s) = self.state.lock() {
                    s.state = DropletState::Error(format!("GET droplets → {code}"));
                }
            }
            Err(e) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("curl error on GET droplets: {e}"), true);
                }
            }
        }
    }

    /// Fold a freshly-parsed droplet into shared state. Handles
    /// detecting the active→* transition so we can reset `active_since`
    /// correctly.
    fn apply_droplet(&self, mut d: Droplet) {
        let now_sec = unix_now();
        if let Ok(mut s) = self.state.lock() {
            let was_active = s.droplet.status == "active" && s.droplet.id == d.id;
            let now_active = d.status == "active";

            if now_active {
                if was_active && s.droplet.active_since != 0 {
                    d.active_since = s.droplet.active_since;
                } else {
                    d.active_since = now_sec;
                    if !was_active {
                        s.push_log(format!("droplet {} is now ACTIVE", d.id), false);
                    }
                }
            } else {
                // Drop the counter when leaving active.
                d.active_since = 0;
            }

            // Humanise size slug.
            d.gpu_label = humanise_size(&d.size_slug);

            s.state = match d.status.as_str() {
                "new" => DropletState::New,
                "active" => DropletState::Active,
                "off" => DropletState::Off,
                "archive" => DropletState::Archive,
                other => DropletState::Transitioning(other.to_string()),
            };
            s.droplet = d;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Live GPU/CPU metrics via SSH
    // ─────────────────────────────────────────────────────────────────────

    fn poll_live_metrics(&self) {
        let (is_active, host) = {
            let s = self.state.lock().unwrap();
            (s.state.is_active(), self.ssh_host.clone())
        };
        if !is_active || host.is_empty() {
            return;
        }

        // Pack all probes into one ssh call — saves round-trips.
        // Note: nvidia-smi will fail on non-GPU hosts; we handle that by
        // treating the whole block as best-effort and parsing whatever
        // fields we recognise.
        let remote_cmd = "\
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null; \
            echo '---CPU---'; \
            head -1 /proc/stat; \
            echo '---MEM---'; \
            head -3 /proc/meminfo";
        let out = Command::new("/usr/bin/ssh")
            .args([
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=4",
                &host,
                remote_cmd,
            ])
            .output();

        let body = match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            Ok(o) => {
                let err = String::from_utf8_lossy(&o.stderr).to_string();
                // Only log on transition to avoid spamming.
                if let Ok(mut s) = self.state.lock() {
                    if s.metrics.sampled_at != 0 {
                        s.metrics.sampled_at = 0;
                        s.push_log(format!("ssh probe failed: {}", err.lines().next().unwrap_or("")), true);
                    }
                }
                return;
            }
            Err(e) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("ssh error: {e}"), true);
                }
                return;
            }
        };

        let mut gpu_util = 0.0f32;
        let mut vram_used = 0u64;
        let mut vram_total = 0u64;
        let mut gpu_temp = 0.0f32;
        let mut gpu_power = 0.0f32;
        let mut cpu_snapshot: Option<CpuSnapshot> = None;
        let mut mem_total_kb = 0u64;
        let mut mem_avail_kb = 0u64;

        let mut section = "gpu";
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed == "---CPU---" {
                section = "cpu";
                continue;
            }
            if trimmed == "---MEM---" {
                section = "mem";
                continue;
            }
            match section {
                "gpu" => {
                    if trimmed.is_empty() {
                        continue;
                    }
                    let parts: Vec<&str> = trimmed.split(',').map(str::trim).collect();
                    if parts.len() >= 5 {
                        gpu_util = parts[0].parse().unwrap_or(0.0);
                        vram_used = parts[1].parse().unwrap_or(0);
                        vram_total = parts[2].parse().unwrap_or(0);
                        gpu_temp = parts[3].parse().unwrap_or(0.0);
                        gpu_power = parts[4].parse().unwrap_or(0.0);
                    }
                }
                "cpu" => {
                    // Line shape: "cpu  12345 67 890 112233 ..."
                    if trimmed.starts_with("cpu ") || trimmed.starts_with("cpu\t") {
                        let nums: Vec<u64> = trimmed
                            .split_whitespace()
                            .skip(1)
                            .map(|s| s.parse::<u64>().unwrap_or(0))
                            .collect();
                        if nums.len() >= 5 {
                            let idle = nums[3] + nums.get(4).copied().unwrap_or(0); // idle + iowait
                            let total: u64 = nums.iter().sum();
                            cpu_snapshot = Some(CpuSnapshot { total, idle });
                        }
                    }
                }
                "mem" => {
                    // Lines: "MemTotal: N kB" / "MemFree: N kB" / "MemAvailable: N kB"
                    if let Some(rest) = trimmed.strip_prefix("MemTotal:") {
                        mem_total_kb = rest.trim().split_whitespace().next()
                            .and_then(|s| s.parse().ok()).unwrap_or(0);
                    } else if let Some(rest) = trimmed.strip_prefix("MemAvailable:") {
                        mem_avail_kb = rest.trim().split_whitespace().next()
                            .and_then(|s| s.parse().ok()).unwrap_or(0);
                    }
                }
                _ => {}
            }
        }

        let ram_total_mb = mem_total_kb / 1024;
        let ram_used_mb = if mem_total_kb > mem_avail_kb {
            (mem_total_kb - mem_avail_kb) / 1024
        } else {
            0
        };

        // CPU utilisation is a delta between samples.
        let cpu_util = if let (Some(prev), Some(curr)) =
            (self.state.lock().unwrap().last_cpu_snapshot, cpu_snapshot)
        {
            let d_total = curr.total.saturating_sub(prev.total) as f32;
            let d_idle = curr.idle.saturating_sub(prev.idle) as f32;
            if d_total > 0.0 {
                ((d_total - d_idle) / d_total * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let now_sec = unix_now();
        if let Ok(mut s) = self.state.lock() {
            s.metrics = LiveMetrics {
                sampled_at: now_sec,
                gpu_util_pct: gpu_util,
                vram_used_mb: vram_used,
                vram_total_mb: vram_total,
                gpu_temp_c: gpu_temp,
                gpu_power_w: gpu_power,
                cpu_util_pct: cpu_util,
                ram_used_mb,
                ram_total_mb,
            };
            s.last_cpu_snapshot = cpu_snapshot;
            s.gpu_history.push(GpuSample { ts: now_sec, gpu_pct: gpu_util });
            let excess = s.gpu_history.len().saturating_sub(MAX_GPU_HISTORY);
            if excess > 0 {
                s.gpu_history.drain(0..excess);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Billing
    // ─────────────────────────────────────────────────────────────────────

    fn poll_billing(&self) {
        let token_present = self.state.lock().unwrap().token_present;
        if !token_present {
            return;
        }
        let url = format!("{DO_API_BASE}/customers/my/balance");
        match do_get(&url) {
            Ok((200, body)) => {
                let mtd = extract_json_f64(&body, "month_to_date_usage").unwrap_or(0.0);
                let bal = extract_json_f64(&body, "account_balance").unwrap_or(0.0);
                let generated = extract_json_string(&body, "generated_at").unwrap_or_default();
                if let Ok(mut s) = self.state.lock() {
                    s.billing = Billing {
                        month_to_date_usage: mtd,
                        account_balance: bal,
                        generated_at: generated,
                    };
                }
            }
            Ok((code, _)) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("GET balance → {code}"), code >= 400);
                }
            }
            Err(e) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("curl error on GET balance: {e}"), true);
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Actions (Start / Stop / Shutdown / Destroy)
    // ─────────────────────────────────────────────────────────────────────

    fn send_action(&self, action_type: &'static str) {
        let (token_present, id) = {
            let s = self.state.lock().unwrap();
            (s.token_present, s.droplet.id)
        };
        if !token_present || id == 0 {
            return;
        }

        if let Ok(mut s) = self.state.lock() {
            s.pending_action = Some(action_type.to_string());
            s.state = DropletState::Transitioning(action_type.to_string());
        }

        let url = format!("{DO_API_BASE}/droplets/{id}/actions");
        let body = format!(r#"{{"type":"{action_type}"}}"#);
        match do_post(&url, &body) {
            Ok((code, resp)) => {
                if let Ok(mut s) = self.state.lock() {
                    let note = format!("POST /v2/droplets/{id}/actions {action_type} → {code}");
                    s.push_log(note, !(200..300).contains(&code));
                    let _ = resp; // body contains the action object; not needed for display
                    s.pending_action = None;
                }
            }
            Err(e) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("action {action_type} failed: {e}"), true);
                    s.pending_action = None;
                }
            }
        }
    }

    fn start_droplet_from_snapshot(&self) {
        let token_present = self.state.lock().unwrap().token_present;
        if !token_present {
            return;
        }
        let snapshot_id = match std::env::var("NEXUS_DROPLET_SNAPSHOT_ID") {
            Ok(s) => s,
            Err(_) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log("NEXUS_DROPLET_SNAPSHOT_ID unset — can't provision", true);
                }
                return;
            }
        };
        let size = std::env::var("NEXUS_DROPLET_SIZE")
            .unwrap_or_else(|_| "gpu-h200x1-141gb".to_string());
        let region = std::env::var("NEXUS_DROPLET_REGION")
            .unwrap_or_else(|_| "nyc2".to_string());
        let name = std::env::var("NEXUS_DROPLET_NAME")
            .unwrap_or_else(|_| format!("nexus-gpu-{}", now_stamp_compact()));

        // Fetch ssh keys to include them all.
        let key_ids = match do_get(&format!("{DO_API_BASE}/account/keys?per_page=200")) {
            Ok((200, body)) => parse_ssh_key_ids(&body),
            _ => Vec::new(),
        };

        let ssh_array = key_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let body = format!(
            r#"{{"name":"{name}","region":"{region}","size":"{size}","image":{snapshot_id},"ssh_keys":[{ssh_array}]}}"#,
        );

        if let Ok(mut s) = self.state.lock() {
            s.pending_action = Some("create".to_string());
            s.state = DropletState::Transitioning("creating".to_string());
            s.push_log(format!("POST /v2/droplets {name} ({size} in {region})"), false);
        }

        match do_post(&format!("{DO_API_BASE}/droplets"), &body) {
            Ok((code, resp)) => {
                let id = parse_single_droplet(&resp).map(|d| d.id).unwrap_or(0);
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("POST /v2/droplets → {code} (id={id})"), code >= 400);
                    s.pending_action = None;
                    if id != 0 {
                        s.pinned_droplet_id = Some(id);
                    }
                }
                if id != 0 {
                    save_persisted_droplet_id(id);
                }
            }
            Err(e) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("create failed: {e}"), true);
                    s.pending_action = None;
                }
            }
        }
    }

    fn destroy_droplet(&self, snapshot_first: bool) {
        let (token_present, id, d_name) = {
            let s = self.state.lock().unwrap();
            (s.token_present, s.droplet.id, s.droplet.name.clone())
        };
        if !token_present || id == 0 {
            return;
        }

        if let Ok(mut s) = self.state.lock() {
            s.pending_action = Some("destroy".to_string());
        }

        if snapshot_first {
            // 1) shutdown (graceful). We don't block on completion here — the
            //    DO docs are clear that shutdown-then-snapshot-then-destroy
            //    needs the droplet in `off` state. We kick the shutdown and
            //    let subsequent polls pick up the transition. The snapshot
            //    action itself is also async on DO's side.
            if let Ok(mut s) = self.state.lock() {
                s.push_log(format!("destroy flow: shutdown {d_name}"), false);
            }
            let shutdown_url = format!("{DO_API_BASE}/droplets/{id}/actions");
            let _ = do_post(&shutdown_url, r#"{"type":"shutdown"}"#);

            let snap_name = format!("nexus-{}-{}", short_size(&d_name), now_stamp_compact());
            let snap_body = format!(r#"{{"type":"snapshot","name":"{snap_name}"}}"#);
            if let Ok(mut s) = self.state.lock() {
                s.push_log(format!("destroy flow: snapshot → {snap_name}"), false);
            }
            match do_post(&shutdown_url, &snap_body) {
                Ok((code, _)) => {
                    if let Ok(mut s) = self.state.lock() {
                        s.push_log(format!("POST snapshot → {code}"), code >= 400);
                    }
                }
                Err(e) => {
                    if let Ok(mut s) = self.state.lock() {
                        s.push_log(format!("snapshot failed: {e}"), true);
                    }
                }
            }
        }

        // Final step: DELETE the droplet.
        let del_url = format!("{DO_API_BASE}/droplets/{id}");
        match do_delete(&del_url) {
            Ok((code, _)) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("DELETE /v2/droplets/{id} → {code}"), code >= 400);
                    s.pending_action = None;
                    if (200..300).contains(&code) {
                        s.droplet = Droplet::default();
                        s.state = DropletState::NoDroplet;
                        s.pinned_droplet_id = None;
                        clear_persisted_droplet_id();
                    }
                }
            }
            Err(e) => {
                if let Ok(mut s) = self.state.lock() {
                    s.push_log(format!("destroy failed: {e}"), true);
                    s.pending_action = None;
                }
            }
        }
    }
}

// =============================================================================
// HTTP via curl — zero-dep API client
// =============================================================================

/// Returns `(http_status, body)` on successful curl exec (the exit code
/// just means curl ran; HTTP errors still come back here). On curl exec
/// failure returns `Err(msg)`.
fn do_get(url: &str) -> Result<(u16, String), String> {
    do_curl("GET", url, None)
}
fn do_post(url: &str, body: &str) -> Result<(u16, String), String> {
    do_curl("POST", url, Some(body))
}
fn do_delete(url: &str) -> Result<(u16, String), String> {
    do_curl("DELETE", url, None)
}

fn do_curl(method: &str, url: &str, body: Option<&str>) -> Result<(u16, String), String> {
    let token = std::env::var("DO_API_TOKEN").unwrap_or_default();

    // We ask curl to append a marker + status code to stdout so we can
    // separate the HTTP body from the status in a single round-trip
    // without needing --dump-header etc.
    //
    //     <body><\n__HTTP_CODE__:<status>
    //
    // Works cleanly even when the body has no trailing newline.
    const MARKER: &str = "\n__HTTP_CODE__:";

    let mut cmd = Command::new("/usr/bin/curl");
    cmd.args([
        "-sS",                      // silent but show errors on stderr
        "-X", method,
        "-H", &format!("Authorization: Bearer {token}"),
        "-H", "Content-Type: application/json",
        "--max-time", "20",
        "--write-out", &format!("{MARKER}%{{http_code}}"),
    ]);
    if let Some(b) = body {
        cmd.args(["-d", b]);
    }
    cmd.arg(url);

    let out = cmd.output().map_err(|e| format!("curl exec: {e}"))?;
    if !out.status.success() && out.stdout.is_empty() {
        return Err(format!(
            "curl exit {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let (body_part, code_part) = match text.rfind(MARKER) {
        Some(i) => (text[..i].to_string(), text[i + MARKER.len()..].to_string()),
        None => (text.clone(), "0".to_string()),
    };
    let code: u16 = code_part.trim().parse().unwrap_or(0);
    Ok((code, body_part))
}

// =============================================================================
// JSON parsing — hand-rolled to avoid adding serde_json to the build graph
// (reqwest/tokio/serde_json are already workspace deps for other bins but
// keeping this file parser-free keeps its surface area tiny).
// =============================================================================

fn extract_json_string(text: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let mut cursor = 0;
    while let Some(pos) = text[cursor..].find(&pat) {
        let abs = cursor + pos + pat.len();
        let rest = &text[abs..];
        if let Some(col) = rest.find(':') {
            let after = rest[col + 1..].trim_start();
            if let Some(inner) = after.strip_prefix('"') {
                if let Some(end) = find_unescaped_quote(inner) {
                    return Some(inner[..end].to_string());
                }
            } else if after.starts_with("null") {
                return None;
            } else {
                // Bare value (number/bool) — not a string.
                return None;
            }
        }
        cursor = abs;
    }
    None
}

fn extract_json_number<T: std::str::FromStr>(text: &str, key: &str) -> Option<T> {
    let pat = format!("\"{key}\"");
    let start = text.find(&pat)? + pat.len();
    let rest = &text[start..];
    let colon = rest.find(':')? + 1;
    let after = rest[colon..].trim_start();
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

fn extract_json_f64(text: &str, key: &str) -> Option<f64> {
    extract_json_number::<f64>(text, key)
}

/// Walk a string looking for the first unescaped `"` character. Assumes
/// the input has already had the opening quote stripped.
fn find_unescaped_quote(s: &str) -> Option<usize> {
    let mut prev_slash = false;
    for (i, c) in s.char_indices() {
        if c == '"' && !prev_slash {
            return Some(i);
        }
        prev_slash = c == '\\' && !prev_slash;
    }
    None
}

/// Parse a single-droplet response. The body is shaped
/// `{"droplet": { ... }}`. Extract the inner object and pull out the
/// fields we care about. On `POST /v2/droplets`, the top-level key is
/// also `droplet` (not `droplets`), so this helper works for both.
fn parse_single_droplet(body: &str) -> Option<Droplet> {
    let obj = slice_inner_object(body, "\"droplet\"")?;
    parse_droplet_obj(obj)
}

/// Parse `{"droplets":[ ... ]}` into a vector of droplets.
fn parse_droplet_list(body: &str) -> Vec<Droplet> {
    let arr = match slice_inner_array(body, "\"droplets\"") {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let bytes = arr.as_bytes();
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(s) = start {
                        let obj = &arr[s..=i];
                        if let Some(d) = parse_droplet_obj(obj) {
                            out.push(d);
                        }
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Given a JSON text and a quoted key like `"droplet"`, find the
/// object that is that key's value and return it as a slice including
/// the surrounding braces.
fn slice_inner_object<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let k = text.find(key)?;
    let after_key = &text[k + key.len()..];
    let colon = after_key.find(':')?;
    let after = &after_key[colon + 1..];
    let brace = after.find('{')?;
    let start_abs = (k + key.len() + colon + 1) + brace;
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escape = false;
    for i in start_abs..bytes.len() {
        let b = bytes[i];
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start_abs..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn slice_inner_array<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let k = text.find(key)?;
    let after_key = &text[k + key.len()..];
    let colon = after_key.find(':')?;
    let after = &after_key[colon + 1..];
    let bracket = after.find('[')?;
    let start_abs = (k + key.len() + colon + 1) + bracket;
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escape = false;
    for i in start_abs..bytes.len() {
        let b = bytes[i];
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start_abs + 1..i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_droplet_obj(obj: &str) -> Option<Droplet> {
    let id = extract_json_number::<u64>(obj, "id")?;
    let name = extract_json_string(obj, "name").unwrap_or_default();
    let status = extract_json_string(obj, "status").unwrap_or_default();
    let memory_mb = extract_json_number::<u64>(obj, "memory").unwrap_or(0);
    let vcpus = extract_json_number::<u32>(obj, "vcpus").unwrap_or(0);
    let disk_gb = extract_json_number::<u64>(obj, "disk").unwrap_or(0);

    // Region slug is inside `"region":{"slug":"..."}`.
    let region = slice_inner_object(obj, "\"region\"")
        .and_then(|r| extract_json_string(r, "slug"))
        .unwrap_or_default();

    // Size is `"size":{"slug":"...","price_hourly":X}`.
    let size_obj = slice_inner_object(obj, "\"size\"");
    let size_slug = size_obj
        .and_then(|s| extract_json_string(s, "slug"))
        .unwrap_or_default();
    let price_hourly = size_obj
        .and_then(|s| extract_json_f64(s, "price_hourly"))
        .unwrap_or(0.0);

    // Public IP is inside `"networks":{"v4":[{"ip_address":"...","type":"public"},...]}`.
    let ip_public = extract_public_ipv4(obj).unwrap_or_default();

    Some(Droplet {
        id,
        name,
        region,
        size_slug,
        price_hourly,
        status,
        ip_public,
        memory_mb,
        vcpus,
        disk_gb,
        gpu_label: String::new(),
        active_since: 0,
    })
}

fn extract_public_ipv4(obj: &str) -> Option<String> {
    let nets = slice_inner_object(obj, "\"networks\"")?;
    let v4 = slice_inner_array(nets, "\"v4\"")?;
    // Walk objects inside v4, pick first with type=public.
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let bytes = v4.as_bytes();
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(s) = start {
                        let o = &v4[s..=i];
                        let ty = extract_json_string(o, "type").unwrap_or_default();
                        if ty == "public" {
                            return extract_json_string(o, "ip_address");
                        }
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_ssh_key_ids(body: &str) -> Vec<u64> {
    let arr = match slice_inner_array(body, "\"ssh_keys\"") {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let bytes = arr.as_bytes();
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escape { escape = false; }
            else if b == b'\\' { escape = true; }
            else if b == b'"' { in_str = false; }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if depth == 0 { start = Some(i); }
                depth += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(s) = start {
                        let o = &arr[s..=i];
                        if let Some(id) = extract_json_number::<u64>(o, "id") {
                            out.push(id);
                        }
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

// =============================================================================
// State persistence — tiny TOML at ~/.config/nexus-droplet-ticker/state.toml
// =============================================================================

fn state_file_path() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = std::path::PathBuf::from(home).join(STATE_DIR_REL);
    let _ = std::fs::create_dir_all(&dir);
    Some(dir.join(STATE_FILE_NAME))
}

fn load_persisted_droplet_id() -> Option<u64> {
    let p = state_file_path()?;
    let txt = std::fs::read_to_string(&p).ok()?;
    // Look for `droplet_id = <n>`.
    for line in txt.lines() {
        if let Some(rest) = line.trim().strip_prefix("droplet_id") {
            let rest = rest.trim_start_matches(|c: char| c == '=' || c.is_whitespace());
            if let Ok(id) = rest.trim().parse::<u64>() {
                return Some(id);
            }
        }
    }
    None
}

fn save_persisted_droplet_id(id: u64) {
    if let Some(p) = state_file_path() {
        let _ = std::fs::write(
            &p,
            format!(
                "# nexus-droplet-ticker state\ndroplet_id = {id}\nupdated = \"{}\"\n",
                now_hms()
            ),
        );
    }
}

fn clear_persisted_droplet_id() {
    if let Some(p) = state_file_path() {
        let _ = std::fs::remove_file(&p);
    }
}

// =============================================================================
// Time + formatting helpers
// =============================================================================

fn now_hms() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

fn now_stamp_compact() -> String {
    chrono::Local::now().format("%Y%m%dT%H%M%S").to_string()
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn format_hms(secs: u64) -> String {
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

fn format_mb(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1}G", mb as f64 / 1024.0)
    } else {
        format!("{mb}M")
    }
}

/// Map a DigitalOcean size slug to a short human label.
/// Known mappings fall back to the slug itself.
fn humanise_size(slug: &str) -> String {
    match slug {
        "gpu-h100x1-80gb" => "H100 · 80GB".to_string(),
        "gpu-h100x8-640gb" => "H100×8 · 640GB".to_string(),
        "gpu-h200x1-141gb" => "H200 · 141GB".to_string(),
        "gpu-h200x8-1128gb" => "H200×8 · 1128GB".to_string(),
        "gpu-mi300x1-192gb" => "MI300 · 192GB".to_string(),
        other => other.to_string(),
    }
}

/// Shorten a droplet/size identifier for use in a snapshot name, so we
/// don't inherit a 50-character DO slug.
fn short_size(name: &str) -> String {
    // grab the last 12 alphanumeric chars so snapshot names stay readable
    name.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>()
        .chars().rev().take(12).collect::<String>()
        .chars().rev().collect()
}

// =============================================================================
// UI — eframe::App
// =============================================================================

impl eframe::App for DropletTickerApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(250));

        // Persisted theme on first frame.
        static LOADED: std::sync::Once = std::sync::Once::new();
        LOADED.call_once(|| set_active_theme(Theme::load()));

        let is_light = ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) == 1;
        let mut visuals = if is_light { egui::Visuals::light() } else { egui::Visuals::dark() };
        visuals.panel_fill = egui::Color32::TRANSPARENT;
        visuals.window_fill = egui::Color32::TRANSPARENT;
        visuals.override_text_color = Some(text_color());
        ctx.set_visuals(visuals);

        // ---- Background polls (UI thread; each probe is short + guarded) ----
        let now = Instant::now();
        if now.duration_since(self.last_droplet_poll).as_secs_f32() >= DROPLET_POLL_SECS {
            self.last_droplet_poll = now;
            self.refresh_droplet();
        }
        if now.duration_since(self.last_metrics_poll).as_secs_f32() >= METRICS_POLL_SECS {
            self.last_metrics_poll = now;
            self.poll_live_metrics();
        }
        if now.duration_since(self.last_billing_poll).as_secs_f32() >= BILLING_POLL_SECS {
            self.last_billing_poll = now;
            self.poll_billing();
        }

        // Update session seconds every frame so the billing display ticks.
        if let Ok(mut s) = self.state.lock() {
            s.session_seconds = if s.droplet.active_since != 0 {
                unix_now().saturating_sub(s.droplet.active_since)
            } else {
                0
            };
        }

        let snap = self.state.lock().unwrap().snapshot();

        let outer = egui::Frame::none()
            .fill(bg_color())
            .rounding(egui::Rounding::same(10.0))
            .stroke(egui::Stroke::new(
                1.0,
                if is_light {
                    egui::Color32::from_rgba_unmultiplied(text_dim().r(), text_dim().g(), text_dim().b(), 90)
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
            .inner_margin(egui::Margin { left: 10.0, right: 10.0, top: 8.0, bottom: 10.0 });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(egui::Margin::same(6.0)))
            .show(ctx, |ui| {
                outer.show(ui, |ui| {
                    ui.set_clip_rect(ui.max_rect());
                    self.render_content(ui, ctx, &snap, is_light);
                });
            });

        // Destroy-confirm modal (rendered on top of CentralPanel).
        if snap.destroy_confirm_open {
            self.render_destroy_modal(ctx, &snap);
        }
    }
}

impl DropletTickerApp {
    fn render_content(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        snap: &Snapshot,
        _is_light: bool,
    ) {
        // ── Titlebar ──
        let pulse = snap.state.is_active() || snap.state.is_transitional();
        let title_resp = ui.horizontal(|ui| {
            draw_led(ui, status_color(&snap.state), 5.0, pulse);
            ui.label(
                egui::RichText::new("NEXUS-DROPLET")
                    .strong()
                    .size(11.0)
                    .color(text_color()),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(
                    egui::Button::new(egui::RichText::new("✕").size(12.0).color(text_dim()))
                        .frame(false)
                        .min_size(egui::vec2(20.0, 20.0)),
                ).on_hover_text("close").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    std::process::exit(0);
                }
                if ui.add(
                    egui::Button::new(egui::RichText::new("—").size(12.0).color(text_dim()))
                        .frame(false)
                        .min_size(egui::vec2(20.0, 20.0)),
                ).on_hover_text("minimize").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                }
                let is_light_now = ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) == 1;
                let glyph = if is_light_now { "☾" } else { "☀" };
                if ui.add(
                    egui::Button::new(
                        egui::RichText::new(glyph).size(11.0).color(amber()).strong(),
                    )
                    .frame(false)
                    .min_size(egui::vec2(16.0, 16.0)),
                ).on_hover_text("Toggle light/dark theme").clicked() {
                    let new_theme = if is_light_now { Theme::Dark } else { Theme::Light };
                    set_active_theme(new_theme);
                    new_theme.save();
                }
            });
        }).response;
        let drag_sense = ui.interact(
            title_resp.rect,
            egui::Id::new("droplet-ticker-drag"),
            egui::Sense::click_and_drag(),
        );
        if drag_sense.drag_started_by(egui::PointerButton::Primary) {
            ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
        }
        ui.separator();

        // ── Droplet identity row ──
        ui.horizontal_wrapped(|ui| {
            draw_led(ui, status_color(&snap.state), 5.0, pulse);
            ui.label(
                egui::RichText::new(snap.state.label())
                    .strong()
                    .size(11.0)
                    .color(text_color()),
            );
            let name = if snap.droplet.name.is_empty() {
                "—".to_string()
            } else {
                snap.droplet.name.clone()
            };
            ui.label(egui::RichText::new(format!(" {name}")).size(10.0).color(text_dim()));
            if !snap.droplet.region.is_empty() {
                ui.label(
                    egui::RichText::new(format!("· {}", snap.droplet.region))
                        .size(9.0)
                        .color(text_dim()),
                );
            }
            if !snap.droplet.gpu_label.is_empty() {
                ui.label(
                    egui::RichText::new(format!("· {}", snap.droplet.gpu_label))
                        .size(9.0)
                        .color(text_dim()),
                );
            }
        });

        // IP address + click-to-copy.
        if !snap.droplet.ip_public.is_empty() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("ip").size(9.0).color(slate()));
                let ip_btn = ui.add(
                    egui::Button::new(
                        egui::RichText::new(&snap.droplet.ip_public).size(10.0).color(teal()),
                    )
                    .frame(false),
                ).on_hover_text("click to copy");
                if ip_btn.clicked() {
                    ctx.copy_text(snap.droplet.ip_public.clone());
                    if let Ok(mut s) = self.state.lock() {
                        s.push_log(format!("copied {} to clipboard", snap.droplet.ip_public), false);
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("id={}", snap.droplet.id))
                            .size(8.0)
                            .color(slate()),
                    );
                });
            });
        }

        // ── Droplet picker (only shown when there's more than one GPU droplet and no pin) ──
        if snap.droplet_options.len() > 1 && snap.droplet.id == 0 {
            ui.separator();
            ui.label(egui::RichText::new("select a droplet:").size(10.0).color(text_dim()));
            egui::ScrollArea::vertical()
                .max_height(80.0)
                .id_salt("droplet-picker")
                .show(ui, |ui| {
                    for opt in &snap.droplet_options {
                        let label = format!("{} — {} — {}", opt.name, opt.size_slug, opt.status);
                        if ui.button(egui::RichText::new(label).size(10.0).color(teal())).clicked() {
                            let id = opt.id;
                            save_persisted_droplet_id(id);
                            if let Ok(mut s) = self.state.lock() {
                                s.pinned_droplet_id = Some(id);
                                s.push_log(format!("pinned droplet {id}"), false);
                            }
                        }
                    }
                });
        }

        ui.add_space(2.0);
        ui.separator();

        // ── Metrics strip ──
        self.render_metrics(ui, snap);

        // ── GPU% history plot (collapsed on short windows) ──
        let avail = ui.available_rect_before_wrap();
        if avail.height() > PLOT_COLLAPSE_BELOW - 260.0 && ui.ctx().screen_rect().height() > PLOT_COLLAPSE_BELOW {
            let plot_h = (avail.height() * 0.28).clamp(60.0, 120.0);
            draw_gpu_plot(ui, plot_h, &snap.gpu_history);
        }

        ui.separator();

        // ── Billing panel ──
        self.render_billing(ui, snap);

        ui.add_space(2.0);
        ui.separator();

        // ── Action row ──
        self.render_actions(ui, snap);

        ui.add_space(2.0);
        ui.separator();

        // ── Log panel ──
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("log").size(9.0).color(text_dim()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(
                    egui::Button::new(egui::RichText::new("clear").size(9.0).color(terracotta()))
                        .frame(false),
                ).on_hover_text("clear log").clicked() {
                    if let Ok(mut s) = self.state.lock() {
                        s.log.clear();
                        s.push_log("log cleared", false);
                    }
                }
                ui.add_space(6.0);
                if ui.add(
                    egui::Button::new(egui::RichText::new("copy all").size(9.0).color(teal()))
                        .frame(false),
                ).on_hover_text("copy log to clipboard").clicked() {
                    let full: String = snap.log.iter()
                        .map(|(ts, msg, _)| format!("{ts}  {msg}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    ctx.copy_text(full);
                }
            });
        });

        let avail_after = ui.available_height().max(40.0);
        egui::ScrollArea::vertical()
            .id_salt("droplet-log")
            .max_height(avail_after)
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded)
            .show(ui, |ui| {
                for (ts, msg, is_err) in &snap.log {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(ts).size(8.0).color(slate()));
                        let c = if *is_err { terracotta() } else { text_color() };
                        ui.label(egui::RichText::new(msg).size(9.0).color(c));
                    });
                }
            });
    }

    fn render_metrics(&self, ui: &mut egui::Ui, snap: &Snapshot) {
        let m = &snap.metrics;
        let vram_pct = if m.vram_total_mb > 0 {
            m.vram_used_mb as f32 / m.vram_total_mb as f32
        } else {
            0.0
        };
        let ram_pct = if m.ram_total_mb > 0 {
            m.ram_used_mb as f32 / m.ram_total_mb as f32
        } else {
            0.0
        };
        let stale = m.sampled_at == 0
            || unix_now().saturating_sub(m.sampled_at) > (METRICS_POLL_SECS as u64 * 4);

        // GPU utilisation bar.
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("GPU").size(9.0).color(text_dim()));
            let bar = egui::ProgressBar::new((m.gpu_util_pct / 100.0).clamp(0.0, 1.0))
                .desired_width(ui.available_width() - 60.0)
                .text(
                    egui::RichText::new(if stale { "—".to_string() } else { format!("{:.0}%", m.gpu_util_pct) })
                        .size(9.0)
                        .color(text_color()),
                );
            ui.add(bar);
        });

        // VRAM bar.
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("VRAM").size(9.0).color(text_dim()));
            let bar = egui::ProgressBar::new(vram_pct.clamp(0.0, 1.0))
                .desired_width(ui.available_width() - 60.0)
                .text(
                    egui::RichText::new(if m.vram_total_mb > 0 {
                        format!("{} / {}", format_mb(m.vram_used_mb), format_mb(m.vram_total_mb))
                    } else {
                        "—".to_string()
                    })
                    .size(9.0)
                    .color(text_color()),
                );
            ui.add(bar);
        });

        // CPU + RAM + temp + power — one line of little chips.
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(format!("CPU {:.0}%", m.cpu_util_pct)).size(9.0).color(text_dim()));
            ui.label(egui::RichText::new(format!(
                "RAM {:.0}% ({}/{})",
                ram_pct * 100.0,
                format_mb(m.ram_used_mb),
                format_mb(m.ram_total_mb),
            )).size(9.0).color(text_dim()));
            if m.gpu_temp_c > 0.0 {
                ui.label(egui::RichText::new(format!("{:.0}°C", m.gpu_temp_c)).size(9.0).color(text_dim()));
            }
            if m.gpu_power_w > 0.0 {
                ui.label(egui::RichText::new(format!("{:.0} W", m.gpu_power_w)).size(9.0).color(text_dim()));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let age = if m.sampled_at == 0 {
                    "no data".to_string()
                } else {
                    format!("{}s ago", unix_now().saturating_sub(m.sampled_at))
                };
                ui.label(egui::RichText::new(age).size(8.0).color(slate()));
            });
        });
    }

    fn render_billing(&self, ui: &mut egui::Ui, snap: &Snapshot) {
        let rate = snap.droplet.price_hourly;
        let runtime = snap.session_seconds;
        let session_cost = rate * (runtime as f64 / 3600.0);

        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("session").size(9.0).color(slate()));
            ui.label(
                egui::RichText::new(format!("${session_cost:.4}"))
                    .size(13.0)
                    .strong()
                    .color(if snap.state.is_active() { teal() } else { text_dim() }),
            );
            ui.label(egui::RichText::new(format!("· {}", format_hms(runtime))).size(9.0).color(text_dim()));
            ui.label(egui::RichText::new(format!("· ${rate:.3}/hr")).size(9.0).color(text_dim()));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("MTD").size(9.0).color(slate()));
            ui.label(
                egui::RichText::new(format!("${:.2}", snap.billing.month_to_date_usage))
                    .size(10.0)
                    .color(text_color()),
            );
            ui.label(
                egui::RichText::new(format!("· balance ${:.2}", snap.billing.account_balance))
                    .size(9.0)
                    .color(text_dim()),
            );
        });
    }

    fn render_actions(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        let is_active = snap.state.is_active();
        let is_off = snap.state.is_off() || matches!(snap.state, DropletState::NoDroplet);
        let pending = snap.pending_action.is_some();
        let has_token = snap.token_present;

        ui.horizontal_wrapped(|ui| {
            let start_enabled = has_token && is_off && !pending;
            if ui.add_enabled(
                start_enabled,
                egui::Button::new(
                    egui::RichText::new("Start")
                        .size(10.0)
                        .color(if start_enabled { teal() } else { text_dim() }),
                ),
            ).on_hover_text("Provision a new droplet from $NEXUS_DROPLET_SNAPSHOT_ID, or power on the existing one").clicked() {
                if matches!(snap.state, DropletState::NoDroplet) {
                    self.start_droplet_from_snapshot();
                } else {
                    self.send_action("power_on");
                }
            }

            let stop_enabled = has_token && is_active && !pending;
            if ui.add_enabled(
                stop_enabled,
                egui::Button::new(
                    egui::RichText::new("Stop")
                        .size(10.0)
                        .color(if stop_enabled { amber() } else { text_dim() }),
                ),
            ).on_hover_text("Hard power-off (no graceful shutdown)").clicked() {
                self.send_action("power_off");
            }

            let shutdown_enabled = has_token && is_active && !pending;
            if ui.add_enabled(
                shutdown_enabled,
                egui::Button::new(
                    egui::RichText::new("Shutdown")
                        .size(10.0)
                        .color(if shutdown_enabled { amber() } else { text_dim() }),
                ),
            ).on_hover_text("Graceful OS shutdown, then droplet stops").clicked() {
                self.send_action("shutdown");
            }

            let destroy_enabled = has_token && snap.droplet.id != 0 && !pending;
            if ui.add_enabled(
                destroy_enabled,
                egui::Button::new(
                    egui::RichText::new("Destroy")
                        .size(10.0)
                        .color(if destroy_enabled { terracotta() } else { text_dim() }),
                ),
            ).on_hover_text("Permanently delete the droplet (optional snapshot first)").clicked() {
                if let Ok(mut s) = self.state.lock() {
                    s.destroy_confirm_open = true;
                }
            }
        });

        if !has_token {
            ui.label(
                egui::RichText::new("set DO_API_TOKEN in the environment to enable actions")
                    .size(8.0)
                    .color(slate()),
            );
        }
    }

    fn render_destroy_modal(&self, ctx: &egui::Context, snap: &Snapshot) {
        let mut still_open = true;
        let mut do_destroy = false;
        let mut snapshot_first = snap.destroy_snapshot_first;

        egui::Window::new(
            egui::RichText::new("Confirm destroy")
                .size(11.0)
                .color(terracotta())
                .strong(),
        )
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .default_size([320.0, 160.0])
        .open(&mut still_open)
        .frame(
            egui::Frame::none()
                .fill(bg_color())
                .rounding(egui::Rounding::same(8.0))
                .stroke(egui::Stroke::new(1.0, terracotta()))
                .inner_margin(egui::Margin::same(12.0)),
        )
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Destroy droplet {} (id {})?",
                    snap.droplet.name, snap.droplet.id
                ))
                .size(10.0)
                .color(text_color()),
            );
            ui.label(
                egui::RichText::new("This is permanent. Storage is released once the action completes.")
                    .size(9.0)
                    .color(text_dim()),
            );
            ui.add_space(6.0);
            ui.checkbox(&mut snapshot_first, "Snapshot first (shutdown → snapshot → destroy)");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button(
                    egui::RichText::new("Cancel").size(10.0).color(text_dim()),
                ).clicked() {
                    if let Ok(mut s) = self.state.lock() {
                        s.destroy_confirm_open = false;
                    }
                }
                if ui.button(
                    egui::RichText::new("Destroy").size(10.0).color(terracotta()).strong(),
                ).clicked() {
                    do_destroy = true;
                }
            });
        });

        // Sync the checkbox back to shared state before potentially acting.
        if let Ok(mut s) = self.state.lock() {
            s.destroy_snapshot_first = snapshot_first;
            if !still_open {
                s.destroy_confirm_open = false;
            }
        }
        if do_destroy {
            if let Ok(mut s) = self.state.lock() {
                s.destroy_confirm_open = false;
            }
            self.destroy_droplet(snapshot_first);
        }
    }
}

// =============================================================================
// GPU% history plot (dotted polyline, same visual language as training_ticker)
// =============================================================================

fn draw_gpu_plot(ui: &mut egui::Ui, height: f32, history: &[GpuSample]) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, egui::Rounding::same(4.0), bg_row());

    if history.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no GPU samples yet",
            egui::FontId::proportional(9.0),
            slate(),
        );
        return;
    }

    // Show the last 5 minutes window.
    let now = unix_now();
    let window_start = now.saturating_sub(300);
    let points: Vec<&GpuSample> = history.iter().filter(|p| p.ts >= window_start).collect();
    if points.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "collecting GPU history…",
            egui::FontId::proportional(9.0),
            slate(),
        );
        return;
    }

    let x_min = points.first().map(|p| p.ts).unwrap_or(now);
    let x_max = points.last().map(|p| p.ts).unwrap_or(now);
    let x_span = (x_max - x_min).max(1) as f32;

    let pad = 4.0;
    let inner = egui::Rect::from_min_max(
        egui::pos2(rect.min.x + pad, rect.min.y + pad),
        egui::pos2(rect.max.x - pad, rect.max.y - 12.0),
    );

    // Grid: 3 horizontal rules at 0 / 50 / 100.
    let grid = egui::Color32::from_rgba_unmultiplied(slate().r(), slate().g(), slate().b(), 40);
    for t in [0.0, 0.5, 1.0] {
        let y = inner.max.y - t * inner.height();
        painter.line_segment(
            [egui::pos2(inner.min.x, y), egui::pos2(inner.max.x, y)],
            egui::Stroke::new(1.0, grid),
        );
    }

    let map = |p: &GpuSample| -> egui::Pos2 {
        let tx = (p.ts - x_min) as f32 / x_span;
        let ty = (p.gpu_pct / 100.0).clamp(0.0, 1.0);
        egui::pos2(
            inner.min.x + tx * inner.width(),
            inner.max.y - ty * inner.height(),
        )
    };

    let pts: Vec<egui::Pos2> = points.iter().map(|p| map(p)).collect();
    draw_dotted_polyline(&painter, &pts, teal());

    // Axis label: "GPU% · last Nm".
    let label = format!("GPU% · last {}m", ((x_max - x_min) / 60).max(1));
    let galley = painter.layout_no_wrap(label, egui::FontId::proportional(8.0), slate());
    painter.galley(
        egui::pos2(rect.min.x + 6.0, rect.max.y - 11.0),
        galley,
        slate(),
    );
}

fn draw_dotted_polyline(painter: &egui::Painter, points: &[egui::Pos2], color: egui::Color32) {
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
        let mut t = -acc;
        while t < seg_len {
            if t >= 0.0 {
                let p = a + dir * t;
                painter.circle_filled(p, DOT_RADIUS, color);
            }
            t += DOT_SPACING;
        }
        acc = (seg_len + acc) % DOT_SPACING;
    }
}

// =============================================================================
// Theme — mirrors training_ticker.rs
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
    fn save(&self) {
        let _ = std::fs::write(THEME_FILE, match self {
            Self::Dark => "dark",
            Self::Light => "light",
        });
    }
}

static ACTIVE_THEME: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
fn set_active_theme(t: Theme) {
    ACTIVE_THEME.store(
        match t { Theme::Dark => 0, Theme::Light => 1 },
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

fn status_color(d: &DropletState) -> egui::Color32 {
    match d {
        DropletState::Active => teal(),
        DropletState::New | DropletState::Transitioning(_) => amber(),
        DropletState::Off | DropletState::NoDroplet | DropletState::Archive => terracotta(),
        DropletState::NoToken | DropletState::Unknown => slate(),
        DropletState::Error(_) => terracotta(),
    }
}

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
    token_present: bool,
    state: DropletState,
    droplet: Droplet,
    droplet_options: Vec<Droplet>,
    metrics: LiveMetrics,
    gpu_history: Vec<GpuSample>,
    billing: Billing,
    session_seconds: u64,
    destroy_confirm_open: bool,
    destroy_snapshot_first: bool,
    pending_action: Option<String>,
    log: Vec<(String, String, bool)>,
}

impl SharedState {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            token_present: self.token_present,
            state: self.state.clone(),
            droplet: self.droplet.clone(),
            droplet_options: self.droplet_options.clone(),
            metrics: self.metrics.clone(),
            gpu_history: self.gpu_history.clone(),
            billing: self.billing.clone(),
            session_seconds: self.session_seconds,
            destroy_confirm_open: self.destroy_confirm_open,
            destroy_snapshot_first: self.destroy_snapshot_first,
            pending_action: self.pending_action.clone(),
            log: self.log.iter()
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
    fn parse_droplet_list_basic() {
        let body = r#"{
            "droplets": [
                {
                    "id": 111,
                    "name": "nexus-h100",
                    "status": "active",
                    "memory": 491520,
                    "vcpus": 20,
                    "disk": 200,
                    "region": {"slug": "nyc2", "name": "New York 2"},
                    "size": {"slug": "gpu-h100x1-80gb", "price_hourly": 4.47},
                    "networks": {"v4": [{"ip_address": "1.2.3.4", "type": "public"}]}
                },
                {
                    "id": 222,
                    "name": "regular",
                    "status": "active",
                    "memory": 2048,
                    "vcpus": 1,
                    "disk": 50,
                    "region": {"slug": "nyc2"},
                    "size": {"slug": "s-1vcpu-2gb", "price_hourly": 0.01},
                    "networks": {"v4": []}
                }
            ]
        }"#;
        let droplets = parse_droplet_list(body);
        assert_eq!(droplets.len(), 2);
        assert_eq!(droplets[0].id, 111);
        assert_eq!(droplets[0].region, "nyc2");
        assert_eq!(droplets[0].size_slug, "gpu-h100x1-80gb");
        assert!((droplets[0].price_hourly - 4.47).abs() < 1e-6);
        assert_eq!(droplets[0].ip_public, "1.2.3.4");
        assert_eq!(droplets[0].vcpus, 20);
        let gpu: Vec<_> = droplets.iter().filter(|d| d.size_slug.starts_with("gpu-")).collect();
        assert_eq!(gpu.len(), 1);
    }

    #[test]
    fn parse_single_droplet_basic() {
        let body = r#"{"droplet":{"id":999,"name":"one","status":"off","memory":8192,"vcpus":4,"disk":100,"region":{"slug":"sfo3"},"size":{"slug":"gpu-h200x1-141gb","price_hourly":6.75},"networks":{"v4":[{"ip_address":"9.9.9.9","type":"public"}]}}}"#;
        let d = parse_single_droplet(body).unwrap();
        assert_eq!(d.id, 999);
        assert_eq!(d.region, "sfo3");
        assert_eq!(d.size_slug, "gpu-h200x1-141gb");
        assert!((d.price_hourly - 6.75).abs() < 1e-6);
        assert_eq!(d.ip_public, "9.9.9.9");
    }

    #[test]
    fn balance_json_extracts() {
        let body = r#"{"month_to_date_balance":"15.00","account_balance":"-12.50","month_to_date_usage":"27.50","generated_at":"2026-04-18T12:00:00Z"}"#;
        assert!((extract_json_f64(body, "month_to_date_usage").unwrap() - 27.5).abs() < 1e-6);
        assert!((extract_json_f64(body, "account_balance").unwrap() + 12.5).abs() < 1e-6);
        assert_eq!(extract_json_string(body, "generated_at").unwrap(), "2026-04-18T12:00:00Z");
    }

    #[test]
    fn ssh_key_id_extraction() {
        let body = r#"{"ssh_keys":[{"id":100,"fingerprint":"aa","name":"a"},{"id":200,"fingerprint":"bb","name":"b"}]}"#;
        let ids = parse_ssh_key_ids(body);
        assert_eq!(ids, vec![100, 200]);
    }

    #[test]
    fn humanise_size_fallback() {
        assert_eq!(humanise_size("gpu-h200x1-141gb"), "H200 · 141GB");
        assert_eq!(humanise_size("gpu-unknown-slug"), "gpu-unknown-slug");
    }

    #[test]
    fn dotted_polyline_short_inputs() {
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
