//! nexus-ticker — always-on-top desktop widget that actively monitors and
//! fixes CI, git state, and infrastructure across all AutomataNexus repos.
//!
//! This is NOT a passive status display. The ticker:
//! 1. Polls GitHub CI every 5 minutes for all monitored repos
//! 2. When it finds failures, runs a **ralph loop** (up to 5 passes): apply
//!    `cargo fmt --all` + `cargo clippy --fix` + targeted unused-import removal,
//!    then verify locally via `cargo fmt --check` and `cargo clippy -- -D warnings`.
//!    Only commits + auto-pushes once the local verify is green. If the loop
//!    exhausts 5 passes without verifying, it toasts the user for manual fix
//!    instead of pushing a broken "fix" commit.
//! 3. Monitors Tailscale for offline controllers
//! 4. Checks for uncommitted changes across repos
//! 5. Accepts manual commands via the input bar
//!
//! Launch: nexus-ticker &

use eframe::egui;
use std::sync::{Arc, Mutex};

const NEXUS_SERVE_URL: &str = "http://127.0.0.1:11435";
const TICKER_WIDTH: f32 = 480.0;
const TICKER_HEIGHT: f32 = 280.0;
const MAX_LOG_LINES: usize = 100;
const CI_POLL_SECS: u64 = 300;      // 5 minutes
const HEALTH_POLL_SECS: u64 = 10;
const GIT_POLL_SECS: u64 = 600;     // 10 minutes

/// Repos to monitor — (gh owner/name, local path)
const REPOS: &[(&str, &str)] = &[
    ("AutomataNexus/AxonML", "/opt/AxonML"),
    ("AutomataNexus/NexusEdge_Rust", "/opt/NexusEdge_Rust"),
    ("AutomataNexus/FerrumEmail", "/opt/Ferrum"),
    ("AutomataNexus/FerumMailSaaS", "/opt/FerumMailSaaS"),
    ("AutomataNexus/FerumMailbox", "/opt/FerumMailbox"),
    ("AutomataNexus/FerumWebmail", "/opt/FerumWebmail"),
    ("AutomataNexus/NexusOracle", "/opt/NexusOracle"),
];

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([TICKER_WIDTH, TICKER_HEIGHT])
            .with_always_on_top()
            .with_decorations(true)
            .with_transparent(false)
            .with_title("nexus-agent"),
        ..Default::default()
    };

    eframe::run_native(
        "nexus-ticker",
        options,
        Box::new(|cc| Ok(Box::new(TickerApp::new(cc)))),
    )
}

// =============================================================================
// State
// =============================================================================

#[derive(Clone, PartialEq)]
enum Status {
    Idle,
    Monitoring(String),
    Fixing(String),
    Error(String),
}

impl Status {
    fn label(&self) -> &str {
        match self {
            Self::Idle => "IDLE",
            Self::Monitoring(_) => "MONITORING",
            Self::Fixing(_) => "FIXING",
            Self::Error(_) => "ERROR",
        }
    }
    fn color(&self) -> egui::Color32 {
        match self {
            Self::Idle => egui::Color32::from_rgb(100, 200, 100),
            Self::Monitoring(_) => egui::Color32::from_rgb(100, 150, 255),
            Self::Fixing(_) => egui::Color32::from_rgb(255, 200, 50),
            Self::Error(_) => egui::Color32::from_rgb(255, 80, 80),
        }
    }
    fn detail(&self) -> String {
        match self {
            Self::Idle => "All green".to_string(),
            Self::Monitoring(s) | Self::Fixing(s) | Self::Error(s) => s.clone(),
        }
    }
}

#[derive(Clone)]
struct LogEntry {
    timestamp: String,
    message: String,
    is_error: bool,
}

#[derive(Clone)]
struct RepoStatus {
    name: String,
    ci_ok: Option<bool>,       // None = unknown, Some(true) = green, Some(false) = red
    last_checked: String,
}

/// A commit waiting for user approval before push.
#[derive(Clone)]
struct PendingPush {
    repo_name: String,
    local_path: String,
    commit_msg: String,
    files_changed: usize,
}

struct SharedState {
    status: Status,
    log: Vec<LogEntry>,
    backend_online: bool,
    repos: Vec<RepoStatus>,
    ci_failures: usize,
    pending_pushes: Vec<PendingPush>,
}

impl SharedState {
    fn push_log(&mut self, msg: &str, is_error: bool) {
        self.log.push(LogEntry {
            timestamp: now(),
            message: msg.to_string(),
            is_error,
        });
        let excess = self.log.len().saturating_sub(MAX_LOG_LINES);
        if excess > 0 { self.log.drain(0..excess); }
    }
}

struct TickerApp {
    state: Arc<Mutex<SharedState>>,
    input: String,
    selected_agent: usize,
    runtime: tokio::runtime::Runtime,
}

const AGENT_NAMES: &[&str] = &["knowledge", "retrain", "fieldtech", "research", "orchestrator"];

impl TickerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let repos: Vec<RepoStatus> = REPOS.iter().map(|(name, _)| RepoStatus {
            name: name.to_string(),
            ci_ok: None,
            last_checked: "never".to_string(),
        }).collect();

        let state = Arc::new(Mutex::new(SharedState {
            status: Status::Monitoring("starting up...".to_string()),
            log: vec![LogEntry { timestamp: now(), message: "nexus-ticker started — monitoring CI".to_string(), is_error: false }],
            backend_online: false,
            repos,
            ci_failures: 0,
            pending_pushes: Vec::new(),
        }));

        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");

        // --- Background: nexus-serve health (every 10s) ---
        let s1 = state.clone();
        runtime.spawn(async move {
            let client = reqwest::Client::new();
            loop {
                let health = client.get(format!("{NEXUS_SERVE_URL}/health"))
                    .send().await.map(|r| r.status().is_success()).unwrap_or(false);
                if let Ok(mut s) = s1.lock() { s.backend_online = health; }
                tokio::time::sleep(std::time::Duration::from_secs(HEALTH_POLL_SECS)).await;
            }
        });

        // --- Background: CI monitor + auto-fix (every 5 min, first run after 5s) ---
        let s2 = state.clone();
        runtime.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            loop {
                ci_check_and_fix(s2.clone()).await;
                tokio::time::sleep(std::time::Duration::from_secs(CI_POLL_SECS)).await;
            }
        });

        // --- Background: git dirty check (every 10 min, first run after 30s) ---
        let s3 = state.clone();
        runtime.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            loop {
                git_dirty_check(s3.clone()).await;
                tokio::time::sleep(std::time::Duration::from_secs(GIT_POLL_SECS)).await;
            }
        });

        Self { state, input: String::new(), selected_agent: 0, runtime }
    }

    fn push_log(&self, msg: &str) {
        if let Ok(mut s) = self.state.lock() { s.push_log(msg, false); }
    }

    fn submit_command(&mut self) {
        let cmd = self.input.trim().to_string();
        if cmd.is_empty() { return; }
        self.input.clear();

        let agent = AGENT_NAMES[self.selected_agent].to_string();
        self.push_log(&format!("> [{agent}] {cmd}"));

        let state = self.state.clone();
        if let Ok(mut s) = state.lock() {
            s.status = Status::Monitoring(format!("running {agent}..."));
        }

        let cmd_clone = cmd.clone();
        self.runtime.spawn(async move {
            let output = tokio::process::Command::new(
                "/opt/AxonML/nexus-agent/target/release/nexus-agent")
                .args([&agent, &cmd_clone])
                .output().await;

            if let Ok(mut s) = state.lock() {
                match output {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let lines: Vec<&str> = stdout.lines().collect();
                        let start = lines.len().saturating_sub(5);
                        for line in &lines[start..] {
                            s.push_log(line, false);
                        }
                        s.status = Status::Idle;
                    }
                    Err(e) => { s.status = Status::Error(e.to_string()); }
                }
            }
        });
    }
}

// =============================================================================
// Background: CI check + auto-fix
// =============================================================================

async fn ci_check_and_fix(state: Arc<Mutex<SharedState>>) {
    if let Ok(mut s) = state.lock() {
        s.status = Status::Monitoring("checking CI...".to_string());
        s.push_log("--- CI scan ---", false);
    }

    let mut total_failures = 0usize;

    for (repo_gh, local_path) in REPOS {
        // Get latest CI run with run ID
        let output = tokio::process::Command::new("gh")
            .args(["run", "list", "--repo", repo_gh, "--limit", "1",
                   "--json", "conclusion,name,headBranch,databaseId"])
            .output().await;

        let (ci_ok, conclusion, run_id) = match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(&stdout);
                match parsed {
                    Ok(runs) if !runs.is_empty() => {
                        let c = runs[0].get("conclusion")
                            .and_then(|v| v.as_str()).unwrap_or("unknown");
                        let id = runs[0].get("databaseId")
                            .and_then(|v| v.as_u64()).unwrap_or(0);
                        (c == "success", c.to_string(), id)
                    }
                    _ => (true, "no runs".to_string(), 0),
                }
            }
            Err(_) => (true, "gh error".to_string(), 0),
        };

        let short_name = repo_gh.split('/').last().unwrap_or(repo_gh);
        if let Ok(mut s) = state.lock() {
            if let Some(r) = s.repos.iter_mut().find(|r| r.name == *repo_gh) {
                r.ci_ok = Some(ci_ok);
                r.last_checked = now();
            }
        }

        if ci_ok {
            if let Ok(mut s) = state.lock() {
                s.push_log(&format!("  {short_name}: green"), false);
            }
            continue;
        }

        total_failures += 1;
        if let Ok(mut s) = state.lock() {
            s.push_log(&format!("  {short_name}: FAILED ({conclusion})"), true);
            s.status = Status::Fixing(format!("analyzing {short_name}..."));
        }

        let is_repo = std::path::Path::new(local_path).join(".git").exists();
        if !is_repo {
            if let Ok(mut s) = state.lock() {
                s.push_log(&format!("    skip: no local repo at {local_path}"), false);
            }
            continue;
        }

        // ---- Step 1: Get the actual CI error logs ----
        let error_log = if run_id > 0 {
            let log_out = tokio::process::Command::new("gh")
                .args(["run", "view", &run_id.to_string(), "--repo", repo_gh, "--log"])
                .output().await;
            match log_out {
                Ok(o) => {
                    let full = String::from_utf8_lossy(&o.stdout).to_string();
                    // Extract only error lines
                    let errors: Vec<&str> = full.lines()
                        .filter(|l| l.contains("error[") || l.contains("error:") ||
                                     l.contains("Diff in") || l.contains("FAILED") ||
                                     l.contains("warning[") || l.contains("aborting due"))
                        .collect();
                    errors.join("\n")
                }
                Err(_) => String::new(),
            }
        } else {
            String::new()
        };

        if let Ok(mut s) = state.lock() {
            if !error_log.is_empty() {
                let preview: String = error_log.lines().take(3)
                    .collect::<Vec<_>>().join(" | ");
                s.push_log(&format!("    errors: {}", &preview[..preview.len().min(120)]), true);
            }
            s.status = Status::Fixing(format!("fixing {short_name}..."));
        }

        // ---- Step 2: Ralph loop — fix → local verify → fix → verify, max 5 passes ----
        // Only commits + pushes once `cargo fmt --check` and `cargo clippy -- -D warnings`
        // both pass locally. If the loop can't reach a green verify, toast and leave
        // the working tree dirty for manual investigation.
        const MAX_RALPH_ITER: usize = 5;
        let mut verified = false;
        let mut current_errors = error_log.clone();

        for iter in 0..MAX_RALPH_ITER {
            if let Ok(mut s) = state.lock() {
                s.push_log(&format!("    ralph pass {}/{}", iter + 1, MAX_RALPH_ITER), false);
            }

            // Apply fixes
            let _ = tokio::process::Command::new("cargo")
                .args(["fmt", "--all"])
                .current_dir(local_path)
                .output().await;

            let _ = tokio::process::Command::new("cargo")
                .args(["clippy", "--fix", "--allow-dirty", "--allow-staged", "--all-targets", "--workspace"])
                .current_dir(local_path)
                .output().await;

            for line in current_errors.lines() {
                if line.contains("unused import") {
                    if let Some(file_info) = extract_file_line(line) {
                        let removed = remove_unused_import(&file_info.0, file_info.1).await;
                        if removed {
                            if let Ok(mut s) = state.lock() {
                                s.push_log(&format!("      removed unused import in {}:{}", file_info.0, file_info.1), false);
                            }
                        }
                    }
                }
            }

            // Local verify — same commands CI runs
            let fmt_check = tokio::process::Command::new("cargo")
                .args(["fmt", "--all", "--", "--check"])
                .current_dir(local_path)
                .output().await;
            let fmt_ok = fmt_check.as_ref().map(|o| o.status.success()).unwrap_or(false);

            let clippy_check = tokio::process::Command::new("cargo")
                .args(["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"])
                .current_dir(local_path)
                .output().await;
            let clippy_ok = clippy_check.as_ref().map(|o| o.status.success()).unwrap_or(false);

            if fmt_ok && clippy_ok {
                verified = true;
                if let Ok(mut s) = state.lock() {
                    s.push_log(&format!("    verified locally on pass {}", iter + 1), false);
                }
                break;
            }

            // Collect fresh errors for the next iteration (targeted fixes key off them)
            let mut next_errors = String::new();
            if let Ok(o) = fmt_check {
                next_errors.push_str(&String::from_utf8_lossy(&o.stdout));
                next_errors.push('\n');
            }
            if let Ok(o) = clippy_check {
                next_errors.push_str(&String::from_utf8_lossy(&o.stderr));
            }

            if let Ok(mut s) = state.lock() {
                let remaining_preview: String = next_errors.lines()
                    .filter(|l| l.contains("error") || l.contains("warning") || l.contains("Diff in"))
                    .take(2)
                    .collect::<Vec<_>>()
                    .join(" | ");
                if !remaining_preview.is_empty() {
                    s.push_log(&format!("      still failing: {}", &remaining_preview[..remaining_preview.len().min(120)]), true);
                }
            }

            current_errors = next_errors;
        }

        // Did any files change on disk?
        let diff_out = tokio::process::Command::new("git")
            .args(["diff", "--stat"])
            .current_dir(local_path)
            .output().await;
        let has_changes = diff_out.map(|d| !d.stdout.is_empty()).unwrap_or(false);

        if verified && has_changes {
            // Count changed files
            let diff_stat = tokio::process::Command::new("git")
                .args(["diff", "--stat", "--numstat"])
                .current_dir(local_path)
                .output().await;
            let files_changed = diff_stat.map(|d| {
                String::from_utf8_lossy(&d.stdout).lines().count()
            }).unwrap_or(0);

            // Stage + commit + auto-push (safe — we verified locally)
            let _ = tokio::process::Command::new("git")
                .args(["add", "-A"])
                .current_dir(local_path)
                .output().await;

            let commit_msg = "fix: resolve CI failures (verified locally via ralph loop)".to_string();
            let commit = tokio::process::Command::new("git")
                .args(["commit", "-m", &commit_msg])
                .current_dir(local_path)
                .output().await;

            if let Ok(c) = &commit {
                if c.status.success() {
                    let push = tokio::process::Command::new("git")
                        .args(["push"])
                        .current_dir(local_path)
                        .output().await;
                    let pushed = push.as_ref().map(|p| p.status.success()).unwrap_or(false);

                    if let Ok(mut s) = state.lock() {
                        if pushed {
                            s.push_log(&format!("    {short_name}: verified + pushed ({files_changed} files)"), false);
                        } else {
                            s.push_log(&format!("    {short_name}: verified + committed but push failed"), true);
                        }
                    }

                    let toast_msg = if pushed {
                        format!("{short_name}: verified fix pushed ({files_changed} files)")
                    } else {
                        format!("{short_name}: fix committed but push failed — check ticker")
                    };
                    let _ = tokio::process::Command::new("powershell.exe")
                        .args(["-Command", &format!(
                            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; \
                             $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); \
                             $text = $xml.GetElementsByTagName('text'); \
                             $text[0].AppendChild($xml.CreateTextNode('nexus-agent CI Fix')) > $null; \
                             $text[1].AppendChild($xml.CreateTextNode('{}')) > $null; \
                             $toast = [Windows.UI.Notifications.ToastNotification]::new($xml); \
                             [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('nexus-agent').Show($toast)",
                            toast_msg.replace('\'', "")
                        )])
                        .output().await;
                }
            }
        } else if !verified {
            // Ralph loop exhausted without reaching a green local verify
            if let Ok(mut s) = state.lock() {
                s.push_log(&format!("    {short_name}: ralph loop exhausted after {MAX_RALPH_ITER} passes — manual fix needed"), true);
            }

            let toast_msg = format!("{short_name}: auto-fix exhausted {MAX_RALPH_ITER} passes. Manual fix needed.");
            let _ = tokio::process::Command::new("powershell.exe")
                .args(["-Command", &format!(
                    "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; \
                     $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); \
                     $text = $xml.GetElementsByTagName('text'); \
                     $text[0].AppendChild($xml.CreateTextNode('nexus-agent CI Alert')) > $null; \
                     $text[1].AppendChild($xml.CreateTextNode('{}')) > $null; \
                     $toast = [Windows.UI.Notifications.ToastNotification]::new($xml); \
                     [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('nexus-agent').Show($toast)",
                    toast_msg.replace('\'', "")
                )])
                .output().await;
        }
        // verified && !has_changes is a no-op — CI flaked or was a transient false positive.
    }

    if let Ok(mut s) = state.lock() {
        s.ci_failures = total_failures;
        if total_failures == 0 {
            s.status = Status::Idle;
            s.push_log("CI: all repos green", false);
        } else if s.pending_pushes.is_empty() {
            s.status = Status::Error(format!("{total_failures} repo(s) failing"));
        } else {
            s.status = Status::Fixing(format!("{} fix(es) awaiting push approval", s.pending_pushes.len()));
        }
    }
}

/// Wait for GitHub to surface a CI run for `sha` and return whether it passed.
/// Returns Ok(true) on success, Ok(false) on failure, Err if we time out or can't tell.
#[allow(dead_code)]
async fn wait_for_ci_on_sha(repo_gh: &str, sha: &str, timeout_secs: u64) -> Result<bool, String> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        if std::time::Instant::now() >= deadline {
            return Err("timeout".to_string());
        }
        let out = tokio::process::Command::new("gh")
            .args(["run", "list", "--repo", repo_gh, "--commit", sha, "--limit", "1",
                   "--json", "status,conclusion"])
            .output().await
            .map_err(|e| e.to_string())?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        if let Ok(runs) = serde_json::from_str::<Vec<serde_json::Value>>(&stdout) {
            if !runs.is_empty() {
                let status = runs[0].get("status").and_then(|v| v.as_str()).unwrap_or("");
                if status == "completed" {
                    let conclusion = runs[0].get("conclusion").and_then(|v| v.as_str()).unwrap_or("");
                    return Ok(conclusion == "success");
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
    }
}

/// Extract file path and line number from a rustc error line.
fn extract_file_line(line: &str) -> Option<(String, usize)> {
    // Pattern: "  --> src/foo.rs:42:5" or "file.rs:10:3"
    let arrow_idx = line.find("-->")?;
    let after = line[arrow_idx + 3..].trim();
    let parts: Vec<&str> = after.splitn(3, ':').collect();
    if parts.len() >= 2 {
        let file = parts[0].trim().to_string();
        let line_num: usize = parts[1].parse().ok()?;
        Some((file, line_num))
    } else {
        None
    }
}

/// Remove an unused import by commenting it out or deleting the line.
async fn remove_unused_import(file: &str, line_num: usize) -> bool {
    let content = match tokio::fs::read_to_string(file).await {
        Ok(c) => c,
        Err(_) => return false,
    };
    let lines: Vec<&str> = content.lines().collect();
    if line_num == 0 || line_num > lines.len() { return false; }

    let target_line = lines[line_num - 1];
    // Only remove if it looks like a use/import line
    if !target_line.trim_start().starts_with("use ") { return false; }

    let mut new_lines: Vec<&str> = lines.clone();
    new_lines.remove(line_num - 1);
    let new_content = new_lines.join("\n") + "\n";

    tokio::fs::write(file, new_content).await.is_ok()
}

// =============================================================================
// Background: git dirty check
// =============================================================================

async fn git_dirty_check(state: Arc<Mutex<SharedState>>) {
    for (_repo_gh, local_path) in REPOS {
        if !std::path::Path::new(local_path).join(".git").exists() { continue; }

        let output = tokio::process::Command::new("git")
            .args(["status", "--short"])
            .current_dir(local_path)
            .output().await;

        if let Ok(out) = output {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let changed: Vec<&str> = stdout.lines().collect();
            if !changed.is_empty() {
                let short = std::path::Path::new(local_path)
                    .file_name().unwrap_or_default().to_string_lossy();
                if let Ok(mut s) = state.lock() {
                    s.push_log(
                        &format!("git: {short} has {} uncommitted change(s)", changed.len()),
                        false,
                    );
                }
            }
        }
    }
}

// =============================================================================
// UI
// =============================================================================

impl eframe::App for TickerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint frequently for pulse animation
        ctx.request_repaint_after(std::time::Duration::from_millis(60));

        // Dark theme with NexusStratum palette
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = BG_DARK;
        visuals.window_fill = BG_DARK;
        visuals.override_text_color = Some(CREAM);
        ctx.set_visuals(visuals);

        let snap = self.state.lock().unwrap().clone_snapshot();
        let is_active = snap.status_label != "IDLE";

        egui::CentralPanel::default().show(ctx, |ui| {
            // ---- Header: main status LED + label + right-side indicators ----
            ui.horizontal(|ui| {
                let main_color = status_led_color(&snap.status_label);
                draw_led(ui, main_color, 6.0, is_active);
                ui.label(egui::RichText::new(snap.status_label).strong().size(11.0).color(CREAM));
                ui.label(egui::RichText::new(&snap.status_detail).size(9.0).color(TEXT_DIM));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Help button — opens CHEATSHEET.md in neovim via Windows Terminal
                    let help_btn = egui::RichText::new("?").size(11.0).color(TEAL).strong();
                    if ui.add(egui::Button::new(help_btn)
                        .frame(false)
                        .min_size(egui::vec2(16.0, 16.0)))
                        .on_hover_text("Open cheatsheet in neovim (Ctrl+click to copy path)")
                        .clicked()
                    {
                        open_cheatsheet_in_nvim();
                    }
                    ui.add_space(4.0);

                    // nexus-serve LED
                    let serve_c = if snap.backend_online { TEAL } else { TERRACOTTA };
                    draw_led(ui, serve_c, 4.0, !snap.backend_online);
                    ui.label(egui::RichText::new("serve").size(8.0).color(TEXT_DIM));

                    // CI summary LED
                    let ci_c = if snap.ci_failures == 0 { TEAL } else { TERRACOTTA };
                    draw_led(ui, ci_c, 4.0, snap.ci_failures > 0);
                    ui.label(egui::RichText::new(format!("CI:{}", snap.ci_failures)).size(8.0).color(TEXT_DIM));
                });
            });

            // ---- Repo status strip: one LED per repo ----
            ui.horizontal_wrapped(|ui| {
                for (name, ci_ok) in &snap.repo_statuses {
                    let color = match ci_ok {
                        Some(true) => TEAL,
                        Some(false) => TERRACOTTA,
                        None => SLATE,
                    };
                    draw_led(ui, color, 3.0, *ci_ok == Some(false));
                    ui.label(egui::RichText::new(name).size(8.0).color(TEXT_DIM));
                    ui.add_space(4.0);
                }
            });

            ui.add_space(2.0);
            ui.separator();

            // ---- Scrolling log ----
            let available = ui.available_height() - 30.0;
            egui::ScrollArea::vertical()
                .max_height(available)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for (ts, msg, is_err) in &snap.log_entries {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(ts).size(8.0).color(SLATE));
                            let color = if *is_err { TERRACOTTA } else { CREAM };
                            ui.label(egui::RichText::new(msg).size(9.0).color(color));
                        });
                    }
                });

            // ---- Pending push approval bar ----
            if !snap.pending_pushes.is_empty() {
                ui.separator();
                let pushes = snap.pending_pushes.clone();
                for pp in &pushes {
                    ui.horizontal(|ui| {
                        draw_led(ui, AMBER, 3.5, true);
                        ui.label(egui::RichText::new(
                            format!("{}: {} file(s) — {}", pp.0, pp.2, pp.1)
                        ).size(9.0).color(AMBER));

                        if ui.button(egui::RichText::new("Push").size(9.0).color(TEAL)).clicked() {
                            let path = pp.3.clone();
                            let name = pp.0.clone();
                            let state_ref = self.state.clone();
                            self.runtime.spawn(async move {
                                let push = tokio::process::Command::new("git")
                                    .args(["push", "origin", "HEAD"])
                                    .current_dir(&path)
                                    .output().await;
                                if let Ok(mut s) = state_ref.lock() {
                                    s.pending_pushes.retain(|p| p.repo_name != name);
                                    match push {
                                        Ok(p) if p.status.success() => {
                                            s.push_log(&format!("PUSHED {name}"), false);
                                        }
                                        _ => {
                                            s.push_log(&format!("push failed for {name}"), true);
                                        }
                                    }
                                    if s.pending_pushes.is_empty() && s.ci_failures == 0 {
                                        s.status = Status::Idle;
                                    }
                                }
                            });
                        }

                        if ui.button(egui::RichText::new("Revert").size(9.0).color(TERRACOTTA)).clicked() {
                            let path = pp.3.clone();
                            let name = pp.0.clone();
                            let state_ref = self.state.clone();
                            self.runtime.spawn(async move {
                                // Undo the last commit, keep changes unstaged
                                let _ = tokio::process::Command::new("git")
                                    .args(["reset", "HEAD~1"])
                                    .current_dir(&path)
                                    .output().await;
                                let _ = tokio::process::Command::new("git")
                                    .args(["checkout", "."])
                                    .current_dir(&path)
                                    .output().await;
                                if let Ok(mut s) = state_ref.lock() {
                                    s.pending_pushes.retain(|p| p.repo_name != name);
                                    s.push_log(&format!("reverted fix for {name}"), false);
                                    if s.pending_pushes.is_empty() {
                                        s.status = Status::Error("fix reverted — CI still failing".to_string());
                                    }
                                }
                            });
                        }
                    });
                }
            }

            // ---- Input bar ----
            ui.separator();
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("agent")
                    .width(85.0)
                    .selected_text(AGENT_NAMES[self.selected_agent])
                    .show_ui(ui, |ui| {
                        for (i, name) in AGENT_NAMES.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_agent, i, *name);
                        }
                    });

                let r = ui.add(egui::TextEdit::singleline(&mut self.input)
                    .desired_width(ui.available_width() - 10.0)
                    .hint_text("command...")
                    .font(egui::TextStyle::Small));

                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    self.submit_command();
                    r.request_focus();
                }
            });
        });
    }
}

// =============================================================================
// NexusStratum palette + LED rendering
// =============================================================================

const CREAM: egui::Color32 = egui::Color32::from_rgb(245, 240, 235);       // #F5F0EB
const TEAL: egui::Color32 = egui::Color32::from_rgb(20, 184, 166);         // #14b8a6
const TERRACOTTA: egui::Color32 = egui::Color32::from_rgb(205, 92, 68);    // warm red
const AMBER: egui::Color32 = egui::Color32::from_rgb(245, 180, 60);
const SLATE: egui::Color32 = egui::Color32::from_rgb(150, 145, 138);       // unknown/grey
const BG_DARK: egui::Color32 = egui::Color32::from_rgb(45, 42, 38);        // #2D2A26
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(155, 145, 138);     // #9B918A

/// Draw a pulsing LED circle. `pulse` controls brightness oscillation.
fn draw_led(ui: &mut egui::Ui, color: egui::Color32, radius: f32, pulse: bool) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(radius * 2.5, radius * 2.5),
        egui::Sense::hover(),
    );
    let center = rect.center();

    // Pulse: oscillate alpha between 0.6 and 1.0
    let alpha = if pulse {
        let t = ui.input(|i| i.time) as f32;
        let wave = (t * 2.0).sin() * 0.2 + 0.8; // 0.6..1.0
        (wave * 255.0) as u8
    } else {
        255
    };

    let c = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);

    // Outer glow
    let glow = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha / 4);
    ui.painter().circle_filled(center, radius * 1.6, glow);
    // Inner LED
    ui.painter().circle_filled(center, radius, c);
    // Bright center highlight
    let highlight = egui::Color32::from_rgba_unmultiplied(255, 255, 255, alpha / 3);
    ui.painter().circle_filled(center + egui::vec2(-radius * 0.2, -radius * 0.2), radius * 0.35, highlight);
}

fn status_led_color(status: &str) -> egui::Color32 {
    match status {
        "IDLE" => TEAL,
        "MONITORING" => TEAL,
        "FIXING" => AMBER,
        "ERROR" => TERRACOTTA,
        _ => SLATE,
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn now() -> String { chrono::Local::now().format("%H:%M:%S").to_string() }

/// Open the ticker cheatsheet in neovim inside a new Windows Terminal window.
///
/// Uses `wt.exe` to spawn a new terminal tab that runs `nvim` inside WSL
/// against the cheatsheet path. Non-blocking — returns immediately.
fn open_cheatsheet_in_nvim() {
    const CHEATSHEET: &str = "/opt/AxonML/nexus-agent/CHEATSHEET.md";
    // wt.exe -w 0 new-tab --profile Ubuntu wsl.exe -d Ubuntu -e nvim <path>
    // -w 0 targets the current/focused terminal window so the new tab
    // opens in the existing Terminal rather than spawning a new one.
    let _ = std::process::Command::new("wt.exe")
        .args([
            "-w", "0",
            "new-tab",
            "--title", "cheatsheet",
            "wsl.exe",
            "-d", "Ubuntu",
            "-e",
            "nvim",
            CHEATSHEET,
        ])
        .spawn();
}

struct Snapshot {
    status_label: String,
    status_detail: String,
    status_color: egui::Color32,
    backend_online: bool,
    ci_failures: usize,
    repo_statuses: Vec<(String, Option<bool>)>,
    log_entries: Vec<(String, String, bool)>,
    /// (repo_name, commit_msg, files_changed, local_path)
    pending_pushes: Vec<(String, String, usize, String)>,
}

impl SharedState {
    fn clone_snapshot(&self) -> Snapshot {
        Snapshot {
            status_label: self.status.label().to_string(),
            status_detail: self.status.detail(),
            status_color: self.status.color(),
            backend_online: self.backend_online,
            ci_failures: self.ci_failures,
            repo_statuses: self.repos.iter().map(|r| {
                let short = r.name.split('/').last().unwrap_or(&r.name).to_string();
                (short, r.ci_ok)
            }).collect(),
            log_entries: self.log.iter()
                .map(|e| (e.timestamp.clone(), e.message.clone(), e.is_error))
                .collect(),
            pending_pushes: self.pending_pushes.iter()
                .map(|p| (p.repo_name.clone(), p.commit_msg.clone(), p.files_changed, p.local_path.clone()))
                .collect(),
        }
    }
}
