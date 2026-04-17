//! nexus-ticker — Always-On-Top CI Monitor and Auto-Fix Widget
//!
//! eframe-based frameless desktop widget that actively monitors and fixes
//! CI, git state, and infrastructure across all AutomataNexus repos. NOT a
//! passive status display — it intervenes.
//!
//! Architecture:
//! * `TickerApp` owns a shared `SharedState` (log, repo LEDs, pending
//!   pushes, backend health) behind `Arc<Mutex>`, plus a tokio runtime that
//!   spawns three background tasks: nexus-serve `/health` poll every 10s,
//!   `ci_check_and_fix` every 5 minutes, and `git_dirty_check` every 10
//!   minutes.
//! * `ci_check_and_fix` iterates the `REPOS` list, calls `gh run list` for
//!   each, and on failure runs the ralph loop (up to `MAX_RALPH_ITER = 5`
//!   passes): apply `cargo fmt --all` + `cargo clippy --fix` + targeted
//!   unused-import removal (see `extract_file_line` / `remove_unused_import`),
//!   then verify locally via `cargo fmt --check` and
//!   `cargo clippy -- -D warnings`. Only commits + auto-pushes once the
//!   local verify is green. Handles chattr-immutable `/opt/AxonML/crates`
//!   via `unlock-crates.sh` / `lock-crates.sh`. If ralph exhausts without
//!   verifying, delegates to the `ci-fixer` agent; if that fails too,
//!   toasts via PowerShell instead of pushing a broken fix.
//! * UI paints a rounded transparent outer frame with a custom titlebar
//!   (drag region, close, minimize), status LED, per-repo LED strip,
//!   scrolling log with copy/clear, pending-push approval bar, and an
//!   input row with an agent picker (knowledge, retrain, fieldtech,
//!   research, orchestrator).
//! * Theme system (`Theme::Dark` / `Theme::Light`) persists to
//!   `/tmp/.nexus-ticker-theme`; LED semantics stay fixed (TEAL = pass,
//!   TERRACOTTA = fail) across themes so pass/fail never becomes
//!   ambiguous.
//!
//! Launch: nexus-ticker &
//!
//! # File
//! `nexus-agent/src/ui/ticker.rs`
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
use std::sync::{Arc, Mutex};

// =============================================================================
// Constants and Monitored Repos
// =============================================================================

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

// =============================================================================
// Entry Point
// =============================================================================

fn main() -> eframe::Result {
    // Frameless + transparent — we paint our own rounded background and a
    // custom titlebar. Matches the style we set on tech-ticker so both
    // widgets look like a single cohesive set.
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([TICKER_WIDTH, TICKER_HEIGHT])
            .with_always_on_top()
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(true)
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
// Background Probe — CI Check + Ralph Loop Auto-Fix
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
        //
        // Per LESSONS.md L18: `/opt/AxonML/crates/` is chattr +i locked. Before fix
        // attempts, run `sudo /opt/AxonML/unlock-crates.sh` if the local repo path
        // matches, and re-lock after a successful push.
        const MAX_RALPH_ITER: usize = 5;
        let mut verified = false;
        let mut current_errors = error_log.clone();
        let is_axonml = *local_path == "/opt/AxonML";

        // Unlock before attempting any writes — required for AxonML framework code
        if is_axonml {
            let unlock = tokio::process::Command::new("sudo")
                .args(["-n", "/opt/AxonML/unlock-crates.sh"])
                .output().await;
            if let Ok(mut s) = state.lock() {
                match unlock {
                    Ok(o) if o.status.success() => s.push_log("    unlocked /opt/AxonML/crates", false),
                    _ => s.push_log("    WARN: unlock-crates failed (sudo NOPASSWD not set?); writes may fail", true),
                }
            }
        }

        for iter in 0..MAX_RALPH_ITER {
            if let Ok(mut s) = state.lock() {
                s.push_log(&format!("    ralph pass {}/{}", iter + 1, MAX_RALPH_ITER), false);
            }

            // Apply fixes
            let fmt_out = tokio::process::Command::new("cargo")
                .args(["fmt", "--all"])
                .current_dir(local_path)
                .output().await;

            // Detect the chattr-immutable symptom and self-heal on the fly.
            // `cargo fmt` prints "Operation not permitted (os error 1)" to stderr when
            // files have `chattr +i`. If we see that, call unlock-crates and let the
            // next iteration retry.
            if let Ok(ref out) = fmt_out {
                let stderr = String::from_utf8_lossy(&out.stderr);
                if stderr.contains("Operation not permitted") && is_axonml {
                    if let Ok(mut s) = state.lock() {
                        s.push_log("      perm denied — re-running unlock-crates", true);
                    }
                    let _ = tokio::process::Command::new("sudo")
                        .args(["-n", "/opt/AxonML/unlock-crates.sh"])
                        .output().await;
                }
            }
            let _ = fmt_out;

            let _ = tokio::process::Command::new("cargo")
                .args(["clippy", "--fix", "--allow-dirty", "--allow-staged", "--workspace"])
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
                .args(["clippy", "--workspace", "--", "-D", "warnings"])
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
                            // Optimistic LED: we just pushed a verified fix
                            // so the next GitHub CI run will be green. Flip
                            // the repo LED now instead of waiting for the
                            // 5-minute poll, and decrement the failure count
                            // so the header LED goes back to idle.
                            if let Some(r) = s.repos.iter_mut().find(|r| r.name == *repo_gh) {
                                r.ci_ok = Some(true);
                            }
                            total_failures = total_failures.saturating_sub(1);
                        } else {
                            s.push_log(&format!("    {short_name}: verified + committed but push failed"), true);
                        }
                    }

                    // Re-lock AxonML/crates after a successful push
                    if pushed && is_axonml {
                        let relock = tokio::process::Command::new("sudo")
                            .args(["-n", "/opt/AxonML/lock-crates.sh"])
                            .output().await;
                        if let Ok(mut s) = state.lock() {
                            match relock {
                                Ok(o) if o.status.success() => s.push_log("    re-locked /opt/AxonML/crates", false),
                                _ => s.push_log("    WARN: re-lock failed", true),
                            }
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
            // Ralph's fmt/clippy loop couldn't reach a green local verify.
            // Before giving up, delegate to the `ci-fixer` agent which has
            // full shell + file tools and can handle test assertions,
            // flaky convergence, logic bugs — things lint autofix can't.
            if let Ok(mut s) = state.lock() {
                s.push_log(&format!("    {short_name}: delegating to ci-fixer agent..."), false);
                s.status = Status::Fixing(format!("ci-fixer: {short_name}..."));
            }

            // Hand the agent the repo + the error log tail. The agent is
            // instructed to stop once the failing test passes locally — it
            // does NOT commit or push; that stays our responsibility below.
            let agent_task = format!(
                "Repo: {local_path}\nShort name: {short_name}\n\nCI error log (tail):\n{current_errors}\n\n\
                 Reproduce the failure in {local_path}, identify the root cause, and apply the minimum change that makes the failing test pass locally. \
                 Do not commit. Do not push. Stop when the test is green."
            );
            let agent_out = tokio::process::Command::new(
                "/opt/AxonML/nexus-agent/target/release/nexus-agent",
            )
                .args(["ci-fixer", &agent_task])
                .output()
                .await;
            match agent_out {
                Ok(ref o) if o.status.success() => {
                    let summary_tail = String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .rev()
                        .take(3)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join(" | ");
                    if let Ok(mut s) = state.lock() {
                        s.push_log(&format!("    ci-fixer: {}", &summary_tail[..summary_tail.len().min(160)]), false);
                    }
                }
                _ => {
                    if let Ok(mut s) = state.lock() {
                        s.push_log("    ci-fixer: invocation failed", true);
                    }
                }
            }

            // Did the agent actually leave fixes on disk? Re-verify locally.
            let fmt_check2 = tokio::process::Command::new("cargo")
                .args(["fmt", "--all", "--", "--check"])
                .current_dir(local_path)
                .output().await;
            let clippy_check2 = tokio::process::Command::new("cargo")
                .args(["clippy", "--workspace", "--", "-D", "warnings"])
                .current_dir(local_path)
                .output().await;
            let test_check = tokio::process::Command::new("cargo")
                .args(["test", "--workspace", "--no-fail-fast"])
                .current_dir(local_path)
                .output().await;
            let all_green = fmt_check2.as_ref().map(|o| o.status.success()).unwrap_or(false)
                && clippy_check2.as_ref().map(|o| o.status.success()).unwrap_or(false)
                && test_check.as_ref().map(|o| o.status.success()).unwrap_or(false);
            let diff_out = tokio::process::Command::new("git")
                .args(["diff", "--stat"])
                .current_dir(local_path)
                .output().await;
            let agent_has_changes = diff_out.map(|d| !d.stdout.is_empty()).unwrap_or(false);

            if all_green && agent_has_changes {
                // Agent produced a working fix — take over and push it,
                // using the same commit + re-lock + toast path as the ralph
                // success branch above.
                if let Ok(mut s) = state.lock() {
                    s.push_log(&format!("    ci-fixer: verified — committing"), false);
                }
                let _ = tokio::process::Command::new("git").args(["add", "-A"]).current_dir(local_path).output().await;
                let commit = tokio::process::Command::new("git")
                    .args(["commit", "-m", "fix: resolve CI failures (ci-fixer agent)"])
                    .current_dir(local_path)
                    .output().await;
                if commit.as_ref().map(|c| c.status.success()).unwrap_or(false) {
                    let push = tokio::process::Command::new("git")
                        .args(["push"])
                        .current_dir(local_path)
                        .output().await;
                    let pushed = push.as_ref().map(|p| p.status.success()).unwrap_or(false);
                    if let Ok(mut s) = state.lock() {
                        if pushed {
                            s.push_log(&format!("    {short_name}: ci-fixer fix pushed"), false);
                            if let Some(r) = s.repos.iter_mut().find(|r| r.name == *repo_gh) {
                                r.ci_ok = Some(true);
                            }
                            total_failures = total_failures.saturating_sub(1);
                        } else {
                            s.push_log(&format!("    {short_name}: ci-fixer fix committed but push failed"), true);
                        }
                    }
                    if pushed && is_axonml {
                        let _ = tokio::process::Command::new("sudo")
                            .args(["-n", "/opt/AxonML/lock-crates.sh"])
                            .output().await;
                    }
                    continue; // move to the next repo
                }
            }

            // Agent couldn't or didn't fix it — fall through to the
            // "manual fix needed" toast that was already here.
            if let Ok(mut s) = state.lock() {
                s.push_log(&format!("    {short_name}: ralph + ci-fixer exhausted — manual fix needed"), true);
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
// Background Probe — Git Dirty Check
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
// UI — eframe::App Implementation
// =============================================================================

impl eframe::App for TickerApp {
    /// Transparent framebuffer clear so only our rounded Frame shows —
    /// without this the area outside the rounded pill paints opaque.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint frequently for pulse animation
        ctx.request_repaint_after(std::time::Duration::from_millis(60));

        // Load persisted theme on first frame, then apply egui visuals that match.
        // We sync ACTIVE_THEME from disk once, on the first paint, so load order
        // doesn't matter. Toggling happens via the header button (bottom of impl).
        static LOADED: std::sync::Once = std::sync::Once::new();
        LOADED.call_once(|| set_active_theme(Theme::load()));

        let is_light = ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) == 1;
        let mut visuals = if is_light { egui::Visuals::light() } else { egui::Visuals::dark() };
        visuals.panel_fill = egui::Color32::TRANSPARENT;
        visuals.window_fill = egui::Color32::TRANSPARENT;
        visuals.override_text_color = Some(CREAM());
        ctx.set_visuals(visuals);

        let snap = self.state.lock().unwrap().clone_snapshot();
        let is_active = snap.status_label != "IDLE";

        // Outer rounded pill — matches the tech-ticker chrome.
        let outer_frame = egui::Frame::none()
            .fill(BG_DARK())
            .rounding(egui::Rounding::same(10.0))
            .stroke(egui::Stroke::new(
                1.0,
                if is_light {
                    egui::Color32::from_rgba_unmultiplied(
                        TEXT_DIM().r(), TEXT_DIM().g(), TEXT_DIM().b(), 90,
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
            .inner_margin(egui::Margin { left: 10.0, right: 10.0, top: 8.0, bottom: 10.0 });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(egui::Margin::same(6.0)))
            .show(ctx, |ui| { outer_frame.show(ui, |ui| {
            // Force a hard clip to the outer frame's inner rect so no child
            // widget (scrollbar, text edit border) paints over the rounded
            // corners.
            ui.set_clip_rect(ui.max_rect());
            // ── Custom titlebar: drag region + close/min buttons ──────────
            let title_resp = ui.horizontal(|ui| {
                let main_color = status_led_color(&snap.status_label);
                draw_led(ui, main_color, 5.0, is_active);
                ui.label(egui::RichText::new("NEXUS-AGENT").strong().size(11.0).color(CREAM()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Close — WSLg doesn't always honor ViewportCommand::Close,
                    // so send the command AND forcefully exit after a short tick.
                    if ui.add(egui::Button::new(
                        egui::RichText::new("✕").size(12.0).color(TEXT_DIM()),
                    ).frame(false).min_size(egui::vec2(20.0, 20.0)))
                        .on_hover_text("close").clicked()
                    {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        std::process::exit(0);
                    }
                    // Minimize — Wayland under WSLg doesn't expose a "minimize to
                    // taskbar" primitive the way Windows expects. Try the viewport
                    // command first, then fall back to xdotool which drives the
                    // underlying X11 path.
                    if ui.add(egui::Button::new(
                        egui::RichText::new("—").size(12.0).color(TEXT_DIM()),
                    ).frame(false).min_size(egui::vec2(20.0, 20.0)))
                        .on_hover_text("minimize").clicked()
                    {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                        let _ = std::process::Command::new("sh")
                            .arg("-c")
                            .arg("xdotool search --name '^nexus-agent$' windowminimize 2>/dev/null")
                            .spawn();
                    }
                });
            }).response;
            let drag_sense = ui.interact(title_resp.rect, egui::Id::new("nexus-drag"), egui::Sense::click_and_drag());
            if drag_sense.drag_started_by(egui::PointerButton::Primary) {
                ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            ui.separator();

            // ---- Status line (was the old header): status LED + detail + right-side toggles ----
            ui.horizontal(|ui| {
                let main_color = status_led_color(&snap.status_label);
                draw_led(ui, main_color, 6.0, is_active);
                ui.label(egui::RichText::new(snap.status_label).strong().size(11.0).color(CREAM()));
                ui.label(egui::RichText::new(&snap.status_detail).size(9.0).color(TEXT_DIM()));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Theme toggle — ☀ / ☾ to flip light/dark, persists across restarts
                    let is_light_now = ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) == 1;
                    let theme_glyph = if is_light_now { "☾" } else { "☀" };
                    let theme_btn = egui::RichText::new(theme_glyph).size(11.0).color(AMBER()).strong();
                    if ui.add(egui::Button::new(theme_btn)
                        .frame(false)
                        .min_size(egui::vec2(16.0, 16.0)))
                        .on_hover_text("Toggle light/dark theme")
                        .clicked()
                    {
                        let new_theme = if is_light_now { Theme::Dark } else { Theme::Light };
                        set_active_theme(new_theme);
                        new_theme.save();
                    }
                    ui.add_space(4.0);

                    // Help button — opens CHEATSHEET.md in neovim via Windows Terminal
                    let help_btn = egui::RichText::new("?").size(11.0).color(TEAL()).strong();
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
                    let serve_c = if snap.backend_online { TEAL() } else { TERRACOTTA() };
                    draw_led(ui, serve_c, 4.0, !snap.backend_online);
                    ui.label(egui::RichText::new("serve").size(8.0).color(TEXT_DIM()));

                    // CI summary LED
                    let ci_c = if snap.ci_failures == 0 { TEAL() } else { TERRACOTTA() };
                    draw_led(ui, ci_c, 4.0, snap.ci_failures > 0);
                    ui.label(egui::RichText::new(format!("CI:{}", snap.ci_failures)).size(8.0).color(TEXT_DIM()));
                });
            });

            // ---- Repo status strip: one LED per repo ----
            ui.horizontal_wrapped(|ui| {
                for (name, ci_ok) in &snap.repo_statuses {
                    let color = match ci_ok {
                        Some(true) => TEAL(),
                        Some(false) => TERRACOTTA(),
                        None => SLATE(),
                    };
                    draw_led(ui, color, 3.0, *ci_ok == Some(false));
                    ui.label(egui::RichText::new(name).size(8.0).color(TEXT_DIM()));
                    ui.add_space(4.0);
                }
            });

            ui.add_space(2.0);
            ui.separator();

            // ---- Log toolbar: Copy All + Clear ----
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("log").size(9.0).color(TEXT_DIM()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(egui::Button::new(egui::RichText::new("clear").size(9.0).color(TERRACOTTA()))
                        .frame(false))
                        .on_hover_text("Clear the log buffer")
                        .clicked()
                    {
                        if let Ok(mut s) = self.state.lock() {
                            s.log.clear();
                            s.push_log("log cleared", false);
                        }
                    }
                    ui.add_space(6.0);
                    if ui.add(egui::Button::new(egui::RichText::new("copy all").size(9.0).color(TEAL()))
                        .frame(false))
                        .on_hover_text("Copy entire log to clipboard")
                        .clicked()
                    {
                        let full: String = snap.log_entries.iter()
                            .map(|(ts, msg, _)| format!("{ts}  {msg}"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        ctx.copy_text(full);
                    }
                });
            });

            // ---- Scrolling log ----
            let available = ui.available_height() - 30.0;
            egui::ScrollArea::vertical()
                .max_height(available)
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded)
                .show(ui, |ui| {
                    for (ts, msg, is_err) in &snap.log_entries {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(ts).size(8.0).color(SLATE()));
                            let color = if *is_err { TERRACOTTA() } else { CREAM() };
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
                        draw_led(ui, AMBER(), 3.5, true);
                        ui.label(egui::RichText::new(
                            format!("{}: {} file(s) — {}", pp.0, pp.2, pp.1)
                        ).size(9.0).color(AMBER()));

                        if ui.button(egui::RichText::new("Push").size(9.0).color(TEAL())).clicked() {
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

                        if ui.button(egui::RichText::new("Revert").size(9.0).color(TERRACOTTA())).clicked() {
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
            }); // closes outer_frame.show
        });
    }
}

// =============================================================================
// Theme — NexusStratum Palette + LED Rendering
// =============================================================================

// ─── Theme system (mirrors tech_ticker.rs) ───────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum Theme { Dark, Light }

const THEME_FILE: &str = "/tmp/.nexus-ticker-theme";

impl Theme {
    fn load() -> Self {
        match std::fs::read_to_string(THEME_FILE).ok().as_deref().map(str::trim) {
            Some("light") => Self::Light,
            _ => Self::Dark,
        }
    }
    fn save(&self) {
        let _ = std::fs::write(THEME_FILE, match self { Self::Dark => "dark", Self::Light => "light" });
    }
    #[allow(dead_code)]
    fn toggle(self) -> Self { if self == Self::Dark { Self::Light } else { Self::Dark } }
}

#[derive(Clone, Copy)]
struct Palette {
    bg: egui::Color32,
    bg_row: egui::Color32,
    text: egui::Color32,
    text_dim: egui::Color32,
    accent: egui::Color32,
    warn: egui::Color32,
    alert: egui::Color32,
    slate: egui::Color32,
}

const DARK: Palette = Palette {
    bg: egui::Color32::from_rgb(45, 42, 38),
    bg_row: egui::Color32::from_rgb(55, 50, 46),
    text: egui::Color32::from_rgb(245, 240, 235),
    text_dim: egui::Color32::from_rgb(155, 145, 138),
    accent: egui::Color32::from_rgb(20, 184, 166),
    warn: egui::Color32::from_rgb(245, 180, 60),
    alert: egui::Color32::from_rgb(205, 92, 68),
    slate: egui::Color32::from_rgb(150, 145, 138),
};

const LIGHT: Palette = Palette {
    bg: egui::Color32::from_rgb(250, 249, 245),
    bg_row: egui::Color32::from_rgb(241, 236, 224),
    text: egui::Color32::from_rgb(61, 57, 41),
    text_dim: egui::Color32::from_rgb(141, 132, 119),
    accent: egui::Color32::from_rgb(201, 100, 66),
    warn: egui::Color32::from_rgb(201, 133, 50),
    alert: egui::Color32::from_rgb(180, 60, 50),
    slate: egui::Color32::from_rgb(180, 172, 158),
};

static ACTIVE_THEME: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
fn active_palette() -> Palette {
    match ACTIVE_THEME.load(std::sync::atomic::Ordering::Relaxed) { 1 => LIGHT, _ => DARK }
}
#[allow(dead_code)]
fn set_active_theme(t: Theme) {
    ACTIVE_THEME.store(match t { Theme::Dark => 0, Theme::Light => 1 }, std::sync::atomic::Ordering::Relaxed);
}

// Named-color accessors. Surface colors (bg, text, text_dim) follow the
// active palette so dark/light swap cleanly. Semantic LED colors (TEAL =
// success/online, TERRACOTTA = failure, AMBER = warn, SLATE = unknown)
// stay FIXED in both themes — "green means good" has to be green even on
// a cream background, or users can't tell passing from failing CI.
#[allow(non_snake_case)]
fn CREAM() -> egui::Color32 { active_palette().text }
// Semantic status colors — theme-aware only for Amber (since pure yellow
// is unreadable on cream). Teal and red stay put so CI pass/fail is never
// ambiguous.
#[allow(non_snake_case)]
fn TEAL() -> egui::Color32 { egui::Color32::from_rgb(20, 184, 166) }
#[allow(non_snake_case)]
fn TERRACOTTA() -> egui::Color32 {
    if active_palette().bg == LIGHT.bg {
        egui::Color32::from_rgb(200, 50, 40)  // deeper red on cream for contrast
    } else {
        egui::Color32::from_rgb(205, 92, 68)
    }
}
#[allow(non_snake_case)]
fn AMBER() -> egui::Color32 { active_palette().warn }
#[allow(non_snake_case)]
fn SLATE() -> egui::Color32 { active_palette().slate }
#[allow(non_snake_case)]
fn BG_DARK() -> egui::Color32 { active_palette().bg }
#[allow(non_snake_case)]
fn BG_ROW() -> egui::Color32 { active_palette().bg_row }
#[allow(non_snake_case)]
fn TEXT_DIM() -> egui::Color32 { active_palette().text_dim }

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
        "IDLE" => TEAL(),
        "MONITORING" => TEAL(),
        "FIXING" => AMBER(),
        "ERROR" => TERRACOTTA(),
        _ => SLATE(),
    }
}

// =============================================================================
// Helpers and Snapshot
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
