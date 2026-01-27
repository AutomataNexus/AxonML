//! Dashboard - Dashboard and Server Management Commands
//!
//! This module handles starting, stopping, and monitoring the AxonML
//! dashboard (frontend) and API server (backend).
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use colored::Colorize;

use crate::cli::{LogsArgs, StartArgs, StatusArgs, StopArgs};
use crate::error::{CliError, CliResult};

// =============================================================================
// Data Directory
// =============================================================================

/// Get the AxonML data directory
fn get_data_dir() -> PathBuf {
    std::env::var("AXONML_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("axonml")
        })
}

/// Get the PID file path for a service
fn get_pid_file(service: &str) -> PathBuf {
    get_data_dir().join(format!("{}.pid", service))
}

/// Get the log file path for a service
fn get_log_file(service: &str) -> PathBuf {
    get_data_dir().join("logs").join(format!("{}.log", service))
}

/// Read PID from file
fn read_pid(service: &str) -> Option<u32> {
    let pid_file = get_pid_file(service);
    if pid_file.exists() {
        fs::read_to_string(&pid_file)
            .ok()
            .and_then(|s| s.trim().parse().ok())
    } else {
        None
    }
}

/// Write PID to file
fn write_pid(service: &str, pid: u32) -> CliResult<()> {
    let pid_file = get_pid_file(service);
    if let Some(parent) = pid_file.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&pid_file, pid.to_string())?;
    Ok(())
}

/// Remove PID file
fn remove_pid(service: &str) -> CliResult<()> {
    let pid_file = get_pid_file(service);
    if pid_file.exists() {
        fs::remove_file(&pid_file)?;
    }
    Ok(())
}

/// Check if a process is running
#[cfg(unix)]
fn is_process_running(pid: u32) -> bool {
    // Use kill -0 to check if process exists
    let result = unsafe { libc::kill(pid as i32, 0) };
    result == 0
}

#[cfg(not(unix))]
fn is_process_running(pid: u32) -> bool {
    // On Windows, try to open the process
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {}", pid)])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

/// Stop a process by PID
#[cfg(unix)]
fn stop_process(pid: u32, force: bool) -> bool {
    let signal = if force { libc::SIGKILL } else { libc::SIGTERM };
    let result = unsafe { libc::kill(pid as i32, signal) };
    result == 0
}

#[cfg(not(unix))]
fn stop_process(pid: u32, _force: bool) -> bool {
    Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

// =============================================================================
// Start Command
// =============================================================================

/// Execute the start command
pub fn execute_start(args: StartArgs) -> CliResult<()> {
    let data_dir = get_data_dir();

    // Create necessary directories
    fs::create_dir_all(&data_dir)?;
    fs::create_dir_all(data_dir.join("logs"))?;

    let start_server = !args.dashboard;
    let start_dashboard = !args.server;

    // Print banner
    println!();
    println!(
        "{}",
        "  ╔═══════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        "  ║         AxonML Dashboard Starting         ║".cyan()
    );
    println!(
        "{}",
        "  ╚═══════════════════════════════════════════╝".cyan()
    );
    println!();

    if start_server {
        start_server_process(&args)?;
    }

    if start_dashboard {
        start_dashboard_process(&args)?;
    }

    println!();
    if start_server && start_dashboard {
        println!(
            "{} Dashboard: {}",
            "✓".green().bold(),
            format!("http://{}:{}", args.host, args.dashboard_port).cyan()
        );
        println!(
            "{} API Server: {}",
            "✓".green().bold(),
            format!("http://{}:{}", args.host, args.port).cyan()
        );
    } else if start_server {
        println!(
            "{} API Server: {}",
            "✓".green().bold(),
            format!("http://{}:{}", args.host, args.port).cyan()
        );
    } else if start_dashboard {
        println!(
            "{} Dashboard: {}",
            "✓".green().bold(),
            format!("http://{}:{}", args.host, args.dashboard_port).cyan()
        );
    }
    println!();
    println!("Use {} to stop services", "axon stop".yellow());
    println!("Use {} to view logs", "axon logs -f".yellow());
    println!();

    Ok(())
}

fn start_server_process(args: &StartArgs) -> CliResult<()> {
    // Check if already running
    if let Some(pid) = read_pid("server") {
        if is_process_running(pid) {
            println!(
                "{} API server already running (PID: {})",
                "!".yellow().bold(),
                pid
            );
            return Ok(());
        }
        // Stale PID file, remove it
        remove_pid("server")?;
    }

    println!(
        "{} Starting API server on port {}...",
        "→".blue().bold(),
        args.port
    );

    let log_file = get_log_file("server");
    if let Some(parent) = log_file.parent() {
        fs::create_dir_all(parent)?;
    }

    // Try to find axonml-server binary
    let server_bin = find_server_binary()?;

    let log_handle = fs::File::create(&log_file)?;
    let err_handle = log_handle.try_clone()?;

    let mut cmd = Command::new(&server_bin);
    cmd.args(["--host", &args.host, "--port", &args.port.to_string()]);

    if let Some(ref config) = args.config {
        cmd.args(["--config", config]);
    }

    if args.foreground {
        // Run in foreground
        let status = cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()?;

        if !status.success() {
            return Err(CliError::Other(format!(
                "Server exited with status: {:?}",
                status.code()
            )));
        }
    } else {
        // Run in background
        let child = cmd
            .stdout(Stdio::from(log_handle))
            .stderr(Stdio::from(err_handle))
            .spawn()?;

        write_pid("server", child.id())?;
        println!(
            "{} API server started (PID: {})",
            "✓".green().bold(),
            child.id()
        );
    }

    Ok(())
}

fn start_dashboard_process(args: &StartArgs) -> CliResult<()> {
    // Check if already running
    if let Some(pid) = read_pid("dashboard") {
        if is_process_running(pid) {
            println!(
                "{} Dashboard already running (PID: {})",
                "!".yellow().bold(),
                pid
            );
            return Ok(());
        }
        // Stale PID file, remove it
        remove_pid("dashboard")?;
    }

    println!(
        "{} Starting dashboard on port {}...",
        "→".blue().bold(),
        args.dashboard_port
    );

    let log_file = get_log_file("dashboard");
    if let Some(parent) = log_file.parent() {
        fs::create_dir_all(parent)?;
    }

    // Check if we have a pre-built dashboard or need to use trunk
    let data_dir = get_data_dir();
    let dashboard_dist = data_dir.join("dashboard");

    if dashboard_dist.exists() && dashboard_dist.join("index.html").exists() {
        // Serve pre-built dashboard using a simple HTTP server
        start_static_server(&dashboard_dist, args)?;
    } else {
        // Try to use trunk for development
        start_trunk_server(args)?;
    }

    Ok(())
}

fn start_static_server(dist_path: &PathBuf, args: &StartArgs) -> CliResult<()> {
    let log_file = get_log_file("dashboard");
    let log_handle = fs::File::create(&log_file)?;
    let err_handle = log_handle.try_clone()?;

    // Use Python's http.server or a Rust-based server
    // First try python
    let child = if Command::new("python3").arg("--version").output().is_ok() {
        Command::new("python3")
            .args([
                "-m",
                "http.server",
                &args.dashboard_port.to_string(),
                "--bind",
                &args.host,
            ])
            .current_dir(dist_path)
            .stdout(Stdio::from(log_handle))
            .stderr(Stdio::from(err_handle))
            .spawn()?
    } else if Command::new("python").arg("--version").output().is_ok() {
        Command::new("python")
            .args([
                "-m",
                "http.server",
                &args.dashboard_port.to_string(),
                "--bind",
                &args.host,
            ])
            .current_dir(dist_path)
            .stdout(Stdio::from(log_handle))
            .stderr(Stdio::from(err_handle))
            .spawn()?
    } else {
        // Fallback: try npx serve
        Command::new("npx")
            .args(["serve", "-s", "-l", &args.dashboard_port.to_string(), "-n"])
            .current_dir(dist_path)
            .stdout(Stdio::from(log_handle))
            .stderr(Stdio::from(err_handle))
            .spawn()
            .map_err(|_| {
                CliError::Other(
                    "No static file server found. Install Python or run: npm install -g serve"
                        .to_string(),
                )
            })?
    };

    write_pid("dashboard", child.id())?;
    println!(
        "{} Dashboard started (PID: {})",
        "✓".green().bold(),
        child.id()
    );

    Ok(())
}

fn start_trunk_server(args: &StartArgs) -> CliResult<()> {
    // Find the dashboard crate directory
    let dashboard_dir = find_dashboard_dir()?;

    let log_file = get_log_file("dashboard");
    let log_handle = fs::File::create(&log_file)?;
    let err_handle = log_handle.try_clone()?;

    let proxy_backend = format!("http://{}:{}/api", args.host, args.port);

    let child = Command::new("trunk")
        .args([
            "serve",
            "--port",
            &args.dashboard_port.to_string(),
            "--address",
            &args.host,
            &format!("--proxy-backend={}", proxy_backend),
        ])
        .current_dir(&dashboard_dir)
        .stdout(Stdio::from(log_handle))
        .stderr(Stdio::from(err_handle))
        .spawn()
        .map_err(|_| {
            CliError::Other("trunk not found. Install with: cargo install trunk".to_string())
        })?;

    write_pid("dashboard", child.id())?;
    println!(
        "{} Dashboard started via trunk (PID: {})",
        "✓".green().bold(),
        child.id()
    );

    Ok(())
}

fn find_server_binary() -> CliResult<PathBuf> {
    // Check in PATH first
    if let Ok(output) = Command::new("which").arg("axonml-server").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Ok(PathBuf::from(path));
        }
    }

    // Check common install locations
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let locations = [
        home.join(".local/bin/axonml-server"),
        home.join(".cargo/bin/axonml-server"),
        PathBuf::from("/usr/local/bin/axonml-server"),
        PathBuf::from("./target/release/axonml-server"),
        PathBuf::from("./target/debug/axonml-server"),
    ];

    for loc in &locations {
        if loc.exists() {
            return Ok(loc.clone());
        }
    }

    Err(CliError::Other(
        "axonml-server not found. Build with: cargo build -p axonml-server --release".to_string(),
    ))
}

fn find_dashboard_dir() -> CliResult<PathBuf> {
    // Check current directory structure
    let locations = [
        PathBuf::from("./crates/axonml-dashboard"),
        PathBuf::from("../axonml-dashboard"),
        PathBuf::from("../../crates/axonml-dashboard"),
    ];

    for loc in &locations {
        if loc.join("Trunk.toml").exists() || loc.join("index.html").exists() {
            return Ok(loc.clone());
        }
    }

    // Check AXONML_HOME
    let data_dir = get_data_dir();
    let dashboard_src = data_dir.join("dashboard-src");
    if dashboard_src.join("Trunk.toml").exists() {
        return Ok(dashboard_src);
    }

    Err(CliError::Other(
        "Dashboard source not found. Run from the AxonML repository root or set AXONML_HOME."
            .to_string(),
    ))
}

// =============================================================================
// Stop Command
// =============================================================================

/// Execute the stop command
pub fn execute_stop(args: StopArgs) -> CliResult<()> {
    let stop_server = !args.dashboard;
    let stop_dashboard = !args.server;

    let mut stopped_any = false;

    if stop_server {
        if let Some(pid) = read_pid("server") {
            if is_process_running(pid) {
                println!(
                    "{} Stopping API server (PID: {})...",
                    "→".blue().bold(),
                    pid
                );
                if stop_process(pid, args.force) {
                    // Wait a moment for graceful shutdown
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    println!("{} API server stopped", "✓".green().bold());
                    stopped_any = true;
                } else {
                    println!("{} Failed to stop API server", "✗".red().bold());
                }
            }
            remove_pid("server")?;
        } else {
            println!("{} API server is not running", "!".yellow().bold());
        }
    }

    if stop_dashboard {
        if let Some(pid) = read_pid("dashboard") {
            if is_process_running(pid) {
                println!("{} Stopping dashboard (PID: {})...", "→".blue().bold(), pid);
                if stop_process(pid, args.force) {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    println!("{} Dashboard stopped", "✓".green().bold());
                    stopped_any = true;
                } else {
                    println!("{} Failed to stop dashboard", "✗".red().bold());
                }
            }
            remove_pid("dashboard")?;
        } else {
            println!("{} Dashboard is not running", "!".yellow().bold());
        }
    }

    if stopped_any {
        println!();
        println!("All services stopped.");
    }

    Ok(())
}

// =============================================================================
// Status Command
// =============================================================================

/// Execute the status command
pub fn execute_status(args: StatusArgs) -> CliResult<()> {
    let server_pid = read_pid("server");
    let dashboard_pid = read_pid("dashboard");

    let server_running = server_pid.map(is_process_running).unwrap_or(false);
    let dashboard_running = dashboard_pid.map(is_process_running).unwrap_or(false);

    if args.format == "json" {
        let status = serde_json::json!({
            "server": {
                "running": server_running,
                "pid": server_pid
            },
            "dashboard": {
                "running": dashboard_running,
                "pid": dashboard_pid
            }
        });
        println!("{}", serde_json::to_string_pretty(&status).unwrap());
    } else {
        println!();
        println!("{}", "AxonML Service Status".cyan().bold());
        println!("{}", "═".repeat(40).cyan());
        println!();

        // Server status
        if server_running {
            println!(
                "{} API Server:  {} (PID: {})",
                "●".green().bold(),
                "Running".green(),
                server_pid.unwrap()
            );
        } else {
            println!("{} API Server:  {}", "●".red().bold(), "Stopped".red());
        }

        // Dashboard status
        if dashboard_running {
            println!(
                "{} Dashboard:   {} (PID: {})",
                "●".green().bold(),
                "Running".green(),
                dashboard_pid.unwrap()
            );
        } else {
            println!("{} Dashboard:   {}", "●".red().bold(), "Stopped".red());
        }

        println!();

        if args.detailed {
            let data_dir = get_data_dir();
            println!("{}", "Configuration".cyan().bold());
            println!("{}", "─".repeat(40).cyan());
            println!("Data directory: {}", data_dir.display());
            println!("Server log:     {}", get_log_file("server").display());
            println!("Dashboard log:  {}", get_log_file("dashboard").display());
            println!();
        }
    }

    Ok(())
}

// =============================================================================
// Logs Command
// =============================================================================

/// Execute the logs command
pub fn execute_logs(args: LogsArgs) -> CliResult<()> {
    let show_server = !args.dashboard;
    let show_dashboard = !args.server;

    if show_server {
        let log_file = get_log_file("server");
        if log_file.exists() {
            if args.follow {
                follow_log(&log_file, "server", args.level.as_deref())?;
            } else {
                show_log(&log_file, "server", args.lines, args.level.as_deref())?;
            }
        } else {
            println!("{} No server logs found", "!".yellow().bold());
        }
    }

    if show_dashboard {
        let log_file = get_log_file("dashboard");
        if log_file.exists() {
            if args.follow {
                follow_log(&log_file, "dashboard", args.level.as_deref())?;
            } else {
                show_log(&log_file, "dashboard", args.lines, args.level.as_deref())?;
            }
        } else {
            println!("{} No dashboard logs found", "!".yellow().bold());
        }
    }

    Ok(())
}

fn show_log(
    log_file: &PathBuf,
    name: &str,
    lines: usize,
    level_filter: Option<&str>,
) -> CliResult<()> {
    println!(
        "{} {} logs (last {} lines)",
        "━".cyan(),
        name.cyan().bold(),
        lines
    );
    println!();

    let file = fs::File::open(log_file)?;
    let reader = BufReader::new(file);

    let all_lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
    let start = all_lines.len().saturating_sub(lines);

    for line in &all_lines[start..] {
        if let Some(level) = level_filter {
            if !line.to_lowercase().contains(level) {
                continue;
            }
        }
        print_log_line(line);
    }

    println!();
    Ok(())
}

fn follow_log(log_file: &PathBuf, name: &str, level_filter: Option<&str>) -> CliResult<()> {
    use std::io::Seek;

    println!(
        "{} Following {} logs (Ctrl+C to stop)",
        "━".cyan(),
        name.cyan().bold()
    );
    println!();

    let mut file = fs::File::open(log_file)?;
    // Seek to end
    file.seek(std::io::SeekFrom::End(0))?;
    let mut reader = BufReader::new(file);

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // No new data, wait a bit
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Ok(_) => {
                if let Some(level) = level_filter {
                    if !line.to_lowercase().contains(level) {
                        continue;
                    }
                }
                print_log_line(&line);
            }
            Err(e) => {
                eprintln!("Error reading log: {}", e);
                return Err(CliError::Other(format!("Error reading log: {}", e)));
            }
        }
    }
}

fn print_log_line(line: &str) {
    // Simple log colorization
    if line.contains("ERROR") || line.contains("error") {
        println!("{}", line.red());
    } else if line.contains("WARN") || line.contains("warn") {
        println!("{}", line.yellow());
    } else if line.contains("INFO") || line.contains("info") {
        println!("{}", line);
    } else if line.contains("DEBUG") || line.contains("debug") {
        println!("{}", line.dimmed());
    } else {
        println!("{}", line);
    }
}
