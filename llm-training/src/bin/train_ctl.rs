//! `train_ctl` — operator CLI for any running llm-training binary.
//!
//! Every `train_*` binary that uses [`llm_training::TrainingLifecycle`]
//! (all of them, once Phase 0 is propagated) binds a Unix control socket at
//! `/tmp/axonml-train-<pid>.sock` with a convenience symlink at
//! `/tmp/axonml-train-latest.sock`. This tool is a thin client over that
//! socket so you don't have to `nc -U` by hand.
//!
//! Usage:
//!   train_ctl                  # same as `status`
//!   train_ctl status           # show JSON status of latest run
//!   train_ctl pause            # pause after current step
//!   train_ctl resume           # resume
//!   train_ctl stop             # graceful stop with final checkpoint
//!   train_ctl checkpoint       # flush an ad-hoc checkpoint now
//!   train_ctl list             # list every running training job on this box
//!   train_ctl --socket PATH <cmd>   # target a specific socket (multi-run)
//!   train_ctl --pid PID <cmd>       # target /tmp/axonml-train-<PID>.sock

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

const LATEST: &str = "/tmp/axonml-train-latest.sock";
const TMP_PREFIX: &str = "axonml-train-";
const TMP_SUFFIX: &str = ".sock";

fn print_help() -> ! {
    println!(
        "train_ctl — operator CLI for running AxonML training jobs

USAGE:
  train_ctl [OPTIONS] [COMMAND]

COMMANDS:
  status         Dump JSON status of the run (default if no command given)
  pause          Pause after the current step
  resume         Resume a paused run
  stop           Graceful stop + flush final checkpoint, then exit
  checkpoint     Flush an ad-hoc checkpoint on the next poll
  list           List every `/tmp/axonml-train-*.sock` on this box with status

OPTIONS:
  --socket PATH  Target a specific socket path
  --pid N        Target /tmp/axonml-train-<N>.sock
  --help, -h     Show this help

If no target is given, uses /tmp/axonml-train-latest.sock."
    );
    std::process::exit(0);
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut socket_override: Option<PathBuf> = None;
    let mut command: Option<String> = None;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--socket" => {
                i += 1;
                socket_override = Some(PathBuf::from(&argv[i]));
            }
            "--pid" => {
                i += 1;
                socket_override =
                    Some(PathBuf::from(format!("/tmp/axonml-train-{}.sock", &argv[i])));
            }
            "--help" | "-h" => print_help(),
            "list" => {
                list_jobs();
                return;
            }
            other => {
                if command.is_none() {
                    command = Some(other.to_string());
                } else {
                    eprintln!("error: unexpected positional argument {other:?}");
                    print_help();
                }
            }
        }
        i += 1;
    }

    let cmd = command.unwrap_or_else(|| "status".to_string());
    let path = socket_override.unwrap_or_else(|| PathBuf::from(LATEST));

    if !path.exists() {
        eprintln!(
            "error: no training socket at {} — is a training binary running?\n\
             hint: try `train_ctl list` to discover active jobs.",
            path.display(),
        );
        std::process::exit(2);
    }

    match send_command(&path, &cmd) {
        Ok(reply) => {
            print!("{reply}");
            if !reply.ends_with('\n') {
                println!();
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn send_command(path: &Path, cmd: &str) -> std::io::Result<String> {
    let mut stream = UnixStream::connect(path)?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    writeln!(stream, "{cmd}")?;
    stream.flush()?;

    // Read single line reply.
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(line)
}

fn list_jobs() {
    let dir = Path::new("/tmp");
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("error: cannot read /tmp");
        std::process::exit(1);
    };

    let mut rows: Vec<(u32, PathBuf, String)> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().into_string().unwrap_or_default();
        if !name.starts_with(TMP_PREFIX) || !name.ends_with(TMP_SUFFIX) {
            continue;
        }
        if name == format!("{TMP_PREFIX}latest{TMP_SUFFIX}") {
            continue;
        }
        let pid_str = &name[TMP_PREFIX.len()..name.len() - TMP_SUFFIX.len()];
        let Ok(pid) = pid_str.parse::<u32>() else {
            continue;
        };
        let path = entry.path();
        let status = match send_status(&path) {
            Ok(s) => s,
            Err(e) => format!("(socket present but unresponsive: {e})"),
        };
        rows.push((pid, path, status));
    }

    if rows.is_empty() {
        println!("No running training jobs found.");
        return;
    }

    rows.sort_by_key(|r| r.0);
    for (pid, path, status) in rows {
        println!("=== PID {pid} — {}", path.display());
        println!("{}", status.trim_end());
        println!();
    }
}

fn send_status(path: &Path) -> std::io::Result<String> {
    send_command(path, "status")
}
