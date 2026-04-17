//! Training Lifecycle Controls — Pause / Resume / Stop / Checkpoint / Monitor
//!
//! Shared subsystem adopted by every `train_*.rs` binary. The hard rule is
//! that weeks-long runs must never lose progress to Ctrl+C or an orphaned
//! shell, so this module wires the training loop into signals, a control
//! socket, and an always-on browser monitor with rotating checkpoint saves.
//!
//! # What it provides
//!
//! - **Signal handlers** — `SIGINT` / `SIGTERM` flush a final checkpoint then
//!   exit cleanly; `SIGUSR1` pauses; `SIGUSR2` resumes. A dedicated
//!   `axonml-signal-dispatch` thread polls the raw `signal_hook` flags and
//!   translates them into the [`ControlFlags`] used by [`TrainingLifecycle`].
//! - **Unix control socket** at `/tmp/axonml-train-<pid>.sock` with a
//!   convenience symlink at `/tmp/axonml-train-latest.sock`. Commands
//!   (plaintext, one per line): `pause`, `resume`, `stop`, `checkpoint`,
//!   `status` (returns a JSON status blob with epoch, global step, last
//!   loss, monitor URL, param count, etc.).
//! - **Step-level checkpoint rotation** — [`TrainingLifecycle::save_step`]
//!   writes every N steps and prunes older files beyond `keep_last_k`.
//! - **End-of-epoch + best-model saves** — [`TrainingLifecycle::save_epoch`]
//!   writes `checkpoint_latest.axonml` plus a numbered
//!   `checkpoint_epoch_NNNN.axonml`; [`TrainingLifecycle::save_if_best`]
//!   writes `best_model.axonml` + `checkpoint_best.axonml` when a new best
//!   metric is observed.
//! - **Always-on training monitor** — `axonml::TrainingMonitor` is launched
//!   automatically at `127.0.0.1:<auto>` with no opt-out flag.
//! - **Graceful final save** — [`TrainingLifecycle::save_final`] flushes
//!   `checkpoint_final.axonml` on stop before exit.
//! - **[`LoopAction`]** — enum returned by
//!   [`TrainingLifecycle::poll`] telling the loop to `Continue`,
//!   `CheckpointNow`, or `Stop`.
//!
//! # Usage in a training binary
//!
//! ```ignore
//! let lifecycle = TrainingLifecycle::builder()
//!     .model_name("Trident-Coder-1B")
//!     .output_dir(&cfg.output_dir)
//!     .param_count(param_count)
//!     .total_epochs(cfg.epochs)
//!     .batch_size(cfg.batch_size)
//!     .checkpoint_every_steps(500)
//!     .keep_last_k(5)
//!     .start();
//!
//! 'outer: for epoch in (start_epoch + 1)..=cfg.epochs {
//!     for step in 1..=cfg.steps_per_epoch {
//!         match lifecycle.poll() {
//!             LoopAction::Stop => {
//!                 lifecycle.save_final(&model, &optimizer, &training_state, epoch);
//!                 break 'outer;
//!             }
//!             LoopAction::CheckpointNow => {
//!                 lifecycle.save_step(&model, &optimizer, &training_state, epoch, global_step);
//!             }
//!             LoopAction::Continue => {}
//!         }
//!         // ... training step ...
//!         lifecycle.tick(global_step, loss_val);
//!         if lifecycle.should_step_checkpoint(global_step) {
//!             lifecycle.save_step(&model, &optimizer, &training_state, epoch, global_step);
//!         }
//!     }
//!     lifecycle.save_epoch(&model, &optimizer, &training_state, epoch, epoch_avg_loss);
//! }
//! lifecycle.finish();
//! ```
//!
//! # File
//! `llm-training/src/lifecycle.rs`
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

// =============================================================================
// Imports
// =============================================================================

use std::fs;
use std::io::{BufRead, BufReader, Write as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axonml_nn::Module;
use axonml_serialize::{save_checkpoint, save_model, Checkpoint, StateDict, TrainingState};

// =============================================================================
// Shared control flags
// =============================================================================

/// Atomic flags shared between the main training thread, signal handlers,
/// and the Unix-socket controller thread.
///
/// All writers (signal handlers, socket thread) are async-signal-safe on
/// Linux: `AtomicBool::store` compiles to a single `mov` with memory barrier.
#[derive(Default)]
struct ControlFlags {
    /// Set by SIGUSR1 or socket `pause`. Poll loop spins until cleared.
    paused: AtomicBool,
    /// Set by SIGINT/SIGTERM or socket `stop`. Poll returns `LoopAction::Stop`.
    should_stop: AtomicBool,
    /// Set by socket `checkpoint`. Consumed by poll — returns
    /// `LoopAction::CheckpointNow` once, then clears.
    checkpoint_requested: AtomicBool,

    // -- telemetry readable via `status` --
    global_step: AtomicU64,
    current_epoch: AtomicUsize,
    /// Last observed loss, stored as f32 bits for atomic read/write.
    last_loss_bits: AtomicU32,
}

impl ControlFlags {
    fn current_loss(&self) -> f32 {
        f32::from_bits(self.last_loss_bits.load(Ordering::Relaxed))
    }
    fn set_loss(&self, loss: f32) {
        self.last_loss_bits.store(loss.to_bits(), Ordering::Relaxed);
    }
}

// =============================================================================
// Signal handler installation
// =============================================================================

fn install_signals(flags: Arc<ControlFlags>) -> std::io::Result<()> {
    use signal_hook::consts::{SIGINT, SIGTERM, SIGUSR1, SIGUSR2};

    let stop = flags.clone();
    signal_hook::flag::register(SIGINT, Arc::new(AtomicBool::new(false)))?;
    signal_hook::flag::register(SIGTERM, Arc::new(AtomicBool::new(false)))?;
    signal_hook::flag::register(SIGUSR1, Arc::new(AtomicBool::new(false)))?;
    signal_hook::flag::register(SIGUSR2, Arc::new(AtomicBool::new(false)))?;

    // Thread that polls the raw signal flags and translates into our
    // control flags. Using a dispatcher thread avoids doing any non-
    // async-signal-safe work inside the handler itself.
    let stop_flag = Arc::new(AtomicBool::new(false));
    let pause_flag = Arc::new(AtomicBool::new(false));
    let resume_flag = Arc::new(AtomicBool::new(false));

    signal_hook::flag::register(SIGINT, stop_flag.clone())?;
    signal_hook::flag::register(SIGTERM, stop_flag.clone())?;
    signal_hook::flag::register(SIGUSR1, pause_flag.clone())?;
    signal_hook::flag::register(SIGUSR2, resume_flag.clone())?;

    thread::Builder::new()
        .name("axonml-signal-dispatch".into())
        .spawn(move || loop {
            if stop_flag.swap(false, Ordering::SeqCst) {
                eprintln!("\n[lifecycle] stop signal received — finishing current step, then flushing final checkpoint");
                stop.should_stop.store(true, Ordering::SeqCst);
                stop.paused.store(false, Ordering::SeqCst);
            }
            if pause_flag.swap(false, Ordering::SeqCst) {
                eprintln!("[lifecycle] SIGUSR1 — pausing after current step");
                stop.paused.store(true, Ordering::SeqCst);
            }
            if resume_flag.swap(false, Ordering::SeqCst) {
                eprintln!("[lifecycle] SIGUSR2 — resuming");
                stop.paused.store(false, Ordering::SeqCst);
            }
            thread::sleep(Duration::from_millis(100));
        })?;
    Ok(())
}

// =============================================================================
// Unix control socket
// =============================================================================

const LATEST_SOCKET: &str = "/tmp/axonml-train-latest.sock";

fn socket_path() -> PathBuf {
    PathBuf::from(format!("/tmp/axonml-train-{}.sock", std::process::id()))
}

fn start_control_socket(flags: Arc<ControlFlags>, meta: SocketMetadata) -> std::io::Result<PathBuf> {
    let path = socket_path();
    // Stale socket from a crashed prior run: remove it.
    let _ = fs::remove_file(&path);
    let listener = UnixListener::bind(&path)?;
    eprintln!("[lifecycle] control socket: {}", path.display());

    // Convenience "latest" symlink so operators don't have to look up the PID.
    let _ = fs::remove_file(LATEST_SOCKET);
    let _ = std::os::unix::fs::symlink(&path, LATEST_SOCKET);

    let meta = Arc::new(meta);
    thread::Builder::new()
        .name("axonml-control-socket".into())
        .spawn(move || {
            for conn in listener.incoming() {
                let Ok(stream) = conn else { continue };
                let flags = flags.clone();
                let meta = meta.clone();
                thread::spawn(move || handle_connection(stream, flags, meta));
            }
        })?;
    Ok(path)
}

struct SocketMetadata {
    model_name: String,
    output_dir: PathBuf,
    started_at: u64,
    param_count: usize,
    total_epochs: usize,
    batch_size: usize,
    monitor_port: u16,
}

fn handle_connection(stream: UnixStream, flags: Arc<ControlFlags>, meta: Arc<SocketMetadata>) {
    let Ok(peer) = stream.try_clone() else { return };
    let reader = BufReader::new(stream);
    let mut writer = peer;

    for line in reader.lines() {
        let Ok(line) = line else { return };
        let cmd = line.trim();
        match cmd {
            "pause" => {
                flags.paused.store(true, Ordering::SeqCst);
                let _ = writeln!(writer, "ok: pausing after current step");
            }
            "resume" => {
                flags.paused.store(false, Ordering::SeqCst);
                let _ = writeln!(writer, "ok: resuming");
            }
            "stop" => {
                flags.should_stop.store(true, Ordering::SeqCst);
                flags.paused.store(false, Ordering::SeqCst);
                let _ = writeln!(writer, "ok: stopping after current step (final checkpoint will flush)");
            }
            "checkpoint" | "checkpoint-now" => {
                flags.checkpoint_requested.store(true, Ordering::SeqCst);
                let _ = writeln!(writer, "ok: checkpoint queued (will flush on next poll)");
            }
            "status" => {
                let now_secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let uptime = now_secs.saturating_sub(meta.started_at);
                let json = format!(
                    "{{\"model\":\"{}\",\"pid\":{},\"output_dir\":\"{}\",\
                      \"epoch\":{},\"global_step\":{},\"last_loss\":{:.6},\
                      \"paused\":{},\"stopping\":{},\
                      \"started_at\":{},\"uptime_sec\":{},\
                      \"params\":{},\"total_epochs\":{},\"batch_size\":{},\
                      \"monitor\":\"http://127.0.0.1:{}\"}}",
                    meta.model_name,
                    std::process::id(),
                    meta.output_dir.display(),
                    flags.current_epoch.load(Ordering::Relaxed),
                    flags.global_step.load(Ordering::Relaxed),
                    flags.current_loss(),
                    flags.paused.load(Ordering::Relaxed),
                    flags.should_stop.load(Ordering::Relaxed),
                    meta.started_at,
                    uptime,
                    meta.param_count,
                    meta.total_epochs,
                    meta.batch_size,
                    meta.monitor_port,
                );
                let _ = writeln!(writer, "{json}");
            }
            "" => {}
            other => {
                let _ = writeln!(
                    writer,
                    "err: unknown command {other:?} (try: pause | resume | stop | checkpoint | status)"
                );
            }
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// What the training loop should do on the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopAction {
    /// Run another training step.
    Continue,
    /// Flush a checkpoint before the next step.
    CheckpointNow,
    /// Flush a final checkpoint and exit the outer loop.
    Stop,
}

/// Builder for a [`TrainingLifecycle`].
pub struct TrainingLifecycleBuilder {
    model_name: String,
    output_dir: PathBuf,
    param_count: usize,
    total_epochs: usize,
    batch_size: usize,
    checkpoint_every_steps: u64,
    keep_last_k: usize,
}

impl TrainingLifecycleBuilder {
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }
    pub fn output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_dir = path.into();
        self
    }
    pub fn param_count(mut self, n: usize) -> Self {
        self.param_count = n;
        self
    }
    pub fn total_epochs(mut self, n: usize) -> Self {
        self.total_epochs = n;
        self
    }
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }
    /// How often to write a rotating step-level checkpoint. `0` disables
    /// step-level checkpoints (epoch + best still run).
    pub fn checkpoint_every_steps(mut self, n: u64) -> Self {
        self.checkpoint_every_steps = n;
        self
    }
    /// How many rotating step-level checkpoints to keep on disk.
    pub fn keep_last_k(mut self, n: usize) -> Self {
        self.keep_last_k = n;
        self
    }

    /// Install signal handlers, start the control socket, launch the monitor.
    ///
    /// Panics if the output directory cannot be created — training cannot
    /// proceed without somewhere to write checkpoints, so this is fatal.
    pub fn start(self) -> TrainingLifecycle {
        fs::create_dir_all(&self.output_dir).expect("create checkpoint output dir");

        let flags = Arc::new(ControlFlags::default());
        if let Err(e) = install_signals(flags.clone()) {
            eprintln!("[lifecycle] WARN: could not install signal handlers: {e}");
        }

        // Monitor is always on — no opt-out flag.
        let monitor = axonml::TrainingMonitor::new(&self.model_name, self.param_count)
            .total_epochs(self.total_epochs)
            .batch_size(self.batch_size)
            .launch();
        let monitor_port = monitor.port();
        eprintln!(
            "[lifecycle] training monitor: http://127.0.0.1:{}",
            monitor_port
        );

        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let socket_meta = SocketMetadata {
            model_name: self.model_name.clone(),
            output_dir: self.output_dir.clone(),
            started_at,
            param_count: self.param_count,
            total_epochs: self.total_epochs,
            batch_size: self.batch_size,
            monitor_port,
        };
        let socket = match start_control_socket(flags.clone(), socket_meta) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("[lifecycle] WARN: could not start control socket: {e}");
                None
            }
        };

        TrainingLifecycle {
            flags,
            monitor,
            model_name: self.model_name,
            output_dir: self.output_dir,
            checkpoint_every_steps: self.checkpoint_every_steps,
            keep_last_k: self.keep_last_k,
            socket_path: socket,
            started_at: Instant::now(),
        }
    }
}

/// The lifecycle handle returned by the builder.
///
/// Owns the monitor, control-socket thread, and checkpoint rotation state.
/// Dropping it removes the socket file but leaves signal handlers in place
/// (they are per-process, not per-handle).
pub struct TrainingLifecycle {
    flags: Arc<ControlFlags>,
    monitor: axonml::TrainingMonitor,
    model_name: String,
    output_dir: PathBuf,
    checkpoint_every_steps: u64,
    keep_last_k: usize,
    socket_path: Option<PathBuf>,
    started_at: Instant,
}

impl TrainingLifecycle {
    pub fn builder() -> TrainingLifecycleBuilder {
        TrainingLifecycleBuilder {
            model_name: "unnamed".into(),
            output_dir: PathBuf::from("checkpoints"),
            param_count: 0,
            total_epochs: 0,
            batch_size: 1,
            checkpoint_every_steps: 0,
            keep_last_k: 5,
        }
    }

    /// Called at the top of every training step.
    ///
    /// - Blocks while paused (polling every 200 ms).
    /// - Returns `Stop` if a stop was requested — caller must flush a final
    ///   checkpoint and exit the outer loop.
    /// - Returns `CheckpointNow` if an ad-hoc checkpoint was queued via socket.
    /// - Returns `Continue` otherwise.
    pub fn poll(&self) -> LoopAction {
        // Block while paused. Stop-while-paused wakes immediately.
        while self.flags.paused.load(Ordering::SeqCst) {
            if self.flags.should_stop.load(Ordering::SeqCst) {
                return LoopAction::Stop;
            }
            thread::sleep(Duration::from_millis(200));
        }
        if self.flags.should_stop.load(Ordering::SeqCst) {
            return LoopAction::Stop;
        }
        if self
            .flags
            .checkpoint_requested
            .swap(false, Ordering::SeqCst)
        {
            return LoopAction::CheckpointNow;
        }
        LoopAction::Continue
    }

    /// Called after every step to update telemetry visible over the socket.
    pub fn tick(&self, global_step: u64, loss: f32) {
        self.flags.global_step.store(global_step, Ordering::Relaxed);
        self.flags.set_loss(loss);
    }

    /// Called at the start of each epoch to update telemetry.
    pub fn set_epoch(&self, epoch: usize) {
        self.flags.current_epoch.store(epoch, Ordering::Relaxed);
    }

    /// True when a step-level rotating checkpoint is due on this `global_step`.
    /// False when `checkpoint_every_steps = 0` (the feature is disabled).
    pub fn should_step_checkpoint(&self, global_step: u64) -> bool {
        self.checkpoint_every_steps > 0
            && global_step > 0
            && global_step % self.checkpoint_every_steps == 0
    }

    // ---- Checkpoint helpers ------------------------------------------------

    /// Save a rotating step-level checkpoint and prune older ones beyond
    /// `keep_last_k`. Non-fatal on error — prints a warning and returns.
    pub fn save_step<M: Module>(
        &self,
        model: &M,
        training_state: &TrainingState,
        epoch: usize,
    ) {
        let global_step = self.flags.global_step.load(Ordering::Relaxed);
        let path = self
            .output_dir
            .join(format!("checkpoint_step_{global_step:010}.axonml"));
        let cp = Checkpoint::builder()
            .model_state(StateDict::from_module(model))
            .training_state(training_state.clone())
            .epoch(epoch)
            .build();
        if let Err(e) = save_checkpoint(&cp, &path) {
            eprintln!("[lifecycle] WARN: step checkpoint save failed: {e}");
            return;
        }
        self.rotate_step_checkpoints();
    }

    /// Save the end-of-epoch "latest" + optional "epoch_NNNN" checkpoints.
    pub fn save_epoch<M: Module>(
        &self,
        model: &M,
        training_state: &TrainingState,
        epoch: usize,
    ) {
        let latest = self.output_dir.join("checkpoint_latest.axonml");
        let numbered = self
            .output_dir
            .join(format!("checkpoint_epoch_{epoch:04}.axonml"));
        let cp = Checkpoint::builder()
            .model_state(StateDict::from_module(model))
            .training_state(training_state.clone())
            .epoch(epoch)
            .build();
        if let Err(e) = save_checkpoint(&cp, &latest) {
            eprintln!("[lifecycle] WARN: latest checkpoint save failed: {e}");
        }
        if let Err(e) = save_checkpoint(&cp, &numbered) {
            eprintln!("[lifecycle] WARN: epoch checkpoint save failed: {e}");
        }
    }

    /// If `metric` beats the previous best, write `best_model.axonml` +
    /// `checkpoint_best.axonml`. Returns true when a new best was recorded.
    pub fn save_if_best<M: Module>(
        &self,
        model: &M,
        training_state: &TrainingState,
        epoch: usize,
        metric: f32,
        previous_best: f32,
    ) -> bool {
        if !(metric < previous_best) {
            return false;
        }
        let best_model = self.output_dir.join("best_model.axonml");
        let best_ckpt = self.output_dir.join("checkpoint_best.axonml");
        if let Err(e) = save_model(model, &best_model) {
            eprintln!("[lifecycle] WARN: best model save failed: {e}");
            return false;
        }
        let cp = Checkpoint::builder()
            .model_state(StateDict::from_module(model))
            .training_state(training_state.clone())
            .epoch(epoch)
            .build();
        if let Err(e) = save_checkpoint(&cp, &best_ckpt) {
            eprintln!("[lifecycle] WARN: best checkpoint save failed: {e}");
        }
        true
    }

    /// Flush a final checkpoint on graceful stop. This is the
    /// "don't-lose-a-week-of-training" path.
    pub fn save_final<M: Module>(
        &self,
        model: &M,
        training_state: &TrainingState,
        epoch: usize,
    ) {
        let path = self.output_dir.join("checkpoint_final.axonml");
        let cp = Checkpoint::builder()
            .model_state(StateDict::from_module(model))
            .training_state(training_state.clone())
            .epoch(epoch)
            .build();
        match save_checkpoint(&cp, &path) {
            Ok(_) => eprintln!(
                "[lifecycle] final checkpoint flushed → {} (epoch {epoch}, step {})",
                path.display(),
                self.flags.global_step.load(Ordering::Relaxed),
            ),
            Err(e) => eprintln!("[lifecycle] CRITICAL: final checkpoint save failed: {e}"),
        }
    }

    fn rotate_step_checkpoints(&self) {
        if self.keep_last_k == 0 {
            return;
        }
        let Ok(entries) = fs::read_dir(&self.output_dir) else {
            return;
        };
        let mut step_ckpts: Vec<(u64, PathBuf)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let rest = name.strip_prefix("checkpoint_step_")?;
                let step_str = rest.strip_suffix(".axonml")?;
                let step: u64 = step_str.parse().ok()?;
                Some((step, e.path()))
            })
            .collect();
        if step_ckpts.len() <= self.keep_last_k {
            return;
        }
        step_ckpts.sort_by_key(|(s, _)| *s);
        let to_delete = step_ckpts.len() - self.keep_last_k;
        for (_, path) in step_ckpts.into_iter().take(to_delete) {
            let _ = fs::remove_file(path);
        }
    }

    // ---- Monitor passthrough ----------------------------------------------

    /// Forward epoch metrics to the browser monitor.
    pub fn log_epoch(
        &self,
        epoch: usize,
        train_loss: f32,
        val_loss: Option<f32>,
        extras: Vec<(&str, f32)>,
    ) {
        self.monitor.log_epoch(epoch, train_loss, val_loss, extras);
    }

    /// Set monitor status (`"training"`, `"paused"`, `"complete"`, `"stopped"`).
    pub fn set_status(&self, status: &str) {
        self.monitor.set_status(status);
    }

    /// Stop the monitor, mark as complete, clean up the socket file.
    /// Safe to call multiple times.
    pub fn finish(&self) {
        self.monitor.set_status("complete");
        eprintln!(
            "[lifecycle] run complete — total wall time {:.1}s",
            self.started_at.elapsed().as_secs_f32(),
        );
        if let Some(path) = &self.socket_path {
            let _ = fs::remove_file(path);
        }
        let _ = fs::remove_file(LATEST_SOCKET);
    }

    // ---- Introspection ----------------------------------------------------

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn monitor_port(&self) -> u16 {
        self.monitor.port()
    }

    /// True when a stop has been requested — useful for loops that can't use
    /// `poll()` directly (e.g. validation sweeps inside a step).
    pub fn stopping(&self) -> bool {
        self.flags.should_stop.load(Ordering::Relaxed)
    }
}

impl Drop for TrainingLifecycle {
    fn drop(&mut self) {
        if let Some(path) = &self.socket_path {
            let _ = fs::remove_file(path);
        }
        let _ = fs::remove_file(LATEST_SOCKET);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_returns_continue_by_default() {
        let flags = Arc::new(ControlFlags::default());
        // Build a minimal lifecycle without going through start() so we don't
        // fight the OS (signals, ports). We only need to exercise poll() logic.
        let lc = TrainingLifecycle {
            flags: flags.clone(),
            monitor: axonml::TrainingMonitor::new("test", 0)
                .total_epochs(1)
                .batch_size(1)
                .launch(),
            model_name: "test".into(),
            output_dir: std::env::temp_dir().join("axonml-lifecycle-test"),
            checkpoint_every_steps: 100,
            keep_last_k: 3,
            socket_path: None,
            started_at: Instant::now(),
        };
        assert_eq!(lc.poll(), LoopAction::Continue);

        // Stop request → Stop
        flags.should_stop.store(true, Ordering::SeqCst);
        assert_eq!(lc.poll(), LoopAction::Stop);
        flags.should_stop.store(false, Ordering::SeqCst);

        // Checkpoint request → CheckpointNow once, then clears
        flags.checkpoint_requested.store(true, Ordering::SeqCst);
        assert_eq!(lc.poll(), LoopAction::CheckpointNow);
        assert_eq!(lc.poll(), LoopAction::Continue);
    }

    #[test]
    fn step_checkpoint_gate() {
        let flags = Arc::new(ControlFlags::default());
        let lc = TrainingLifecycle {
            flags,
            monitor: axonml::TrainingMonitor::new("test", 0)
                .total_epochs(1)
                .batch_size(1)
                .launch(),
            model_name: "test".into(),
            output_dir: std::env::temp_dir().join("axonml-lifecycle-gate-test"),
            checkpoint_every_steps: 500,
            keep_last_k: 3,
            socket_path: None,
            started_at: Instant::now(),
        };
        assert!(!lc.should_step_checkpoint(0));
        assert!(!lc.should_step_checkpoint(499));
        assert!(lc.should_step_checkpoint(500));
        assert!(lc.should_step_checkpoint(1000));
        assert!(!lc.should_step_checkpoint(501));
    }

    #[test]
    fn zero_disables_step_checkpoints() {
        let flags = Arc::new(ControlFlags::default());
        let lc = TrainingLifecycle {
            flags,
            monitor: axonml::TrainingMonitor::new("test", 0)
                .total_epochs(1)
                .batch_size(1)
                .launch(),
            model_name: "test".into(),
            output_dir: std::env::temp_dir().join("axonml-lifecycle-zero-test"),
            checkpoint_every_steps: 0,
            keep_last_k: 3,
            socket_path: None,
            started_at: Instant::now(),
        };
        assert!(!lc.should_step_checkpoint(500));
        assert!(!lc.should_step_checkpoint(100_000));
    }
}
