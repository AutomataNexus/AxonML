//! Terminal WebSocket API
//!
//! Provides a PTY-based terminal interface accessible from the webapp.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::StatusCode,
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde::Deserialize;
use std::io::{Read, Write};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::api::AppState;

/// Query params for WebSocket auth
#[derive(Debug, Deserialize)]
pub struct WsAuthQuery {
    pub token: Option<String>,
}

/// WebSocket handler for terminal
/// Authenticates via query param since WebSocket can't use Authorization header
pub async fn terminal_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<WsAuthQuery>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Validate the token from query params
    let token = query.token.ok_or((
        StatusCode::UNAUTHORIZED,
        "Missing token parameter".to_string(),
    ))?;

    // Verify the JWT token
    let _claims = state
        .jwt
        .validate_access_token(&token)
        .map_err(|e| (StatusCode::UNAUTHORIZED, format!("Invalid token: {}", e)))?;

    Ok(ws.on_upgrade(handle_terminal))
}

/// Handle terminal WebSocket connection with proper PTY
async fn handle_terminal(socket: WebSocket) {
    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Get the PTY system
    let pty_system = native_pty_system();

    // Create a new PTY pair with initial size
    let pair = match pty_system.openpty(PtySize {
        rows: 24,
        cols: 80,
        pixel_width: 0,
        pixel_height: 0,
    }) {
        Ok(pair) => pair,
        Err(e) => {
            error!("Failed to create PTY: {}", e);
            let _ = ws_sender
                .send(Message::Text(format!(
                    "\r\nError: Failed to create PTY: {}\r\n",
                    e
                )))
                .await;
            return;
        }
    };

    // Determine shell
    let shell = if cfg!(target_os = "windows") {
        "powershell.exe".to_string()
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string())
    };

    info!("Starting PTY terminal session with shell: {}", shell);

    // Build command
    let mut cmd = CommandBuilder::new(&shell);
    cmd.env("TERM", "xterm-256color");

    // Spawn the shell in the PTY
    let mut child = match pair.slave.spawn_command(cmd) {
        Ok(child) => child,
        Err(e) => {
            error!("Failed to spawn shell: {}", e);
            let _ = ws_sender
                .send(Message::Text(format!(
                    "\r\nError: Failed to spawn shell: {}\r\n",
                    e
                )))
                .await;
            return;
        }
    };

    // Get reader for the PTY master
    let mut reader = match pair.master.try_clone_reader() {
        Ok(r) => r,
        Err(e) => {
            error!("Failed to get PTY reader: {}", e);
            return;
        }
    };

    // Get writer for the PTY master
    let pty_writer: Box<dyn Write + Send> = match pair.master.take_writer() {
        Ok(w) => w,
        Err(e) => {
            error!("Failed to get PTY writer: {}", e);
            return;
        }
    };

    // Keep master for resize operations, writer for data
    let master = Arc::new(std::sync::Mutex::new(pair.master));
    let writer = Arc::new(std::sync::Mutex::new(pty_writer));

    // Channel for PTY output -> WebSocket
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(256);

    // Task to read from PTY and send to channel
    let read_handle = std::thread::spawn(move || {
        let mut buffer = [0u8; 4096];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => {
                    // EOF
                    break;
                }
                Ok(n) => {
                    if tx.blocking_send(buffer[..n].to_vec()).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    // Check if it's just the PTY closing
                    if e.kind() != std::io::ErrorKind::Other {
                        warn!("PTY read error: {}", e);
                    }
                    break;
                }
            }
        }
    });

    // Task to send PTY output to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(data) = rx.recv().await {
            let text = String::from_utf8_lossy(&data).to_string();
            if ws_sender.send(Message::Text(text)).await.is_err() {
                break;
            }
        }
    });

    // Task to receive from WebSocket and write to PTY
    let master_clone = master.clone();
    let writer_clone = writer.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // Handle resize command
                    if text.starts_with("\x1b[8;") {
                        // Parse resize: ESC[8;rows;colst
                        if let Some(size) = parse_resize_sequence(&text) {
                            if let Ok(master) = master_clone.lock() {
                                let _ = master.resize(size);
                            }
                            continue;
                        }
                    }

                    // Write to PTY
                    if let Ok(mut pty_writer) = writer_clone.lock() {
                        if pty_writer.write_all(text.as_bytes()).is_err() {
                            break;
                        }
                    }
                }
                Message::Binary(data) => {
                    if let Ok(mut pty_writer) = writer_clone.lock() {
                        if pty_writer.write_all(&data).is_err() {
                            break;
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    // Wait for tasks or child process to end
    tokio::select! {
        _ = send_task => {
            info!("Send task ended");
        }
        _ = recv_task => {
            info!("Recv task ended");
        }
        _ = tokio::task::spawn_blocking(move || {
            let _ = child.wait();
        }) => {
            info!("Shell process ended");
        }
    }

    // Clean up the reader thread
    drop(writer);
    drop(master);
    let _ = read_handle.join();

    info!("Terminal session ended");
}

/// Parse terminal resize sequence ESC[8;rows;colst
fn parse_resize_sequence(s: &str) -> Option<PtySize> {
    // Format: \x1b[8;ROWS;COLSt
    if !s.starts_with("\x1b[8;") || !s.ends_with('t') {
        return None;
    }

    let inner = &s[4..s.len() - 1]; // Remove prefix and suffix
    let parts: Vec<&str> = inner.split(';').collect();

    if parts.len() != 2 {
        return None;
    }

    let rows: u16 = parts[0].parse().ok()?;
    let cols: u16 = parts[1].parse().ok()?;

    Some(PtySize {
        rows,
        cols,
        pixel_width: 0,
        pixel_height: 0,
    })
}

/// Get terminal info
pub async fn terminal_info(
    State(state): State<AppState>,
    Query(query): Query<WsAuthQuery>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Validate the token from query params
    let token = query.token.ok_or((
        StatusCode::UNAUTHORIZED,
        "Missing token parameter".to_string(),
    ))?;

    // Verify the JWT token
    let _claims = state
        .jwt
        .validate_access_token(&token)
        .map_err(|e| (StatusCode::UNAUTHORIZED, format!("Invalid token: {}", e)))?;

    let shell = if cfg!(target_os = "windows") {
        "powershell.exe".to_string()
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string())
    };

    Ok(axum::Json(serde_json::json!({
        "available": true,
        "shell": shell,
        "features": ["pty", "resize", "colors"],
    })))
}
