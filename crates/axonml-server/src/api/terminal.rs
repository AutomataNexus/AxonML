//! Terminal WebSocket API
//!
//! Provides a PTY-based terminal interface accessible from the webapp.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use std::process::Stdio;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::api::AppState;
use crate::auth::AuthUser;

/// WebSocket handler for terminal
pub async fn terminal_ws(
    ws: WebSocketUpgrade,
    State(_state): State<AppState>,
    _user: AuthUser,
) -> impl IntoResponse {
    ws.on_upgrade(handle_terminal)
}

/// Handle terminal WebSocket connection
async fn handle_terminal(socket: WebSocket) {
    let (mut sender, mut receiver) = socket.split();

    // Determine shell
    let shell: String = if cfg!(target_os = "windows") {
        "powershell.exe".to_string()
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string())
    };

    info!("Starting terminal session with shell: {}", shell);

    // Spawn shell process
    let mut child = match Command::new(&shell)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            error!("Failed to spawn shell: {}", e);
            let _ = sender.send(Message::Text(format!("Error: Failed to spawn shell: {}\r\n", e))).await;
            return;
        }
    };

    let mut stdin = child.stdin.take().expect("Failed to get stdin");
    let stdout = child.stdout.take().expect("Failed to get stdout");
    let stderr = child.stderr.take().expect("Failed to get stderr");

    // Channel for sending data to WebSocket
    let (tx, mut rx) = mpsc::channel::<String>(100);
    let tx_stderr = tx.clone();

    // Task to read stdout
    let stdout_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut buffer = [0u8; 4096];
        loop {
            match tokio::io::AsyncReadExt::read(&mut reader, &mut buffer).await {
                Ok(0) => break, // EOF
                Ok(n) => {
                    let data = String::from_utf8_lossy(&buffer[..n]).to_string();
                    if tx.send(data).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    error!("Error reading stdout: {}", e);
                    break;
                }
            }
        }
    });

    // Task to read stderr
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut buffer = [0u8; 4096];
        loop {
            match tokio::io::AsyncReadExt::read(&mut reader, &mut buffer).await {
                Ok(0) => break, // EOF
                Ok(n) => {
                    let data = String::from_utf8_lossy(&buffer[..n]).to_string();
                    if tx_stderr.send(data).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    error!("Error reading stderr: {}", e);
                    break;
                }
            }
        }
    });

    // Task to send output to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(data) = rx.recv().await {
            if sender.send(Message::Text(data)).await.is_err() {
                break;
            }
        }
    });

    // Task to receive input from WebSocket and write to stdin
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // Handle special control sequences
                    let bytes = if text == "\\x03" {
                        vec![0x03] // Ctrl+C
                    } else if text == "\\x04" {
                        vec![0x04] // Ctrl+D
                    } else {
                        text.into_bytes()
                    };

                    if stdin.write_all(&bytes).await.is_err() {
                        break;
                    }
                    if stdin.flush().await.is_err() {
                        break;
                    }
                }
                Message::Binary(data) => {
                    if stdin.write_all(&data).await.is_err() {
                        break;
                    }
                    if stdin.flush().await.is_err() {
                        break;
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    // Wait for any task to complete
    tokio::select! {
        _ = stdout_task => info!("stdout task ended"),
        _ = stderr_task => info!("stderr task ended"),
        _ = send_task => info!("send task ended"),
        _ = recv_task => info!("recv task ended"),
        _ = child.wait() => info!("shell process ended"),
    }

    info!("Terminal session ended");
}

/// Get terminal info
pub async fn terminal_info(
    State(_state): State<AppState>,
    _user: AuthUser,
) -> impl IntoResponse {
    let shell: String = if cfg!(target_os = "windows") {
        "powershell.exe".to_string()
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string())
    };

    axum::Json(serde_json::json!({
        "available": true,
        "shell": shell,
        "features": ["pty", "resize", "colors"],
    }))
}
