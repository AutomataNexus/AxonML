//! Terminal Panel — WebSocket-Backed Shell Embedded in the Dashboard
//!
//! A collapsible terminal widget that proxies a shell over a WebSocket
//! (`/api/terminal?token=...`) so an authenticated user can run commands
//! without leaving the dashboard.
//!
//! `TerminalState` (`Closed`/`Open`/`Minimized`) drives the panel's
//! visual mode. `Terminal` owns four signals: `terminal_state`,
//! `connected` (live socket indicator), `output` (accumulated lines
//! joined into a `<pre>`), and `input` (current command buffer). The
//! actual `WebSocket` is held in a `StoredValue<Option<WebSocket>>`.
//!
//! `connect` picks `ws`/`wss` from `location.protocol`, targets
//! `localhost:3021` when the page is served from the known dev ports
//! (8083/8081), otherwise reuses the current host, and opens the
//! socket with the current access token as a query parameter. It wires
//! `onopen`/`onmessage`/`onclose`/`onerror` closures via `wasm_bindgen`
//! and `.forget()` to persist them. `disconnect` closes the socket.
//! `send_input` pushes the input plus `\n` on Enter; Ctrl+C sends
//! `\\x03` to the remote shell.
//!
//! `toggle` is the floating button action (connect on first click,
//! minimize/restore thereafter); `minimize` and `close` are the header
//! controls (close also clears output). The rendered panel has a header
//! with a connection badge, an output `<pre>`, and a disabled-when-
//! disconnected input row with a `$ ` prompt.
//!
//! # File
//! `crates/axonml-dashboard/src/components/terminal.rs`
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

use leptos::*;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CloseEvent, ErrorEvent, MessageEvent, WebSocket};

use crate::state::use_app_state;

// =============================================================================
// JS Interop
// =============================================================================

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// =============================================================================
// TerminalState
// =============================================================================

/// Terminal panel state
#[derive(Clone, Copy, PartialEq)]
pub enum TerminalState {
    Closed,
    Open,
    Minimized,
}

// =============================================================================
// Terminal Component
// =============================================================================

/// Terminal component
#[component]
pub fn Terminal() -> impl IntoView {
    let state = use_app_state();
    let (terminal_state, set_terminal_state) = create_signal(TerminalState::Closed);
    let (connected, set_connected) = create_signal(false);
    let (output, set_output) = create_signal(Vec::<String>::new());
    let (input, set_input) = create_signal(String::new());
    let ws: StoredValue<Option<WebSocket>> = store_value(None);

    // -------------------------------------------------------------------------
    // WebSocket Lifecycle
    // -------------------------------------------------------------------------

    // Connect to terminal WebSocket
    let connect = move |_| {
        let token = state.get_access_token().unwrap_or_default();
        let protocol = if web_sys::window()
            .and_then(|w| w.location().protocol().ok())
            .map(|p| p == "https:")
            .unwrap_or(false)
        {
            "wss"
        } else {
            "ws"
        };

        let host = web_sys::window()
            .and_then(|w| w.location().host().ok())
            .unwrap_or_else(|| "localhost:3021".to_string());

        // For development, use the API port directly
        let ws_host = if host.contains("8083") || host.contains("8081") {
            "localhost:3021".to_string()
        } else {
            host
        };

        let url = format!("{}://{}/api/terminal?token={}", protocol, ws_host, token);

        match WebSocket::new(&url) {
            Ok(socket) => {
                socket.set_binary_type(web_sys::BinaryType::Arraybuffer);

                // On open
                let set_connected_clone = set_connected;
                let set_output_clone = set_output;
                let onopen = Closure::wrap(Box::new(move |_| {
                    set_connected_clone.set(true);
                    set_output_clone.update(|o| o.push("Connected to terminal.\r\n".to_string()));
                }) as Box<dyn FnMut(JsValue)>);
                socket.set_onopen(Some(onopen.as_ref().unchecked_ref()));
                onopen.forget();

                // On message
                let set_output_clone = set_output;
                let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
                    if let Ok(text) = e.data().dyn_into::<js_sys::JsString>() {
                        let text: String = text.into();
                        set_output_clone.update(|o| o.push(text));
                    }
                }) as Box<dyn FnMut(MessageEvent)>);
                socket.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
                onmessage.forget();

                // On close
                let set_connected_clone = set_connected;
                let set_output_clone = set_output;
                let onclose = Closure::wrap(Box::new(move |_: CloseEvent| {
                    set_connected_clone.set(false);
                    set_output_clone.update(|o| o.push("\r\nConnection closed.\r\n".to_string()));
                }) as Box<dyn FnMut(CloseEvent)>);
                socket.set_onclose(Some(onclose.as_ref().unchecked_ref()));
                onclose.forget();

                // On error
                let set_output_clone = set_output;
                let onerror = Closure::wrap(Box::new(move |_: ErrorEvent| {
                    set_output_clone.update(|o| o.push("\r\nConnection error.\r\n".to_string()));
                }) as Box<dyn FnMut(ErrorEvent)>);
                socket.set_onerror(Some(onerror.as_ref().unchecked_ref()));
                onerror.forget();

                ws.set_value(Some(socket));
                set_terminal_state.set(TerminalState::Open);
            }
            Err(e) => {
                log(&format!("Failed to create WebSocket: {:?}", e));
                set_output.update(|o| o.push("Failed to connect.\r\n".to_string()));
            }
        }
    };

    // Disconnect
    let disconnect = move |_| {
        if let Some(socket) = ws.get_value() {
            let _ = socket.close();
        }
        ws.set_value(None);
        set_connected.set(false);
    };

    // -------------------------------------------------------------------------
    // Input & Keyboard
    // -------------------------------------------------------------------------

    // Send input
    let send_input = move |_| {
        let text = input.get();
        if !text.is_empty() {
            if let Some(socket) = ws.get_value() {
                if socket.ready_state() == WebSocket::OPEN {
                    let _ = socket.send_with_str(&format!("{}\n", text));
                    set_input.set(String::new());
                }
            }
        }
    };

    // Handle key press
    let on_keydown = move |e: web_sys::KeyboardEvent| {
        if e.key() == "Enter" {
            send_input(());
        } else if e.key() == "c" && e.ctrl_key() {
            // Send Ctrl+C
            if let Some(socket) = ws.get_value() {
                if socket.ready_state() == WebSocket::OPEN {
                    let _ = socket.send_with_str("\\x03");
                }
            }
        }
    };

    // -------------------------------------------------------------------------
    // Panel State Handlers
    // -------------------------------------------------------------------------

    // Toggle terminal (for main button)
    let toggle = move |_| match terminal_state.get() {
        TerminalState::Closed => {
            connect(());
        }
        TerminalState::Open => {
            set_terminal_state.set(TerminalState::Minimized);
        }
        TerminalState::Minimized => {
            set_terminal_state.set(TerminalState::Open);
        }
    };

    // Minimize terminal (for header button)
    let minimize = move |_| {
        set_terminal_state.set(TerminalState::Minimized);
    };

    // Close terminal
    let close = move |_| {
        disconnect(());
        set_terminal_state.set(TerminalState::Closed);
        set_output.set(Vec::new());
    };

    // -------------------------------------------------------------------------
    // View
    // -------------------------------------------------------------------------

    view! {
        // Terminal toggle button (always visible)
        <button
            class="terminal-toggle-btn"
            on:click=toggle
            title="Toggle Terminal"
        >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="4,17 10,11 4,5"></polyline>
                <line x1="12" y1="19" x2="20" y2="19"></line>
            </svg>
            <Show when=move || connected.get()>
                <span class="terminal-status-dot connected"></span>
            </Show>
        </button>

        // Terminal panel
        <div class=move || format!("terminal-panel {}",
            match terminal_state.get() {
                TerminalState::Closed => "closed",
                TerminalState::Open => "open",
                TerminalState::Minimized => "minimized",
            }
        )>
            // Header
            <div class="terminal-header">
                <div class="terminal-title">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="4,17 10,11 4,5"></polyline>
                        <line x1="12" y1="19" x2="20" y2="19"></line>
                    </svg>
                    <span>"Terminal"</span>
                    <Show when=move || connected.get()>
                        <span class="badge badge-success">"Connected"</span>
                    </Show>
                    <Show when=move || !connected.get()>
                        <span class="badge badge-secondary">"Disconnected"</span>
                    </Show>
                </div>
                <div class="terminal-controls">
                    <button class="btn btn-ghost btn-xs" on:click=minimize title="Minimize">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                        </svg>
                    </button>
                    <button class="btn btn-ghost btn-xs" on:click=close title="Close">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
            </div>

            // Terminal output
            <div class="terminal-output">
                <pre>{move || output.get().join("")}</pre>
            </div>

            // Input
            <div class="terminal-input">
                <span class="terminal-prompt">"$ "</span>
                <input
                    type="text"
                    placeholder="Enter command..."
                    prop:value=move || input.get()
                    on:input=move |e| set_input.set(event_target_value(&e))
                    on:keydown=on_keydown
                    disabled=move || !connected.get()
                />
            </div>
        </div>
    }
}
