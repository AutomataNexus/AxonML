//! Training Run Detail Page with Real-time Metrics

use leptos::*;
use leptos_router::*;
use web_sys::{WebSocket, MessageEvent, CloseEvent};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{icons::*, spinner::*, charts::*, progress::*, modal::*, table::StatusBadge};

/// Training run detail page
#[component]
pub fn TrainingDetailPage() -> impl IntoView {
    let params = use_params_map();
    let state = use_app_state();
    let navigate = use_navigate();

    let run_id = move || params.get().get("id").cloned().unwrap_or_default();

    let (loading, set_loading) = create_signal(true);
    let (run, set_run) = create_signal::<Option<TrainingRun>>(None);
    let (metrics_history, set_metrics_history) = create_signal::<Vec<TrainingMetrics>>(Vec::new());
    let (logs, set_logs) = create_signal::<Vec<LogEntry>>(Vec::new());
    let (ws_connected, set_ws_connected) = create_signal(false);
    let stop_modal = create_rw_signal(false);
    let delete_modal = create_rw_signal(false);
    let (active_tab, set_active_tab) = create_signal("metrics".to_string());

    // Clone state and navigate for different closures
    let state_for_effect = state.clone();
    let navigate_for_effect = navigate.clone();
    let state_for_stop = state.clone();
    let state_for_delete = state.clone();
    let navigate_for_delete = navigate.clone();

    // Fetch initial data
    create_effect(move |_| {
        let id = run_id();
        if id.is_empty() {
            return;
        }

        let state = state_for_effect.clone();
        let navigate = navigate_for_effect.clone();
        spawn_local(async move {
            // Fetch run details
            match api::training::get_run(&id).await {
                Ok(data) => {
                    set_run.set(Some(data));
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                    navigate("/training", Default::default());
                    return;
                }
            }

            // Fetch metrics history
            if let Ok(history) = api::training::get_metrics(&id, None, None).await {
                set_metrics_history.set(history.metrics);
            }

            // Fetch logs
            if let Ok(run_logs) = api::training::get_logs(&id, Some(100)).await {
                set_logs.set(run_logs.logs);
            }

            set_loading.set(false);
        });
    });

    // Set up WebSocket for real-time updates
    create_effect(move |_| {
        let id = run_id();
        if id.is_empty() {
            return;
        }

        let run_status = run.get().map(|r| r.status);
        if run_status != Some(RunStatus::Running) {
            return;
        }

        let ws_url = api::training::stream_url(&id);

        match WebSocket::new(&ws_url) {
            Ok(ws) => {
                ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

                // On open
                let on_open = Closure::<dyn Fn()>::new(move || {
                    set_ws_connected.set(true);
                });
                ws.set_onopen(Some(on_open.as_ref().unchecked_ref()));
                on_open.forget();

                // On message
                let on_message = Closure::<dyn Fn(MessageEvent)>::new(move |e: MessageEvent| {
                    if let Ok(text) = e.data().dyn_into::<js_sys::JsString>() {
                        let text: String = text.into();
                        if let Ok(msg) = serde_json::from_str::<WsMessage>(&text) {
                            match msg {
                                WsMessage::Metrics(m) => {
                                    set_metrics_history.update(|h| h.push(m.clone()));
                                    set_run.update(|r| {
                                        if let Some(run) = r {
                                            run.latest_metrics = Some(m);
                                        }
                                    });
                                }
                                WsMessage::Status { run_id: msg_run_id, status } => {
                                    // Only update if the message is for the current run
                                    if msg_run_id == id {
                                        set_run.update(|r| {
                                            if let Some(run) = r {
                                                run.status = status;
                                            }
                                        });
                                    }
                                }
                                WsMessage::Log(entry) => {
                                    set_logs.update(|l| l.push(entry));
                                }
                                _ => {}
                            }
                        }
                    }
                });
                ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
                on_message.forget();

                // On close
                let on_close = Closure::<dyn Fn(CloseEvent)>::new(move |_e: CloseEvent| {
                    set_ws_connected.set(false);
                });
                ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));
                on_close.forget();
            }
            Err(_) => {
                // WebSocket not supported or failed to connect
            }
        }
    });

    // Stop run handler
    let stop_run = move |_| {
        let id = run_id();
        let state = state_for_stop.clone();
        spawn_local(async move {
            match api::training::stop_run(&id).await {
                Ok(updated) => {
                    set_run.set(Some(updated));
                    state.toast_success("Stopped", "Training run stopped");
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
        });
        stop_modal.set(false);
    };

    // Delete run handler
    let delete_run = move |_| {
        let id = run_id();
        let state = state_for_delete.clone();
        let navigate = navigate_for_delete.clone();
        spawn_local(async move {
            match api::training::delete_run(&id).await {
                Ok(_) => {
                    state.toast_success("Deleted", "Training run deleted");
                    navigate("/training", Default::default());
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
        });
        delete_modal.set(false);
    };

    // Chart data
    let loss_series = move || {
        let history = metrics_history.get();
        vec![ChartSeries {
            name: "Loss".to_string(),
            data: history
                .iter()
                .map(|m| DataPoint {
                    x: m.epoch as f64 + (m.step as f64 / 1000.0),
                    y: m.loss,
                    label: None,
                })
                .collect(),
            color: "var(--teal)".to_string(),
        }]
    };

    let accuracy_series = move || {
        let history = metrics_history.get();
        let has_accuracy = history.iter().any(|m| m.accuracy.is_some());
        if !has_accuracy {
            return vec![];
        }
        vec![ChartSeries {
            name: "Accuracy".to_string(),
            data: history
                .iter()
                .filter_map(|m| {
                    m.accuracy.map(|acc| DataPoint {
                        x: m.epoch as f64 + (m.step as f64 / 1000.0),
                        y: acc * 100.0,
                        label: None,
                    })
                })
                .collect(),
            color: "var(--success)".to_string(),
        }]
    };

    view! {
        <div class="page training-detail-page">
            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading training run..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get() && run.get().is_some()>
                {move || run.get().map(|r| {
                    let is_running = r.status == RunStatus::Running;
                    let status_class = r.status.color_class();
                    // Clone latest_metrics to avoid borrow after move
                    let latest_metrics_for_progress = r.latest_metrics.clone();
                    let latest_metrics_for_stats = r.latest_metrics.clone();
                    let config_epochs = r.config.epochs;
                    let config_steps_per_epoch = r.config.steps_per_epoch;

                    view! {
                        // Header
                        <div class="page-header">
                            <div class="header-breadcrumb">
                                <A href="/training" class="btn btn-ghost btn-sm">
                                    <IconArrowLeft size=IconSize::Sm />
                                    <span>"Training Runs"</span>
                                </A>
                            </div>
                            <div class="header-content">
                                <div class="header-title-row">
                                    <h1>{r.name.clone()}</h1>
                                    <StatusBadge status=r.status.as_str().to_string() class=status_class.to_string() />
                                    <Show when=move || ws_connected.get()>
                                        <span class="live-indicator">
                                            <span class="live-dot"></span>
                                            "Live"
                                        </span>
                                    </Show>
                                </div>
                                <p class="page-subtitle">{r.model_type.clone()}</p>
                            </div>
                            <div class="header-actions">
                                <Show when=move || is_running>
                                    <button class="btn btn-warning" on:click=move |_| stop_modal.set(true)>
                                        <IconStop size=IconSize::Sm />
                                        <span>"Stop"</span>
                                    </button>
                                </Show>
                                <button class="btn btn-danger-outline" on:click=move |_| delete_modal.set(true)>
                                    <IconTrash size=IconSize::Sm />
                                    <span>"Delete"</span>
                                </button>
                            </div>
                        </div>

                        // Progress section for running jobs
                        <Show when=move || is_running>
                            {latest_metrics_for_progress.as_ref().map(|m| {
                                let epoch = m.epoch;
                                let step = m.step;
                                let total_epochs = config_epochs;
                                let total_steps = config_steps_per_epoch;
                                view! {
                                    <div class="card progress-card">
                                        <TrainingProgress
                                            epoch=MaybeSignal::derive(move || epoch)
                                            total_epochs=MaybeSignal::derive(move || total_epochs)
                                            step=MaybeSignal::derive(move || step)
                                            total_steps=MaybeSignal::derive(move || total_steps)
                                        />
                                    </div>
                                }
                            })}
                        </Show>

                        // Stats cards
                        <div class="stats-grid">
                            {latest_metrics_for_stats.as_ref().map(|m| view! {
                                <div class="stat-card">
                                    <div class="stat-icon"><IconLineChart size=IconSize::Md /></div>
                                    <div class="stat-content">
                                        <span class="stat-value">{format!("{:.4}", m.loss)}</span>
                                        <span class="stat-label">"Loss"</span>
                                    </div>
                                </div>
                                {m.accuracy.map(|acc| view! {
                                    <div class="stat-card">
                                        <div class="stat-icon"><IconTrendingUp size=IconSize::Md /></div>
                                        <div class="stat-content">
                                            <span class="stat-value">{format!("{:.2}%", acc * 100.0)}</span>
                                            <span class="stat-label">"Accuracy"</span>
                                        </div>
                                    </div>
                                })}
                                <div class="stat-card">
                                    <div class="stat-icon"><IconActivity size=IconSize::Md /></div>
                                    <div class="stat-content">
                                        <span class="stat-value">{format!("{}/{}", m.epoch, config_epochs)}</span>
                                        <span class="stat-label">"Epoch"</span>
                                    </div>
                                </div>
                                {m.gpu_utilization.map(|gpu| view! {
                                    <div class="stat-card">
                                        <div class="stat-icon"><IconCpu size=IconSize::Md /></div>
                                        <div class="stat-content">
                                            <span class="stat-value">{format!("{:.0}%", gpu)}</span>
                                            <span class="stat-label">"GPU"</span>
                                        </div>
                                    </div>
                                })}
                            })}
                        </div>

                        // Tabs
                        <div class="tabs-container">
                            <div class="tabs">
                                <button
                                    class=move || format!("tab {}", if active_tab.get() == "metrics" { "active" } else { "" })
                                    on:click=move |_| set_active_tab.set("metrics".to_string())
                                >
                                    <IconLineChart size=IconSize::Sm />
                                    <span>"Metrics"</span>
                                </button>
                                <button
                                    class=move || format!("tab {}", if active_tab.get() == "config" { "active" } else { "" })
                                    on:click=move |_| set_active_tab.set("config".to_string())
                                >
                                    <IconSettings size=IconSize::Sm />
                                    <span>"Configuration"</span>
                                </button>
                                <button
                                    class=move || format!("tab {}", if active_tab.get() == "logs" { "active" } else { "" })
                                    on:click=move |_| set_active_tab.set("logs".to_string())
                                >
                                    <IconTerminal size=IconSize::Sm />
                                    <span>"Logs"</span>
                                </button>
                            </div>

                            // Tab content
                            <div class="tab-content">
                                // Metrics tab
                                <Show when=move || active_tab.get() == "metrics">
                                    <div class="charts-grid">
                                        <div class="card chart-card">
                                            <div class="card-header">
                                                <h3>"Loss"</h3>
                                            </div>
                                            <div class="card-body">
                                                <LineChart
                                                    series=MaybeSignal::derive(loss_series)
                                                    config=ChartConfig {
                                                        height: 250,
                                                        y_label: Some("Loss".to_string()),
                                                        x_label: Some("Epoch".to_string()),
                                                        ..Default::default()
                                                    }
                                                />
                                            </div>
                                        </div>

                                        <Show when=move || !accuracy_series().is_empty()>
                                            <div class="card chart-card">
                                                <div class="card-header">
                                                    <h3>"Accuracy"</h3>
                                                </div>
                                                <div class="card-body">
                                                    <LineChart
                                                        series=MaybeSignal::derive(accuracy_series)
                                                        config=ChartConfig {
                                                            height: 250,
                                                            y_min: Some(0.0),
                                                            y_max: Some(100.0),
                                                            y_label: Some("Accuracy (%)".to_string()),
                                                            x_label: Some("Epoch".to_string()),
                                                            ..Default::default()
                                                        }
                                                    />
                                                </div>
                                            </div>
                                        </Show>
                                    </div>
                                </Show>

                                // Config tab
                                <Show when=move || active_tab.get() == "config">
                                    <div class="card">
                                        <div class="card-body">
                                            <div class="config-grid">
                                                <div class="config-item">
                                                    <span class="config-label">"Learning Rate"</span>
                                                    <span class="config-value">{format!("{:.0e}", r.config.learning_rate)}</span>
                                                </div>
                                                <div class="config-item">
                                                    <span class="config-label">"Batch Size"</span>
                                                    <span class="config-value">{r.config.batch_size}</span>
                                                </div>
                                                <div class="config-item">
                                                    <span class="config-label">"Epochs"</span>
                                                    <span class="config-value">{config_epochs}</span>
                                                </div>
                                                <div class="config-item">
                                                    <span class="config-label">"Optimizer"</span>
                                                    <span class="config-value">{r.config.optimizer.clone()}</span>
                                                </div>
                                                <div class="config-item">
                                                    <span class="config-label">"Loss Function"</span>
                                                    <span class="config-value">{r.config.loss_function.clone()}</span>
                                                </div>
                                                <div class="config-item">
                                                    <span class="config-label">"Started"</span>
                                                    <span class="config-value">{r.started_at.format("%Y-%m-%d %H:%M:%S").to_string()}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </Show>

                                // Logs tab
                                <Show when=move || active_tab.get() == "logs">
                                    <div class="card logs-card">
                                        <div class="card-body">
                                            <div class="logs-viewer">
                                                {move || {
                                                    let log_entries = logs.get();
                                                    if log_entries.is_empty() {
                                                        view! {
                                                            <div class="empty-state">
                                                                <p>"No logs available"</p>
                                                            </div>
                                                        }.into_view()
                                                    } else {
                                                        log_entries.into_iter().map(|entry| {
                                                            let level_class = match entry.level.as_str() {
                                                                "error" | "ERROR" => "log-error",
                                                                "warn" | "WARN" => "log-warn",
                                                                "info" | "INFO" => "log-info",
                                                                _ => "log-debug",
                                                            };
                                                            view! {
                                                                <div class=format!("log-entry {}", level_class)>
                                                                    <span class="log-time">
                                                                        {entry.timestamp.format("%H:%M:%S").to_string()}
                                                                    </span>
                                                                    <span class="log-level">{entry.level}</span>
                                                                    <span class="log-message">{entry.message}</span>
                                                                </div>
                                                            }
                                                        }).collect_view()
                                                    }
                                                }}
                                            </div>
                                        </div>
                                    </div>
                                </Show>
                            </div>
                        </div>

                    }
                })}
            </Show>

            // Modals - outside the Show to avoid FnOnce issues
            <ConfirmDialog
                show=stop_modal
                title="Stop Training?"
                message="This will stop the training run. You can resume from the last checkpoint later."
                confirm_text="Stop".to_string()
                danger=false
                on_confirm=stop_run
            />

            <ConfirmDialog
                show=delete_modal
                title="Delete Training Run?"
                message="This action cannot be undone. All metrics and logs will be permanently deleted."
                confirm_text="Delete".to_string()
                danger=true
                on_confirm=delete_run
            />
        </div>
    }
}
