//! Inference Metrics Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::components::{charts::*, icons::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// Inference metrics page
#[component]
pub fn InferenceMetricsPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (endpoints, set_endpoints) = create_signal::<Vec<InferenceEndpoint>>(Vec::new());
    let (selected_endpoint, set_selected_endpoint) = create_signal::<Option<String>>(None);
    let (metrics, set_metrics) = create_signal::<Vec<InferenceMetrics>>(Vec::new());

    // Clone state for the effect
    let state_for_effect = state.clone();

    // Fetch endpoints
    create_effect(move |_| {
        let state = state_for_effect.clone();
        spawn_local(async move {
            match api::inference::list_endpoints().await {
                Ok(data) => {
                    if !data.is_empty() {
                        set_selected_endpoint.set(Some(data[0].id.clone()));
                    }
                    set_endpoints.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    });

    // Fetch metrics when endpoint changes
    create_effect(move |_| {
        if let Some(id) = selected_endpoint.get() {
            spawn_local(async move {
                if let Ok(data) = api::inference::get_metrics(&id, None, None).await {
                    set_metrics.set(data.metrics);
                }
            });
        }
    });

    // Chart data
    let latency_series = move || {
        let m = metrics.get();
        vec![
            ChartSeries {
                name: "P50".to_string(),
                data: m
                    .iter()
                    .enumerate()
                    .map(|(i, metric)| DataPoint {
                        x: i as f64,
                        y: metric.latency_p50,
                        label: None,
                    })
                    .collect(),
                color: "var(--teal)".to_string(),
            },
            ChartSeries {
                name: "P95".to_string(),
                data: m
                    .iter()
                    .enumerate()
                    .map(|(i, metric)| DataPoint {
                        x: i as f64,
                        y: metric.latency_p95,
                        label: None,
                    })
                    .collect(),
                color: "var(--terracotta)".to_string(),
            },
            ChartSeries {
                name: "P99".to_string(),
                data: m
                    .iter()
                    .enumerate()
                    .map(|(i, metric)| DataPoint {
                        x: i as f64,
                        y: metric.latency_p99,
                        label: None,
                    })
                    .collect(),
                color: "var(--error)".to_string(),
            },
        ]
    };

    let requests_series = move || {
        let m = metrics.get();
        vec![
            ChartSeries {
                name: "Success".to_string(),
                data: m
                    .iter()
                    .enumerate()
                    .map(|(i, metric)| DataPoint {
                        x: i as f64,
                        y: metric.requests_success as f64,
                        label: None,
                    })
                    .collect(),
                color: "var(--success)".to_string(),
            },
            ChartSeries {
                name: "Error".to_string(),
                data: m
                    .iter()
                    .enumerate()
                    .map(|(i, metric)| DataPoint {
                        x: i as f64,
                        y: metric.requests_error as f64,
                        label: None,
                    })
                    .collect(),
                color: "var(--error)".to_string(),
            },
        ]
    };

    // Aggregate stats
    let total_requests = move || metrics.get().iter().map(|m| m.requests_total).sum::<u64>();

    let avg_latency = move || {
        let m = metrics.get();
        if m.is_empty() {
            0.0
        } else {
            m.iter().map(|m| m.latency_p50).sum::<f64>() / m.len() as f64
        }
    };

    let error_rate = move || {
        let m = metrics.get();
        let total: u64 = m.iter().map(|m| m.requests_total).sum();
        let errors: u64 = m.iter().map(|m| m.requests_error).sum();
        if total > 0 {
            (errors as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    };

    view! {
        <div class="page inference-metrics-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Inference Metrics"</h1>
                    <p class="page-subtitle">"Monitor latency and throughput"</p>
                </div>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading metrics..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                // Endpoint selector
                <Show when=move || !endpoints.get().is_empty()>
                    <div class="filters-bar">
                        <div class="form-group">
                            <label class="form-label">"Endpoint"</label>
                            <div class="select-wrapper">
                                <select
                                    class="form-select"
                                    on:change=move |e| {
                                        let value = event_target_value(&e);
                                        set_selected_endpoint.set(Some(value));
                                    }
                                >
                                    {move || endpoints.get().into_iter().map(|ep| {
                                        let id = ep.id.clone();
                                        let selected = selected_endpoint.get() == Some(id.clone());
                                        view! {
                                            <option value=id selected=selected>
                                                {ep.name}
                                            </option>
                                        }
                                    }).collect_view()}
                                </select>
                                <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
                            </div>
                        </div>
                    </div>
                </Show>

                // Stats
                <div class="stats-grid">
                    <div class="stat-card stat-card-teal">
                        <div class="stat-icon"><IconZap size=IconSize::Lg /></div>
                        <div class="stat-content">
                            <span class="stat-value">{total_requests}</span>
                            <span class="stat-label">"Total Requests"</span>
                        </div>
                    </div>
                    <div class="stat-card stat-card-info">
                        <div class="stat-icon"><IconClock size=IconSize::Lg /></div>
                        <div class="stat-content">
                            <span class="stat-value">{move || format!("{:.1}ms", avg_latency())}</span>
                            <span class="stat-label">"Avg Latency (P50)"</span>
                        </div>
                    </div>
                    <div class="stat-card stat-card-error">
                        <div class="stat-icon"><IconAlertCircle size=IconSize::Lg /></div>
                        <div class="stat-content">
                            <span class="stat-value">{move || format!("{:.2}%", error_rate())}</span>
                            <span class="stat-label">"Error Rate"</span>
                        </div>
                    </div>
                </div>

                // Charts
                <div class="charts-grid">
                    <div class="card chart-card">
                        <div class="card-header">
                            <h2>"Latency Distribution"</h2>
                        </div>
                        <div class="card-body">
                            <Show
                                when=move || !metrics.get().is_empty()
                                fallback=|| view! {
                                    <div class="empty-state">
                                        <IconLineChart size=IconSize::Xl class="text-muted".to_string() />
                                        <p>"No metrics data available"</p>
                                    </div>
                                }
                            >
                                <LineChart
                                    series=Signal::derive(latency_series)
                                    config=ChartConfig {
                                        height: 300,
                                        y_label: Some("Latency (ms)".to_string()),
                                        ..Default::default()
                                    }
                                />
                            </Show>
                        </div>
                    </div>

                    <div class="card chart-card">
                        <div class="card-header">
                            <h2>"Request Volume"</h2>
                        </div>
                        <div class="card-body">
                            <Show
                                when=move || !metrics.get().is_empty()
                                fallback=|| view! {
                                    <div class="empty-state">
                                        <IconBarChart size=IconSize::Xl class="text-muted".to_string() />
                                        <p>"No metrics data available"</p>
                                    </div>
                                }
                            >
                                <LineChart
                                    series=Signal::derive(requests_series)
                                    config=ChartConfig {
                                        height: 300,
                                        y_label: Some("Requests".to_string()),
                                        ..Default::default()
                                    }
                                />
                            </Show>
                        </div>
                    </div>
                </div>

                // Empty state
                <Show when=move || endpoints.get().is_empty()>
                    <div class="empty-state">
                        <IconServer size=IconSize::Xl class="text-muted".to_string() />
                        <h3>"No Endpoints"</h3>
                        <p>"Deploy a model to start collecting metrics"</p>
                        <A href="/models" class="btn btn-primary">
                            "Deploy Model"
                        </A>
                    </div>
                </Show>
            </Show>
        </div>
    }
}
