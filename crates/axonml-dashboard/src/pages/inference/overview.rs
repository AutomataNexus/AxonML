//! Inference Overview Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{icons::*, spinner::*};

/// Inference overview page
#[component]
pub fn InferenceOverviewPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (endpoints, set_endpoints) = create_signal::<Vec<InferenceEndpoint>>(Vec::new());
    let (total_requests, set_total_requests) = create_signal(0u64);
    let (avg_latency, set_avg_latency) = create_signal(0.0f64);
    let (error_rate, set_error_rate) = create_signal(0.0f64);

    // Fetch data
    let state_for_effect = state.clone();
    create_effect(move |_| {
        let state = state_for_effect.clone();
        spawn_local(async move {
            match api::inference::list_endpoints().await {
                Ok(data) => {
                    // Fetch metrics for each active endpoint and aggregate
                    let mut total_reqs: u64 = 0;
                    let mut total_errors: u64 = 0;
                    let mut latency_sum: f64 = 0.0;
                    let mut latency_count: u32 = 0;

                    for endpoint in &data {
                        if endpoint.status == EndpointStatus::Running {
                            if let Ok(metrics_response) = api::inference::get_metrics(&endpoint.id, None, None).await {
                                for m in &metrics_response.metrics {
                                    total_reqs += m.requests_total;
                                    total_errors += m.requests_error;
                                    latency_sum += m.latency_p50;
                                    latency_count += 1;
                                }
                            }
                        }
                    }

                    set_total_requests.set(total_reqs);
                    if latency_count > 0 {
                        set_avg_latency.set(latency_sum / latency_count as f64);
                    }
                    if total_reqs > 0 {
                        set_error_rate.set((total_errors as f64 / total_reqs as f64) * 100.0);
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

    let active_endpoints = move || {
        endpoints.get().iter().filter(|e| e.status == EndpointStatus::Running).count()
    };

    let stopped_endpoints = move || {
        endpoints.get().iter().filter(|e| e.status == EndpointStatus::Stopped).count()
    };

    view! {
        <div class="page inference-overview-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Inference Overview"</h1>
                    <p class="page-subtitle">"Monitor your model serving infrastructure"</p>
                </div>
                <A href="/inference/endpoints" class="btn btn-primary">
                    <IconPlus size=IconSize::Sm />
                    <span>"New Endpoint"</span>
                </A>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading inference data..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                // Stats Cards
                <div class="stats-grid">
                    <div class="stat-card stat-card-success">
                        <div class="stat-icon"><IconServer size=IconSize::Lg /></div>
                        <div class="stat-content">
                            <span class="stat-value">{active_endpoints}</span>
                            <span class="stat-label">"Active Endpoints"</span>
                        </div>
                    </div>
                    <div class="stat-card stat-card-warning">
                        <div class="stat-icon"><IconPause size=IconSize::Lg /></div>
                        <div class="stat-content">
                            <span class="stat-value">{stopped_endpoints}</span>
                            <span class="stat-label">"Stopped"</span>
                        </div>
                    </div>
                    <div class="stat-card stat-card-teal">
                        <div class="stat-icon"><IconZap size=IconSize::Lg /></div>
                        <div class="stat-content">
                            <span class="stat-value">{move || format!("{:.0}", total_requests.get())}</span>
                            <span class="stat-label">"Total Requests"</span>
                        </div>
                    </div>
                    <div class="stat-card stat-card-info">
                        <div class="stat-icon"><IconClock size=IconSize::Lg /></div>
                        <div class="stat-content">
                            <span class="stat-value">{move || format!("{:.1}ms", avg_latency.get())}</span>
                            <span class="stat-label">"Avg Latency"</span>
                        </div>
                    </div>
                    <div class="stat-card stat-card-error">
                        <div class="stat-icon"><IconAlertCircle size=IconSize::Lg /></div>
                        <div class="stat-content">
                            <span class="stat-value">{move || format!("{:.2}%", error_rate.get())}</span>
                            <span class="stat-label">"Error Rate"</span>
                        </div>
                    </div>
                </div>

                // Main content
                <div class="inference-grid">
                    // Active Endpoints
                    <div class="card">
                        <div class="card-header">
                            <h2>"Active Endpoints"</h2>
                            <A href="/inference/endpoints" class="btn btn-ghost btn-sm">
                                "View All"
                                <IconArrowRight size=IconSize::Sm />
                            </A>
                        </div>
                        <div class="card-body">
                            <Show
                                when=move || !endpoints.get().is_empty()
                                fallback=|| view! {
                                    <div class="empty-state">
                                        <IconServer size=IconSize::Xl class="text-muted".to_string() />
                                        <h3>"No Endpoints"</h3>
                                        <p>"Deploy a model to create your first endpoint"</p>
                                        <A href="/models" class="btn btn-primary btn-sm">
                                            "Deploy Model"
                                        </A>
                                    </div>
                                }
                            >
                                <div class="endpoint-list">
                                    {move || endpoints.get().into_iter().take(5).map(|endpoint| {
                                        view! { <EndpointListItem endpoint=endpoint /> }
                                    }).collect_view()}
                                </div>
                            </Show>
                        </div>
                    </div>

                    // Quick Actions
                    <div class="card quick-actions-card">
                        <div class="card-header">
                            <h2>"Quick Actions"</h2>
                        </div>
                        <div class="card-body">
                            <div class="quick-actions">
                                <A href="/inference/endpoints" class="quick-action">
                                    <div class="action-icon teal">
                                        <IconServer size=IconSize::Md />
                                    </div>
                                    <div class="action-text">
                                        <span class="action-title">"Manage Endpoints"</span>
                                        <span class="action-desc">"View and control endpoints"</span>
                                    </div>
                                </A>
                                <A href="/inference/metrics" class="quick-action">
                                    <div class="action-icon terracotta">
                                        <IconBarChart size=IconSize::Md />
                                    </div>
                                    <div class="action-text">
                                        <span class="action-title">"View Metrics"</span>
                                        <span class="action-desc">"Latency and throughput"</span>
                                    </div>
                                </A>
                                <A href="/models" class="quick-action">
                                    <div class="action-icon info">
                                        <IconBox size=IconSize::Md />
                                    </div>
                                    <div class="action-text">
                                        <span class="action-title">"Model Registry"</span>
                                        <span class="action-desc">"Deploy new models"</span>
                                    </div>
                                </A>
                            </div>
                        </div>
                    </div>
                </div>
            </Show>
        </div>
    }
}

/// Endpoint list item
#[component]
fn EndpointListItem(endpoint: InferenceEndpoint) -> impl IntoView {
    let status_class = endpoint.status.color_class();

    view! {
        <A href=format!("/inference/endpoints/{}", endpoint.id) class="endpoint-list-item">
            <div class="endpoint-info">
                <span class="endpoint-name">{endpoint.name}</span>
                <span class="endpoint-port">{format!(":{}", endpoint.port)}</span>
            </div>
            <div class="endpoint-meta">
                <span class="replicas">
                    <IconServer size=IconSize::Xs />
                    {format!("{}", endpoint.replicas)}
                </span>
                {endpoint.model_name.as_ref().map(|name| view! {
                    <span class="model-name">
                        <IconBox size=IconSize::Xs />
                        {name.clone()}
                    </span>
                })}
            </div>
            <span class=format!("badge {}", status_class)>
                {endpoint.status.as_str()}
            </span>
        </A>
    }
}
