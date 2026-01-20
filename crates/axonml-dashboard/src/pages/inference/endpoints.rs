//! Inference Endpoints Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{icons::*, spinner::*, modal::*};

/// Endpoints list page
#[component]
pub fn EndpointsListPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (endpoints, set_endpoints) = create_signal::<Vec<InferenceEndpoint>>(Vec::new());
    let delete_modal = create_rw_signal(false);
    let (endpoint_to_delete, set_endpoint_to_delete) = create_signal::<Option<String>>(None);

    // Clone state for different closures
    let state_for_effect = state.clone();
    let state_for_refresh = state.clone();
    let state_for_delete = state.clone();
    let state_for_toggle = state.clone();

    // Initial fetch
    create_effect(move |_| {
        let state = state_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::inference::list_endpoints().await {
                Ok(data) => {
                    set_endpoints.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    });

    // Refresh handler
    let on_refresh = move |_| {
        let state = state_for_refresh.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::inference::list_endpoints().await {
                Ok(data) => {
                    set_endpoints.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    };

    // Delete endpoint handler
    let delete_endpoint = move |_| {
        if let Some(id) = endpoint_to_delete.get() {
            let state = state_for_delete.clone();
            spawn_local(async move {
                match api::inference::delete_endpoint(&id).await {
                    Ok(_) => {
                        state.toast_success("Deleted", "Endpoint deleted");
                        // Inline refresh
                        set_loading.set(true);
                        if let Ok(data) = api::inference::list_endpoints().await {
                            set_endpoints.set(data);
                        }
                        set_loading.set(false);
                    }
                    Err(e) => {
                        state.toast_error("Error", e.message);
                    }
                }
            });
        }
        delete_modal.set(false);
        set_endpoint_to_delete.set(None);
    };

    // Store delete handler for ConfirmDialog
    let delete_endpoint_stored = store_value(delete_endpoint);

    // Toggle endpoint handler - stored for use in nested closures
    let toggle_endpoint_fn = move |id: String, current_status: EndpointStatus| {
        let state = state_for_toggle.clone();
        spawn_local(async move {
            let result = if current_status == EndpointStatus::Running {
                api::inference::stop_endpoint(&id).await
            } else {
                api::inference::start_endpoint(&id).await
            };

            match result {
                Ok(_) => {
                    state.toast_success(
                        "Success",
                        if current_status == EndpointStatus::Running {
                            "Endpoint stopped"
                        } else {
                            "Endpoint started"
                        },
                    );
                    // Inline refresh
                    set_loading.set(true);
                    if let Ok(data) = api::inference::list_endpoints().await {
                        set_endpoints.set(data);
                    }
                    set_loading.set(false);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
        });
    };
    let toggle_endpoint = store_value(toggle_endpoint_fn);

    view! {
        <div class="page endpoints-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Inference Endpoints"</h1>
                    <p class="page-subtitle">"Manage your model serving endpoints"</p>
                </div>
                <div class="header-actions">
                    <button class="btn btn-ghost" on:click=on_refresh>
                        <IconRefresh size=IconSize::Sm />
                        <span>"Refresh"</span>
                    </button>
                    <A href="/models" class="btn btn-primary">
                        <IconPlus size=IconSize::Sm />
                        <span>"Deploy Model"</span>
                    </A>
                </div>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading endpoints..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                <Show
                    when=move || !endpoints.get().is_empty()
                    fallback=|| view! {
                        <div class="empty-state">
                            <IconServer size=IconSize::Xl class="text-muted".to_string() />
                            <h3>"No Endpoints"</h3>
                            <p>"Deploy a model to create your first inference endpoint"</p>
                            <A href="/models" class="btn btn-primary">
                                <IconPlus size=IconSize::Sm />
                                <span>"Deploy Model"</span>
                            </A>
                        </div>
                    }
                >
                    <div class="endpoints-grid">
                        {move || endpoints.get().into_iter().map(|endpoint| {
                            let id_for_toggle = endpoint.id.clone();
                            let id_for_delete = endpoint.id.clone();
                            let status = endpoint.status;

                            view! {
                                <EndpointCard
                                    endpoint=endpoint
                                    on_toggle={
                                        let id = id_for_toggle.clone();
                                        move |_| toggle_endpoint.with_value(|f| f(id.clone(), status))
                                    }
                                    on_delete=move |_| {
                                        set_endpoint_to_delete.set(Some(id_for_delete.clone()));
                                        delete_modal.set(true);
                                    }
                                />
                            }
                        }).collect_view()}
                    </div>
                </Show>
            </Show>

            <ConfirmDialog
                show=delete_modal
                title="Delete Endpoint?"
                message="This will stop and remove the endpoint. You can redeploy the model later."
                confirm_text="Delete".to_string()
                danger=true
                on_confirm=move |_| delete_endpoint_stored.with_value(|f| f(()))
            />
        </div>
    }
}

/// Endpoint card component
#[component]
fn EndpointCard(
    endpoint: InferenceEndpoint,
    #[prop(into)] on_toggle: Callback<()>,
    #[prop(into)] on_delete: Callback<()>,
) -> impl IntoView {
    let status_class = endpoint.status.color_class();
    let is_running = endpoint.status == EndpointStatus::Running;
    let endpoint_name_for_url = endpoint.name.clone();

    view! {
        <div class="endpoint-card card">
            <div class="card-header">
                <div class="endpoint-header-content">
                    <A href=format!("/inference/endpoints/{}", endpoint.id) class="endpoint-name">
                        {endpoint.name.clone()}
                    </A>
                    <span class=format!("badge {}", status_class)>
                        {endpoint.status.as_str()}
                    </span>
                </div>
                <div class="endpoint-actions">
                    <div class="dropdown">
                        <button class="btn btn-ghost btn-sm">
                            <IconMoreVertical size=IconSize::Sm />
                        </button>
                        <div class="dropdown-menu dropdown-menu-right">
                            <A href=format!("/inference/endpoints/{}", endpoint.id) class="dropdown-item">
                                <IconEye size=IconSize::Sm />
                                <span>"View Details"</span>
                            </A>
                            <button class="dropdown-item" on:click=move |_| on_toggle.call(())>
                                {if is_running {
                                    view! {
                                        <IconStop size=IconSize::Sm />
                                        <span>"Stop"</span>
                                    }.into_view()
                                } else {
                                    view! {
                                        <IconPlay size=IconSize::Sm />
                                        <span>"Start"</span>
                                    }.into_view()
                                }}
                            </button>
                            <div class="dropdown-divider"></div>
                            <button class="dropdown-item text-danger" on:click=move |_| on_delete.call(())>
                                <IconTrash size=IconSize::Sm />
                                <span>"Delete"</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card-body">
                <div class="endpoint-info-grid">
                    <div class="info-item">
                        <span class="info-label">"Port"</span>
                        <span class="info-value">{endpoint.port}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">"Replicas"</span>
                        <span class="info-value">{endpoint.replicas}</span>
                    </div>
                    {endpoint.model_name.as_ref().map(|name| view! {
                        <div class="info-item">
                            <span class="info-label">"Model"</span>
                            <span class="info-value">{name.clone()}</span>
                        </div>
                    })}
                    {endpoint.version.map(|v| view! {
                        <div class="info-item">
                            <span class="info-label">"Version"</span>
                            <span class="info-value">{format!("v{}", v)}</span>
                        </div>
                    })}
                </div>

                // API URL
                <div class="endpoint-url">
                    <span class="url-label">"API:"</span>
                    <code class="url-value">{format!("/api/inference/predict/{}", endpoint_name_for_url)}</code>
                    <button class="btn btn-ghost btn-sm copy-btn">
                        <IconCopy size=IconSize::Xs />
                    </button>
                </div>
            </div>

            <div class="card-footer">
                <button
                    class=format!("btn btn-sm {}", if is_running { "btn-warning" } else { "btn-success" })
                    on:click=move |_| on_toggle.call(())
                >
                    {if is_running {
                        view! {
                            <IconStop size=IconSize::Sm />
                            <span>"Stop"</span>
                        }
                    } else {
                        view! {
                            <IconPlay size=IconSize::Sm />
                            <span>"Start"</span>
                        }
                    }}
                </button>
                <A href=format!("/inference/endpoints/{}", endpoint.id) class="btn btn-ghost btn-sm">
                    "Details"
                    <IconArrowRight size=IconSize::Sm />
                </A>
            </div>
        </div>
    }
}

/// Endpoint detail page
#[component]
pub fn EndpointDetailPage() -> impl IntoView {
    let params = use_params_map();
    let state = use_app_state();
    let navigate = use_navigate();

    let endpoint_id = move || params.get().get("id").cloned().unwrap_or_default();

    let (loading, set_loading) = create_signal(true);
    let (endpoint, set_endpoint) = create_signal::<Option<InferenceEndpoint>>(None);
    let (metrics, set_metrics) = create_signal::<Vec<InferenceMetrics>>(Vec::new());

    // Fetch endpoint data
    let state_for_effect = state.clone();
    let navigate_for_effect = navigate.clone();
    create_effect(move |_| {
        let id = endpoint_id();
        if id.is_empty() {
            return;
        }

        let state = state_for_effect.clone();
        let navigate = navigate_for_effect.clone();
        spawn_local(async move {
            match api::inference::get_endpoint(&id).await {
                Ok(data) => {
                    set_endpoint.set(Some(data));
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                    navigate("/inference/endpoints", Default::default());
                    return;
                }
            }

            // Fetch metrics
            if let Ok(metrics_data) = api::inference::get_metrics(&id, None, None).await {
                set_metrics.set(metrics_data.metrics);
            }

            set_loading.set(false);
        });
    });

    view! {
        <div class="page endpoint-detail-page">
            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading endpoint..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get() && endpoint.get().is_some()>
                {move || endpoint.get().map(|e| {
                    let status_class = e.status.color_class();
                    let is_running = e.status == EndpointStatus::Running;

                    // Calculate metrics aggregates
                    let current_metrics = metrics.get();
                    let total_requests: u64 = current_metrics.iter().map(|m| m.requests_total).sum();
                    let avg_latency: f64 = if current_metrics.is_empty() {
                        0.0
                    } else {
                        current_metrics.iter().map(|m| m.latency_p50).sum::<f64>() / current_metrics.len() as f64
                    };
                    let error_count: u64 = current_metrics.iter().map(|m| m.requests_error).sum();

                    view! {
                        <div class="page-header">
                            <div class="header-breadcrumb">
                                <A href="/inference/endpoints" class="btn btn-ghost btn-sm">
                                    <IconArrowLeft size=IconSize::Sm />
                                    <span>"Endpoints"</span>
                                </A>
                            </div>
                            <div class="header-content">
                                <div class="header-title-row">
                                    <h1>{e.name.clone()}</h1>
                                    <span class=format!("badge {}", status_class)>
                                        {e.status.as_str()}
                                    </span>
                                </div>
                                <p class="page-subtitle">{format!("Port: {}", e.port)}</p>
                            </div>
                            <div class="header-actions">
                                {if is_running {
                                    view! {
                                        <span class="live-indicator">
                                            <span class="live-dot"></span>
                                            "Running"
                                        </span>
                                    }.into_view()
                                } else {
                                    view! {
                                        <span class="badge badge-warning">"Stopped"</span>
                                    }.into_view()
                                }}
                            </div>
                        </div>

                        // Stats
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-icon"><IconServer size=IconSize::Md /></div>
                                <div class="stat-content">
                                    <span class="stat-value">{e.replicas}</span>
                                    <span class="stat-label">"Replicas"</span>
                                </div>
                            </div>
                            <div class="stat-card stat-card-teal">
                                <div class="stat-icon"><IconZap size=IconSize::Md /></div>
                                <div class="stat-content">
                                    <span class="stat-value">{total_requests}</span>
                                    <span class="stat-label">"Total Requests"</span>
                                </div>
                            </div>
                            <div class="stat-card stat-card-info">
                                <div class="stat-icon"><IconClock size=IconSize::Md /></div>
                                <div class="stat-content">
                                    <span class="stat-value">{format!("{:.1}ms", avg_latency)}</span>
                                    <span class="stat-label">"Avg Latency (P50)"</span>
                                </div>
                            </div>
                            <div class="stat-card stat-card-error">
                                <div class="stat-icon"><IconAlertCircle size=IconSize::Md /></div>
                                <div class="stat-content">
                                    <span class="stat-value">{error_count}</span>
                                    <span class="stat-label">"Errors"</span>
                                </div>
                            </div>
                        </div>

                        // Configuration
                        <div class="card">
                            <div class="card-header">
                                <h2>"Configuration"</h2>
                            </div>
                            <div class="card-body">
                                <div class="config-grid">
                                    <div class="config-item">
                                        <span class="config-label">"Batch Size"</span>
                                        <span class="config-value">{e.config.batch_size}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="config-label">"Timeout"</span>
                                        <span class="config-value">{format!("{}ms", e.config.timeout_ms)}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="config-label">"Max Concurrent"</span>
                                        <span class="config-value">{e.config.max_concurrent}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        // API Info
                        <div class="card">
                            <div class="card-header">
                                <h2>"API Information"</h2>
                            </div>
                            <div class="card-body">
                                <div class="api-info">
                                    <div class="api-url">
                                        <span class="url-method">"POST"</span>
                                        <code class="url-value">{format!("/api/inference/predict/{}", e.name)}</code>
                                    </div>
                                    <div class="api-example">
                                        <h4>"Example Request"</h4>
                                        <pre><code>{r#"curl -X POST http://localhost:3000/api/inference/predict/"#}{e.name.clone()}{r#" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [1, 2, 3]}'"#}</code></pre>
                                    </div>
                                </div>
                            </div>
                        </div>
                    }
                })}
            </Show>
        </div>
    }
}
