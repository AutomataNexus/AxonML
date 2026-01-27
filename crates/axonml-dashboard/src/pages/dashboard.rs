//! Main Dashboard Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::components::{icons::*, spinner::*, table::*};
use crate::state::use_app_state;
use crate::types::*;

/// Main dashboard overview page
#[component]
pub fn DashboardPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (stats, set_stats) = create_signal(DashboardStats::default());
    let (recent_runs, set_recent_runs) = create_signal::<Vec<TrainingRun>>(Vec::new());
    let (active_endpoints, set_active_endpoints) =
        create_signal::<Vec<InferenceEndpoint>>(Vec::new());

    // Fetch dashboard data
    create_effect(move |_| {
        spawn_local(async move {
            // Fetch training runs
            if let Ok(runs) = api::training::list_runs(None, Some(5)).await {
                let active = runs
                    .iter()
                    .filter(|r| r.status == RunStatus::Running)
                    .count();
                let completed = runs
                    .iter()
                    .filter(|r| r.status == RunStatus::Completed)
                    .count();
                let failed = runs
                    .iter()
                    .filter(|r| r.status == RunStatus::Failed)
                    .count();

                set_stats.update(|s| {
                    s.active_runs = active as u32;
                    s.completed_runs = completed as u32;
                    s.failed_runs = failed as u32;
                });
                set_recent_runs.set(runs);
            }

            // Fetch models
            if let Ok(models) = api::models::list().await {
                set_stats.update(|s| s.total_models = models.len() as u32);
            }

            // Fetch endpoints
            if let Ok(endpoints) = api::inference::list_endpoints().await {
                let active = endpoints
                    .iter()
                    .filter(|e| e.status == EndpointStatus::Running)
                    .count();
                set_stats.update(|s| s.active_endpoints = active as u32);
                set_active_endpoints.set(
                    endpoints
                        .into_iter()
                        .filter(|e| e.status == EndpointStatus::Running)
                        .take(5)
                        .collect(),
                );
            }

            set_loading.set(false);
        });
    });

    view! {
        <div class="dashboard-page">
            <div class="page-header">
                <h1>"Dashboard"</h1>
                <p class="page-subtitle">
                    "Welcome back, " {move || state.user.get().map(|u| u.name).unwrap_or_else(|| "User".to_string())}
                </p>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading dashboard..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                // Stats Cards
                <div class="stats-grid">
                    <StatsCard
                        title="Active Training"
                        value=move || stats.get().active_runs.to_string()
                        subtitle="runs in progress"
                        icon=view! { <IconActivity size=IconSize::Lg /> }
                        color="teal"
                    />
                    <StatsCard
                        title="Completed"
                        value=move || stats.get().completed_runs.to_string()
                        subtitle="successful runs"
                        icon=view! { <IconCheckCircle size=IconSize::Lg /> }
                        color="success"
                    />
                    <StatsCard
                        title="Models"
                        value=move || stats.get().total_models.to_string()
                        subtitle="in registry"
                        icon=view! { <IconBox size=IconSize::Lg /> }
                        color="terracotta"
                    />
                    <StatsCard
                        title="Endpoints"
                        value=move || stats.get().active_endpoints.to_string()
                        subtitle="serving inference"
                        icon=view! { <IconServer size=IconSize::Lg /> }
                        color="info"
                    />
                </div>

                // Main Content Grid
                <div class="dashboard-grid">
                    // Recent Training Runs
                    <div class="card">
                        <div class="card-header">
                            <h2>"Recent Training Runs"</h2>
                            <A href="/training" class="btn btn-ghost btn-sm">
                                "View All"
                                <IconArrowRight size=IconSize::Sm />
                            </A>
                        </div>
                        <div class="card-body">
                            <Show
                                when=move || !recent_runs.get().is_empty()
                                fallback=|| view! {
                                    <div class="empty-state">
                                        <IconActivity size=IconSize::Xl class="text-muted".to_string() />
                                        <p>"No training runs yet"</p>
                                        <A href="/training/new" class="btn btn-primary btn-sm">
                                            "Start Training"
                                        </A>
                                    </div>
                                }
                            >
                                <div class="run-list">
                                    {move || recent_runs.get().into_iter().map(|run| {
                                        view! { <RunListItem run=run /> }
                                    }).collect_view()}
                                </div>
                            </Show>
                        </div>
                    </div>

                    // Active Endpoints
                    <div class="card">
                        <div class="card-header">
                            <h2>"Active Endpoints"</h2>
                            <A href="/inference/endpoints" class="btn btn-ghost btn-sm">
                                "Manage"
                                <IconArrowRight size=IconSize::Sm />
                            </A>
                        </div>
                        <div class="card-body">
                            <Show
                                when=move || !active_endpoints.get().is_empty()
                                fallback=|| view! {
                                    <div class="empty-state">
                                        <IconServer size=IconSize::Xl class="text-muted".to_string() />
                                        <p>"No active endpoints"</p>
                                        <A href="/models" class="btn btn-primary btn-sm">
                                            "Deploy Model"
                                        </A>
                                    </div>
                                }
                            >
                                <div class="endpoint-list">
                                    {move || active_endpoints.get().into_iter().map(|endpoint| {
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
                                <A href="/training/new" class="quick-action">
                                    <div class="action-icon teal">
                                        <IconPlus size=IconSize::Md />
                                    </div>
                                    <div class="action-text">
                                        <span class="action-title">"New Training Run"</span>
                                        <span class="action-desc">"Start training a model"</span>
                                    </div>
                                </A>
                                <A href="/models/upload" class="quick-action">
                                    <div class="action-icon terracotta">
                                        <IconUpload size=IconSize::Md />
                                    </div>
                                    <div class="action-text">
                                        <span class="action-title">"Upload Model"</span>
                                        <span class="action-desc">"Add to registry"</span>
                                    </div>
                                </A>
                                <A href="/inference/endpoints" class="quick-action">
                                    <div class="action-icon info">
                                        <IconServer size=IconSize::Md />
                                    </div>
                                    <div class="action-text">
                                        <span class="action-title">"Deploy Endpoint"</span>
                                        <span class="action-desc">"Serve predictions"</span>
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

/// Dashboard stats state
#[derive(Clone, Default)]
struct DashboardStats {
    active_runs: u32,
    completed_runs: u32,
    failed_runs: u32,
    total_models: u32,
    active_endpoints: u32,
}

/// Stats card component
#[component]
fn StatsCard(
    #[prop(into)] title: String,
    value: impl Fn() -> String + 'static,
    #[prop(into)] subtitle: String,
    icon: View,
    #[prop(into)] color: String,
) -> impl IntoView {
    view! {
        <div class=format!("stats-card stats-card-{}", color)>
            <div class="stats-icon">{icon}</div>
            <div class="stats-content">
                <span class="stats-value">{value}</span>
                <span class="stats-title">{title}</span>
                <span class="stats-subtitle">{subtitle}</span>
            </div>
        </div>
    }
}

/// Training run list item
#[component]
fn RunListItem(run: TrainingRun) -> impl IntoView {
    let status_class = run.status.color_class();

    view! {
        <A href=format!("/training/{}", run.id) class="run-list-item">
            <div class="run-info">
                <span class="run-name">{run.name}</span>
                <span class="run-model">{run.model_type}</span>
            </div>
            <div class="run-metrics">
                {run.latest_metrics.as_ref().map(|m| {
                    view! {
                        <span class="metric">
                            <span class="metric-label">"Loss:"</span>
                            <span class="metric-value">{format!("{:.4}", m.loss)}</span>
                        </span>
                    }
                })}
            </div>
            <StatusBadge status=run.status.as_str().to_string() class=status_class.to_string() />
        </A>
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
                    {format!("{} replicas", endpoint.replicas)}
                </span>
            </div>
            <StatusBadge status=endpoint.status.as_str().to_string() class=status_class.to_string() />
        </A>
    }
}

/// Application shell with sidebar and navbar
#[component]
pub fn AppShell(children: Children) -> impl IntoView {
    let state = use_app_state();
    let collapsed = move || state.sidebar_collapsed.get();

    view! {
        <div class=move || format!("app-shell {}", if collapsed() { "sidebar-collapsed" } else { "" })>
            <crate::components::navbar::Navbar />
            <div class="app-content">
                <crate::components::sidebar::Sidebar />
                <main class="main-content">
                    {children()}
                </main>
            </div>
            <crate::components::toast::ToastContainer />
            <crate::components::terminal::Terminal />
        </div>
    }
}
