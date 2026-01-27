//! Training Runs List Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::components::{icons::*, modal::*, spinner::*, table::*};
use crate::state::use_app_state;
use crate::types::*;

/// Training runs list page
#[component]
pub fn TrainingListPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (runs, set_runs) = create_signal::<Vec<TrainingRun>>(Vec::new());
    let (filter, set_filter) = create_signal::<Option<RunStatus>>(None);
    let (search, set_search) = create_signal(String::new());
    let delete_modal = create_rw_signal(false);
    let (run_to_delete, set_run_to_delete) = create_signal::<Option<String>>(None);

    // Clone state for different closures
    let state_for_effect = state.clone();
    let state_for_refresh = state.clone();
    let state_for_delete = state.clone();

    // Initial fetch on filter changes
    create_effect(move |_| {
        filter.get(); // Subscribe to filter changes
        let state = state_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::training::list_runs(filter.get(), None).await {
                Ok(data) => {
                    set_runs.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    });

    // Filtered runs based on search
    let filtered_runs = move || {
        let search_term = search.get().to_lowercase();
        runs.get()
            .into_iter()
            .filter(|r| {
                search_term.is_empty()
                    || r.name.to_lowercase().contains(&search_term)
                    || r.model_type.to_lowercase().contains(&search_term)
            })
            .collect::<Vec<_>>()
    };

    // Refresh handler
    let on_refresh = move |_| {
        let state = state_for_refresh.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::training::list_runs(filter.get(), None).await {
                Ok(data) => {
                    set_runs.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    };

    let delete_run = move |_| {
        if let Some(id) = run_to_delete.get() {
            let state = state_for_delete.clone();
            spawn_local(async move {
                match api::training::delete_run(&id).await {
                    Ok(_) => {
                        state.toast_success("Deleted", "Training run deleted successfully");
                        // Refresh the list
                        set_loading.set(true);
                        if let Ok(data) = api::training::list_runs(filter.get(), None).await {
                            set_runs.set(data);
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
        set_run_to_delete.set(None);
    };

    view! {
        <div class="page training-list-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Training Runs"</h1>
                    <p class="page-subtitle">"Monitor and manage your training experiments"</p>
                </div>
                <A href="/training/new" class="btn btn-primary">
                    <IconPlus size=IconSize::Sm />
                    <span>"New Run"</span>
                </A>
            </div>

            // Filters
            <div class="filters-bar">
                <div class="search-box">
                    <IconSearch size=IconSize::Sm />
                    <input
                        type="text"
                        placeholder="Search runs..."
                        prop:value=move || search.get()
                        on:input=move |e| set_search.set(event_target_value(&e))
                    />
                </div>

                <div class="filter-buttons">
                    <button
                        class=move || format!("btn btn-sm {}", if filter.get().is_none() { "btn-primary" } else { "btn-ghost" })
                        on:click=move |_| set_filter.set(None)
                    >
                        "All"
                    </button>
                    <button
                        class=move || format!("btn btn-sm {}", if filter.get() == Some(RunStatus::Running) { "btn-primary" } else { "btn-ghost" })
                        on:click=move |_| set_filter.set(Some(RunStatus::Running))
                    >
                        "Running"
                    </button>
                    <button
                        class=move || format!("btn btn-sm {}", if filter.get() == Some(RunStatus::Completed) { "btn-primary" } else { "btn-ghost" })
                        on:click=move |_| set_filter.set(Some(RunStatus::Completed))
                    >
                        "Completed"
                    </button>
                    <button
                        class=move || format!("btn btn-sm {}", if filter.get() == Some(RunStatus::Failed) { "btn-primary" } else { "btn-ghost" })
                        on:click=move |_| set_filter.set(Some(RunStatus::Failed))
                    >
                        "Failed"
                    </button>
                </div>

                <button class="btn btn-ghost btn-sm" on:click=on_refresh>
                    <IconRefresh size=IconSize::Sm />
                    <span>"Refresh"</span>
                </button>
            </div>

            // Content
            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading training runs..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                <Show
                    when=move || !filtered_runs().is_empty()
                    fallback=move || view! {
                        <div class="empty-state">
                            <IconActivity size=IconSize::Xl class="text-muted".to_string() />
                            <h3>"No Training Runs"</h3>
                            <p>"Start your first training run to see it here"</p>
                            <A href="/training/new" class="btn btn-primary">
                                <IconPlus size=IconSize::Sm />
                                <span>"Start Training"</span>
                            </A>
                        </div>
                    }
                >
                    <div class="runs-grid">
                        {move || filtered_runs().into_iter().map(|run| {
                            let run_id_for_delete = run.id.clone();

                            view! {
                                <TrainingRunCard
                                    run=run
                                    on_delete=move |_| {
                                        set_run_to_delete.set(Some(run_id_for_delete.clone()));
                                        delete_modal.set(true);
                                    }
                                />
                            }
                        }).collect_view()}
                    </div>
                </Show>
            </Show>

            // Delete confirmation modal
            <ConfirmDialog
                show=delete_modal
                title="Delete Training Run?"
                message="This action cannot be undone. All metrics and logs for this run will be permanently deleted."
                confirm_text="Delete".to_string()
                danger=true
                on_confirm=delete_run
            />
        </div>
    }
}

/// Training run card component
#[component]
fn TrainingRunCard(run: TrainingRun, #[prop(into)] on_delete: Callback<()>) -> impl IntoView {
    let status_class = run.status.color_class();
    let is_running = run.status == RunStatus::Running;
    let run_id = run.id.clone();

    view! {
        <div class="run-card card">
            <div class="card-header">
                <div class="run-header-content">
                    <A href=format!("/training/{}", run_id) class="run-name">
                        {run.name.clone()}
                    </A>
                    <StatusBadge status=run.status.as_str().to_string() class=status_class.to_string() />
                </div>
                <div class="run-actions">
                    <div class="dropdown">
                        <button class="btn btn-ghost btn-sm">
                            <IconMoreVertical size=IconSize::Sm />
                        </button>
                        <div class="dropdown-menu dropdown-menu-right">
                            <A href=format!("/training/{}", run.id) class="dropdown-item">
                                <IconEye size=IconSize::Sm />
                                <span>"View Details"</span>
                            </A>
                            <Show when=move || is_running>
                                <button class="dropdown-item">
                                    <IconStop size=IconSize::Sm />
                                    <span>"Stop Run"</span>
                                </button>
                            </Show>
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
                <div class="run-meta">
                    <span class="meta-item">
                        <IconBrain size=IconSize::Xs />
                        {run.model_type}
                    </span>
                    <span class="meta-item">
                        <IconClock size=IconSize::Xs />
                        {format_time(&run.started_at)}
                    </span>
                </div>

                // Metrics
                {run.latest_metrics.as_ref().map(|m| {
                    view! {
                        <div class="run-metrics">
                            <div class="metric">
                                <span class="metric-label">"Epoch"</span>
                                <span class="metric-value">{format!("{}/{}", m.epoch, run.config.epochs)}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">"Loss"</span>
                                <span class="metric-value">{format!("{:.4}", m.loss)}</span>
                            </div>
                            {m.accuracy.map(|acc| view! {
                                <div class="metric">
                                    <span class="metric-label">"Accuracy"</span>
                                    <span class="metric-value">{format!("{:.2}%", acc * 100.0)}</span>
                                </div>
                            })}
                        </div>
                    }
                })}

                // Progress bar for running jobs
                <Show when=move || is_running>
                    {run.latest_metrics.as_ref().map(|m| {
                        let progress = (m.epoch as f64 / run.config.epochs as f64) * 100.0;
                        view! {
                            <div class="run-progress">
                                <div class="progress">
                                    <div
                                        class="progress-bar progress-primary"
                                        style=format!("width: {}%", progress)
                                    />
                                </div>
                            </div>
                        }
                    })}
                </Show>

                // Config summary
                <div class="run-config">
                    <span class="config-item">
                        "LR: "{format!("{:.0e}", run.config.learning_rate)}
                    </span>
                    <span class="config-item">
                        "Batch: "{run.config.batch_size}
                    </span>
                    <span class="config-item">
                        {run.config.optimizer.clone()}
                    </span>
                </div>
            </div>

            <div class="card-footer">
                <A href=format!("/training/{}", run.id) class="btn btn-ghost btn-sm">
                    "View Details"
                    <IconArrowRight size=IconSize::Sm />
                </A>
            </div>
        </div>
    }
}

fn format_time(dt: &chrono::DateTime<chrono::Utc>) -> String {
    dt.format("%b %d, %H:%M").to_string()
}
