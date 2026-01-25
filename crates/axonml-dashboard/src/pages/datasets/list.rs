//! Datasets List Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{icons::*, spinner::*, modal::*};

/// Datasets list page
#[component]
pub fn DatasetsListPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (datasets, set_datasets) = create_signal::<Vec<Dataset>>(Vec::new());
    let (search, set_search) = create_signal(String::new());
    let delete_modal = create_rw_signal(false);
    let (dataset_to_delete, set_dataset_to_delete) = create_signal::<Option<String>>(None);

    let state_for_effect = state.clone();
    let state_for_refresh = state.clone();
    let state_for_delete = state.clone();

    // Initial fetch
    create_effect(move |_| {
        let state = state_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::datasets::list().await {
                Ok(data) => {
                    set_datasets.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    });

    // Filtered datasets
    let filtered_datasets = move || {
        let search_term = search.get().to_lowercase();
        datasets
            .get()
            .into_iter()
            .filter(|d| {
                search_term.is_empty()
                    || d.name.to_lowercase().contains(&search_term)
                    || d.dataset_type.to_lowercase().contains(&search_term)
                    || d.description.as_ref().map(|desc| desc.to_lowercase().contains(&search_term)).unwrap_or(false)
            })
            .collect::<Vec<_>>()
    };

    // Refresh handler
    let on_refresh = move |_| {
        let state = state_for_refresh.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::datasets::list().await {
                Ok(data) => {
                    set_datasets.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    };

    let delete_dataset = move |_| {
        if let Some(id) = dataset_to_delete.get() {
            let state = state_for_delete.clone();
            spawn_local(async move {
                match api::datasets::delete(&id).await {
                    Ok(_) => {
                        state.toast_success("Deleted", "Dataset deleted successfully");
                        set_loading.set(true);
                        if let Ok(data) = api::datasets::list().await {
                            set_datasets.set(data);
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
        set_dataset_to_delete.set(None);
    };

    view! {
        <div class="page datasets-list-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Datasets"</h1>
                    <p class="page-subtitle">"Manage your training datasets"</p>
                </div>
                <A href="/datasets/upload" class="btn btn-primary">
                    <IconUpload size=IconSize::Sm />
                    <span>"Upload Dataset"</span>
                </A>
            </div>

            // Search bar
            <div class="filters-bar">
                <div class="search-box">
                    <IconSearch size=IconSize::Sm />
                    <input
                        type="text"
                        placeholder="Search datasets..."
                        prop:value=move || search.get()
                        on:input=move |e| set_search.set(event_target_value(&e))
                    />
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
                    <p>"Loading datasets..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                <Show
                    when=move || !filtered_datasets().is_empty()
                    fallback=move || view! {
                        <div class="empty-state">
                            <IconDatabase size=IconSize::Xl class="text-muted".to_string() />
                            <h3>"No Datasets Yet"</h3>
                            <p>"Upload your first dataset to get started"</p>
                            <A href="/datasets/upload" class="btn btn-primary">
                                <IconUpload size=IconSize::Sm />
                                <span>"Upload Dataset"</span>
                            </A>
                        </div>
                    }
                >
                    <div class="datasets-grid">
                        {move || filtered_datasets().into_iter().map(|dataset| {
                            let dataset_id_for_delete = dataset.id.clone();

                            view! {
                                <DatasetCard
                                    dataset=dataset
                                    on_delete=move |_| {
                                        set_dataset_to_delete.set(Some(dataset_id_for_delete.clone()));
                                        delete_modal.set(true);
                                    }
                                />
                            }
                        }).collect_view()}
                    </div>
                </Show>
            </Show>

            // Delete modal
            <ConfirmDialog
                show=delete_modal
                title="Delete Dataset?"
                message="This will permanently delete this dataset. This action cannot be undone."
                confirm_text="Delete".to_string()
                danger=true
                on_confirm=delete_dataset
            />
        </div>
    }
}

/// Dataset card component
#[component]
fn DatasetCard(
    dataset: Dataset,
    #[prop(into)] on_delete: Callback<()>,
) -> impl IntoView {
    // Extract values for display
    let name = dataset.name.clone();
    let dataset_type = dataset.dataset_type.clone();
    let description = dataset.description.clone();
    let num_samples = dataset.num_samples;
    let num_features = dataset.num_features;
    let file_size = dataset.file_size;
    let created_at = dataset.created_at.format("%b %d, %Y").to_string();

    view! {
        <div class="dataset-card card">
            <div class="card-header">
                <div class="dataset-header-content">
                    <span class="dataset-name">{name}</span>
                    <span class="dataset-type badge badge-default">{dataset_type}</span>
                </div>
                <div class="dataset-actions">
                    <div class="dropdown">
                        <button class="btn btn-ghost btn-sm">
                            <IconMoreVertical size=IconSize::Sm />
                        </button>
                        <div class="dropdown-menu dropdown-menu-right">
                            <button class="dropdown-item text-danger" on:click=move |_| on_delete.call(())>
                                <IconTrash size=IconSize::Sm />
                                <span>"Delete"</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card-body">
                {description.map(|d| view! {
                    <p class="dataset-description">{d}</p>
                })}

                <div class="dataset-stats">
                    {num_samples.map(|n| view! {
                        <div class="stat-item">
                            <IconLayers size=IconSize::Sm />
                            <span>{format!("{} samples", n)}</span>
                        </div>
                    })}
                    {num_features.map(|n| view! {
                        <div class="stat-item">
                            <IconCpu size=IconSize::Sm />
                            <span>{format!("{} features", n)}</span>
                        </div>
                    })}
                    <div class="stat-item">
                        <IconBox size=IconSize::Sm />
                        <span>{format_file_size(file_size)}</span>
                    </div>
                </div>

                <div class="dataset-meta">
                    <span class="meta-item">
                        <IconCalendar size=IconSize::Xs />
                        {created_at}
                    </span>
                </div>
            </div>
        </div>
    }
}

fn format_file_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
