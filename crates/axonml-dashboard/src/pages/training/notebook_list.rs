//! Training Notebooks List Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{icons::*, spinner::*, modal::*, StatusBadge};

/// Notebooks list page
#[component]
pub fn NotebookListPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (notebooks, set_notebooks) = create_signal::<Vec<TrainingNotebook>>(Vec::new());
    let (search, set_search) = create_signal(String::new());
    let delete_modal = create_rw_signal(false);
    let (notebook_to_delete, set_notebook_to_delete) = create_signal::<Option<String>>(None);

    let state_for_effect = state.clone();
    let state_for_refresh = state.clone();
    let state_for_delete = state.clone();

    // Fetch notebooks on mount
    create_effect(move |_| {
        let state = state_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::notebooks::list_notebooks().await {
                Ok(data) => {
                    set_notebooks.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    });

    // Filtered notebooks based on search
    let filtered_notebooks = move || {
        let search_term = search.get().to_lowercase();
        notebooks.get()
            .into_iter()
            .filter(|n| {
                search_term.is_empty()
                    || n.name.to_lowercase().contains(&search_term)
                    || n.description.as_ref().map(|d| d.to_lowercase().contains(&search_term)).unwrap_or(false)
            })
            .collect::<Vec<_>>()
    };

    // Refresh handler
    let on_refresh = move |_| {
        let state = state_for_refresh.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::notebooks::list_notebooks().await {
                Ok(data) => {
                    set_notebooks.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    };

    let delete_notebook = move |_| {
        if let Some(id) = notebook_to_delete.get() {
            let state = state_for_delete.clone();
            spawn_local(async move {
                match api::notebooks::delete_notebook(&id).await {
                    Ok(_) => {
                        state.toast_success("Deleted", "Notebook deleted successfully");
                        set_loading.set(true);
                        if let Ok(data) = api::notebooks::list_notebooks().await {
                            set_notebooks.set(data);
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
        set_notebook_to_delete.set(None);
    };

    view! {
        <div class="page notebooks-list-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Training Notebooks"</h1>
                    <p class="page-subtitle">"Create and manage training experiments with notebook-style interface"</p>
                </div>
                <div class="header-actions">
                    <A href="/training/notebooks/import" class="btn btn-ghost">
                        <IconUpload size=IconSize::Sm />
                        <span>"Import"</span>
                    </A>
                    <A href="/training/notebooks/new" class="btn btn-primary">
                        <IconPlus size=IconSize::Sm />
                        <span>"New Notebook"</span>
                    </A>
                </div>
            </div>

            // Filters
            <div class="filters-bar">
                <div class="search-box">
                    <IconSearch size=IconSize::Sm />
                    <input
                        type="text"
                        placeholder="Search notebooks..."
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
                    <p>"Loading notebooks..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                <Show
                    when=move || !filtered_notebooks().is_empty()
                    fallback=move || view! {
                        <div class="empty-state">
                            <IconFileText size=IconSize::Xl class="text-muted".to_string() />
                            <h3>"No Training Notebooks"</h3>
                            <p>"Create your first notebook to start training with an interactive interface"</p>
                            <A href="/training/notebooks/new" class="btn btn-primary">
                                <IconPlus size=IconSize::Sm />
                                <span>"Create Notebook"</span>
                            </A>
                        </div>
                    }
                >
                    <div class="notebooks-grid">
                        {move || filtered_notebooks().into_iter().map(|notebook| {
                            let notebook_id_for_delete = notebook.id.clone();

                            view! {
                                <NotebookCard
                                    notebook=notebook
                                    on_delete=move |_| {
                                        set_notebook_to_delete.set(Some(notebook_id_for_delete.clone()));
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
                title="Delete Notebook?"
                message="This action cannot be undone. All cells and checkpoints will be permanently deleted."
                confirm_text="Delete".to_string()
                danger=true
                on_confirm=delete_notebook
            />
        </div>
    }
}

/// Notebook card component
#[component]
fn NotebookCard(
    notebook: TrainingNotebook,
    #[prop(into)] on_delete: Callback<()>,
) -> impl IntoView {
    let status_class = notebook.status.color_class();
    let notebook_id = notebook.id.clone();
    let notebook_id_for_edit = notebook.id.clone();
    let notebook_id_for_export = notebook.id.clone();
    let notebook_id_for_link = notebook.id.clone();
    let cell_count = notebook.cells.len();
    let tags = notebook.metadata.tags.clone();
    let has_tags = !tags.is_empty();
    let description = notebook.description.clone();
    let name = notebook.name.clone();
    let updated_at = notebook.updated_at;

    view! {
        <div class="notebook-card card">
            <div class="card-header">
                <div class="notebook-header-content">
                    <A href=format!("/training/notebooks/{}", notebook_id) class="notebook-name">
                        {name}
                    </A>
                    <StatusBadge status=notebook.status.as_str().to_string() class=status_class.to_string() />
                </div>
                <div class="notebook-actions">
                    <div class="dropdown">
                        <button class="btn btn-ghost btn-sm">
                            <IconMoreVertical size=IconSize::Sm />
                        </button>
                        <div class="dropdown-menu dropdown-menu-right">
                            <A href=format!("/training/notebooks/{}", notebook_id_for_edit) class="dropdown-item">
                                <IconEdit size=IconSize::Sm />
                                <span>"Open"</span>
                            </A>
                            <A href=format!("/training/notebooks/{}/export?format=ipynb", notebook_id_for_export) class="dropdown-item">
                                <IconDownload size=IconSize::Sm />
                                <span>"Export"</span>
                            </A>
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
                {description.map(|desc| view! {
                    <p class="notebook-description">{desc}</p>
                })}

                <div class="notebook-meta">
                    <span class="meta-item">
                        <IconCode size=IconSize::Xs />
                        {format!("{} cells", cell_count)}
                    </span>
                    <span class="meta-item">
                        <IconClock size=IconSize::Xs />
                        {format_time(&updated_at)}
                    </span>
                </div>

                // Tags/metadata
                <Show when=move || has_tags>
                    <div class="notebook-tags">
                        {tags.iter().map(|tag| view! {
                            <span class="tag">{tag}</span>
                        }).collect_view()}
                    </div>
                </Show>
            </div>

            <div class="card-footer">
                <A href=format!("/training/notebooks/{}", notebook_id_for_link) class="btn btn-primary btn-sm">
                    "Open Notebook"
                    <IconArrowRight size=IconSize::Sm />
                </A>
            </div>
        </div>
    }
}

fn format_time(dt: &chrono::DateTime<chrono::Utc>) -> String {
    dt.format("%b %d, %H:%M").to_string()
}
