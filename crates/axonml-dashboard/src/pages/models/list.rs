//! Models Registry List Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{icons::*, spinner::*, modal::*};

/// Models list page
#[component]
pub fn ModelsListPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (models, set_models) = create_signal::<Vec<Model>>(Vec::new());
    let (search, set_search) = create_signal(String::new());
    let delete_modal = create_rw_signal(false);
    let (model_to_delete, set_model_to_delete) = create_signal::<Option<String>>(None);

    // Clone state for different closures
    let state_for_effect = state.clone();
    let state_for_refresh = state.clone();
    let state_for_delete = state.clone();

    // Initial fetch
    create_effect(move |_| {
        let state = state_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::models::list().await {
                Ok(data) => {
                    set_models.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    });

    // Filtered models
    let filtered_models = move || {
        let search_term = search.get().to_lowercase();
        models
            .get()
            .into_iter()
            .filter(|m| {
                search_term.is_empty()
                    || m.name.to_lowercase().contains(&search_term)
                    || m.model_type.to_lowercase().contains(&search_term)
                    || m.description.as_ref().map(|d| d.to_lowercase().contains(&search_term)).unwrap_or(false)
            })
            .collect::<Vec<_>>()
    };

    // Refresh handler
    let on_refresh = move |_| {
        let state = state_for_refresh.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::models::list().await {
                Ok(data) => {
                    set_models.set(data);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    };

    let delete_model = move |_| {
        if let Some(id) = model_to_delete.get() {
            let state = state_for_delete.clone();
            spawn_local(async move {
                match api::models::delete(&id).await {
                    Ok(_) => {
                        state.toast_success("Deleted", "Model deleted successfully");
                        // Inline the fetch logic to avoid FnOnce issue
                        set_loading.set(true);
                        if let Ok(data) = api::models::list().await {
                            set_models.set(data);
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
        set_model_to_delete.set(None);
    };

    view! {
        <div class="page models-list-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Model Registry"</h1>
                    <p class="page-subtitle">"Manage and version your trained models"</p>
                </div>
                <A href="/models/upload" class="btn btn-primary">
                    <IconUpload size=IconSize::Sm />
                    <span>"Upload Model"</span>
                </A>
            </div>

            // Search bar
            <div class="filters-bar">
                <div class="search-box">
                    <IconSearch size=IconSize::Sm />
                    <input
                        type="text"
                        placeholder="Search models..."
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
                    <p>"Loading models..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                <Show
                    when=move || !filtered_models().is_empty()
                    fallback=move || view! {
                        <div class="empty-state">
                            <IconBox size=IconSize::Xl class="text-muted".to_string() />
                            <h3>"No Models Yet"</h3>
                            <p>"Upload your first model to get started"</p>
                            <A href="/models/upload" class="btn btn-primary">
                                <IconUpload size=IconSize::Sm />
                                <span>"Upload Model"</span>
                            </A>
                        </div>
                    }
                >
                    <div class="models-grid">
                        {move || filtered_models().into_iter().map(|model| {
                            let model_id_for_delete = model.id.clone();

                            view! {
                                <ModelCard
                                    model=model
                                    on_delete=move |_| {
                                        set_model_to_delete.set(Some(model_id_for_delete.clone()));
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
                title="Delete Model?"
                message="This will delete the model and all its versions. This action cannot be undone."
                confirm_text="Delete".to_string()
                danger=true
                on_confirm=delete_model
            />
        </div>
    }
}

/// Model card component
#[component]
fn ModelCard(
    model: Model,
    #[prop(into)] on_delete: Callback<()>,
) -> impl IntoView {
    view! {
        <div class="model-card card">
            <div class="card-header">
                <div class="model-header-content">
                    <A href=format!("/models/{}", model.id) class="model-name">
                        {model.name.clone()}
                    </A>
                    <span class="model-type badge badge-default">{model.model_type.clone()}</span>
                </div>
                <div class="model-actions">
                    <div class="dropdown">
                        <button class="btn btn-ghost btn-sm">
                            <IconMoreVertical size=IconSize::Sm />
                        </button>
                        <div class="dropdown-menu dropdown-menu-right">
                            <A href=format!("/models/{}", model.id) class="dropdown-item">
                                <IconEye size=IconSize::Sm />
                                <span>"View Details"</span>
                            </A>
                            <A href=format!("/models/{}/upload", model.id) class="dropdown-item">
                                <IconUpload size=IconSize::Sm />
                                <span>"Upload Version"</span>
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
                {model.description.as_ref().map(|d| view! {
                    <p class="model-description">{d.clone()}</p>
                })}

                <div class="model-stats">
                    <div class="stat-item">
                        <IconLayers size=IconSize::Sm />
                        <span>{format!("{} versions", model.version_count)}</span>
                    </div>
                    {model.latest_version.map(|v| view! {
                        <div class="stat-item">
                            <span class="badge badge-primary">{format!("v{}", v)}</span>
                        </div>
                    })}
                </div>

                <div class="model-meta">
                    <span class="meta-item">
                        <IconCalendar size=IconSize::Xs />
                        {model.created_at.format("%b %d, %Y").to_string()}
                    </span>
                </div>
            </div>

            <div class="card-footer">
                <A href=format!("/models/{}", model.id) class="btn btn-ghost btn-sm">
                    "View Details"
                    <IconArrowRight size=IconSize::Sm />
                </A>
            </div>
        </div>
    }
}
