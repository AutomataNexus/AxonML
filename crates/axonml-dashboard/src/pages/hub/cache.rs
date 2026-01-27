//! Model Hub Cache Page
//!
//! View and manage cached pretrained models.

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::components::{icons::*, modal::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// Hub cache page
#[component]
pub fn HubCachePage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (cache_info, set_cache_info) = create_signal::<Option<CacheInfo>>(None);
    let clear_all_modal = create_rw_signal(false);
    let (model_to_clear, set_model_to_clear) = create_signal::<Option<String>>(None);
    let clear_model_modal = create_rw_signal(false);

    let state_for_effect = state.clone();
    let state_for_clear_all = state.clone();
    let state_for_clear_model = state.clone();

    // Initial fetch
    create_effect(move |_| {
        let state = state_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::hub::get_cache_info().await {
                Ok(info) => set_cache_info.set(Some(info)),
                Err(e) => state.toast_error("Error", e.message),
            }
            set_loading.set(false);
        });
    });

    // Clear all cache
    let clear_all = move |_| {
        let state = state_for_clear_all.clone();
        spawn_local(async move {
            match api::hub::clear_cache(None).await {
                Ok(_) => {
                    state.toast_success("Cache Cleared", "All cached models have been removed");
                    if let Ok(info) = api::hub::get_cache_info().await {
                        set_cache_info.set(Some(info));
                    }
                }
                Err(e) => state.toast_error("Error", e.message),
            }
        });
        clear_all_modal.set(false);
    };

    // Clear single model
    let clear_model = move |_| {
        if let Some(name) = model_to_clear.get() {
            let state = state_for_clear_model.clone();
            spawn_local(async move {
                match api::hub::clear_cache(Some(&name)).await {
                    Ok(_) => {
                        state.toast_success(
                            "Model Removed",
                            &format!("{} removed from cache", name),
                        );
                        if let Ok(info) = api::hub::get_cache_info().await {
                            set_cache_info.set(Some(info));
                        }
                    }
                    Err(e) => state.toast_error("Error", e.message),
                }
            });
        }
        clear_model_modal.set(false);
        set_model_to_clear.set(None);
    };

    let has_cached_models = move || {
        cache_info
            .get()
            .map(|c| !c.models.is_empty())
            .unwrap_or(false)
    };

    view! {
        <div class="page hub-cache-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Model Cache"</h1>
                    <p class="page-subtitle">"Manage downloaded pretrained models"</p>
                </div>
                <div class="header-actions">
                    <A href="/hub" class="btn btn-secondary">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Back to Hub"</span>
                    </A>
                    <button
                        class="btn btn-danger"
                        on:click=move |_| clear_all_modal.set(true)
                        disabled=move || !has_cached_models()
                    >
                        <IconTrash size=IconSize::Sm />
                        <span>"Clear All"</span>
                    </button>
                </div>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading cache info..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                <CacheContent
                    cache_info=cache_info
                    set_model_to_clear=set_model_to_clear
                    clear_model_modal=clear_model_modal
                />
            </Show>

            // Clear all confirmation modal
            <ConfirmDialog
                show=clear_all_modal
                title="Clear All Cache?"
                message="This will remove all cached pretrained models. You will need to download them again."
                confirm_text="Clear All".to_string()
                danger=true
                on_confirm=clear_all
            />

            // Clear single model modal
            <ConfirmDialog
                show=clear_model_modal
                title="Remove from Cache?"
                message="This will remove this model from the cache. You will need to download it again."
                confirm_text="Remove".to_string()
                danger=true
                on_confirm=clear_model
            />
        </div>
    }
}

/// Cache Content component
#[component]
fn CacheContent(
    cache_info: ReadSignal<Option<CacheInfo>>,
    set_model_to_clear: WriteSignal<Option<String>>,
    clear_model_modal: RwSignal<bool>,
) -> impl IntoView {
    view! {
        {move || cache_info.get().map(|info| {
            let cache_dir = info.cache_directory.clone();
            let total_size = info.total_size_bytes;
            let total_models = info.total_models;
            let models = info.models.clone();
            let models_empty = models.is_empty();

            view! {
                <CacheContentInner
                    cache_dir=cache_dir
                    total_size=total_size
                    total_models=total_models
                    models=models
                    models_empty=models_empty
                    set_model_to_clear=set_model_to_clear
                    clear_model_modal=clear_model_modal
                />
            }
        })}
    }
}

/// Cache Content Inner component
#[component]
fn CacheContentInner(
    cache_dir: String,
    total_size: u64,
    total_models: usize,
    models: Vec<CachedModel>,
    models_empty: bool,
    set_model_to_clear: WriteSignal<Option<String>>,
    clear_model_modal: RwSignal<bool>,
) -> impl IntoView {
    view! {
        <div class="cache-summary card">
            <div class="card-body">
                <div class="summary-stats">
                    <div class="stat">
                        <span class="stat-value">{total_models}</span>
                        <span class="stat-label">"Models Cached"</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value">{format_bytes(total_size)}</span>
                        <span class="stat-label">"Total Size"</span>
                    </div>
                </div>
                <div class="cache-path">
                    <span class="label">"Cache Directory:"</span>
                    <code>{cache_dir}</code>
                </div>
            </div>
        </div>

        {if models_empty {
            view! {
                <div class="empty-state">
                    <IconBox size=IconSize::Xl class="text-muted".to_string() />
                    <h3>"No Cached Models"</h3>
                    <p>"Download models from the hub to cache them locally"</p>
                    <A href="/hub" class="btn btn-primary">
                        <IconDownload size=IconSize::Sm />
                        <span>"Browse Models"</span>
                    </A>
                </div>
            }.into_view()
        } else {
            view! {
                <div class="cached-models-list">
                    {models.into_iter().map(|model| {
                        let model_name = model.name.clone();
                        let model_name_for_click = model.name.clone();
                        let model_path = model.path.clone();
                        let model_size = model.size_bytes;

                        view! {
                            <div class="cached-model-row">
                                <div class="model-info">
                                    <IconBox size=IconSize::Md />
                                    <div class="model-details">
                                        <span class="model-name">{model_name}</span>
                                        <span class="model-path text-muted">{model_path}</span>
                                    </div>
                                </div>
                                <div class="model-size">
                                    {format_bytes(model_size)}
                                </div>
                                <button
                                    class="btn btn-ghost btn-sm text-danger"
                                    on:click=move |_| {
                                        set_model_to_clear.set(Some(model_name_for_click.clone()));
                                        clear_model_modal.set(true);
                                    }
                                >
                                    <IconTrash size=IconSize::Sm />
                                </button>
                            </div>
                        }
                    }).collect_view()}
                </div>
            }.into_view()
        }}
    }
}

fn format_bytes(bytes: u64) -> String {
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
