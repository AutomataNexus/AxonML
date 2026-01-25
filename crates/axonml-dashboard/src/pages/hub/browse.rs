//! Model Hub Browse Page
//!
//! Browse and download pretrained models.

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{icons::*, spinner::*};

/// Hub browse page
#[component]
pub fn HubBrowsePage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (models, set_models) = create_signal::<Vec<PretrainedModel>>(Vec::new());
    let (search, set_search) = create_signal(String::new());
    let (architecture_filter, set_architecture_filter) = create_signal::<Option<String>>(None);
    let (downloading, set_downloading) = create_signal::<Option<String>>(None);

    let state_for_effect = state.clone();

    // Initial fetch
    create_effect(move |_| {
        let state = state_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            match api::hub::list_models(None, None, None).await {
                Ok(data) => set_models.set(data),
                Err(e) => state.toast_error("Error", e.message),
            }
            set_loading.set(false);
        });
    });

    // Get unique architectures for filter
    let architectures = move || {
        let mut archs: Vec<String> = models.get()
            .iter()
            .map(|m| m.architecture.clone())
            .collect();
        archs.sort();
        archs.dedup();
        archs
    };

    // Filtered models
    let filtered_models = move || {
        let search_term = search.get().to_lowercase();
        let arch_filter = architecture_filter.get();

        models.get()
            .into_iter()
            .filter(|m| {
                let matches_search = search_term.is_empty()
                    || m.name.to_lowercase().contains(&search_term)
                    || m.description.to_lowercase().contains(&search_term)
                    || m.architecture.to_lowercase().contains(&search_term);

                let matches_arch = arch_filter.as_ref()
                    .map(|a| m.architecture == *a)
                    .unwrap_or(true);

                matches_search && matches_arch
            })
            .collect::<Vec<_>>()
    };

    view! {
        <div class="page hub-browse-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Model Hub"</h1>
                    <p class="page-subtitle">"Browse and download pretrained models"</p>
                </div>
                <A href="/hub/cache" class="btn btn-secondary">
                    <IconBox size=IconSize::Sm />
                    <span>"View Cache"</span>
                </A>
            </div>

            // Filters
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

                <select
                    class="select"
                    on:change=move |e| {
                        let val = event_target_value(&e);
                        set_architecture_filter.set(if val.is_empty() { None } else { Some(val) });
                    }
                >
                    <option value="">"All Architectures"</option>
                    {move || architectures().into_iter().map(|arch| {
                        view! { <option value={arch.clone()}>{arch}</option> }
                    }).collect_view()}
                </select>
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
                    fallback=|| view! {
                        <div class="empty-state">
                            <IconBox size=IconSize::Xl class="text-muted".to_string() />
                            <h3>"No Models Found"</h3>
                            <p>"Try adjusting your search filters"</p>
                        </div>
                    }
                >
                    <ModelsGrid
                        models=filtered_models
                        downloading=downloading
                        set_downloading=set_downloading
                        set_models=set_models
                    />
                </Show>
            </Show>
        </div>
    }
}

/// Models Grid component
#[component]
fn ModelsGrid<F>(
    models: F,
    downloading: ReadSignal<Option<String>>,
    set_downloading: WriteSignal<Option<String>>,
    set_models: WriteSignal<Vec<PretrainedModel>>,
) -> impl IntoView
where
    F: Fn() -> Vec<PretrainedModel> + 'static,
{
    let state = use_app_state();

    view! {
        <div class="models-grid">
            {move || {
                let current_downloading = downloading.get();
                models().into_iter().map(|model| {
                    let model_name = model.name.clone();
                    let is_downloading = current_downloading.as_ref() == Some(&model_name);

                    let state = state.clone();
                    let name_for_download = model_name.clone();

                    let on_download = move |_| {
                        let state = state.clone();
                        let model_name = name_for_download.clone();
                        set_downloading.set(Some(model_name.clone()));

                        spawn_local(async move {
                            match api::hub::download_model(&model_name, false).await {
                                Ok(response) => {
                                    if response.was_cached {
                                        state.toast_success("Already Cached", &format!("{} is already in cache", model_name));
                                    } else {
                                        state.toast_success("Downloaded", &format!("{} downloaded successfully", model_name));
                                    }
                                    // Refresh to update cached status
                                    if let Ok(data) = api::hub::list_models(None, None, None).await {
                                        set_models.set(data);
                                    }
                                }
                                Err(e) => state.toast_error("Download Failed", e.message),
                            }
                            set_downloading.set(None);
                        });
                    };

                    view! {
                        <PretrainedModelCard
                            model=model
                            downloading=is_downloading
                            on_download=on_download
                        />
                    }
                }).collect_view()
            }}
        </div>
    }
}

/// Pretrained Model Card component
#[component]
fn PretrainedModelCard<F>(
    model: PretrainedModel,
    downloading: bool,
    on_download: F,
) -> impl IntoView
where
    F: Fn(ev::MouseEvent) + 'static,
{
    let name = model.name.clone();
    let description = model.description.clone();
    let architecture = model.architecture.clone();
    let size_mb = model.size_mb;
    let accuracy = model.accuracy;
    let dataset = model.dataset.clone();
    let input_size = model.input_size;
    let num_classes = model.num_classes;
    let num_parameters = model.num_parameters;
    let is_cached = model.is_cached;

    view! {
        <div class="pretrained-model-card card">
            <div class="card-header">
                <div class="model-header-content">
                    <span class="model-name">{name}</span>
                    <span class="model-architecture badge badge-primary">{architecture}</span>
                </div>
                {if is_cached {
                    view! {
                        <span class="badge badge-success">"Cached"</span>
                    }.into_view()
                } else {
                    view! {}.into_view()
                }}
            </div>

            <div class="card-body">
                <p class="model-description">{description}</p>

                <div class="model-stats">
                    <div class="stat-item">
                        <IconActivity size=IconSize::Xs />
                        <span>{format!("{:.1}% accuracy", accuracy)}</span>
                    </div>
                    <div class="stat-item">
                        <IconLayers size=IconSize::Xs />
                        <span>{format_params(num_parameters)}</span>
                    </div>
                    <div class="stat-item">
                        <IconBox size=IconSize::Xs />
                        <span>{format!("{:.1} MB", size_mb)}</span>
                    </div>
                </div>

                <div class="model-details">
                    <div class="detail-row">
                        <span class="label">"Dataset"</span>
                        <span class="value">{dataset}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">"Input Size"</span>
                        <span class="value">{format!("{}x{}", input_size.0, input_size.1)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">"Classes"</span>
                        <span class="value">{num_classes}</span>
                    </div>
                </div>
            </div>

            <div class="card-footer">
                <button
                    class="btn btn-primary btn-block"
                    on:click=on_download
                    disabled=downloading || is_cached
                >
                    {if downloading {
                        view! {
                            <Spinner size=SpinnerSize::Sm />
                            <span>"Downloading..."</span>
                        }.into_view()
                    } else if is_cached {
                        view! {
                            <IconCheck size=IconSize::Sm />
                            <span>"Already Cached"</span>
                        }.into_view()
                    } else {
                        view! {
                            <IconDownload size=IconSize::Sm />
                            <span>"Download"</span>
                        }.into_view()
                    }}
                </button>
            </div>
        </div>
    }
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B params", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M params", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.1}K params", params as f64 / 1_000.0)
    } else {
        format!("{} params", params)
    }
}
