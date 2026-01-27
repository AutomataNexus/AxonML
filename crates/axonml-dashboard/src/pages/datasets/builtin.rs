//! Built-in Datasets Browser Page
//!
//! Browse and prepare built-in datasets from multiple sources.

use leptos::*;

use crate::api;
use crate::components::{icons::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// Built-in datasets browser page
#[component]
pub fn BuiltinDatasetsPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (datasets, set_datasets) = create_signal::<Vec<BuiltinDataset>>(vec![]);
    let (sources, set_sources) = create_signal::<Vec<DatasetSource>>(vec![]);
    let (selected_source, set_selected_source) = create_signal::<Option<String>>(None);
    let (search_query, set_search_query) = create_signal(String::new());
    let (search_results, set_search_results) =
        create_signal::<Option<Vec<DatasetSearchResult>>>(None);
    let (searching, set_searching) = create_signal(false);
    let (selected_dataset, set_selected_dataset) = create_signal::<Option<BuiltinDataset>>(None);
    let (preparing, set_preparing) = create_signal::<Option<String>>(None);

    let state_for_fetch = state.clone();

    // Fetch initial data
    create_effect(move |_| {
        let state = state_for_fetch.clone();
        spawn_local(async move {
            // Fetch sources
            match api::builtin_datasets::list_sources().await {
                Ok(s) => set_sources.set(s),
                Err(e) => state.toast_error("Error", e.message.clone()),
            }

            // Fetch all datasets
            match api::builtin_datasets::list(None).await {
                Ok(d) => set_datasets.set(d),
                Err(e) => state.toast_error("Error", e.message),
            }

            set_loading.set(false);
        });
    });

    // Refetch when source changes
    let state_for_source = state.clone();
    create_effect(move |_| {
        let source = selected_source.get();
        let state = state_for_source.clone();

        set_loading.set(true);
        spawn_local(async move {
            match api::builtin_datasets::list(source.as_deref()).await {
                Ok(d) => set_datasets.set(d),
                Err(e) => state.toast_error("Error", e.message),
            }
            set_loading.set(false);
        });
    });

    view! {
        <div class="page builtin-datasets-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Dataset Library"</h1>
                    <p class="page-subtitle">"Browse built-in datasets and external sources"</p>
                </div>
            </div>

            // Source Filter
            <div class="source-filters">
                <button
                    class={move || if selected_source.get().is_none() { "source-btn active" } else { "source-btn" }}
                    on:click=move |_| set_selected_source.set(None)
                >
                    "All Sources"
                </button>
                {move || sources.get().into_iter().map(|source| {
                    let source_id = source.id.clone();
                    let is_active = selected_source.get().as_ref() == Some(&source_id);

                    view! {
                        <button
                            class={if is_active { "source-btn active" } else { "source-btn" }}
                            on:click={
                                let source_id = source_id.clone();
                                move |_| set_selected_source.set(Some(source_id.clone()))
                            }
                        >
                            {source.name.clone()}
                            <span class="source-count">{source.dataset_count.clone()}</span>
                        </button>
                    }
                }).collect_view()}
            </div>

            // Search Bar
            <SearchBar
                search_query=search_query
                set_search_query=set_search_query
                set_search_results=set_search_results
                searching=searching
                set_searching=set_searching
                selected_source=selected_source
            />

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading datasets..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                // Show search results or main grid
                {move || {
                    if let Some(results) = search_results.get() {
                        view! {
                            <SearchResultsGrid
                                results=results
                                set_selected_dataset=set_selected_dataset
                                preparing=preparing
                                set_preparing=set_preparing
                            />
                        }.into_view()
                    } else {
                        view! {
                            <DatasetsGrid
                                datasets=datasets
                                set_selected_dataset=set_selected_dataset
                                preparing=preparing
                                set_preparing=set_preparing
                            />
                        }.into_view()
                    }
                }}
            </Show>

            // Dataset Detail Modal
            <Show when=move || selected_dataset.get().is_some()>
                <DatasetDetailModal
                    dataset=selected_dataset
                    set_dataset=set_selected_dataset
                    preparing=preparing
                    set_preparing=set_preparing
                />
            </Show>
        </div>
    }
}

/// Search bar component
#[component]
fn SearchBar(
    search_query: ReadSignal<String>,
    set_search_query: WriteSignal<String>,
    set_search_results: WriteSignal<Option<Vec<DatasetSearchResult>>>,
    searching: ReadSignal<bool>,
    set_searching: WriteSignal<bool>,
    selected_source: ReadSignal<Option<String>>,
) -> impl IntoView {
    let state = use_app_state();

    let do_search = {
        let state = state.clone();
        move || {
            let query = search_query.get();
            if query.is_empty() {
                set_search_results.set(None);
                return;
            }

            let source = selected_source.get();
            let state = state.clone();
            set_searching.set(true);

            spawn_local(async move {
                match api::builtin_datasets::search(&query, source.as_deref(), Some(50)).await {
                    Ok(results) => set_search_results.set(Some(results)),
                    Err(e) => state.toast_error("Search Error", e.message),
                }
                set_searching.set(false);
            });
        }
    };

    let do_search_keypress = do_search.clone();
    let do_search_click = do_search;

    let on_clear = move |_| {
        set_search_query.set(String::new());
        set_search_results.set(None);
    };

    view! {
        <div class="search-bar">
            <div class="search-input-wrapper">
                <IconSearch size=IconSize::Sm class="search-icon".to_string() />
                <input
                    type="text"
                    class="input search-input"
                    placeholder="Search datasets..."
                    prop:value=move || search_query.get()
                    on:input=move |ev| set_search_query.set(event_target_value(&ev))
                    on:keypress=move |ev| {
                        if ev.key() == "Enter" {
                            do_search_keypress();
                        }
                    }
                />
                <Show when=move || !search_query.get().is_empty()>
                    <button class="btn-clear" on:click=on_clear>
                        <IconX size=IconSize::Sm />
                    </button>
                </Show>
            </div>
            <button
                class="btn btn-primary"
                on:click=move |_| do_search_click()
                disabled=move || searching.get() || search_query.get().is_empty()
            >
                {move || if searching.get() {
                    view! {
                        <Spinner size=SpinnerSize::Sm />
                    }.into_view()
                } else {
                    view! {
                        <span>"Search"</span>
                    }.into_view()
                }}
            </button>
        </div>
    }
}

/// Datasets grid component
#[component]
fn DatasetsGrid(
    datasets: ReadSignal<Vec<BuiltinDataset>>,
    set_selected_dataset: WriteSignal<Option<BuiltinDataset>>,
    preparing: ReadSignal<Option<String>>,
    set_preparing: WriteSignal<Option<String>>,
) -> impl IntoView {
    let state = use_app_state();

    view! {
        <div class="datasets-grid">
            {move || {
                let ds = datasets.get();
                if ds.is_empty() {
                    view! {
                        <div class="empty-state">
                            <IconDatabase size=IconSize::Lg class="text-muted".to_string() />
                            <p>"No datasets found"</p>
                        </div>
                    }.into_view()
                } else {
                    ds.into_iter().map(|dataset| {
                        let dataset_for_click = dataset.clone();
                        let dataset_id = dataset.id.clone();
                        let is_preparing = preparing.get().as_ref() == Some(&dataset_id);
                        let state = state.clone();

                        let on_prepare = {
                            let dataset_id = dataset_id.clone();
                            move |ev: web_sys::MouseEvent| {
                                ev.stop_propagation();
                                let state = state.clone();
                                let dataset_id = dataset_id.clone();
                                set_preparing.set(Some(dataset_id.clone()));

                                spawn_local(async move {
                                    match api::builtin_datasets::prepare(&dataset_id).await {
                                        Ok(ds) => {
                                            state.toast_success("Ready", format!("{} is ready to use", ds.name));
                                        }
                                        Err(e) => state.toast_error("Error", e.message),
                                    }
                                    set_preparing.set(None);
                                });
                            }
                        };

                        view! {
                            <div
                                class="dataset-card"
                                on:click=move |_| set_selected_dataset.set(Some(dataset_for_click.clone()))
                            >
                                <div class="card-header">
                                    <div class="dataset-icon">
                                        {match dataset.data_type.as_str() {
                                            "image" => view! { <IconImage size=IconSize::Lg /> }.into_view(),
                                            "text" => view! { <IconFileText size=IconSize::Lg /> }.into_view(),
                                            "audio" => view! { <IconVolume size=IconSize::Lg /> }.into_view(),
                                            _ => view! { <IconDatabase size=IconSize::Lg /> }.into_view(),
                                        }}
                                    </div>
                                    <span class={format!("badge badge-{}", dataset.source.clone())}>{dataset.source.clone()}</span>
                                </div>
                                <div class="card-body">
                                    <h4 class="dataset-name">{dataset.name.clone()}</h4>
                                    <p class="dataset-description">{dataset.description.clone()}</p>
                                    <div class="dataset-stats">
                                        <span class="stat">
                                            <span class="stat-value">{format_number(dataset.num_samples)}</span>
                                            <span class="stat-label">"samples"</span>
                                        </span>
                                        <span class="stat">
                                            <span class="stat-value">{dataset.num_classes.to_string()}</span>
                                            <span class="stat-label">"classes"</span>
                                        </span>
                                        <span class="stat">
                                            <span class="stat-value">{format!("{:.1} MB", dataset.size_mb)}</span>
                                            <span class="stat-label">"size"</span>
                                        </span>
                                    </div>
                                    <div class="dataset-tags">
                                        <span class="tag">{dataset.data_type.clone()}</span>
                                        <span class="tag">{dataset.task_type.clone()}</span>
                                    </div>
                                </div>
                                <div class="card-footer">
                                    <button
                                        class="btn btn-primary btn-sm"
                                        on:click=on_prepare
                                        disabled=is_preparing
                                    >
                                        {if is_preparing {
                                            view! {
                                                <Spinner size=SpinnerSize::Sm />
                                                <span>"Preparing..."</span>
                                            }.into_view()
                                        } else {
                                            view! {
                                                <IconDownload size=IconSize::Sm />
                                                <span>"Prepare"</span>
                                            }.into_view()
                                        }}
                                    </button>
                                </div>
                            </div>
                        }
                    }).collect_view()
                }
            }}
        </div>
    }
}

/// Search results grid
#[component]
fn SearchResultsGrid(
    results: Vec<DatasetSearchResult>,
    set_selected_dataset: WriteSignal<Option<BuiltinDataset>>,
    preparing: ReadSignal<Option<String>>,
    set_preparing: WriteSignal<Option<String>>,
) -> impl IntoView {
    let state = use_app_state();

    view! {
        <div class="search-results">
            <h3 class="results-header">{format!("{} results found", results.len())}</h3>
            <div class="results-list">
                {results.into_iter().map(|result| {
                    let result_id = result.id.clone();
                    let is_preparing = preparing.get().as_ref() == Some(&result_id);
                    let state_prepare = state.clone();
                    let state_view = state.clone();

                    let on_prepare = {
                        let result_id = result_id.clone();
                        move |_| {
                            let state = state_prepare.clone();
                            let result_id = result_id.clone();
                            set_preparing.set(Some(result_id.clone()));

                            spawn_local(async move {
                                match api::builtin_datasets::prepare(&result_id).await {
                                    Ok(ds) => {
                                        state.toast_success("Ready", format!("{} is ready to use", ds.name));
                                    }
                                    Err(e) => state.toast_error("Error", e.message),
                                }
                                set_preparing.set(None);
                            });
                        }
                    };

                    let on_view = {
                        let result_id = result_id.clone();
                        move |_| {
                            let result_id = result_id.clone();
                            let state = state_view.clone();
                            spawn_local(async move {
                                match api::builtin_datasets::get(&result_id).await {
                                    Ok(ds) => set_selected_dataset.set(Some(ds)),
                                    Err(e) => state.toast_error("Error", e.message),
                                }
                            });
                        }
                    };

                    view! {
                        <div class="result-item">
                            <div class="result-info">
                                <h4 class="result-name">{result.name.clone()}</h4>
                                <p class="result-description">{result.description.clone()}</p>
                                <div class="result-meta">
                                    <span class="badge">{result.source.clone()}</span>
                                    <span class="meta-item">{result.size.clone()}</span>
                                    <span class="meta-item">{format!("{} downloads", result.download_count)}</span>
                                </div>
                            </div>
                            <div class="result-actions">
                                <button class="btn btn-secondary btn-sm" on:click=on_view>
                                    <IconEye size=IconSize::Sm />
                                    <span>"View"</span>
                                </button>
                                <button
                                    class="btn btn-primary btn-sm"
                                    on:click=on_prepare
                                    disabled=is_preparing
                                >
                                    {if is_preparing {
                                        view! {
                                            <Spinner size=SpinnerSize::Sm />
                                        }.into_view()
                                    } else {
                                        view! {
                                            <IconDownload size=IconSize::Sm />
                                            <span>"Prepare"</span>
                                        }.into_view()
                                    }}
                                </button>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}

/// Dataset detail modal
#[component]
fn DatasetDetailModal(
    dataset: ReadSignal<Option<BuiltinDataset>>,
    set_dataset: WriteSignal<Option<BuiltinDataset>>,
    preparing: ReadSignal<Option<String>>,
    set_preparing: WriteSignal<Option<String>>,
) -> impl IntoView {
    let state = use_app_state();

    view! {
        {move || dataset.get().map(|ds| {
            let dataset_id = ds.id.clone();
            let is_preparing = preparing.get().as_ref() == Some(&dataset_id);
            let state = state.clone();

            let on_prepare = {
                let dataset_id = dataset_id.clone();
                move |_| {
                    let state = state.clone();
                    let dataset_id = dataset_id.clone();
                    set_preparing.set(Some(dataset_id.clone()));

                    spawn_local(async move {
                        match api::builtin_datasets::prepare(&dataset_id).await {
                            Ok(d) => {
                                state.toast_success("Ready", format!("{} is ready to use", d.name));
                            }
                            Err(e) => state.toast_error("Error", e.message),
                        }
                        set_preparing.set(None);
                    });
                }
            };

            view! {
                <div class="modal-overlay" on:click=move |_| set_dataset.set(None)>
                    <div class="modal modal-lg" on:click=move |ev| ev.stop_propagation()>
                        <div class="modal-header">
                            <h3>{ds.name.clone()}</h3>
                            <button class="btn-close" on:click=move |_| set_dataset.set(None)>
                                <IconX size=IconSize::Sm />
                            </button>
                        </div>
                        <div class="modal-body">
                            <p class="dataset-description">{ds.description.clone()}</p>

                            <div class="detail-grid">
                                <div class="detail-item">
                                    <span class="label">"Source"</span>
                                    <span class="value badge">{ds.source.clone()}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="label">"Data Type"</span>
                                    <span class="value">{ds.data_type.clone()}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="label">"Task Type"</span>
                                    <span class="value">{ds.task_type.clone()}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="label">"Samples"</span>
                                    <span class="value">{format_number(ds.num_samples)}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="label">"Features"</span>
                                    <span class="value">{ds.num_features.to_string()}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="label">"Classes"</span>
                                    <span class="value">{ds.num_classes.to_string()}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="label">"Size"</span>
                                    <span class="value">{format!("{:.2} MB", ds.size_mb)}</span>
                                </div>
                            </div>

                            {ds.loading_code.clone().map(|code| view! {
                                <div class="code-section">
                                    <h4>"Loading Code"</h4>
                                    <pre class="code-block"><code>{code}</code></pre>
                                </div>
                            })}
                        </div>
                        <div class="modal-footer">
                            <button
                                class="btn btn-secondary"
                                on:click=move |_| set_dataset.set(None)
                            >
                                "Close"
                            </button>
                            <button
                                class="btn btn-primary"
                                on:click=on_prepare
                                disabled=is_preparing
                            >
                                {if is_preparing {
                                    view! {
                                        <Spinner size=SpinnerSize::Sm />
                                        <span>"Preparing..."</span>
                                    }.into_view()
                                } else {
                                    view! {
                                        <IconDownload size=IconSize::Sm />
                                        <span>"Prepare Dataset"</span>
                                    }.into_view()
                                }}
                            </button>
                        </div>
                    </div>
                </div>
            }
        })}
    }
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
