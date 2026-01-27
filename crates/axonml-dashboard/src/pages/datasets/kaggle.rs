//! Kaggle Integration Page
//!
//! Search, browse, and download datasets from Kaggle.

use leptos::*;

use crate::api;
use crate::components::{icons::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// Kaggle integration page
#[component]
pub fn KagglePage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (kaggle_status, set_kaggle_status) = create_signal::<Option<KaggleStatusResponse>>(None);
    let (search_query, set_search_query) = create_signal(String::new());
    let (search_results, set_search_results) = create_signal::<Option<KaggleSearchResponse>>(None);
    let (downloaded, set_downloaded) = create_signal::<Vec<KaggleLocalDataset>>(vec![]);
    let (searching, set_searching) = create_signal(false);
    let (downloading, set_downloading) = create_signal::<Option<String>>(None);
    let (active_tab, set_active_tab) = create_signal("search".to_string());
    let (show_credentials_modal, set_show_credentials_modal) = create_signal(false);

    let state_for_fetch = state.clone();

    // Fetch Kaggle status on mount
    create_effect(move |_| {
        let state = state_for_fetch.clone();
        spawn_local(async move {
            match api::kaggle::get_status().await {
                Ok(status) => set_kaggle_status.set(Some(status)),
                Err(e) => state.toast_error("Error", e.message),
            }

            match api::kaggle::list_downloaded().await {
                Ok(d) => set_downloaded.set(d),
                Err(_) => {} // Not critical
            }

            set_loading.set(false);
        });
    });

    view! {
        <div class="page kaggle-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Kaggle Datasets"</h1>
                    <p class="page-subtitle">"Search, browse, and download datasets from Kaggle"</p>
                </div>
                <div class="header-actions">
                    <button
                        class="btn btn-secondary"
                        on:click=move |_| set_show_credentials_modal.set(true)
                    >
                        <IconSettings size=IconSize::Sm />
                        <span>"Configure"</span>
                    </button>
                </div>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading Kaggle status..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                // Status banner
                {move || kaggle_status.get().map(|status| {
                    if status.configured {
                        view! {
                            <div class="status-banner status-success">
                                <IconCheck size=IconSize::Sm />
                                <span>"Connected as "{status.username.unwrap_or_default()}</span>
                            </div>
                        }.into_view()
                    } else {
                        view! {
                            <div class="status-banner status-warning">
                                <IconAlert size=IconSize::Sm />
                                <span>"Kaggle credentials not configured. "</span>
                                <button
                                    class="btn-link"
                                    on:click=move |_| set_show_credentials_modal.set(true)
                                >
                                    "Configure now"
                                </button>
                            </div>
                        }.into_view()
                    }
                })}

                // Tab Navigation
                <div class="tab-nav">
                    <button
                        class={move || if active_tab.get() == "search" { "tab active" } else { "tab" }}
                        on:click=move |_| set_active_tab.set("search".to_string())
                    >
                        <IconSearch size=IconSize::Sm />
                        <span>"Search"</span>
                    </button>
                    <button
                        class={move || if active_tab.get() == "downloaded" { "tab active" } else { "tab" }}
                        on:click=move |_| set_active_tab.set("downloaded".to_string())
                    >
                        <IconDownload size=IconSize::Sm />
                        <span>"Downloaded"</span>
                    </button>
                </div>

                // Tab Content
                <div class="tab-content">
                    <Show when=move || active_tab.get() == "search">
                        <KaggleSearchTab
                            search_query=search_query
                            set_search_query=set_search_query
                            search_results=search_results
                            set_search_results=set_search_results
                            searching=searching
                            set_searching=set_searching
                            downloading=downloading
                            set_downloading=set_downloading
                            set_downloaded=set_downloaded
                            kaggle_configured=move || kaggle_status.get().map(|s| s.configured).unwrap_or(false)
                        />
                    </Show>

                    <Show when=move || active_tab.get() == "downloaded">
                        <DownloadedTab downloaded=downloaded />
                    </Show>
                </div>
            </Show>

            // Credentials Modal
            <Show when=move || show_credentials_modal.get()>
                <CredentialsModal
                    set_show_modal=set_show_credentials_modal
                    set_status=set_kaggle_status
                />
            </Show>
        </div>
    }
}

/// Kaggle search tab
#[component]
fn KaggleSearchTab(
    search_query: ReadSignal<String>,
    set_search_query: WriteSignal<String>,
    search_results: ReadSignal<Option<KaggleSearchResponse>>,
    set_search_results: WriteSignal<Option<KaggleSearchResponse>>,
    searching: ReadSignal<bool>,
    set_searching: WriteSignal<bool>,
    downloading: ReadSignal<Option<String>>,
    set_downloading: WriteSignal<Option<String>>,
    set_downloaded: WriteSignal<Vec<KaggleLocalDataset>>,
    kaggle_configured: impl Fn() -> bool + 'static + Copy,
) -> impl IntoView {
    let state = use_app_state();
    let state_for_search = state.clone();
    let state_for_results = state.clone();

    let do_search = {
        let state = state_for_search;
        move || {
            let query = search_query.get();
            if query.is_empty() {
                return;
            }

            let state = state.clone();
            set_searching.set(true);

            spawn_local(async move {
                match api::kaggle::search(&query, Some(20), None).await {
                    Ok(results) => set_search_results.set(Some(results)),
                    Err(e) => state.toast_error("Search Error", e.message),
                }
                set_searching.set(false);
            });
        }
    };

    let do_search_keypress = do_search.clone();
    let do_search_click = do_search;

    view! {
        <div class="search-tab">
            // Search Form
            <div class="search-form">
                <input
                    type="text"
                    class="input search-input"
                    placeholder="Search Kaggle datasets..."
                    prop:value=move || search_query.get()
                    on:input=move |ev| set_search_query.set(event_target_value(&ev))
                    on:keypress=move |ev| {
                        if ev.key() == "Enter" {
                            do_search_keypress();
                        }
                    }
                    disabled=move || !kaggle_configured()
                />
                <button
                    class="btn btn-primary"
                    on:click=move |_| do_search_click()
                    disabled=move || searching.get() || search_query.get().is_empty() || !kaggle_configured()
                >
                    {move || if searching.get() {
                        view! {
                            <Spinner size=SpinnerSize::Sm />
                            <span>"Searching..."</span>
                        }.into_view()
                    } else {
                        view! {
                            <IconSearch size=IconSize::Sm />
                            <span>"Search"</span>
                        }.into_view()
                    }}
                </button>
            </div>

            // Results
            <div class="search-results">
                {move || search_results.get().map(|results| {
                    if results.datasets.is_empty() {
                        view! {
                            <div class="empty-state">
                                <IconSearch size=IconSize::Lg class="text-muted".to_string() />
                                <p>"No datasets found"</p>
                            </div>
                        }.into_view()
                    } else {
                        let datasets = results.datasets.clone();
                        view! {
                            <div class="results-grid">
                                {datasets.into_iter().map(|dataset| {
                                    let dataset_ref = dataset.ref_name.clone();
                                    let downloading_this = downloading.get().as_ref() == Some(&dataset_ref);
                                    let state = state_for_results.clone();

                                    let on_download = {
                                        let dataset_ref = dataset_ref.clone();
                                        move |_| {
                                            let state = state.clone();
                                            let dataset_ref = dataset_ref.clone();
                                            set_downloading.set(Some(dataset_ref.clone()));

                                            spawn_local(async move {
                                                let request = KaggleDownloadRequest {
                                                    dataset_ref: dataset_ref.clone(),
                                                    output_dir: None,
                                                };

                                                match api::kaggle::download(&request).await {
                                                    Ok(_) => {
                                                        state.toast_success("Download Complete", format!("Downloaded {}", dataset_ref));
                                                        // Refresh downloaded list
                                                        if let Ok(d) = api::kaggle::list_downloaded().await {
                                                            set_downloaded.set(d);
                                                        }
                                                    }
                                                    Err(e) => state.toast_error("Download Error", e.message),
                                                }
                                                set_downloading.set(None);
                                            });
                                        }
                                    };

                                    view! {
                                        <div class="dataset-card">
                                            <div class="card-header">
                                                <h4 class="dataset-title">{dataset.title.clone()}</h4>
                                                <span class="dataset-ref">{dataset.ref_name.clone()}</span>
                                            </div>
                                            <div class="card-body">
                                                {dataset.description.clone().map(|d| view! {
                                                    <p class="dataset-description">{d}</p>
                                                })}
                                                <div class="dataset-meta">
                                                    <span class="meta-item">
                                                        <IconDatabase size=IconSize::Xs />
                                                        {dataset.size.clone()}
                                                    </span>
                                                    <span class="meta-item">
                                                        <IconDownload size=IconSize::Xs />
                                                        {format!("{} downloads", dataset.download_count)}
                                                    </span>
                                                    <span class="meta-item">
                                                        <IconStar size=IconSize::Xs />
                                                        {format!("{} votes", dataset.vote_count)}
                                                    </span>
                                                </div>
                                            </div>
                                            <div class="card-footer">
                                                <button
                                                    class="btn btn-primary btn-sm"
                                                    on:click=on_download
                                                    disabled=move || downloading_this
                                                >
                                                    {if downloading_this {
                                                        view! {
                                                            <Spinner size=SpinnerSize::Sm />
                                                            <span>"Downloading..."</span>
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
                                }).collect_view()}
                            </div>
                        }.into_view()
                    }
                }).unwrap_or_else(|| view! {
                    <div class="empty-state">
                        <IconSearch size=IconSize::Lg class="text-muted".to_string() />
                        <p>"Search for Kaggle datasets"</p>
                        <p class="text-muted">"Enter a search term and click Search"</p>
                    </div>
                }.into_view())}
            </div>
        </div>
    }
}

/// Downloaded datasets tab
#[component]
fn DownloadedTab(downloaded: ReadSignal<Vec<KaggleLocalDataset>>) -> impl IntoView {
    view! {
        <div class="downloaded-tab">
            {move || {
                let datasets = downloaded.get();
                if datasets.is_empty() {
                    view! {
                        <div class="empty-state">
                            <IconDownload size=IconSize::Lg class="text-muted".to_string() />
                            <p>"No downloaded datasets"</p>
                            <p class="text-muted">"Downloaded datasets will appear here"</p>
                        </div>
                    }.into_view()
                } else {
                    view! {
                        <div class="downloaded-list">
                            {datasets.into_iter().map(|dataset| view! {
                                <div class="downloaded-item">
                                    <div class="item-icon">
                                        <IconFolder size=IconSize::Md />
                                    </div>
                                    <div class="item-info">
                                        <span class="item-name">{dataset.filename.clone()}</span>
                                        <span class="item-path">{dataset.path.clone()}</span>
                                    </div>
                                    <div class="item-size">
                                        {format!("{:.2} MB", dataset.size_mb)}
                                    </div>
                                </div>
                            }).collect_view()}
                        </div>
                    }.into_view()
                }
            }}
        </div>
    }
}

/// Credentials modal
#[component]
fn CredentialsModal(
    set_show_modal: WriteSignal<bool>,
    set_status: WriteSignal<Option<KaggleStatusResponse>>,
) -> impl IntoView {
    let state = use_app_state();

    let (username, set_username) = create_signal(String::new());
    let (key, set_key) = create_signal(String::new());
    let (saving, set_saving) = create_signal(false);

    let on_save = move |_| {
        let username_val = username.get();
        let key_val = key.get();

        if username_val.is_empty() || key_val.is_empty() {
            state.toast_error("Error", "Username and API key are required");
            return;
        }

        let state = state.clone();
        set_saving.set(true);

        spawn_local(async move {
            let credentials = KaggleCredentials {
                username: username_val,
                key: key_val,
            };

            match api::kaggle::save_credentials(&credentials).await {
                Ok(status) => {
                    set_status.set(Some(status));
                    state.toast_success("Success", "Kaggle credentials saved");
                    set_show_modal.set(false);
                }
                Err(e) => state.toast_error("Error", e.message),
            }
            set_saving.set(false);
        });
    };

    view! {
        <div class="modal-overlay" on:click=move |_| set_show_modal.set(false)>
            <div class="modal" on:click=move |ev| ev.stop_propagation()>
                <div class="modal-header">
                    <h3>"Kaggle API Credentials"</h3>
                    <button class="btn-close" on:click=move |_| set_show_modal.set(false)>
                        <IconX size=IconSize::Sm />
                    </button>
                </div>
                <div class="modal-body">
                    <p class="modal-description">
                        "Enter your Kaggle API credentials. You can find these in your "
                        <a href="https://www.kaggle.com/account" target="_blank">"Kaggle account settings"</a>
                        "."
                    </p>

                    <div class="form-group">
                        <label>"Username"</label>
                        <input
                            type="text"
                            class="input"
                            placeholder="your_kaggle_username"
                            prop:value=move || username.get()
                            on:input=move |ev| set_username.set(event_target_value(&ev))
                        />
                    </div>

                    <div class="form-group">
                        <label>"API Key"</label>
                        <input
                            type="password"
                            class="input"
                            placeholder="Your Kaggle API key"
                            prop:value=move || key.get()
                            on:input=move |ev| set_key.set(event_target_value(&ev))
                        />
                    </div>
                </div>
                <div class="modal-footer">
                    <button
                        class="btn btn-secondary"
                        on:click=move |_| set_show_modal.set(false)
                    >
                        "Cancel"
                    </button>
                    <button
                        class="btn btn-primary"
                        on:click=on_save
                        disabled=move || saving.get() || username.get().is_empty() || key.get().is_empty()
                    >
                        {move || if saving.get() {
                            view! {
                                <Spinner size=SpinnerSize::Sm />
                                <span>"Saving..."</span>
                            }.into_view()
                        } else {
                            view! {
                                <IconCheck size=IconSize::Sm />
                                <span>"Save"</span>
                            }.into_view()
                        }}
                    </button>
                </div>
            </div>
        </div>
    }
}
