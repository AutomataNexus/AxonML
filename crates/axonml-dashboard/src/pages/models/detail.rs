//! Model Detail Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::components::{forms::*, icons::*, modal::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// Model detail page
#[component]
pub fn ModelDetailPage() -> impl IntoView {
    let params = use_params_map();
    let state = use_app_state();
    let navigate = use_navigate();

    let model_id = move || params.get().get("id").cloned().unwrap_or_default();

    let (loading, set_loading) = create_signal(true);
    let (model, set_model) = create_signal::<Option<Model>>(None);
    let (versions, set_versions) = create_signal::<Vec<ModelVersion>>(Vec::new());
    let delete_modal = create_rw_signal(false);
    let delete_version_modal = create_rw_signal(false);
    let (version_to_delete, set_version_to_delete) = create_signal::<Option<u32>>(None);
    let deploy_modal = create_rw_signal(false);
    let (version_to_deploy, set_version_to_deploy) = create_signal::<Option<u32>>(None);
    let endpoint_name = create_rw_signal(String::new());
    let (deploying, set_deploying) = create_signal(false);

    // Clone state and navigate for different closures
    let state_for_effect = state.clone();
    let navigate_for_effect = navigate.clone();
    let state_for_delete = state.clone();
    let navigate_for_delete = navigate.clone();
    let state_for_delete_version = state.clone();
    let state_for_deploy = state.clone();
    let navigate_for_deploy = navigate.clone();

    // Fetch model data on ID changes
    create_effect(move |_| {
        let id = model_id();
        if id.is_empty() {
            return;
        }

        let state = state_for_effect.clone();
        let navigate = navigate_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            // Fetch model
            match api::models::get(&id).await {
                Ok(data) => {
                    set_model.set(Some(data));
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                    navigate("/models", Default::default());
                    return;
                }
            }

            // Fetch versions
            if let Ok(vers) = api::models::list_versions(&id).await {
                set_versions.set(vers);
            }

            set_loading.set(false);
        });
    });

    // Delete model
    let delete_model = move |_| {
        let id = model_id();
        let state = state_for_delete.clone();
        let navigate = navigate_for_delete.clone();
        spawn_local(async move {
            match api::models::delete(&id).await {
                Ok(_) => {
                    state.toast_success("Deleted", "Model deleted successfully");
                    navigate("/models", Default::default());
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
        });
        delete_modal.set(false);
    };

    // Delete version
    let delete_version = move |_| {
        if let Some(version) = version_to_delete.get() {
            let id = model_id();
            let state = state_for_delete_version.clone();
            spawn_local(async move {
                match api::models::delete_version(&id, version).await {
                    Ok(_) => {
                        state.toast_success("Deleted", "Version deleted successfully");
                        // Inline fetch to avoid capturing fetch_data closure
                        set_loading.set(true);
                        if let Ok(vers) = api::models::list_versions(&id).await {
                            set_versions.set(vers);
                        }
                        set_loading.set(false);
                    }
                    Err(e) => {
                        state.toast_error("Error", e.message);
                    }
                }
            });
        }
        delete_version_modal.set(false);
        set_version_to_delete.set(None);
    };

    // Deploy version
    let deploy_version = move |_| {
        let name = endpoint_name.get();
        let state = state_for_deploy.clone();
        let navigate = navigate_for_deploy.clone();
        if name.is_empty() {
            state.toast_error("Error", "Please enter an endpoint name");
            return;
        }

        if let Some(version) = version_to_deploy.get() {
            let id = model_id();
            set_deploying.set(true);

            spawn_local(async move {
                match api::models::deploy_version(&id, version, &name).await {
                    Ok(endpoint) => {
                        state.toast_success(
                            "Deployed",
                            format!("Model deployed to endpoint: {}", endpoint.name),
                        );
                        navigate(
                            &format!("/inference/endpoints/{}", endpoint.id),
                            Default::default(),
                        );
                    }
                    Err(e) => {
                        state.toast_error("Error", e.message);
                        set_deploying.set(false);
                    }
                }
            });
        }
    };

    // Store closures for use in modals
    let delete_model_stored = store_value(delete_model);
    let delete_version_stored = store_value(delete_version);
    let deploy_version_stored = store_value(deploy_version);

    view! {
        <div class="page model-detail-page">
            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading model..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get() && model.get().is_some()>
                {move || model.get().map(|m| {
                    // Store m.name for use in nested closures to avoid FnOnce
                    let model_name_for_deploy = store_value(m.name.clone());

                    view! {
                        // Header
                        <div class="page-header">
                            <div class="header-breadcrumb">
                                <A href="/models" class="btn btn-ghost btn-sm">
                                    <IconArrowLeft size=IconSize::Sm />
                                    <span>"Models"</span>
                                </A>
                            </div>
                            <div class="header-content">
                                <div class="header-title-row">
                                    <h1>{m.name.clone()}</h1>
                                    <span class="badge badge-default">{m.model_type.clone()}</span>
                                </div>
                                {m.description.as_ref().map(|d| view! {
                                    <p class="page-subtitle">{d.clone()}</p>
                                })}
                            </div>
                            <div class="header-actions">
                                <A href=format!("/models/{}/upload", m.id) class="btn btn-primary">
                                    <IconUpload size=IconSize::Sm />
                                    <span>"Upload Version"</span>
                                </A>
                                <button class="btn btn-danger-outline" on:click=move |_| delete_modal.set(true)>
                                    <IconTrash size=IconSize::Sm />
                                </button>
                            </div>
                        </div>

                        // Stats
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-icon"><IconLayers size=IconSize::Md /></div>
                                <div class="stat-content">
                                    <span class="stat-value">{m.version_count}</span>
                                    <span class="stat-label">"Versions"</span>
                                </div>
                            </div>
                            {m.latest_version.map(|v| view! {
                                <div class="stat-card">
                                    <div class="stat-icon"><IconBox size=IconSize::Md /></div>
                                    <div class="stat-content">
                                        <span class="stat-value">{format!("v{}", v)}</span>
                                        <span class="stat-label">"Latest"</span>
                                    </div>
                                </div>
                            })}
                            <div class="stat-card">
                                <div class="stat-icon"><IconCalendar size=IconSize::Md /></div>
                                <div class="stat-content">
                                    <span class="stat-value">{m.created_at.format("%b %d").to_string()}</span>
                                    <span class="stat-label">"Created"</span>
                                </div>
                            </div>
                        </div>

                        // Versions Table
                        <div class="card">
                            <div class="card-header">
                                <h2>"Versions"</h2>
                            </div>
                            <div class="card-body">
                                <Show
                                    when=move || !versions.get().is_empty()
                                    fallback=|| view! {
                                        <div class="empty-state">
                                            <IconLayers size=IconSize::Xl class="text-muted".to_string() />
                                            <p>"No versions uploaded yet"</p>
                                        </div>
                                    }
                                >
                                    <table class="data-table">
                                        <thead>
                                            <tr>
                                                <th>"Version"</th>
                                                <th>"Size"</th>
                                                <th>"Training Run"</th>
                                                <th>"Created"</th>
                                                <th>"Actions"</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {move || versions.get().into_iter().map(|v| {
                                                let version = v.version;
                                                let version_for_deploy = v.version;
                                                let version_for_delete = v.version;
                                                let model_id_for_download = model_id();
                                                let training_run_id = v.training_run_id.clone();
                                                let file_size = v.file_size;
                                                let created_at = v.created_at;

                                                view! {
                                                    <tr>
                                                        <td>
                                                            <span class="badge badge-primary">{format!("v{}", version)}</span>
                                                        </td>
                                                        <td>{format_file_size(file_size)}</td>
                                                        <td>
                                                            {training_run_id.as_ref().map(|id| {
                                                                let id_clone = id.clone();
                                                                let id_short = id.chars().take(8).collect::<String>();
                                                                view! {
                                                                    <A href=format!("/training/{}", id_clone) class="link">
                                                                        {id_short}
                                                                    </A>
                                                                }
                                                            }).unwrap_or_else(|| view! { <span class="text-muted">"-"</span> }.into_view())}
                                                        </td>
                                                        <td>{created_at.format("%b %d, %H:%M").to_string()}</td>
                                                        <td>
                                                            <div class="action-buttons">
                                                                <a
                                                                    href=api::models::download_url(&model_id_for_download, version)
                                                                    class="btn btn-ghost btn-sm"
                                                                    download
                                                                >
                                                                    <IconDownload size=IconSize::Sm />
                                                                </a>
                                                                <button
                                                                    class="btn btn-primary btn-sm"
                                                                    on:click=move |_| {
                                                                        set_version_to_deploy.set(Some(version_for_deploy));
                                                                        let model_name = model_name_for_deploy.get_value();
                                                                        endpoint_name.set(format!("{}-v{}", model_name.to_lowercase().replace(' ', "-"), version_for_deploy));
                                                                        deploy_modal.set(true);
                                                                    }
                                                                >
                                                                    <IconServer size=IconSize::Sm />
                                                                    <span>"Deploy"</span>
                                                                </button>
                                                                <button
                                                                    class="btn btn-ghost btn-sm text-danger"
                                                                    on:click=move |_| {
                                                                        set_version_to_delete.set(Some(version_for_delete));
                                                                        delete_version_modal.set(true);
                                                                    }
                                                                >
                                                                    <IconTrash size=IconSize::Sm />
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                }
                                            }).collect_view()}
                                        </tbody>
                                    </table>
                                </Show>
                            </div>
                        </div>

                        // Delete Model Modal
                        <ConfirmDialog
                            show=delete_modal
                            title="Delete Model?"
                            message="This will delete the model and all its versions. This action cannot be undone."
                            confirm_text="Delete".to_string()
                            danger=true
                            on_confirm=move |_| delete_model_stored.with_value(|f| f(()))
                        />

                        // Delete Version Modal
                        <ConfirmDialog
                            show=delete_version_modal
                            title="Delete Version?"
                            message="This version will be permanently deleted. Any endpoints using this version will stop working."
                            confirm_text="Delete".to_string()
                            danger=true
                            on_confirm=move |_| delete_version_stored.with_value(|f| f(()))
                        />

                        // Deploy Modal
                        <Modal
                            show=deploy_modal
                            title="Deploy Model Version"
                            size=ModalSize::Small
                            footer=std::rc::Rc::new(move || view! {
                                <button class="btn btn-ghost" on:click=move |_| deploy_modal.set(false)>
                                    "Cancel"
                                </button>
                                <button
                                    class="btn btn-primary"
                                    disabled=move || deploying.get() || endpoint_name.get().is_empty()
                                    on:click=move |e| deploy_version_stored.with_value(|f| f(e))
                                >
                                    <Show when=move || deploying.get() fallback=|| "Deploy">
                                        <Spinner size=SpinnerSize::Sm />
                                        <span>"Deploying..."</span>
                                    </Show>
                                </button>
                            }.into_view().into())
                        >
                            <div class="deploy-form">
                                <p class="deploy-info">
                                    "Create an inference endpoint for "
                                    <strong>{format!("v{}", version_to_deploy.get().unwrap_or(0))}</strong>
                                </p>
                                <TextInput
                                    value=endpoint_name
                                    label="Endpoint Name"
                                    placeholder="my-model-endpoint"
                                    helper_text="Used in the API URL: /api/inference/predict/{name}"
                                    required=true
                                />
                            </div>
                        </Modal>
                    }
                })}
            </Show>
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
