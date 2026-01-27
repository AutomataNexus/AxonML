//! Model Upload Page

use leptos::*;
use leptos_router::*;
use web_sys::{File, FileList};

use crate::api;
use crate::components::{forms::*, icons::*, progress::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// Model upload page
#[component]
pub fn ModelUploadPage() -> impl IntoView {
    let params = use_params_map();
    let state = use_app_state();
    let navigate = use_navigate();

    // Check if uploading to existing model
    let existing_model_id = move || params.get().get("id").cloned();
    let is_new_model = move || existing_model_id().is_none();

    // Form state
    let model_name = create_rw_signal(String::new());
    let model_description = create_rw_signal(String::new());
    let model_type = create_rw_signal("neural_network".to_string());
    let (selected_file, set_selected_file) = create_signal::<Option<File>>(None);
    let training_run_id = create_rw_signal(String::new());
    let (uploading, set_uploading) = create_signal(false);
    let (upload_progress, set_upload_progress) = create_signal(0.0f64);
    let error = create_rw_signal::<Option<String>>(None);

    // Load existing model data if editing
    let (existing_model, set_existing_model) = create_signal::<Option<Model>>(None);
    create_effect(move |_| {
        if let Some(id) = existing_model_id() {
            spawn_local(async move {
                if let Ok(model) = api::models::get(&id).await {
                    set_existing_model.set(Some(model));
                }
            });
        }
    });

    let model_types = vec![
        ("neural_network".to_string(), "Neural Network".to_string()),
        ("transformer".to_string(), "Transformer".to_string()),
        ("cnn".to_string(), "CNN".to_string()),
        ("rnn".to_string(), "RNN".to_string()),
        ("lstm".to_string(), "LSTM".to_string()),
        ("custom".to_string(), "Custom".to_string()),
    ];

    let on_file_select = move |files: FileList| {
        if files.length() > 0 {
            if let Some(file) = files.get(0) {
                set_selected_file.set(Some(file));
            }
        }
    };

    let on_submit = move |e: web_sys::SubmitEvent| {
        e.prevent_default();

        let file = match selected_file.get() {
            Some(f) => f,
            None => {
                error.set(Some("Please select a model file".to_string()));
                return;
            }
        };

        set_uploading.set(true);
        error.set(None);
        set_upload_progress.set(0.0);

        // Clone state and navigate for async block
        let state = state.clone();
        let navigate = navigate.clone();

        // If new model, create it first
        if is_new_model() {
            if model_name.get().trim().is_empty() {
                error.set(Some("Please enter a model name".to_string()));
                set_uploading.set(false);
                return;
            }

            let request = CreateModelRequest {
                name: model_name.get(),
                description: if model_description.get().is_empty() {
                    None
                } else {
                    Some(model_description.get())
                },
                model_type: model_type.get(),
            };

            spawn_local(async move {
                // Create model
                match api::models::create(&request).await {
                    Ok(model) => {
                        set_upload_progress.set(20.0);

                        // Upload version
                        let run_id = if training_run_id.get().is_empty() {
                            None
                        } else {
                            Some(training_run_id.get())
                        };

                        match api::models::upload_version(&model.id, file, run_id.as_deref()).await
                        {
                            Ok(_version) => {
                                set_upload_progress.set(100.0);
                                state.toast_success("Uploaded", "Model uploaded successfully");
                                navigate(&format!("/models/{}", model.id), Default::default());
                            }
                            Err(e) => {
                                error.set(Some(e.message));
                                set_uploading.set(false);
                            }
                        }
                    }
                    Err(e) => {
                        error.set(Some(e.message));
                        set_uploading.set(false);
                    }
                }
            });
        } else {
            // Upload to existing model
            let model_id = existing_model_id().unwrap();
            let run_id = if training_run_id.get().is_empty() {
                None
            } else {
                Some(training_run_id.get())
            };

            spawn_local(async move {
                set_upload_progress.set(20.0);

                match api::models::upload_version(&model_id, file, run_id.as_deref()).await {
                    Ok(_version) => {
                        set_upload_progress.set(100.0);
                        state.toast_success("Uploaded", "New version uploaded successfully");
                        navigate(&format!("/models/{}", model_id), Default::default());
                    }
                    Err(e) => {
                        error.set(Some(e.message));
                        set_uploading.set(false);
                    }
                }
            });
        }
    };

    view! {
        <div class="page model-upload-page">
            <div class="page-header">
                <div class="header-breadcrumb">
                    <A href="/models" class="btn btn-ghost btn-sm">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Models"</span>
                    </A>
                </div>
                <h1>
                    {move || if is_new_model() {
                        "Upload New Model"
                    } else {
                        "Upload New Version"
                    }}
                </h1>
                <p class="page-subtitle">
                    {move || if is_new_model() {
                        "Create a new model and upload its first version".to_string()
                    } else {
                        existing_model.get().map(|m| format!("Add a new version to {}", m.name)).unwrap_or_default()
                    }}
                </p>
            </div>

            <form class="upload-form" on:submit=on_submit>
                <Show when=move || error.get().is_some()>
                    <div class="alert alert-error">
                        <IconAlertCircle size=IconSize::Sm />
                        <span>{move || error.get().unwrap_or_default()}</span>
                    </div>
                </Show>

                // Model Info (only for new models)
                <Show when=is_new_model>
                    <div class="card form-section">
                        <div class="card-header">
                            <h2>"Model Information"</h2>
                        </div>
                        <div class="card-body">
                            <div class="form-grid">
                                <TextInput
                                    value=model_name
                                    label="Model Name"
                                    placeholder="e.g., ResNet-50 ImageNet"
                                    required=true
                                />

                                <div class="form-group">
                                    <label class="form-label">"Model Type"</label>
                                    <div class="select-wrapper">
                                        <select
                                            class="form-select"
                                            on:change=move |e| model_type.set(event_target_value(&e))
                                        >
                                            {model_types.iter().map(|(value, label)| {
                                                let v = value.clone();
                                                let v_for_cmp = value.clone();
                                                view! {
                                                    <option value=v selected=move || model_type.get() == v_for_cmp>
                                                        {label.clone()}
                                                    </option>
                                                }
                                            }).collect_view()}
                                        </select>
                                        <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
                                    </div>
                                </div>
                            </div>

                            <TextArea
                                value=model_description
                                label="Description"
                                placeholder="Describe your model..."
                                rows=3
                            />
                        </div>
                    </div>
                </Show>

                // File Upload Section
                <div class="card form-section">
                    <div class="card-header">
                        <h2>"Model File"</h2>
                    </div>
                    <div class="card-body">
                        <FileInput
                            on_select=Callback::new(on_file_select)
                            label="Model File"
                            accept=".py,.pt,.pth,.onnx,.h5,.pb,.safetensors,.bin,.pkl,.npy,.npz,.json,.yaml,.yml"
                            helper_text="Supported formats: Python (.py), PyTorch (.pt, .pth), ONNX (.onnx), TensorFlow (.h5, .pb), SafeTensors, NumPy, configs"
                            required=true
                        />

                        // Show selected file info
                        <Show when=move || selected_file.get().is_some()>
                            {move || selected_file.get().map(|f| {
                                view! {
                                    <div class="selected-file">
                                        <IconBox size=IconSize::Md />
                                        <div class="file-info">
                                            <span class="file-name">{f.name()}</span>
                                            <span class="file-size">{format_file_size(f.size() as u64)}</span>
                                        </div>
                                        <button
                                            type="button"
                                            class="btn btn-ghost btn-sm"
                                            on:click=move |_| set_selected_file.set(None)
                                        >
                                            <IconX size=IconSize::Sm />
                                        </button>
                                    </div>
                                }
                            })}
                        </Show>
                    </div>
                </div>

                // Optional: Link to training run
                <div class="card form-section">
                    <div class="card-header">
                        <h2>"Link to Training Run"</h2>
                        <span class="badge badge-default">"Optional"</span>
                    </div>
                    <div class="card-body">
                        <TextInput
                            value=training_run_id
                            label="Training Run ID"
                            placeholder="e.g., run_abc123..."
                            helper_text="Link this model version to a training run for traceability"
                        />
                    </div>
                </div>

                // Upload Progress
                <Show when=move || uploading.get()>
                    <div class="card upload-progress-card">
                        <div class="card-body">
                            <div class="upload-progress">
                                <Spinner size=SpinnerSize::Md />
                                <div class="progress-info">
                                    <span>"Uploading model..."</span>
                                    <ProgressBar
                                        value=upload_progress
                                        show_label=true
                                        variant=ProgressVariant::Primary
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </Show>

                // Form Actions
                <div class="form-actions">
                    <A href="/models" class="btn btn-ghost">"Cancel"</A>
                    <button
                        type="submit"
                        class="btn btn-primary"
                        disabled=move || uploading.get() || selected_file.get().is_none()
                    >
                        <Show when=move || uploading.get() fallback=|| {
                            view! {
                                <IconUpload size=IconSize::Sm />
                                <span>"Upload"</span>
                            }
                        }>
                            <Spinner size=SpinnerSize::Sm />
                            <span>"Uploading..."</span>
                        </Show>
                    </button>
                </div>
            </form>
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
