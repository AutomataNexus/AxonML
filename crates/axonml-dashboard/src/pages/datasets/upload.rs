//! Dataset Upload Page

use leptos::*;
use leptos_router::*;
use web_sys::{File, FileList};

use crate::api;
use crate::state::use_app_state;
use crate::components::{icons::*, spinner::*, forms::*, progress::*};

/// Dataset upload page
#[component]
pub fn DatasetUploadPage() -> impl IntoView {
    let state = use_app_state();
    let navigate = use_navigate();

    // Form state
    let dataset_name = create_rw_signal(String::new());
    let dataset_description = create_rw_signal(String::new());
    let dataset_type = create_rw_signal("tabular".to_string());
    let (selected_file, set_selected_file) = create_signal::<Option<File>>(None);
    let (uploading, set_uploading) = create_signal(false);
    let (upload_progress, set_upload_progress) = create_signal(0.0f64);
    let error = create_rw_signal::<Option<String>>(None);

    let dataset_types = vec![
        ("tabular".to_string(), "Tabular (CSV, TSV)".to_string()),
        ("image".to_string(), "Image".to_string()),
        ("text".to_string(), "Text".to_string()),
        ("audio".to_string(), "Audio".to_string()),
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

        if dataset_name.get().trim().is_empty() {
            error.set(Some("Please enter a dataset name".to_string()));
            return;
        }

        let file = match selected_file.get() {
            Some(f) => f,
            None => {
                error.set(Some("Please select a file".to_string()));
                return;
            }
        };

        set_uploading.set(true);
        error.set(None);
        set_upload_progress.set(0.0);

        let state = state.clone();
        let navigate = navigate.clone();
        let name = dataset_name.get();
        let description = if dataset_description.get().is_empty() {
            None
        } else {
            Some(dataset_description.get())
        };
        let dtype = dataset_type.get();

        spawn_local(async move {
            set_upload_progress.set(20.0);

            match api::datasets::upload(file, &name, description.as_deref(), Some(&dtype)).await {
                Ok(_dataset) => {
                    set_upload_progress.set(100.0);
                    state.toast_success("Uploaded", "Dataset uploaded successfully");
                    navigate("/datasets", Default::default());
                }
                Err(e) => {
                    error.set(Some(e.message));
                    set_uploading.set(false);
                }
            }
        });
    };

    view! {
        <div class="page dataset-upload-page">
            <div class="page-header">
                <div class="header-breadcrumb">
                    <A href="/datasets" class="btn btn-ghost btn-sm">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Datasets"</span>
                    </A>
                </div>
                <h1>"Upload Dataset"</h1>
                <p class="page-subtitle">"Upload a new dataset for training"</p>
            </div>

            <form class="upload-form" on:submit=on_submit>
                <Show when=move || error.get().is_some()>
                    <div class="alert alert-error">
                        <IconAlertCircle size=IconSize::Sm />
                        <span>{move || error.get().unwrap_or_default()}</span>
                    </div>
                </Show>

                // Dataset Info
                <div class="card form-section">
                    <div class="card-header">
                        <h2>"Dataset Information"</h2>
                    </div>
                    <div class="card-body">
                        <div class="form-grid">
                            <TextInput
                                value=dataset_name
                                label="Dataset Name"
                                placeholder="e.g., MNIST Training Data"
                                required=true
                            />

                            <div class="form-group">
                                <label class="form-label">"Dataset Type"</label>
                                <div class="select-wrapper">
                                    <select
                                        class="form-select"
                                        on:change=move |e| dataset_type.set(event_target_value(&e))
                                    >
                                        {dataset_types.iter().map(|(value, label)| {
                                            let v = value.clone();
                                            let v_for_cmp = value.clone();
                                            view! {
                                                <option value=v selected=move || dataset_type.get() == v_for_cmp>
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
                            value=dataset_description
                            label="Description"
                            placeholder="Describe your dataset..."
                            rows=3
                        />
                    </div>
                </div>

                // File Upload Section
                <div class="card form-section">
                    <div class="card-header">
                        <h2>"Dataset File"</h2>
                    </div>
                    <div class="card-body">
                        <FileInput
                            on_select=Callback::new(on_file_select)
                            label="Data File"
                            accept=".csv,.tsv,.txt,.json,.jsonl,.parquet,.npy,.npz,.tar,.tar.gz,.zip"
                            helper_text="Supported formats: CSV, TSV, JSON, JSONL, Parquet, NumPy, archives"
                            required=true
                        />

                        // Show selected file info
                        <Show when=move || selected_file.get().is_some()>
                            {move || selected_file.get().map(|f| {
                                view! {
                                    <div class="selected-file">
                                        <IconDatabase size=IconSize::Md />
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

                // Upload Progress
                <Show when=move || uploading.get()>
                    <div class="card upload-progress-card">
                        <div class="card-body">
                            <div class="upload-progress">
                                <Spinner size=SpinnerSize::Md />
                                <div class="progress-info">
                                    <span>"Uploading dataset..."</span>
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
                    <A href="/datasets" class="btn btn-ghost">"Cancel"</A>
                    <button
                        type="submit"
                        class="btn btn-primary"
                        disabled=move || uploading.get() || selected_file.get().is_none() || dataset_name.get().trim().is_empty()
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
