//! Notebook Import Page
//!
//! Page for importing Jupyter notebooks (.ipynb files)

use leptos::*;
use leptos_router::*;
use gloo_file::callbacks::FileReader;
use std::rc::Rc;
use std::cell::RefCell;

use crate::api;
use crate::state::use_app_state;
use crate::components::{icons::*, spinner::*};

/// Notebook import page
#[component]
pub fn NotebookImportPage() -> impl IntoView {
    let state = use_app_state();
    let navigate = use_navigate();

    let (loading, set_loading) = create_signal(false);
    let (file_name, set_file_name) = create_signal::<Option<String>>(None);
    let (file_content, set_file_content) = create_signal::<Option<String>>(None);
    let (error, set_error) = create_signal::<Option<String>>(None);

    // Store the file reader to keep it alive
    let reader_ref: Rc<RefCell<Option<FileReader>>> = Rc::new(RefCell::new(None));

    let on_file_select = {
        let reader_ref = reader_ref.clone();
        move |ev: web_sys::Event| {
            set_error.set(None);
            set_file_content.set(None);

            let input: web_sys::HtmlInputElement = event_target(&ev);
            let files = input.files();

            if let Some(files) = files {
                if let Some(file) = files.get(0) {
                    let name = file.name();
                    if !name.ends_with(".ipynb") {
                        set_error.set(Some("Please select a .ipynb file".to_string()));
                        return;
                    }

                    set_file_name.set(Some(name));

                    let gloo_file = gloo_file::File::from(file);
                    let reader = gloo_file::callbacks::read_as_text(&gloo_file, move |result| {
                        match result {
                            Ok(content) => {
                                set_file_content.set(Some(content));
                            }
                            Err(e) => {
                                set_error.set(Some(format!("Failed to read file: {:?}", e)));
                            }
                        }
                    });
                    *reader_ref.borrow_mut() = Some(reader);
                }
            }
        }
    };

    let on_import = {
        let state = state.clone();
        let navigate = navigate.clone();
        move |_| {
            if let Some(content) = file_content.get() {
                let state = state.clone();
                let navigate = navigate.clone();

                set_loading.set(true);
                spawn_local(async move {
                    match api::notebooks::import_notebook(&content, "ipynb").await {
                        Ok(notebook) => {
                            state.toast_success("Imported", "Notebook imported successfully");
                            navigate(&format!("/training/notebooks/{}", notebook.id), Default::default());
                        }
                        Err(e) => {
                            state.toast_error("Import Failed", e.message);
                            set_loading.set(false);
                        }
                    }
                });
            }
        }
    };

    view! {
        <div class="page notebook-import-page">
            <div class="page-header">
                <div class="header-content">
                    <A href="/training/notebooks" class="back-link">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Back to Notebooks"</span>
                    </A>
                    <h1>"Import Jupyter Notebook"</h1>
                    <p class="page-subtitle">"Upload a .ipynb file to import as an AxonML training notebook"</p>
                </div>
            </div>

            <div class="import-container card">
                <div class="card-body">
                    // File drop zone
                    <div class="file-drop-zone">
                        <input
                            type="file"
                            id="notebook-file"
                            accept=".ipynb"
                            class="file-input"
                            on:change=on_file_select
                        />
                        <label for="notebook-file" class="file-label">
                            <IconUpload size=IconSize::Xl />
                            <p class="file-label-text">"Drop a .ipynb file here or click to browse"</p>
                            <p class="file-label-hint">"Supports Jupyter Notebook format"</p>
                        </label>
                    </div>

                    // Error message
                    <Show when=move || error.get().is_some()>
                        <div class="alert alert-danger">
                            {move || error.get().unwrap_or_default()}
                        </div>
                    </Show>

                    // Selected file info
                    <Show when=move || file_name.get().is_some()>
                        <div class="selected-file">
                            <IconFileText size=IconSize::Md />
                            <div class="file-info">
                                <span class="file-name">{move || file_name.get().unwrap_or_default()}</span>
                                <span class="file-status">
                                    {move || if file_content.get().is_some() {
                                        "Ready to import"
                                    } else {
                                        "Reading file..."
                                    }}
                                </span>
                            </div>
                        </div>
                    </Show>

                    // Import button
                    <div class="import-actions">
                        <button
                            class="btn btn-primary btn-lg"
                            on:click=on_import
                            disabled=move || loading.get() || file_content.get().is_none()
                        >
                            {move || if loading.get() {
                                view! { <Spinner size=SpinnerSize::Sm /> }.into_view()
                            } else {
                                view! { <IconUpload size=IconSize::Sm /> }.into_view()
                            }}
                            <span>{move || if loading.get() { "Importing..." } else { "Import Notebook" }}</span>
                        </button>
                    </div>
                </div>
            </div>

            // Info section
            <div class="import-info card">
                <div class="card-header">
                    <h3>"Supported Features"</h3>
                </div>
                <div class="card-body">
                    <ul class="feature-list">
                        <li>
                            <IconCheck size=IconSize::Sm />
                            <span>"Code cells with Python/Rust code"</span>
                        </li>
                        <li>
                            <IconCheck size=IconSize::Sm />
                            <span>"Markdown cells for documentation"</span>
                        </li>
                        <li>
                            <IconCheck size=IconSize::Sm />
                            <span>"Cell outputs (text, images, HTML)"</span>
                        </li>
                        <li>
                            <IconCheck size=IconSize::Sm />
                            <span>"Notebook metadata and kernel info"</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    }
}
