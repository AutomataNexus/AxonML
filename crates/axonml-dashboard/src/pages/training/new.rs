//! New Training Run Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{icons::*, spinner::*, forms::*};

/// New training run form page
#[component]
pub fn NewTrainingPage() -> impl IntoView {
    let state = use_app_state();
    let navigate = use_navigate();

    // Form state
    let name = create_rw_signal(String::new());
    let model_type = create_rw_signal("neural_network".to_string());
    let selected_model_id = create_rw_signal::<Option<String>>(None);
    let selected_version_id = create_rw_signal::<Option<String>>(None);
    let selected_dataset_id = create_rw_signal::<Option<String>>(None);
    let learning_rate = create_rw_signal(0.001f64);
    let batch_size = create_rw_signal(32f64);
    let epochs = create_rw_signal(100f64);
    let optimizer = create_rw_signal("adam".to_string());
    let loss_function = create_rw_signal("cross_entropy".to_string());
    let (loading, set_loading) = create_signal(false);
    let error = create_rw_signal::<Option<String>>(None);

    // Load available models
    let (models, set_models) = create_signal::<Vec<Model>>(vec![]);
    let (versions, set_versions) = create_signal::<Vec<ModelVersion>>(vec![]);
    let (models_loading, set_models_loading) = create_signal(true);

    // Load available datasets
    let (datasets, set_datasets) = create_signal::<Vec<Dataset>>(vec![]);
    let (datasets_loading, set_datasets_loading) = create_signal(true);

    // Fetch models and datasets on mount
    create_effect(move |_| {
        spawn_local(async move {
            match api::models::list().await {
                Ok(m) => set_models.set(m),
                Err(_) => {}
            }
            set_models_loading.set(false);
        });
    });

    create_effect(move |_| {
        spawn_local(async move {
            match api::datasets::list().await {
                Ok(d) => set_datasets.set(d),
                Err(_) => {}
            }
            set_datasets_loading.set(false);
        });
    });

    // Fetch versions when model changes
    create_effect(move |_| {
        if let Some(model_id) = selected_model_id.get() {
            spawn_local(async move {
                match api::models::get_with_versions(&model_id).await {
                    Ok(mwv) => {
                        // Auto-select latest version before setting versions
                        if let Some(latest) = mwv.model.latest_version {
                            if let Some(v) = mwv.versions.iter().find(|v| v.version == latest) {
                                selected_version_id.set(Some(v.id.clone()));
                            }
                        }
                        set_versions.set(mwv.versions);
                    }
                    Err(_) => set_versions.set(vec![]),
                }
            });
        } else {
            set_versions.set(vec![]);
            selected_version_id.set(None);
        }
    });

    // Store option lists for use in closures
    let model_types = store_value(vec![
        ("neural_network".to_string(), "Neural Network".to_string()),
        ("transformer".to_string(), "Transformer".to_string()),
        ("cnn".to_string(), "Convolutional Neural Network".to_string()),
        ("rnn".to_string(), "Recurrent Neural Network".to_string()),
        ("lstm".to_string(), "LSTM".to_string()),
        ("gpt".to_string(), "GPT-style LLM".to_string()),
        ("vae".to_string(), "Variational Autoencoder".to_string()),
        ("gan".to_string(), "GAN".to_string()),
        ("custom".to_string(), "Custom".to_string()),
    ]);

    let optimizers = store_value(vec![
        ("adam".to_string(), "Adam".to_string()),
        ("adamw".to_string(), "AdamW".to_string()),
        ("sgd".to_string(), "SGD".to_string()),
        ("rmsprop".to_string(), "RMSprop".to_string()),
        ("adagrad".to_string(), "Adagrad".to_string()),
    ]);

    let loss_functions = store_value(vec![
        ("cross_entropy".to_string(), "Cross Entropy".to_string()),
        ("mse".to_string(), "Mean Squared Error".to_string()),
        ("mae".to_string(), "Mean Absolute Error".to_string()),
        ("binary_cross_entropy".to_string(), "Binary Cross Entropy".to_string()),
        ("huber".to_string(), "Huber Loss".to_string()),
    ]);

    let on_submit = move |e: web_sys::SubmitEvent| {
        e.prevent_default();

        if name.get().trim().is_empty() {
            error.set(Some("Please enter a run name".to_string()));
            return;
        }

        set_loading.set(true);
        error.set(None);

        let request = CreateRunRequest {
            name: name.get(),
            model_type: model_type.get(),
            model_version_id: selected_version_id.get(),
            dataset_id: selected_dataset_id.get(),
            config: TrainingConfig {
                learning_rate: learning_rate.get(),
                batch_size: batch_size.get() as u32,
                epochs: epochs.get() as u32,
                optimizer: optimizer.get(),
                loss_function: loss_function.get(),
                extra: std::collections::HashMap::new(),
            },
        };

        let state = state.clone();
        let navigate = navigate.clone();
        spawn_local(async move {
            match api::training::create_run(&request).await {
                Ok(run) => {
                    state.toast_success("Created", "Training run started successfully");
                    navigate(&format!("/training/{}", run.id), Default::default());
                }
                Err(e) => {
                    error.set(Some(e.message));
                    set_loading.set(false);
                }
            }
        });
    };

    view! {
        <div class="page new-training-page">
            <div class="page-header">
                <div class="header-breadcrumb">
                    <A href="/training" class="btn btn-ghost btn-sm">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Training Runs"</span>
                    </A>
                </div>
                <h1>"New Training Run"</h1>
                <p class="page-subtitle">"Configure and start a new model training experiment"</p>
            </div>

            <form class="training-form" on:submit=on_submit>
                <Show when=move || error.get().is_some()>
                    <div class="alert alert-error">
                        <IconAlertCircle size=IconSize::Sm />
                        <span>{move || error.get().unwrap_or_default()}</span>
                    </div>
                </Show>

                <div class="form-sections">
                    // Model Selection Section
                    <div class="card form-section">
                        <div class="card-header">
                            <h2>"Select Model"</h2>
                        </div>
                        <div class="card-body">
                            <Show when=move || models_loading.get()>
                                <div class="loading-models">
                                    <Spinner size=SpinnerSize::Sm />
                                    <span>"Loading models..."</span>
                                </div>
                            </Show>
                            <Show when=move || !models_loading.get()>
                                <div class="form-grid">
                                    <div class="form-group">
                                        <label class="form-label">"Model" <span class="required">"*"</span></label>
                                        <div class="select-wrapper">
                                            <select
                                                class="form-select"
                                                required=true
                                                on:change=move |e| {
                                                    let val = event_target_value(&e);
                                                    if val.is_empty() {
                                                        selected_model_id.set(None);
                                                    } else {
                                                        selected_model_id.set(Some(val));
                                                    }
                                                }
                                            >
                                                <option value="" selected=move || selected_model_id.get().is_none()>
                                                    "-- Select a model --"
                                                </option>
                                                {move || models.get().into_iter().map(|m| {
                                                    let id = m.id.clone();
                                                    let id_for_check = m.id.clone();
                                                    view! {
                                                        <option value=id selected=move || selected_model_id.get() == Some(id_for_check.clone())>
                                                            {m.name}
                                                        </option>
                                                    }
                                                }).collect_view()}
                                            </select>
                                            <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
                                        </div>
                                        <p class="form-helper">"Select the model you want to train"</p>
                                    </div>

                                    <Show when=move || selected_model_id.get().is_some() && !versions.get().is_empty()>
                                        <div class="form-group">
                                            <label class="form-label">"Version"</label>
                                            <div class="select-wrapper">
                                                <select
                                                    class="form-select"
                                                    on:change=move |e| {
                                                        let val = event_target_value(&e);
                                                        if val.is_empty() {
                                                            selected_version_id.set(None);
                                                        } else {
                                                            selected_version_id.set(Some(val));
                                                        }
                                                    }
                                                >
                                                    {move || versions.get().into_iter().map(|v| {
                                                        let id = v.id.clone();
                                                        let id_for_check = v.id.clone();
                                                        view! {
                                                            <option value=id selected=move || selected_version_id.get() == Some(id_for_check.clone())>
                                                                {format!("v{}", v.version)}
                                                            </option>
                                                        }
                                                    }).collect_view()}
                                                </select>
                                                <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
                                            </div>
                                            <p class="form-helper">"Select which version to train"</p>
                                        </div>
                                    </Show>
                                </div>
                                <Show when=move || models.get().is_empty()>
                                    <div class="empty-models">
                                        <IconBox size=IconSize::Lg />
                                        <p>"No models found. "</p>
                                        <A href="/models/upload" class="btn btn-primary btn-sm">
                                            "Upload a Model"
                                        </A>
                                    </div>
                                </Show>
                            </Show>
                        </div>
                    </div>

                    // Dataset Selection Section
                    <div class="card form-section">
                        <div class="card-header">
                            <h2>"Select Dataset"</h2>
                        </div>
                        <div class="card-body">
                            <Show when=move || datasets_loading.get()>
                                <div class="loading-models">
                                    <Spinner size=SpinnerSize::Sm />
                                    <span>"Loading datasets..."</span>
                                </div>
                            </Show>
                            <Show when=move || !datasets_loading.get()>
                                <div class="form-grid">
                                    <div class="form-group">
                                        <label class="form-label">"Dataset" <span class="required">"*"</span></label>
                                        <div class="select-wrapper">
                                            <select
                                                class="form-select"
                                                required=true
                                                on:change=move |e| {
                                                    let val = event_target_value(&e);
                                                    if val.is_empty() {
                                                        selected_dataset_id.set(None);
                                                    } else {
                                                        selected_dataset_id.set(Some(val));
                                                    }
                                                }
                                            >
                                                <option value="" selected=move || selected_dataset_id.get().is_none()>
                                                    "-- Select a dataset --"
                                                </option>
                                                {move || datasets.get().into_iter().map(|d| {
                                                    let id = d.id.clone();
                                                    let id_for_check = d.id.clone();
                                                    let info = format!("{} ({} samples)", d.name, d.num_samples.unwrap_or(0));
                                                    view! {
                                                        <option value=id selected=move || selected_dataset_id.get() == Some(id_for_check.clone())>
                                                            {info}
                                                        </option>
                                                    }
                                                }).collect_view()}
                                            </select>
                                            <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
                                        </div>
                                        <p class="form-helper">"Select the dataset to use for training"</p>
                                    </div>
                                </div>
                                <Show when=move || datasets.get().is_empty()>
                                    <div class="empty-models">
                                        <IconDatabase size=IconSize::Lg />
                                        <p>"No datasets found. "</p>
                                        <A href="/datasets/upload" class="btn btn-primary btn-sm">
                                            "Upload a Dataset"
                                        </A>
                                    </div>
                                </Show>
                            </Show>
                        </div>
                    </div>

                    // Basic Info Section
                    <div class="card form-section">
                        <div class="card-header">
                            <h2>"Run Configuration"</h2>
                        </div>
                        <div class="card-body">
                            <div class="form-grid">
                                <TextInput
                                    value=name
                                    label="Run Name"
                                    placeholder="e.g., ResNet-50 Experiment 1"
                                    helper_text="A descriptive name for this training run"
                                    required=true
                                />

                                <div class="form-group">
                                    <label class="form-label">"Model Type"</label>
                                    <div class="select-wrapper">
                                        <select
                                            class="form-select"
                                            on:change=move |e| model_type.set(event_target_value(&e))
                                        >
                                            {model_types.with_value(|types| types.iter().map(|(value, label)| {
                                                let v = value.clone();
                                                let val_for_check = value.clone();
                                                view! {
                                                    <option value=v selected=move || model_type.get() == val_for_check>
                                                        {label.clone()}
                                                    </option>
                                                }
                                            }).collect_view())}
                                        </select>
                                        <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    // Hyperparameters Section
                    <div class="card form-section">
                        <div class="card-header">
                            <h2>"Hyperparameters"</h2>
                        </div>
                        <div class="card-body">
                            <div class="form-grid form-grid-3">
                                <div class="form-group">
                                    <label class="form-label">"Learning Rate"</label>
                                    <input
                                        type="number"
                                        class="form-input"
                                        step="any"
                                        min="0.0000001"
                                        max="1"
                                        prop:value=move || learning_rate.get()
                                        on:input=move |e| {
                                            if let Ok(v) = event_target_value(&e).parse() {
                                                learning_rate.set(v);
                                            }
                                        }
                                    />
                                    <p class="form-helper">"Initial learning rate"</p>
                                </div>

                                <NumberInput
                                    value=batch_size
                                    label="Batch Size"
                                    min=1.0
                                    max=4096.0
                                    step=1.0
                                    helper_text="Samples per training batch"
                                />

                                <NumberInput
                                    value=epochs
                                    label="Epochs"
                                    min=1.0
                                    max=10000.0
                                    step=1.0
                                    helper_text="Number of training epochs"
                                />
                            </div>

                            <div class="form-grid">
                                <div class="form-group">
                                    <label class="form-label">"Optimizer"</label>
                                    <div class="select-wrapper">
                                        <select
                                            class="form-select"
                                            on:change=move |e| optimizer.set(event_target_value(&e))
                                        >
                                            {optimizers.with_value(|opts| opts.iter().map(|(value, label)| {
                                                let v = value.clone();
                                                let val_for_check = value.clone();
                                                view! {
                                                    <option value=v selected=move || optimizer.get() == val_for_check>
                                                        {label.clone()}
                                                    </option>
                                                }
                                            }).collect_view())}
                                        </select>
                                        <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
                                    </div>
                                </div>

                                <div class="form-group">
                                    <label class="form-label">"Loss Function"</label>
                                    <div class="select-wrapper">
                                        <select
                                            class="form-select"
                                            on:change=move |e| loss_function.set(event_target_value(&e))
                                        >
                                            {loss_functions.with_value(|funcs| funcs.iter().map(|(value, label)| {
                                                let v = value.clone();
                                                let val_for_check = value.clone();
                                                view! {
                                                    <option value=v selected=move || loss_function.get() == val_for_check>
                                                        {label.clone()}
                                                    </option>
                                                }
                                            }).collect_view())}
                                        </select>
                                        <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    // Summary Section
                    <div class="card form-section summary-section">
                        <div class="card-header">
                            <h2>"Configuration Summary"</h2>
                        </div>
                        <div class="card-body">
                            <div class="summary-grid">
                                <div class="summary-item">
                                    <span class="summary-label">"Model"</span>
                                    <span class="summary-value">{move || {
                                        match selected_model_id.get() {
                                            Some(id) => models.get().iter().find(|m| m.id == id).map(|m| m.name.clone()).unwrap_or("Unknown".to_string()),
                                            None => "Not selected".to_string()
                                        }
                                    }}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Dataset"</span>
                                    <span class="summary-value">{move || {
                                        match selected_dataset_id.get() {
                                            Some(id) => datasets.get().iter().find(|d| d.id == id).map(|d| d.name.clone()).unwrap_or("Unknown".to_string()),
                                            None => "Not selected".to_string()
                                        }
                                    }}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Run Name"</span>
                                    <span class="summary-value">{move || { let n = name.get(); if n.is_empty() { "Not set".to_string() } else { n } }}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Model Type"</span>
                                    <span class="summary-value">{move || model_type.get()}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Learning Rate"</span>
                                    <span class="summary-value">{move || format!("{:.0e}", learning_rate.get())}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Batch Size"</span>
                                    <span class="summary-value">{move || batch_size.get() as u32}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Epochs"</span>
                                    <span class="summary-value">{move || epochs.get() as u32}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Optimizer"</span>
                                    <span class="summary-value">{move || optimizer.get()}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                // Form Actions
                <div class="form-actions">
                    <A href="/training" class="btn btn-ghost">
                        "Cancel"
                    </A>
                    <button
                        type="submit"
                        class="btn btn-primary"
                        disabled=move || loading.get() || name.get().trim().is_empty() || selected_model_id.get().is_none() || selected_dataset_id.get().is_none()
                    >
                        <Show when=move || loading.get() fallback=|| {
                            view! {
                                <IconPlay size=IconSize::Sm />
                                <span>"Start Training"</span>
                            }
                        }>
                            <Spinner size=SpinnerSize::Sm />
                            <span>"Starting..."</span>
                        </Show>
                    </button>
                </div>
            </form>
        </div>
    }
}
