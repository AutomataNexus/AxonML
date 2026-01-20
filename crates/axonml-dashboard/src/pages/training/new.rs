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
    let learning_rate = create_rw_signal(0.001f64);
    let batch_size = create_rw_signal(32f64);
    let epochs = create_rw_signal(100f64);
    let optimizer = create_rw_signal("adam".to_string());
    let loss_function = create_rw_signal("cross_entropy".to_string());
    let (loading, set_loading) = create_signal(false);
    let error = create_rw_signal::<Option<String>>(None);

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
                    // Basic Info Section
                    <div class="card form-section">
                        <div class="card-header">
                            <h2>"Basic Information"</h2>
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
                                        step="0.0001"
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
                        disabled=move || loading.get() || name.get().trim().is_empty()
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
