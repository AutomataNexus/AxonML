//! Training Notebook Editor Page
//!
//! Cell-based notebook interface for interactive ML training.

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::components::{icons::*, modal::*, spinner::*, StatusBadge};
use crate::state::use_app_state;
use crate::types::*;

/// Notebook editor page
#[component]
pub fn NotebookEditorPage() -> impl IntoView {
    let params = use_params_map();
    let state = use_app_state();
    let navigate = use_navigate();

    let (loading, set_loading) = create_signal(true);
    let (saving, set_saving) = create_signal(false);
    let (notebook, set_notebook) = create_signal::<Option<TrainingNotebook>>(None);
    let (selected_cell, set_selected_cell) = create_signal::<Option<String>>(None);
    let ai_modal = create_rw_signal(false);
    let (ai_prompt, set_ai_prompt) = create_signal(String::new());
    let (ai_loading, set_ai_loading) = create_signal(false);
    let (ai_suggestion, set_ai_suggestion) = create_signal::<Option<AiAssistResponse>>(None);

    let state_for_effect = state.clone();
    let state_for_save = state.clone();

    // Fetch notebook on mount
    create_effect(move |_| {
        let id = params.get().get("id").cloned().unwrap_or_default();
        let state = state_for_effect.clone();

        if id.is_empty() || id == "new" {
            // Create a new notebook
            set_loading.set(false);
            let new_notebook = TrainingNotebook {
                id: String::new(),
                user_id: String::new(),
                name: "Untitled Notebook".to_string(),
                description: None,
                cells: vec![
                    NotebookCell {
                        id: uuid::Uuid::new_v4().to_string(),
                        cell_type: CellType::Markdown,
                        source: "# Training Notebook\n\nDescribe your experiment here.".to_string(),
                        outputs: vec![],
                        status: CellStatus::Idle,
                        execution_count: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    NotebookCell {
                        id: uuid::Uuid::new_v4().to_string(),
                        cell_type: CellType::Code,
                        source: r#"# AxonML Training Imports
use axonml::prelude::*;
use axonml::nn::{Linear, Conv2d, BatchNorm2d, Dropout, Sequential};
use axonml::optim::{Adam, SGD, AdamW};
use axonml::data::{DataLoader, Dataset};
use axonml::tensor::Tensor;
use axonml::autograd::Variable;

# Training configuration
let device = Device::cuda_if_available();
let batch_size = 32;
let learning_rate = 0.001;
let epochs = 10;"#
                            .to_string(),
                        outputs: vec![],
                        status: CellStatus::Idle,
                        execution_count: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    NotebookCell {
                        id: uuid::Uuid::new_v4().to_string(),
                        cell_type: CellType::Markdown,
                        source: "## Model Definition".to_string(),
                        outputs: vec![],
                        status: CellStatus::Idle,
                        execution_count: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    NotebookCell {
                        id: uuid::Uuid::new_v4().to_string(),
                        cell_type: CellType::Code,
                        source: r#"# Define your model here
struct MyModel {
    fc1: Linear,
    fc2: Linear,
}

impl MyModel {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            fc1: Linear::new(input_size, hidden_size),
            fc2: Linear::new(hidden_size, output_size),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.fc1.forward(x).relu();
        self.fc2.forward(&x)
    }
}

let model = MyModel::new(784, 128, 10).to(&device);"#
                            .to_string(),
                        outputs: vec![],
                        status: CellStatus::Idle,
                        execution_count: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    NotebookCell {
                        id: uuid::Uuid::new_v4().to_string(),
                        cell_type: CellType::Markdown,
                        source: "## Training Loop".to_string(),
                        outputs: vec![],
                        status: CellStatus::Idle,
                        execution_count: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    NotebookCell {
                        id: uuid::Uuid::new_v4().to_string(),
                        cell_type: CellType::Code,
                        source: r#"# Setup optimizer and loss
let optimizer = Adam::new(model.parameters(), learning_rate);
let criterion = CrossEntropyLoss::new();

# Training loop
for epoch in 0..epochs {
    let mut total_loss = 0.0;
    let mut correct = 0;
    let mut total = 0;

    for (batch_idx, (data, target)) in train_loader.iter().enumerate() {
        let data = data.to(&device);
        let target = target.to(&device);

        optimizer.zero_grad();
        let output = model.forward(&data);
        let loss = criterion.forward(&output, &target);
        loss.backward();
        optimizer.step();

        total_loss += loss.item();
        let pred = output.argmax(1);
        correct += pred.eq(&target).sum().item() as usize;
        total += target.size(0);
    }

    let accuracy = 100.0 * correct as f32 / total as f32;
    println!("Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", epoch + 1, total_loss, accuracy);
}"#
                        .to_string(),
                        outputs: vec![],
                        status: CellStatus::Idle,
                        execution_count: None,
                        metadata: std::collections::HashMap::new(),
                    },
                ],
                metadata: NotebookMetadata::default(),
                checkpoints: vec![],
                model_id: None,
                dataset_id: None,
                status: RunStatus::Pending,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };
            set_notebook.set(Some(new_notebook));
            return;
        }

        set_loading.set(true);
        spawn_local(async move {
            match api::notebooks::get_notebook(&id).await {
                Ok(data) => {
                    set_notebook.set(Some(data));
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
            set_loading.set(false);
        });
    });

    // Save notebook action
    let save_action = create_action(move |_: &()| {
        let state = state_for_save.clone();
        let navigate_ref = navigate.clone();
        let nb_opt = notebook.get();

        async move {
            if let Some(nb) = nb_opt {
                set_saving.set(true);
                let result = if nb.id.is_empty() {
                    api::notebooks::create_notebook(CreateNotebookRequest {
                        name: nb.name.clone(),
                        description: nb.description.clone(),
                        cells: nb.cells.clone(),
                        model_id: nb.model_id.clone(),
                        dataset_id: nb.dataset_id.clone(),
                    })
                    .await
                } else {
                    api::notebooks::update_notebook(
                        &nb.id,
                        UpdateNotebookRequest {
                            name: Some(nb.name.clone()),
                            description: nb.description.clone(),
                            cells: Some(nb.cells.clone()),
                            model_id: nb.model_id.clone(),
                            dataset_id: nb.dataset_id.clone(),
                        },
                    )
                    .await
                };

                match result {
                    Ok(saved) => {
                        state.toast_success("Saved", "Notebook saved successfully");
                        if nb.id.is_empty() {
                            navigate_ref(
                                &format!("/training/notebooks/{}", saved.id),
                                Default::default(),
                            );
                        }
                        set_notebook.set(Some(saved));
                    }
                    Err(e) => {
                        state.toast_error("Error", e.message);
                    }
                }
                set_saving.set(false);
            }
        }
    });

    // Add new cell
    let add_cell = move |cell_type: CellType| {
        if let Some(mut nb) = notebook.get() {
            let new_cell = NotebookCell {
                id: uuid::Uuid::new_v4().to_string(),
                cell_type,
                source: match cell_type {
                    CellType::Code => "# Your code here".to_string(),
                    CellType::Markdown => "## Section".to_string(),
                },
                outputs: vec![],
                status: CellStatus::Idle,
                execution_count: None,
                metadata: std::collections::HashMap::new(),
            };

            if let Some(selected_id) = selected_cell.get() {
                if let Some(pos) = nb.cells.iter().position(|c| c.id == selected_id) {
                    nb.cells.insert(pos + 1, new_cell.clone());
                } else {
                    nb.cells.push(new_cell.clone());
                }
            } else {
                nb.cells.push(new_cell.clone());
            }

            set_selected_cell.set(Some(new_cell.id.clone()));
            set_notebook.set(Some(nb));
        }
    };

    // Delete cell
    let delete_cell = move |cell_id: String| {
        if let Some(mut nb) = notebook.get() {
            nb.cells.retain(|c| c.id != cell_id);
            if selected_cell.get() == Some(cell_id) {
                set_selected_cell.set(None);
            }
            set_notebook.set(Some(nb));
        }
    };

    // Update cell source
    let update_cell_source = move |cell_id: String, source: String| {
        if let Some(mut nb) = notebook.get() {
            if let Some(cell) = nb.cells.iter_mut().find(|c| c.id == cell_id) {
                cell.source = source;
                set_notebook.set(Some(nb));
            }
        }
    };

    // Format code on blur (basic auto-formatting)
    let format_cell_source = move |cell_id: String, is_code: bool| {
        if let Some(mut nb) = notebook.get() {
            if let Some(cell) = nb.cells.iter_mut().find(|c| c.id == cell_id) {
                let formatted = if is_code {
                    format_code(&cell.source)
                } else {
                    format_markdown(&cell.source)
                };
                if formatted != cell.source {
                    cell.source = formatted;
                    set_notebook.set(Some(nb));
                }
            }
        }
    };

    // Request AI assist
    let request_ai_assist = {
        let state = state.clone();
        move |_| {
            let prompt = ai_prompt.get();
            if prompt.is_empty() {
                state.toast_error("Error", "Please enter a prompt first");
                return;
            }

            let nb = notebook.get();
            if nb.is_none() {
                state.toast_error("Error", "No notebook loaded");
                return;
            }
            let nb = nb.unwrap();

            // Need a saved notebook to use AI assist
            if nb.id.is_empty() {
                state.toast_error(
                    "Save Required",
                    "Please save the notebook first before using AI assist",
                );
                set_ai_suggestion.set(Some(AiAssistResponse {
                    suggestion: "# Please save the notebook first\n# AI assistance requires the notebook to be saved so it can analyze the full context.".to_string(),
                    explanation: Some("Save the notebook first to enable AI assistance.".to_string()),
                    confidence: 0.0,
                    model: String::new(),
                    tokens_generated: 0,
                }));
                return;
            }

            state.toast_info("AI Assist", "Generating suggestion...");

            let notebook_id = nb.id.clone();
            let selected = selected_cell.get();
            let state = state.clone();

            set_ai_loading.set(true);
            spawn_local(async move {
                let request = AiAssistRequest {
                    prompt,
                    selected_cell_id: selected,
                    cell_type: CellType::Code,
                    include_imports: true,
                };

                match api::notebooks::ai_assist(&notebook_id, request).await {
                    Ok(response) => {
                        set_ai_suggestion.set(Some(response));
                    }
                    Err(e) => {
                        let msg = e.message;
                        state.toast_error("AI Error", msg.clone());
                        set_ai_suggestion.set(Some(AiAssistResponse {
                            suggestion: format!("# Error: {}", msg),
                            explanation: Some(msg),
                            confidence: 0.0,
                            model: String::new(),
                            tokens_generated: 0,
                        }));
                    }
                }
                set_ai_loading.set(false);
            });
        }
    };

    // Apply AI suggestion
    let apply_ai_suggestion = move |_| {
        if let (Some(suggestion), Some(cell_id)) = (ai_suggestion.get(), selected_cell.get()) {
            update_cell_source(cell_id, suggestion.suggestion);
            ai_modal.set(false);
            set_ai_suggestion.set(None);
            set_ai_prompt.set(String::new());
        }
    };

    view! {
        <div class="page notebook-editor-page">
            // Loading state
            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading notebook..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get() && notebook.get().is_some()>
                // Notebook header
                <div class="notebook-header">
                    <div class="notebook-title-section">
                        <input
                            type="text"
                            class="notebook-title-input"
                            placeholder="Notebook Name"
                            prop:value=move || notebook.get().map(|n| n.name.clone()).unwrap_or_default()
                            on:input=move |e| {
                                if let Some(mut nb) = notebook.get() {
                                    nb.name = event_target_value(&e);
                                    set_notebook.set(Some(nb));
                                }
                            }
                        />
                        {move || notebook.get().map(|nb| {
                            view! {
                                <StatusBadge status=nb.status.as_str().to_string() class=nb.status.color_class().to_string() />
                            }
                        })}
                    </div>

                    <div class="notebook-actions">
                        <button class="btn btn-ghost" on:click=move |_| ai_modal.set(true)>
                            <IconZap size=IconSize::Sm />
                            <span>"AI Assist"</span>
                        </button>
                        <button
                            class="btn btn-primary"
                            on:click=move |_| save_action.dispatch(())
                            disabled=move || saving.get()
                        >
                            {move || if saving.get() {
                                view! { <Spinner size=SpinnerSize::Xs /> }.into_view()
                            } else {
                                view! { <IconCheck size=IconSize::Sm /> }.into_view()
                            }}
                            <span>{move || if saving.get() { "Saving..." } else { "Save" }}</span>
                        </button>
                    </div>
                </div>

                // Toolbar
                <div class="notebook-toolbar">
                    <button class="btn btn-ghost btn-sm" on:click=move |_| add_cell(CellType::Code)>
                        <IconPlus size=IconSize::Xs />
                        <span>"Code"</span>
                    </button>
                    <button class="btn btn-ghost btn-sm" on:click=move |_| add_cell(CellType::Markdown)>
                        <IconPlus size=IconSize::Xs />
                        <span>"Markdown"</span>
                    </button>
                    <div class="toolbar-separator"></div>
                    <Show when=move || selected_cell.get().is_some()>
                        <button class="btn btn-ghost btn-sm text-danger" on:click=move |_| {
                            if let Some(id) = selected_cell.get() {
                                delete_cell(id);
                            }
                        }>
                            <IconTrash size=IconSize::Xs />
                        </button>
                    </Show>
                </div>

                // Cells
                <div class="notebook-cells">
                    {move || notebook.get().map(|nb| {
                        nb.cells.iter().map(|cell| {
                            let cell_id = cell.id.clone();
                            let cell_id_for_select = cell.id.clone();
                            let cell_id_for_update = cell.id.clone();
                            let cell_id_for_delete = cell.id.clone();
                            let cell_id_for_format = cell.id.clone();
                            let is_code = cell.cell_type == CellType::Code;
                            let is_code_for_format = is_code;
                            let is_selected = move || selected_cell.get() == Some(cell_id.clone());
                            let source = cell.source.clone();
                            let exec_count = cell.execution_count;
                            let outputs = cell.outputs.clone();
                            let outputs_for_check = outputs.clone();
                            let has_outputs = !outputs.is_empty();

                            view! {
                                <div
                                    class=move || format!("notebook-cell {} {}",
                                        if is_code { "cell-code" } else { "cell-markdown" },
                                        if is_selected() { "selected" } else { "" }
                                    )
                                    on:click=move |_| set_selected_cell.set(Some(cell_id_for_select.clone()))
                                >
                                    <div class="cell-gutter">
                                        {if is_code {
                                            view! {
                                                <span class="execution-count">
                                                    {exec_count.map(|c| format!("[{}]", c)).unwrap_or_else(|| "[ ]".to_string())}
                                                </span>
                                            }.into_view()
                                        } else {
                                            view! {
                                                <IconFileText size=IconSize::Xs />
                                            }.into_view()
                                        }}
                                        <button
                                            class="cell-delete-btn"
                                            title="Delete cell"
                                            on:click={
                                                let id = cell_id_for_delete.clone();
                                                move |e: web_sys::MouseEvent| {
                                                    e.stop_propagation();
                                                    delete_cell(id.clone());
                                                }
                                            }
                                        >
                                            <IconTrash size=IconSize::Xs />
                                        </button>
                                    </div>

                                    <div class="cell-content">
                                        <div class="cell-input">
                                            <textarea
                                                class="cell-source"
                                                prop:value=source.clone()
                                                on:input={
                                                    let id = cell_id_for_update.clone();
                                                    move |e| update_cell_source(id.clone(), event_target_value(&e))
                                                }
                                                on:blur={
                                                    let id = cell_id_for_format.clone();
                                                    move |_| format_cell_source(id.clone(), is_code_for_format)
                                                }
                                                spellcheck="false"
                                                rows={std::cmp::max(3, source.lines().count())}
                                            />
                                        </div>

                                        // Outputs
                                        <Show when=move || has_outputs>
                                            <div class="cell-outputs">
                                                {outputs_for_check.iter().map(|output| {
                                                    let text = output.text.clone();
                                                    let is_error = output.output_type == "error" || output.error_name.is_some();
                                                    view! {
                                                        <div class=format!("cell-output {}", if is_error { "output-error" } else { "" })>
                                                            {text.map(|t| view! { <pre class="output-text">{t}</pre> })}
                                                        </div>
                                                    }
                                                }).collect_view()}
                                            </div>
                                        </Show>
                                    </div>
                                </div>
                            }
                        }).collect_view()
                    })}
                </div>

                // Add cell button at the end
                <div class="add-cell-footer">
                    <button class="btn btn-ghost" on:click=move |_| add_cell(CellType::Code)>
                        <IconPlus size=IconSize::Sm />
                        <span>"Add Code Cell"</span>
                    </button>
                    <button class="btn btn-ghost" on:click=move |_| add_cell(CellType::Markdown)>
                        <IconPlus size=IconSize::Sm />
                        <span>"Add Markdown Cell"</span>
                    </button>
                </div>
            </Show>

            // AI Assist Modal
            <Modal
                show=ai_modal
                title="AI Assistant"
            >
                <div class="ai-assist-modal">
                    <div class="form-group">
                        <label>"What would you like help with?"</label>
                        <textarea
                            class="form-control"
                            rows=3
                            placeholder="E.g., Create a training loop for image classification with early stopping"
                            prop:value=move || ai_prompt.get()
                            on:input=move |e| set_ai_prompt.set(event_target_value(&e))
                        />
                    </div>

                    <button
                        class="btn btn-primary"
                        on:click=request_ai_assist
                        disabled=move || ai_loading.get() || ai_prompt.get().is_empty()
                    >
                        {move || if ai_loading.get() {
                            view! { <Spinner size=SpinnerSize::Xs /> }.into_view()
                        } else {
                            view! { <IconZap size=IconSize::Sm /> }.into_view()
                        }}
                        <span>{move || if ai_loading.get() { "Generating..." } else { "Generate" }}</span>
                    </button>

                    <Show when=move || ai_suggestion.get().is_some()>
                        <div class="ai-suggestion">
                            <div class="ai-suggestion-header">
                                <h4>"Suggestion"</h4>
                                {move || {
                                    let suggestion = ai_suggestion.get();
                                    if let Some(ref s) = suggestion {
                                        if !s.model.is_empty() {
                                            Some(view! {
                                                <span class="ai-model-info">
                                                    {format!("{} ({} tokens)", s.model, s.tokens_generated)}
                                                </span>
                                            })
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                }}
                            </div>
                            <pre class="suggestion-code">{move || ai_suggestion.get().map(|s| s.suggestion).unwrap_or_default()}</pre>
                            {move || ai_suggestion.get().and_then(|s| s.explanation).map(|exp| view! {
                                <p class="suggestion-explanation">{exp}</p>
                            })}
                            <div class="suggestion-actions">
                                <button class="btn btn-primary" on:click=apply_ai_suggestion>
                                    "Apply to Selected Cell"
                                </button>
                                <button class="btn btn-ghost" on:click=move |_| {
                                    set_ai_suggestion.set(None);
                                }>
                                    "Discard"
                                </button>
                            </div>
                        </div>
                    </Show>
                </div>
            </Modal>
        </div>
    }
}

/// Format code with basic auto-formatting
fn format_code(source: &str) -> String {
    let mut result = String::new();
    let mut indent_level: i32 = 0;

    for line in source.lines() {
        let trimmed = line.trim();

        // Skip empty lines but preserve them
        if trimmed.is_empty() {
            result.push('\n');
            continue;
        }

        // Decrease indent for closing braces/brackets at start
        if trimmed.starts_with('}') || trimmed.starts_with(']') || trimmed.starts_with(')') {
            indent_level = (indent_level - 1).max(0);
        }

        // Add proper indentation (4 spaces per level)
        for _ in 0..indent_level {
            result.push_str("    ");
        }

        // Add the trimmed line
        result.push_str(trimmed);
        result.push('\n');

        // Increase indent after opening braces/brackets
        if trimmed.ends_with('{') || trimmed.ends_with('[') || trimmed.ends_with('(') {
            indent_level += 1;
        }
        // Handle else/elif on same line as closing brace
        if trimmed.contains("} else") || trimmed.contains("} elif") {
            // Don't change indent
        }
    }

    // Remove trailing newlines but ensure one at end
    result.trim_end().to_string()
}

/// Format markdown with basic cleanup
fn format_markdown(source: &str) -> String {
    source
        .lines()
        .map(|line| line.trim_end()) // Remove trailing whitespace
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}
