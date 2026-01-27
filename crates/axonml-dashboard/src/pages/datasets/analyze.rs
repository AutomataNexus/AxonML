//! Data Analysis Page
//!
//! Analyze datasets, preview samples, validate, and generate training configs.

use leptos::*;

use crate::api;
use crate::components::{icons::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// Data analysis page
#[component]
pub fn DataAnalyzePage() -> impl IntoView {
    let state = use_app_state();

    let (datasets, set_datasets) = create_signal::<Vec<Dataset>>(vec![]);
    let (selected_dataset, set_selected_dataset) = create_signal::<Option<String>>(None);
    let (analysis, set_analysis) = create_signal::<Option<DatasetAnalysis>>(None);
    let (preview, set_preview) = create_signal::<Option<DataPreviewResponse>>(None);
    let (validation, set_validation) = create_signal::<Option<ValidationResult>>(None);
    let (generated_config, set_generated_config) =
        create_signal::<Option<GeneratedTrainingConfig>>(None);
    let (loading, set_loading) = create_signal(true);
    let (analyzing, set_analyzing) = create_signal(false);
    let (active_tab, set_active_tab) = create_signal("analysis".to_string());

    let state_for_fetch = state.clone();

    // Fetch datasets on mount
    create_effect(move |_| {
        let state = state_for_fetch.clone();
        spawn_local(async move {
            match api::datasets::list().await {
                Ok(data) => set_datasets.set(data),
                Err(e) => state.toast_error("Error", e.message),
            }
            set_loading.set(false);
        });
    });

    view! {
        <div class="page data-analyze-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"Data Analysis"</h1>
                    <p class="page-subtitle">"Analyze, preview, validate datasets and generate training configurations"</p>
                </div>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading datasets..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                <div class="analysis-layout">
                    // Dataset Selection Panel
                    <div class="dataset-select-panel card">
                        <div class="card-header">
                            <h3>
                                <IconDatabase size=IconSize::Md />
                                <span>"Select Dataset"</span>
                            </h3>
                        </div>
                        <div class="card-body">
                            <DatasetSelector
                                datasets=datasets
                                selected=selected_dataset
                                set_selected=set_selected_dataset
                                set_analysis=set_analysis
                                set_preview=set_preview
                                set_validation=set_validation
                                set_generated_config=set_generated_config
                                set_analyzing=set_analyzing
                            />
                        </div>
                    </div>

                    // Analysis Results Panel
                    <Show when=move || selected_dataset.get().is_some()>
                        <div class="analysis-panel">
                            // Tab Navigation
                            <div class="tab-nav">
                                <button
                                    class={move || if active_tab.get() == "analysis" { "tab active" } else { "tab" }}
                                    on:click=move |_| set_active_tab.set("analysis".to_string())
                                >
                                    <IconActivity size=IconSize::Sm />
                                    <span>"Analysis"</span>
                                </button>
                                <button
                                    class={move || if active_tab.get() == "preview" { "tab active" } else { "tab" }}
                                    on:click=move |_| set_active_tab.set("preview".to_string())
                                >
                                    <IconEye size=IconSize::Sm />
                                    <span>"Preview"</span>
                                </button>
                                <button
                                    class={move || if active_tab.get() == "validation" { "tab active" } else { "tab" }}
                                    on:click=move |_| set_active_tab.set("validation".to_string())
                                >
                                    <IconCheck size=IconSize::Sm />
                                    <span>"Validation"</span>
                                </button>
                                <button
                                    class={move || if active_tab.get() == "config" { "tab active" } else { "tab" }}
                                    on:click=move |_| set_active_tab.set("config".to_string())
                                >
                                    <IconSettings size=IconSize::Sm />
                                    <span>"Config"</span>
                                </button>
                            </div>

                            // Tab Content
                            <div class="tab-content">
                                <Show when=move || analyzing.get()>
                                    <div class="loading-state">
                                        <Spinner size=SpinnerSize::Lg />
                                        <p>"Analyzing dataset..."</p>
                                    </div>
                                </Show>

                                <Show when=move || !analyzing.get()>
                                    <Show when=move || active_tab.get() == "analysis">
                                        <AnalysisTab analysis=analysis />
                                    </Show>

                                    <Show when=move || active_tab.get() == "preview">
                                        <PreviewTab preview=preview />
                                    </Show>

                                    <Show when=move || active_tab.get() == "validation">
                                        <ValidationTab validation=validation />
                                    </Show>

                                    <Show when=move || active_tab.get() == "config">
                                        <ConfigTab config=generated_config />
                                    </Show>
                                </Show>
                            </div>
                        </div>
                    </Show>
                </div>
            </Show>
        </div>
    }
}

/// Dataset selector component
#[component]
fn DatasetSelector(
    datasets: ReadSignal<Vec<Dataset>>,
    selected: ReadSignal<Option<String>>,
    set_selected: WriteSignal<Option<String>>,
    set_analysis: WriteSignal<Option<DatasetAnalysis>>,
    set_preview: WriteSignal<Option<DataPreviewResponse>>,
    set_validation: WriteSignal<Option<ValidationResult>>,
    set_generated_config: WriteSignal<Option<GeneratedTrainingConfig>>,
    set_analyzing: WriteSignal<bool>,
) -> impl IntoView {
    let state = use_app_state();

    view! {
        <div class="dataset-list">
            {move || datasets.get().into_iter().map(|dataset| {
                let dataset_id = dataset.id.clone();
                let dataset_name_display = dataset.name.clone();
                let dataset_type_display = dataset.dataset_type.clone();
                let is_selected = selected.get().as_ref() == Some(&dataset_id);
                let state = state.clone();

                let on_click = move |_| {
                    let id = dataset_id.clone();
                    let state = state.clone();

                    set_selected.set(Some(id.clone()));
                    set_analyzing.set(true);

                    spawn_local(async move {
                        // Run analysis
                        let analyze_query = AnalyzeQuery {
                            data_type: None,
                            max_samples: None,
                        };

                        match api::data::analyze(&id, &analyze_query).await {
                            Ok(a) => set_analysis.set(Some(a)),
                            Err(e) => {
                                state.toast_error("Analysis Error", e.message.clone());
                            }
                        }

                        // Run preview
                        let preview_query = PreviewQuery {
                            num_samples: Some(5),
                        };

                        match api::data::preview(&id, &preview_query).await {
                            Ok(p) => set_preview.set(Some(p)),
                            Err(_) => {} // Preview failure is not critical
                        }

                        // Run validation
                        let validate_query = ValidateQuery {
                            num_classes: None,
                            check_balance: Some(true),
                        };

                        match api::data::validate(&id, &validate_query).await {
                            Ok(v) => set_validation.set(Some(v)),
                            Err(_) => {} // Validation failure is not critical
                        }

                        // Generate config
                        let config_request = GenerateConfigRequest {
                            format: Some("toml".to_string()),
                        };

                        match api::data::generate_config(&id, &config_request).await {
                            Ok(c) => set_generated_config.set(Some(c)),
                            Err(_) => {} // Config generation failure is not critical
                        }

                        set_analyzing.set(false);
                    });
                };

                view! {
                    <div
                        class={if is_selected { "dataset-item selected" } else { "dataset-item" }}
                        on:click=on_click
                    >
                        <div class="dataset-icon">
                            <IconDatabase size=IconSize::Md />
                        </div>
                        <div class="dataset-info">
                            <span class="dataset-name">{dataset_name_display}</span>
                            <span class="dataset-type badge">{dataset_type_display}</span>
                        </div>
                    </div>
                }
            }).collect_view()}

            {move || if datasets.get().is_empty() {
                view! {
                    <div class="empty-state">
                        <IconDatabase size=IconSize::Lg class="text-muted".to_string() />
                        <p>"No datasets found"</p>
                        <p class="text-muted">"Upload a dataset first"</p>
                    </div>
                }.into_view()
            } else {
                view! {}.into_view()
            }}
        </div>
    }
}

/// Analysis tab component
#[component]
fn AnalysisTab(analysis: ReadSignal<Option<DatasetAnalysis>>) -> impl IntoView {
    view! {
        {move || {
            if let Some(a) = analysis.get() {
                view! {
                    <div class="analysis-content">
                        <div class="analysis-grid">
                            <div class="stat-card">
                                <span class="stat-label">"Data Type"</span>
                                <span class="stat-value">{a.data_type.clone()}</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-label">"Task Type"</span>
                                <span class="stat-value">{a.task_type.clone()}</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-label">"Samples"</span>
                                <span class="stat-value">{a.num_samples.to_string()}</span>
                            </div>
                            <div class="stat-card">
                                <span class="stat-label">"Classes"</span>
                                <span class="stat-value">{a.num_classes.map(|n| n.to_string()).unwrap_or("-".to_string())}</span>
                            </div>
                        </div>

                        // Statistics
                        <div class="section">
                            <h4>"Statistics"</h4>
                            <div class="stats-grid">
                                <div class="stat-item">
                                    <span class="label">"Mean"</span>
                                    <span class="value">{a.statistics.mean.map(|v| format!("{:.4}", v)).unwrap_or("-".to_string())}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="label">"Std Dev"</span>
                                    <span class="value">{a.statistics.std.map(|v| format!("{:.4}", v)).unwrap_or("-".to_string())}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="label">"Min"</span>
                                    <span class="value">{a.statistics.min.map(|v| format!("{:.4}", v)).unwrap_or("-".to_string())}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="label">"Max"</span>
                                    <span class="value">{a.statistics.max.map(|v| format!("{:.4}", v)).unwrap_or("-".to_string())}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="label">"Missing"</span>
                                    <span class="value">{format!("{} ({:.1}%)", a.statistics.missing_count, a.statistics.missing_percentage)}</span>
                                </div>
                            </div>
                        </div>

                        // Recommendations
                        <div class="section">
                            <h4>"Training Recommendations"</h4>
                            <div class="recommendations">
                                <div class="rec-item">
                                    <span class="label">"Model"</span>
                                    <span class="value">{a.recommendations.suggested_model.clone()}</span>
                                </div>
                                <div class="rec-item">
                                    <span class="label">"Batch Size"</span>
                                    <span class="value">{a.recommendations.suggested_batch_size.to_string()}</span>
                                </div>
                                <div class="rec-item">
                                    <span class="label">"Learning Rate"</span>
                                    <span class="value">{format!("{:.6}", a.recommendations.suggested_lr)}</span>
                                </div>
                                <div class="rec-item">
                                    <span class="label">"Epochs"</span>
                                    <span class="value">{a.recommendations.suggested_epochs.to_string()}</span>
                                </div>
                                <div class="rec-item">
                                    <span class="label">"Optimizer"</span>
                                    <span class="value">{a.recommendations.suggested_optimizer.clone()}</span>
                                </div>
                            </div>
                            {if !a.recommendations.notes.is_empty() {
                                view! {
                                    <div class="notes">
                                        <h5>"Notes"</h5>
                                        <ul>
                                            {a.recommendations.notes.clone().into_iter().map(|note| view! {
                                                <li>{note}</li>
                                            }).collect_view()}
                                        </ul>
                                    </div>
                                }.into_view()
                            } else {
                                view! {}.into_view()
                            }}
                        </div>
                    </div>
                }.into_view()
            } else {
                view! {
                    <div class="empty-state">
                        <p>"Select a dataset to analyze"</p>
                    </div>
                }.into_view()
            }
        }}
    }
}

/// Preview tab component
#[component]
fn PreviewTab(preview: ReadSignal<Option<DataPreviewResponse>>) -> impl IntoView {
    view! {
        {move || {
            if let Some(p) = preview.get() {
                view! {
                    <div class="preview-content">
                        <div class="preview-header">
                            <span class="badge">{p.data_type.clone()}</span>
                            <span class="total">{format!("{} total samples", p.total_samples)}</span>
                        </div>
                        <div class="samples-list">
                            {p.samples.clone().into_iter().map(|sample| view! {
                                <div class="sample-card">
                                    <span class="sample-index">"#"{sample.index.to_string()}</span>
                                    {sample.label.clone().map(|l| view! {
                                        <span class="sample-label badge">{l}</span>
                                    })}
                                    {sample.text.clone().map(|t| view! {
                                        <p class="sample-text">{if t.len() > 200 { format!("{}...", &t[..200]) } else { t }}</p>
                                    })}
                                    {sample.features.clone().map(|f| view! {
                                        <div class="sample-features">
                                            {f.into_iter().take(10).map(|v| view! {
                                                <span class="feature-value">{format!("{:.3}", v)}</span>
                                            }).collect_view()}
                                        </div>
                                    })}
                                    {sample.image_dimensions.map(|(w, h)| view! {
                                        <span class="image-dims">{format!("{}x{}", w, h)}</span>
                                    })}
                                </div>
                            }).collect_view()}
                        </div>
                    </div>
                }.into_view()
            } else {
                view! {
                    <div class="empty-state">
                        <p>"No preview available"</p>
                    </div>
                }.into_view()
            }
        }}
    }
}

/// Validation tab component
#[component]
fn ValidationTab(validation: ReadSignal<Option<ValidationResult>>) -> impl IntoView {
    view! {
        {move || {
            if let Some(v) = validation.get() {
                view! {
                    <div class="validation-content">
                        <div class="validation-header">
                            <span class={if v.is_valid { "badge badge-success" } else { "badge badge-error" }}>
                                {if v.is_valid { "Valid" } else { "Invalid" }}
                            </span>
                            <span class="sample-count">{format!("{} samples", v.num_samples)}</span>
                        </div>

                        {if !v.issues.is_empty() {
                            view! {
                                <div class="issues-section">
                                    <h4>"Issues"</h4>
                                    <ul class="issues-list">
                                        {v.issues.clone().into_iter().map(|issue| view! {
                                            <li class={format!("issue issue-{}", issue.severity)}>
                                                <span class="issue-category badge">{issue.category.clone()}</span>
                                                <span class="issue-message">{issue.message.clone()}</span>
                                                {issue.file_path.clone().map(|p| view! {
                                                    <span class="issue-path">{p}</span>
                                                })}
                                            </li>
                                        }).collect_view()}
                                    </ul>
                                </div>
                            }.into_view()
                        } else {
                            view! {}.into_view()
                        }}

                        {if !v.warnings.is_empty() {
                            view! {
                                <div class="warnings-section">
                                    <h4>"Warnings"</h4>
                                    <ul class="warnings-list">
                                        {v.warnings.clone().into_iter().map(|warning| view! {
                                            <li class="warning">{warning}</li>
                                        }).collect_view()}
                                    </ul>
                                </div>
                            }.into_view()
                        } else {
                            view! {}.into_view()
                        }}

                        {v.class_distribution.clone().map(|dist| view! {
                            <div class="distribution-section">
                                <h4>"Class Distribution"</h4>
                                <div class="distribution-grid">
                                    {dist.into_iter().map(|(class, count)| view! {
                                        <div class="dist-item">
                                            <span class="class-name">{class}</span>
                                            <span class="class-count">{count.to_string()}</span>
                                        </div>
                                    }).collect_view()}
                                </div>
                            </div>
                        })}
                    </div>
                }.into_view()
            } else {
                view! {
                    <div class="empty-state">
                        <p>"No validation results"</p>
                    </div>
                }.into_view()
            }
        }}
    }
}

/// Config tab component
#[component]
fn ConfigTab(config: ReadSignal<Option<GeneratedTrainingConfig>>) -> impl IntoView {
    view! {
        {move || {
            if let Some(c) = config.get() {
                view! {
                    <div class="config-content">
                        <div class="config-header">
                            <h4>{c.name.clone()}</h4>
                            <span class="badge">{c.model_type.clone()}</span>
                        </div>

                        <div class="config-sections">
                            <div class="config-section">
                                <h5>"Training Config"</h5>
                                <div class="config-grid">
                                    <div class="config-item">
                                        <span class="label">"Learning Rate"</span>
                                        <span class="value">{format!("{:.6}", c.config.learning_rate)}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Batch Size"</span>
                                        <span class="value">{c.config.batch_size.to_string()}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Epochs"</span>
                                        <span class="value">{c.config.epochs.to_string()}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Optimizer"</span>
                                        <span class="value">{c.config.optimizer.clone()}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Loss Function"</span>
                                        <span class="value">{c.config.loss_function.clone()}</span>
                                    </div>
                                </div>
                            </div>

                            <div class="config-section">
                                <h5>"Data Config"</h5>
                                <div class="config-grid">
                                    <div class="config-item">
                                        <span class="label">"Train Split"</span>
                                        <span class="value">{format!("{:.0}%", c.data_config.train_split * 100.0)}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Val Split"</span>
                                        <span class="value">{format!("{:.0}%", c.data_config.val_split * 100.0)}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Test Split"</span>
                                        <span class="value">{format!("{:.0}%", c.data_config.test_split * 100.0)}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Shuffle"</span>
                                        <span class="value">{if c.data_config.shuffle { "Yes" } else { "No" }}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Augmentation"</span>
                                        <span class="value">{if c.data_config.augmentation { "Yes" } else { "No" }}</span>
                                    </div>
                                    <div class="config-item">
                                        <span class="label">"Normalize"</span>
                                        <span class="value">{if c.data_config.normalize { "Yes" } else { "No" }}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {if !c.notes.is_empty() {
                            view! {
                                <div class="config-notes">
                                    <h5>"Notes"</h5>
                                    <ul>
                                        {c.notes.clone().into_iter().map(|note| view! {
                                            <li>{note}</li>
                                        }).collect_view()}
                                    </ul>
                                </div>
                            }.into_view()
                        } else {
                            view! {}.into_view()
                        }}
                    </div>
                }.into_view()
            } else {
                view! {
                    <div class="empty-state">
                        <p>"No config generated"</p>
                    </div>
                }.into_view()
            }
        }}
    }
}
