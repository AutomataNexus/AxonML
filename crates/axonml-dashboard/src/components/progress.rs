//! Progress Bar Components

use leptos::*;

/// Progress bar variant
#[derive(Debug, Clone, Copy, Default)]
pub enum ProgressVariant {
    #[default]
    Primary,
    Success,
    Warning,
    Error,
    Info,
}

impl ProgressVariant {
    pub fn class(&self) -> &'static str {
        match self {
            Self::Primary => "progress-primary",
            Self::Success => "progress-success",
            Self::Warning => "progress-warning",
            Self::Error => "progress-error",
            Self::Info => "progress-info",
        }
    }
}

/// Progress bar component
#[component]
pub fn ProgressBar(
    #[prop(into)] value: MaybeSignal<f64>,
    #[prop(default = 100.0)] max: f64,
    #[prop(default = ProgressVariant::Primary)] variant: ProgressVariant,
    #[prop(default = false)] show_label: bool,
    #[prop(default = false)] striped: bool,
    #[prop(default = false)] animated: bool,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let percentage = move || {
        let v = value.get();
        ((v / max) * 100.0).clamp(0.0, 100.0)
    };

    let bar_class = move || {
        let mut classes = vec!["progress-bar", variant.class()];
        if striped {
            classes.push("progress-striped");
        }
        if animated {
            classes.push("progress-animated");
        }
        classes.join(" ")
    };

    view! {
        <div class=format!("progress {}", class)>
            <div
                class=bar_class
                style=move || format!("width: {}%", percentage())
                role="progressbar"
                aria-valuenow=move || value.get()
                aria-valuemin="0"
                aria-valuemax=max
            >
                <Show when=move || show_label>
                    <span class="progress-label">
                        {move || format!("{:.0}%", percentage())}
                    </span>
                </Show>
            </div>
        </div>
    }
}

/// Circular progress indicator
#[component]
pub fn CircularProgress(
    #[prop(into)] value: MaybeSignal<f64>,
    #[prop(default = 100.0)] max: f64,
    #[prop(default = 48)] size: u32,
    #[prop(default = 4)] stroke_width: u32,
    #[prop(default = ProgressVariant::Primary)] variant: ProgressVariant,
    #[prop(default = false)] show_label: bool,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let radius = (size / 2 - stroke_width) as f64;
    let circumference = 2.0 * std::f64::consts::PI * radius;
    let center = size as f64 / 2.0;

    let stroke_dashoffset = move || {
        let percentage = (value.get() / max).clamp(0.0, 1.0);
        circumference * (1.0 - percentage)
    };

    let color = match variant {
        ProgressVariant::Primary => "var(--teal)",
        ProgressVariant::Success => "var(--success)",
        ProgressVariant::Warning => "var(--warning)",
        ProgressVariant::Error => "var(--error)",
        ProgressVariant::Info => "var(--info)",
    };

    view! {
        <div class=format!("circular-progress {}", class) style=format!("width: {}px; height: {}px;", size, size)>
            <svg
                viewBox=format!("0 0 {} {}", size, size)
                class="circular-progress-svg"
            >
                // Background circle
                <circle
                    cx=center
                    cy=center
                    r=radius
                    fill="none"
                    stroke="var(--slate-bg)"
                    stroke-width=stroke_width
                />
                // Progress circle
                <circle
                    cx=center
                    cy=center
                    r=radius
                    fill="none"
                    stroke=color
                    stroke-width=stroke_width
                    stroke-dasharray=circumference
                    stroke-dashoffset=stroke_dashoffset
                    stroke-linecap="round"
                    transform=format!("rotate(-90 {} {})", center, center)
                    class="circular-progress-indicator"
                />
            </svg>
            <Show when=move || show_label>
                <div class="circular-progress-label">
                    {move || format!("{:.0}%", (value.get() / max * 100.0).clamp(0.0, 100.0))}
                </div>
            </Show>
        </div>
    }
}

/// Progress steps component
#[component]
pub fn ProgressSteps(
    #[prop(into)] steps: Vec<String>,
    #[prop(into)] current: MaybeSignal<usize>,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    view! {
        <div class=format!("progress-steps {}", class)>
            {steps.into_iter().enumerate().map(|(i, label)| {
                let is_completed = move || i < current.get();
                let is_current = move || i == current.get();

                view! {
                    <div class=move || format!(
                        "progress-step {} {}",
                        if is_completed() { "completed" } else { "" },
                        if is_current() { "current" } else { "" }
                    )>
                        <div class="step-indicator">
                            <Show
                                when=is_completed
                                fallback=move || view! { <span>{i + 1}</span> }
                            >
                                <svg viewBox="0 0 24 24" class="step-check">
                                    <polyline points="20 6 9 17 4 12" fill="none" stroke="currentColor" stroke-width="2" />
                                </svg>
                            </Show>
                        </div>
                        <div class="step-label">{label.clone()}</div>
                    </div>
                }
            }).collect_view()}
        </div>
    }
}

/// Training progress component with epoch/step info
#[component]
pub fn TrainingProgress(
    #[prop(into)] epoch: MaybeSignal<u32>,
    #[prop(into)] total_epochs: MaybeSignal<u32>,
    #[prop(into)] step: MaybeSignal<u32>,
    #[prop(into)] total_steps: MaybeSignal<u32>,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let epoch_for_progress = epoch.clone();
    let total_epochs_for_progress = total_epochs.clone();
    let step_for_progress = step.clone();
    let total_steps_for_progress = total_steps.clone();

    let epoch_progress = Signal::derive(move || {
        let t = total_epochs_for_progress.get();
        if t > 0 {
            (epoch_for_progress.get() as f64 / t as f64) * 100.0
        } else {
            0.0
        }
    });

    let step_progress = Signal::derive(move || {
        let t = total_steps_for_progress.get();
        if t > 0 {
            (step_for_progress.get() as f64 / t as f64) * 100.0
        } else {
            0.0
        }
    });

    view! {
        <div class=format!("training-progress {}", class)>
            <div class="training-progress-row">
                <div class="training-progress-label">
                    <span class="label">"Epoch"</span>
                    <span class="value">
                        {move || format!("{}/{}", epoch.get(), total_epochs.get())}
                    </span>
                </div>
                <ProgressBar
                    value=epoch_progress
                    variant=ProgressVariant::Primary
                />
            </div>
            <div class="training-progress-row">
                <div class="training-progress-label">
                    <span class="label">"Step"</span>
                    <span class="value">
                        {move || format!("{}/{}", step.get(), total_steps.get())}
                    </span>
                </div>
                <ProgressBar
                    value=step_progress
                    variant=ProgressVariant::Info
                />
            </div>
        </div>
    }
}
