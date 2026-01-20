//! Loading Spinner Components

use leptos::*;

/// Spinner size variants
#[derive(Debug, Clone, Copy, Default)]
pub enum SpinnerSize {
    Xs,
    Sm,
    #[default]
    Md,
    Lg,
    Xl,
}

impl SpinnerSize {
    pub fn class(&self) -> &'static str {
        match self {
            Self::Xs => "spinner-xs",
            Self::Sm => "spinner-sm",
            Self::Md => "spinner-md",
            Self::Lg => "spinner-lg",
            Self::Xl => "spinner-xl",
        }
    }

    pub fn size(&self) -> u32 {
        match self {
            Self::Xs => 16,
            Self::Sm => 20,
            Self::Md => 32,
            Self::Lg => 48,
            Self::Xl => 64,
        }
    }
}

/// Basic spinning loader
#[component]
pub fn Spinner(
    #[prop(default = SpinnerSize::Md)] size: SpinnerSize,
    #[prop(default = "var(--teal)".to_string())] color: String,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let s = size.size();

    view! {
        <svg
            class=format!("spinner {} {}", size.class(), class)
            width=s
            height=s
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
        >
            <circle
                cx="12"
                cy="12"
                r="10"
                fill="none"
                stroke="var(--slate-bg)"
                stroke-width="3"
            />
            <circle
                cx="12"
                cy="12"
                r="10"
                fill="none"
                stroke=color
                stroke-width="3"
                stroke-linecap="round"
                stroke-dasharray="31.4 31.4"
                class="spinner-arc"
            />
        </svg>
    }
}

/// Dots loading animation
#[component]
pub fn DotsLoader(
    #[prop(default = SpinnerSize::Md)] size: SpinnerSize,
    #[prop(default = "var(--teal)".to_string())] color: String,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let dot_size = match size {
        SpinnerSize::Xs => 4,
        SpinnerSize::Sm => 6,
        SpinnerSize::Md => 8,
        SpinnerSize::Lg => 10,
        SpinnerSize::Xl => 12,
    };

    view! {
        <div class=format!("dots-loader {} {}", size.class(), class)>
            <span class="dot" style=format!("width: {}px; height: {}px; background: {};", dot_size, dot_size, color)></span>
            <span class="dot" style=format!("width: {}px; height: {}px; background: {};", dot_size, dot_size, color)></span>
            <span class="dot" style=format!("width: {}px; height: {}px; background: {};", dot_size, dot_size, color)></span>
        </div>
    }
}

/// Pulse loading animation
#[component]
pub fn PulseLoader(
    #[prop(default = SpinnerSize::Md)] size: SpinnerSize,
    #[prop(default = "var(--teal)".to_string())] color: String,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let s = size.size();

    view! {
        <div
            class=format!("pulse-loader {} {}", size.class(), class)
            style=format!("width: {}px; height: {}px;", s, s)
        >
            <span class="pulse-ring" style=format!("border-color: {};", color)></span>
            <span class="pulse-ring" style=format!("border-color: {};", color)></span>
            <span class="pulse-dot" style=format!("background: {};", color)></span>
        </div>
    }
}

/// Loading overlay for sections
#[component]
pub fn LoadingOverlay(
    #[prop(into)] loading: MaybeSignal<bool>,
    #[prop(optional, into)] text: String,
    #[prop(default = SpinnerSize::Lg)] size: SpinnerSize,
    children: Children,
) -> impl IntoView {
    let text_empty = text.is_empty();
    let text_stored = store_value(text);
    let children_stored = store_value(children());

    view! {
        <div class="loading-overlay-container">
            {children_stored.get_value()}
            <Show when=move || loading.get()>
                <div class="loading-overlay">
                    <div class="loading-overlay-content">
                        <Spinner size=size />
                        <Show when=move || !text_empty>
                            <p class="loading-text">{text_stored.get_value()}</p>
                        </Show>
                    </div>
                </div>
            </Show>
        </div>
    }
}

/// Full page loading state
#[component]
pub fn PageLoader(
    #[prop(optional, into)] text: String,
) -> impl IntoView {
    let text_empty = text.is_empty();
    let text_stored = store_value(text);

    view! {
        <div class="page-loader">
            <div class="page-loader-content">
                <img src="/assets/logo.svg" alt="AxonML" class="page-loader-logo" />
                <Spinner size=SpinnerSize::Lg />
                <Show when=move || !text_empty>
                    <p class="loading-text">{text_stored.get_value()}</p>
                </Show>
            </div>
        </div>
    }
}

/// Skeleton loading placeholder
#[component]
pub fn Skeleton(
    #[prop(default = "100%".to_string())] width: String,
    #[prop(default = "1rem".to_string())] height: String,
    #[prop(default = "4px".to_string())] border_radius: String,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    view! {
        <div
            class=format!("skeleton {}", class)
            style=format!("width: {}; height: {}; border-radius: {};", width, height, border_radius)
        />
    }
}

/// Skeleton text lines
#[component]
pub fn SkeletonText(
    #[prop(default = 3)] lines: u32,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    view! {
        <div class=format!("skeleton-text {}", class)>
            {(0..lines).map(|i| {
                let width = if i == lines - 1 { "60%" } else { "100%" };
                view! { <Skeleton width=width.to_string() height="0.875rem".to_string() /> }
            }).collect_view()}
        </div>
    }
}

/// Skeleton card
#[component]
pub fn SkeletonCard(
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    view! {
        <div class=format!("card skeleton-card {}", class)>
            <Skeleton width="40%".to_string() height="1.5rem".to_string() />
            <div class="skeleton-card-body">
                <SkeletonText lines=3 />
            </div>
            <div class="skeleton-card-footer">
                <Skeleton width="80px".to_string() height="32px".to_string() border_radius="6px".to_string() />
            </div>
        </div>
    }
}

/// Skeleton table
#[component]
pub fn SkeletonTable(
    #[prop(default = 5)] rows: u32,
    #[prop(default = 4)] columns: u32,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    view! {
        <div class=format!("skeleton-table {}", class)>
            // Header
            <div class="skeleton-table-header">
                {(0..columns).map(|_| {
                    view! { <Skeleton width="80%".to_string() height="1rem".to_string() /> }
                }).collect_view()}
            </div>
            // Rows
            {(0..rows).map(|_| {
                view! {
                    <div class="skeleton-table-row">
                        {(0..columns).map(|_| {
                            view! { <Skeleton width="70%".to_string() height="0.875rem".to_string() /> }
                        }).collect_view()}
                    </div>
                }
            }).collect_view()}
        </div>
    }
}
