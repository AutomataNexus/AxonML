//! Toast Notification Components

use crate::components::icons::*;
use crate::state::{use_app_state, Toast, ToastType};
use leptos::*;

/// Toast container component - renders all active toasts
#[component]
pub fn ToastContainer() -> impl IntoView {
    let state = use_app_state();

    view! {
        <div class="toast-container">
            <For
                each=move || state.toasts.get()
                key=|toast| toast.id
                children=move |toast| {
                    let id = toast.id;
                    let state = state.clone();
                    view! { <ToastItem toast=toast on_close=Callback::new(move |_| state.remove_toast(id)) /> }
                }
            />
        </div>
    }
}

/// Individual toast item
#[component]
fn ToastItem(toast: Toast, on_close: Callback<()>) -> impl IntoView {
    let visible = create_rw_signal(true);
    let exiting = create_rw_signal(false);

    let handle_close = move |_| {
        exiting.set(true);
        set_timeout(
            move || {
                visible.set(false);
                on_close.call(());
            },
            std::time::Duration::from_millis(300),
        );
    };

    let toast_type = toast.toast_type;
    let title = store_value(toast.title);
    let message = store_value(toast.message);

    view! {
        <Show when=move || visible.get()>
            <div
                class=move || format!(
                    "toast {} {}",
                    toast_type.class(),
                    if exiting.get() { "toast-exit" } else { "toast-enter" }
                )
            >
                <div class="toast-icon">
                    {match toast_type {
                        ToastType::Success => view! { <IconCheckCircle size=IconSize::Md /> }.into_view(),
                        ToastType::Error => view! { <IconXCircle size=IconSize::Md /> }.into_view(),
                        ToastType::Warning => view! { <IconAlertTriangle size=IconSize::Md /> }.into_view(),
                        ToastType::Info => view! { <IconInfo size=IconSize::Md /> }.into_view(),
                    }}
                </div>
                <div class="toast-content">
                    <div class="toast-title">{title.get_value()}</div>
                    <div class="toast-message">{message.get_value()}</div>
                </div>
                <button class="toast-close" on:click=handle_close>
                    <IconX size=IconSize::Sm />
                </button>
            </div>
        </Show>
    }
}

/// Standalone toast function (for use without context)
#[component]
pub fn StandaloneToast(
    #[prop(into)] toast_type: ToastType,
    #[prop(into)] title: String,
    #[prop(into)] message: String,
    #[prop(default = 5000)] duration_ms: u64,
    #[prop(into)] on_close: Callback<()>,
) -> impl IntoView {
    let (visible, set_visible) = create_signal(true);
    let (exiting, set_exiting) = create_signal(false);

    // Auto-close timer
    let on_close_clone = on_close.clone();
    set_timeout(
        move || {
            set_exiting.set(true);
            set_timeout(
                move || {
                    set_visible.set(false);
                    on_close_clone.call(());
                },
                std::time::Duration::from_millis(300),
            );
        },
        std::time::Duration::from_millis(duration_ms),
    );

    let handle_close = move |_| {
        set_exiting.set(true);
        set_timeout(
            move || {
                set_visible.set(false);
                on_close.call(());
            },
            std::time::Duration::from_millis(300),
        );
    };

    let title_stored = store_value(title);
    let message_stored = store_value(message);

    view! {
        <Show when=move || visible.get()>
            <div
                class=move || format!(
                    "toast {} {}",
                    toast_type.class(),
                    if exiting.get() { "toast-exit" } else { "toast-enter" }
                )
            >
                <div class="toast-icon">
                    {match toast_type {
                        ToastType::Success => view! { <IconCheckCircle size=IconSize::Md /> }.into_view(),
                        ToastType::Error => view! { <IconXCircle size=IconSize::Md /> }.into_view(),
                        ToastType::Warning => view! { <IconAlertTriangle size=IconSize::Md /> }.into_view(),
                        ToastType::Info => view! { <IconInfo size=IconSize::Md /> }.into_view(),
                    }}
                </div>
                <div class="toast-content">
                    <div class="toast-title">{title_stored.get_value()}</div>
                    <div class="toast-message">{message_stored.get_value()}</div>
                </div>
                <button class="toast-close" on:click=handle_close>
                    <IconX size=IconSize::Sm />
                </button>
            </div>
        </Show>
    }
}

/// Inline notification banner (not a toast, stays in place)
#[component]
pub fn NotificationBanner(
    #[prop(into)] variant: ToastType,
    #[prop(into)] title: String,
    #[prop(optional, into)] message: String,
    #[prop(default = true)] dismissible: bool,
    #[prop(optional)] on_dismiss: Option<Callback<()>>,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let (visible, set_visible) = create_signal(true);

    let on_dismiss_stored = store_value(on_dismiss);

    let handle_dismiss = move |_| {
        set_visible.set(false);
        on_dismiss_stored.with_value(|cb| {
            if let Some(cb) = cb.as_ref() {
                cb.call(());
            }
        });
    };

    let variant_class = match variant {
        ToastType::Success => "banner-success",
        ToastType::Error => "banner-error",
        ToastType::Warning => "banner-warning",
        ToastType::Info => "banner-info",
    };

    let title_stored = store_value(title);
    let message_empty = message.is_empty();
    let message_stored = store_value(message);

    view! {
        <Show when=move || visible.get()>
            <div class=format!("notification-banner {} {}", variant_class, class)>
                <div class="banner-icon">
                    {match variant {
                        ToastType::Success => view! { <IconCheckCircle size=IconSize::Md /> }.into_view(),
                        ToastType::Error => view! { <IconXCircle size=IconSize::Md /> }.into_view(),
                        ToastType::Warning => view! { <IconAlertTriangle size=IconSize::Md /> }.into_view(),
                        ToastType::Info => view! { <IconInfo size=IconSize::Md /> }.into_view(),
                    }}
                </div>
                <div class="banner-content">
                    <div class="banner-title">{title_stored.get_value()}</div>
                    <Show when=move || !message_empty>
                        <div class="banner-message">{message_stored.get_value()}</div>
                    </Show>
                </div>
                <Show when=move || dismissible>
                    <button class="banner-dismiss" on:click=handle_dismiss>
                        <IconX size=IconSize::Sm />
                    </button>
                </Show>
            </div>
        </Show>
    }
}
