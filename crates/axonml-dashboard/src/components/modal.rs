//! Modal Dialog Components

use crate::components::icons::*;
use leptos::*;
use wasm_bindgen::JsCast;

/// Modal size variants
#[derive(Debug, Clone, Copy, Default)]
pub enum ModalSize {
    Small,
    #[default]
    Medium,
    Large,
    FullScreen,
}

impl ModalSize {
    pub fn class(&self) -> &'static str {
        match self {
            Self::Small => "modal-sm",
            Self::Medium => "modal-md",
            Self::Large => "modal-lg",
            Self::FullScreen => "modal-fullscreen",
        }
    }
}

/// Modal component
#[component]
pub fn Modal(
    #[prop(into)] show: RwSignal<bool>,
    #[prop(optional, into)] title: String,
    #[prop(default = ModalSize::Medium)] size: ModalSize,
    #[prop(default = true)] close_on_backdrop: bool,
    #[prop(default = true)] close_on_escape: bool,
    #[prop(default = true)] show_close_button: bool,
    #[prop(optional, into)] class: String,
    children: Children,
    #[prop(optional)] footer: Option<ChildrenFn>,
) -> impl IntoView {
    let modal_ref = create_node_ref::<html::Div>();

    // Handle escape key
    let close_on_esc = close_on_escape;
    create_effect(move |_| {
        if show.get() && close_on_esc {
            let handler = wasm_bindgen::closure::Closure::<dyn Fn(web_sys::KeyboardEvent)>::new(
                move |e: web_sys::KeyboardEvent| {
                    if e.key() == "Escape" {
                        show.set(false);
                    }
                },
            );

            let window = web_sys::window().unwrap();
            let _ = window
                .add_event_listener_with_callback("keydown", handler.as_ref().unchecked_ref());
            handler.forget();
        }
    });

    let on_backdrop_click = move |_| {
        if close_on_backdrop {
            show.set(false);
        }
    };

    let on_close = move |_| {
        show.set(false);
    };

    let title_empty = title.is_empty();
    let title_display = store_value(title);
    let children_stored = store_value(children());
    let has_footer = footer.is_some();
    let footer_stored = store_value(footer);

    view! {
        <Show when=move || show.get()>
            <div class="modal-backdrop" on:click=on_backdrop_click>
                <div
                    class=format!("modal {} {}", size.class(), class)
                    on:click=|e| e.stop_propagation()
                    node_ref=modal_ref
                >
                    // Header
                    <Show when=move || !title_empty || show_close_button>
                        <div class="modal-header">
                            <Show when=move || !title_empty>
                                <h3 class="modal-title">{title_display.get_value()}</h3>
                            </Show>
                            <Show when=move || show_close_button>
                                <button class="btn btn-ghost modal-close" on:click=on_close>
                                    <IconX />
                                </button>
                            </Show>
                        </div>
                    </Show>

                    // Body
                    <div class="modal-body">
                        {children_stored.get_value()}
                    </div>

                    // Footer
                    <Show when=move || has_footer>
                        <div class="modal-footer">
                            {move || footer_stored.with_value(|f| f.as_ref().map(|f| f()))}
                        </div>
                    </Show>
                </div>
            </div>
        </Show>
    }
}

/// Confirmation dialog
#[component]
pub fn ConfirmDialog(
    #[prop(into)] show: RwSignal<bool>,
    #[prop(into)] title: String,
    #[prop(into)] message: String,
    #[prop(default = "Confirm".to_string())] confirm_text: String,
    #[prop(default = "Cancel".to_string())] cancel_text: String,
    #[prop(default = false)] danger: bool,
    #[prop(into)] on_confirm: Callback<()>,
    #[prop(optional, into)] on_cancel: Option<Callback<()>>,
) -> impl IntoView {
    let on_confirm_click = move |_| {
        on_confirm.call(());
        show.set(false);
    };

    let on_cancel_click = {
        let on_cancel = on_cancel.clone();
        move |_| {
            if let Some(cb) = on_cancel.as_ref() {
                cb.call(());
            }
            show.set(false);
        }
    };

    let confirm_text_stored = store_value(confirm_text);

    view! {
        <Modal show=show size=ModalSize::Small>
            <div class="confirm-dialog">
                <div class=move || format!("confirm-icon {}", if danger { "danger" } else { "" })>
                    <Show
                        when=move || danger
                        fallback=|| view! { <IconAlertCircle size=IconSize::Xl /> }
                    >
                        <IconAlertTriangle size=IconSize::Xl />
                    </Show>
                </div>
                <h3 class="confirm-title">{title}</h3>
                <p class="confirm-message">{message}</p>
                <div class="confirm-actions">
                    <button class="btn btn-ghost" on:click=on_cancel_click>
                        {cancel_text}
                    </button>
                    <button
                        class=move || format!("btn {}", if danger { "btn-danger" } else { "btn-primary" })
                        on:click=on_confirm_click
                    >
                        {confirm_text_stored.get_value()}
                    </button>
                </div>
            </div>
        </Modal>
    }
}

/// Alert dialog (just displays message with OK button)
#[component]
pub fn AlertDialog(
    #[prop(into)] show: RwSignal<bool>,
    #[prop(into)] title: String,
    #[prop(into)] message: String,
    #[prop(default = "OK".to_string())] button_text: String,
    #[prop(default = false)] is_error: bool,
) -> impl IntoView {
    let on_ok = move |_| {
        show.set(false);
    };

    view! {
        <Modal show=show size=ModalSize::Small>
            <div class="alert-dialog">
                <div class=move || format!("alert-icon {}", if is_error { "error" } else { "info" })>
                    <Show
                        when=move || is_error
                        fallback=|| view! { <IconInfo size=IconSize::Xl /> }
                    >
                        <IconXCircle size=IconSize::Xl />
                    </Show>
                </div>
                <h3 class="alert-title">{title}</h3>
                <p class="alert-message">{message}</p>
                <div class="alert-actions">
                    <button class="btn btn-primary" on:click=on_ok>
                        {button_text}
                    </button>
                </div>
            </div>
        </Modal>
    }
}

/// Drawer component (slides in from side)
#[component]
pub fn Drawer(
    #[prop(into)] show: RwSignal<bool>,
    #[prop(optional, into)] title: String,
    #[prop(default = "right")] position: &'static str,
    #[prop(default = true)] show_close_button: bool,
    #[prop(optional, into)] class: String,
    children: Children,
) -> impl IntoView {
    let on_backdrop_click = move |_| {
        show.set(false);
    };

    let on_close = move |_| {
        show.set(false);
    };

    let title_empty = title.is_empty();
    let title_display = store_value(title);
    let children_stored = store_value(children());

    view! {
        <Show when=move || show.get()>
            <div class="drawer-backdrop" on:click=on_backdrop_click>
                <div
                    class=format!("drawer drawer-{} {} {}", position, if show.get() { "open" } else { "" }, class)
                    on:click=|e| e.stop_propagation()
                >
                    // Header
                    <div class="drawer-header">
                        <Show when=move || !title_empty>
                            <h3 class="drawer-title">{title_display.get_value()}</h3>
                        </Show>
                        <Show when=move || show_close_button>
                            <button class="btn btn-ghost drawer-close" on:click=on_close>
                                <IconX />
                            </button>
                        </Show>
                    </div>

                    // Body
                    <div class="drawer-body">
                        {children_stored.get_value()}
                    </div>
                </div>
            </div>
        </Show>
    }
}
