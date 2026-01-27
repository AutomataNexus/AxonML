//! Form Input Components

use crate::components::icons::*;
use leptos::*;
use wasm_bindgen::JsCast;

/// Input field types
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InputType {
    #[default]
    Text,
    Email,
    Password,
    Number,
    Tel,
    Url,
    Search,
    Date,
    Time,
    DateTime,
}

impl InputType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Email => "email",
            Self::Password => "password",
            Self::Number => "number",
            Self::Tel => "tel",
            Self::Url => "url",
            Self::Search => "search",
            Self::Date => "date",
            Self::Time => "time",
            Self::DateTime => "datetime-local",
        }
    }
}

/// Text input component
#[component]
pub fn TextInput(
    #[prop(into)] value: RwSignal<String>,
    #[prop(default = InputType::Text)] input_type: InputType,
    #[prop(optional, into)] label: String,
    #[prop(optional, into)] placeholder: String,
    #[prop(optional, into)] helper_text: String,
    #[prop(optional, into)] error: MaybeSignal<Option<String>>,
    #[prop(default = false)] required: bool,
    #[prop(default = false)] disabled: bool,
    #[prop(default = false)] readonly: bool,
    #[prop(optional)] icon: Option<View>,
    #[prop(optional, into)] class: String,
    #[prop(optional)] on_blur: Option<Callback<()>>,
) -> impl IntoView {
    let (show_password, set_show_password) = create_signal(false);
    let is_password = input_type == InputType::Password;

    // Store values for use in closures
    let label_empty = label.is_empty();
    let label_display = store_value(label);
    let helper_empty = helper_text.is_empty();
    let helper_display = store_value(helper_text);
    let has_icon = icon.is_some();
    let icon_stored = store_value(icon);
    let on_blur_stored = store_value(on_blur);

    // Store error MaybeSignal for use in closures
    let error_stored = store_value(error);

    let actual_type = move || {
        if is_password && show_password.get() {
            "text"
        } else {
            input_type.as_str()
        }
    };

    view! {
        <div class=format!("form-group {}", class)>
            <Show when=move || !label_empty>
                <label class="form-label">
                    {label_display.get_value()}
                    <Show when=move || required>
                        <span class="required-mark">"*"</span>
                    </Show>
                </label>
            </Show>

            <div class=move || format!("input-wrapper {}", if error_stored.with_value(|e| e.get().is_some()) { "has-error" } else { "" })>
                <Show when=move || has_icon>
                    <span class="input-icon">{icon_stored.get_value()}</span>
                </Show>

                <input
                    type=actual_type
                    class="form-input"
                    placeholder=placeholder
                    required=required
                    disabled=disabled
                    readonly=readonly
                    prop:value=move || value.get()
                    on:input=move |e| {
                        value.set(event_target_value(&e));
                    }
                    on:blur=move |_| {
                        if let Some(cb_opt) = on_blur_stored.try_with_value(|cb| cb.clone()) {
                            if let Some(cb) = cb_opt {
                                cb.call(());
                            }
                        }
                    }
                />

                <Show when=move || is_password>
                    <button
                        type="button"
                        class="input-toggle-password"
                        on:click=move |_| set_show_password.update(|v| *v = !*v)
                    >
                        <Show
                            when=move || show_password.get()
                            fallback=|| view! { <IconEye size=IconSize::Sm /> }
                        >
                            <IconEyeOff size=IconSize::Sm />
                        </Show>
                    </button>
                </Show>
            </div>

            <Show when=move || error_stored.with_value(|e| e.get().is_some())>
                <p class="form-error">{move || error_stored.with_value(|e| e.get().unwrap_or_default())}</p>
            </Show>

            <Show when=move || !helper_empty && error_stored.with_value(|e| e.get().is_none())>
                <p class="form-helper">{helper_display.get_value()}</p>
            </Show>
        </div>
    }
}

/// Textarea component
#[component]
pub fn TextArea(
    #[prop(into)] value: RwSignal<String>,
    #[prop(optional, into)] label: String,
    #[prop(optional, into)] placeholder: String,
    #[prop(optional, into)] helper_text: String,
    #[prop(optional, into)] error: MaybeSignal<Option<String>>,
    #[prop(default = 4)] rows: u32,
    #[prop(default = false)] required: bool,
    #[prop(default = false)] disabled: bool,
    #[prop(default = false)] readonly: bool,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let label_empty = label.is_empty();
    let label_display = store_value(label);
    let helper_empty = helper_text.is_empty();
    let helper_display = store_value(helper_text);

    // Store error MaybeSignal for use in closures
    let error_stored = store_value(error);

    view! {
        <div class=format!("form-group {}", class)>
            <Show when=move || !label_empty>
                <label class="form-label">
                    {label_display.get_value()}
                    <Show when=move || required>
                        <span class="required-mark">"*"</span>
                    </Show>
                </label>
            </Show>

            <textarea
                class=move || format!("form-textarea {}", if error_stored.with_value(|e| e.get().is_some()) { "has-error" } else { "" })
                placeholder=placeholder
                rows=rows
                required=required
                disabled=disabled
                readonly=readonly
                prop:value=move || value.get()
                on:input=move |e| {
                    value.set(event_target_value(&e));
                }
            />

            <Show when=move || error_stored.with_value(|e| e.get().is_some())>
                <p class="form-error">{move || error_stored.with_value(|e| e.get().unwrap_or_default())}</p>
            </Show>

            <Show when=move || !helper_empty && error_stored.with_value(|e| e.get().is_none())>
                <p class="form-helper">{helper_display.get_value()}</p>
            </Show>
        </div>
    }
}

/// Select dropdown component
#[component]
pub fn Select<T: Clone + PartialEq + 'static>(
    #[prop(into)] value: RwSignal<T>,
    #[prop(into)] options: Vec<(T, String)>,
    #[prop(optional, into)] label: String,
    #[prop(optional, into)] placeholder: String,
    #[prop(optional, into)] helper_text: String,
    #[prop(optional, into)] error: MaybeSignal<Option<String>>,
    #[prop(default = false)] required: bool,
    #[prop(default = false)] disabled: bool,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let label_empty = label.is_empty();
    let label_display = store_value(label);
    let placeholder_empty = placeholder.is_empty();
    let placeholder_display = store_value(placeholder);
    let helper_empty = helper_text.is_empty();
    let helper_display = store_value(helper_text);
    let options_for_change = options.clone();

    // Store error MaybeSignal for use in closures
    let error_stored = store_value(error);

    view! {
        <div class=format!("form-group {}", class)>
            <Show when=move || !label_empty>
                <label class="form-label">
                    {label_display.get_value()}
                    <Show when=move || required>
                        <span class="required-mark">"*"</span>
                    </Show>
                </label>
            </Show>

            <div class="select-wrapper">
                <select
                    class=move || format!("form-select {}", if error_stored.with_value(|e| e.get().is_some()) { "has-error" } else { "" })
                    required=required
                    disabled=disabled
                    on:change=move |e| {
                        let idx: usize = event_target_value(&e).parse().unwrap_or(0);
                        if let Some((val, _)) = options_for_change.get(idx) {
                            value.set(val.clone());
                        }
                    }
                >
                    <Show when=move || !placeholder_empty>
                        <option value="" disabled selected>{placeholder_display.get_value()}</option>
                    </Show>
                    {options.iter().enumerate().map(|(i, (_, opt_label))| {
                        view! {
                            <option value=i>{opt_label.clone()}</option>
                        }
                    }).collect_view()}
                </select>
                <IconChevronDown size=IconSize::Sm class="select-icon".to_string() />
            </div>

            <Show when=move || error_stored.with_value(|e| e.get().is_some())>
                <p class="form-error">{move || error_stored.with_value(|e| e.get().unwrap_or_default())}</p>
            </Show>

            <Show when=move || !helper_empty && error_stored.with_value(|e| e.get().is_none())>
                <p class="form-helper">{helper_display.get_value()}</p>
            </Show>
        </div>
    }
}

/// Checkbox component
#[component]
pub fn Checkbox(
    #[prop(into)] checked: RwSignal<bool>,
    #[prop(into)] label: String,
    #[prop(optional, into)] helper_text: String,
    #[prop(default = false)] disabled: bool,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let helper_empty = helper_text.is_empty();
    let helper_display = store_value(helper_text);

    view! {
        <div class=format!("form-checkbox {}", class)>
            <label class="checkbox-label">
                <input
                    type="checkbox"
                    class="checkbox-input"
                    disabled=disabled
                    prop:checked=move || checked.get()
                    on:change=move |e| {
                        checked.set(event_target_checked(&e));
                    }
                />
                <span class="checkbox-box">
                    <IconCheck size=IconSize::Sm />
                </span>
                <span class="checkbox-text">{label}</span>
            </label>
            <Show when=move || !helper_empty>
                <p class="form-helper">{helper_display.get_value()}</p>
            </Show>
        </div>
    }
}

/// Toggle switch component
#[component]
pub fn Toggle(
    #[prop(into)] checked: RwSignal<bool>,
    #[prop(optional, into)] label: String,
    #[prop(optional, into)] helper_text: String,
    #[prop(default = false)] disabled: bool,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let label_empty = label.is_empty();
    let label_display = store_value(label);
    let helper_empty = helper_text.is_empty();
    let helper_display = store_value(helper_text);

    view! {
        <div class=format!("form-toggle {}", class)>
            <label class="toggle-label">
                <input
                    type="checkbox"
                    class="toggle-input"
                    disabled=disabled
                    prop:checked=move || checked.get()
                    on:change=move |e| {
                        checked.set(event_target_checked(&e));
                    }
                />
                <span class="toggle-switch"></span>
                <Show when=move || !label_empty>
                    <span class="toggle-text">{label_display.get_value()}</span>
                </Show>
            </label>
            <Show when=move || !helper_empty>
                <p class="form-helper">{helper_display.get_value()}</p>
            </Show>
        </div>
    }
}

/// Radio button group
#[component]
pub fn RadioGroup<T: Clone + PartialEq + 'static>(
    #[prop(into)] value: RwSignal<T>,
    #[prop(into)] options: Vec<(T, String)>,
    #[prop(into)] name: String,
    #[prop(optional, into)] label: String,
    #[prop(optional, into)] helper_text: String,
    #[prop(default = false)] disabled: bool,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let label_empty = label.is_empty();
    let label_display = store_value(label);
    let helper_empty = helper_text.is_empty();
    let helper_display = store_value(helper_text);

    view! {
        <div class=format!("form-radio-group {}", class)>
            <Show when=move || !label_empty>
                <label class="form-label">{label_display.get_value()}</label>
            </Show>

            <div class="radio-options">
                {options.into_iter().map(|(val, text)| {
                    let val_clone = val.clone();
                    let val_for_check = val.clone();
                    let name_clone = name.clone();

                    view! {
                        <label class="radio-label">
                            <input
                                type="radio"
                                class="radio-input"
                                name=name_clone
                                disabled=disabled
                                checked=move || value.get() == val_for_check
                                on:change=move |_| {
                                    value.set(val_clone.clone());
                                }
                            />
                            <span class="radio-circle"></span>
                            <span class="radio-text">{text}</span>
                        </label>
                    }
                }).collect_view()}
            </div>

            <Show when=move || !helper_empty>
                <p class="form-helper">{helper_display.get_value()}</p>
            </Show>
        </div>
    }
}

/// File input component
#[component]
pub fn FileInput(
    #[prop(into)] on_select: Callback<web_sys::FileList>,
    #[prop(optional, into)] label: String,
    #[prop(optional, into)] accept: String,
    #[prop(default = false)] multiple: bool,
    #[prop(default = false)] required: bool,
    #[prop(default = false)] disabled: bool,
    #[prop(optional, into)] helper_text: String,
    #[prop(optional, into)] error: MaybeSignal<Option<String>>,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let (dragging, set_dragging) = create_signal(false);

    // Generate unique ID for the input using random number
    let input_id = format!("file-input-{}", js_sys::Math::random().to_bits());
    let input_id_for_label = input_id.clone();

    let label_empty = label.is_empty();
    let label_display = store_value(label);
    let accept_empty = accept.is_empty();
    let accept_display = store_value(accept.clone());
    let helper_empty = helper_text.is_empty();
    let helper_display = store_value(helper_text);

    let on_select_for_drop = on_select.clone();

    // Store error MaybeSignal for use in closures
    let error_stored = store_value(error);

    let on_file_change = move |e: web_sys::Event| {
        let input: web_sys::HtmlInputElement = event_target(&e);
        if let Some(files) = input.files() {
            if files.length() > 0 {
                on_select.call(files);
            }
        }
    };

    let on_drop = move |e: web_sys::DragEvent| {
        e.prevent_default();
        set_dragging.set(false);
        if let Some(dt) = e.data_transfer() {
            if let Some(files) = dt.files() {
                if files.length() > 0 {
                    on_select_for_drop.call(files);
                }
            }
        }
    };

    view! {
        <div class=format!("form-group {}", class)>
            <Show when=move || !label_empty>
                <span class="form-label">
                    {label_display.get_value()}
                    <Show when=move || required>
                        <span class="required-mark">"*"</span>
                    </Show>
                </span>
            </Show>

            // Use label element to trigger file input - this is the standard HTML approach
            <label
                for=input_id_for_label
                class=move || format!(
                    "file-dropzone {} {}",
                    if dragging.get() { "dragging" } else { "" },
                    if error_stored.with_value(|e| e.get().is_some()) { "has-error" } else { "" }
                )
                on:dragover=move |e: web_sys::DragEvent| {
                    e.prevent_default();
                    set_dragging.set(true);
                }
                on:dragleave=move |_| set_dragging.set(false)
                on:drop=on_drop
            >
                <input
                    type="file"
                    id=input_id
                    class="file-input-hidden"
                    accept=accept
                    multiple=multiple
                    required=required
                    disabled=disabled
                    on:change=on_file_change
                />
                <div class="dropzone-content">
                    <IconUpload size=IconSize::Xl class="text-muted".to_string() />
                    <p class="dropzone-text">"Drop files here or click to upload"</p>
                    <Show when=move || !accept_empty>
                        <p class="dropzone-hint">{move || format!("Accepted: {}", accept_display.get_value())}</p>
                    </Show>
                </div>
            </label>

            <Show when=move || error_stored.with_value(|e| e.get().is_some())>
                <p class="form-error">{move || error_stored.with_value(|e| e.get().unwrap_or_default())}</p>
            </Show>

            <Show when=move || !helper_empty && error_stored.with_value(|e| e.get().is_none())>
                <p class="form-helper">{helper_display.get_value()}</p>
            </Show>
        </div>
    }
}

/// Number input with increment/decrement buttons
#[component]
pub fn NumberInput(
    #[prop(into)] value: RwSignal<f64>,
    #[prop(optional, into)] label: String,
    #[prop(optional)] min: Option<f64>,
    #[prop(optional)] max: Option<f64>,
    #[prop(default = 1.0)] step: f64,
    #[prop(optional, into)] helper_text: String,
    #[prop(optional, into)] error: MaybeSignal<Option<String>>,
    #[prop(default = false)] required: bool,
    #[prop(default = false)] disabled: bool,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let label_empty = label.is_empty();
    let label_display = store_value(label);
    let helper_empty = helper_text.is_empty();
    let helper_display = store_value(helper_text);

    let increment = move |_| {
        value.update(|v| {
            let new_val = *v + step;
            *v = max.map(|m| new_val.min(m)).unwrap_or(new_val);
        });
    };

    let decrement = move |_| {
        value.update(|v| {
            let new_val = *v - step;
            *v = min.map(|m| new_val.max(m)).unwrap_or(new_val);
        });
    };

    // Store error MaybeSignal for use in closures
    let error_stored = store_value(error);

    view! {
        <div class=format!("form-group {}", class)>
            <Show when=move || !label_empty>
                <label class="form-label">
                    {label_display.get_value()}
                    <Show when=move || required>
                        <span class="required-mark">"*"</span>
                    </Show>
                </label>
            </Show>

            <div class=move || format!("number-input-wrapper {}", if error_stored.with_value(|e| e.get().is_some()) { "has-error" } else { "" })>
                <button
                    type="button"
                    class="number-btn"
                    disabled=disabled
                    on:click=decrement
                >
                    <IconMinus size=IconSize::Sm />
                </button>
                <input
                    type="number"
                    class="form-input number-input"
                    min=min
                    max=max
                    step=step
                    required=required
                    disabled=disabled
                    prop:value=move || value.get()
                    on:input=move |e| {
                        if let Ok(v) = event_target_value(&e).parse() {
                            value.set(v);
                        }
                    }
                />
                <button
                    type="button"
                    class="number-btn"
                    disabled=disabled
                    on:click=increment
                >
                    <IconPlus size=IconSize::Sm />
                </button>
            </div>

            <Show when=move || error_stored.with_value(|e| e.get().is_some())>
                <p class="form-error">{move || error_stored.with_value(|e| e.get().unwrap_or_default())}</p>
            </Show>

            <Show when=move || !helper_empty && error_stored.with_value(|e| e.get().is_none())>
                <p class="form-helper">{helper_display.get_value()}</p>
            </Show>
        </div>
    }
}

/// Code input for TOTP codes
#[component]
pub fn CodeInput(
    #[prop(into)] value: RwSignal<String>,
    #[prop(default = 6)] length: usize,
    #[prop(optional, into)] label: String,
    #[prop(optional, into)] error: MaybeSignal<Option<String>>,
    #[prop(optional)] on_complete: Option<Callback<String>>,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let (digits, set_digits) = create_signal(vec!["".to_string(); length]);

    let label_empty = label.is_empty();
    let label_display = store_value(label);

    // Update the combined value when digits change
    create_effect(move |_| {
        let combined: String = digits.get().concat();
        value.set(combined.clone());
        if combined.len() == length {
            if let Some(cb) = on_complete.as_ref() {
                cb.call(combined);
            }
        }
    });

    // Store error MaybeSignal for use in closures
    let error_stored = store_value(error);

    view! {
        <div class=format!("form-group {}", class)>
            <Show when=move || !label_empty>
                <label class="form-label">{label_display.get_value()}</label>
            </Show>

            <div class=move || format!("code-input-wrapper {}", if error_stored.with_value(|e| e.get().is_some()) { "has-error" } else { "" })>
                {(0..length).map(|i| {
                    let id = format!("code-{}", i);
                    view! {
                        <input
                            type="text"
                            maxlength="1"
                            class="code-digit"
                            id=id.clone()
                            prop:value=move || digits.get().get(i).cloned().unwrap_or_default()
                            on:input=move |e| {
                                let val = event_target_value(&e);
                                let char = val.chars().last().filter(|c| c.is_ascii_digit()).map(|c| c.to_string()).unwrap_or_default();

                                set_digits.update(|d| {
                                    if i < d.len() {
                                        d[i] = char.clone();
                                    }
                                });

                                // Auto-focus next input
                                if !char.is_empty() && i < length - 1 {
                                    if let Some(window) = web_sys::window() {
                                        if let Some(doc) = window.document() {
                                            let next_id = format!("code-{}", i + 1);
                                            if let Some(next) = doc.get_element_by_id(&next_id) {
                                                if let Some(input) = next.dyn_ref::<web_sys::HtmlInputElement>() {
                                                    let _ = input.focus();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            on:keydown=move |e: web_sys::KeyboardEvent| {
                                // Handle backspace
                                if e.key() == "Backspace" && digits.get().get(i).map(|s| s.is_empty()).unwrap_or(true) && i > 0 {
                                    if let Some(window) = web_sys::window() {
                                        if let Some(doc) = window.document() {
                                            let prev_id = format!("code-{}", i - 1);
                                            if let Some(prev) = doc.get_element_by_id(&prev_id) {
                                                if let Some(input) = prev.dyn_ref::<web_sys::HtmlInputElement>() {
                                                    let _ = input.focus();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        />
                    }
                }).collect_view()}
            </div>

            <Show when=move || error_stored.with_value(|e| e.get().is_some())>
                <p class="form-error">{move || error_stored.with_value(|e| e.get().unwrap_or_default())}</p>
            </Show>
        </div>
    }
}
