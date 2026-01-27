//! Profile Settings Page

use leptos::*;
use leptos_router::*;

use crate::components::{forms::*, icons::*, spinner::*};
use crate::state::use_app_state;

/// Profile settings page
#[component]
pub fn ProfileSettingsPage() -> impl IntoView {
    let state = use_app_state();

    let name = create_rw_signal(String::new());
    let email = create_rw_signal(String::new());
    let saving = create_rw_signal(false);
    let (error, set_error) = create_signal::<Option<String>>(None);

    // Clone state for different usages
    let state_for_effect = state.clone();
    let state_for_submit = state.clone();
    let state_for_view = state.clone();

    // Load current user data
    create_effect(move |_| {
        if let Some(user) = state_for_effect.user.get() {
            name.set(user.name);
            email.set(user.email);
        }
    });

    let on_submit = move |e: web_sys::SubmitEvent| {
        e.prevent_default();
        saving.set(true);
        set_error.set(None);

        // In a real app, we'd call an API to update the profile
        // For now, just show success
        let state = state_for_submit.clone();
        set_timeout(
            move || {
                state.toast_success("Saved", "Profile updated successfully");
                saving.set(false);
            },
            std::time::Duration::from_millis(500),
        );
    };

    view! {
        <div class="page settings-profile-page">
            <div class="page-header">
                <div class="header-breadcrumb">
                    <A href="/settings" class="btn btn-ghost btn-sm">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Settings"</span>
                    </A>
                </div>
                <h1>"Profile"</h1>
                <p class="page-subtitle">"Update your personal information"</p>
            </div>

            <form class="settings-form" on:submit=on_submit>
                <div class="card">
                    <div class="card-header">
                        <h2>"Personal Information"</h2>
                    </div>
                    <div class="card-body">
                        <Show when=move || error.get().is_some()>
                            <div class="alert alert-error">
                                <IconAlertCircle size=IconSize::Sm />
                                <span>{move || error.get().unwrap_or_default()}</span>
                            </div>
                        </Show>

                        <div class="form-grid">
                            <TextInput
                                value=name
                                label="Full Name"
                                placeholder="Your name"
                                required=true
                            />

                            <TextInput
                                value=email
                                input_type=InputType::Email
                                label="Email"
                                placeholder="you@example.com"
                                required=true
                                disabled=true
                                helper_text="Contact support to change your email"
                            />
                        </div>

                        // Avatar section
                        <div class="avatar-section">
                            <label class="form-label">"Profile Picture"</label>
                            <div class="avatar-preview">
                                <div class="avatar-large">
                                    {move || name.get().chars().next().unwrap_or('U').to_string()}
                                </div>
                                <div class="avatar-actions">
                                    <button type="button" class="btn btn-secondary btn-sm" disabled>
                                        <IconUpload size=IconSize::Sm />
                                        <span>"Upload"</span>
                                    </button>
                                    <p class="form-helper">"Contact admin to update avatar"</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer">
                        <button
                            type="submit"
                            class="btn btn-primary"
                            disabled=move || saving.get()
                        >
                            <Show when=move || saving.get() fallback=|| "Save Changes">
                                <Spinner size=SpinnerSize::Sm />
                                <span>"Saving..."</span>
                            </Show>
                        </button>
                    </div>
                </div>
            </form>

            // Account info
            <div class="card">
                <div class="card-header">
                    <h2>"Account Information"</h2>
                </div>
                <div class="card-body">
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">"Role"</span>
                            <span class="info-value">
                                {move || state_for_view.user.get().map(|u| format!("{:?}", u.role)).unwrap_or_default()}
                            </span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">"Member Since"</span>
                            <span class="info-value">
                                {move || state_for_view.user.get().map(|u| u.created_at.format("%b %d, %Y").to_string()).unwrap_or_default()}
                            </span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">"User ID"</span>
                            <code class="info-value">
                                {move || state_for_view.user.get().map(|u| u.id).unwrap_or_default()}
                            </code>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    }
}
