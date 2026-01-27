//! Security Settings Page

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::components::{forms::*, icons::*, modal::*, spinner::*};
use crate::state::use_app_state;

/// Security settings page
#[component]
pub fn SecuritySettingsPage() -> impl IntoView {
    let state = use_app_state();

    let loading = create_rw_signal(true);
    let totp_enabled = create_rw_signal(false);
    let webauthn_enabled = create_rw_signal(false);

    // Password change state
    let show_password_modal = create_rw_signal(false);
    let current_password = create_rw_signal(String::new());
    let new_password = create_rw_signal(String::new());
    let confirm_password = create_rw_signal(String::new());
    let changing_password = create_rw_signal(false);
    let password_error = create_rw_signal::<Option<String>>(None);

    // Disable TOTP modal
    let show_disable_totp = create_rw_signal(false);

    // Clone state for different closures
    let state_for_effect = state.clone();
    let state_for_password = state.clone();
    let state_for_totp = state.clone();

    // Load user MFA status
    create_effect(move |_| {
        if let Some(user) = state_for_effect.user.get() {
            totp_enabled.set(user.totp_enabled);
            webauthn_enabled.set(user.webauthn_enabled);
        }
        loading.set(false);
    });

    let change_password = move |_| {
        let current = current_password.get();
        let new_pw = new_password.get();
        let confirm = confirm_password.get();

        if new_pw != confirm {
            password_error.set(Some("Passwords do not match".to_string()));
            return;
        }

        if new_pw.len() < 8 {
            password_error.set(Some("Password must be at least 8 characters".to_string()));
            return;
        }

        changing_password.set(true);
        password_error.set(None);

        let state = state_for_password.clone();
        spawn_local(async move {
            match api::auth::change_password(&current, &new_pw).await {
                Ok(_) => {
                    state.toast_success("Password Changed", "Your password has been updated");
                    show_password_modal.set(false);
                    current_password.set(String::new());
                    new_password.set(String::new());
                    confirm_password.set(String::new());
                }
                Err(e) => {
                    password_error.set(Some(e.message));
                }
            }
            changing_password.set(false);
        });
    };

    let disable_totp = move |_| {
        let state = state_for_totp.clone();
        spawn_local(async move {
            match api::auth::totp_disable().await {
                Ok(_) => {
                    state.toast_success(
                        "TOTP Disabled",
                        "Two-factor authentication has been disabled",
                    );
                    totp_enabled.set(false);
                }
                Err(e) => {
                    state.toast_error("Error", e.message);
                }
            }
        });
        show_disable_totp.set(false);
    };

    // Store closures for use in Modal footers
    let change_password_stored = store_value(change_password);
    let disable_totp_stored = store_value(disable_totp);

    view! {
        <div class="page security-settings-page">
            <div class="page-header">
                <div class="header-breadcrumb">
                    <A href="/settings" class="btn btn-ghost btn-sm">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Settings"</span>
                    </A>
                </div>
                <h1>"Security"</h1>
                <p class="page-subtitle">"Manage your password and two-factor authentication"</p>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                </div>
            </Show>

            <Show when=move || !loading.get()>
                // Password Section
                <div class="card security-section">
                    <div class="card-header">
                        <div class="section-title">
                            <IconLock size=IconSize::Md />
                            <h2>"Password"</h2>
                        </div>
                    </div>
                    <div class="card-body">
                        <p class="section-description">
                            "Use a strong password that you don't use elsewhere."
                        </p>
                        <button
                            class="btn btn-secondary"
                            on:click=move |_| show_password_modal.set(true)
                        >
                            "Change Password"
                        </button>
                    </div>
                </div>

                // Two-Factor Authentication Section
                <div class="card security-section">
                    <div class="card-header">
                        <div class="section-title">
                            <IconShield size=IconSize::Md />
                            <h2>"Two-Factor Authentication"</h2>
                        </div>
                    </div>
                    <div class="card-body">
                        <p class="section-description">
                            "Add an extra layer of security to your account by requiring a second form of verification."
                        </p>

                        <div class="mfa-options">
                            // TOTP
                            <div class="mfa-option">
                                <div class="mfa-option-info">
                                    <div class="mfa-option-icon">
                                        <IconSmartphone size=IconSize::Md />
                                    </div>
                                    <div class="mfa-option-text">
                                        <h4>"Authenticator App"</h4>
                                        <p>"Use an app like Google Authenticator or Authy"</p>
                                    </div>
                                </div>
                                <div class="mfa-option-action">
                                    <Show
                                        when=move || totp_enabled.get()
                                        fallback=|| view! {
                                            <A href="/settings/security/totp" class="btn btn-primary btn-sm">
                                                "Set Up"
                                            </A>
                                        }
                                    >
                                        <span class="badge badge-success">"Enabled"</span>
                                        <button
                                            class="btn btn-danger-outline btn-sm"
                                            on:click=move |_| show_disable_totp.set(true)
                                        >
                                            "Disable"
                                        </button>
                                    </Show>
                                </div>
                            </div>

                            // WebAuthn
                            <div class="mfa-option">
                                <div class="mfa-option-info">
                                    <div class="mfa-option-icon">
                                        <IconFingerprint size=IconSize::Md />
                                    </div>
                                    <div class="mfa-option-text">
                                        <h4>"Security Key"</h4>
                                        <p>"Use a hardware key or biometric"</p>
                                    </div>
                                </div>
                                <div class="mfa-option-action">
                                    <Show
                                        when=move || webauthn_enabled.get()
                                        fallback=|| view! {
                                            <A href="/settings/security/webauthn" class="btn btn-secondary btn-sm">
                                                "Add Key"
                                            </A>
                                        }
                                    >
                                        <span class="badge badge-success">"Enabled"</span>
                                        <A href="/settings/security/webauthn" class="btn btn-ghost btn-sm">
                                            "Manage"
                                        </A>
                                    </Show>
                                </div>
                            </div>

                            // Recovery Codes
                            <div class="mfa-option">
                                <div class="mfa-option-info">
                                    <div class="mfa-option-icon">
                                        <IconKey size=IconSize::Md />
                                    </div>
                                    <div class="mfa-option-text">
                                        <h4>"Recovery Codes"</h4>
                                        <p>"Backup codes for when you lose access"</p>
                                    </div>
                                </div>
                                <div class="mfa-option-action">
                                    <Show when=move || totp_enabled.get() || webauthn_enabled.get()>
                                        <A href="/settings/security/recovery" class="btn btn-ghost btn-sm">
                                            "View Codes"
                                        </A>
                                    </Show>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                // Sessions Section
                <div class="card security-section">
                    <div class="card-header">
                        <div class="section-title">
                            <IconActivity size=IconSize::Md />
                            <h2>"Active Sessions"</h2>
                        </div>
                    </div>
                    <div class="card-body">
                        <p class="section-description">
                            "Manage your active sessions across devices."
                        </p>
                        <div class="session-list">
                            <div class="session-item current">
                                <div class="session-info">
                                    <span class="session-device">"Current Session"</span>
                                    <span class="session-location">"This browser"</span>
                                </div>
                                <span class="badge badge-success">"Active"</span>
                            </div>
                        </div>
                        <button class="btn btn-danger-outline btn-sm" disabled>
                            "Sign Out All Other Sessions"
                        </button>
                    </div>
                </div>
            </Show>

            // Password Change Modal
            <Modal
                show=show_password_modal
                title="Change Password"
                size=ModalSize::Small
                footer=std::rc::Rc::new(move || view! {
                    <button class="btn btn-ghost" on:click=move |_| show_password_modal.set(false)>
                        "Cancel"
                    </button>
                    <button
                        class="btn btn-primary"
                        disabled=move || changing_password.get()
                        on:click=move |e| change_password_stored.with_value(|f| f(e))
                    >
                        <Show when=move || changing_password.get() fallback=|| "Change Password">
                            <Spinner size=SpinnerSize::Sm />
                            <span>"Changing..."</span>
                        </Show>
                    </button>
                }.into_view().into())
            >
                <Show when=move || password_error.get().is_some()>
                    <div class="alert alert-error">
                        <IconAlertCircle size=IconSize::Sm />
                        <span>{move || password_error.get().unwrap_or_default()}</span>
                    </div>
                </Show>

                <TextInput
                    value=current_password
                    input_type=InputType::Password
                    label="Current Password"
                    required=true
                />

                <TextInput
                    value=new_password
                    input_type=InputType::Password
                    label="New Password"
                    helper_text="At least 8 characters"
                    required=true
                />

                <TextInput
                    value=confirm_password
                    input_type=InputType::Password
                    label="Confirm New Password"
                    required=true
                />
            </Modal>

            // Disable TOTP Modal
            <ConfirmDialog
                show=show_disable_totp
                title="Disable Authenticator App?".to_string()
                message="This will remove two-factor authentication from your account. You can re-enable it at any time.".to_string()
                confirm_text="Disable".to_string()
                danger=true
                on_confirm=move |_| disable_totp_stored.with_value(|f| f(()))
            />
        </div>
    }
}
