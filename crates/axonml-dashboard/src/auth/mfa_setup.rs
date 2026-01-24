//! MFA Setup Pages (TOTP, WebAuthn, Recovery Codes)

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::state::use_app_state;
use crate::types::*;
use crate::components::{forms::*, icons::*, spinner::*, modal::*};
use crate::utils::webauthn;

/// TOTP Setup Page
#[component]
pub fn TotpSetupPage() -> impl IntoView {
    let state = use_app_state();
    let _navigate = use_navigate(); // Uses link-based navigation instead

    let loading = create_rw_signal(true);
    let (setup_data, set_setup_data) = create_signal::<Option<TotpSetupResponse>>(None);
    let verification_code = create_rw_signal(String::new());
    let verifying = create_rw_signal(false);
    let error = create_rw_signal::<Option<String>>(None);
    let (step, set_step) = create_signal(1u32);
    let (copied, set_copied) = create_signal(false);

    // Fetch TOTP setup data on mount
    create_effect(move |_| {
        spawn_local(async move {
            match api::auth::totp_setup().await {
                Ok(data) => {
                    set_setup_data.set(Some(data));
                }
                Err(e) => {
                    error.set(Some(e.message));
                }
            }
            loading.set(false);
        });
    });

    let copy_secret = {
        move |_| {
            if let Some(data) = setup_data.get() {
                let window = web_sys::window().unwrap();
                let nav = window.navigator();
                let clipboard = nav.clipboard();

                let secret = data.secret.clone();
                spawn_local(async move {
                    let promise = clipboard.write_text(&secret);
                    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
                    set_copied.set(true);
                    set_timeout(move || set_copied.set(false), std::time::Duration::from_secs(2));
                });
            }
        }
    };

    let do_verify = {
        let state = state.clone();
        move || {
            let code = verification_code.get();
            if code.len() != 6 {
                error.set(Some("Please enter a valid 6-digit code".to_string()));
                return;
            }

            verifying.set(true);
            error.set(None);

            let state = state.clone();
            spawn_local(async move {
                match api::auth::totp_enable(&code).await {
                    Ok(_) => {
                        set_step.set(3);
                        state.toast_success("TOTP Enabled", "Two-factor authentication is now active");
                    }
                    Err(e) => {
                        error.set(Some(e.message));
                    }
                }
                verifying.set(false);
            });
        }
    };

    let on_click_verify = {
        let do_verify = do_verify.clone();
        std::rc::Rc::new(move |_: web_sys::MouseEvent| {
            do_verify();
        })
    };
    let click_verify = on_click_verify.clone();

    let on_complete_verify = Callback::new({
        let do_verify = do_verify.clone();
        move |_: String| {
            do_verify();
        }
    });

    view! {
        <div class="mfa-setup-page">
            <div class="setup-container">
                <div class="setup-header">
                    <A href="/settings/security" class="btn btn-ghost">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Back to Security"</span>
                    </A>
                </div>

                <div class="setup-content">
                    <h1>"Set Up Authenticator App"</h1>

                    // Progress steps
                    <div class="setup-steps">
                        <div class=move || format!("setup-step {}", if step.get() >= 1 { "active" } else { "" })>
                            <span class="step-number">"1"</span>
                            <span class="step-label">"Scan QR"</span>
                        </div>
                        <div class="step-line"></div>
                        <div class=move || format!("setup-step {}", if step.get() >= 2 { "active" } else { "" })>
                            <span class="step-number">"2"</span>
                            <span class="step-label">"Verify"</span>
                        </div>
                        <div class="step-line"></div>
                        <div class=move || format!("setup-step {}", if step.get() >= 3 { "active" } else { "" })>
                            <span class="step-number">"3"</span>
                            <span class="step-label">"Done"</span>
                        </div>
                    </div>

                    <Show when=move || loading.get()>
                        <div class="loading-state">
                            <Spinner size=SpinnerSize::Lg />
                            <p>"Generating setup code..."</p>
                        </div>
                    </Show>

                    <Show when=move || error.get().is_some()>
                        <div class="alert alert-error">
                            <IconAlertCircle size=IconSize::Sm />
                            <span>{move || error.get().unwrap_or_default()}</span>
                        </div>
                    </Show>

                    // Step 1: Scan QR Code
                    <Show when=move || !loading.get() && step.get() == 1>
                        <div class="setup-step-content">
                            <p class="step-instructions">
                                "Scan this QR code with your authenticator app (Google Authenticator, Authy, 1Password, etc.)"
                            </p>

                            {move || setup_data.get().map(|data| {
                                view! {
                                    <div class="qr-container">
                                        <img
                                            src=format!("data:image/svg+xml;base64,{}", data.qr_code)
                                            alt="TOTP QR Code"
                                            class="qr-image"
                                        />
                                    </div>

                                    <div class="secret-display">
                                        <p class="secret-label">"Or enter this code manually:"</p>
                                        <div class="secret-value">
                                            <code>{data.secret.clone()}</code>
                                            <button
                                                class="btn btn-ghost btn-sm"
                                                on:click=copy_secret
                                            >
                                                {move || if copied.get() {
                                                    view! { <IconCheck size=IconSize::Sm /> }
                                                } else {
                                                    view! { <IconCopy size=IconSize::Sm /> }
                                                }}
                                            </button>
                                        </div>
                                    </div>
                                }
                            })}

                            <button
                                class="btn btn-primary"
                                on:click=move |_| set_step.set(2)
                            >
                                "Continue"
                                <IconArrowRight size=IconSize::Sm />
                            </button>
                        </div>
                    </Show>

                    // Step 2: Verify Code
                    <Show when=move || step.get() == 2>
                        <div class="setup-step-content">
                            <p class="step-instructions">
                                "Enter the 6-digit code from your authenticator app to verify setup"
                            </p>

                            <CodeInput
                                value=verification_code
                                length=6
                                label="Verification Code"
                                on_complete=on_complete_verify
                            />

                            <div class="step-actions">
                                <button
                                    class="btn btn-ghost"
                                    on:click=move |_| set_step.set(1)
                                >
                                    <IconArrowLeft size=IconSize::Sm />
                                    "Back"
                                </button>
                                <button
                                    class="btn btn-primary"
                                    disabled=move || verifying.get() || verification_code.get().len() != 6
                                    on:click={let f = click_verify.clone(); move |e| f(e)}
                                >
                                    <Show when=move || verifying.get() fallback=|| "Verify & Enable">
                                        <Spinner size=SpinnerSize::Sm />
                                        "Verifying..."
                                    </Show>
                                </button>
                            </div>
                        </div>
                    </Show>

                    // Step 3: Success + Backup Codes
                    <Show when=move || step.get() == 3>
                        <div class="setup-step-content setup-success">
                            <div class="success-icon">
                                <IconCheckCircle size=IconSize::Xl />
                            </div>
                            <h2>"Authenticator App Enabled!"</h2>
                            <p>"Two-factor authentication is now active on your account."</p>

                            {move || setup_data.get().map(|data| {
                                view! {
                                    <div class="backup-codes">
                                        <h3>"Save Your Backup Codes"</h3>
                                        <p class="backup-warning">
                                            <IconAlertTriangle size=IconSize::Sm />
                                            "Store these codes safely. Each code can only be used once."
                                        </p>
                                        <div class="codes-grid">
                                            {data.backup_codes.iter().map(|code| {
                                                view! { <code class="backup-code">{code.clone()}</code> }
                                            }).collect_view()}
                                        </div>
                                    </div>
                                }
                            })}

                            <A href="/settings/security" class="btn btn-primary">
                                "Done"
                            </A>
                        </div>
                    </Show>
                </div>
            </div>
        </div>
    }
}

/// WebAuthn Setup Page
#[component]
pub fn WebAuthnSetupPage() -> impl IntoView {
    let state = use_app_state();

    let loading = create_rw_signal(false);
    let error = create_rw_signal::<Option<String>>(None);
    let (success, set_success) = create_signal(false);
    let device_name = create_rw_signal(String::new());

    let start_registration = move |_| {
        if device_name.get().is_empty() {
            error.set(Some("Please enter a name for this device".to_string()));
            return;
        }

        // Check if WebAuthn is available
        if !webauthn::is_webauthn_available() {
            error.set(Some("WebAuthn is not supported in this browser".to_string()));
            return;
        }

        loading.set(true);
        error.set(None);

        let name = device_name.get();
        let user_id = state.user.get()
            .map(|u| u.id.clone())
            .unwrap_or_else(|| "user".to_string());
        let user_email = state.user.get()
            .map(|u| u.email.clone())
            .unwrap_or_else(|| "user@example.com".to_string());

        spawn_local(async move {
            // Step 1: Get challenge from server
            match api::auth::webauthn_register_start().await {
                Ok(challenge) => {
                    // Step 2: Create credential using WebAuthn API
                    let rp_id = web_sys::window()
                        .and_then(|w| w.location().hostname().ok())
                        .unwrap_or_else(|| "localhost".to_string());

                    match webauthn::create_credential(
                        &challenge.challenge,
                        "AxonML",
                        &rp_id,
                        &challenge.user_id.unwrap_or(user_id),
                        &user_email,
                        &name,
                    ).await {
                        Ok(registration) => {
                            // Step 3: Send credential to server for registration
                            let finish_request = api::auth::WebAuthnRegisterFinishRequest {
                                credential_id: registration.id,
                                attestation_object: registration.attestation_object,
                                client_data_json: registration.client_data_json,
                                device_name: name,
                            };

                            match api::auth::webauthn_register_finish(&finish_request).await {
                                Ok(_) => {
                                    set_success.set(true);
                                }
                                Err(e) => {
                                    error.set(Some(format!("Registration failed: {}", e.message)));
                                }
                            }
                        }
                        Err(webauthn::WebAuthnError::UserCancelled) => {
                            error.set(Some("Registration cancelled".to_string()));
                        }
                        Err(e) => {
                            error.set(Some(e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    error.set(Some(e.message));
                }
            }
            loading.set(false);
        });
    };

    view! {
        <div class="mfa-setup-page">
            <div class="setup-container">
                <div class="setup-header">
                    <A href="/settings/security" class="btn btn-ghost">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Back to Security"</span>
                    </A>
                </div>

                <div class="setup-content">
                    <h1>"Add Security Key"</h1>

                    <Show when=move || !success.get()>
                        <div class="webauthn-setup">
                            <div class="setup-icon">
                                <IconFingerprint size=IconSize::Xl />
                            </div>

                            <p class="step-instructions">
                                "Register a hardware security key, fingerprint, or Face ID for passwordless authentication."
                            </p>

                            <Show when=move || error.get().is_some()>
                                <div class="alert alert-error">
                                    <IconAlertCircle size=IconSize::Sm />
                                    <span>{move || error.get().unwrap_or_default()}</span>
                                </div>
                            </Show>

                            <TextInput
                                value=device_name
                                label="Device Name"
                                placeholder="e.g., MacBook Pro TouchID, YubiKey"
                                helper_text="Give this device a recognizable name"
                            />

                            <button
                                class="btn btn-primary btn-block"
                                disabled=move || loading.get() || device_name.get().is_empty()
                                on:click=start_registration
                            >
                                <Show when=move || loading.get() fallback=|| {
                                    view! {
                                        <IconFingerprint size=IconSize::Sm />
                                        <span>"Register Security Key"</span>
                                    }
                                }>
                                    <Spinner size=SpinnerSize::Sm />
                                    <span>"Waiting for device..."</span>
                                </Show>
                            </button>
                        </div>
                    </Show>

                    <Show when=move || success.get()>
                        <div class="setup-success">
                            <div class="success-icon">
                                <IconCheckCircle size=IconSize::Xl />
                            </div>
                            <h2>"Security Key Added!"</h2>
                            <p>"You can now use this device for two-factor authentication."</p>
                            <A href="/settings/security" class="btn btn-primary">"Done"</A>
                        </div>
                    </Show>
                </div>
            </div>
        </div>
    }
}

/// Recovery Codes Page (view/regenerate)
#[component]
pub fn RecoveryCodesPage() -> impl IntoView {
    let state = use_app_state();

    let loading = create_rw_signal(true);
    let (codes, set_codes) = create_signal::<Vec<String>>(Vec::new());
    let error = create_rw_signal::<Option<String>>(None);
    let show_regenerate_modal = create_rw_signal(false);
    let (_regenerating, set_regenerating) = create_signal(false); // State tracked but ConfirmDialog handles UI
    let (copied, set_copied) = create_signal(false);

    // Fetch codes on mount
    create_effect(move |_| {
        spawn_local(async move {
            match api::auth::get_recovery_codes().await {
                Ok(response) => {
                    set_codes.set(response.codes);
                }
                Err(e) => {
                    error.set(Some(e.message));
                }
            }
            loading.set(false);
        });
    });

    let copy_codes = move |_| {
        let codes_text = codes.get().join("\n");
        let window = web_sys::window().unwrap();
        let nav = window.navigator();
        let clipboard = nav.clipboard();

        spawn_local(async move {
            let promise = clipboard.write_text(&codes_text);
            let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
            set_copied.set(true);
            set_timeout(move || set_copied.set(false), std::time::Duration::from_secs(2));
        });
    };

    let regenerate_codes = {
        let state = state.clone();
        move |_| {
            set_regenerating.set(true);
            error.set(None);

            let state = state.clone();
            spawn_local(async move {
                match api::auth::regenerate_recovery_codes().await {
                    Ok(response) => {
                        set_codes.set(response.codes);
                        show_regenerate_modal.set(false);
                        state.toast_success("Codes Regenerated", "New recovery codes have been generated");
                    }
                    Err(e) => {
                        error.set(Some(e.message));
                    }
                }
                set_regenerating.set(false);
            });
        }
    };

    view! {
        <div class="mfa-setup-page">
            <div class="setup-container">
                <div class="setup-header">
                    <A href="/settings/security" class="btn btn-ghost">
                        <IconArrowLeft size=IconSize::Sm />
                        <span>"Back to Security"</span>
                    </A>
                </div>

                <div class="setup-content">
                    <h1>"Recovery Codes"</h1>

                    <Show when=move || loading.get()>
                        <div class="loading-state">
                            <Spinner size=SpinnerSize::Lg />
                            <p>"Loading recovery codes..."</p>
                        </div>
                    </Show>

                    <Show when=move || error.get().is_some()>
                        <div class="alert alert-error">
                            <IconAlertCircle size=IconSize::Sm />
                            <span>{move || error.get().unwrap_or_default()}</span>
                        </div>
                    </Show>

                    <Show when=move || !loading.get() && error.get().is_none()>
                        <div class="recovery-codes-section">
                            <div class="codes-info">
                                <IconAlertTriangle size=IconSize::Md class="text-warning".to_string() />
                                <div>
                                    <p class="codes-warning">
                                        "These codes can be used to access your account if you lose your authenticator device."
                                    </p>
                                    <p class="codes-note">
                                        "Each code can only be used once. Store them in a safe place."
                                    </p>
                                </div>
                            </div>

                            <div class="codes-display">
                                <div class="codes-grid">
                                    {move || codes.get().iter().enumerate().map(|(i, code)| {
                                        view! {
                                            <div class="code-item">
                                                <span class="code-number">{i + 1}"."</span>
                                                <code>{code.clone()}</code>
                                            </div>
                                        }
                                    }).collect_view()}
                                </div>
                            </div>

                            <div class="codes-actions">
                                <button class="btn btn-secondary" on:click=copy_codes>
                                    {move || if copied.get() {
                                        view! {
                                            <IconCheck size=IconSize::Sm />
                                            <span>"Copied!"</span>
                                        }.into_view()
                                    } else {
                                        view! {
                                            <IconCopy size=IconSize::Sm />
                                            <span>"Copy All"</span>
                                        }.into_view()
                                    }}
                                </button>
                                <button
                                    class="btn btn-danger-outline"
                                    on:click=move |_| show_regenerate_modal.set(true)
                                >
                                    <IconRefresh size=IconSize::Sm />
                                    <span>"Regenerate Codes"</span>
                                </button>
                            </div>
                        </div>
                    </Show>
                </div>
            </div>

            // Regenerate confirmation modal
            <ConfirmDialog
                show=show_regenerate_modal
                title="Regenerate Recovery Codes?".to_string()
                message="This will invalidate all existing recovery codes. Make sure to save the new codes.".to_string()
                confirm_text="Regenerate".to_string()
                danger=true
                on_confirm=Callback::new(regenerate_codes)
            />
        </div>
    }
}
