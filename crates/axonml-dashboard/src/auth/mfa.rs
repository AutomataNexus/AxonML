//! MFA Verification Components

use leptos::*;

use crate::api;
use crate::types::*;
use crate::components::{forms::*, icons::*, spinner::*};
use crate::utils::webauthn;

/// MFA challenge page (shown when login requires MFA)
#[component]
pub fn MfaChallengePage(
    #[prop(into)] mfa_token: String,
    #[prop(into)] on_success: Callback<TokenPair>,
    #[prop(into)] on_cancel: Callback<()>,
) -> impl IntoView {
    let (method, set_method) = create_signal::<MfaMethod>(MfaMethod::Totp);
    let code = create_rw_signal(String::new());
    let loading = create_rw_signal(false);
    let error = create_rw_signal::<Option<String>>(None);

    let mfa_token_for_click = mfa_token.clone();
    let mfa_token_for_complete = mfa_token.clone();
    let on_success_for_click = on_success.clone();
    let on_success_for_complete = on_success.clone();

    let do_verify = move |on_success: Callback<TokenPair>, token: String, current_method: MfaMethod| {
        let code_val = code.get();
        if code_val.is_empty() {
            error.set(Some("Please enter a code".to_string()));
            return;
        }

        loading.set(true);
        error.set(None);

        spawn_local(async move {
            let result = match current_method {
                MfaMethod::Totp => {
                    api::auth::verify_mfa(&MfaVerifyRequest {
                        mfa_token: token,
                        code: code_val,
                    })
                    .await
                }
                MfaMethod::Recovery => api::auth::use_recovery_code(&token, &code_val).await,
                MfaMethod::WebAuthn => {
                    // WebAuthn uses the dedicated WebAuthnAuthenticator component, not code input
                    Err(api::ApiClientError {
                        status: 400,
                        message: "Use the security key button instead of entering a code".to_string(),
                    })
                }
            };

            match result {
                Ok(token_pair) => {
                    on_success.call(token_pair);
                }
                Err(e) => {
                    error.set(Some(e.message));
                }
            }
            loading.set(false);
        });
    };

    let on_click_verify = {
        let token = mfa_token_for_click.clone();
        move |_: web_sys::MouseEvent| {
            do_verify(on_success_for_click.clone(), token.clone(), method.get());
        }
    };

    let on_complete_verify = Callback::new({
        let token = mfa_token_for_complete.clone();
        move |_: String| {
            do_verify(on_success_for_complete.clone(), token.clone(), method.get());
        }
    });

    view! {
        <div class="mfa-challenge">
            <div class="mfa-challenge-header">
                <button class="btn btn-ghost" on:click=move |_| on_cancel.call(())>
                    <IconArrowLeft size=IconSize::Sm />
                    <span>"Cancel"</span>
                </button>
            </div>

            <div class="mfa-challenge-content">
                <div class="mfa-icon">
                    <IconShield size=IconSize::Xl />
                </div>

                <h1>"Verify Your Identity"</h1>
                <p>"Choose a verification method to continue"</p>

                <Show when=move || error.get().is_some()>
                    <div class="alert alert-error">
                        <IconAlertCircle size=IconSize::Sm />
                        <span>{move || error.get().unwrap_or_default()}</span>
                    </div>
                </Show>

                // Method selector
                <div class="mfa-methods">
                    <button
                        class=move || format!("mfa-method-btn {}", if method.get() == MfaMethod::Totp { "active" } else { "" })
                        on:click=move |_| set_method.set(MfaMethod::Totp)
                    >
                        <IconSmartphone size=IconSize::Md />
                        <span>"Authenticator App"</span>
                    </button>

                    <button
                        class=move || format!("mfa-method-btn {}", if method.get() == MfaMethod::Recovery { "active" } else { "" })
                        on:click=move |_| set_method.set(MfaMethod::Recovery)
                    >
                        <IconKey size=IconSize::Md />
                        <span>"Recovery Code"</span>
                    </button>
                </div>

                // Input based on method
                <div class="mfa-input-section">
                    <Show
                        when=move || method.get() == MfaMethod::Totp
                        fallback=move || view! {
                            <TextInput
                                value=code
                                label="Recovery Code"
                                placeholder="Enter your recovery code"
                                helper_text="Enter one of your backup recovery codes"
                            />
                        }
                    >
                        <CodeInput
                            value=code
                            length=6
                            label="6-Digit Code"
                            on_complete=on_complete_verify
                        />
                        <p class="form-helper">"Enter the code from your authenticator app"</p>
                    </Show>
                </div>

                <button
                    type="button"
                    class="btn btn-primary btn-block"
                    disabled=move || loading.get() || code.get().is_empty()
                    on:click=on_click_verify
                >
                    <Show when=move || loading.get() fallback=|| "Verify">
                        <Spinner size=SpinnerSize::Sm />
                        <span>"Verifying..."</span>
                    </Show>
                </button>
            </div>
        </div>
    }
}

/// MFA method enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MfaMethod {
    Totp,
    WebAuthn,
    Recovery,
}

/// WebAuthn authenticator component
#[component]
pub fn WebAuthnAuthenticator(
    #[prop(into)] mfa_token: String,
    #[prop(into)] on_success: Callback<TokenPair>,
    #[prop(optional)] on_error: Option<Callback<String>>,
) -> impl IntoView {
    let loading = create_rw_signal(false);
    let error = create_rw_signal::<Option<String>>(None);

    let start_auth = {
        let mfa_token = mfa_token.clone();
        move |_| {
            // Check if WebAuthn is available
            if !webauthn::is_webauthn_available() {
                error.set(Some("WebAuthn is not supported in this browser".to_string()));
                return;
            }

            loading.set(true);
            error.set(None);

            let token = mfa_token.clone();
            let on_success = on_success.clone();
            let on_error = on_error.clone();

            spawn_local(async move {
                // Step 1: Get challenge from server
                match api::auth::webauthn_authenticate_start(&token).await {
                    Ok(challenge) => {
                        // Step 2: Use WebAuthn API to get credential assertion
                        let rp_id = web_sys::window()
                            .and_then(|w| w.location().hostname().ok())
                            .unwrap_or_else(|| "localhost".to_string());

                        match webauthn::get_assertion(
                            &challenge.challenge,
                            &rp_id,
                            &challenge.allowed_credentials,
                        ).await {
                            Ok(assertion) => {
                                // Step 3: Send assertion to server for verification
                                let finish_request = api::auth::WebAuthnAuthFinishRequest {
                                    credential_id: assertion.id,
                                    authenticator_data: assertion.authenticator_data,
                                    client_data_json: assertion.client_data_json,
                                    signature: assertion.signature,
                                    user_handle: assertion.user_handle,
                                };

                                match api::auth::webauthn_authenticate_finish(&token, &finish_request).await {
                                    Ok(token_pair) => {
                                        on_success.call(token_pair);
                                    }
                                    Err(e) => {
                                        error.set(Some(format!("Verification failed: {}", e.message)));
                                        if let Some(cb) = on_error.as_ref() {
                                            cb.call(e.message);
                                        }
                                    }
                                }
                            }
                            Err(webauthn::WebAuthnError::UserCancelled) => {
                                error.set(Some("Authentication cancelled".to_string()));
                            }
                            Err(e) => {
                                let msg = e.to_string();
                                error.set(Some(msg.clone()));
                                if let Some(cb) = on_error.as_ref() {
                                    cb.call(msg);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let msg = e.message.clone();
                        error.set(Some(e.message));
                        if let Some(cb) = on_error.as_ref() {
                            cb.call(msg);
                        }
                    }
                }
                loading.set(false);
            });
        }
    };

    view! {
        <div class="webauthn-authenticator">
            <Show when=move || error.get().is_some()>
                <div class="alert alert-error">
                    <IconAlertCircle size=IconSize::Sm />
                    <span>{move || error.get().unwrap_or_default()}</span>
                </div>
            </Show>

            <div class="webauthn-prompt">
                <div class="webauthn-icon">
                    <IconFingerprint size=IconSize::Xl />
                </div>
                <p>"Use your security key or biometric to verify your identity"</p>
            </div>

            <button
                type="button"
                class="btn btn-primary btn-block"
                disabled=move || loading.get()
                on:click=start_auth
            >
                <Show when=move || loading.get() fallback=|| {
                    view! {
                        <IconFingerprint size=IconSize::Sm />
                        <span>"Use Security Key"</span>
                    }
                }>
                    <Spinner size=SpinnerSize::Sm />
                    <span>"Waiting for device..."</span>
                </Show>
            </button>
        </div>
    }
}
