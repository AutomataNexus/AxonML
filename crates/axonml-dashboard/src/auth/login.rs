//! Login Page Component

use leptos::*;
use leptos_router::*;

use crate::api;
use crate::components::{forms::*, icons::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// Login page component
#[component]
pub fn LoginPage() -> impl IntoView {
    let state = use_app_state();
    let navigate = use_navigate();

    let email = create_rw_signal(String::new());
    let password = create_rw_signal(String::new());
    let loading = create_rw_signal(false);
    let error = create_rw_signal::<Option<String>>(None);
    let mfa_required = create_rw_signal(false);
    let mfa_token = create_rw_signal::<Option<String>>(None);

    let handle_submit = {
        let state = state.clone();
        let navigate = navigate.clone();
        move |e: web_sys::SubmitEvent| {
            e.prevent_default();
            loading.set(true);
            error.set(None);

            let email_val = email.get();
            let password_val = password.get();
            let state = state.clone();
            let navigate = navigate.clone();

            spawn_local(async move {
                let request = LoginRequest {
                    email: email_val,
                    password: password_val,
                };

                match api::auth::login(&request).await {
                    Ok(response) => {
                        if response.requires_mfa {
                            // MFA required - store token and show MFA form
                            mfa_token.set(response.mfa_token);
                            mfa_required.set(true);
                        } else if let (Some(access), Some(refresh), Some(user)) =
                            (response.access_token, response.refresh_token, response.user)
                        {
                            // No MFA required - login complete
                            state.set_auth(&access, &refresh, user);
                            navigate("/dashboard", Default::default());
                        }
                    }
                    Err(e) => {
                        error.set(Some(e.message));
                    }
                }
                loading.set(false);
            });
        }
    };

    // Clones for the MFA fallback view
    let state_mfa = state.clone();
    let navigate_mfa = navigate.clone();
    let handle_submit = std::rc::Rc::new(handle_submit);
    let on_submit = handle_submit.clone();

    view! {
        <div class="auth-page">
            <div class="auth-container">
                <div class="auth-header">
                    <img src="/assets/AxonML-logo.png" alt="AxonML" class="auth-logo" />
                    <h1 class="auth-title">"Welcome Back"</h1>
                    <p class="auth-subtitle">"Sign in to your AxonML account"</p>
                </div>

                <Show
                    when=move || !mfa_required.get()
                    fallback=move || view! {
                        <MfaVerificationForm
                            mfa_token=mfa_token.get().unwrap_or_default()
                            on_success=Callback::new({
                                let state = state_mfa.clone();
                                let navigate = navigate_mfa.clone();
                                move |token_pair: TokenPair| {
                                    state.set_auth(&token_pair.access_token, &token_pair.refresh_token, token_pair.user);
                                    navigate("/dashboard", Default::default());
                                }
                            })
                            on_back=Callback::new(move |_: ()| {
                                mfa_required.set(false);
                                mfa_token.set(None);
                            })
                        />
                    }
                >
                    <form class="auth-form" on:submit={let f = on_submit.clone(); move |e| f(e)}>
                        <Show when=move || error.get().is_some()>
                            <div class="alert alert-error">
                                <IconAlertCircle size=IconSize::Sm />
                                <span>{move || error.get().unwrap_or_default()}</span>
                            </div>
                        </Show>

                        <TextInput
                            value=email
                            input_type=InputType::Text
                            label="Email or Username"
                            placeholder="you@example.com or username"
                            required=true
                            icon=view! { <IconUser size=IconSize::Sm /> }.into_view()
                        />

                        <TextInput
                            value=password
                            input_type=InputType::Password
                            label="Password"
                            placeholder="Enter your password"
                            required=true
                            icon=view! { <IconLock size=IconSize::Sm /> }.into_view()
                        />

                        <div class="auth-options">
                            <A href="/forgot-password" class="link">"Forgot password?"</A>
                        </div>

                        <button
                            type="submit"
                            class="btn btn-primary btn-block"
                            disabled=move || loading.get()
                        >
                            <Show when=move || loading.get() fallback=|| "Sign In">
                                <Spinner size=SpinnerSize::Sm />
                                <span>"Signing in..."</span>
                            </Show>
                        </button>
                    </form>

                    <div class="auth-footer">
                        <p>
                            "Don't have an account? "
                            <A href="/register" class="link">"Create one"</A>
                        </p>
                    </div>
                </Show>
            </div>
        </div>
    }
}

/// MFA verification form component
#[component]
fn MfaVerificationForm(
    mfa_token: String,
    #[prop(into)] on_success: Callback<TokenPair>,
    #[prop(into)] on_back: Callback<()>,
) -> impl IntoView {
    let code = create_rw_signal(String::new());
    let loading = create_rw_signal(false);
    let error = create_rw_signal::<Option<String>>(None);
    let use_recovery = create_rw_signal(false);

    let mfa_token_clone = mfa_token.clone();
    let mfa_token_for_complete = mfa_token.clone();
    let on_success_for_click = on_success.clone();
    let on_success_for_complete = on_success.clone();

    let do_verify = move |on_success: Callback<TokenPair>, token: String| {
        let code_val = code.get();
        if code_val.len() < 6 {
            error.set(Some("Please enter a valid 6-digit code".to_string()));
            return;
        }

        loading.set(true);
        error.set(None);

        let use_recovery_val = use_recovery.get();

        spawn_local(async move {
            let result = if use_recovery_val {
                api::auth::use_recovery_code(&token, &code_val).await
            } else {
                api::auth::verify_mfa(&MfaVerifyRequest {
                    mfa_token: token,
                    code: code_val,
                })
                .await
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
        let token = mfa_token_clone.clone();
        move |_: web_sys::MouseEvent| {
            do_verify(on_success_for_click.clone(), token.clone());
        }
    };

    let on_complete_verify = Callback::new({
        let token = mfa_token_for_complete.clone();
        move |_: String| {
            do_verify(on_success_for_complete.clone(), token.clone());
        }
    });

    view! {
        <div class="mfa-form">
            <div class="mfa-header">
                <button class="btn btn-ghost" on:click=move |_| on_back.call(())>
                    <IconArrowLeft size=IconSize::Sm />
                    <span>"Back"</span>
                </button>
            </div>

            <div class="mfa-icon">
                <IconShield size=IconSize::Xl />
            </div>

            <h2 class="mfa-title">"Two-Factor Authentication"</h2>
            <p class="mfa-subtitle">
                {move || if use_recovery.get() {
                    "Enter one of your recovery codes"
                } else {
                    "Enter the 6-digit code from your authenticator app"
                }}
            </p>

            <Show when=move || error.get().is_some()>
                <div class="alert alert-error">
                    <IconAlertCircle size=IconSize::Sm />
                    <span>{move || error.get().unwrap_or_default()}</span>
                </div>
            </Show>

            <Show
                when=move || !use_recovery.get()
                fallback=move || view! {
                    <TextInput
                        value=code
                        label="Recovery Code"
                        placeholder="Enter recovery code"
                        required=true
                    />
                }
            >
                <CodeInput
                    value=code
                    length=6
                    label="Verification Code"
                    on_complete=on_complete_verify
                />
            </Show>

            <button
                type="button"
                class="btn btn-primary btn-block"
                disabled=move || loading.get() || code.get().len() < 6
                on:click=on_click_verify
            >
                <Show when=move || loading.get() fallback=|| "Verify">
                    <Spinner size=SpinnerSize::Sm />
                    <span>"Verifying..."</span>
                </Show>
            </button>

            <div class="mfa-options">
                <button
                    type="button"
                    class="btn btn-ghost btn-sm"
                    on:click=move |_| {
                        use_recovery.update(|v| *v = !*v);
                        code.set(String::new());
                        error.set(None);
                    }
                >
                    {move || if use_recovery.get() {
                        "Use authenticator app instead"
                    } else {
                        "Use recovery code instead"
                    }}
                </button>
            </div>
        </div>
    }
}

/// Registration page component
#[component]
pub fn RegisterPage() -> impl IntoView {
    let state = use_app_state();
    let navigate = use_navigate();

    let name = create_rw_signal(String::new());
    let email = create_rw_signal(String::new());
    let password = create_rw_signal(String::new());
    let confirm_password = create_rw_signal(String::new());
    let loading = create_rw_signal(false);
    let error = create_rw_signal::<Option<String>>(None);

    let password_match = move || password.get() == confirm_password.get();
    let password_error = move || {
        if !confirm_password.get().is_empty() && !password_match() {
            Some("Passwords do not match".to_string())
        } else {
            None
        }
    };

    let on_submit = {
        let state = state.clone();
        let navigate = navigate.clone();
        move |e: web_sys::SubmitEvent| {
            e.prevent_default();

            if !password_match() {
                error.set(Some("Passwords do not match".to_string()));
                return;
            }

            if password.get().len() < 8 {
                error.set(Some("Password must be at least 8 characters".to_string()));
                return;
            }

            loading.set(true);
            error.set(None);

            let request = RegisterRequest {
                email: email.get(),
                name: name.get(),
                password: password.get(),
            };

            let state = state.clone();
            let navigate = navigate.clone();
            spawn_local(async move {
                match api::auth::register(&request).await {
                    Ok(token_pair) => {
                        state.set_auth(
                            &token_pair.access_token,
                            &token_pair.refresh_token,
                            token_pair.user,
                        );
                        navigate("/dashboard", Default::default());
                    }
                    Err(e) => {
                        error.set(Some(e.message));
                    }
                }
                loading.set(false);
            });
        }
    };

    view! {
        <div class="auth-page">
            <div class="auth-container">
                <div class="auth-header">
                    <img src="/assets/AxonML-logo.png" alt="AxonML" class="auth-logo" />
                    <h1 class="auth-title">"Create Account"</h1>
                    <p class="auth-subtitle">"Get started with AxonML"</p>
                </div>

                <form class="auth-form" on:submit=on_submit>
                    <Show when=move || error.get().is_some()>
                        <div class="alert alert-error">
                            <IconAlertCircle size=IconSize::Sm />
                            <span>{move || error.get().unwrap_or_default()}</span>
                        </div>
                    </Show>

                    <TextInput
                        value=name
                        label="Full Name"
                        placeholder="John Doe"
                        required=true
                        icon=view! { <IconUser size=IconSize::Sm /> }.into_view()
                    />

                    <TextInput
                        value=email
                        input_type=InputType::Email
                        label="Email"
                        placeholder="you@example.com"
                        required=true
                        icon=view! { <IconUser size=IconSize::Sm /> }.into_view()
                    />

                    <TextInput
                        value=password
                        input_type=InputType::Password
                        label="Password"
                        placeholder="Create a password (min. 8 characters)"
                        required=true
                        helper_text="Must be at least 8 characters"
                        icon=view! { <IconLock size=IconSize::Sm /> }.into_view()
                    />

                    <TextInput
                        value=confirm_password
                        input_type=InputType::Password
                        label="Confirm Password"
                        placeholder="Confirm your password"
                        required=true
                        error=MaybeSignal::derive(password_error)
                        icon=view! { <IconLock size=IconSize::Sm /> }.into_view()
                    />

                    <button
                        type="submit"
                        class="btn btn-primary btn-block"
                        disabled=move || loading.get() || !password_match()
                    >
                        <Show when=move || loading.get() fallback=|| "Create Account">
                            <Spinner size=SpinnerSize::Sm />
                            <span>"Creating account..."</span>
                        </Show>
                    </button>
                </form>

                <div class="auth-footer">
                    <p>
                        "Already have an account? "
                        <A href="/login" class="link">"Sign in"</A>
                    </p>
                </div>
            </div>
        </div>
    }
}
