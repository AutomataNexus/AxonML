//! Session Management Utilities

use leptos::*;
use gloo_storage::{LocalStorage, Storage};
use gloo_timers::callback::Interval;
use wasm_bindgen::JsCast;

use crate::api;
use crate::state::use_app_state;
use crate::types::User;

const ACCESS_TOKEN_KEY: &str = "access_token";
const REFRESH_TOKEN_KEY: &str = "refresh_token";
const USER_KEY: &str = "user";
const TOKEN_REFRESH_INTERVAL_MS: u32 = 5 * 60 * 1000; // 5 minutes

/// Session manager that handles token refresh and session validation
pub struct SessionManager {
    refresh_interval: Option<Interval>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            refresh_interval: None,
        }
    }

    /// Start automatic token refresh
    pub fn start_refresh_timer(&mut self) {
        let interval = Interval::new(TOKEN_REFRESH_INTERVAL_MS, move || {
            spawn_local(async {
                refresh_token_if_needed().await;
            });
        });
        self.refresh_interval = Some(interval);
    }

    /// Stop automatic token refresh
    pub fn stop_refresh_timer(&mut self) {
        self.refresh_interval = None;
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if we have a stored session
pub fn has_stored_session() -> bool {
    LocalStorage::get::<String>(ACCESS_TOKEN_KEY).is_ok()
}

/// Get stored access token
pub fn get_access_token() -> Option<String> {
    LocalStorage::get(ACCESS_TOKEN_KEY).ok()
}

/// Get stored refresh token
pub fn get_refresh_token() -> Option<String> {
    LocalStorage::get(REFRESH_TOKEN_KEY).ok()
}

/// Get stored user
pub fn get_stored_user() -> Option<User> {
    LocalStorage::get(USER_KEY).ok()
}

/// Store session data
pub fn store_session(access_token: &str, refresh_token: &str, user: &User) {
    let _ = LocalStorage::set(ACCESS_TOKEN_KEY, access_token);
    let _ = LocalStorage::set(REFRESH_TOKEN_KEY, refresh_token);
    let _ = LocalStorage::set(USER_KEY, user);
}

/// Clear session data
pub fn clear_session() {
    LocalStorage::delete(ACCESS_TOKEN_KEY);
    LocalStorage::delete(REFRESH_TOKEN_KEY);
    LocalStorage::delete(USER_KEY);
}

/// Update access token only
pub fn update_access_token(token: &str) {
    let _ = LocalStorage::set(ACCESS_TOKEN_KEY, token);
}

/// Refresh the access token if we have a refresh token
pub async fn refresh_token_if_needed() {
    if let Some(refresh) = get_refresh_token() {
        match api::auth::refresh(&refresh).await {
            Ok(token_pair) => {
                store_session(
                    &token_pair.access_token,
                    &token_pair.refresh_token,
                    &token_pair.user,
                );
            }
            Err(_) => {
                // Refresh failed - session is invalid
                clear_session();
            }
        }
    }
}

/// Validate current session by checking with the server
pub async fn validate_session() -> bool {
    if !has_stored_session() {
        return false;
    }

    match api::auth::me().await {
        Ok(user) => {
            // Update stored user with latest data
            if let Some(access) = get_access_token() {
                if let Some(refresh) = get_refresh_token() {
                    store_session(&access, &refresh, &user);
                }
            }
            true
        }
        Err(e) => {
            if e.status == 401 {
                // Token expired - try to refresh
                refresh_token_if_needed().await;
                // Check again after refresh
                api::auth::me().await.is_ok()
            } else {
                false
            }
        }
    }
}

/// Session initialization component
/// This runs once at app startup to validate/refresh the session
#[component]
pub fn SessionInitializer() -> impl IntoView {
    let state = use_app_state();

    create_effect(move |_| {
        let state = state.clone();

        spawn_local(async move {
            state.loading.set(true);

            if has_stored_session() {
                // Try to validate existing session
                if validate_session().await {
                    // Session is valid - load user data
                    if let Some(user) = get_stored_user() {
                        state.user.set(Some(user));
                    }
                } else {
                    // Session is invalid - clear it
                    clear_session();
                    state.user.set(None);
                }
            }

            state.loading.set(false);
        });
    });

    // This component doesn't render anything
    view! {}
}

/// Protected route wrapper
/// Redirects to login if not authenticated
#[component]
pub fn ProtectedRoute(
    children: Children,
    #[prop(default = "/login")] redirect: &'static str,
) -> impl IntoView {
    let state = use_app_state();
    let navigate = leptos_router::use_navigate();

    let is_authenticated = move || state.user.get().is_some();
    let is_loading = move || state.loading.get();

    create_effect(move |_| {
        if !is_loading() && !is_authenticated() {
            navigate(redirect, Default::default());
        }
    });

    let children = store_value(children());

    view! {
        <Show
            when=move || !is_loading()
            fallback=|| view! {
                <div class="page-loader">
                    <div class="page-loader-content">
                        <img src="/assets/logo.svg" alt="AxonML" class="page-loader-logo" />
                        <crate::components::spinner::Spinner size=crate::components::spinner::SpinnerSize::Lg />
                        <p>"Loading..."</p>
                    </div>
                </div>
            }
        >
            <Show
                when=is_authenticated
                fallback=|| view! {} // Will redirect via effect
            >
                {children.get_value()}
            </Show>
        </Show>
    }
}

/// Admin-only route wrapper
#[component]
pub fn AdminRoute(
    children: Children,
    #[prop(default = "/")] redirect: &'static str,
) -> impl IntoView {
    let state = use_app_state();
    let navigate = leptos_router::use_navigate();

    let state_for_admin = state.clone();
    let state_for_loading = state.clone();
    let state_for_effect_admin = state.clone();
    let state_for_effect_loading = state.clone();

    let is_admin = move || state_for_admin.is_admin();
    let is_loading = move || state_for_loading.loading.get();

    create_effect(move |_| {
        let loading = state_for_effect_loading.loading.get();
        let admin = state_for_effect_admin.is_admin();
        if !loading && !admin {
            navigate(redirect, Default::default());
        }
    });

    let children = store_value(children());

    view! {
        <ProtectedRoute>
            {move || {
                if is_loading() {
                    view! {
                        <div class="loading-state">
                            <p>"Checking permissions..."</p>
                        </div>
                    }.into_view()
                } else if is_admin() {
                    children.get_value().into_view()
                } else {
                    view! {
                        <div class="access-denied">
                            <h1>"Access Denied"</h1>
                            <p>"You don't have permission to view this page."</p>
                        </div>
                    }.into_view()
                }
            }}
        </ProtectedRoute>
    }
}

/// Logout action
pub fn logout() {
    clear_session();
    // Navigate to login page
    if let Some(window) = web_sys::window() {
        let _ = window.location().set_href("/login");
    }
}

/// Hook to handle session timeout
pub fn use_session_timeout(timeout_minutes: u32) {
    let state = use_app_state();
    let (last_activity, set_last_activity) = create_signal(js_sys::Date::now());

    // Update activity on any user interaction - clone set_last_activity for each closure
    let set_last_activity_mouse = set_last_activity;
    let set_last_activity_key = set_last_activity;
    let set_last_activity_click = set_last_activity;

    // Set up event listeners for activity
    create_effect(move |_| {
        let window = web_sys::window().unwrap();

        // Mouse move
        let handler = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            set_last_activity_mouse.set(js_sys::Date::now());
        });
        let _ = window.add_event_listener_with_callback("mousemove", handler.as_ref().unchecked_ref());
        handler.forget();

        // Keypress
        let handler = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            set_last_activity_key.set(js_sys::Date::now());
        });
        let _ = window.add_event_listener_with_callback("keypress", handler.as_ref().unchecked_ref());
        handler.forget();

        // Click
        let handler = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            set_last_activity_click.set(js_sys::Date::now());
        });
        let _ = window.add_event_listener_with_callback("click", handler.as_ref().unchecked_ref());
        handler.forget();
    });

    // Check for timeout periodically
    let timeout_ms = (timeout_minutes * 60 * 1000) as f64;
    let state_for_timeout = state.clone();
    create_effect(move |_| {
        let state = state_for_timeout.clone();
        let _interval = Interval::new(60_000, move || { // Check every minute
            let now = js_sys::Date::now();
            let last = last_activity.get();
            let elapsed = now - last;

            if elapsed > timeout_ms {
                // Session timed out
                state.toast_warning("Session Expired", "You have been logged out due to inactivity");
                logout();
            }
        });
    });
}
