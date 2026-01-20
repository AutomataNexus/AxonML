//! AxonML Dashboard - Leptos/WASM Frontend
//!
//! A modern, reactive web dashboard for the AxonML Machine Learning Framework.

use leptos::*;
use leptos_router::*;

pub mod api;
pub mod auth;
pub mod components;
pub mod pages;
pub mod state;
pub mod types;

use auth::{LoginPage, RegisterPage, session::{SessionInitializer, ProtectedRoute}};
use components::toast::ToastContainer;
use pages::{
    landing::LandingPage,
    dashboard::{DashboardPage, AppShell},
    training::{TrainingListPage, TrainingDetailPage, NewTrainingPage},
    models::{ModelsListPage, ModelDetailPage, ModelUploadPage},
    inference::{InferenceOverviewPage, EndpointsListPage, EndpointDetailPage, InferenceMetricsPage},
    settings::{SettingsPage, ProfileSettingsPage, SecuritySettingsPage},
};
use auth::mfa_setup::{TotpSetupPage, WebAuthnSetupPage, RecoveryCodesPage};
use state::provide_app_state;

/// Main application component
#[component]
pub fn App() -> impl IntoView {
    // Provide global state
    let _state = provide_app_state();

    view! {
        <Router>
            <SessionInitializer />
            <Routes>
                // Public routes
                <Route path="/" view=|| view! {
                    <PublicOrDashboard />
                } />
                <Route path="/login" view=LoginPage />
                <Route path="/register" view=RegisterPage />

                // Protected dashboard routes
                <Route path="/dashboard" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <DashboardPage />
                        </AppShell>
                    </ProtectedRoute>
                } />

                // Training routes
                <Route path="/training" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <TrainingListPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/training/new" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <NewTrainingPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/training/:id" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <TrainingDetailPage />
                        </AppShell>
                    </ProtectedRoute>
                } />

                // Models routes
                <Route path="/models" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <ModelsListPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/models/upload" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <ModelUploadPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/models/:id" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <ModelDetailPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/models/:id/upload" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <ModelUploadPage />
                        </AppShell>
                    </ProtectedRoute>
                } />

                // Inference routes
                <Route path="/inference" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <InferenceOverviewPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/inference/endpoints" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <EndpointsListPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/inference/endpoints/:id" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <EndpointDetailPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/inference/metrics" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <InferenceMetricsPage />
                        </AppShell>
                    </ProtectedRoute>
                } />

                // Settings routes
                <Route path="/settings" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <SettingsPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/settings/profile" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <ProfileSettingsPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/settings/security" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <SecuritySettingsPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/settings/security/totp" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <TotpSetupPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/settings/security/webauthn" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <WebAuthnSetupPage />
                        </AppShell>
                    </ProtectedRoute>
                } />
                <Route path="/settings/security/recovery" view=|| view! {
                    <ProtectedRoute>
                        <AppShell>
                            <RecoveryCodesPage />
                        </AppShell>
                    </ProtectedRoute>
                } />

                // 404
                <Route path="/*any" view=NotFound />
            </Routes>
            <ToastContainer />
        </Router>
    }
}

/// Show landing page if not authenticated, dashboard if authenticated
#[component]
fn PublicOrDashboard() -> impl IntoView {
    let state = state::use_app_state();
    let is_authenticated = move || state.user.get().is_some();
    let is_loading = move || state.loading.get();

    view! {
        <Show
            when=move || !is_loading()
            fallback=|| view! {
                <div class="page-loader">
                    <div class="page-loader-content">
                        <img src="/assets/logo.svg" alt="AxonML" class="page-loader-logo" />
                        <components::spinner::Spinner size=components::spinner::SpinnerSize::Lg />
                        <p>"Loading..."</p>
                    </div>
                </div>
            }
        >
            <Show
                when=is_authenticated
                fallback=LandingPage
            >
                <AppShell>
                    <DashboardPage />
                </AppShell>
            </Show>
        </Show>
    }
}

/// 404 Not Found page
#[component]
fn NotFound() -> impl IntoView {
    view! {
        <div class="not-found-page">
            <div class="not-found-content">
                <h1>"404"</h1>
                <h2>"Page Not Found"</h2>
                <p>"The page you're looking for doesn't exist or has been moved."</p>
                <A href="/" class="btn btn-primary">
                    "Go Home"
                </A>
            </div>
        </div>
    }
}

/// WASM entry point
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    // Set up panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize tracing for WASM
    tracing_wasm::set_as_global_default();

    // Mount the app
    mount_to_body(App);
}
