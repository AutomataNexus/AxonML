//! Settings Pages

pub mod profile;
pub mod security;

pub use profile::*;
pub use security::*;

use leptos::*;
use leptos_router::*;

use crate::components::icons::*;

/// Settings layout page
#[component]
pub fn SettingsPage() -> impl IntoView {
    view! {
        <div class="page settings-page">
            <div class="page-header">
                <h1>"Settings"</h1>
                <p class="page-subtitle">"Manage your account and preferences"</p>
            </div>

            <div class="settings-grid">
                <A href="/settings/profile" class="settings-card">
                    <div class="settings-icon">
                        <IconUser size=IconSize::Lg />
                    </div>
                    <div class="settings-content">
                        <h3>"Profile"</h3>
                        <p>"Update your personal information"</p>
                    </div>
                    <IconChevronRight size=IconSize::Sm />
                </A>

                <A href="/settings/security" class="settings-card">
                    <div class="settings-icon">
                        <IconShield size=IconSize::Lg />
                    </div>
                    <div class="settings-content">
                        <h3>"Security"</h3>
                        <p>"Password, MFA, and session management"</p>
                    </div>
                    <IconChevronRight size=IconSize::Sm />
                </A>

                <div class="settings-card disabled">
                    <div class="settings-icon">
                        <IconSettings size=IconSize::Lg />
                    </div>
                    <div class="settings-content">
                        <h3>"Preferences"</h3>
                        <p>"Notifications and display settings"</p>
                    </div>
                    <span class="badge badge-default">"Coming Soon"</span>
                </div>

                <div class="settings-card disabled">
                    <div class="settings-icon">
                        <IconKey size=IconSize::Lg />
                    </div>
                    <div class="settings-content">
                        <h3>"API Keys"</h3>
                        <p>"Manage API access tokens"</p>
                    </div>
                    <span class="badge badge-default">"Coming Soon"</span>
                </div>
            </div>
        </div>
    }
}
