//! Top Navigation Bars — Authenticated Navbar + Public Landing Navbar
//!
//! Defines the two top-of-page navigation components used across the
//! dashboard: `Navbar` (authenticated shell) and `PublicNavbar` (landing /
//! auth pages).
//!
//! `Navbar` reads `use_app_state()` to derive the current user and
//! authentication status, wires a sidebar-toggle button to
//! `state.toggle_sidebar()`, and hosts the global search input (with a
//! keyboard hint `kbd`). When signed in, it shows an avatar initial, the
//! user's name + email, and a logout button that clears auth state and
//! navigates back to `/login`. When signed out, the right side falls back
//! to a "Sign In" button linking to `/login`.
//!
//! `PublicNavbar` is the marketing/landing variant: fixed brand on the
//! left, a centered link group (Features, Documentation, GitHub — the
//! external link opens in a new tab with `noopener`), and "Sign In" +
//! "Get Started" calls-to-action on the right.
//!
//! # File
//! `crates/axonml-dashboard/src/components/navbar.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use leptos::*;
use leptos_router::*;

use crate::components::icons::*;
use crate::state::use_app_state;

// =============================================================================
// Authenticated Navbar
// =============================================================================

/// Top navigation bar
#[component]
pub fn Navbar() -> impl IntoView {
    let state = use_app_state();
    let navigate = use_navigate();

    let user = move || state.user.get();
    let is_authenticated = move || user().is_some();

    let on_logout = std::rc::Rc::new({
        let state = state.clone();
        let navigate = navigate.clone();
        move |_: web_sys::MouseEvent| {
            state.clear_auth();
            navigate("/login", Default::default());
        }
    });

    let toggle_sidebar = move |_| {
        state.toggle_sidebar();
    };

    view! {
        <nav class="navbar">
            <div class="navbar-left">
                <button class="btn btn-ghost" on:click=toggle_sidebar>
                    <IconMenu />
                </button>
                <A href="/" class="navbar-brand">
                    <img src="/assets/AxonML-logo.png" alt="AxonML" class="navbar-logo" />
                    <span class="navbar-title">"AxonML"</span>
                </A>
            </div>

            <div class="navbar-center">
                <div class="search-container">
                    <IconSearch size=IconSize::Sm />
                    <input
                        type="text"
                        placeholder="Search models, runs, endpoints..."
                        class="search-input"
                    />
                    <kbd class="search-shortcut">"/"</kbd>
                </div>
            </div>

            <div class="navbar-right">
                <Show
                    when=is_authenticated
                    fallback=move || view! {
                        <A href="/login" class="btn btn-primary">"Sign In"</A>
                    }
                >
                    <div class="navbar-actions">
                        // User info display
                        <div class="user-display">
                            <div class="avatar">
                                {move || user().map(|u| u.name.chars().next().unwrap_or('U').to_string()).unwrap_or_else(|| "U".to_string())}
                            </div>
                            <div class="user-details">
                                <span class="user-name">
                                    {move || user().map(|u| u.name.clone()).unwrap_or_default()}
                                </span>
                                <span class="user-email">
                                    {move || user().map(|u| u.email.clone()).unwrap_or_default()}
                                </span>
                            </div>
                        </div>
                        <button class="btn btn-ghost btn-sm" on:click={let on_logout = on_logout.clone(); move |e| on_logout(e)}>
                            <IconLogout size=IconSize::Sm />
                            <span>"Logout"</span>
                        </button>
                    </div>
                </Show>
            </div>
        </nav>
    }
}

// =============================================================================
// Public Navbar
// =============================================================================

/// Public navbar for landing/auth pages
#[component]
pub fn PublicNavbar() -> impl IntoView {
    view! {
        <nav class="navbar navbar-public">
            <div class="navbar-left">
                <A href="/" class="navbar-brand">
                    <img src="/assets/AxonML-logo.png" alt="AxonML" class="navbar-logo" />
                    <span class="navbar-title">"AxonML"</span>
                </A>
            </div>

            <div class="navbar-center">
                <div class="navbar-links">
                    <a href="#features" class="nav-link">"Features"</a>
                    <a href="#docs" class="nav-link">"Documentation"</a>
                    <a href="https://github.com/AutomataNexus/AxonML" class="nav-link" target="_blank" rel="noopener">
                        "GitHub"
                        <IconExternalLink size=IconSize::Sm />
                    </a>
                </div>
            </div>

            <div class="navbar-right">
                <A href="/login" class="btn btn-ghost">"Sign In"</A>
                <A href="/register" class="btn btn-primary">"Get Started"</A>
            </div>
        </nav>
    }
}
