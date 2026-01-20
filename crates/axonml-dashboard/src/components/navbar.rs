//! Top Navigation Bar Component

use leptos::*;
use leptos_router::*;

use crate::state::use_app_state;
use crate::components::icons::*;

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
    let logout_handler = on_logout.clone();

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
                    <img src="/assets/logo.svg" alt="AxonML" class="navbar-logo" />
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
                        // Notifications dropdown placeholder
                        <button class="btn btn-ghost">
                            <IconActivity />
                        </button>

                        // User dropdown
                        <div class="dropdown">
                            <button class="btn btn-ghost user-button">
                                <div class="avatar">
                                    {move || user().map(|u| u.name.chars().next().unwrap_or('U').to_string()).unwrap_or_else(|| "U".to_string())}
                                </div>
                                <span class="user-name">
                                    {move || user().map(|u| u.name.clone()).unwrap_or_default()}
                                </span>
                                <IconChevronDown size=IconSize::Sm />
                            </button>
                            <div class="dropdown-menu dropdown-menu-right">
                                <div class="dropdown-header">
                                    <div class="user-info">
                                        <strong>{move || user().map(|u| u.name.clone()).unwrap_or_default()}</strong>
                                        <span class="text-muted">{move || user().map(|u| u.email.clone()).unwrap_or_default()}</span>
                                    </div>
                                </div>
                                <div class="dropdown-divider"></div>
                                <A href="/settings/profile" class="dropdown-item">
                                    <IconUser size=IconSize::Sm />
                                    <span>"Profile"</span>
                                </A>
                                <A href="/settings/security" class="dropdown-item">
                                    <IconShield size=IconSize::Sm />
                                    <span>"Security"</span>
                                </A>
                                <A href="/settings" class="dropdown-item">
                                    <IconSettings size=IconSize::Sm />
                                    <span>"Settings"</span>
                                </A>
                                <div class="dropdown-divider"></div>
                                <button class="dropdown-item text-danger" on:click={let f = logout_handler.clone(); move |e| f(e)}>
                                    <IconLogout size=IconSize::Sm />
                                    <span>"Sign Out"</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </Show>
            </div>
        </nav>
    }
}

/// Public navbar for landing/auth pages
#[component]
pub fn PublicNavbar() -> impl IntoView {
    view! {
        <nav class="navbar navbar-public">
            <div class="navbar-left">
                <A href="/" class="navbar-brand">
                    <img src="/assets/logo.svg" alt="AxonML" class="navbar-logo" />
                    <span class="navbar-title">"AxonML"</span>
                </A>
            </div>

            <div class="navbar-center">
                <div class="navbar-links">
                    <a href="#features" class="nav-link">"Features"</a>
                    <a href="#docs" class="nav-link">"Documentation"</a>
                    <a href="https://github.com/yourusername/axonml" class="nav-link" target="_blank" rel="noopener">
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
