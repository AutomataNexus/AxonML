//! Sidebar Navigation Component

use leptos::*;
use leptos_router::*;

use crate::state::use_app_state;
use crate::components::icons::*;

/// Sidebar navigation item
#[component]
fn SidebarItem<F, V>(
    #[prop(into)] href: String,
    #[prop(into)] label: String,
    icon: F,
    #[prop(optional)] badge: Option<u32>,
) -> impl IntoView
where
    F: Fn() -> V + 'static,
    V: IntoView,
{
    let state = use_app_state();
    let collapsed = move || state.sidebar_collapsed.get();
    let location = use_location();

    let is_active = {
        let href = href.clone();
        move || {
            let path = location.pathname.get();
            if href == "/" {
                path == "/"
            } else {
                path.starts_with(&href)
            }
        }
    };

    view! {
        <A
            href=href
            class=move || format!("sidebar-item {}", if is_active() { "active" } else { "" })
        >
            <span class="sidebar-icon">{icon()}</span>
            <Show when=move || !collapsed()>
                <span class="sidebar-label">{label.clone()}</span>
                {badge.map(|b| view! {
                    <span class="badge badge-primary">{b}</span>
                })}
            </Show>
        </A>
    }
}

/// Sidebar section header
#[component]
fn SidebarSection(
    #[prop(into)] label: String,
    children: Children,
) -> impl IntoView {
    let state = use_app_state();
    let collapsed = move || state.sidebar_collapsed.get();

    view! {
        <div class="sidebar-section">
            <Show when=move || !collapsed()>
                <div class="sidebar-section-header">{label.clone()}</div>
            </Show>
            {children()}
        </div>
    }
}

/// Main sidebar component
#[component]
pub fn Sidebar() -> impl IntoView {
    let state = use_app_state();
    let state_for_collapsed = state.clone();
    let state_for_admin = state.clone();
    let state_for_toggle = state.clone();
    let collapsed = move || state_for_collapsed.sidebar_collapsed.get();
    let is_admin = move || state_for_admin.is_admin();

    view! {
        <aside class=move || format!("sidebar {}", if collapsed() { "collapsed" } else { "" })>
            <div class="sidebar-content">
                // Main Navigation
                <SidebarSection label="Main">
                    <SidebarItem
                        href="/"
                        label="Dashboard"
                        icon=|| view! { <IconDashboard /> }
                    />
                </SidebarSection>

                // Training Section
                <SidebarSection label="Training">
                    <SidebarItem
                        href="/training"
                        label="Training Runs"
                        icon=|| view! { <IconActivity /> }
                    />
                    <SidebarItem
                        href="/training/new"
                        label="New Run"
                        icon=|| view! { <IconPlus /> }
                    />
                </SidebarSection>

                // Models Section
                <SidebarSection label="Models">
                    <SidebarItem
                        href="/models"
                        label="Model Registry"
                        icon=|| view! { <IconBox /> }
                    />
                    <SidebarItem
                        href="/models/upload"
                        label="Upload Model"
                        icon=|| view! { <IconUpload /> }
                    />
                </SidebarSection>

                // Inference Section
                <SidebarSection label="Inference">
                    <SidebarItem
                        href="/inference"
                        label="Overview"
                        icon=|| view! { <IconServer /> }
                    />
                    <SidebarItem
                        href="/inference/endpoints"
                        label="Endpoints"
                        icon=|| view! { <IconZap /> }
                    />
                    <SidebarItem
                        href="/inference/metrics"
                        label="Metrics"
                        icon=|| view! { <IconBarChart /> }
                    />
                </SidebarSection>

                // Admin Section (only for admins)
                <Show when=is_admin>
                    <SidebarSection label="Admin">
                        <SidebarItem
                            href="/admin/users"
                            label="User Management"
                            icon=|| view! { <IconUsers /> }
                        />
                        <SidebarItem
                            href="/admin/system"
                            label="System Stats"
                            icon=|| view! { <IconCpu /> }
                        />
                    </SidebarSection>
                </Show>

                // Settings at bottom
                <div class="sidebar-spacer"></div>
                <SidebarSection label="Settings">
                    <SidebarItem
                        href="/settings"
                        label="Settings"
                        icon=|| view! { <IconSettings /> }
                    />
                </SidebarSection>
            </div>

            // Collapse toggle
            <div class="sidebar-footer">
                <button
                    class="btn btn-ghost sidebar-toggle"
                    on:click=move |_| state_for_toggle.toggle_sidebar()
                >
                    <Show
                        when=collapsed
                        fallback=|| view! { <IconChevronLeft /> }
                    >
                        <IconChevronRight />
                    </Show>
                </button>
            </div>
        </aside>
    }
}
