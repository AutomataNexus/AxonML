//! User Management Admin Page

use crate::components::icons::*;
use leptos::*;

#[component]
pub fn UserManagementPage() -> impl IntoView {
    view! {
        <div class="page-container">
            <div class="page-header">
                <div class="page-header-content">
                    <h1 class="page-title">
                        <IconUsers />
                        <span>"User Management"</span>
                    </h1>
                    <p class="page-description">
                        "Manage users, roles, and permissions across your AxonML instance"
                    </p>
                </div>
                <div class="page-header-actions">
                    <button class="btn btn-primary">
                        <IconPlus size=IconSize::Sm />
                        <span>"Add User"</span>
                    </button>
                </div>
            </div>

            <div class="page-content">
                <div class="card">
                    <div class="card-header">
                        <h3>"Users"</h3>
                        <div class="card-actions">
                            <input
                                type="text"
                                placeholder="Search users..."
                                class="form-input"
                            />
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="empty-state">
                            <IconUsers size=IconSize::Xl />
                            <h3>"User Management"</h3>
                            <p>"No users found matching your search"</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    }
}
