//! User Management Admin Page — Stub Screen For User CRUD
//!
//! Placeholder admin screen for managing user accounts, roles, and
//! permissions. `UserManagementPage` renders a page header with an
//! `IconUsers` title, a primary "Add User" button (`IconPlus`), and a single
//! search-filterable card whose body currently displays an empty-state block.
//! The real user list, search wiring, and mutation calls are not yet
//! implemented — this file is the scaffold that routing targets.
//!
//! # File
//! `crates/axonml-dashboard/src/pages/admin/users.rs`
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

use crate::components::icons::*;
use leptos::*;

// =============================================================================
// User Management Page
// =============================================================================

#[component]
pub fn UserManagementPage() -> impl IntoView {
    view! {
        <div class="page-container">
            // -----------------------------------------------------------------
            // Page Header
            // -----------------------------------------------------------------
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

            // -----------------------------------------------------------------
            // Users Card
            // -----------------------------------------------------------------
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
