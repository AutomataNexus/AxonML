//! Error Boundary Component — Panic Catching Page Wrapper for WASM
//!
//! Provides the `PageErrorBoundary` Leptos component which wraps page content
//! to catch component errors and display a user-friendly fallback UI instead
//! of crashing the WASM module. The fallback shows the list of error messages
//! from Leptos' `ErrorBoundary` and includes a "Reload Page" button that calls
//! `window.location().reload()` via `web_sys` to recover the application.
//!
//! # File
//! `crates/axonml-dashboard/src/components/error_boundary.rs`
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

// =============================================================================
// Page Error Boundary Component
// =============================================================================

/// A reusable error boundary that catches component errors and displays
/// a user-friendly fallback instead of crashing the page.
///
/// # Usage
/// ```ignore
/// <PageErrorBoundary>
///     <SomePageContent />
/// </PageErrorBoundary>
/// ```
#[component]
pub fn PageErrorBoundary(children: Children) -> impl IntoView {
    view! {
        <ErrorBoundary fallback=|errors| {
            view! {
                <div class="error-boundary">
                    <div class="error-boundary-content">
                        <h2>"Something went wrong"</h2>
                        <p class="error-message">
                            "An error occurred while rendering this page."
                        </p>
                        <div class="error-details">
                            <ul>
                                {move || errors.get()
                                    .into_iter()
                                    .map(|(_, e)| view! { <li>{e.to_string()}</li> })
                                    .collect_view()
                                }
                            </ul>
                        </div>
                        <button
                            class="btn btn-primary"
                            on:click=move |_| {
                                // Reload the page to recover
                                if let Some(window) = web_sys::window() {
                                    let _ = window.location().reload();
                                }
                            }
                        >
                            "Reload Page"
                        </button>
                    </div>
                </div>
            }
        }>
            {children()}
        </ErrorBoundary>
    }
}
