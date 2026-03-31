//! Error Boundary Component
//!
//! Wraps page content to catch panics and render errors gracefully
//! instead of crashing the entire WASM module.

use leptos::*;

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
