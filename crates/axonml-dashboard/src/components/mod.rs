//! Dashboard Component Library — Sub-Module Aggregator
//!
//! Declares and re-exports every reusable Leptos component used by the
//! AxonML dashboard. Sub-modules: `charts` (time-series / metric plots),
//! `error_boundary` (page-level error trap), `forms` (text, code, select,
//! and checkbox inputs), `icons` (SVG icon set), `modal` (dialog + confirm
//! primitives), `navbar` (top bar), `progress` (progress bars/indicators),
//! `sidebar` (left nav), `spinner` (loading indicators), `table`
//! (sortable data grid), `terminal` (streaming log viewer), and `toast`
//! (transient notifications). Re-exports are glob-style so consumers can
//! write `use components::*;`.
//!
//! # File
//! `crates/axonml-dashboard/src/components/mod.rs`
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
// Sub-Modules
// =============================================================================

pub mod charts;
pub mod error_boundary;
pub mod forms;
pub mod icons;
pub mod modal;
pub mod navbar;
pub mod progress;
pub mod sidebar;
pub mod spinner;
pub mod table;
pub mod terminal;
pub mod toast;

// =============================================================================
// Re-Exports
// =============================================================================

pub use charts::*;
pub use error_boundary::*;
pub use forms::*;
pub use icons::*;
pub use modal::*;
pub use navbar::*;
pub use progress::*;
pub use sidebar::*;
pub use spinner::*;
pub use table::*;
pub use terminal::*;
pub use toast::*;
