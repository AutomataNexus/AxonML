//! Admin pages module
//!
//! # File
//! `crates/axonml-dashboard/src/pages/admin/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

mod system;
mod users;

pub use system::SystemStatsPage;
pub use users::UserManagementPage;
