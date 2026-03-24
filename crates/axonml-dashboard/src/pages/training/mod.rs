//! Training Pages
//!
//! # File
//! `crates/axonml-dashboard/src/pages/training/mod.rs`
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

pub mod detail;
pub mod list;
pub mod new;
pub mod notebook_editor;
pub mod notebook_import;
pub mod notebook_list;

pub use detail::*;
pub use list::*;
pub use new::*;
pub use notebook_editor::*;
pub use notebook_import::*;
pub use notebook_list::*;
