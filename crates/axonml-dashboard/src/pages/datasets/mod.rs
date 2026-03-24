//! Datasets Pages
//!
//! # File
//! `crates/axonml-dashboard/src/pages/datasets/mod.rs`
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

pub mod analyze;
pub mod builtin;
pub mod kaggle;
pub mod list;
pub mod upload;

pub use analyze::*;
pub use builtin::*;
pub use kaggle::*;
pub use list::*;
pub use upload::*;
