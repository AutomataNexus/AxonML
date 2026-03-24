//! Dashboard Pages
//!
//! # File
//! `crates/axonml-dashboard/src/pages/mod.rs`
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

pub mod admin;
pub mod dashboard;
pub mod datasets;
pub mod hub;
pub mod inference;
pub mod landing;
pub mod models;
pub mod settings;
pub mod system;
pub mod training;

pub use dashboard::*;
pub use landing::*;
