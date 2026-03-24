//! Commands - CLI Command Implementations
//!
//! # File
//! `crates/axonml-cli/src/commands/mod.rs`
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
pub mod bench;
pub mod convert;
pub mod dashboard;
pub mod data;
pub mod dataset;
pub mod eval;
pub mod export;
pub mod gpu;
pub mod hub;
pub mod init;
pub mod inspect;
pub mod kaggle;
pub mod load;
pub mod new;
pub mod predict;
pub mod quant;
pub mod rename;
pub mod report;
pub mod resume;
pub mod scaffold;
pub mod train;
pub mod tui;
pub mod upload;
pub mod zip;

#[cfg(feature = "wandb")]
pub mod wandb;
#[cfg(feature = "wandb")]
pub mod wandb_client;

#[cfg(feature = "serve")]
pub mod serve;

#[cfg(feature = "server-sync")]
pub mod sync;

// Re-export common utilities for commands
pub(crate) mod utils;
