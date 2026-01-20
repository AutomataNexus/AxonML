//! Commands - CLI Command Implementations
//!
//! This module contains the implementations for all CLI commands.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

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

// Re-export common utilities for commands
pub(crate) mod utils;
