//! Commands — CLI Subcommand Aggregator
//!
//! Declares every subcommand implementation module for the `axonml` CLI.
//! Always-on modules cover project lifecycle (`new`, `init`, `scaffold`),
//! training and evaluation (`train`, `resume`, `eval`, `predict`), model
//! tooling (`convert`, `export`, `inspect`, `report`, `quant`, `load`,
//! `rename`, `zip`, `upload`), data/compute (`data`, `analyze`, `bench`,
//! `gpu`), discovery (`hub`, `kaggle`, `dataset`), and UI
//! (`dashboard`, `tui`). Feature-gated: `wandb` + `wandb_client` under
//! `wandb`, `serve` under `serve`, and `sync` under `server-sync`. The
//! crate-visible `utils` module holds shared helpers for the command
//! implementations.
//!
//! # File
//! `crates/axonml-cli/src/commands/mod.rs`
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

// -----------------------------------------------------------------------------
// Feature-Gated Modules
// -----------------------------------------------------------------------------

#[cfg(feature = "wandb")]
pub mod wandb;
#[cfg(feature = "wandb")]
pub mod wandb_client;

#[cfg(feature = "serve")]
pub mod serve;

#[cfg(feature = "server-sync")]
pub mod sync;

// -----------------------------------------------------------------------------
// Internal Utilities
// -----------------------------------------------------------------------------

// Re-export common utilities for commands
pub(crate) mod utils;
