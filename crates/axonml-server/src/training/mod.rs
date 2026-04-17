//! Training Module — Submodule Aggregator
//!
//! Groups the training-runtime pieces used by the AxonML server. Declares:
//! - `executor` — `TrainingExecutor` that spawns and supervises training jobs
//! - `notebook_executor` — Jupyter-style cell runner (`NotebookExecutor`)
//! - `tracker` — `TrainingTracker` for metrics broadcast / history
//! - `websocket` — WebSocket routes for streaming training updates
//!
//! # File
//! `crates/axonml-server/src/training/mod.rs`
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

pub mod executor;
pub mod notebook_executor;
pub mod tracker;
pub mod websocket;
