//! Inference Module — Model Serving, Metrics, and Connection Pooling
//!
//! Re-exports the three inference sub-modules: `metrics` (per-endpoint latency
//! and throughput tracking), `pool` (LRU model instance pool with capacity
//! management), and `server` (the inference HTTP server).
//!
//! # File
//! `crates/axonml-server/src/inference/mod.rs`
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
// Sub-modules
// =============================================================================

pub mod metrics;
pub mod pool;
pub mod server;
