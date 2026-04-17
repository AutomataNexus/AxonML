//! LLM Integration — Ollama Client Re-Exports
//!
//! Aggregator for the server's LLM integration layer. Declares the `ollama`
//! submodule (which provides `OllamaClient` and `DEFAULT_OLLAMA_URL` used by
//! `main.rs` for AI assistance) and re-exports its public items with a glob.
//!
//! # File
//! `crates/axonml-server/src/llm/mod.rs`
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
// Sub-Modules and Re-Exports
// =============================================================================

pub mod ollama;

pub use ollama::*;
