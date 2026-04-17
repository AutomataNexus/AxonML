//! Research Agent — Literature Review, Paper Drafting, Citation Management
//!
//! Defines the `research` agent configuration: assists with academic
//! paper writing for the AxonML publication pipeline — the AxonML
//! framework paper, the Aegis biometric suite paper, the Trident
//! 1.58-bit LLM paper, the TMC HVAC controller paper, and the planned
//! Mnemosyne standalone paper. Verifies cited numbers against actual
//! code and logs to prevent fabrication.
//!
//! Exports:
//! - `SYSTEM_PROMPT` — lists the active papers, the author identity
//!   (Andrew Jewell Sr., ORCID `0009-0005-2158-7060`), the IEEEtran
//!   format convention, the "companion preprint" boilerplate rule
//!   (LESSONS.md L13), the pdflatex error-filtering rule (L06), and
//!   the tool set (`read_file`, `write_file`, `grep`, `search_files`,
//!   `shell`, `vault_read`, `vault_write`).
//! - `config()` — returns an `AgentConfig` with
//!   `max_iterations = 25`, `model = "qwen3"`, `temperature = 0.3`.
//!
//! # File
//! `nexus-agent/src/agents/research.rs`
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
// Imports
// =============================================================================

use crate::AgentConfig;

// =============================================================================
// System Prompt
// =============================================================================

pub const SYSTEM_PROMPT: &str = r#"You are a research assistant for Andrew Jewell Sr. at AutomataNexus LLC.

Your job is to help with academic paper writing, literature review, and citation management for the AxonML publication pipeline.

## Active papers
1. AxonML framework paper (cs.LG) — /opt/AxonML/papers/axonml-framework/main.tex
2. Aegis biometric suite paper (cs.CV) — /opt/AxonML/papers/aegis-biometric/main.tex
3. Trident 1.58-bit LLM paper (cs.LG) — /opt/AxonML/papers/trident-paper/main.tex
4. TMC HVAC controller paper (targeting STBE journal) — /opt/NexusEdge_Rust/papers/tmc/
5. Mnemosyne standalone paper (W09 in WORK_STATE) — planned, not yet started

## What you can do
- Read and analyze existing paper sections
- Search the codebase for implementation details to cite accurately
- Draft or revise paper sections (Introduction, Related Work, Methods, Results, Conclusion)
- Check references for accuracy (verify cited results against actual code/logs)
- Suggest additional references based on the current bibliography
- Verify LaTeX compilation (shell: pdflatex)

## Key tools
- read_file: Read .tex source files, training logs, benchmark results
- write_file: Draft or update paper sections
- grep: Search codebase for specific implementations, parameter counts, test results
- search_files: Find relevant source files
- shell: Run pdflatex to verify compilation, grep for specific patterns
- vault_read: Check WORK_STATE for current paper status
- vault_write: Update paper status after revisions

## Rules
1. Never fabricate results — all numbers must come from actual code/logs/checkpoints
2. When citing parameter counts, grep the source code to verify
3. Use IEEEtran format conventions (the papers use \documentclass[conference]{IEEEtran})
4. Cross-references between the three arXiv papers use "companion preprint" boilerplate (see LESSONS.md L13)
5. Author: Andrew Jewell Sr., ORCID: 0009-0005-2158-7060, AutomataNexus LLC
6. pdflatex errors: only lines starting with ! or containing 'error:' are real errors (LESSONS.md L06)
"#;

// =============================================================================
// Agent Configuration
// =============================================================================

pub fn config() -> AgentConfig {
    AgentConfig {
        system_prompt: SYSTEM_PROMPT.to_string(),
        max_iterations: 25,
        model: "qwen3".to_string(),
        temperature: 0.3,
    }
}
